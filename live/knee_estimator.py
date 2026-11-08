"""knee_estimator.py - knee angle from two 6-bit accelerometers plus a laggy camera.

One Kalman filter with state [theta, omega, alpha, b]: the accelerometers measure theta + b, the camera
measures theta, so b soaks up accel/camera disagreement (mounting error, strap slip, and the board error
that changes with the knee angle). Camera frames arrive 60-150 ms late and are stamped late (t_cap is
taken when the frame reaches the program, not at the exposure): each frame is moved back by a delay the
filter learns from motion, then rewinds a short history to that time, updates there, and replays the
accel samples since. The swinging shank board also feels centripetal and tangential acceleration, which
tilts its gravity reading by up to ~10 deg in a brisk rep; it is removed using the estimated omega/alpha
and the sensor's distance from the hinge. A One Euro filter on theta gives the display: still at rest,
immediate in motion. Angles in degrees, 0 = straight, flexion positive. Times are seconds on the caller's
clock (time.monotonic()); nothing in here reads a clock. Standard library only.
"""
import math

# Constants marked (tuned) were fitted with tuning/optimize.py on the mock (tuning/fusion_bench.py scenarios,
# train seeds) and checked on held-out seeds; see tuning/TUNING_NOTES.md.
W_DEFAULT = 0.75      # camera share of trust (0 = accelerometers only, 1 = camera only): camera leads
RESP_DEFAULT = 0.5    # responsiveness 0..1: higher follows faster, lower is smoother at rest
R_ACC0 = 1.7          # (tuned) accel knee variance per 100 Hz sample at w = 0 (deg^2)
R_CAM0 = 11.0         # (tuned) floor of the camera variance per frame at w = 1 (deg^2); also learned (cam_var)
DEV_G = 0.2           # (tuned) accel R grows by (dev/DEV_G)^2, dev = ||a| - 1 g| beyond DEAD_G and the swing
DEAD_G = 0.0          # (tuned) |a| error ignored as noise / zero-g offset (g)
Q_B = 0.03            # random walk of b while the camera is fused (deg^2/s): strap slip (tuning/ slip scenario)
Q_J0 = 1.2e6          # (tuned) white jerk density at resp = 0.5 (deg^2/s^5)
ALPHA_TAU = 0.075     # (tuned) angular acceleration decorrelates over this time (s)
Q_CAM_A = 3000.0      # camera only: white angular-acceleration density at resp = 0.5 (deg^2/s^3)
GATE = 3.0            # (tuned) reject camera frames whose innovation exceeds this many sigma
RELOCK_N = 3          # ... unless this many rejected frames
RELOCK_S = 0.4        # ... within this long
RELOCK_SD = 3.0       # ... lie on a line to within this (deg): then the camera is right, relock to it
HIST_S = 0.5          # history kept for late camera frames (s); older frames are dropped
GAP_S = 0.05          # accel silence that starts coasting (s)
COAST_TAU = 0.2       # while coasting omega decays with this time constant (s)
CAM_STALE_S = 0.3     # camera-only: no accepted frame for this long also coasts (s)
CAM_OK_S = 0.5        # cam_ok while a frame was accepted this recently (s)
EURO_FMIN = (0.033, 0.133)  # (tuned) display (One Euro) cutoff at rest: a + b * resp (Hz)
EURO_BETA = (0.035, 0.14)   # (tuned) ... plus this (a + b * resp, Hz) per deg/s of speed
# Without camera frames (no webcam, markers lost, src = accel) the accelerometers are on their own and want
# a calmer model and display than when the camera steadies the estimate:
# a constant-velocity model like the original (pre-rework) filter, which tuning/ablate_accel.py found best
# without a camera (the swing model needs a good alpha, which only the camera-steadied model has), retuned.
Q_ACC_A = 6000.0      # (tuned) white angular-acceleration density at resp = 0.5, no camera (deg^2/s^3)
R_ACC0_ACC = 5.7      # (tuned) accel knee variance per sample, no camera (deg^2)
DEV_G_ACC, DEAD_G_ACC = 0.084, 0.0275   # (tuned) |a| weighting, no camera (g)
BLIP_SIGMA_ACC = 3.0  # jolt threshold, no camera
SWING_ACC = 0         # swing model without camera: 0 off, 1 centripetal only (omega^2; alpha is not estimated)
EURO_FMIN_ACC = (0.033, 0.133)  # (tuned) display cutoff at rest, no camera (Hz)
EURO_BETA_ACC = (0.071, 0.283)  # (tuned) display speed slope, no camera
OMEGA_LIM, ALPHA_LIM = 1500.0, 30000.0   # hard state limits (deg/s, deg/s^2): a guard, never reached by a leg
SPEED_DEAD = 5.0      # display: speeds below this (deg/s) count as rest noise
BLIP_SIGMA = 2.2      # (tuned) accel samples further than this from the prediction are jolts: down-weighted
G = 9.80665           # m/s^2
R_SHANK_M = 0.118     # hinge -> shank sensor pocket centre (hardware/brace/brace.py sensor_pos)
OMEGA_MAX = 600.0     # the swing model trusts at most this knee speed (deg/s) ...
ALPHA_MAX = 4000.0    # ... and acceleration (deg/s^2); a brisk 90 deg rep in 0.5 s peaks near 2100
DYN_RES = 0.06        # (tuned) share of the modelled swing acceleration kept as uncertainty (lever arm, axes)
CAM_DELAY0 = 0.04     # first guess of exposure -> t_cap (s); learned while the knee moves
CAM_DELAY_MAX = 0.25
CAM_DELAY_GAIN = 0.17 # (tuned) learning rate of the camera delay (per frame in motion)
CAM_VAR_GAIN = 0.02   # learning rate of the camera noise (per frame at rest)
CAM_VAR_SPEED = 20.0  # ... frames count as "at rest" below this speed (deg/s)
CAM_DELAY_SPEED = 30.0  # frames are used for the delay only above this speed (deg/s)
CAM_H = (1.0, 0.0, 0.0, 0.0)  # camera row: camera = theta (accel = theta + b + swing, see _accel_z)
N = 4


def wrap(a):
    """Wrap an angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def tilt(x, y):
    """Sagittal tilt (deg) of one accelerometer's gravity vector. The boards lie flat on the brace
    bars, so the knee bends in each board's x-y plane and z is the hinge axis."""
    return math.degrees(math.atan2(x, y))


def raw_knee(v6):
    """Knee angle straight from one sample: thigh xyz then shank xyz, in g."""
    return wrap(tilt(v6[3], v6[4]) - tilt(v6[0], v6[1]))


def q_from_resp(resp, q0=None):
    """Responsiveness 0..1 -> white jerk density (deg^2/s^5); x100 per +1."""
    return (Q_J0 if q0 is None else q0) * 10.0 ** (2.0 * (resp - 0.5))


def _predict(x, P, dt, q, qb, coast):
    """Constant acceleration (alpha decaying over ALPHA_TAU) driven by white jerk. Coasting (no data):
    alpha is dropped and omega decays (tau COAST_TAU) so theta settles instead of running away."""
    if q < 0.0:                          # constant velocity (camera only), white acceleration density -q
        F = ((1.0, dt, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    if coast:
        f = math.exp(-dt / COAST_TAU)
        F = ((1.0, COAST_TAU * (1.0 - f), 0.0, 0.0), (0.0, f, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    elif q >= 0.0:
        e = math.exp(-dt / ALPHA_TAU)
        F = ((1.0, dt, 0.5 * dt * dt, 0.0), (0.0, 1.0, dt, 0.0), (0.0, 0.0, e, 0.0), (0.0, 0.0, 0.0, 1.0))
    xn = [sum(F[i][k] * x[k] for k in range(N)) for i in range(N)]
    FP = [[sum(F[i][k] * P[k][j] for k in range(N) if F[i][k]) for j in range(N)] for i in range(N)]
    Pn = [[sum(FP[i][k] * F[j][k] for k in range(N) if F[j][k]) for j in range(N)] for i in range(N)]
    d2, d3 = dt * dt, dt * dt * dt
    if q < 0.0:
        Pn[0][0] -= q * d3 / 3.0
        Pn[0][1] -= q * d2 / 2.0
        Pn[1][0] -= q * d2 / 2.0
        Pn[1][1] -= q * dt
    else:
        Q = ((d3 * d2 / 20.0, d2 * d2 / 8.0, d3 / 6.0), (d2 * d2 / 8.0, d3 / 3.0, d2 / 2.0), (d3 / 6.0, d2 / 2.0, dt))
        for i in range(3):
            for j in range(3):
                Pn[i][j] += q * Q[i][j]
    Pn[3][3] += qb * dt
    return xn, Pn


def _innovation(x, P, h, z, r):
    ph = [sum(P[i][k] * h[k] for k in range(N) if h[k]) for i in range(N)]   # P h^T
    return ph, sum(ph[k] * h[k] for k in range(N) if h[k]) + r, z - sum(x[k] * h[k] for k in range(N))


def _update(x, P, ph, s, nu):
    k = [p / s for p in ph]
    return ([x[i] + k[i] * nu for i in range(N)],
            [[P[i][j] - k[i] * ph[j] for j in range(N)] for i in range(N)])


class OneEuro:
    """One Euro filter (Casiez 2012) using the Kalman omega, low-passed at 2 Hz, as its speed."""

    def __init__(self, y=None, t=None):
        self.y, self.t, self.speed = y, t, 0.0

    def _next(self, t, x, omega, fmin, beta):
        dt = t - self.t
        speed = self.speed + (omega - self.speed) / (1.0 + 1.0 / (2.0 * math.pi * 2.0 * dt))
        fc = fmin + beta * max(0.0, abs(speed) - SPEED_DEAD)
        return self.y + (x - self.y) / (1.0 + 1.0 / (2.0 * math.pi * fc * dt)), speed

    def __call__(self, t, x, omega, fmin, beta):
        if self.y is None or t <= self.t:
            self.y = x if self.y is None else self.y
            self.t = t if self.t is None else max(self.t, t)
            return self.y
        self.y, self.speed = self._next(t, x, omega, fmin, beta)
        self.t = t
        return self.y

    def peek(self, t, x, omega, fmin, beta):
        """The output at t without committing it (coasting previews must not move the filter's clock)."""
        if self.y is None or t <= self.t:
            return x if self.y is None else self.y
        return self._next(t, x, omega, fmin, beta)[0]


class KneeFilter:
    """Accelerometer samples and late camera frames in, display (theta, omega) out."""

    def __init__(self, w=W_DEFAULT, resp=RESP_DEFAULT):
        self.w, self.resp, self.src = w, resp, "fused"
        self.k = 0.0                 # accel reference: theta = raw_knee - k (+ b)
        self.calibrated = False
        self.r_shank = R_SHANK_M     # lever arm of the shank board (m); 0 turns the swing correction off
        self.x = self.P = None       # state and covariance at time self.t
        self.t = None                # time of the newest history entry (never ahead of the accel stream)
        self.t_acc = None            # time of the newest accel sample
        self.z_acc = None            # newest accel knee reading (theta + b)
        # history: [t, kind (0 accel, 1 camera, 2 camera relock), meas, r, x, P, t_acc, z], last HIST_S
        self.hist = []
        self.t_cam = -1e9            # (delay-corrected) time of the last accepted camera frame
        self.cam_delay = CAM_DELAY0  # learned exposure -> t_cap delay (s)
        self.cam_var = R_CAM0        # learned camera angle variance (deg^2) at w = 1
        self.rej = []                # recently rejected camera frames [(t, knee)]
        self.n_relock = 0
        self.euro = OneEuro()
        self.set(w, resp)

    def set(self, w=None, resp=None, src=None):
        self.w = self.w if w is None else min(1.0, max(0.0, w))
        self.resp = self.resp if resp is None else min(1.0, max(0.0, resp))
        self.src = src or self.src
        self.w_eff = {"accel": 0.0, "camera": 1.0}.get(self.src, self.w)
        # camera only: nothing between the 30 Hz frames, so a jerk model would overshoot every stop;
        # use constant velocity instead (q < 0 flags it: white acceleration density -q)
        self.q = -Q_CAM_A * 10.0 ** (2.0 * (self.resp - 0.5)) if self.w_eff >= 1.0 else q_from_resp(self.resp)
        self.q_acc = -Q_ACC_A * 10.0 ** (2.0 * (self.resp - 0.5))   # no camera frames: constant velocity
        self.euro_cam = (EURO_FMIN[0] + EURO_FMIN[1] * self.resp,      # display cutoff at rest (Hz)
                         EURO_BETA[0] + EURO_BETA[1] * self.resp)      # extra cutoff (Hz) per deg/s of speed
        self.euro_acc = (EURO_FMIN_ACC[0] + EURO_FMIN_ACC[1] * self.resp, EURO_BETA_ACC[0] + EURO_BETA_ACC[1] * self.resp)

    @property
    def fmin(self):
        return (self.euro_cam if self.cam_ok else self.euro_acc)[0]

    @property
    def beta(self):
        return (self.euro_cam if self.cam_ok else self.euro_acc)[1]

    @property
    def cam_ok(self):
        return self.t is not None and self.w_eff > 0.0 and self.t - self.t_cam < CAM_OK_S

    # ---- measurement models ----
    def _accel_z(self, x, m, w_eff=None, fused=True):
        """Accel measurement for state x: (z, h, r, zc). z = raw knee - k (unwrapped near the prediction),
        h = its linearised row, r = its variance, zc = z with the modelled swing removed (theta + b).

        The shank board sits r_shank from the hinge, so it also feels r*omega^2 towards the hinge and r*alpha
        along its path. Seen through its tilt, that adds d = (r/g)(omega^2 cos theta + alpha sin theta) rad
        to the knee reading (the board hangs straight down at theta = 90; true for any mounting in the
        mapped frame, where the reading rises with theta). It is part of the model (an EKF), not subtracted
        from the input: alpha is too noisy to subtract, but the filter can weigh it."""
        tau_t, sx, sy, sz, dev_t = m
        th = math.radians(x[0])
        w = math.radians(max(-OMEGA_MAX, min(OMEGA_MAX, x[1])))
        a = math.radians(max(-ALPHA_MAX, min(ALPHA_MAX, x[2]))) if fused or SWING_ACC > 1 else 0.0
        if not fused and not SWING_ACC:
            w = 0.0
        c, sn = math.cos(th), math.sin(th)
        kr = self.r_shank / G
        d = math.degrees(kr * (w * w * c + a * sn))
        h = (1.0 + kr * (a * c - w * w * sn),                    # d(theta + b + d)/dx (0 where clipped)
             kr * 2.0 * w * c if abs(x[1]) < OMEGA_MAX else 0.0,
             kr * sn if abs(x[2]) < ALPHA_MAX else 0.0, 1.0)
        pred = x[0] + x[3] + d
        z = pred + wrap(tilt(sx, sy) - tau_t - self.k - pred)    # continuous near the prediction
        # |a| of the shank also changes with the swing (by (r/g)(omega^2 sin theta - alpha cos theta));
        # what is left beyond that is unmodelled motion or a jolt
        norm = math.sqrt(sx * sx + sy * sy + sz * sz) - 1.0 - kr * (w * w * sn - a * c)
        dev = max(dev_t, abs(norm))
        w_eff = self.w_eff if w_eff is None else w_eff
        r0, dg, dd = (R_ACC0, DEV_G, DEAD_G) if fused else (R_ACC0_ACC, DEV_G_ACC, DEAD_G_ACC)
        r = r0 / max(1e-6, 1.0 - w_eff) * (1.0 + (max(0.0, dev - dd) / dg) ** 2) + (DYN_RES * d) ** 2
        return z, h, r, z - d, pred

    def _step(self, x, P, prev, e):
        """Advance (x, P) from history entry prev to entry e and apply e's measurement (fills e[4:])."""
        t, kind = e[0], e[1]
        if kind == 2:                                 # camera relock: a reset, not a measurement
            e[6] = prev[6]
            return e[4], e[5]
        dt = t - prev[0]
        camera_fused = self.w_eff > 0.0 and t - self.t_cam < CAM_OK_S
        coast = t - prev[6] > GAP_S or (self.w_eff >= 1.0 and t - self.t_cam > CAM_STALE_S)
        qb = Q_B if camera_fused else 0.0
        x, P = _predict(x, P, dt, self.q if camera_fused or self.q < 0.0 else self.q_acc, qb, coast)
        z = None
        if kind == 0:
            # the camera share only discounts the accelerometers while camera frames are actually fused
            zm, h, r, z, pred = self._accel_z(x, e[2], self.w_eff if camera_fused else 0.0, camera_fused)
            if self.w_eff < 1.0:
                zl = zm - pred + sum(h[k] * x[k] for k in range(N))     # linearised about x
                ph, s, nu = _innovation(x, P, h, zl, r)
                bs = BLIP_SIGMA if camera_fused else BLIP_SIGMA_ACC
                if nu * nu > bs * bs * s:             # jolt, not a real angle change (Huber)
                    ph, s, nu = _innovation(x, P, h, zl, r * nu * nu / (bs * bs * s))
                x, P = _update(x, P, ph, s, nu)
                if abs(x[1]) > OMEGA_LIM or abs(x[2]) > ALPHA_LIM:    # guard: never let the state run away
                    x = [x[0], max(-OMEGA_LIM, min(OMEGA_LIM, x[1])), max(-ALPHA_LIM, min(ALPHA_LIM, x[2])), x[3]]
        elif e[3] is not None:
            ph, s, nu = _innovation(x, P, CAM_H, x[0] + wrap(e[2] - x[0]), e[3])
            x, P = _update(x, P, ph, s, nu)
        e[4:] = [x, P, t if kind == 0 else prev[6], z]
        return x, P

    def _replay(self, i):
        """Recompute history from entry i on (entry i-1 is the starting state)."""
        H = self.hist
        x, P = H[i - 1][4], H[i - 1][5]
        for j in range(i, len(H)):
            x, P = self._step(x, P, H[j - 1], H[j])
        self.x, self.P = x, P

    def accel(self, t, v6):
        """One accelerometer sample at time t. Returns the display (theta, omega)."""
        tau_t = tilt(v6[0], v6[1])
        m = (tau_t, v6[3], v6[4], v6[5], abs(math.sqrt(v6[0] ** 2 + v6[1] ** 2 + v6[2] ** 2) - 1.0))
        if self.x is None:
            z = wrap(raw_knee(v6) - self.k)
            self.x, self.P, self.t = [z, 0.0, 0.0, 0.0], self._p0(4.0, 3.0 ** 2), t
            self.t_acc, self.z_acc = t, z
            self.hist = [[t, 0, m, None, self.x, self.P, t, z]]
            return self.euro(t, z, 0.0, self.fmin, self.beta), 0.0
        if t <= self.t:
            return self.euro.y, self.x[1]       # out-of-order or duplicate stamp
        e = [t, 0, m, None, None, None, None, None]
        self.x, self.P = self._step(self.x, self.P, self.hist[-1], e)
        self.t, self.t_acc, self.z_acc = t, t, e[7]
        self.hist.append(e)
        n = 0
        while n < len(self.hist) - 1 and self.hist[n + 1][0] < t - HIST_S:
            n += 1
        del self.hist[:n]
        return self.euro(t, self.x[0], self.x[1], self.fmin, self.beta), self.x[1]

    @staticmethod
    def _p0(var_theta, var_b):
        P = [[0.0] * N for _ in range(N)]
        P[0][0], P[1][1], P[2][2], P[3][3] = var_theta, 100.0 ** 2, 1000.0 ** 2, var_b
        return P

    def camera(self, t_cap, knee, conf=1.0):
        """One camera knee angle stamped t_cap (its true exposure is ~cam_delay earlier, up to HIST_S
        old). conf (0..1) scales its weight. Returns True if it was used."""
        if knee is None or self.x is None or self.w_eff <= 0.0:
            return False
        H = self.hist
        t = min(t_cap - self.cam_delay, self.t)          # never ahead of the accel stream
        i = len(H)
        while i > 0 and H[i - 1][0] > t:
            i -= 1
        if i == 0:
            return False                                  # older than the history
        r = max(R_CAM0, self.cam_var) / self.w_eff / max(0.5, min(1.0, conf if conf is not None else 1.0)) ** 2
        e = [t, 1, knee, r, None, None, None, None]
        x, P = _predict(H[i - 1][4], H[i - 1][5], t - H[i - 1][0], self.q, 0.0, t - H[i - 1][6] > GAP_S)
        nu = wrap(knee - x[0])
        s = P[0][0] + r
        if nu * nu > GATE * GATE * s:
            self.rej = [f for f in self.rej if t - RELOCK_S <= f[0] < t] + [(t, x[0] + nu)]
            fit = self._line(self.rej) if len(self.rej) >= RELOCK_N else None
            if fit is None:
                return False
            # Several frames agree with each other and not with the filter: the filter is wrong (camera
            # back after a dropout, strap slipped, or a runaway prediction). Restart theta/omega on them and
            # re-derive b from the accel reading at that time.
            theta, omega = fit
            z_acc = H[i - 1][7] if H[i - 1][7] is not None else self.z_acc
            x = [theta, omega, 0.0, (z_acc - theta) if self.w_eff < 1.0 else x[3]]
            P = self._p0(r, r + R_ACC0)
            P[1][1], P[2][2] = 30.0 ** 2, 300.0 ** 2
            e[1], e[3] = 2, None                          # a reset entry: replays keep it
            e[4:] = [x, P, H[i - 1][6], None]
            self.rej, self.n_relock = [], self.n_relock + 1
        else:
            if self.w_eff < 1.0 and abs(x[1]) > CAM_DELAY_SPEED:
                # The camera shows the knee as it was cam_delay before t_cap; a wrong delay leaves an
                # innovation of -omega * (delay error). Learn it (frames in both directions average out
                # angle offsets).
                err = max(-0.1, min(0.1, -nu / x[1]))
                wt = x[1] * x[1] / (x[1] * x[1] + 100.0 ** 2)
                self.cam_delay = max(0.0, min(CAM_DELAY_MAX, self.cam_delay + CAM_DELAY_GAIN * wt * err))
            if abs(x[1]) < CAM_VAR_SPEED:
                # Learn how noisy this camera really is (innovation^2 minus the filter's own share), from
                # frames at rest where timing errors do not matter. Clipped so outliers count only a little.
                v = min(nu * nu, (GATE * GATE / 4.0) * s) - P[0][0]
                self.cam_var += CAM_VAR_GAIN * (max(0.0, v) - self.cam_var)
            ph, s, nu = _innovation(x, P, CAM_H, x[0] + nu, r)
            x, P = _update(x, P, ph, s, nu)
            e[4:] = [x, P, H[i - 1][6], None]
            self.rej = []
        self.t_cam = max(self.t_cam, t)
        H.insert(i, e)
        if i + 1 < len(H):
            self._replay(i + 1)
        else:
            self.x, self.P = x, P
        return True

    @staticmethod
    def _line(fr):
        """Least-squares line through frames [(t, knee)] -> (knee at the newest t, slope) if they all lie
        within RELOCK_SD of it, else None."""
        n = len(fr)
        tm = sum(f[0] for f in fr) / n
        km = sum(f[1] for f in fr) / n
        stt = sum((f[0] - tm) ** 2 for f in fr)
        slope = sum((f[0] - tm) * (f[1] - km) for f in fr) / stt if stt > 1e-9 else 0.0
        slope = max(-400.0, min(400.0, slope))
        if max(abs(f[1] - km - slope * (f[0] - tm)) for f in fr) > RELOCK_SD:
            return None
        return km + slope * (fr[-1][0] - tm), slope

    def tick(self, t):
        """Coast output while the accelerometers are silent; None while they are streaming."""
        if self.x is None or t - self.t_acc <= GAP_S:
            return None
        x, _ = _predict(self.x, self.P, t - self.t, 0.0, 0.0, True)
        return self.euro.peek(t, x[0], x[1], self.fmin, self.beta), x[1]

    def _rebase(self, k_new, theta_now):
        """New accel reference: theta := theta_now, b := 0, history restarts (snap the display)."""
        if self.x is None:
            return
        self.k = k_new
        self.x = [theta_now, self.x[1], self.x[2], 0.0]
        P = [row[:] for row in self.P]
        for j in range(N):
            P[3][j] = P[j][3] = 0.0
        P[3][3] = 3.0 ** 2
        self.P = P
        self.hist = [[self.t, 0, self.hist[-1][2], None, self.x, self.P, self.t_acc, theta_now]]
        self.rej = []
        self.euro = OneEuro(theta_now, self.t)

    def zero(self, theta_now_target=0.0):
        """Quick zero: the current (filtered) accel reading becomes theta_now_target.
        Not a calibration: the extension deficit is unknown, so calibrated becomes False."""
        if self.x is not None:
            self._rebase(self.k + self.x[0] + self.x[3] - theta_now_target, theta_now_target)
            self.calibrated = False

    def set_reference(self, k_deg):
        """Calibrated offset from calibrate.py: displayed theta = raw knee - k_deg."""
        if self.x is not None:
            self._rebase(k_deg, wrap(self.x[0] + self.x[3] + self.k - k_deg))
        else:
            self.k = k_deg
        self.calibrated = True
