"""knee_estimator.py - knee angle from two 6-bit accelerometers plus a laggy camera.

One Kalman filter with state [theta, omega, b]: the accelerometers measure theta + b, the camera
measures theta, so b soaks up slow accel/camera disagreement (mounting error, strap slip). Camera
frames arrive 60-120 ms late, so each one rewinds a short history to its capture time, updates
there, and replays the accel samples since. A One Euro filter on theta gives the display: still at
rest, immediate in motion. Angles in degrees, 0 = straight, flexion positive. Times are seconds on
the caller's clock (time.monotonic()); nothing in here reads a clock. Standard library only.
"""
import math

W_DEFAULT = 0.15      # camera share of trust (0 = accelerometers only, 1 = camera only); eval_fusion.py
RESP_DEFAULT = 0.5    # responsiveness 0..1: higher follows faster, lower is smoother at rest
R_ACC0 = 4.0          # accel knee variance per 100 Hz sample at w = 0 (deg^2); 6-bit steps are 2.7 deg
R_CAM0 = R_ACC0 * 0.3 # camera variance per 30 Hz frame at w = 1, so w = 0.5 is equal information
DEV_G = 0.05          # accel R grows by (dev/DEV_G)^2 with dev = ||a| - 1 g| beyond DEAD_G (motion)
DEAD_G = 0.04         # |a| error ignored as noise / zero-g offset (g)
Q_B = 0.5 ** 2        # random walk of b while the camera is fused (deg^2/s): strap slip, not jitter
GATE = 4.0            # reject camera frames whose innovation exceeds this many sigma
RELOCK_S = 1.0        # ... unless they have disagreed this long: then trust the camera and reset b
HIST_S = 0.5          # history kept for late camera frames (s); older frames are dropped
GAP_S = 0.05          # accel silence that starts coasting (s)
COAST_TAU = 0.2       # while coasting omega decays with this time constant (s)
CAM_STALE_S = 0.3     # camera-only: no accepted frame for this long also coasts (s)
CAM_OK_S = 0.5        # cam_ok while a frame was accepted this recently (s)
SPEED_DEAD = 5.0      # display: speeds below this (deg/s) count as rest noise
ACC_H, CAM_H = (1.0, 0.0, 1.0), (1.0, 0.0, 0.0)  # measurement rows: accel = theta + b, camera = theta


def wrap(a):
    """Wrap an angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def tilt(x, z):
    """Sagittal tilt (deg) of one accelerometer's gravity vector."""
    return math.degrees(math.atan2(x, z))


def raw_knee(v6):
    """Knee angle straight from one sample: thigh xyz then shank xyz, in g."""
    return wrap(tilt(v6[3], v6[5]) - tilt(v6[0], v6[2]))


def q_from_resp(resp):
    """Responsiveness 0..1 -> white angular-acceleration density (deg^2/s^3); 0.5 -> 3000."""
    return 3000.0 * 10.0 ** (2.0 * (resp - 0.5))


def _predict(x, P, dt, q, qb, coast):
    """Constant velocity; when coasting omega decays (tau COAST_TAU) instead of running away."""
    f = math.exp(-dt / COAST_TAU) if coast else 1.0
    a = COAST_TAU * (1.0 - f) if coast else dt          # how far omega carries theta over dt
    F = ((1.0, a, 0.0), (0.0, f, 0.0), (0.0, 0.0, 1.0))
    FP = [[sum(F[i][k] * P[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    P = [[sum(FP[i][k] * F[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    P[0][0] += q * dt ** 3 / 3.0
    P[0][1] += q * dt * dt / 2.0
    P[1][0] += q * dt * dt / 2.0
    P[1][1] += q * dt
    P[2][2] += qb * dt
    return [x[0] + a * x[1], f * x[1], x[2]], P


def _innovation(x, P, h, z, r):
    ph = [P[i][0] * h[0] + P[i][2] * h[2] for i in range(3)]   # P h^T (h[1] is always 0)
    return ph, ph[0] * h[0] + ph[2] * h[2] + r, z - (x[0] * h[0] + x[2] * h[2])


def _update(x, P, ph, s, nu):
    k = [p / s for p in ph]
    return ([x[i] + k[i] * nu for i in range(3)],
            [[P[i][j] - k[i] * ph[j] for j in range(3)] for i in range(3)])


class OneEuro:
    """One Euro filter (Casiez 2012) using the Kalman omega, low-passed at 2 Hz, as its speed."""

    def __init__(self, y=None, t=None):
        self.y, self.t, self.speed = y, t, 0.0

    def __call__(self, t, x, omega, fmin, beta):
        if self.y is None or t <= self.t:
            self.y = x if self.y is None else self.y
            self.t = t
            return self.y
        dt, self.t = t - self.t, t
        self.speed += (omega - self.speed) / (1.0 + 1.0 / (2.0 * math.pi * 2.0 * dt))
        fc = fmin + beta * max(0.0, abs(self.speed) - SPEED_DEAD)
        self.y += (x - self.y) / (1.0 + 1.0 / (2.0 * math.pi * fc * dt))
        return self.y


class KneeFilter:
    """Accelerometer samples and late camera frames in, display (theta, omega) out."""

    def __init__(self, w=W_DEFAULT, resp=RESP_DEFAULT):
        self.w, self.resp, self.src = w, resp, "fused"
        self.k = 0.0                 # accel reference: theta = raw_knee - k (+ b)
        self.calibrated = False
        self.x = self.P = None       # state and covariance at time self.t
        self.t = None
        self.hist = []               # [t, z, r, h, x, P] per measurement, last HIST_S seconds
        self.t_cam = -1e9            # capture time of the last accepted camera frame
        self.rej_since = None        # capture time of the first of the current run of rejects
        self.euro = OneEuro()
        self.set(w, resp)

    def set(self, w=None, resp=None, src=None):
        self.w = self.w if w is None else min(1.0, max(0.0, w))
        self.resp = self.resp if resp is None else min(1.0, max(0.0, resp))
        self.src = src or self.src
        self.w_eff = {"accel": 0.0, "camera": 1.0}.get(self.src, self.w)
        self.q = q_from_resp(self.resp)
        self.fmin = 0.1 + 0.4 * self.resp        # display cutoff at rest (Hz)
        self.beta = 0.1 + 0.4 * self.resp        # extra cutoff (Hz) per deg/s of speed

    @property
    def cam_ok(self):
        return self.t is not None and self.w_eff > 0.0 and self.t - self.t_cam < CAM_OK_S

    def _step(self, x, P, t0, e):
        """Advance (x, P) from t0 to history entry e and apply its measurement (r None = skip)."""
        t, z, r, h = e[0], e[1], e[2], e[3]
        dt = t - t0
        camera_fused = self.w_eff > 0.0 and t - self.t_cam < CAM_OK_S
        coast = dt > GAP_S or (self.w_eff >= 1.0 and t - self.t_cam > CAM_STALE_S)
        x, P = _predict(x, P, dt, self.q, Q_B if camera_fused else 0.0, coast)
        if r is not None:
            x, P = _update(x, P, *_innovation(x, P, h, z, r))
        return x, P

    def accel(self, t, v6):
        """One accelerometer sample at time t. Returns the display (theta, omega)."""
        z = self.z_acc = raw_knee(v6) - self.k
        if self.x is None:
            self.x, self.P, self.t = [z, 0.0, 0.0], [[4.0, 0, 0], [0, 100.0 ** 2, 0], [0, 0, 3.0 ** 2]], t
            self.hist = [[t, z, None, ACC_H, self.x, self.P]]
            return self.euro(t, z, 0.0, self.fmin, self.beta), 0.0
        if t <= self.t:
            return self.euro.y, self.x[1]       # out-of-order or duplicate stamp
        r = None
        if self.w_eff < 1.0:
            dev = max(abs(math.sqrt(v6[0] ** 2 + v6[1] ** 2 + v6[2] ** 2) - 1.0),
                      abs(math.sqrt(v6[3] ** 2 + v6[4] ** 2 + v6[5] ** 2) - 1.0))
            r = R_ACC0 / (1.0 - self.w_eff) * (1.0 + (max(0.0, dev - DEAD_G) / DEV_G) ** 2)
        e = [t, z, r, ACC_H]
        self.x, self.P = self._step(self.x, self.P, self.t, e)
        self.t = t
        self.hist.append(e + [self.x, self.P])
        n = 0
        while n < len(self.hist) - 1 and self.hist[n + 1][0] < t - HIST_S:
            n += 1
        del self.hist[:n]
        return self.euro(t, self.x[0], self.x[1], self.fmin, self.beta), self.x[1]

    def camera(self, t_cap, knee):
        """One camera knee angle captured at t_cap (may be up to HIST_S old). Returns accepted."""
        if knee is None or self.x is None or self.w_eff <= 0.0:
            return False
        H = self.hist
        i = len(H)
        while i > 0 and H[i - 1][0] > t_cap:
            i -= 1
        if i == 0:
            return False                          # older than the history
        r = R_CAM0 / self.w_eff
        x, P = self._step(H[i - 1][4], H[i - 1][5], H[i - 1][0], [t_cap, 0.0, None, CAM_H])
        ph, s, nu = _innovation(x, P, CAM_H, knee, r)
        if nu * nu > GATE * GATE * s:
            if self.rej_since is None:
                self.rej_since = t_cap
            if t_cap - self.rej_since < RELOCK_S and t_cap - self.t_cam < RELOCK_S:
                return False
            # Camera disagreed for a second or is back after a dropout: believe it and re-derive b
            # from the latest accel reading (theta may be a stale prediction in camera-only mode)
            x = [knee, x[1], self.z_acc - knee]
            P = [[r, 0.0, -r], [0.0, P[1][1], 0.0], [-r, 0.0, P[0][0] + 2.0 * P[0][2] + P[2][2] + r]]
        else:
            x, P = _update(x, P, ph, s, nu)
        self.rej_since = None
        self.t_cam = max(self.t_cam, t_cap)
        H.insert(i, [t_cap, knee, r, CAM_H, x, P])
        for k in range(i + 1, len(H)):            # replay everything measured since the capture
            x, P = self._step(x, P, H[k - 1][0], H[k])
            H[k][4], H[k][5] = x, P
        self.x, self.P, self.t = x, P, max(self.t, t_cap)
        return True

    def tick(self, t):
        """Coast output while the accelerometers are silent; None while they are streaming."""
        if self.x is None or t - self.t <= GAP_S:
            return None
        x, _ = _predict(self.x, self.P, t - self.t, 0.0, 0.0, True)
        return self.euro(t, x[0], x[1], self.fmin, self.beta), x[1]

    def _rebase(self, k_new, theta_now):
        """New accel reference: theta := theta_now, b := 0, history restarts (snap the display)."""
        if self.x is None:
            return
        self.k = k_new
        self.x = [theta_now, self.x[1], 0.0]
        self.P = [[self.P[0][0], self.P[0][1], 0.0], [self.P[1][0], self.P[1][1], 0.0], [0.0, 0.0, 3.0 ** 2]]
        self.hist = [[self.t, theta_now, None, ACC_H, self.x, self.P]]
        self.euro = OneEuro(theta_now, self.t)

    def zero(self, theta_now_target=0.0):
        """Quick zero: the current (filtered) accel reading becomes theta_now_target.
        Not a calibration: the extension deficit is unknown, so calibrated becomes False."""
        if self.x is not None:
            self._rebase(self.k + self.x[0] + self.x[2] - theta_now_target, theta_now_target)
            self.calibrated = False

    def set_reference(self, k_deg):
        """Calibrated offset from calibrate.py: displayed theta = raw knee - k_deg."""
        if self.x is not None:
            self._rebase(k_deg, self.x[0] + self.x[2] + self.k - k_deg)
        else:
            self.k = k_deg
        self.calibrated = True
