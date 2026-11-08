#!/usr/bin/env python3
"""serial_bridge.py - Arduino (brace_sensors.ino) -> stdout, for the C++ viewer.

Reads the Arduino's USB serial port and prints only the sample lines
    t_ms,thigh_x,thigh_y,thigh_z,shank_x,shank_y,shank_z
so main.cpp can use it in place of fake_pi.py:
    viewer --pi-cmd "python serial_bridge.py"          (Windows)
    viewer --pi-cmd "python3 serial_bridge.py"         (Mac/Linux)

Also handy on its own:
    python serial_bridge.py --pretty                   # readable check + sample rate
    python serial_bridge.py --log session.csv          # also save every sample
    python serial_bridge.py --port COM5                # pick the port yourself

Position of each sensor (new)
    Every sample line gets four extra columns appended:
        ...,thigh_px,thigh_py,shank_px,shank_py            (cm, 2 decimals)
    Sagittal plane, origin = knee hinge, +x forward, +y up (the same frame main.cpp draws:
    the thigh runs along -x, and flexion swings the shin down, towards -y).
    main.cpp reads the first 7 fields with sscanf and ignores the rest, so the viewer
    needs no change. Use --no-position to get the plain 7-field lines back.

    How it is derived (accelerometers only, no gyro):
      1. Gravity is estimated per sensor with a slow low-pass and subtracted, and the
         remainder (linear acceleration) is rotated into the world frame by the tilt.
      2. delta-v = a*dt is integrated to velocity, then to position.
      3. ZUPT: when a sensor has been still for ~0.15 s its velocity is set to zero and
         its gravity estimate snaps to the reading (this stops the drift between reps).
      4. Knee-hinge constraint: the tilts also fix where each sensor must be (it sits a
         known distance along its segment from the knee), which is exact at low
         frequency. A complementary filter (--fuse-tau) pulls the integrated position
         towards that geometry, so the integration supplies the fast motion and the
         geometry supplies the slow part that integration alone cannot hold.
    Be aware (measured in simulation): without a gyro, a tilt looks like an acceleration,
    and a real leg-extension acceleration is only ~1 sensor step (0.047 g). Integrated
    delta-v therefore adds almost nothing; the position is essentially the knee-hinge
    geometry (~0.4 cm RMS in simulation), with delta-v a small refinement. That is why
    --fuse-tau defaults to a small 0.05 s. Raising it makes the output worse, not better.
    Errors that no filter can fix: a wrong --thigh/shank-sensor-cm shifts the position by
    the same amount, and a 3-count sensor bias costs ~1-2 cm. The camera is the cure.
    Assumes the knee stays put (seated leg extension). Set --thigh-sensor-cm and
    --shank-sensor-cm to the real distances from the knee to each sensor.

Status messages go to stderr, so stdout carries nothing but samples.
Close the Arduino IDE's Serial Monitor first: only one program can open the port.
Needs: pip install pyserial
"""
import argparse
import math
import sys
import time
from collections import deque

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit("pyserial is missing. Install it with:  pip install pyserial")

HINTS = ("arduino", "usbmodem", "usbserial", "wchusbserial", "ch340", "ch910",
         "cp210", "ftdi", "ttyacm", "ttyusb", "usb serial", "seeed")


def log(msg):
    print(f"[bridge] {msg}", file=sys.stderr, flush=True)


def find_port():
    best, best_score = None, 0
    for p in serial.tools.list_ports.comports():
        text = " ".join(filter(None, (p.device, p.description, p.manufacturer))).lower()
        score = sum(h in text for h in HINTS) - (5 if "bluetooth" in text else 0)
        if score > best_score:
            best, best_score = p.device, score
    return best


def parse(line):
    parts = line.split(",")
    if len(parts) != 7:
        return None
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in vals):   # float() accepts nan/inf
        return None
    # The MMA7660 only reads +/-1.5 g. Anything bigger is a corrupted line (two programs reading the
    # same port split the bytes and produce numbers like 857557780937).
    if any(abs(v) > 2.0 for v in vals[1:]) or vals[0] < 0 or vals[0] > 1e9:
        return None
    return vals


# ----------------------------------------------------------- sensor positions ----
G = 9.80665          # m/s^2 per g
CUTOFF_HZ = 5.0      # tilt / knee-angle low-pass, same as kCutoffHz in main.cpp
ACC_LIMIT_G = 0.15   # cap on the linear acceleration fed to the integrator (g)


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class PositionEstimator:
    """Sagittal-plane position of the thigh and shank sensors, origin at the knee.

    Output frame: +x forward, +y up, centimetres. Thigh heading phi is clockwise-positive
    and u(phi) = (cos phi, -sin phi), so the thigh sensor sits at -d_t*u(phi_thigh) and the
    shank sensor at +d_s*u(phi_shank), with phi_shank = phi_thigh + knee (see module doc).
    """

    STILL_WIN = 15  # samples (~0.15 s at 100 Hz)

    def __init__(self, a):
        self.a = a
        self.s = float(a.knee_sign)
        self.d = [a.thigh_sensor_cm / 100.0, a.shank_sensor_cm / 100.0]   # metres; the camera refines these
        self.thigh_zero = a.thigh_zero      # thigh tilt (deg) when the thigh is horizontal
        self.knee_zero = a.knee_zero        # knee reading (deg) when the leg is straight
        self.last_t = None
        self.y = None                       # 5 Hz low-pass of [thigh xyz, shank xyz]
        self.g = None                       # slow gravity estimate per sensor, in the sagittal plane
        self.win = [deque(maxlen=self.STILL_WIN), deque(maxlen=self.STILL_WIN)]
        self.p = [[0.0, 0.0], [0.0, 0.0]]   # position, metres (x fwd, y up)
        self.v = [[0.0, 0.0], [0.0, 0.0]]   # velocity, m/s
        self.still = [False, False]
        self.zero_end = None
        self.z_sin = [0.0, 0.0]
        self.z_cos = [0.0, 0.0]
        self.zeroing = a.zero_secs > 0
        self.cam_tau = getattr(a, "cam_tau", 2.0)   # seconds; smoothing of the camera's distance measurement
        self.knee_deg = 0.0                         # last knee angle from the accelerometers (deg)
        self.cam_zero = getattr(a, "cam_zero", True)
        self.cam_zero_set = False                   # first camera fix sets the zeros outright

    def _use_camera(self, cam, dt):
        """Camera fix -> learn the true knee-to-sensor distances (the largest error the filter cannot fix).

        The marker next to each sensor gives |rel| = distance from the knee marker. Its norm does not
        depend on the leg angle or on camera latency, so it can be averaged over the whole rep.
        """
        if cam is None or getattr(cam, "confidence", 0.0) < 0.5:
            return
        k = dt / (self.cam_tau + dt)
        for i, rel in enumerate((cam.thigh_rel_cm, cam.shank_rel_cm)):
            if rel is None:
                continue
            meas = math.hypot(rel[0], rel[1]) / 100.0
            if not (0.03 <= meas <= 0.60) or abs(meas - self.d[i]) > 0.5 * self.d[i]:
                continue                                  # implausible detection: ignore
            self.d[i] += k * (meas - self.d[i])

    def _camera_zero(self, cam, tilt, dt):
        """Camera segment headings -> accelerometer zero offsets (replaces --zero-secs).

        The camera sees the true thigh and shank headings (from the knee and arm-end markers), so
        thigh_zero and knee_zero follow directly:
            phi_thigh = s*(tilt_thigh - thigh_zero)   ->  thigh_zero = tilt_thigh - s*phi_thigh
            knee      = s*(tilt_shank - tilt_thigh) - knee_zero  ->  knee_zero = s*(tilt_shank - tilt_thigh) - knee
        Updated only while both sensors are still (camera latency then does not matter), smoothed over
        cam_tau, and a single frame cannot jump the zero by more than 20 degrees once it is set."""
        if (not self.cam_zero or cam is None or getattr(cam, "confidence", 0.0) < 0.5
                or cam.thigh_heading_deg is None or cam.shank_heading_deg is None):
            return
        if self.cam_zero_set and not (self.still[0] and self.still[1]):
            return
        cam_knee = wrap_deg(cam.shank_heading_deg - cam.thigh_heading_deg)
        tz = wrap_deg(tilt[0] - self.s * cam.thigh_heading_deg)
        kz = wrap_deg(self.s * (tilt[1] - tilt[0]) - cam_knee)
        if not self.cam_zero_set:
            self.thigh_zero, self.knee_zero, self.cam_zero_set = tz, kz, True
            self.zeroing = False                       # the camera replaces --zero-secs
            log("camera set the zeros: thigh %+.1f deg, knee %+.1f deg" % (tz, kz))
            return
        k = dt / (self.cam_tau + dt)
        for attr, target in (("thigh_zero", tz), ("knee_zero", kz)):
            cur = getattr(self, attr)
            err = wrap_deg(target - cur)
            if abs(err) <= 20.0:
                setattr(self, attr, wrap_deg(cur + k * err))

    # --- helpers -----------------------------------------------------------
    def _plane(self, a3):
        """(sagittal, vertical) components after the axis map, same as tiltDeg() in main.cpp."""
        a = self.a
        return [a.sag_sign * a3[a.sag_axis], a.vert_sign * a3[a.vert_axis]]

    def _tilt(self, a3):
        sag, vert = self._plane(a3)
        return math.degrees(math.atan2(sag, vert))

    def _kinematic(self, tilt):
        """Where the knee-hinge geometry says each sensor is (metres)."""
        phi_t = self.s * (tilt[0] - self.thigh_zero)
        knee = wrap_deg(self.s * (tilt[1] - tilt[0]) - self.knee_zero)
        phi_s = phi_t + knee
        rt, rs = math.radians(phi_t), math.radians(phi_s)
        return [[-self.d[0] * math.cos(rt), self.d[0] * math.sin(rt)],
                [self.d[1] * math.cos(rs), -self.d[1] * math.sin(rs)]]

    # --- main step -----------------------------------------------------------
    def update(self, t_ms, raw, cam=None):
        """raw = 6 floats (g), cam = optional camera fix. Returns [(thigh_x, thigh_y), (shank_x, shank_y)] in cm."""
        a = self.a
        dt = 0.01 if self.last_t is None else (t_ms - self.last_t) / 1000.0
        self.last_t = t_ms
        if dt <= 0.0 or dt > 0.5:
            dt = 0.01
        self._use_camera(cam, dt)

        first = self.y is None
        if first:
            self.y = list(raw)
            self.g = [self._plane(raw[0:3]), self._plane(raw[3:6])]
        else:
            rc = 1.0 / (2.0 * math.pi * CUTOFF_HZ)
            al = dt / (rc + dt)
            for i in range(6):
                self.y[i] += al * (raw[i] - self.y[i])
        tilt = [self._tilt(self.y[0:3]), self._tilt(self.y[3:6])]
        self.knee_deg = wrap_deg(self.s * (tilt[1] - tilt[0]) - self.knee_zero)
        kin = self._kinematic(tilt)
        if first:
            self.p = [list(kin[0]), list(kin[1])]

        self._auto_zero(t_ms, tilt)

        for i in range(2):
            r3 = raw[3 * i:3 * i + 3]
            yp = self._plane(self.y[3 * i:3 * i + 3])
            self.win[i].append(r3)

            # ZUPT detector: every axis flat over the window, and |a| close to 1 g.
            w = self.win[i]
            still = False
            if a.zupt and len(w) == w.maxlen:
                still = all(max(s[k] for s in w) - min(s[k] for s in w) <= a.still_g for k in range(3))
                mag = math.sqrt(sum(c * c for c in r3))
                still = still and abs(mag - 1.0) <= 0.15
            self.still[i] = still

            # Slow gravity estimate; snaps to the reading while still (re-anchors the orientation).
            tau_g = 0.1 if still else a.gravity_tau
            ag = dt / (tau_g + dt)
            self.g[i][0] += ag * (yp[0] - self.g[i][0])
            self.g[i][1] += ag * (yp[1] - self.g[i][1])

            # Linear acceleration = reading - gravity, rotated into the world frame by the tilt of g.
            dx, dz = yp[0] - self.g[i][0], yp[1] - self.g[i][1]
            tg = math.atan2(self.g[i][0], self.g[i][1])
            ax = self.s * G * (-dx * math.cos(tg) + dz * math.sin(tg))
            ay = G * (dx * math.sin(tg) + dz * math.cos(tg))
            # Real leg-extension acceleration is ~0.03-0.07 g (about one sensor step); anything
            # much larger is tilt leaking through the gravity filter, so cap it.
            amax = ACC_LIMIT_G * G
            ax = max(-amax, min(amax, ax))
            ay = max(-amax, min(amax, ay))

            if first:
                continue
            # delta-v = a*dt, then position.
            p, v = self.p[i], self.v[i]
            p[0] += v[0] * dt + 0.5 * ax * dt * dt
            p[1] += v[1] * dt + 0.5 * ay * dt * dt
            v[0] += ax * dt
            v[1] += ay * dt

            # Knee-hinge constraint: complementary filter towards the geometric position.
            if a.fuse_tau > 0:
                w0 = min(1.0 / a.fuse_tau, 0.5 / dt)   # per-sample Euler step is unstable above ~0.85/dt
                for k in range(2):
                    r = kin[i][k] - p[k]
                    p[k] += 2.0 * w0 * dt * r
                    v[k] += w0 * w0 * dt * r

            if still:
                v[0] = v[1] = 0.0
            if not all(math.isfinite(c) for c in (p[0], p[1], v[0], v[1])):
                p[0], p[1] = kin[i]          # never let a bad value poison the stream
                v[0] = v[1] = 0.0

        self._camera_zero(cam, tilt, dt)
        return [(100.0 * self.p[0][0], 100.0 * self.p[0][1]),
                (100.0 * self.p[1][0], 100.0 * self.p[1][1])]

    def _auto_zero(self, t_ms, tilt):
        """--zero-secs: average the first N seconds (leg straight, thigh horizontal) into the zeros."""
        if not self.zeroing:
            return
        if self.zero_end is None:
            self.zero_end = t_ms + 1000.0 * self.a.zero_secs
            log("zeroing: hold the leg straight for %.0f s..." % self.a.zero_secs)
        knee_raw = wrap_deg(self.s * (tilt[1] - tilt[0]))
        for k, ang in enumerate((tilt[0], knee_raw)):
            self.z_sin[k] += math.sin(math.radians(ang))
            self.z_cos[k] += math.cos(math.radians(ang))
        if t_ms >= self.zero_end:
            self.thigh_zero = math.degrees(math.atan2(self.z_sin[0], self.z_cos[0]))
            self.knee_zero = math.degrees(math.atan2(self.z_sin[1], self.z_cos[1]))
            self.zeroing = False
            kin = self._kinematic(tilt)
            self.p = [list(kin[0]), list(kin[1])]
            self.v = [[0.0, 0.0], [0.0, 0.0]]
            log("zero set: thigh %+.2f deg, knee %+.2f deg" % (self.thigh_zero, self.knee_zero))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/cu.usbmodem1101 (default: auto)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--pretty", action="store_true", help="human-readable output with sample rate")
    ap.add_argument("--log", help="append every sample line to this CSV file")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")

    pg = ap.add_argument_group("sensor position (appended as thigh_px,thigh_py,shank_px,shank_py in cm)")
    pg.add_argument("--no-position", action="store_true", help="plain 7-field lines, no position columns")
    pg.add_argument("--thigh-sensor-cm", type=float, default=15.0, help="knee -> thigh sensor, along the thigh (default 15)")
    pg.add_argument("--shank-sensor-cm", type=float, default=15.0, help="knee -> shank sensor, along the shin (default 15)")
    pg.add_argument("--fuse-tau", type=float, default=0.05, help="seconds; small = trust the knee-hinge geometry, large = trust the integrated acceleration (0 = pure integration, drifts badly)")
    pg.add_argument("--gravity-tau", type=float, default=0.3, help="seconds; low-pass that separates gravity from motion")
    pg.add_argument("--no-zupt", action="store_true", help="disable zero-velocity updates")
    pg.add_argument("--still-g", type=float, default=0.07, help="ZUPT: max change per axis (g) over 0.15 s to count as still")
    pg.add_argument("--sag-axis", type=int, default=0, choices=(0, 1, 2), help="sensor axis in the leg's plane of motion (main.cpp kSagAxis)")
    pg.add_argument("--vert-axis", type=int, default=2, choices=(0, 1, 2), help="sensor axis pointing up at rest (main.cpp kVertAxis)")
    pg.add_argument("--sag-sign", type=float, default=1.0, choices=(-1.0, 1.0))
    pg.add_argument("--vert-sign", type=float, default=1.0, choices=(-1.0, 1.0))
    pg.add_argument("--knee-sign", type=float, default=1.0, choices=(-1.0, 1.0), help="-1 if bending the knee makes the angle negative")
    pg.add_argument("--thigh-zero", type=float, default=0.0, help="thigh tilt (deg) when the thigh is horizontal")
    pg.add_argument("--knee-zero", type=float, default=0.0, help="knee reading (deg) when the leg is straight")
    pg.add_argument("--zero-secs", type=float, default=0.0, help="if >0, hold the leg straight and horizontal for this long at start to set both zeros")

    cg = ap.add_argument_group("camera (ArUco markers, see camera_origin.py); needs: pip install opencv-contrib-python")
    cg.add_argument("--camera", type=int, nargs="?", const=0, default=None, metavar="N",
                    help="use webcam N (default 0): knee + arm-end ArUco markers give the leg angles (auto-zero) and arm lengths")
    cg.add_argument("--marker-cm", type=float, default=5.0, help="printed side length of the ArUco markers (cm)")
    cg.add_argument("--cam-flip-x", action="store_true", help="camera is on the other side of the leg (mirrors x)")
    cg.add_argument("--cam-tau", type=float, default=2.0, help="seconds; smoothing of the camera measurements")
    cg.add_argument("--no-cam-zero", action="store_true", help="do not let the camera set the accelerometer zero offsets")
    cg.add_argument("--cam-sensor-markers", action="store_true",
                    help="also use optional markers 10/11 placed next to the sensors (learns the sensor-to-knee distances)")
    args = ap.parse_args()
    args.zupt = not args.no_zupt
    args.cam_zero = not args.no_cam_zero

    tracker = None
    if args.camera is not None:
        try:
            import camera_origin  # lazy: OpenCV is only needed when --camera is used
        except ImportError as e:
            sys.exit(f"--camera needs camera_origin.py next to this file and OpenCV ({e}). "
                     "Install with:  pip install opencv-contrib-python")
        tracker = camera_origin.CameraTracker(cam_index=args.camera, marker_cm=args.marker_cm,
                                              flip_x=args.cam_flip_x,
                                              thigh_id=10 if args.cam_sensor_markers else None,
                                              shank_id=11 if args.cam_sensor_markers else None)
        if not tracker.start():
            log(f"camera {args.camera} could not be opened ({tracker.error}); continuing without it")
            tracker = None

    if args.list:
        for p in serial.tools.list_ports.comports():
            print(f"{p.device}  {p.description}")
        return

    logf = open(args.log, "a", buffering=1) if args.log else None
    out = sys.stdout

    while True:  # reconnect loop: survives the cable being unplugged
        port = args.port or find_port()
        if not port:
            log("no Arduino found yet (is it plugged in?). Retrying...")
            time.sleep(1.0)
            continue
        try:
            ser = serial.Serial(port, args.baud, timeout=1)
        except serial.SerialException as e:
            log(f"cannot open {port}: {e}  (Serial Monitor still open?) Retrying...")
            time.sleep(1.0)
            continue

        log(f"connected to {port} at {args.baud} baud")
        n, t_rate, n_rate = 0, time.monotonic(), 0
        est = None if args.no_position else PositionEstimator(args)  # fresh on every (re)connect: t_ms restarts
        try:
            time.sleep(2.0)            # most Arduinos reboot when the port opens
            ser.reset_input_buffer()
            ser.write(b"S")            # start streaming (t_ms restarts at 0)
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="ignore").strip()
                if line.startswith("#"):
                    log("arduino: " + line[1:].strip())
                    continue
                vals = parse(line)
                if vals is None:
                    continue  # partial line at startup, noise, etc.
                cam = tracker.latest() if tracker else None
                pos = est.update(vals[0], vals[1:], cam) if est else None
                if pos:
                    line = "%s,%.2f,%.2f,%.2f,%.2f" % (line, pos[0][0], pos[0][1], pos[1][0], pos[1][1])
                if logf:
                    logf.write(line + "\n")
                n += 1
                n_rate += 1
                if args.pretty:
                    now = time.monotonic()
                    if now - t_rate >= 1.0:
                        hz = n_rate / (now - t_rate)
                        t_rate, n_rate = now, 0
                        t, tx, ty, tz, sx, sy, sz = vals
                        txt = (f"t={t:8.0f} ms  thigh=[{tx:+.3f} {ty:+.3f} {tz:+.3f}]  "
                               f"shank=[{sx:+.3f} {sy:+.3f} {sz:+.3f}]  {hz:5.1f} Hz")
                        if pos:
                            txt += (f"  pos cm thigh=({pos[0][0]:+6.1f},{pos[0][1]:+6.1f}) "
                                    f"shank=({pos[1][0]:+6.1f},{pos[1][1]:+6.1f})  knee={est.knee_deg:+5.1f} deg")
                            if tracker:
                                ck = cam.camera_knee_deg if cam else None
                                txt += (f"  cam: knee={'--' if ck is None else format(ck, '+5.1f')} deg "
                                        f"d=({100 * est.d[0]:.1f},{100 * est.d[1]:.1f}) cm "
                                        f"conf={cam.confidence if cam else 0.0:.2f}")
                                arms = tracker.viewer_args()
                                if arms:
                                    txt += f"  arms: {arms}"
                        out.write(txt + "\n")
                        out.flush()
                else:
                    out.write(line + "\n")
                    out.flush()
        except serial.SerialException as e:
            log(f"lost the port ({e}). Reconnecting...")
            try:
                ser.close()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)
        except (BrokenPipeError, KeyboardInterrupt):
            # Viewer closed the pipe, or Ctrl+C: stop the Arduino and exit quietly.
            try:
                ser.write(b"X")
                ser.close()
            except Exception:  # noqa: BLE001
                pass
            if tracker:
                tracker.stop()
            try:
                sys.stdout = None  # avoid a second BrokenPipeError at shutdown
            finally:
                return


if __name__ == "__main__":
    main()
