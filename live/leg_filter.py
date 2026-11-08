#!/usr/bin/env python3
"""leg_filter.py - sits between serial_bridge.py (or fake_pi.py) and the viewer.

    leg_filter.py --state STATE.json -- <data source command ...>

main.cpp is not touched. Its axis map is compile-time and it only knows the two
accelerometers, so this filter rewrites the sample lines it receives.

1) Leg choice (STATE.json, re-read ~10x/s, written by leg_panel.py / run_live.py):
     {"leg", "invert": bool, "rot180": bool, "assist": bool}
   invert : negate the sagittal axis (x) of BOTH sensors -> knee bend direction flips
   rot180 : negate x and z of the SHANK sensor -> shin tilt +180 deg

2) CAMERA ASSIST (the camera helps the motion sensors). leg_panel.py publishes the
   camera's knee angle (from the ArUco markers) to STATE.json.cam. Accelerometer-only
   knee angle has an unknown zero offset, tilt drift and 2.7 deg steps; the camera
   knee angle is absolute but slow/laggy. A complementary filter combines them:

        knee_out = knee_accel + offset,   offset tracks (knee_camera - knee_accel)
                                          with a ~0.4 s time constant

   i.e. slow motion and the zero follow the camera, quick motion follows the sensors.
   The correction is applied by rotating the shank accelerometer vector in the
   sagittal (x,z) plane, so main.cpp's own math (tilt, low-pass, knee = shank-thigh,
   quality from |a|) is unchanged and simply sees the corrected angle. The camera
   also fixes the bend direction automatically (accelerometer sign vs camera sign are
   correlated while you move) and the camera's own mirror sign (flexion is >= 0).
   If the camera drops out the last offset is held. Turn off with assist=false.

Line format in/out: t_ms,tx,ty,tz,sx,sy,sz[,thigh_px,thigh_py,shank_px,shank_py]

Also: emits a blank heartbeat line when the source is silent (the viewer ignores blank
lines) so the viewer can always quit; stops the source when the viewer closes the pipe;
writes STATE.json.stats {rate, wall, fusion, offset_deg, ...} for the panel.
"""
import json
import math
from collections import deque
import os
import select
import signal
import subprocess
import sys
import time

KSAG, KVERT = 0, 2          # main.cpp kSagAxis / kVertAxis (x / z)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class Fusion:
    TAU_OFFSET = 0.6      # s: how fast the offset follows the camera
    CAM_LAG = 0.12        # s: typical webcam latency; the accelerometer angle is compared with the
                          #    camera angle from this long ago so motion does not bias the offset
    RC = 1.0 / (2.0 * math.pi * 5.0)   # same 5 Hz low-pass as main.cpp
    CAM_MAX_AGE = 0.25    # s
    TAU_STAT = 6.0        # s window for the sign decision

    def __init__(self):
        self.y = None          # low-passed [tx,ty,tz,sx,sy,sz]
        self.o = None          # offset (deg), None until the first camera lock
        self.s = 1.0           # accelerometer sign relative to the camera
        self.cs = 1.0          # camera sign (so that flexion is >= 0)
        self.cmean = 0.0
        self.stat = None       # EW means/variances for the correlation
        self.last_t = None
        self.cam = None
        self.fresh = False
        self.status = "waiting"
        self.hist = deque()    # (t_ms, a_lp) for the latency compensation
        self.rej = 0

    def set_cam(self, d):
        self.cam = d

    def _cam_value(self):
        d = self.cam
        if not d or d.get("knee") is None:
            return None
        if time.time() - d.get("wall", 0) > self.CAM_MAX_AGE or d.get("conf", 0) < 0.5:
            return None
        return float(d["knee"])

    def process(self, v, t_ms):
        """v: 6 floats (g). Returns corrected 6 floats."""
        dt = 0.01
        if self.last_t is not None:
            d = (t_ms - self.last_t) / 1000.0
            if 0.002 < d < 0.05:
                dt = d
        self.last_t = t_ms
        if self.y is None:
            self.y = list(v)
        else:
            al = dt / (self.RC + dt)
            for i in range(6):
                self.y[i] += al * (v[i] - self.y[i])

        def tilt(a, base):
            return math.degrees(math.atan2(a[base + KSAG], a[base + KVERT]))
        a_inst = wrap(tilt(v, 3) - tilt(v, 0))
        a_lp = wrap(tilt(self.y, 3) - tilt(self.y, 0))

        self.hist.append((t_ms, a_lp))
        while len(self.hist) > 2 and self.hist[1][0] <= t_ms - 1000.0 * self.CAM_LAG:
            self.hist.popleft()
        a_cmp = self.hist[0][1]                  # a_lp about CAM_LAG ago (matches the camera frame)

        c = self._cam_value()
        self.fresh = c is not None
        if c is not None:
            # camera sign: flexion should be >= 0 (never hyperextends past ~8 deg)
            self.cmean += dt / 3.0 * (c - self.cmean)
            ncs = self.cs
            if self.cmean > 8.0:
                ncs = 1.0
            elif self.cmean < -8.0:
                ncs = -1.0
            if ncs != self.cs:
                self.cs, self.o, self.stat = ncs, None, None
            c *= self.cs
            # accelerometer sign: correlate accel knee with camera knee while moving
            if self.stat is None:
                self.stat = [a_cmp, c, 0.0, 0.0, 0.0]            # ma, mc, cov, va, vc
            else:
                al = dt / self.TAU_STAT
                ma, mc, cov, va, vc = self.stat
                da, dc = a_cmp - ma, c - mc
                ma += al * da
                mc += al * dc
                cov += al * (da * dc - cov)
                va += al * (da * da - va)
                vc += al * (dc * dc - vc)
                self.stat = [ma, mc, cov, va, vc]
                if va > 25.0 and vc > 25.0:
                    corr = cov / math.sqrt(va * vc)
                    if abs(corr) > 0.6:
                        ns = 1.0 if corr > 0 else -1.0
                        if ns != self.s:
                            self.s, self.o = ns, None
            err = wrap(c - self.s * a_cmp)
            if self.o is None:
                self.o = err
            elif abs(wrap(err - self.o)) < 60.0:             # ignore wild detections ...
                self.rej = 0
                self.o = wrap(self.o + dt / (self.TAU_OFFSET + dt) * wrap(err - self.o))
            else:
                self.rej += 1
                if self.rej > 40:                            # ... unless they persist (>0.4 s): re-lock
                    self.o, self.rej = err, 0

        if self.o is None:
            self.status = "waiting"
            return v
        self.status = "locked" if self.fresh else "holding"
        target = wrap(self.s * a_inst + self.o)
        delta = math.radians(wrap(target - a_inst))
        cd, sd = math.cos(delta), math.sin(delta)
        x, z = v[3 + KSAG], v[3 + KVERT]
        out = list(v)
        out[3 + KSAG] = x * cd + z * sd                       # rotate the shank vector in the
        out[3 + KVERT] = z * cd - x * sd                      # sagittal plane: tilt += delta
        return out


def load_state(path, prev, prev_mtime):
    try:
        m = os.stat(path).st_mtime
        if m == prev_mtime:
            return prev, prev_mtime
        with open(path) as f:
            d = json.load(f)
        return {"invert": bool(d.get("invert")), "rot180": bool(d.get("rot180")),
                "assist": bool(d.get("assist", True))}, m
    except (OSError, ValueError):
        return prev, prev_mtime


def transform(line, st, fusion):
    f = line.rstrip("\r\n").split(",")
    if len(f) < 7:
        return line
    try:
        if st["invert"]:
            for i in (1, 4, 7, 9):          # thigh_x, shank_x, thigh_px, shank_px
                if i < len(f):
                    f[i] = _neg(f[i])
        if st["rot180"]:
            for i in (4, 6):                # shank x and z
                f[i] = _neg(f[i])
        if st["assist"] and fusion is not None:
            v = [float(x) for x in f[1:7]]
            out = fusion.process(v, float(f[0]))
            if out is not v:
                f[1:7] = ["%.4f" % x for x in out]
        elif not (st["invert"] or st["rot180"]):
            return line
    except ValueError:
        return line
    return ",".join(f) + "\n"


def write_stats(path, rate, fusion):
    d = {"rate": rate, "wall": time.time()}
    if fusion is not None:
        d.update(fusion=fusion.status, offset_deg=None if fusion.o is None else fusion.o,
                 accel_sign=fusion.s, cam_sign=fusion.cs, cam_fresh=fusion.fresh)
    try:
        with open(path + ".tmp", "w") as f:
            json.dump(d, f)
        os.replace(path + ".tmp", path)
    except OSError:
        pass


def _neg(s):
    return "%.4f" % -float(s)


def main():
    argv = sys.argv[1:]
    if "--state" not in argv or "--" not in argv:
        sys.exit(__doc__)
    state_path = argv[argv.index("--state") + 1]
    cmd = argv[argv.index("--") + 1:]
    if not cmd:
        sys.exit("no source command after --")

    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    stop = {"now": False}

    def _sig(signum, frame):
        stop["now"] = True
    for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sg, _sig)

    fusion = Fusion()
    cam_path = state_path + ".cam"
    st, mt = {"invert": False, "rot180": False, "assist": True}, None
    st, mt = load_state(state_path, st, mt)
    last_poll = last_cam = time.monotonic()
    cam_mt = None
    n_samples, stats_t, stats_path = 0, time.monotonic(), state_path + ".stats"
    buf = b""
    out = sys.stdout.buffer
    fd = child.stdout.fileno()
    try:
        while not stop["now"]:
            r, _, _ = select.select([fd], [], [], 0.2)
            now = time.monotonic()
            if now - last_poll > 0.1:
                st, mt = load_state(state_path, st, mt)
                last_poll = now
            if now - last_cam > 0.02:                # camera knee angle from the panel
                last_cam = now
                try:
                    m = os.stat(cam_path).st_mtime
                    if m != cam_mt:
                        cam_mt = m
                        fusion.set_cam(read_json(cam_path))
                except OSError:
                    pass
            if now - stats_t >= 0.5:                 # readout for the panel
                write_stats(stats_path, n_samples / (now - stats_t), fusion)
                n_samples, stats_t = 0, now
            if not r:
                out.write(b"\n")            # heartbeat: keeps the viewer's fgets from blocking forever
                out.flush()
                if child.poll() is not None:
                    break
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                if ln.count(b",") >= 6:
                    n_samples += 1
                out.write(transform(ln.decode("ascii", "replace") + "\n", st, fusion).encode("ascii", "replace"))
            out.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)    # bridge sends 'X' and releases the port
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
    os._exit(0)   # skip interpreter-shutdown flush of the (possibly closed) pipe


if __name__ == "__main__":
    sys.exit(main())
