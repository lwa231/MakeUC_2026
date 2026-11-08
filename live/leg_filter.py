#!/usr/bin/env python3
"""leg_filter.py - between the data source and the viewer: sensor lines in, knee angle out.

    leg_filter.py [--state PATH] [--auto-zero SECS] [--no-cal] -- <source command ...>

Per sample (`t_ms,tx,ty,tz,sx,sy,sz[,...]`, g): bench correction -> leg mapping (calibration sign, or
STATE invert/rot180 while uncalibrated) -> knee_estimator.KneeFilter, fused with STATE.cam frames.
STATE.tune (panel) and STATE.viewer (viewer keys) set w, resp, src; a rise in zero_seq / cal_seq runs
a quick zero / a calibration. stdout: `K,<t_ms>,<theta>,<omega>,<flags>,<w>` per sample (50 Hz when
coasting; flags 1 camera fused, 2 coasting, 4 calibrating, 8 uncalibrated, 16 reset), `M,<level>,<text>`
messages and blank heartbeat lines. STATE.stats (2 Hz) feeds the panel. Standard library only.
"""
import argparse
import os
import select
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate  # noqa: E402
from knee_estimator import KneeFilter, wrap  # noqa: E402

BENCH_PATH = os.path.join(HERE, "brace_cal.json")   # bench check result (once per brace)
LEG_CAL_PATH = os.path.join(HERE, "leg_cal.json")   # {leg: hang/anchor result}
SERIAL_LAT_S = 0.008    # best-case sample-to-arrival delay (I2C reads + one 115200-baud line)
CLOCK_CREEP = 0.005     # s per s the device-to-host clock offset may rise (crystal skew)
CAM_LATENCY_S = 0.08    # capture-to-file delay assumed for camera frames without t_cap
CAM_MIN_CONF = 0.5      # camera frames below this confidence are ignored
REANCHOR_S = 1.0       # a device->host offset this much higher for this long is a new clock, not a stall
COAST_AFTER_S = 0.1     # accel silence before coasting output starts (USB stalls are shorter)
CAL_SILENT_S = 15.0     # a calibration gives up after this long without sensor data


class LegFilter:
    def __init__(self, state_path, auto_zero, use_saved):
        self.paths = {k: state_path + k for k in ("", ".cam", ".tune", ".viewer", ".stats")}
        self.auto_zero, self.use_saved = auto_zero, use_saved
        self.bench = calibrate.load(BENCH_PATH) if use_saved else None
        self.st, self.leg, self.leg_cal, self.mapping = {}, None, None, None
        self.kf, self.reset, self.out, self.t_out = KneeFilter(), False, [], 0.0   # t_out: last K line
        self.off = self.t_last = self.t_arr = self.t0 = None       # device clock -> host clock
        self.t_dev, self.t_high = 0.0, None                        # last device time, offset-high since
        self.cal, self.last_cal, self.cal_msg, self.cal_t = None, None, "", 0.0   # running / last CalSession
        self.cam_mtime, self.cam_key, self.cam_mean, self.cam_sign = None, None, 0.0, 1.0
        self.n_acc = self.n_cam = 0
        # (file, key) -> last zero_seq / cal_seq seen; whatever the files hold at startup is old news
        self.seqs = {(p, k): int((calibrate.load(p) or {}).get(k) or 0)
                     for p in (self.paths[".tune"], self.paths[".viewer"]) for k in ("zero_seq", "cal_seq")}
        self.poll_files()
        if self.leg_cal:
            self.auto_zero = 0.0        # calibrated: a quick zero would throw the deficit away
        else:
            self.say("info", "Not calibrated: press C (or Calibrate in the panel) for Hang and Straighten.")

    def say(self, level, text):
        self.out.append("M,%s,%s\n" % (level, " ".join(str(text).split())[:150]))

    def emit(self, t, theta, omega, coast=False):
        flags = ((1 if self.kf.cam_ok else 0) | (2 if coast else 0) | (4 if self.cal else 0)
                 | (0 if self.kf.calibrated else 8) | (16 if self.reset else 0))
        self.reset, self.t_out = False, max(t, self.t_out)
        self.out.append("K,%.1f,%.2f,%.1f,%d,%.2f\n" % (self.t_out * 1000.0, theta, omega, flags, self.kf.w))

    def apply_mapping(self, force=False):
        """Calibrated: the calibration's sign and k. Otherwise STATE invert/rot180, k unknown."""
        if self.leg_cal:
            inv, k = self.leg_cal["sign"] < 0, float(self.leg_cal["k_deg"])
            rot = abs(k) > 90.0         # shank mounted upside down: rotate it so k stays small
            k = wrap(k + 180.0) if rot else k
        else:
            inv, rot, k = bool(self.st.get("invert")), bool(self.st.get("rot180")), None
        if force or (inv, rot, k) != self.mapping:
            self.mapping, old = (inv, rot, k), self.kf
            self.kf = KneeFilter(old.w, old.resp)     # the raw angle jumped: start afresh
            self.kf.set(src=old.src)
            self.kf.cam_delay, self.kf.cam_var = old.cam_delay, old.cam_var   # the camera did not change
            if k is not None:
                self.kf.set_reference(k)
            self.reset = True

    def on_line(self, line, now):
        if line.startswith("#"):
            return sys.stderr.write(line + "\n")
        f = line.split(",")
        try:
            t_dev, raw = float(f[0]) / 1000.0, tuple(float(x) for x in f[1:7])
            tx, ty, tz, sx, sy, sz = calibrate.apply_bench(self.bench, raw)
        except ValueError:
            return                      # not a sample line
        # Device clock -> host clock: lower envelope of (arrival - t_dev), so samples keep their
        # device spacing through USB bursts; it creeps up for crystal skew and re-anchors on a reset.
        # Re-anchor only on a device restart (t_ms went back) or an offset that stays high for a while
        # (bridge reconnect); a host stall also raises d for one burst, but the lower envelope rides it out.
        d = now - t_dev
        if self.off is None or t_dev < self.t_dev - 0.001:
            self.off, self.t_high = d, None
        elif d > self.off + 0.5:
            self.t_high = now if self.t_high is None else self.t_high
            if now - self.t_high > REANCHOR_S:
                self.off, self.t_high = d, None
        else:
            self.off, self.t_high = min(self.off + CLOCK_CREEP * (now - self.t_arr), d), None
        self.t_dev = t_dev
        t = t_dev + self.off - SERIAL_LAT_S
        t = t if self.t_last is None else max(t, self.t_last + 1e-4)
        self.t_last, self.t_arr, self.n_acc = t, now, self.n_acc + 1
        self.t0 = t if self.t0 is None else self.t0
        inv, rot, _ = self.mapping
        if inv:
            tx, sx = -tx, -sx
        if rot:
            sx, sy = -sx, -sy
        theta, omega = self.kf.accel(t, (tx, ty, tz, sx, sy, sz))
        if self.auto_zero and t - self.t0 >= self.auto_zero:
            self.auto_zero = 0.0
            if not self.kf.calibrated:
                self.kf.zero()
                theta, self.reset = 0.0, True
        if self.cal:
            self.feed_cal(t, raw)
        self.emit(t, theta, omega)

    def poll_cam(self, now):
        p = self.paths[".cam"]
        m = os.stat(p).st_mtime_ns if os.path.exists(p) else None    # replaced atomically, never missing
        if m is None or m == self.cam_mtime:
            return
        self.cam_mtime, d = m, calibrate.load(p) or {}
        key = d.get("seq", d.get("wall"))
        if key == self.cam_key or d.get("knee") is None or (d.get("conf") or 0.0) < CAM_MIN_CONF:
            return
        self.cam_key = key
        try:
            knee = float(d["knee"])
            t_cap = now - CAM_LATENCY_S if d.get("t_cap") is None else float(d["t_cap"])
        except (TypeError, ValueError):
            return
        if not now - 1.0 < t_cap <= now + 0.05:
            return                      # a stale file, or stamped on another clock
        # The 2D camera angle's sign depends on the side it films from; flexion is mostly positive.
        self.cam_mean += 0.01 * (knee - self.cam_mean)
        if abs(self.cam_mean) > 8.0:
            sign = 1.0 if self.cam_mean > 0 else -1.0
            if sign != self.cam_sign:          # what was learned from mirrored frames is wrong: forget it
                self.cam_sign, self.kf.cmap, self.kf.rej = sign, [0.0] * len(self.kf.cmap), []
        self.n_cam += 1
        self.kf.camera(t_cap, self.cam_sign * knee, float(d.get("conf") or 0.0))

    def poll_files(self):
        if self.cal and time.monotonic() - self.cal_t > CAL_SILENT_S:
            self.cal.state, self.cal.prompt = "failed", "Calibration stopped: no sensor data."
            self.finish_cal()
        self.st = calibrate.load(self.paths[""]) or self.st
        leg = self.st.get("leg") or "left"
        if leg != self.leg:
            self.leg = leg
            rec = (calibrate.load(LEG_CAL_PATH) or {}).get(leg) if self.use_saved else None
            self.leg_cal = rec if calibrate.is_fresh(rec) else None
            if rec and not self.leg_cal:
                self.say("warn", "Leg calibration is over 4 h old: press C to recalibrate.")
        self.apply_mapping()
        want, kf = {}, self.kf
        try:                            # oldest file first, so the newer file's values win
            for _, p, d in sorted((os.stat(p).st_mtime_ns, p, calibrate.load(p) or {}) for p in
                                  (self.paths[".tune"], self.paths[".viewer"]) if os.path.exists(p)):
                want.update({k: d[k] for k in ("w", "resp", "src") if d.get(k) is not None})
                zero_seq, cal_seq = int(d.get("zero_seq") or 0), int(d.get("cal_seq") or 0)
                if zero_seq > self.seqs[p, "zero_seq"] and kf.x is not None:
                    kf.zero()
                    self.reset = True
                    self.say("info", "Zeroed: this pose reads 0 deg. Press C for a full calibration.")
                if cal_seq > self.seqs[p, "cal_seq"] and not self.cal:
                    self.start_cal(d)
                self.seqs[p, "zero_seq"], self.seqs[p, "cal_seq"] = zero_seq, cal_seq
            src = want.get("src", kf.src) if self.st.get("assist", True) else "accel"
            src = src if src in ("fused", "accel", "camera") else kf.src
            w, resp = float(want.get("w", kf.w)), float(want.get("resp", kf.resp))
        except (OSError, TypeError, ValueError):
            return
        if (w, resp, src) != (kf.w, kf.resp, kf.src):
            self.reset = self.reset or src != kf.src
            kf.set(w, resp, src)

    def start_cal(self, d):
        anchor = d.get("cal_anchor_deg")
        sign = self.leg_cal["sign"] if self.leg_cal else (-1 if self.st.get("invert") else 1)
        try:
            self.cal = calibrate.CalSession(d.get("cal_mode") or "hang",
                                            None if anchor is None else float(anchor), self.bench, sign)
        except (ValueError, TypeError, KeyError):
            return self.say("warn", "Anchor mode needs the measured knee angle in degrees.")
        self.last_cal, self.cal_msg, self.cal_t = self.cal, "", time.monotonic()
        self.say("info", self.cal.prompt)

    def feed_cal(self, t, raw):
        s, self.cal_t = self.cal, time.monotonic()
        if s.feed(t, raw) and not s.done:
            self.say(s.level, s.message if s.message and s.message != self.cal_msg else s.prompt)
            self.cal_msg = s.message
        elif s.done:
            self.finish_cal()

    def finish_cal(self):
        s, self.cal = self.cal, None
        if s.result:
            if s.mode == "bench":
                self.bench, path, saved = s.result, BENCH_PATH, s.result
            else:
                self.leg_cal, path = s.result, LEG_CAL_PATH
                saved = dict(calibrate.load(LEG_CAL_PATH) or {}, **{self.leg: s.result})
            self.apply_mapping(force=True)
            try:
                if self.use_saved:
                    calibrate.save(path, saved)
            except OSError as e:
                self.say("warn", "Could not save the calibration: %s" % e)
        self.say(s.level, s.prompt)

    def write_stats(self, dt):
        kf, s = self.kf, self.last_cal
        cal = {"state": "idle", "prompt": "", "progress": 0.0, "result": None} if not s else {
            "state": s.state, "prompt": s.message if self.cal and s.message else s.prompt,
            "progress": s.progress, "result": s.result}
        try:
            calibrate.save(self.paths[".stats"], {
                "rate": self.n_acc / dt, "cam_hz": self.n_cam / dt, "wall": time.time(),
                "fusion": "locked" if kf.cam_ok else ("waiting" if kf.w_eff > 0 else "off"),
                "offset_deg": None if kf.x is None else kf.x[3], "w": kf.w, "resp": kf.resp,
                "src": kf.src, "cal": dict(cal, calibrated=kf.calibrated)})
        except OSError:
            pass
        self.n_acc = self.n_cam = 0


def main():
    ap = argparse.ArgumentParser(description="Sensor lines in, K/M lines out (see the module docstring).")
    ap.add_argument("--state", default="build/leg_state.json")
    ap.add_argument("--auto-zero", type=float, default=0.0, metavar="SECS",
                    help="quick zero after SECS of data, when uncalibrated (demo: 1.5)")
    ap.add_argument("--no-cal", action="store_true", help="ignore saved calibrations and save none")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <source command ...>")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.state)), exist_ok=True)

    child = subprocess.Popen(a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd, stdout=subprocess.PIPE, bufsize=0)
    for sg in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sg, signal.default_int_handler)     # raise KeyboardInterrupt: clean shutdown
    lf = LegFilter(a.state, a.auto_zero, not a.no_cal)
    out, fd, buf = sys.stdout.buffer, child.stdout.fileno(), b""
    t_poll = t_stats = t_coast = t_written = time.monotonic()
    try:
        while True:
            readable, _, _ = select.select([fd], [], [], 0.01)
            now = time.monotonic()
            if readable:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break               # the source ended
                *lines, buf = (buf + chunk).split(b"\n")
                for ln in lines:
                    lf.on_line(ln.decode("ascii", "replace").strip(), now)
            lf.poll_cam(now)
            if now - t_poll >= 0.1:
                lf.poll_files()
                t_poll = now
            if lf.t_arr is not None and now - lf.t_arr > COAST_AFTER_S and now - t_coast >= 0.02:
                t_coast, c = now, lf.kf.tick(now - SERIAL_LAT_S)
                if c:
                    lf.emit(now - SERIAL_LAT_S, c[0], c[1], coast=True)
            if now - t_stats >= 0.5:
                lf.write_stats(now - t_stats)
                t_stats = now
            if not lf.out and now - t_written > 0.2:
                lf.out.append("\n")     # heartbeat, so the viewer's blocking read returns
            if lf.out:
                out.write("".join(lf.out).encode("ascii", "replace"))
                out.flush()
                lf.out, t_written = [], now
    except (OSError, KeyboardInterrupt):
        pass                            # BrokenPipeError: the viewer closed the pipe; or a signal
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)    # serial_bridge sends 'X' and releases the port
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
    os._exit(0)     # skip the interpreter's final flush onto a possibly closed pipe


if __name__ == "__main__":
    main()
