#!/usr/bin/env python3
"""verify_pipeline.py - prove Arduino -> serial_bridge.py -> viewer-pipe works, with NO hardware.

Starts live/sim_arduino.py (a pty that behaves like brace_sensors_3.ino), runs
`python3 serial_bridge.py --port <pty>` against it and checks what the C++ viewer would see:

  1. bridge with position columns (default)         ~5 s of data
  2. bridge with --no-position                      ~5 s of data
  3. viewer-style launch: /bin/sh -c "<cmd>" (what popen() does), then close the pipe
  4. INFO only: viewer closes the pipe while the board is silent (documents a known limitation)

Per run it checks: every stdout line parses with main.cpp's sscanf("%lf,%f,%f,%f,%f,%f,%f")
(extra fields allowed), line length < 256 (fgets buffer), sample rate 90-110 Hz, t_ms monotonic,
|g| sane, knee angle (main.cpp's math) sweeps ~0..90 deg, no stray lines on stdout, the bridge
sent 'S' on connect and 'X' when the pipe closed, exits promptly and releases the pty.

It never touches a real serial port: the bridge is always given the simulator's pty via --port.

    python3 live/verify_pipeline.py [--secs 5] [--bridge path/to/serial_bridge.py] [--python python3]

Prints a PASS/FAIL table; exit status 1 if anything FAILs.
"""
import argparse
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "sim_arduino.py")
DEFAULT_BRIDGE = os.path.join(os.path.dirname(HERE), "serial_bridge.py")

# ---- main.cpp's parser: std::sscanf(line, "%lf,%f,%f,%f,%f,%f,%f") == 7 ----
_NUM = r"[-+]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf(?:inity)?|nan)"
_SSCANF7 = re.compile(r"\s*(%s)(?:\s*,\s*(%s)){6}" % (_NUM, _NUM), re.I)
_SSCANF7_CAP = re.compile(r"\s*(%s)" % _NUM + (r"\s*,\s*(%s)" % _NUM) * 6, re.I)


def sscanf7(line):
    """Emulates the viewer's sscanf: returns 7 floats, or None if it would not return 7."""
    m = _SSCANF7_CAP.match(line)
    return [float(x) for x in m.groups()] if m else None


class Result:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail="", info=False):
        status = "INFO" if info else ("PASS" if ok else "FAIL")
        self.rows.append((status, name, detail))

    def failed(self):
        return any(r[0] == "FAIL" for r in self.rows)

    def table(self):
        w = max(len(r[1]) for r in self.rows)
        out = []
        for status, name, detail in self.rows:
            out.append("  %-4s  %-*s  %s" % (status, w, name, detail))
        return "\n".join(out)


def start_sim(cmd_log, extra=()):
    p = subprocess.Popen([sys.executable, SIM, "--duration", "120", "--cmd-log", cmd_log, *extra],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    port = p.stdout.readline().strip()
    if not port.startswith("/dev/"):
        p.kill()
        raise RuntimeError("sim_arduino did not print a pty path: %r (%s)" % (port, p.stderr.read()))
    return p, port


def stop(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def holders(port, ignore_pids):
    """PIDs (other than ignore_pids) that have the pty open; None if lsof is unavailable."""
    try:
        r = subprocess.run(["lsof", "-t", port], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    pids = {int(x) for x in r.stdout.split() if x.isdigit()}
    return sorted(pids - set(ignore_pids))


def read_cmds(path):
    try:
        with open(path) as f:
            return [ln.split()[1] for ln in f if len(ln.split()) == 2]
    except OSError:
        return []


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class Collector(threading.Thread):
    """Reads the bridge's stdout line by line (like fgets), stamping arrival times."""

    def __init__(self, f):
        super().__init__(daemon=True)
        self.f, self.lines, self.stop_flag = f, [], False

    def run(self):
        for raw in iter(self.f.readline, b""):
            self.lines.append((time.perf_counter(), raw))
            if self.stop_flag:
                break


def collect(proc, secs, first_timeout=10.0, max_lines=None):
    c = Collector(proc.stdout)
    c.start()
    t0 = time.perf_counter()
    while not c.lines and time.perf_counter() - t0 < first_timeout and proc.poll() is None:
        time.sleep(0.02)
    if c.lines:
        t_first = c.lines[0][0]
        while time.perf_counter() - t_first < secs and proc.poll() is None:
            if max_lines and len(c.lines) >= max_lines:
                break
            time.sleep(0.02)
    c.stop_flag = True
    c.join(2.0)
    return c.lines


def knee_sweep(rows):
    """main.cpp's chain (5 Hz low-pass, tilt = atan2(x, z), knee = shank - thigh); skips the first 0.8 s."""
    kPi = math.pi
    rc = 1.0 / (2.0 * kPi * 5.0)
    y, last_t, knees = None, None, []
    for r in rows:
        t, v = r[0], r[1:7]
        dt = 0.01 if last_t is None else (t - last_t) / 1000.0
        last_t = t
        if dt <= 0 or dt > 0.5:
            dt = 0.01
        al = dt / (rc + dt)
        y = list(v) if y is None else [y[i] + al * (v[i] - y[i]) for i in range(6)]
        k = math.degrees(math.atan2(y[3], y[5]) - math.atan2(y[0], y[2]))
        knees.append(((k + 180.0) % 360.0) - 180.0)
        if t - rows[0][0] < 800:
            knees.pop()
    return (min(knees), max(knees)) if knees else (float("nan"), float("nan"))


def check_stream(res, tag, lines, expect_fields):
    """All checks on the captured stdout lines."""
    text = [raw.decode("ascii", "replace") for _, raw in lines]
    n = len(text)
    res.add(tag + "got data", n >= 100, "%d lines" % n)
    if n < 10:
        return
    bad = [s for s in text if sscanf7(s) is None]
    res.add(tag + "every line parses (sscanf 7 fields)", not bad,
            "%d/%d ok" % (n - len(bad), n) + ("; first bad: %r" % bad[0] if bad else ""))
    longest = max(len(s) for s in text)
    res.add(tag + "line length fits fgets(256)", longest <= 255, "longest %d bytes" % longest)
    res.add(tag + "lines end with LF, no CR", all(s.endswith("\n") and "\r" not in s for s in text))
    fields = {len(s.strip().split(",")) for s in text}
    res.add(tag + "field count", fields == {expect_fields}, "got %s, want %d" % (sorted(fields), expect_fields))
    stray = [s for s in text if s.lstrip().startswith("#") or not s.strip()]
    res.add(tag + "no stray status/blank lines on stdout", not stray, "%d stray" % len(stray))

    rows = [sscanf7(s) for s in text if sscanf7(s)]
    arr = [t for (t, _) in lines]
    span = arr[-1] - arr[0]
    hz_arr = (n - 1) / span if span > 0 else 0.0
    res.add(tag + "rate by arrival time 90-110 Hz", 90.0 <= hz_arr <= 110.0, "%.1f Hz over %.1f s" % (hz_arr, span))

    ts = [r[0] for r in rows]
    drops = [i for i in range(1, len(ts)) if ts[i] <= ts[i - 1]]
    late = [i for i in drops if i >= 5]
    note = ""
    if drops and not late:
        note = " (%d stale pre-S line(s) at start: t_ms restarts at 0 after S; main.cpp resets dt, harmless)" % len(drops)
    res.add(tag + "t_ms monotonic", not late, "last backwards step at line %s%s" % (drops[-1] if drops else "-", note))
    s0 = (drops[-1] if drops else -1) + 1
    seg = ts[s0:]
    if len(seg) > 10:
        hz_t = (len(seg) - 1) / ((seg[-1] - seg[0]) / 1000.0)
        res.add(tag + "rate by t_ms 90-110 Hz", 90.0 <= hz_t <= 110.0, "%.1f Hz" % hz_t)
        gaps = max(b - a for a, b in zip(seg, seg[1:]))
        res.add(tag + "no gap > 100 ms (viewer timeout is 500 ms)", gaps <= 100, "max step %d ms" % gaps)
    res.add(tag + "first t_ms small (S reset it)", ts[s0] < 500 if s0 < len(ts) else False, "first t_ms %d" % ts[s0])

    mags = [math.sqrt(sum(c * c for c in r[1 + 3 * k:4 + 3 * k])) for r in rows for k in (0, 1)]
    comp = max(abs(c) for r in rows for c in r[1:7])
    res.add(tag + "|g| sane (0.6..1.5, mean ~1)", min(mags) > 0.6 and max(mags) < 1.5 and 0.9 < sum(mags) / len(mags) < 1.1,
            "min %.2f max %.2f mean %.3f; max |component| %.3f" % (min(mags), max(mags), sum(mags) / len(mags), comp))
    kmin, kmax = knee_sweep(rows)
    res.add(tag + "viewer knee angle sweeps ~0..90 deg", kmin <= 10 and kmax >= 80, "min %.1f max %.1f deg" % (kmin, kmax))

    if expect_fields == 11:
        pos = [[float(x) for x in s.strip().split(",")[7:11]] for s in text if len(s.strip().split(",")) == 11]
        pmax = max(abs(c) for p in pos for c in p)
        res.add(tag + "position columns finite, |p| < 60 cm", all(math.isfinite(c) for p in pos for c in p) and pmax < 60,
                "max |p| %.1f cm" % pmax)


def run_bridge_case(res, tag, extra, secs, py, bridge):
    tag = "[%s] " % tag
    cmdlog = tempfile.mktemp(prefix="sim_cmds_")
    sim = br = None
    try:
        sim, port = start_sim(cmdlog)
        cmd = [py, bridge, "--port", port] + extra
        br = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(bridge))
        errbuf = []
        te = threading.Thread(target=lambda: errbuf.append(br.stderr.read()), daemon=True)
        te.start()
        t_start = time.perf_counter()
        lines = collect(br, secs)
        first_at = (lines[0][0] - t_start) if lines else float("nan")
        res.add(tag + "first sample after ~2 s reset wait", bool(lines) and 1.5 <= first_at <= 6.0, "%.2f s after launch" % first_at)
        cmds_before = read_cmds(cmdlog)
        check_stream(res, tag, lines, 7 if "--no-position" in extra else 11)
        res.add(tag + "bridge sent 'S' on connect (before close)", cmds_before[:1] == ["S"] and "X" not in cmds_before,
                "commands so far: %s" % "".join(cmds_before))
        # viewer exits: pclose() closes the read end of the pipe
        br.stdout.close()
        t_close = time.perf_counter()
        try:
            rc = br.wait(5)
            dt = time.perf_counter() - t_close
        except subprocess.TimeoutExpired:
            rc, dt = None, 5.0
        res.add(tag + "bridge exits promptly when pipe closes", rc is not None and dt < 1.5,
                "exit code %s after %.0f ms" % (rc, dt * 1000))
        res.add(tag + "clean exit status 0", rc == 0, "rc=%s" % rc)
        time.sleep(0.2)
        cmds = read_cmds(cmdlog)
        res.add(tag + "bridge sent 'X' on pipe close", cmds[-1:] == ["X"] and cmds.count("S") == 1,
                "commands: %s" % "".join(cmds))
        h = holders(port, [sim.pid])
        res.add(tag + "pty released (no other holder)", h == [] if h is not None else True,
                "holders: %s" % h if h is not None else "lsof unavailable, skipped")
        te.join(1.0)
        err = (errbuf[0] if errbuf else b"").decode("utf-8", "replace")
        res.add(tag + "no traceback / sample data on stderr", "Traceback" not in err and "Exception ignored" not in err,
                "stderr: %s" % (" | ".join(err.strip().splitlines()[-3:]) or "(empty)"))
    finally:
        stop(br)
        stop(sim)
        try:
            os.unlink(cmdlog)
        except OSError:
            pass


def run_popen_case(res, secs, py, bridge):
    """Mimic main.cpp: popen("<cmd>", "r") == /bin/sh -c "<cmd>" with stdout piped; read lines; pclose."""
    tag = "[popen] "
    cmdlog = tempfile.mktemp(prefix="sim_cmds_")
    sim = sh = None
    try:
        sim, port = start_sim(cmdlog)
        cmd = "%s %s --port %s" % (shlex.quote(py), shlex.quote(bridge), shlex.quote(port))
        sh = subprocess.Popen(["/bin/sh", "-c", cmd], stdout=subprocess.PIPE, cwd=os.path.dirname(bridge))
        lines = collect(sh, secs)
        res.add(tag + "viewer-style fgets loop got samples", len(lines) >= 100 and all(sscanf7(r.decode()) for _, r in lines),
                "%d lines parsed" % len(lines))
        sh.stdout.close()                                     # pclose(): close the pipe, then waitpid
        t0 = time.perf_counter()
        try:
            rc = sh.wait(5)
        except subprocess.TimeoutExpired:
            rc = None
        dt = time.perf_counter() - t0
        res.add(tag + "pclose returns promptly (sh + bridge gone)", rc is not None and dt < 1.5, "rc=%s after %.0f ms" % (rc, dt * 1000))
        time.sleep(0.2)
        cmds = read_cmds(cmdlog)
        res.add(tag + "S sent, X sent on close", cmds[:1] == ["S"] and cmds[-1:] == ["X"], "commands: %s" % "".join(cmds))
        h = holders(port, [sim.pid])
        res.add(tag + "pty released after pclose", h == [] if h is not None else True,
                "holders: %s" % h if h is not None else "lsof unavailable, skipped")
    finally:
        stop(sh)
        stop(sim)
        try:
            os.unlink(cmdlog)
        except OSError:
            pass


def run_silent_case(res, py, bridge):
    """INFO: board never answers. The bridge only notices a closed pipe when it next writes."""
    tag = "[silent board] "
    cmdlog = tempfile.mktemp(prefix="sim_cmds_")
    sim = br = None
    try:
        sim, port = start_sim(cmdlog, ["--mute"])
        br = subprocess.Popen([py, bridge, "--port", port], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              cwd=os.path.dirname(bridge))
        time.sleep(3.5)
        br.stdout.close()
        try:
            br.wait(3)
            exited = True
        except subprocess.TimeoutExpired:
            exited = False
        res.add(tag + "bridge exits after pipe close with no data flowing", exited,
                "STILL RUNNING 3 s after close (it only notices on its next write); viewer's exit join would hang in fgets"
                if not exited else "exited", info=True)
    finally:
        stop(br)
        stop(sim)
        try:
            os.unlink(cmdlog)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--secs", type=float, default=5.0, help="seconds of data to capture per run (default 5)")
    ap.add_argument("--bridge", default=DEFAULT_BRIDGE, help="path to serial_bridge.py")
    ap.add_argument("--python", default=sys.executable, help="python used to run the bridge")
    ap.add_argument("--skip-silent", action="store_true", help="skip the informational silent-board case")
    args = ap.parse_args()
    bridge = os.path.abspath(args.bridge)
    if not os.path.exists(bridge):
        sys.exit("bridge not found: %s" % bridge)
    r = subprocess.run([args.python, "-c", "import serial"], capture_output=True)
    if r.returncode != 0:
        sys.exit("pyserial is not installed for %s (pip install pyserial)" % args.python)

    res = Result()
    print("verify_pipeline: simulated Arduino on a pty, bridge=%s" % bridge, flush=True)
    for name, extra in (("position", []), ("no-position", ["--no-position"])):
        print("... running bridge case '%s' (~%.0f s)" % (name, args.secs + 3), flush=True)
        run_bridge_case(res, name, extra, args.secs, args.python, bridge)
    print("... running popen-style case", flush=True)
    run_popen_case(res, min(args.secs, 3.0), args.python, bridge)
    if not args.skip_silent:
        print("... running silent-board case (informational)", flush=True)
        run_silent_case(res, args.python, bridge)

    print()
    print(res.table())
    n_pass = sum(r[0] == "PASS" for r in res.rows)
    n_fail = sum(r[0] == "FAIL" for r in res.rows)
    n_info = sum(r[0] == "INFO" for r in res.rows)
    print("\n%s: %d passed, %d failed, %d info" % ("FAIL" if n_fail else "PASS", n_pass, n_fail, n_info))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
