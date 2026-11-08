#!/usr/bin/env python3
"""run_live.py - one command that loads the Arduino bridge and the 3D viewer together.

    python3 live/run_live.py                    # real board -> viewer
    python3 live/run_live.py --pretty-check 5   # bridge only, 5 s, prints the sample rate first
    python3 live/run_live.py --demo             # live/mock_sensors.py (accelerometers + camera) instead
    python3 live/run_live.py --dry-run          # just print the command that would run

It only *launches* the programs (build/viewer, live/leg_filter.py, serial_bridge.py or
live/mock_sensors.py, live/leg_panel.py). The viewer starts the data chain through --pi-cmd:

    build/viewer --state build/leg_state.json --pi-cmd "<python> live/leg_filter.py --state ... -- <source>"

Stdlib only; works from any working directory.
"""
import argparse
import glob
import importlib.util
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BRIDGE = os.path.join(REPO, "serial_bridge.py")
MOCK = os.path.join(HERE, "mock_sensors.py")
REQS = os.path.join(REPO, "requirements.txt")
BUILD_SH = os.path.join(HERE, "build_viewer.sh")
IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
VIEWER_NAME = "viewer.exe" if IS_WIN else "viewer"


def say(msg):
    print(f"[run_live] {msg}", flush=True)


def die(msg, code=2):
    print(f"[run_live] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def quote(parts):
    """Join argv into one string for the viewer's popen(): /bin/sh on mac/linux, cmd.exe on Windows."""
    if IS_WIN:
        return subprocess.list2cmdline([str(p) for p in parts])
    return " ".join(shlex.quote(str(p)) for p in parts)


# ----------------------------------------------------------------- dependencies ----

def pip_install(args):
    cmd = [sys.executable, "-m", "pip", "install", *args]
    say("running: " + quote(cmd))
    return subprocess.call(cmd) == 0


def ensure_pyserial(allow_install):
    try:
        import serial  # noqa: F401
        return True
    except ImportError:
        pass
    say(f"pyserial is not installed for {sys.executable}")
    if not allow_install:
        die("install it with:  " + quote([sys.executable, "-m", "pip", "install", "-r", REQS]))
    if not pip_install(["-r", REQS]):
        die("pip install failed; run it yourself:  " + quote([sys.executable, "-m", "pip", "install", "-r", REQS]))
    try:
        import importlib
        importlib.invalidate_caches()
        import serial  # noqa: F401
    except ImportError:
        die("pyserial still not importable after install")
    return True


def ensure_opencv(allow_install):
    """--camera only: camera_origin.py needs cv2 with the aruco module (opencv-contrib-python)."""
    def have():
        try:
            import cv2
            return hasattr(cv2, "aruco")
        except ImportError:
            return False
    if have():
        return True
    say("--camera needs OpenCV with cv2.aruco (package opencv-contrib-python)")
    if not allow_install:
        die("install it with:  " + quote([sys.executable, "-m", "pip", "install", "opencv-contrib-python"]))
    # Two OpenCV wheels fighting over `cv2` is a classic mess: remove the plain one first.
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not pip_install(["opencv-contrib-python"]) or not have():
        die("could not get cv2.aruco; run the pip command above by hand")
    return True


# ------------------------------------------------------------------------ ports ----

def serial_candidates():
    """Serial devices that look like an Arduino. Prefers the bridge's own detection."""
    found = []
    try:
        spec = importlib.util.spec_from_file_location("serial_bridge_probe", BRIDGE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # import only; main() is guarded by __name__
        best = mod.find_port()
        if best:
            found.append(best)
    except (Exception, SystemExit):  # noqa: BLE001 - pyserial missing etc.: fall back to globbing
        pass
    if IS_WIN:
        globbed = []  # COM ports cannot be globbed; the bridge's list_ports handles them
    else:
        pats = ["/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*", "/dev/ttyACM*", "/dev/ttyUSB*"]
        globbed = [p for pat in pats for p in sorted(glob.glob(pat))]
    # macOS: the call-out (cu.) node is the right one to open, never the tty. twin.
    found = [p.replace("/dev/tty.", "/dev/cu.") if IS_MAC else p for p in found]
    for p in globbed:
        if p not in found:
            found.append(p)
    return found


def find_port(override):
    if override:
        return override
    c = serial_candidates()
    if len(c) > 1:
        say("several serial devices found: " + ", ".join(c) + f" -> using {c[0]} (override with --port)")
    return c[0] if c else None


def port_holders(port):
    """[(pid, command)] of other processes that have `port` open. mac/linux via lsof."""
    if IS_WIN:
        return []
    try:
        r = subprocess.run(["lsof", "-F", "pc", "--", port], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    holders, pid = [], None
    for ln in r.stdout.splitlines():
        if ln.startswith("p"):
            pid = int(ln[1:])
        elif ln.startswith("c") and pid is not None and pid != os.getpid():
            full = ps_command(pid) or ln[1:]
            holders.append((pid, full))
            pid = None
    return holders


def ps_command(pid):
    try:
        out = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def check_port_free(port, kill_stale):
    """Return True if free. Prints a clear message and returns False if someone holds it."""
    holders = port_holders(port)
    if not holders:
        return True
    stale = [(p, c) for p, c in holders if "serial_bridge.py" in c]
    if kill_stale and stale:
        for pid, cmd in stale:
            say(f"killing stale bridge PID {pid}: {cmd}")
            try:
                os.kill(pid, signal.SIGINT)
            except OSError:
                pass
        time.sleep(1.5)
        for pid, _ in stale:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.3)
        holders = port_holders(port)
        if not holders:
            return True
    print(f"\n[run_live] PORT BUSY: {port} is already open in another program:", file=sys.stderr)
    for pid, cmd in holders:
        print(f"    PID {pid}: {cmd}", file=sys.stderr)
    print("  -> Close the Arduino IDE Serial Monitor / Serial Plotter (and any other terminal running\n"
          "     serial_bridge.py or `screen`), then run this again. Only one program can read the port.\n"
          + ("  -> If the holder is an old serial_bridge.py, re-run with --kill-stale.\n" if stale else "")
          + "  -> Or free it by hand:  kill <PID>\n", file=sys.stderr)
    return False


# ------------------------------------------------------------------------ viewer ----

def viewer_path(override):
    if override:
        return os.path.abspath(override)
    cands = [os.path.join(REPO, "build", VIEWER_NAME)]
    if IS_WIN:
        cands += [os.path.join(REPO, "build", cfg, VIEWER_NAME) for cfg in ("Release", "Debug")]
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[0]


def ensure_viewer(path, dry_run):
    if os.path.isfile(path):
        return
    say(f"viewer binary not found: {path}")
    if not os.path.isfile(BUILD_SH):
        die(f"and {BUILD_SH} does not exist yet, so it cannot be built automatically")
    if dry_run:
        say("(dry run) would build it with:  bash " + shlex.quote(BUILD_SH))
        return
    say("building it:  bash " + shlex.quote(BUILD_SH))
    if subprocess.call(["bash", BUILD_SH], cwd=REPO) != 0 or not os.path.isfile(path):
        die("viewer build failed (see output above)")


# ---------------------------------------------------------------- source command ----

LEG_FILTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leg_filter.py")
LEG_PANEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leg_panel.py")
LEG_STATE = os.path.join(REPO, "build", "leg_state.json")


def wrap_with_leg_filter(src, demo):
    """viewer -> leg_filter.py -> data source: leg mapping, calibration, camera fusion, smoothing.
    The demo zeroes itself on the mock's opening straight hold (real legs: press C or Z)."""
    return [sys.executable, LEG_FILTER, "--state", LEG_STATE] + (["--auto-zero", "1.5"] if demo else []) + ["--"] + src


def start_leg_panel(leg, cam_index, use_camera, marker_cm):
    """Write the starting leg state, then open the control panel (leg switch + camera view).
    Returns (process or None, reason or None, leg)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import leg_panel
    leg = leg_panel.initial_state(LEG_STATE, leg)
    try:
        import tkinter  # noqa: F401
    except ImportError:
        return None, "tkinter not available", leg
    argv = [sys.executable, LEG_PANEL, "--state", LEG_STATE, "--leg", leg,
            "--cam", str(cam_index), "--marker-cm", str(marker_cm)]
    if not use_camera:
        argv.append("--no-camera")
    return subprocess.Popen(argv, cwd=REPO), None, leg


def write_leg_state(leg):
    import json
    os.makedirs(os.path.dirname(LEG_STATE), exist_ok=True)
    right = leg == "right"
    with open(LEG_STATE, "w") as f:
        json.dump({"leg": leg, "invert": right, "rot180": False, "chosen": True}, f)


def source_argv(args, port):
    """argv of the data helper (what the viewer will popen)."""
    if args.demo:
        return [sys.executable, MOCK, "--scenario", args.scenario, "--cam-out", LEG_STATE + ".cam"]
    argv = [sys.executable, BRIDGE, "--port", port or "<NO-PORT-FOUND>"]
    if args.no_position:
        argv += ["--no-position"]
    if args.log:
        argv += ["--log", os.path.abspath(args.log)]
    if args.bridge_args:
        argv += shlex.split(args.bridge_args, posix=not IS_WIN)
    return argv


# ------------------------------------------------------------------- pretty-check ----

def pretty_check(argv, secs, is_bridge):
    say("PRETTY CHECK: running the data source alone for %.0f s (viewer not opened)" % secs)
    say("source: " + quote(argv))
    p = subprocess.Popen(argv, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=not IS_WIN)
    q = queue.Queue()
    err_lines = []

    def pump_out():
        for ln in p.stdout:
            q.put((time.monotonic(), ln.rstrip("\n")))
        q.put(None)

    def pump_err():
        for ln in p.stderr:
            ln = ln.rstrip("\n")
            err_lines.append(ln)
            print("    " + ln, flush=True)

    threading.Thread(target=pump_out, daemon=True).start()
    threading.Thread(target=pump_err, daemon=True).start()

    rows, t_first, deadline_start = [], None, time.monotonic()
    give_up = deadline_start + secs + 12  # bridge sleeps 2 s after opening; allow slack for a slow start
    ended = False
    try:
        while time.monotonic() < give_up:
            try:
                item = q.get(timeout=0.25)
            except queue.Empty:
                if t_first is not None and time.monotonic() - t_first >= secs:
                    break
                continue
            if item is None:
                ended = True
                break
            ts, ln = item
            rows.append((ts, ln))
            if t_first is None:
                t_first = ts
            if ts - t_first >= secs:
                break
    except KeyboardInterrupt:
        say("interrupted")
    finally:
        stop_group(p, polite=is_bridge)

    n = len(rows)
    print()
    if n == 0:
        say("NO DATA received.")
        busy = any("Resource busy" in e or "cannot open" in e for e in err_lines)
        if busy:
            say("The port looks busy: close the Arduino Serial Monitor and retry.")
        elif ended:
            say("The source exited by itself (see messages above).")
        else:
            say("Nothing arrived. If the log above says 'streaming' but no samples came, the Arduino "
                "can't see its sensors: reseat the Grove cables (thigh -> any I2C port, shank -> D2) "
                "and run  python3 live/sensor_check.py  to see the board's own self-check.\n"
                "Also check: DATA cable, sketch uploaded, right --port, board resets on open (wait ~3 s).")
        return 1
    for tag, sel in (("first", rows[:3]), ("last", rows[-2:])):
        for _, ln in sel:
            print(f"    {tag}: {ln}")
    wall = rows[-1][0] - rows[0][0]
    rate_wall = (n - 1) / wall if wall > 0 else 0.0
    rate_dev = None
    try:
        t0, t1 = float(rows[0][1].split(",")[0]), float(rows[-1][1].split(",")[0])
        if t1 > t0:
            rate_dev = (n - 1) / ((t1 - t0) / 1000.0)
    except ValueError:
        pass
    # sensor sanity: mean |a| per sensor (should be ~1 g at rest) and are the values moving at all
    mags, vals = [[], []], []
    for _, ln in rows:
        f = ln.split(",")
        try:
            v = [float(x) for x in f[1:7]]
        except ValueError:
            continue
        if len(v) < 6:      # '#' status lines
            continue
        vals.append(v)
        mags[0].append((v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5)
        mags[1].append((v[3] ** 2 + v[4] ** 2 + v[5] ** 2) ** 0.5)
    say("%d samples in %.1f s | arrival rate %.1f Hz%s | columns per line: %d"
        % (n, wall, rate_wall, "" if rate_dev is None else " | device-clock rate %.1f Hz" % rate_dev,
           len(rows[-1][1].split(","))))
    if vals:
        m0, m1 = sum(mags[0]) / len(mags[0]), sum(mags[1]) / len(mags[1])
        spread = [max(c) - min(c) for c in zip(*vals)]
        say("mean |a|: thigh %.2f g, shank %.2f g (expect ~1.0 at rest)  | axis spread (g): %s"
            % (m0, m1, " ".join("%.2f" % s for s in spread)))
        if m0 < 0.5 or m1 < 0.5:
            say("WARNING: a sensor reads near 0 g -> not connected / I2C problem (Arduino sends '# ...' status).")
    ok = (rate_dev or rate_wall) >= 80
    say("RESULT: " + ("OK - data is flowing at ~100 Hz. Launch the viewer without --pretty-check."
                      if ok else "LOW RATE (expected ~100 Hz). Another program may be reading the port too, "
                                 "or the USB link is flaky."))
    return 0 if ok else 1


# ---------------------------------------------------------------- process cleanup ----

def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except OSError:
        return False


def stop_group(p, polite=True):
    """Stop a child and its process group. SIGINT first so the bridge can send 'X' to the Arduino."""
    if IS_WIN:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(3)
            except subprocess.TimeoutExpired:
                p.kill()
        return
    pgid = p.pid  # started with start_new_session=True -> pgid == pid
    steps = [(signal.SIGINT if polite else signal.SIGTERM, 2.5), (signal.SIGTERM, 1.5), (signal.SIGKILL, 1.0)]
    for sig, wait_s in steps:
        if not group_alive(pgid):
            break
        try:
            os.killpg(pgid, sig)
        except OSError:
            break
        end = time.monotonic() + wait_s
        while time.monotonic() < end and (p.poll() is None or group_alive(pgid)):
            time.sleep(0.05)
    try:
        p.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def reap_after_viewer(viewer, port, uses_bridge):
    """Viewer is gone. The bridge normally exits when the pipe closes; verify, then clean up."""
    if IS_WIN:
        return
    pgid = viewer.pid
    end = time.monotonic() + 4.0  # grace: bridge notices the closed pipe on its next write (~10 ms)
    while time.monotonic() < end and group_alive(pgid):
        time.sleep(0.1)
    if group_alive(pgid):
        say("helper still running after viewer exit -> stopping it")
        stop_group(viewer, polite=True)
    if uses_bridge and port:
        time.sleep(0.2)
        left = port_holders(port)
        stray = [(p, c) for p, c in left if "serial_bridge.py" in c]
        for pid, cmd in stray:
            say(f"stray bridge PID {pid} still holds {port}; killing it")
            for sig in (signal.SIGINT, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                except OSError:
                    break
                time.sleep(1.0)
        left = port_holders(port)
        if left:
            say(f"note: {port} is still held by: " + "; ".join(f"PID {p} ({c})" for p, c in left))
        else:
            say(f"port {port} is free (no leftover bridge).")


# --------------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser(
        description="Launch the Brace Tracker 3D viewer fed by the Arduino serial bridge.",
        epilog='examples:  run_live.py | run_live.py --pretty-check | run_live.py --demo --scenario fast | '
               'run_live.py --viewer-args "--thigh-len 46 --shin-len 43" --bridge-args "--zero-secs 2"')
    ap.add_argument("--demo", action="store_true", help="use mock_sensors.py (simulated brace + camera) instead of the Arduino")
    ap.add_argument("--scenario", default="default", choices=("default", "fast", "slow", "badcam", "calib"),
                    help="--demo motion: calib = a Hang and Straighten session (press C at the start)")
    ap.add_argument("--port", help="serial port (default: auto-detect, e.g. /dev/cu.usbmodem11401 or COM5)")
    ap.add_argument("--camera", type=int, nargs="?", const=0, default=0, metavar="N",
                    help="camera index for the live ArUco view in the control panel (default 0; "
                         "also switchable in the panel). The panel owns the camera, not the bridge.")
    ap.add_argument("--no-camera", action="store_true", help="no camera view")
    ap.add_argument("--marker-cm", type=float, default=5.0, help="printed ArUco marker side (cm)")
    ap.add_argument("--no-position", action="store_true", help="bridge prints plain 7-field lines")
    ap.add_argument("--pretty-check", type=float, nargs="?", const=5.0, default=None, metavar="SECS",
                    help="run only the data source for SECS (default 5), print sample rate + sample lines, "
                         "then exit without opening the viewer")
    ap.add_argument("--dry-run", action="store_true", help="print the final command and exit (nothing is launched)")
    ap.add_argument("--viewer-args", default="", help='extra viewer options, quoted: "--thigh-len 46 --shin-len 43"')
    ap.add_argument("--bridge-args", default="", help='extra serial_bridge.py options, quoted: "--sag-axis 1"')
    ap.add_argument("--log", metavar="FILE.csv", help="have the bridge also append every sample to this CSV")
    ap.add_argument("--list", action="store_true", help="list serial ports / the detected Arduino port and exit")
    ap.add_argument("--viewer", help="path to the viewer binary (default: <repo>/build/viewer)")
    ap.add_argument("--kill-stale", action="store_true", help="kill a leftover serial_bridge.py that holds the port")
    ap.add_argument("--leg", choices=("left", "right"), help="which leg the brace is on (default: the last one used; switch live in the panel)")
    ap.add_argument("--no-panel", action="store_true", help="no leg window; use --leg (default left) and the saved settings")
    ap.add_argument("--no-install", action="store_true", help="never pip install anything automatically")
    args = ap.parse_args()
    install = not args.no_install

    if not os.path.isfile(BRIDGE) or not os.path.isfile(MOCK):
        die(f"expected {BRIDGE} and {MOCK}")

    use_hw = not args.demo
    if use_hw and not args.dry_run:
        ensure_pyserial(install)
        if not args.no_camera:
            ensure_opencv(install)

    port = None
    if use_hw or args.list:
        port = find_port(args.port)

    if args.list:
        say("python: " + sys.executable)
        cands = serial_candidates()
        say("detected Arduino-like ports: " + (", ".join(cands) if cands else "none"))
        for c in cands:
            h = port_holders(c)
            say(f"  {c}: " + ("free" if not h else "BUSY - " + "; ".join(f"PID {p} ({cmd})" for p, cmd in h)))
        try:
            subprocess.call([sys.executable, BRIDGE, "--list"], cwd=REPO)
        except OSError:
            pass
        return 0

    if use_hw and not port:
        msg = ("no Arduino serial port found. Plug it in (data cable), or pass --port. "
               "On mac look at `ls /dev/cu.*`.")
        if args.dry_run:
            say("WARNING: " + msg)
        else:
            die(msg)

    src = source_argv(args, port)
    viewer = viewer_path(args.viewer)
    vargs = shlex.split(args.viewer_args, posix=not IS_WIN) if args.viewer_args else []
    vcmd = [viewer, "--state", LEG_STATE, "--pi-cmd", quote(wrap_with_leg_filter(src, args.demo)), *vargs]

    if args.dry_run:
        say("python:  " + sys.executable)
        say("cwd:     " + REPO)
        if not os.path.isfile(viewer):
            say(f"viewer binary missing ({viewer}); would build with: bash {shlex.quote(BUILD_SH)}")
        if args.pretty_check is not None:
            say("pretty-check source command (%.0f s):" % args.pretty_check)
            print(quote(src))
        else:
            say("command:")
            print(quote(vcmd) if IS_WIN else " ".join(shlex.quote(c) for c in vcmd))
        if use_hw and port:
            h = port_holders(port)
            say(f"port {port}: " + ("free" if not h else "BUSY - " + "; ".join(f"PID {p} ({c})" for p, c in h)))
        return 0

    if use_hw and not check_port_free(port, args.kill_stale):
        return 3

    if args.pretty_check is not None:
        return pretty_check(src, args.pretty_check, is_bridge=use_hw)

    ensure_viewer(viewer, dry_run=False)
    panel = None
    leg = args.leg or ("left" if args.demo else None)     # the mock is a left leg
    if args.no_panel:
        write_leg_state(leg or "left")
    else:   # in --demo the mock writes the camera file, so the panel leaves the webcam alone
        panel, why, leg = start_leg_panel(leg, args.camera, not (args.no_camera or args.demo), args.marker_cm)
        if panel is None:
            say("control panel unavailable (%s); using the %s leg" % (why, leg))
        else:
            say("control panel open (leg: %s). Switch LEFT/RIGHT there any time." % leg)
    say("source: " + quote(src))
    say("starting viewer (close the window or Ctrl+C here to quit)")
    say("calibrate each session: press C (or Calibrate in the panel), sit, let the shin hang, follow the prompts")
    say("keys: C calibrate | Z zero (only if truly straight) | [ ] camera weight | ; ' responsiveness | "
        "V source | LEFT/RIGHT orbit | ESC quit")
    p = subprocess.Popen(vcmd, cwd=REPO, start_new_session=not IS_WIN)
    code = 0

    def _term(signum, frame):  # also treat SIGTERM/SIGHUP (and ignored-SIGINT background jobs) as Ctrl+C
        raise KeyboardInterrupt
    for sg in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, sg):
            signal.signal(getattr(signal, sg), _term)
    try:
        code = p.wait()
    except KeyboardInterrupt:
        say("Ctrl+C: closing viewer")
        stop_group(p, polite=True)
        code = 130
    if code not in (0, 130):
        say(f"viewer exited with code {code}")
    if panel is not None and panel.poll() is None:
        panel.terminate()
    reap_after_viewer(p, port, uses_bridge=use_hw)
    return code


if __name__ == "__main__":
    sys.exit(main())
