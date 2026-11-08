#!/usr/bin/env python3
"""mock_sensors.py - simulated brace: both MMA7660 accelerometers and the webcam, driven by ONE true knee motion.

Offline (eval_fusion, tests):
    run = generate("default", seed=1, duration=60.0)
    run["accel"]  [(t, v6, truth_deg)]  100 Hz; t = host arrival time (s); v6 = thigh xyz, shank xyz in g
    run["cam"]    [(t_cap, t_arrival, knee_or_None)]  ~30 Hz
    run["truth"]  truth(t) -> true knee angle (deg) at any time
    run["true_k"] raw sensor knee (sign-corrected) minus true knee: what calibration should find
Live (stands in for serial_bridge.py + the camera):
    mock_sensors.py [--scenario S] [--seed N] [--cam-out PATH] [--no-camera] [--duration S]
    prints `t_ms,tx,ty,tz,sx,sy,sz,0,0,0,0,truth` at 100 Hz and writes camera frames to PATH.

Angles: 0 = straight, flexion positive. Mock frame per board: x along the limb, z out of the limb, so a board
on a horizontal limb reads z = +1 g, and the hanging shank (knee 90) reads x = +1 g (x points at the knee).
Standard library only, Python 3.9+.
"""
import argparse
import bisect
import heapq
import itertools
import json
import math
import os
import random
import sys
import time

CPG = 64.0 / 3.0        # MMA7660 counts per g (6-bit, +-1.5 g)
G = 9.80665             # m/s^2
PERIOD = 0.010          # Arduino sample period (device clock)
SKEW = 0.0015           # device clock runs 0.15 % slow
LINE_S = 0.0036 + 0.0044   # I2C reads + serial wire time of one ~50-byte line at 115200 baud
USB_S = 0.004096        # the USB-serial chip flushes on a ~4 ms grid
SLIP_T = 13.75          # calib: the strap slips half-way through lowering back to the hang

DEFAULTS = dict(
    init_hold=4.0,                      # straight hold at the start (the viewer auto-zeros on it)
    t_move=(1.0, 2.0),                  # duration of one flexion or extension (s)
    hold_top=(0.4, 1.2), hold_bot=(0.6, 1.5),   # holds at the extended / flexed end (s)
    flex_to=(75.0, 95.0), ext_to=(-2.0, 5.0),   # rep end points (deg)
    partial=0.15, partial_to=(30.0, 60.0),      # share of half extensions, and how far they get
    tremor=0.04,                        # tremor amplitude (deg) at 7.3 Hz
    sway=1.0,                           # thigh drift scale (1 = a few deg)
    noise=0.4,                          # accel noise, sd in counts (flickers +-1 count)
    offset=3.0, gain=0.08,              # max zero-g offset (counts) and gain error per axis (datasheet)
    mount=5.0,                          # max mounting tilt of each board on the leg (deg)
    leak=(1.0, 4.0),                    # out-of-plane board tilt (deg): part of gravity shows on y
    r=0.15,                             # shank board distance below the knee (m)
    mirror=False,                       # boards turned 180 deg about the limb (right leg): bend sign flips
    hw_seed=None,                       # seed for the board/mounting draw (default: seed)
    stall_rate=0.2, stall=(0.01, 0.04),  # host stalls (per s, s): lines then arrive in a burst
    cam_noise=1.5,                      # camera angle noise, sd (deg), plus 0.01 deg per deg/s of motion
    cam_lat=(0.06, 0.12),               # capture -> file arrival latency (s)
    cam_drop=0.05,                      # frames without a fix
    cam_lost_rate=1 / 20.0, cam_lost=(0.5, 2.0),  # marker-lost dropouts (per s, s)
    cam_outlier=0.01,                   # frames off by 10-30 deg
    script=None,                        # "calib": the Hang and Straighten session, then reps
    hang_err_deg=0.0,                   # calib: shin this far from vertical at the hang
    deficit_deg=5.0,                    # calib: best extension, deg short of straight
    slip_deg=0.0,                       # calib: shank strap slip while lowering to the re-hang
)

SCENARIOS = {
    "default": {},
    "fast": dict(t_move=(0.5, 0.8), hold_top=(0.1, 0.4), hold_bot=(0.2, 0.6)),
    "slow": dict(t_move=(2.5, 4.5), hold_top=(1.0, 2.0), hold_bot=(1.0, 2.0)),
    "badcam": dict(cam_noise=3.0, cam_lat=(0.10, 0.20), cam_drop=0.2, cam_lost_rate=1 / 6.0, cam_outlier=0.04),
    # seated, thigh supported; boards already bench-checked, so offset/gain are corrected and true_k is exact
    "calib": dict(script="calib", sway=0.25, offset=0.0, gain=0.0),
}

# calib script: (phase, seconds, target); "hang" and "deficit" are filled in from the parameters
CALIB_SCRIPT = (("hold_bot", 5.0, "hang"), ("ext", 2.5, "deficit"), ("hold_top", 5.0, "deficit"),
                ("low", 2.5, "hang"), ("hold_bot", 5.0, "hang"))
NEXT_PHASE = {"hold_top": "low", "low": "hold_bot", "hold_bot": "ext", "ext": "hold_top"}


def params(scenario="default", **overrides):
    """DEFAULTS + SCENARIOS[scenario] + overrides; ValueError on unknown names."""
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario %r (have: %s)" % (scenario, ", ".join(SCENARIOS)))
    bad = [k for k in overrides if k not in DEFAULTS]
    if bad:
        raise ValueError("unknown mock parameter(s): %s" % ", ".join(bad))
    p = dict(DEFAULTS)
    p.update(SCENARIOS[scenario])
    p.update(overrides)
    return p


def print_g(count):
    """The Arduino's printG() value: count * 0.046875 g, truncated to 4 decimals with C integer math."""
    v = abs(count) * 46875 // 100
    return math.copysign(v // 10000 + (v % 10000) / 10000.0, count)


def minjerk(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (10 - 15 * u + 6 * u * u)


class Truth:
    """True knee angle: min-jerk moves and holds (segments added on demand, so it runs forever) plus tremor."""

    def __init__(self, p, seed):
        self.p, self.rng = p, random.Random("motion:%d" % seed)
        self.segs = []          # (t0, from_deg, to_deg, duration, phase)
        if p["script"] == "calib":
            to = {"hang": 90.0 - p["hang_err_deg"], "deficit": p["deficit_deg"]}
            t, cur = 0.0, to["hang"]
            for phase, d, target in CALIB_SCRIPT:
                self.segs.append((t, cur, to[target], d, phase))
                t, cur = t + d, to[target]
        else:
            self.segs.append((0.0, 0.0, 0.0, p["init_hold"], "hold_top"))
        self.t0s = [s[0] for s in self.segs]

    def _extend(self, t):
        p, U = self.p, self.rng.uniform
        while self.segs[-1][0] + self.segs[-1][3] <= t:
            t0, _, cur, d, phase = self.segs[-1]
            nxt = NEXT_PHASE[phase]
            if nxt.startswith("hold"):
                to, d = cur, U(*p[nxt])
            else:
                if nxt == "low":
                    to = U(*p["flex_to"])
                else:
                    to = U(*p["partial_to"]) if self.rng.random() < p["partial"] else U(*p["ext_to"])
                d = U(*p["t_move"])
            self.segs.append((t0 + self.segs[-1][3], cur, to, d, nxt))
            self.t0s.append(self.segs[-1][0])

    def motion(self, t):
        """(angle deg, rate deg/s, acceleration deg/s^2) at time t, tremor included."""
        self._extend(t)
        t0, a, b, d, _ = self.segs[max(0, bisect.bisect_right(self.t0s, t) - 1)]
        u = min(1.0, max(0.0, (t - t0) / d))
        span = b - a
        ang = a + span * minjerk(u)
        rate = span / d * 30 * u * u * (1 - u) ** 2
        acc = span / (d * d) * 60 * u * (1 - u) * (1 - 2 * u)
        w, ph = 2 * math.pi * 7.3, 2 * math.pi * 7.3 * t + 1.0
        amp = self.p["tremor"]
        return ang + amp * math.sin(ph), rate + amp * w * math.cos(ph), acc - amp * w * w * math.sin(ph)

    def __call__(self, t):
        return self.motion(t)[0]


def thigh_tilt(p, t):
    """Thigh drift (deg): a few degrees of slow sway."""
    return p["sway"] * (2.0 * math.sin(2 * math.pi * t / 9.0) + 0.7 * math.sin(2 * math.pi * t / 3.1 + 0.5))


def hardware(p, seed):
    """Per-board errors (one draw per brace): zero-g offsets, gains, mounting and out-of-plane tilts."""
    rng = random.Random("hardware:%d" % seed)

    def clipped(m):     # gaussian, sd m/2, never beyond the datasheet limit m
        return max(-m, min(m, rng.gauss(0.0, m / 2)))
    return dict(offset=[clipped(p["offset"]) for _ in range(6)],
                gain=[1.0 + clipped(p["gain"]) for _ in range(6)],
                leak=[math.radians(rng.choice((-1, 1)) * rng.uniform(*p["leak"])) for _ in range(2)],
                mount_th=rng.uniform(-p["mount"], p["mount"]), mount_sh=rng.uniform(-p["mount"], p["mount"]))


def read_board(p, hw, rng, board, tilt_deg, fx=0.0, fz=0.0):
    """printG values (x, y, z) of board 0 (thigh) or 1 (shank) at tilt_deg, plus motion fx/fz (g)."""
    phi, lk = math.radians(tilt_deg), hw["leak"][board]
    x, z = math.sin(phi) + fx, math.cos(phi) + fz
    g3 = (x, z * math.sin(lk), z * math.cos(lk))
    if p["mirror"]:
        g3 = (x, -g3[1], -g3[2])
    out = []
    for j, g in enumerate(g3):
        a = 3 * board + j
        c = round(g * hw["gain"][a] * CPG + hw["offset"][a] + rng.gauss(0.0, p["noise"]))
        out.append(print_g(max(-32, min(31, c))))
    return out


def accel_stream(p, hw, truth, rng):
    """Yields (t_arrival, t_ms, v6, truth_deg) forever, in arrival order."""
    stall, last = (-1.0, -1.0), -1.0
    for k in itertools.count():
        t = k * PERIOD * (1 + SKEW)             # host time this device slot starts
        ts = t + 0.002                          # mid-read
        knee, rate, acc = truth.motion(ts)
        th = thigh_tilt(p, ts)
        slip = p["slip_deg"] * minjerk((ts - SLIP_T) / 0.2) if p["script"] == "calib" else 0.0
        w, alpha = math.radians(rate), math.radians(acc)
        # Shank lever arm: centripetal r*w^2 along x (toward the knee), tangential r*alpha against z.
        v6 = (read_board(p, hw, rng, 0, th + hw["mount_th"]) +
              read_board(p, hw, rng, 1, th + knee + hw["mount_sh"] + slip,
                         p["r"] * w * w / G, -p["r"] * alpha / G))
        arr = math.ceil((t + LINE_S) / USB_S) * USB_S + rng.expovariate(2500.0)
        while p["stall_rate"] > 0 and arr > stall[1]:
            s0 = stall[1] + rng.expovariate(p["stall_rate"])
            stall = (s0, s0 + rng.uniform(*p["stall"]))
        if stall[0] <= arr <= stall[1]:         # inside a stall: released when it ends
            arr = stall[1]
        arr = last = max(arr, last + 0.0001)
        yield arr, int((k * PERIOD + 0.0036) * 1000), v6, knee


def cam_stream(p, truth, rng):
    """Yields (t_arrival, t_cap, knee_or_None) forever, in arrival order."""
    t, prev, lost_until = 0.2, 0.0, -1.0
    while True:
        t += (2 if rng.random() < 0.1 else 1) / 30.0 + rng.gauss(0.0, 0.002)   # ~10% of frames come late
        arr = prev = max(t + rng.uniform(*p["cam_lat"]) + rng.gauss(0.0, 0.005), prev + 0.001)
        if rng.random() < p["cam_lost_rate"] / 30.0:
            lost_until = t + rng.uniform(*p["cam_lost"])
        knee = None
        if t >= lost_until and rng.random() >= p["cam_drop"]:
            ang, rate, _ = truth.motion(t)
            knee = ang + rng.gauss(0.0, p["cam_noise"] + 0.01 * abs(rate))
            if rng.random() < p["cam_outlier"]:
                knee += rng.choice((-1, 1)) * rng.uniform(10.0, 30.0)
        yield arr, t, knee


def _setup(scenario, seed, overrides):
    p = params(scenario, **overrides)
    hw = hardware(p, seed if p["hw_seed"] is None else p["hw_seed"])
    truth = Truth(p, seed)
    return (p, hw, truth, accel_stream(p, hw, truth, random.Random("accel:%d" % seed)),
            cam_stream(p, truth, random.Random("camera:%d" % seed)))


def generate(scenario="default", seed=1, duration=60.0, **overrides):
    """Both sensor streams for `duration` s of host time (see the module docstring for the result)."""
    p, hw, truth, acc, cam = _setup(scenario, seed, overrides)
    run = dict(accel=[], cam=[], truth=truth, params=p, hw=hw,
               true_k=hw["mount_sh"] - hw["mount_th"])      # before any calib strap slip
    for arr, _, v6, knee in acc:
        if arr > duration:
            break
        run["accel"].append((arr, tuple(v6), knee))
    for arr, t_cap, knee in cam:
        if arr > duration:
            break
        run["cam"].append((t_cap, arr, knee))
    return run


def write_json(path, d):
    tmp = path + ".tmp.mock"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, path)


def play(args):
    """Real time: lines to stdout at their arrival times, camera frames to --cam-out."""
    _, _, _, acc, cam = _setup(args.scenario, args.seed, {})
    cam_out = None if args.no_camera else args.cam_out
    events = heapq.merge(((e[0], 0, e) for e in acc), ((e[0], 1, e) for e in cam), key=lambda e: e[0])
    out, t0, seq = sys.stdout, time.monotonic(), 0
    out.write("# streaming (mock scenario=%s seed=%d)\n" % (args.scenario, args.seed))
    for arr, kind, e in events:
        if args.duration and arr > args.duration:
            break
        dt = t0 + arr - time.monotonic()
        if dt > 0:              # lines due together go out in one write, like a USB packet
            out.flush()
            time.sleep(dt)
        if kind == 0:
            _, t_ms, v6, knee = e
            out.write("%d,%s,0,0,0,0,%.2f\n" % (t_ms, ",".join("%.4f" % v for v in v6), knee))
        elif cam_out:
            seq += 1
            knee = e[2]
            write_json(cam_out, {"knee": None if knee is None else round(knee, 2),
                                 "conf": 0.0 if knee is None else 0.85, "wall": time.time(),
                                 "t_cap": t0 + e[1], "seq": seq})
    out.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mock brace: accelerometer lines on stdout, camera frames to a file.")
    ap.add_argument("--scenario", default="default", choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cam-out", metavar="PATH", help="camera JSON to write (e.g. build/leg_state.json.cam)")
    ap.add_argument("--no-camera", action="store_true", help="do not write camera frames")
    ap.add_argument("--duration", type=float, default=0.0, metavar="S", help="stop after S s (default: forever)")
    args = ap.parse_args(argv)
    try:
        play(args)
    except (BrokenPipeError, KeyboardInterrupt):
        # the reader went away: exit quietly, and keep Python's flush at exit off the closed pipe
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return 0


if __name__ == "__main__":
    sys.exit(main())
