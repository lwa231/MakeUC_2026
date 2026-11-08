#!/usr/bin/env python3
"""eval_fusion.py - offline sweep of KneeFilter's camera weight w and responsiveness resp on the mock.

Each mock run (mock_sensors.generate) goes through KneeFilter in arrival order, quick-zeroed at 1.5 s
on the straight start hold like the demo's --auto-zero. It is then drawn the way the viewer draws it:
60 fps, a fixed 30 ms playout, linear interpolation between K samples, omega extrapolation if starved.
Baseline: the HEAD viewer, a 5 Hz one-pole on the raw vectors with the newest value drawn each frame,
zeroed on the same straight hold. Score J = rms + lag/40 + 10*rest_jitter (deg, ms, deg); lower wins.

Lag is timed on a twin run with ideal boards (no zero-g offset or gain error, same motion, camera and
timing): a few degrees of angle-dependent board error would otherwise shift the timing by tens of ms.

Usage: python3 live/eval_fusion.py [--seeds N] [--duration S]     (standard library only)
"""
import argparse
import bisect
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knee_estimator as KE  # noqa: E402
import mock_sensors  # noqa: E402

SCENARIOS = ("default", "fast", "slow", "badcam")
WS = [i / 10.0 for i in range(11)]
RESPS = (0.3, 0.5, 0.7)
FPS = 60.0
PLAYOUT_S = 0.030           # the viewer draws the K stream this far in the past
AUTO_ZERO_S = 1.5           # demo --auto-zero (the mock starts with a 4 s straight hold)
SKIP_S = 4.0                # start-up and the zeroing hold are not scored
MIN_MOVE_DEG = 15.0         # lag is timed on moves at least this big
HOLD_SETTLE_S = 0.3         # rest jitter skips the start of each hold (that is lag, not jitter)
BASE_HZ = 5.0               # HEAD viewer low-pass cutoff
TIE = 0.02                  # configs within 2 % of the best J are reported as equally good


def events(run):
    """Accel samples and camera frames of one run, in arrival order: (t_arrival, is_camera, data)."""
    ev = [(a[0], 0, a[1]) for a in run["accel"]] + [(c[1], 1, c) for c in run["cam"]]
    return sorted(ev, key=lambda e: (e[0], e[1]))


def prepare(scenario, seed, duration):
    """One mock run, its ideal-board twin, and the truth-side data the scoring needs."""
    run = mock_sensors.generate(scenario, seed, duration)
    ideal = mock_sensors.generate(scenario, seed, duration, offset=0.0, gain=0.0)
    truth, frames = run["truth"], [k / FPS for k in range(int(SKIP_S * FPS), int(duration * FPS))]
    holds, moves, segs = [], [], truth.segs
    for k, (t0, a, b, d, phase) in enumerate(segs):
        lo, hi = bisect.bisect_left(frames, t0 + HOLD_SETTLE_S), bisect.bisect_left(frames, t0 + d)
        if phase.startswith("hold"):
            if hi - lo >= 5:
                holds.append(range(lo, hi))
        elif abs(b - a) >= MIN_MOVE_DEG and t0 >= frames[0] and k + 1 < len(segs):
            # (frame at the move start, last frame of the hold after it, truth's mid-move time)
            end = bisect.bisect_left(frames, t0 + d + segs[k + 1][3]) - 1
            if end < len(frames) - 1:
                moves.append((bisect.bisect_left(frames, t0), end, t0 + d / 2.0))
    return dict(name=scenario, frames=frames, g=[truth(t) for t in frames], holds=holds, moves=moves,
                accel=run["accel"], events=events(run),
                ideal_accel=ideal["accel"], ideal_events=events(ideal))


def run_filter(evs, w, resp):
    """KneeFilter over one run's events -> K samples [(t, theta, omega)]."""
    kf = KE.KneeFilter(w, resp)
    ks, zeroed = [], False
    for t, is_cam, d in evs:
        if is_cam:
            kf.camera(d[0], d[2])
            continue
        if not zeroed and t >= AUTO_ZERO_S:
            kf.zero()
            zeroed = True
        theta, omega = kf.accel(t, d)
        ks.append((t, theta, omega))
    return ks


def render(ks, frames):
    """Viewer: at frame time T show the K stream at T - PLAYOUT_S, using only samples that have
    arrived by T (K stamps are arrival times): interpolate, else extrapolate the newest with omega."""
    ts = [k[0] for k in ks]
    out = []
    for T in frames:
        tau = T - PLAYOUT_S
        i = bisect.bisect_right(ts, tau)
        if i == 0:
            out.append(ks[0][1])
            continue
        t0, y0, w0 = ks[i - 1]
        if i < len(ks) and ts[i] <= T:
            t1, y1, _ = ks[i]
            out.append(y0 + (y1 - y0) * (tau - t0) / (t1 - t0))
        else:
            out.append(y0 + w0 * min(tau - t0, 0.3))
    return out


def baseline(accel, frames):
    """HEAD viewer: 5 Hz one-pole on the six raw vectors (100 Hz), newest value drawn each frame,
    zeroed on the mean over the second before AUTO_ZERO_S (a user pressing Z while straight)."""
    al = 0.01 / (1.0 / (2.0 * math.pi * BASE_HZ) + 0.01)
    y, ts, ys = None, [], []
    for t, v6, _ in accel:
        y = list(v6) if y is None else [a + al * (b - a) for a, b in zip(y, v6)]
        ts.append(t)
        ys.append(KE.raw_knee(y))
    hold = [v for t, v in zip(ts, ys) if AUTO_ZERO_S - 1.0 <= t <= AUTO_ZERO_S]
    zero = sum(hold) / len(hold)
    return [KE.wrap(ys[max(0, bisect.bisect_right(ts, T) - 1)] - zero) for T in frames]


def lag_ms(ctx, Y):
    """Median over moves of when the display crosses the middle of its own start and end values,
    minus when the truth crosses the middle of its move (min-jerk: half-way in time)."""
    lags, T = [], ctx["frames"]
    for i0, i1, t_mid in ctx["moves"]:
        mid, up = (Y[i0] + Y[i1]) / 2.0, Y[i1] > Y[i0]
        i = next((i for i in range(i0 + 1, i1 + 1) if (Y[i] >= mid) == up), None)
        if i is not None and abs(Y[i1] - Y[i0]) >= MIN_MOVE_DEG / 2:
            t = T[i - 1] + (T[i] - T[i - 1]) * (mid - Y[i - 1]) / (Y[i] - Y[i - 1])
            lags.append(1000.0 * (t - t_mid))
    return sorted(lags)[len(lags) // 2] if lags else 0.0


def score(ctx, Y, Y_ideal):
    """rms / p95 error (deg), lag (ms, on the ideal twin), rest jitter (deg: pooled SD of the error
    within each hold) and J."""
    err = [y - g for y, g in zip(Y, ctx["g"])]
    n = len(err)
    m = dict(rms=math.sqrt(sum(e * e for e in err) / n), p95=sorted(abs(e) for e in err)[int(0.95 * n)],
             lag=lag_ms(ctx, Y_ideal))
    dev = []
    for h in ctx["holds"]:
        e = [err[i] for i in h]
        mean = sum(e) / len(e)
        dev += [(x - mean) ** 2 for x in e]
    m["jitter"] = math.sqrt(sum(dev) / len(dev)) if dev else 0.0
    m["J"] = m["rms"] + m["lag"] / 40.0 + 10.0 * m["jitter"]
    return m


def evaluate(ctx, w, resp):
    """Metrics of KneeFilter(w, resp) on one prepared run (handy for tests)."""
    frames = ctx["frames"]
    return score(ctx, render(run_filter(ctx["events"], w, resp), frames),
                 render(run_filter(ctx["ideal_events"], w, resp), frames))


def evaluate_baseline(ctx):
    return score(ctx, baseline(ctx["accel"], ctx["frames"]), baseline(ctx["ideal_accel"], ctx["frames"]))


def mean(ms):
    return {k: sum(m[k] for m in ms) / len(ms) for k in ms[0]}


def row(label, m):
    return "%-22s %6.2f %6.2f %6.0f %7.2f %7.2f" % (label, m["rms"], m["p95"], m["lag"], m["jitter"], m["J"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=3, help="seeds per scenario (default 3)")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds per mock run (default 60)")
    args = ap.parse_args(argv)
    ctxs = [prepare(s, seed, args.duration) for s in SCENARIOS for seed in range(1, args.seeds + 1)]
    print("%d runs (%s x %d seeds, %.0f s each); J = rms + lag/40 + 10*rest_jitter, mean over runs"
          % (len(ctxs), "/".join(SCENARIOS), args.seeds, args.duration))
    print("\n   w  " + "".join(" resp=%.1f" % r for r in RESPS))
    res = {}
    for w in WS:
        for r in RESPS:
            res[w, r] = [evaluate(c, w, r) for c in ctxs]
        print("%5.1f " % w + "".join("%9.2f" % mean(res[w, r])["J"] for r in RESPS), flush=True)
    J = {key: mean(v)["J"] for key, v in res.items()}
    w_best, r_best = min(J, key=J.get)
    band = [w for w in WS if J[w, r_best] <= J[w_best, r_best] * (1 + TIE)]
    base = [evaluate_baseline(c) for c in ctxs]

    print("\n%-22s %6s %6s %6s %7s %7s" % ("(deg, ms)", "rms", "p95", "lag", "jitter", "J"))
    for s in SCENARIOS:
        pick = [i for i, c in enumerate(ctxs) if c["name"] == s]
        print(row(s + " HEAD baseline", mean([base[i] for i in pick])))
        print(row(s + " recommended", mean([res[w_best, r_best][i] for i in pick])))
    b, a = mean(base), mean(res[w_best, r_best])
    print(row("all HEAD baseline", b))
    print(row("all accel only w=0", mean(res[0.0, r_best])))
    print(row("all camera only w=1", mean(res[1.0, r_best])))
    print(row("all recommended", a))

    print("\nRecommended: w=%.1f resp=%.1f (J %.2f). Within %d%% of it at this resp: w %s."
          % (w_best, r_best, a["J"], TIE * 100, ", ".join("%.1f" % w for w in band)))
    print("The mock is not the real camera or boards: confirm w on the brace with the slider.")
    print("vs HEAD baseline: rms %.2f -> %.2f deg (%+.0f%%), p95 %.2f -> %.2f deg, lag %.0f -> %.0f ms, "
          "rest jitter %.2f -> %.2f deg (%+.0f%%), J %.2f -> %.2f"
          % (b["rms"], a["rms"], 100 * (a["rms"] / b["rms"] - 1), b["p95"], a["p95"], b["lag"], a["lag"],
             b["jitter"], a["jitter"], 100 * (a["jitter"] / b["jitter"] - 1), b["J"], a["J"]))


if __name__ == "__main__":
    main()
