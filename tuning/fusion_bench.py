#!/usr/bin/env python3
"""fusion_bench.py - tuning bench for live/knee_estimator.KneeFilter on the mock (dev tool, not used live).

Runs the filter over many mock runs (scenarios x seeds) in three modes - fused at the default camera
weight, accelerometers only (w=0) and camera only (w=1) - and scores each the way eval_fusion does
(viewer playout, rms / p95 / lag / rest jitter / J). Constants of knee_estimator can be overridden:

    python3 tuning/fusion_bench.py                       # train seeds, all scenarios
    python3 tuning/fusion_bench.py --test                # held-out seeds
    python3 tuning/fusion_bench.py --set GATE=5 --set RELOCK_S=0.3
"""
import argparse
import math
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "live"))
import eval_fusion as EF  # noqa: E402
if os.environ.get("KE_FILE"):     # benchmark another version of the estimator (e.g. the original)
    import importlib.util
    _spec = importlib.util.spec_from_file_location("knee_estimator_alt", os.environ["KE_FILE"])
    KE = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(KE)
else:
    import knee_estimator as KE  # noqa: E402
import mock_sensors as MS  # noqa: E402

# name -> (mock scenario, overrides)
SCEN = {
    "default": ("default", {}),
    "fast": ("fast", {}),
    "slow": ("slow", {}),
    "badcam": ("badcam", {}),
    "lowfps": ("default", dict(cam_fps=15.0, cam_lat=(0.08, 0.16))),
    "latestamp": ("default", dict(cam_stamp=(0.08, 0.12))),
    "fastbad": ("fast", dict(cam_noise=2.5, cam_drop=0.15, cam_lost_rate=1 / 8.0, cam_outlier=0.03)),
    "nocam": ("default", dict(cam_drop=1.0)),
    "slip": ("default", dict(slip_deg=6.0, slip_t=15.0)),   # the shank strap slips 6 deg mid-session          # webcam missing: what most demos without one see
}
TRAIN, TEST = (1, 2, 3), (11, 12, 13)
DURATION = 40.0
_CACHE = {}


def prepare(name, seed, duration=DURATION):
    """eval_fusion.prepare, with scenario overrides."""
    key = (name, seed, duration)
    if key in _CACHE:
        return _CACHE[key]
    scen, ov = SCEN[name]
    orig = MS.generate

    def gen(scenario, seed, duration, **kw):
        kw2 = dict(ov)
        kw2.update(kw)
        return orig(scenario, seed, duration, **kw2)
    EF.mock_sensors.generate = gen
    try:
        ctx = EF.prepare(scen, seed, duration)
    finally:
        EF.mock_sensors.generate = orig
    ctx["name"] = name
    ctx["truth_fn"] = MS.generate(scen, seed, 1.0, **ov)["truth"]     # same motion, extends on demand
    _CACHE[key] = ctx
    return ctx


def run_filter(evs, w, resp, factory):
    kf = factory(w, resp)
    ks, zeroed = [], False
    for t, is_cam, d in evs:
        if is_cam:
            if d[2] is not None:
                if len(d) > 3 and "conf" in kf.camera.__code__.co_varnames:
                    kf.camera(d[0], d[2], d[3])
                else:
                    kf.camera(d[0], d[2])
            continue
        if not zeroed and t >= EF.AUTO_ZERO_S:
            kf.zero()
            zeroed = True
        theta, omega = kf.accel(t, d)
        ks.append((t, theta, omega))
    return ks


def evaluate(ctx, w, resp, factory=None):
    factory = factory or (lambda w, r: KE.KneeFilter(w, r))
    fr = ctx["frames"]
    return EF.score(ctx, EF.render(run_filter(ctx["events"], w, resp, factory), fr),
                    EF.render(run_filter(ctx["ideal_events"], w, resp, factory), fr))


def _job(args):
    name, seed, w, resp, overrides = args
    for k, v in overrides.items():
        setattr(KE, k, v)
    return (name, seed, w), evaluate(prepare(name, seed), w, resp)


def bench(seeds=TRAIN, ws=None, resp=None, overrides=None, scen=None, procs=None):
    """{(scenario, w): mean metrics over seeds}."""
    ws = ws or (KE.W_DEFAULT, 0.0, 1.0)
    resp = KE.RESP_DEFAULT if resp is None else resp
    jobs = [(n, s, w, resp, overrides or {}) for n in (scen or SCEN) for s in seeds for w in ws]
    with mp.Pool(procs or os.cpu_count()) as pool:
        res = pool.map(_job, jobs)
    out = {}
    for (n, s, w), m in res:
        out.setdefault((n, w), []).append(m)
    return {k: EF.mean(v) for k, v in out.items()}


def table(res, ws):
    lines = ["%-10s %5s %6s %6s %6s %6s %7s" % ("scenario", "w", "rms", "p95", "lag", "jit", "J")]
    for w in ws:
        tot = []
        for n in SCEN:
            if (n, w) in res:
                m = res[n, w]
                tot.append(m)
                lines.append("%-10s %5.2f %6.2f %6.2f %6.0f %6.2f %7.2f" % (n, w, m["rms"], m["p95"], m["lag"],
                                                                         m["jitter"], m["J"]))
        m = EF.mean(tot)
        lines.append("%-10s %5.2f %6.2f %6.2f %6.0f %6.2f %7.2f  <- mean" % ("ALL", w, m["rms"], m["p95"], m["lag"],
                                                                            m["jitter"], m["J"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="held-out seeds")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--w", type=float, action="append")
    ap.add_argument("--resp", type=float)
    ap.add_argument("--scen", action="append")
    a = ap.parse_args()
    ov = {k: float(v) for k, v in (s.split("=", 1) for s in a.set)}
    ws = a.w or [KE.W_DEFAULT, 0.0, 1.0]
    res = bench(TEST if a.test else TRAIN, ws, a.resp, ov, a.scen)
    print(table(res, ws))


if __name__ == "__main__":
    main()
