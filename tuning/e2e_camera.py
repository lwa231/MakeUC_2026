#!/usr/bin/env python3
"""e2e_camera.py - end-to-end sim: mock brace motion -> rendered camera frames -> camera_origin.CameraTracker
-> (stamp, latency, conf gate as in leg_filter) -> KneeFilter fused with the mock accelerometers.

The camera part (rendering + ArUco) is slow, so its output (t_stamp, t_arrival, knee, conf per frame) is
cached in tuning/cache/, keyed by setup and a hash of camera_origin.py. Dev tool.

    python3 tuning/e2e_camera.py                 # build/reuse the cache, score fused / accel / camera-only
"""
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "live"))
sys.path.insert(0, ROOT)
import fusion_bench as FB  # noqa: E402

CACHE = os.path.join(HERE, "cache")
# camera setups: (name, resolution, distance m, view deg (|yaw|; pitch = view/2), light)
SETUPS = (("good", (1280, 720), 1.0, 5.0, "good"), ("angled", (1280, 720), 1.0, 18.0, "good"),
          ("far640", (640, 480), 1.3, 10.0, "good"), ("dim", (1280, 720), 1.0, 10.0, "dim"))
SCENS = ("default", "fast", "slow")
SEEDS = (1, 2)
FPS, LAT, STAMP, CAM_MIN_CONF = 30.0, (0.06, 0.12), (0.02, 0.07), 0.5


def cam_hash():
    if os.environ.get("E2E_HASH"):          # reuse frames rendered with an earlier camera_origin.py
        return os.environ["E2E_HASH"]
    return hashlib.sha1(open(os.path.join(ROOT, "camera_origin.py"), "rb").read()).hexdigest()[:10]


def gen_camera(job):
    """Render and track one run's camera frames. Returns [(t_stamp, t_arrival, knee|None, conf)]."""
    scen, seed, setup = job
    import numpy as np
    import random
    import sim_camera as SC
    from camera_origin import CameraTracker
    name, res, dist, view, light = setup
    ctx = FB.prepare(scen, seed)
    truth = ctx["truth_fn"]
    rng = np.random.default_rng([seed, hash(name) % 1000, 7])
    prng = random.Random("e2e:%s:%d:%s" % (scen, seed, name))
    cam = SC.Cam(w=res[0], h=res[1], hfov=70.0, dist=dist, yaw=view * prng.choice((-1, 1)),
                 pitch=0.5 * view * prng.choice((-1, 1)), k1=-0.05)
    pose0 = SC.sample_pose(SC.Scenario(view=view, speeds=(0,)), 0, rng)
    degr = SC.sample_degr(light, rng)
    tr = CameraTracker(marker_cm=SC.MARKER_CM)
    stamp = prng.uniform(*STAMP)
    out, t, prev = [], 0.2, 0.0
    while t < FB.DURATION:
        ang, rate, _ = truth.motion(t)
        pose = SC.replace(pose0, theta=ang, omega=rate)
        img, _ = SC.render(pose, cam, degr, rng)
        fix = tr.process_frame(img, t + stamp)
        arr = prev = max(t + prng.uniform(*LAT), prev + 0.001)
        knee = fix.camera_knee_deg if fix.confidence >= CAM_MIN_CONF else None
        out.append((min(arr, fix.t), arr, knee, float(fix.confidence), t))
        t += (2 if prng.random() < 0.1 else 1) / FPS + prng.gauss(0.0, 0.002)
    return out


def cache_path(scen, seed, setup):
    return os.path.join(CACHE, "%s_%d_%s_%s.json" % (scen, seed, setup[0], cam_hash()))


def build_cache(procs=None):
    os.makedirs(CACHE, exist_ok=True)
    jobs = [(sc, s, st) for sc in SCENS for s in SEEDS for st in SETUPS if not os.path.exists(cache_path(sc, s, st))]
    if jobs:
        print("rendering %d camera runs ..." % len(jobs), flush=True)
        with mp.Pool(procs or os.cpu_count()) as pool:
            for job, res in zip(jobs, pool.map(gen_camera, jobs)):
                json.dump(res, open(cache_path(*job), "w"))


def with_camera(ctx, frames):
    """ctx with its mock camera frames replaced by the rendered ones."""
    ctx = dict(ctx)
    for key, acc in (("events", ctx["accel"]), ("ideal_events", ctx["ideal_accel"])):
        ev = [(a[0], 0, a[1]) for a in acc] + [(f[1], 1, tuple(f[:4])) for f in frames]
        ctx[key] = sorted(ev, key=lambda e: (e[0], e[1]))
    return ctx


def _job(args):
    scen, seed, setup, w, overrides = args
    for k, v in overrides.items():
        setattr(FB.KE, k, v)
    frames = json.load(open(cache_path(scen, seed, setup)))
    ctx = with_camera(FB.prepare(scen, seed), frames)
    m = FB.evaluate(ctx, w, FB.KE.RESP_DEFAULT)
    got = [f for f in frames if f[2] is not None]
    m["cam_rate"] = len(got) / len(frames)
    errs = [f[2] - ctx["truth_fn"](f[4]) for f in got]
    m["cam_err"] = math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else 0.0
    return (setup[0], w), m


def bench(ws=None, overrides=None, procs=None):
    ws = ws or (FB.KE.W_DEFAULT, 0.0, 1.0)
    jobs = [(sc, s, st, w, overrides or {}) for sc in SCENS for s in SEEDS for st in SETUPS for w in ws]
    with mp.Pool(procs or os.cpu_count()) as pool:
        res = pool.map(_job, jobs)
    out = {}
    for k, m in res:
        out.setdefault(k, []).append(m)
    return {k: FB.EF.mean(v) for k, v in out.items()}


def table(res, ws):
    lines = ["%-8s %5s %6s %6s %6s %6s %7s %6s %7s" % ("setup", "w", "rms", "p95", "lag", "jit", "J", "fix%",
                                                         "camerr")]
    for w in ws:
        tot = []
        for st in SETUPS:
            m = res[st[0], w]
            tot.append(m)
            lines.append("%-8s %5.2f %6.2f %6.2f %6.0f %6.2f %7.2f %6.0f %7.2f" % (
                st[0], w, m["rms"], m["p95"], m["lag"], m["jitter"], m["J"], 100 * m["cam_rate"], m["cam_err"]))
        m = FB.EF.mean(tot)
        lines.append("%-8s %5.2f %6.2f %6.2f %6.0f %6.2f %7.2f  <- mean" % ("ALL", w, m["rms"], m["p95"], m["lag"],
                                                                          m["jitter"], m["J"]))
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--w", type=float, action="append")
    a = ap.parse_args()
    ov = {k: float(v) for k, v in (x.split("=", 1) for x in a.set)}
    build_cache()
    ws = a.w or [FB.KE.W_DEFAULT, 0.0, 1.0]
    print(table(bench(ws, ov), ws))


if __name__ == "__main__":
    main()
