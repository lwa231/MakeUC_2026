"""Scratch: per-run state errors (rest / moving) for the rendered-camera runs, fused vs accel vs camera."""
import json
import math
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import e2e_camera as E  # noqa: E402

KE = E.FB.KE
OV = dict(kv.split("=") for kv in sys.argv[1:])


def rms(v):
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else 0.0


def one(job):
    scen, seed, setup = job
    for k, v in OV.items():
        setattr(KE, k, float(v))
    frames = json.load(open(E.cache_path(scen, seed, setup)))
    ctx = E.with_camera(E.FB.prepare(scen, seed), frames)
    truth = ctx["truth_fn"]
    out = []
    for w in (0.75, 0.0):
        kf = KE.KneeFilter(w, 0.5)
        zeroed = False
        rest, mov = [], []
        for t, c, d in ctx["events"]:
            if c:
                if d[2] is not None:
                    kf.camera(d[0], d[2], d[3])
                continue
            if not zeroed and t >= 1.5:
                kf.zero()
                zeroed = True
            kf.accel(t, d)
            if t > 4:
                rate = truth.motion(t)[1]
                (mov if abs(rate) > 20 else rest).append(kf.x[0] - truth(t))
        out.append("w%.2f rest %5.2f mov %5.2f" % (w, rms(rest), rms(mov)))
    cam_rest = [f[2] - truth(f[4]) for f in frames if f[2] is not None and abs(truth.motion(f[4])[1]) <= 20]
    cam_mov = [f[2] - truth(f[4]) for f in frames if f[2] is not None and abs(truth.motion(f[4])[1]) > 20]
    return "%-7s %d %-6s | %s | camera rest %5.2f mov %5.2f" % (scen, seed, setup[0], " | ".join(out),
                                                                rms(cam_rest), rms(cam_mov))


if __name__ == "__main__":
    jobs = [(sc, s, st) for sc in E.SCENS for s in E.SEEDS for st in E.SETUPS]
    with mp.Pool(8) as p:
        for line in p.map(one, jobs):
            print(line)
