"""Scratch: structure of the rendered camera's error: fit err = a + b*theta (+ c*theta^2) per run; residual sd."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import e2e_camera as E
import numpy as np
for sc in E.SCENS:
    for seed in E.SEEDS:
        tr = E.FB.prepare(sc, seed)["truth_fn"]
        for st in E.SETUPS:
            fr = E.load_frames(sc, seed, st)
            th = np.array([tr(f[4]) for f in fr if f[2] is not None]); er = np.array([f[2] - tr(f[4]) for f in fr if f[2] is not None])
            rate = np.array([abs(tr.motion(f[4])[1]) for f in fr if f[2] is not None])
            A = np.column_stack([np.ones_like(th), th / 90, (th / 90) ** 2]); c, *_ = np.linalg.lstsq(A, er, rcond=None)
            res = er - A @ c
            conf = np.array([f[3] for f in fr if f[2] is not None])
            print("%-7s %d %-6s n%4d fix%3.0f%% err mean %5.2f sd %4.2f | fit a %5.2f b %5.2f c %5.2f resid sd %4.2f (rest %4.2f mov %4.2f) conf %.2f" % (
                sc, seed, st[0], len(er), 100 * len(er) / len(fr), er.mean(), er.std(), *c, res.std(), res[rate < 20].std(), res[rate >= 20].std(), conf.mean()))
