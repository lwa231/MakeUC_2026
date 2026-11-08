"""Scratch: test_smoother numbers (new vs original vs old viewer) and camera-delay learning bias."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "live"))
import eval_fusion as EF  # noqa: E402
import knee_estimator as KE  # noqa: E402
import mock_sensors as MS  # noqa: E402

spec = importlib.util.spec_from_file_location("o", os.path.join(HERE, "..", "knee_estimator_v1_orig.py"))
O = importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)


def f(m):
    return "rms %.2f lag %.0f jit %.2f" % (m["rms"], m["lag"], m["jitter"])


if "smooth" in sys.argv[1:] or len(sys.argv) == 1:
    for seed in (1, 2, 3):
        ctx = EF.prepare("default", seed, 30.0)
        new, base = EF.evaluate(ctx, KE.W_DEFAULT, KE.RESP_DEFAULT), EF.evaluate_baseline(ctx)
        EF.KE = O
        old = EF.evaluate(ctx, O.W_DEFAULT, O.RESP_DEFAULT)
        EF.KE = KE
        print(seed, "new", f(new), "| orig", f(old), "| viewer", f(base), flush=True)

if "delay" in sys.argv[1:] or len(sys.argv) == 1:
    for d in (0.03, 0.06, 0.10, 0.14):
        ds = []
        for seed in (4, 5):
            run = MS.generate("default", seed=seed, duration=60.0, cam_stamp=(d, d))
            kf = KE.KneeFilter(0.75)
            ev = sorted([(t, 0, v) for t, v, g in run["accel"]] + [(ta, 1, (tc, k)) for tc, ta, k in run["cam"]],
                        key=lambda e: e[0])
            hist = []
            for t, c, x in ev:
                if c:
                    if x[1] is not None:
                        kf.camera(*x)
                else:
                    kf.accel(t, x)
                    if t > 30:
                        hist.append(kf.cam_delay)
            ds.append(sum(hist) / len(hist))
        print("true %.2f learned %s" % (d, ["%.3f" % v for v in ds]), flush=True)
