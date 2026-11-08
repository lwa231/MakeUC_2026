"""Scratch: grid over camera-trust / offset-drift constants on the rendered-camera runs (fused w=0.75)."""
import itertools, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import e2e_camera as E
if __name__ == "__main__":
    grid = {"R_CAM0": [0.3, 1.0, 3.0, 11.0], "Q_BW": [0.0, 0.1, 0.5, 2.0], "Q_B": [0.03, 0.3]}
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        ov = dict(zip(keys, vals))
        m = E.bench((0.75,), ov)
        a = E.FB.EF.mean([m[st[0], 0.75] for st in E.SETUPS])
        print("%-45s rms %5.2f p95 %5.2f lag %3.0f jit %4.2f J %6.2f" % (ov, a["rms"], a["p95"], a["lag"], a["jitter"], a["J"]), flush=True)
