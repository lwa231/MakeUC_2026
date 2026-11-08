#!/usr/bin/env python3
"""optimize.py - coordinate search over knee_estimator constants on the mock (dev tool).

Objective = mean J over all bench scenarios, fused at W_DEFAULT, + ACCEL_WEIGHT * mean J accel-only (w=0).
Every evaluation passes the full parameter set to the workers. Prints the best set as it goes.
    python3 tuning/optimize.py [--rounds 3] [--seeds 1,2,3]
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_bench as FB  # noqa: E402

KE = FB.KE
ACCEL_WEIGHT = 0.5
# name -> (log scale?, lower, upper)
SPACE = {
    "Q_J0": (True, 1e4, 1e8), "ALPHA_TAU": (True, 0.03, 2.0), "R_ACC0": (True, 0.5, 30.0),
    "R_CAM0": (True, 0.1, 20.0), "DEV_G": (True, 0.01, 0.5), "DEAD_G": (False, 0.0, 0.2),
    "Q_B": (True, 0.001, 20.0), "GATE": (False, 2.5, 8.0),
    "DYN_RES": (False, 0.0, 1.0), "BLIP_SIGMA": (False, 1.5, 8.0), "CAM_DELAY_GAIN": (True, 0.002, 0.3),
    "EURO_FMIN": (True, 0.1, 5.0), "EURO_BETA": (True, 0.005, 5.0),
    "Q_ACC_A": (True, 30.0, 1e6), "R_ACC0_ACC": (True, 0.3, 50.0), "DEV_G_ACC": (True, 0.01, 0.5),
    "DEAD_G_ACC": (False, 0.0, 0.2), "BLIP_SIGMA_ACC": (False, 1.5, 8.0),
    "EURO_FMIN_ACC": (True, 0.1, 5.0), "EURO_BETA_ACC": (True, 0.005, 5.0),
}
TUPLE = {"EURO_FMIN", "EURO_BETA", "EURO_FMIN_ACC", "EURO_BETA_ACC"}
MODE = "both"     # "accel": objective = accel-only J (w = 0) only   # a + b * resp: the search scales both by one factor


def current():
    return {k: (getattr(KE, k)[0] + getattr(KE, k)[1] * KE.RESP_DEFAULT) if k in TUPLE else getattr(KE, k)
            for k in SPACE}


def to_overrides(p):
    """Tuple constants keep their resp slope ratio: value at resp 0.5 = p[k]."""
    ov = {}
    for k, v in p.items():
        if k in TUPLE:
            a, b = getattr(KE, k)
            f = v / (a + b * KE.RESP_DEFAULT)
            ov[k] = (a * f, b * f)
        else:
            ov[k] = v
    return ov


def objective(p, seeds, detail=False):
    if MODE == "accel":
        res = FB.bench(seeds, (0.0,), None, to_overrides(p))
        ja = sum(res[n, 0.0]["J"] for n in FB.SCEN) / len(FB.SCEN)
        return (ja, 0.0, ja, res) if detail else ja
    res = FB.bench(seeds, (KE.W_DEFAULT, 0.0), None, to_overrides(p))
    jf = sum(res[n, KE.W_DEFAULT]["J"] for n in FB.SCEN) / len(FB.SCEN)
    ja = sum(res[n, 0.0]["J"] for n in FB.SCEN) / len(FB.SCEN)
    return (jf + ACCEL_WEIGHT * ja, jf, ja, res) if detail else jf + ACCEL_WEIGHT * ja


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--only", help="comma list of parameters to search")
    ap.add_argument("--start", help="json file with a starting parameter set")
    ap.add_argument("--mode", default="both", choices=("both", "accel"))
    ap.add_argument("--out", default="best_params.json")
    a = ap.parse_args()
    global MODE
    MODE = a.mode
    seeds = tuple(int(s) for s in a.seeds.split(","))
    p = current()
    if a.start:
        p.update(json.load(open(a.start)))
    names = a.only.split(",") if a.only else list(SPACE)
    best = objective(p, seeds)
    print("start %.3f %s" % (best, json.dumps(p)), flush=True)
    step = {k: (2.0 if SPACE[k][0] else 0.25 * (SPACE[k][2] - SPACE[k][1])) for k in names}
    for rnd in range(a.rounds):
        improved = False
        for k in names:
            log, lo, hi = SPACE[k]
            for direction in (1, -1):
                while True:
                    v = p[k] * step[k] ** direction if log else p[k] + direction * step[k]
                    v = min(hi, max(lo, v))
                    if v == p[k]:
                        break
                    q = dict(p, **{k: v})
                    j = objective(q, seeds)
                    if j < best - 1e-3:
                        best, p, improved = j, q, True
                        print("  %-15s -> %-10.4g J %.3f" % (k, v, best), flush=True)
                    else:
                        break
        print("round %d: J %.3f %s" % (rnd + 1, best, json.dumps(p)), flush=True)
        for k in names:
            step[k] = math.sqrt(step[k]) if SPACE[k][0] else step[k] / 2.0
        if not improved and rnd > 0:
            break
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
    json.dump(p, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
