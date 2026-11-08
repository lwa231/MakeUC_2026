#!/usr/bin/env python3
"""validate.py - score a parameter set (json) on train and held-out seeds. Dev tool.
    python3 tuning/validate.py tuning/best_params.json [NAME=VALUE ...]"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_bench as FB  # noqa: E402
import optimize as O  # noqa: E402


def report(p, label=""):
    for seeds, lab in ((FB.TRAIN, "train"), (FB.TEST, "test")):
        J, jf, ja, res = O.objective(p, seeds, detail=True)
        f = FB.EF.mean([res[n, FB.KE.W_DEFAULT] for n in FB.SCEN])
        a = FB.EF.mean([res[n, 0.0] for n in FB.SCEN])
        print("%-12s %-5s J %6.2f | fused rms %5.2f p95 %5.2f lag %3.0f jit %4.2f | accel rms %5.2f p95 %5.2f "
              "lag %3.0f jit %4.2f" % (label, lab, J, f["rms"], f["p95"], f["lag"], f["jitter"], a["rms"], a["p95"],
                                       a["lag"], a["jitter"]), flush=True)


if __name__ == "__main__":
    p = O.current()
    sets = []
    for arg in sys.argv[1:]:
        if arg.endswith(".json"):
            p.update(json.load(open(arg)))
        else:
            k, v = arg.split("=")
            p[k] = float(v)
            sets.append(arg)
    report(p, " ".join(sets) or "params")
