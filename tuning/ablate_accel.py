"""Accel-only ablation: which part of the new filter costs accuracy without a camera. Dev tool."""
import os, sys, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_bench as FB
import importlib.util
KE = FB.KE
SC = ("default", "slow", "fast", "slip")

def orig():
    spec = importlib.util.spec_from_file_location("ke_orig", os.path.join(FB.ROOT, "tuning", "knee_estimator_v1_orig.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

_DEF = {k: v for k, v in vars(KE).items() if k.isupper()}


def job(a):
    var, sc, seed = a
    for k, v in _DEF.items():          # pool workers are reused: start every job from the defaults
        setattr(KE, k, v)
    ctx = FB.prepare(sc, seed)
    ov = {}
    if var == "orig":
        M = orig(); fac = lambda w, r: M.KneeFilter(w, r)
    else:
        def fac(w, r, var=var):
            kf = KE.KneeFilter(w, r)
            if "noswing" in var: kf.r_shank = 0.0
            return kf
        if "racc4" in var: KE.R_ACC0 = 4.0
        if "swing0" in var: KE.SWING_ACC = 0
        if "swing1" in var: KE.SWING_ACC = 1
        if "euroorig" in var: KE.EURO_FMIN_ACC = (0.1, 0.4); KE.EURO_BETA_ACC = (0.1, 0.4)
        if "atau" in var: KE.ALPHA_TAU = 0.3
        if "devorig" in var: KE.DEV_G = 0.05; KE.DEAD_G = 0.04
    return var, sc, FB.evaluate(ctx, 0.0, 0.5, fac)

if __name__ == "__main__":
    vars_ = sys.argv[1:] or ["orig", "cur", "noswing", "euroorig", "qlow", "qhigh", "atau", "devorig", "racc4"]
    jobs = [(v, s, seed) for v in vars_ for s in SC for seed in (1, 2, 3)]
    with mp.Pool(8) as p:
        res = p.map(job, jobs)
    for v in vars_:
        row = []
        for s in SC:
            ms = [m for vv, ss, m in res if vv == v and ss == s]
            m = FB.EF.mean(ms)
            row.append("%s %.2f/%.0f/%.2f J%.1f" % (s[:4], m["rms"], m["lag"], m["jitter"], m["J"]))
        allm = FB.EF.mean([m for vv, ss, m in res if vv == v])
        print("%-10s J %.2f | %s" % (v, allm["J"], " | ".join(row)), flush=True)
