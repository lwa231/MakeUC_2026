"""Quick single-config diagnostics: state rms and display metrics for fast/default."""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import fusion_bench as FB
KE = FB.KE

def state_rms(name, seed, w, r_shank=None, **ov):
    for k, v in ov.items(): setattr(KE, k, v)
    ctx = FB.prepare(name, seed)
    kf = KE.KneeFilter(w, 0.5)
    if r_shank is not None: kf.r_shank = r_shank
    run_truth = None
    err = []; zeroed = False
    import mock_sensors as MS
    for t, c, d in ctx["events"]:
        if c:
            if d[2] is not None: kf.camera(d[0], d[2])
            continue
        if not zeroed and t >= 1.5: kf.zero(); zeroed = True
        kf.accel(t, d)
    return kf
