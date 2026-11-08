import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "live"))
import knee_estimator as KE, mock_sensors as MS, math
for kv in sys.argv[1:]:
    k, v = kv.split("="); setattr(KE, k, float(v))
run = MS.generate("default", seed=1, duration=30.0)
kf = KE.KneeFilter(0.5)
evs = sorted([(t, 0, v, g) for t, v, g in run["accel"]] + [(ta, 1, (tc, k), None) for tc, ta, k in run["cam"]], key=lambda e: e[0])
err = []
for t, c, d, g in evs:
    if c:
        if d[1] is not None: kf.camera(*d)
    else:
        th, _ = kf.accel(t, d)
        if t > 4: err.append(th - g)
print(" ".join(sys.argv[1:]) or "current", "rms %.2f delay %.3f relock %d cmap %s b %.1f" % (math.sqrt(sum(e*e for e in err)/len(err)), kf.cam_delay, kf.n_relock, [round(c,1) for c in kf.cmap], kf.x[3]))
