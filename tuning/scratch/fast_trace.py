import json, os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import e2e_camera as E
KE = E.FB.KE
for kv in sys.argv[1:]:
    k, v = kv.split("="); setattr(KE, k, float(v))
frames = json.load(open(E.cache_path("fast", 1, E.SETUPS[0])))
ctx = E.with_camera(E.FB.prepare("fast", 1), frames); tr = ctx["truth_fn"]
kf = KE.KneeFilter(0.75, 0.5); z = False; acc = rej = 0; log = []
for t, c, d in ctx["events"]:
    if c:
        if d[2] is None: continue
        x0 = kf.x[0]; ok = kf.camera(d[0], d[2], d[3]); acc += ok; rej += not ok
        if 12 < t < 14: log.append("cam t=%.3f z=%.1f truth_exp=%.1f ok=%d x=%s P00=%.2f relock=%d delay=%.3f" % (t, d[2], tr(d[0] - kf.cam_delay), ok, [round(v) for v in kf.x], kf.P[0][0], kf.n_relock, kf.cam_delay))
        continue
    if not z and t >= 1.5: kf.zero(); z = True
    kf.accel(t, d)
    if 12 < t < 14 and int(t * 100) % 5 == 0: log.append("acc t=%.3f x=%s truth=%.1f" % (t, [round(v) for v in kf.x], tr(t)))
print("acc", acc, "rej", rej, "relock", kf.n_relock, "delay", kf.cam_delay, "cmap", [round(c, 2) for c in kf.cmap])
print("\n".join(log[:90]))
