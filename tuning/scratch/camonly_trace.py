import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import fusion_bench as FB
KE = FB.KE
for kv in sys.argv[1:]:
    k, v = kv.split("="); setattr(KE, k, float(v))
ctx = FB.prepare("fast", 1); tr = ctx["truth_fn"]
kf = KE.KneeFilter(1.0, 0.5); z = False; log = []; first = None
for t, c, d in ctx["events"]:
    if c:
        if d[2] is None: continue
        ok = kf.camera(d[0], d[2]); log.append("cam t=%.3f z=%.1f ok=%d x=%s P00=%.1f P11=%.0f relock=%d" % (t, d[2], ok, [round(v) for v in kf.x], kf.P[0][0], kf.P[1][1], kf.n_relock))
        continue
    if not z and t >= 1.5: kf.zero(); z = True
    kf.accel(t, d)
    log.append("acc t=%.3f x=%s truth=%.1f" % (t, [round(v) for v in kf.x], tr(t)))
    if first is None and abs(kf.x[0] - tr(t)) > 60: first = len(log)
print("\n".join(l for l in log[max(0, first - 60):first + 10] if l.startswith("cam") or "acc" in l[:3] and log.index(l) % 3 == 0))
