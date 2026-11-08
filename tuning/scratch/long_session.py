"""15 min with camera, then the camera is lost for 60 s: accel-only error after the loss, map/b size."""
import sys, os, math, multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "live"))
import knee_estimator as KE, mock_sensors as MS

def one(seed, T=900.0, L=60.0):
    run = MS.generate("default", seed=seed, duration=T + L)
    kf = KE.KneeFilter(0.75)
    evs = sorted([(t, 0, v, g) for t, v, g in run["accel"]] + [(ta, 1, (tc, k), None) for tc, ta, k in run["cam"] if ta < T],
                 key=lambda e: e[0])
    e1, e2, z = [], [], False
    for t, c, d, g in evs:
        if c:
            if d[1] is not None: kf.camera(*d)
            continue
        if not z and t >= 1.5: kf.zero(); z = True
        th, _ = kf.accel(t, d)
        if T - 60 < t < T: e1.append(th - g)
        elif t > T + 1: e2.append(th - g)
    r = lambda e: math.sqrt(sum(x * x for x in e) / len(e))
    return "seed %d: fused rms (last min) %.2f | camera lost: accel rms %.2f | cmap %s b %.1f" % (
        seed, r(e1), r(e2), [round(c, 1) for c in getattr(kf, "cmap", [])], kf.x[-1])

if __name__ == "__main__":
    with mp.Pool(4) as p:
        for line in p.map(one, (1, 2, 3, 4)):
            print(line)
