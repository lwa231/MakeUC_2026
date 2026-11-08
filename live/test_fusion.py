#!/usr/bin/env python3
"""test_fusion.py - offline check of leg_filter.Fusion (camera helps the accelerometers).

Simulates a seated leg extension (knee 90->0->90 deg, 4 s period), an accelerometer pair
with a mounting offset / wrong bend sign / 2.7 deg quantization, and a laggy noisy camera.
Prints the knee-angle error (as main.cpp would compute it) with and without camera assist.
    python3 live/test_fusion.py
"""
import math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leg_filter as lf

Q = 0.0469


def quant(x):
    return round(x / Q) * Q


def knee_of(v):   # what main.cpp computes (no low-pass here)
    t = lambda a, b: math.degrees(math.atan2(a[b], a[b + 2]))
    return lf.wrap(t(v, 3) - t(v, 0))


def run(bias=15.0, sign=1.0, cam_sign=1.0, cam_lag=0.1, cam_noise=1.5, dropout=None, secs=24, assist=True):
    random.seed(1)
    fu = lf.Fusion()
    errs_in, errs_out, hist = [], [], []
    fake_now = [0.0]
    real_time = time.time
    lf.time.time = lambda: fake_now[0]            # drive the freshness check with simulated time
    try:
        for i in range(int(secs * 100)):
            t = i / 100.0
            fake_now[0] = t
            truth = lambda tt: 45.0 * (1.0 + math.cos(2 * math.pi * tt / 4.0))
            k = truth(t)
            tt_ = 2.0 * math.sin(2 * math.pi * t / 7.0)
            ks = sign * k + bias                    # accelerometer-implied knee (wrong sign / offset)
            th = (quant(math.sin(math.radians(tt_)) + random.gauss(0, .01)), 0.0, quant(math.cos(math.radians(tt_))))
            sh_t = tt_ + ks
            sh = (quant(math.sin(math.radians(sh_t)) + random.gauss(0, .01)), 0.0, quant(math.cos(math.radians(sh_t))))
            v = [th[0], th[1], th[2], sh[0], sh[1], sh[2]]
            if assist and i % 3 == 0 and not (dropout and dropout[0] <= t < dropout[1]):   # ~33 Hz camera
                hist.append(truth(t - cam_lag) * cam_sign + random.gauss(0, cam_noise))
                fu.set_cam({"knee": hist[-1], "conf": 1.0, "wall": t})
            out = fu.process(v, t * 1000.0) if assist else v
            if t > 8.0:
                errs_in.append(abs(lf.wrap(knee_of(v) - k)))
                errs_out.append(abs(lf.wrap(knee_of(out) - k)))
    finally:
        lf.time.time = real_time
    rms = lambda e: math.sqrt(sum(x * x for x in e) / len(e))
    return rms(errs_in), rms(errs_out), fu


cases = [
    ("accel offset +15 deg", dict()),
    ("accel offset -25 deg", dict(bias=-25.0)),
    ("accel bend sign wrong (right-leg mount) + offset", dict(sign=-1.0, bias=20.0)),
    ("camera mirrored (negative flexion)", dict(cam_sign=-1.0)),
    ("camera drops out for 4 s (hold)", dict(dropout=(12.0, 16.0))),
    ("camera lag 0.1 s (typical), noise 1.5 deg", dict()),
    ("camera lag 0.2 s, noise 3 deg", dict(cam_lag=0.2, cam_noise=3.0)),
]
bad = 0
print("%-52s %10s %10s  %s" % ("case", "no assist", "assisted", "status"))
for name, kw in cases:
    a, b, fu = run(**kw)
    ok = b < 6.0 and b < a
    bad += not ok
    print("%-52s %8.1f d %8.1f d  %s  (sign %+d, cam sign %+d)" % (name, a, b, "OK" if ok else "FAIL", fu.s, fu.cs))
sys.exit(1 if bad else 0)
