#!/usr/bin/env python3
"""A few end-to-end checks that the demo works: run  python3 -m unittest live/test_live.py"""
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate  # noqa: E402
import eval_fusion  # noqa: E402
import knee_estimator as KE  # noqa: E402
import mock_sensors  # noqa: E402


def run_filter(run, **kw):
    """KneeFilter over a mock run in arrival order -> RMS error vs truth after 4 s."""
    kf = KE.KneeFilter(**kw)
    evs = sorted([(t, 0, v, g) for t, v, g in run["accel"]] +
                 [(ta, 1, (tc, k), None) for tc, ta, k in run["cam"]], key=lambda e: e[0])
    err = []
    for t, is_cam, d, g in evs:
        if is_cam:
            if d[1] is not None:
                kf.camera(*d)
        else:
            theta, _ = kf.accel(t, d)
            if t > 4.0:
                err.append(theta - g)
    return (sum(e * e for e in err) / len(err)) ** 0.5, kf


class Demo(unittest.TestCase):
    def test_smoother_than_old_viewer(self):
        ctxs = [eval_fusion.prepare("default", seed, 30.0) for seed in (1, 2, 3)]
        new = eval_fusion.mean([eval_fusion.evaluate(c, KE.W_DEFAULT, KE.RESP_DEFAULT) for c in ctxs])
        old = eval_fusion.mean([eval_fusion.evaluate_baseline(c) for c in ctxs])
        self.assertLess(new["jitter"], 0.6 * old["jitter"])      # still at rest
        self.assertLess(new["rms"], 1.05 * old["rms"])            # as accurate, despite the playout delay
        self.assertLess(new["lag"], old["lag"] + 15.0)            # and about as responsive

    def test_fusion_beats_either_sensor(self):
        run = mock_sensors.generate("default", seed=1, duration=30.0)
        fused, acc, cam = (run_filter(run, w=w)[0] for w in (0.5, 0.0, 1.0))
        self.assertLess(fused, acc)
        self.assertLess(fused, cam)                               # i.e. the camera delay is compensated

    def test_camera_outlier_rejected_and_w0_ignores_camera(self):
        run = mock_sensors.generate("default", seed=2, duration=10.0)
        _, kf = run_filter(run, w=0.5)
        t = run["accel"][-1][0]
        self.assertFalse(kf.camera(t - 0.05, kf.x[0] + 30.0))
        _, kf0 = run_filter(run, w=0.0)
        self.assertFalse(kf0.camera(t - 0.05, kf0.x[0]))
        _, kfa = run_filter(run, w=0.5)
        kfa.set(src="accel")
        self.assertEqual(kfa.w_eff, 0.0)

    def test_camera_only_relock_keeps_offset_sane(self):
        run = mock_sensors.generate("badcam", seed=1, duration=30.0)
        _, kf = run_filter(run, w=1.0)
        self.assertLess(abs(kf.x[2]), 15.0)                       # b stays usable for a switch to fused

    def test_calibration(self):
        for seed, kw in ((1, {}), (2, {"mirror": True, "mount": 100.0}), (3, {"deficit_deg": 15.0, "mount": 100.0})):
            run = mock_sensors.generate("calib", seed=seed, duration=30.0, **kw)
            s = self._calibrate(run)
            self.assertEqual(s.state, "done", s.prompt)
            self.assertLess(abs(KE.wrap(s.result["k_deg"] - run["true_k"])), 1.5)   # boards turned in pockets
            self.assertAlmostEqual(s.result["ext_deg"], kw.get("deficit_deg", 5.0), delta=2.0)
        s = self._calibrate(mock_sensors.generate("calib", seed=1, duration=30.0, slip_deg=8.0))
        self.assertEqual(s.state, "failed")
        self.assertIn("slip", s.prompt + s.message)

    def test_wrap_and_late_stamps(self):
        """Raw readings past +-180 do not jump the angle, and a camera stamp ahead of the accel stream
        does not swallow the next accel samples."""
        kf = KE.KneeFilter(0.5)
        kf.set_reference(60.0)
        def v6(knee):                       # thigh level, shank at `knee` + 60 (raw wraps past 180)
            a = math.radians(knee + 60.0)
            return (0.0, 1.0, 0.0, math.sin(a), math.cos(a), 0.0)
        for i in range(200):
            th, _ = kf.accel(i * 0.01, v6(min(140.0, i * 1.0)))
        self.assertAlmostEqual(kf.x[0], 140.0, delta=3.0)
        self.assertTrue(kf.camera(2.5, 140.0))           # stamped 0.5 s ahead of the newest sample
        kf.accel(2.0, v6(140.0))
        self.assertEqual(kf.t_acc, 2.0)                  # the next sample was still used

    def test_single_frame_after_dropout_is_not_believed(self):
        run = mock_sensors.generate("default", seed=3, duration=12.0, cam_drop=0.0, cam_lost_rate=0.0,
                                    cam_outlier=0.0)
        _, kf = run_filter(run, w=0.75)
        t = run["accel"][-1][0]
        theta = kf.x[0]
        self.assertFalse(kf.camera(t + kf.cam_delay, theta + 25.0))   # one wild frame: rejected
        self.assertAlmostEqual(kf.x[0], theta, delta=1.0)
        for i in range(1, 4):                                           # three that agree: relock
            kf.accel(t + i * 0.03, run["accel"][-1][1])
            ok = kf.camera(t + i * 0.03 + kf.cam_delay, theta + 25.0)
        self.assertTrue(ok)
        self.assertAlmostEqual(kf.x[0], theta + 25.0, delta=3.0)

    def test_camera_delay_is_learned(self):
        for stamp in (0.01, 0.05):          # the mock stamps frames this long after the exposure
            run = mock_sensors.generate("default", seed=4, duration=60.0, cam_stamp=(stamp, stamp))
            _, kf = run_filter(run, w=0.75)
            self.assertAlmostEqual(kf.cam_delay, stamp, delta=0.02)

    def test_clock_survives_host_stall(self):
        import leg_filter
        with tempfile.TemporaryDirectory() as d:
            lf = leg_filter.LegFilter(os.path.join(d, "s.json"), 0.0, False)
            ts = []
            for k in range(300):
                arr = 100.0 + k * 0.01 + 0.008
                if 100 <= k < 170:                       # 0.7 s stall, then the backlog in one burst
                    arr = 101.708 + (k - 100) * 5e-5
                lf.on_line("%d,0,1,0,0,1,0" % (k * 10), arr)
                ts.append(lf.t_last)
            gaps = [b - a for a, b in zip(ts, ts[1:])]
            self.assertAlmostEqual(min(gaps), 0.01, delta=0.002)
            self.assertAlmostEqual(max(gaps), 0.01, delta=0.002)

    @staticmethod
    def _calibrate(run):
        s = calibrate.CalSession("hang")
        for t, v, _ in run["accel"]:
            s.feed(t, v)
            if s.done:
                break
        return s

    def test_end_to_end_pipe(self):
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "s.json")
            cmd = [sys.executable, os.path.join(HERE, "leg_filter.py"), "--state", state, "--no-cal",
                   "--auto-zero", "1.5", "--", sys.executable, os.path.join(HERE, "mock_sensors.py"),
                   "--cam-out", state + ".cam", "--duration", "7"]
            t0 = time.monotonic()
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=25).stdout.splitlines()
            ks = [list(map(float, ln.split(",")[1:])) for ln in out if ln.startswith("K,")]
            self.assertGreater(len(ks) / (time.monotonic() - t0), 60.0)   # ~100 Hz incl. start-up
            span = max(k[1] for k in ks) - min(k[1] for k in ks)
            self.assertGreater(span, 20.0)                                 # the knee actually moves
            self.assertTrue(os.path.exists(state + ".stats"))


if __name__ == "__main__":
    unittest.main()
