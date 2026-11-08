#!/usr/bin/env python3
"""A few end-to-end checks that the demo works: run  python3 -m unittest live/test_live.py"""
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
        ctx = eval_fusion.prepare("default", 1, 30.0)
        new, old = eval_fusion.evaluate(ctx, KE.W_DEFAULT, KE.RESP_DEFAULT), eval_fusion.evaluate_baseline(ctx)
        self.assertLess(new["jitter"], 0.6 * old["jitter"])      # still at rest
        self.assertLess(new["rms"], 1.1 * old["rms"])             # as accurate, despite the playout delay
        self.assertLess(new["lag"], old["lag"] + 10.0)            # and about as responsive

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
