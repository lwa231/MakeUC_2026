# Camera + sensor fine-tuning (simulation only) — notes

Local clone of `github.com/lwa231/MakeUC_2026` at `MakeUC2026/tuning_clone`, branch **`sim-tuning`**
(based on upstream `31e96a7`). Nothing was pushed: the clone's push URL is set to `DISABLED_NO_PUSH`.
No hardware was connected, so every result below comes from simulation. No new user-facing features:
same files, same commands, same keys/panel. `main.cpp`, `serial_bridge.py` and `brace_sensors_3.ino` are unchanged.

## What changed

### `live/knee_estimator.py` (sensor fusion; same public API)
- **Swing model.** The shank board sits 118 mm from the hinge (`hardware/brace/brace.py`, `sensor_pos`), so in a
  rep it also feels r·ω² and r·α. That tilts its gravity reading by up to 10–30°. It is now part of the
  measurement model, `(r/g)(ω² cos θ + α sin θ)`, which holds for any board mounting, as an extended
  Kalman filter with state [θ, ω, α, b]. Subtracting it from the input instead was unstable, because α is
  too noisy.
- **Camera timing.** `t_cap` is stamped when the frame reaches Python, not at exposure. The filter now *learns*
  the delay from motion (innovation ∝ −ω·delay error) and moves each frame back by it. Frames are never placed
  ahead of the accelerometer stream; before, a frame stamped ahead swallowed the next accel samples.
- **Camera noise is learned** from frames at rest (`cam_var`).
- **Outliers and relock.** One wild frame (e.g. right after a dropout) can no longer take over. The filter
  relocks only when ≥3 rejected frames within 0.4 s lie on a line. Before, any frame after a 1 s dropout
  was believed unconditionally.
- **Learned accelerometer error map.** The boards' zero-g offset and gain errors make the accel error change
  with the knee angle, which one constant offset `b` can't follow.
  - While the camera sees a still knee and has agreed with the filter for 1.5 s, the filter learns the
    accelerometers' total disagreement with the camera. The map is a constant plus sin/cos of θ and 2θ,
    with a slow leak and a ±10° cap.
  - It keeps correcting the accelerometers when the camera drops out.
- **No camera frames** (no webcam, markers lost, or Sensors source): the filter uses a constant-velocity model
  like the original (ablation: the swing model needs a good α, which only the camera-steadied model has), with
  its own retuned settings. Camera share `w` no longer discounts the accelerometers when no camera frames are
  arriving; before, a demo without a webcam was needlessly sluggish.
- **Camera-only source:** constant-velocity model. Before, it could run away to ±180° and stay there.
- **Robustness:**
  - All angle differences are wrapped. Before, a raw reading past ±180° (e.g. k = 60 and knee 140) jumped
    by 360°.
  - NaN/inf input is ignored instead of poisoning the filter for good.
  - The state is kept in a physical range in every mode.
  - The coasting display no longer moves the display filter's clock (no freeze-then-snap after a USB stall).
  - Replay after a late frame keeps relock entries and the accel reading.
- Constants marked `(tuned)` were fitted with `tuning/optimize.py` on mock scenarios plus frames rendered through
  the real ArUco tracker, then checked on held-out seeds.

### `camera_origin.py` (ArUco tracker; same API and `CameraFix` fields)
- **View correction.** A camera not square to the leg distorts the angle (foreshortening).
  - Every marker's shape gives the image→leg-plane stretch (affine, rotation-invariant).
  - The static knee marker's corners, averaged over time, give a full perspective homography. The arm-end
    markers are mapped back through it.
  - Both are averaged over ~1–2 s and ramp in by elapsed time. Only the configured marker ids are used.
- **Thigh hold.** If only the hip-end marker is lost (knee + ankle seen), the last thigh heading is held for
  up to 1 s, at reduced confidence. In seated extension the thigh hardly moves.
- **Confidence** now predicts the expected angle error (view anisotropy, marker size spread / small markers,
  shape disagreement, held heading). Before, every full fix scored ≥ 0.6, so the 0.5 gate never fired.
- Detector parameters were tested and left unchanged: the best variant gave +1.8 points of fix rate for
  1.35× the time.

### `live/leg_filter.py`
- Device→host clock: a host stall over 0.5 s no longer re-anchors the clock and squeezes the buffered
  backlog into ~6 ms. It re-anchors only on a device restart (t_ms goes back) or a lasting jump.
- Camera confidence is passed to the filter.
- On a leg/mapping change the learned camera delay and noise are kept.
- The panel's `offset_deg` stat reads the right state element.

### `live/mock_sensors.py` (the `--demo` mock and test source)
More realistic, so the tuning is not fitted to an over-kind mock:
- Camera stamp delay.
- Camera error like the rendered tracker: small noise plus an angle-dependent bias.
- Frames lost to motion blur.
- Per-brace lever arm (0.10–0.14 m), sensor noise and out-of-plane tilt.
- Optional strap slip.
- Docstring axis fix.

### `live/test_live.py`
New tests:
- Angle wrap and late stamps.
- A single frame after a dropout is not believed.
- The camera delay is learned.
- The clock survives a host stall.

The smoothness test now averages 3 seeds. The original filter fails the old single-seed version on the
realistic mock too. All 10 tests pass: `python3 -m unittest live.test_live`.

### New dev tools (not used live)
| Tool | What it does |
|---|---|
| `live/sim_camera.py` | Renders synthetic ArUco frames (pinhole camera, yaw/pitch, distortion, blur, noise, JPEG, leg or brace mounting) and scores `CameraTracker`. `--selftest`, `--seq`, `--mount brace` |
| `tuning/fusion_bench.py` | Filter on 9 mock scenarios: default, fast, slow, badcam, lowfps, latestamp, fastbad, nocam, slip |
| `tuning/e2e_camera.py` | Mock motion → rendered frames → original *and* new tracker → filter |
| `tuning/optimize.py` | Coordinate search over the filter constants (`--mode both`, `accel` or `combo`) |
| `tuning/validate.py` | Score a parameter set on train and held-out seeds |
| `tuning/ablate_accel.py` | Ablation of the no-camera filter |
| `tuning/knee_estimator_v1_orig.py`, `tuning/camera_origin_v1_orig.py` | The originals, for A/B runs |

## Results (simulation)

Metrics:
- **rms / p95**: error of what the viewer draws vs the true angle (deg).
- **lag**: median ms behind the truth at mid-move.
- **jit**: rest jitter (deg).
- **J** = rms + lag/40 + 10·jit (the teammates' score; lower is better).

**Most realistic: frames rendered through the tracker, markers on the brace as in the CAD.** 24 runs:
default / fast / slow motion × 2 seeds × 4 camera setups (good, 18° angled, far 640×480, dim). Every
frame is tracked by both the original and the new `camera_origin.py`.

| Stack (camera + filter) | Fused rms | p95 | lag | jit | J | Accel-only rms | Camera-only rms |
|---|---|---|---|---|---|---|---|
| original + original | 7.81 | 16.1 | 88 | 0.24 | 12.45 | 6.31 | 23.5 |
| original camera + new filter | 4.77 | 9.2 | 48 | 0.26 | 8.62 | 6.18 | 18.2 |
| new camera + original filter | 7.56 | 15.5 | 88 | 0.27 | 12.49 | 6.31 | 22.5 |
| **new + new** | **4.47** | **8.7** | **47** | **0.25** | **8.10** | 6.18 | 17.7 |

Camera alone, angled setup: 2.27° → 0.42° error. Curved-leg mounting gives the same picture: fused
7.85 → 4.59 rms. Full table: `tuning/e2e_final_matrix.txt`.

**Mock scenarios, held-out seeds** (10 scenarios incl. no-webcam, strap slip, 6–10 s camera loss;
`tuning/bench_v8_test.txt` vs `tuning/bench_orig_test_final.txt`):

| Mode | Original rms / p95 / lag / jit | New rms / p95 / lag / jit |
|---|---|---|
| fused (w = 0.75) | 7.73 / 16.0 / 85 / 0.46 | 5.44 / 10.5 / 53 / 0.45 |
| accel only | 5.70 / 11.4 / 43 / 0.38 | 5.67 / 11.4 / 45 / 0.29 |
| no webcam at default settings | 4.73 / 9.0 / 45 / 0.30 | 4.40 / 8.5 / 43 / 0.28 |
| camera only | 30.4 / 57.3 / 113 / 3.1 | 27.7 / 54.7 / 118 / 1.8 |

Other checks:
- **At rest with the camera** the fused angle now tracks the camera: ≈0.7–2° in default/slow runs (was 3–5°).
- **Long sessions** (`tuning/scratch/long_session.py`: 15 min with camera, then 60 s without, 4 seeds):

  | | Original | New |
  |---|---|---|
  | Fused rms, last minute | 2.4–5.5° | 2.0–4.6° |
  | Accel-only rms after the camera is lost | 3.0–4.7° | 2.0–4.8° (better on 3 of 4 seeds) |

  The learned error map stays small (|c| ≤ 4°).

## Review loop
Three independent Sonnet reviews of the diff; every substantiated finding was fixed:
- **Review 1:**
  - Single-frame relock after a dropout.
  - Late `t_cap` swallowing accel samples.
  - Missing angle wrap.
  - Motion weighting blind to tangential acceleration.
  - Clock squeeze after a host stall.
  - Coast display freeze.
  - Calibration straighten direction: not fixed, see limits.
- **Review 2:**
  - Offset stat index.
  - NaN poisoning.
  - Camera frames piling up while the accel stream is silent.
  - Learned delay lost on remap.
  - Inconsistent predict models.
  - Stray-marker / ramp / heading-source issues in the camera code.
- **Review 3:**
  - Error map and offset drifting together over long sessions. Fixed structurally: the map learns the total
    accel−camera disagreement, plus a leak, a cap and a calm gate.
  - Guard pinning ω.
  - Garbage kept after a camera sign flip.
  - Too-eager frame skipping.
  - Camera ramp steps.

## Honest limits / not done
- Everything is simulated. The real camera latency, lighting, marker print quality and strap behaviour may
  differ. The filter learns delay, camera noise and accel error online for that reason.
- Fast reps (a 90° move in 0.5 s) remain the weak case: the ankle marker blurs, and the shank board's swing
  acceleration reaches 0.5 g.
- Camera-only mode is inherently ~100 ms late. It is a diagnostic view.
- Hang & Straighten calibration still can't tell "straighten" from "flex further". A ±180° board flip makes
  the two look alike. A user who flexes further than the hang can get an inverted sign. Not changed.
- `leg_filter`'s camera sign auto-detection (`cam_mean`) is unchanged.
- Firmware not changed: the 100 Hz polling of a 120 Hz sensor can't be tested without hardware.

## Reproduce
```bash
cd MakeUC2026/tuning_clone
python3 -m unittest live.test_live
python3 live/sim_camera.py --selftest
python3 tuning/fusion_bench.py --test
KE_FILE=tuning/knee_estimator_v1_orig.py python3 tuning/fusion_bench.py --test     # original filter
E2E_MOUNT=brace python3 tuning/e2e_camera.py                                        # renders on first run (~10 min)
E2E_MOUNT=brace E2E_CAM=orig KE_FILE=tuning/knee_estimator_v1_orig.py python3 tuning/e2e_camera.py
```
