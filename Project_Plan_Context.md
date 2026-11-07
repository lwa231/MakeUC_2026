# Brace Tracker (MakeUC) — Project Context

*Snapshot: 2026-11-07. Written to bring teammates (and any AI chat instances building pieces) up to speed. Where something is a decision vs. a suggestion vs. an open question, it is labeled.*

---

## 1. One-line goal

A wearable knee brace system that measures **knee flexion/extension range of motion (ROM)** during a seated leg extension and shows **how ROM evolves over time**, so a physical therapy patient/clinician can track progress.

## 2. Scope and framing (DECIDED)

**In scope**
- Measure knee angle during **seated leg extension** (first iteration).
- Per-repetition and per-session ROM metrics.
- ROM **trend over sessions**, with an honest "is this change real or just noise?" indicator.
- Software runs on the laptop; UI lives on the laptop.

**Explicitly out of scope**
- No diagnosis. No claim that a patient is "healed". No claim about ligament/graft integrity.
- No internal/tibial rotation detection (accelerometers can't see it; it was dropped).
- No dynamic/gait analysis in v1 (published error is roughly double when the leg moves fast). Possible future work only.

**Language rule for the UI/report:** report measurements against a **user/clinician-set target** ("extension 8° short of your set target", "ROM up 12° since last week, beyond measurement noise"). Never use words like "healed", "recovered", or a condition name as a conclusion. Target values are configurable placeholders; we do not claim clinical thresholds.

## 3. Hardware (DECIDED — do not re-litigate; we only use what MLH provides)

Source: MLH hardware list for MakeUC (https://www.mlh.com/resources/hardware?event=makeuc-d5). That page lists no quantities.

**Worn on the patient**
- 1x **Raspberry Pi 4** with a connector hat (assume perfect connectivity between parts)
- 2x **Grove 3-axis digital accelerometer** (listing says ±16g, which suggests the ADXL345 variant — verify the chip printed on the board; the software self-check reads the chip ID)
  - one above the knee (thigh), one below (shank), held by a band, wired together
- Optional: a visual marker on the knee joint (for the camera phase)

**Off-body**
- **Laptop** + its webcam (or the Logitech USB webcam from the MLH list) ~3–5 ft from the knee
- All analysis and the UI run on the laptop

The Pi only samples and streams. No processing on the Pi.

## 4. What the research told us

An agent-assisted literature pass was done and the key claims were spot-checked. Treat this as a prototype-justification summary, not clinical guidance.

**Clinical side**
- ROM (especially regaining full extension and reaching ~90° flexion early) is a recognized rehab milestone, but **specific "X° by week Y" tables are protocol-level, not consensus**. The 2017 JOSPT knee-ligament guideline gives no numeric ROM targets. => Our target ranges must be configurable placeholders.
- ROM alone is a weak predictor of long-term outcome (knee-replacement data: early ROM correlated only ~0.1–0.24 with 12-month ROM). Strength and function matter as much. => We pitch the tool as **ROM tracking**, not a measure of healing.
- Measurement noise matters: a phone-goniometer study (60 knee OA/TKA patients) reported minimal detectable change of about 2.7° and 6.3° (the abstract does not say which is flexion vs. extension). => Changes smaller than ~3–6° should not be called real.

**Sensor/camera side**
- Two body-worn inertial sensors (thigh + shank) can measure knee angle with a few degrees of error on **slow, supported movements** such as seated knee extension. Reported: about 2–3° RMSE in some PT-exercise studies; about 5° RMSE in accelerometer-only gait work; worse (about 10° peak error) during walking due to soft-tissue artifact.
- A **single** tilt-only accelerometer is only valid if the other limb segment stays still. We use two.
- Accelerometers measure gravity plus motion: fast movement corrupts tilt estimates. => We gate by |a| ≈ 1 g and keep the exercise slow.
- **Webcam pose estimation (MediaPipe) is the noisier source.** Some studies on healthy adults report ~3–5° mean error; two 2026 studies found much worse (RMSE ~15° static in one; ~14° underestimation in squats in another). => Camera is a **cross-check and a bias corrector, not the primary measurement.**
- Camera + wearable fusion improves accuracy in the literature (≈2.9° vs ≈3.9° video-only in one gait preprint) but those systems were heavier than what we can build here.

**Sources (opened/checked)**
- Mehta et al., iPhone goniometer app, J Knee Surg 2017 (PMID 27894147): https://thieme-connect.de/products/ejournals/abstract/10.1055/s-0036-1593877
- Köglberger et al., early TKA ROM, Arch Orthop Trauma Surg 2022 (PMID 33768277): https://pmc.ncbi.nlm.nih.gov/articles/PMC9474334
- Perspective on strength/function tests after TKA, Phys Ther: https://pmc.ncbi.nlm.nih.gov/articles/9393064
- Hullfish et al., low-cost wearable knee sensor, J Biomech 2019 (PMID 30981425): https://pmc.ncbi.nlm.nih.gov/articles/PMC6587970
- Hamilton et al., MediaPipe/HRNet vs. mocap, J Bodyw Mov Ther 2024: https://orca.cardiff.ac.uk/168259/1/1-s2.0-S1360859224002213-main.pdf
- Yue et al., BlazePose knee angle, JMIR Rehabil Assist Technol 2026 (PMID 42976294): https://preprints.jmir.org/preprint/102399
- Cosoli et al., MediaPipe squat characterization 2026: https://iris.uniecampus.it/handle/11389/91575
- Peiffer et al., IMU + smartphone video fusion, arXiv 2024: https://arxiv.org/abs/2405.17368

Cited by the research agents but **not independently opened** (verify before relying on them): Djurić-Jovičić 2011 (Sensors, PMC3274301), Bell 2019 (Sensors 19(5):1021), the 2017 JOSPT knee ligament CPG, Russo 2026, Langley 2026.

## 5. Architecture (DECIDED unless marked)

```
Pi 4 (worn)                    Laptop
 2x accelerometer ──TCP──▶  stream server ─▶ raw JSONL logger (always on)
 Python sampler                          ├─▶ preprocess ─▶ tilt ─▶ knee angle
                                         ├─▶ [later] camera + fusion
                                         └─▶ API / WebSocket ─▶ browser UI
```

- **Language:** Python everywhere for sensing/analysis (Pi sampler + laptop engine).
- **Backend (laptop):** Python 3.11, asyncio, FastAPI + WebSocket, NumPy/SciPy, OpenCV + MediaPipe (camera phase), SQLite.
- **UI:** **OPEN DECISION.** Recommendation: browser dashboard (Vite + TypeScript, Three.js for the 3D leg, a light chart library) because the product is trends/history. Alternative: keep the earlier C++/OpenGL window. Pick one; don't build both.
- **Machine learning (honest version):** no end-to-end neural net (no labeled data, no ground truth, no time). ML-adjacent pieces we *do* use: pretrained MediaPipe pose (no training), a Kalman/complementary filter with *adaptive* sensor trust, a small per-session calibration regression (gain + offset), and a noise-aware trend test.
- **Transport (OPEN, recommended):** Pi connects out to the laptop as a TCP client; laptop is the server. Prefer a wired link (direct Ethernet or Pi USB gadget Ethernet) over hackathon Wi-Fi, which tends to isolate clients.

## 6. Current focus: accelerometer pipeline (Phase 1)

Camera comes later. First: a clean, timestamped, trustworthy **knee angle stream** from the two accelerometers. Everything else (reps, metrics, UI, fusion) consumes it.

### 6.1 Pi sampler (`pi/sampler.py`)
- Config file (not hardcoded): per sensor name (`thigh`, `shank`), I2C bus, address, rate.
- **Startup self-check:** read chip ID; exit with a clear error if wrong.
- `smbus2`, burst-read the 6 data bytes. Sensor output rate 200 Hz, full-resolution, ±16g; read at **100 Hz**; convert to g on the Pi (stream is in physical units).
- Schedule reads against absolute deadlines (`next += period`), timestamp with `time.monotonic_ns()`. Expect some Linux jitter; the laptop resamples.
- TCP client, newline-delimited JSON, batches of ~10 samples (~100 ms), auto-reconnect with backoff, short local buffer during drops.
- Optional later upgrade: use the chip FIFO and timestamp by index if jitter hurts.
- `--fake` mode that emits the identical protocol (synthetic extension + noise + jitter + dropped packets) so the laptop side can be built without the Pi.

### 6.2 Wire protocol v1
```json
{"type":"hello","proto":1,"device":"brace-pi","rate_hz":100,
 "sensors":["thigh","shank"],"units":"g","t0_ns":123456789}
{"type":"samples","seq":1042,
 "s":[{"t":123999000,"thigh":[0.01,-0.02,0.99],"shank":[0.50,0.00,0.86]}]}
{"type":"stat","seq":1043,"dropped_reads":0,"loop_jitter_us_p99":900}
{"type":"error","msg":"..."}
```
`t` = Pi monotonic nanoseconds. The laptop also records its own arrival time. Full Pi↔laptop clock sync (repeated round trips, pick the lowest-RTT estimate) is deferred until the camera joins; `hello` reserves room for it.

### 6.3 Laptop ingest (`laptop/ingest/`)
1. asyncio TCP server: validate, check `seq` gaps, track rate/drops/jitter.
2. **Raw logger:** every received message to a timestamped JSONL file, untouched. This is the replay/debug source and lets us retune the algorithms without redoing the exercise.
3. Internal pub/sub bus so logger, preprocessing, and WebSocket never block each other.

### 6.4 Preprocessing and angle (`laptop/core/`)
1. Resample to a uniform 100 Hz grid on Pi timestamps.
2. Low-pass filter (2nd-order Butterworth ~5 Hz; causal live, zero-phase offline).
3. Quality value 0–1 from |a| vs 1 g.
4. Per-segment tilt via `atan2` of two axes, from a **configurable axis map** (sagittal axis, vertical axis, signs) because mounting orientation is unknown until the sensors are on a leg.
5. Knee angle `θ = tilt_shank − tilt_thigh + offset`.
6. **Zero calibration:** hold the leg straight ~2 s; average to get `offset`.
7. Output: `{t, thigh_tilt, shank_tilt, knee_deg, quality}`.

**Angle convention (important — sign bugs are the classic failure):** 0° = full extension, flexion positive.

Possible later upgrade: estimate the knee flexion axis from data during a calibration movement so the axis-map config disappears.

### 6.5 Phase 1 milestone (definition of done)
**M1:** the laptop prints live at 100 Hz: thigh tilt, shank tilt, knee angle, quality, plus measured jitter and dropped-sample counts, and saves a raw log. Works with `--fake` first, real Pi second.

### 6.6 Tests
- Angle math at known tilts (0°, 45°, 90°) and the sign convention.
- Resampling with jittered timestamps.
- Gap/drop handling and reconnect.
- Replay of saved JSONL logs through the pipeline.
- Mock generator includes: thigh motion during extension; sensor dropout.

## 7. Later phases (planned, not started)

**Phase 2 — Reps and metrics**
- Live rep counter: hysteresis state machine. Final numbers: `scipy.signal.find_peaks` on the smoothed (offline) signal with a minimum prominence.
- Per rep: peak extension, peak flexion, ROM (max−min), extension deficit vs. configurable target, duration, peak velocity, smoothness (jerk-based), thigh motion during the rep (compensation flag), confidence.
- Per session: rep count, best/mean ROM, variability.

**Phase 3 — Storage, history, trends**
- SQLite: sessions, reps, calibration, raw-log pointers.
- Trend test: per-session mean rep ROM with a standard error from within-session spread. Between two sessions, call the change meaningful only if `|Δ| > 1.96·sqrt(se₁² + se₂²)`, with a floor equal to the calibration residual. UI labels: improved / declined / within noise.

**Phase 4 — Calibration with known angles**
- Hold 3–4 known angles (e.g., 90°, 60°, 30°, 0°) using an angle block or protractor; fit per-sensor gain and offset by least squares. Show the residual RMSE in the UI as our validity number, and report measured error vs. the anchors in the demo.

**Phase 5 — Camera**
- OpenCV webcam + MediaPipe Pose → sagittal knee angle with a per-frame visibility score. Camera placed side-on, fitted clothing, full leg in view.
- Idea to evaluate: ArUco tags on the thigh and shank bands (OpenCV) for a camera angle that doesn't depend on pose estimation; a single dot at the joint is fine for the visual overlay but not for measurement.
- Fusion: 1D Kalman filter. Accelerometer angle drives at high rate with adaptive noise (worse when |a| ≠ 1 g); camera angle (~30 Hz) corrects slow bias, noise set by visibility. Post-session smoother for the stored numbers.
- Needs full Pi↔laptop clock sync (handshake + cross-correlation refinement).

**Phase 6 — UI polish and demo**
- Live view: big angle, target band, rep counter, 3D leg, confidence indicator.
- History view: ROM per session with noise band, per-rep table, exportable session report.

## 8. Repo layout (proposed; supersedes the earlier Pico/Arduino/C++ draft)

```
brace-tracker-makeuc/
├── README.md
├── PROJECT_CONTEXT.md        # this file
├── docs/                     # plan files handed to each builder
├── pi/
│   ├── sampler.py
│   ├── config.example.yaml
│   └── fake_stream.py
├── laptop/
│   ├── ingest/               # tcp server, logger, bus
│   ├── core/                 # resample, filter, tilt, angle, reps, metrics, fusion
│   ├── store/                # sqlite
│   ├── api/                  # FastAPI + WebSocket
│   ├── vision/               # camera pipeline (later)
│   └── mock/                 # synthetic generators + replay
├── ui/                       # web dashboard (or C++ viewer if chosen)
└── tests/
```

## 9. Planned plan files (for handing to other chat instances)

1. `00-overview.md` — scope, non-goals, demo script
2. `01-protocol.md` — Pi→laptop message schema, sync handshake, angle convention
3. `02-pi-sampler.md` — sensor reader + streamer, self-check, fake mode
4. `03-ingest-and-angle.md` — server, logger, preprocessing, angle, zero calibration
5. `04-engine.md` — reps, metrics, SQLite schema, trends
6. `05-vision.md` — camera pipeline and confidence handling
7. `06-ui.md` — dashboard, 3D leg, history
8. `07-calibration-validation.md` — known-angle procedure and error reporting
9. `08-mock-and-testing.md` — synthetic data and replay

Each file should end with **acceptance criteria**. Everything except `02-pi-sampler.md` can be built and tested against mock data before the hardware is working.

## 10. Suggested split for two people
- **Person A:** Pi sampler, ingest, preprocessing/angle, reps/metrics, storage.
- **Person B:** API, UI, 3D leg, history/trend views.
- They meet at the data contracts (§6.2 and the Rep/Fused-state shapes) and the mock generator unblocks Person B immediately.

## 11. Open decisions
1. **UI stack:** web dashboard (recommended) vs. C++/OpenGL viewer. Which frameworks is each teammate comfortable with (React vs. plain TypeScript)?
2. **Transport:** wired link (recommended) vs. Wi-Fi. Pi as TCP client to the laptop (recommended)?
3. **Sample rate:** 100 Hz assumed; any reason to go higher?
4. **Quantities:** how many accelerometers MLH actually allows (we need 2).
5. **Primary user:** patient doing the exercise, clinician reviewing, or both (default: one app, live patient view + history view).
6. **Target ROM values:** configurable placeholders; who sets them for the demo?

## 12. Known risks
- Both accelerometers may report the same I2C address; if so, give one its own bus (software I2C overlay) — config already supports per-sensor bus/address.
- Mounting orientation of the sensors is unknown until worn => configurable axis map + zero calibration.
- Hackathon Wi-Fi may be unreliable => prefer wired link.
- Webcam knee angle can be several degrees off or worse => camera is a cross-check, not the primary source.
- Strap slippage and soft-tissue motion add error => fixed placement, slow movement, per-session calibration.
- This is a prototype; no clinical validation is claimed.
