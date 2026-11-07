# Brace Tracker (MakeUC) — Project Context

*Snapshot: 2026-11-07, merged from the original context doc and the planning chat. Written to bring teammates (and any AI chat instances building pieces) up to speed. Each item is labeled DECIDED, RECOMMENDED, or OPEN.*

---

## 1. One-line goal

A wearable knee brace system that measures **knee flexion/extension range of motion (ROM)** during a seated leg extension, shows the knee **live as a 3D model**, and shows **how ROM evolves over time**, so a physical therapy patient/clinician can track progress.

## 2. Scope and framing (DECIDED)

**In scope**
- Measure knee angle during **seated leg extension** (first iteration).
- **Live 3D knee** (C++ / OpenGL with GLFW, GLAD, GLM) driven by the measured angle.
- Per-repetition and per-session ROM metrics.
- ROM **trend over sessions**, with an honest "is this change real or just noise?" indicator.
- Software runs on the laptop; UI lives on the laptop.
- Camera used to mark the **origin** (knee position) and **segment lengths/scale** for the 3D model (see §7).

**Explicitly out of scope**
- No diagnosis. No claim that a patient is "healed". No claim about ligament/graft integrity.
- No internal/tibial rotation detection (accelerometers can't see it; it was dropped).
- No dynamic/gait analysis in v1 (published error is roughly double when the leg moves fast). Possible future work only.

**Language rule for the UI/report:** report measurements against a **user/clinician-set target** ("extension 8° short of your set target", "ROM up 12° since last week, beyond measurement noise"). Never use words like "healed", "recovered", or a condition name as a conclusion. Target values are configurable placeholders; we do not claim clinical thresholds.

## 3. Hardware (DECIDED — only what MLH provides)

Source: MLH hardware list for MakeUC (https://www.mlh.com/resources/hardware?event=makeuc-d5). That page lists no quantities.

**Worn on the patient**
- 1x **Raspberry Pi 4 (4 GB)** with a connector hat. Assume the wiring between parts is fine (the two sensors are connected with Grove cables).
- 2x **Grove 3-axis digital accelerometer**, accelerometer only, **no gyro**. The listing says ±16g, which suggests the ADXL345 variant; verify the chip printed on the board (the software self-check reads the chip ID).
  - one above the knee (thigh), one below (shank), held by a band, wired together.
- Printed **ArUco markers** (and/or a visual mark at the knee) for the camera phase.

**Off-body**
- **Laptop (Windows)** + its webcam (or the Logitech USB webcam from the MLH list) ~3–5 ft from the knee, side-on.
- All analysis and the UI run on the laptop.

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
- Accelerometers measure gravity plus motion: fast movement corrupts tilt estimates. => We gate by |a| ≈ 1 g and keep the exercise slow. With no gyro, there is nothing to fuse with, so smoothing and the quality gate are the main defenses.
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

## 5. Architecture

```
Pi 4 (worn)                    Laptop (Windows)
 2x accelerometer ──TCP──▶  stream server ─▶ raw JSONL logger (always on)
 Python sampler                          ├─▶ preprocess ─▶ tilt ─▶ knee angle
                                         ├─▶ camera (ArUco markers; MediaPipe later)
                                         └─▶ localhost stream ─▶ C++/OpenGL viewer
```

- **Language (DECIDED):** Python for sensing/analysis (Pi sampler + laptop engine); C++ for the 3D viewer.
- **Backend (laptop):** Python 3.11, asyncio, NumPy/SciPy, OpenCV (+ MediaPipe in the camera phase), SQLite. FastAPI + WebSocket only if a web UI is added.
- **3D viewer (RECOMMENDED, one open question below):** C++ with **GLFW + GLAD + GLM**, built with CMake + vcpkg on Windows (`glfw3`, `glad`, `glm`, `opencv4`; optionally `imgui` + `implot`). The viewer is a **thin client**: the Python engine publishes `{t, thigh_tilt, shank_tilt, knee_deg, quality}` over localhost and the viewer only renders it. Keeping the math in Python keeps the data contracts in §6.2 intact.
- **History/trend views:** raw OpenGL is awkward for charts. Use **Dear ImGui + ImPlot** inside the same GLFW/GLAD app, or a small web dashboard (Vite + TypeScript, Three.js) if the team prefers web. Pick one; don't build both.
- **Pi-side OpenGL is not planned.** The Pi 4's Mesa driver tops out around OpenGL 3.1, which conflicts with a modern GLAD 3.3 core setup, and the Pi should stay light.
- **Machine learning (honest version):** no end-to-end neural net (no labeled data, no ground truth, no time). ML-adjacent pieces we *do* use: pretrained MediaPipe pose (no training), a Kalman/complementary filter with *adaptive* sensor trust, a small per-session calibration regression (gain + offset), and a noise-aware trend test.
- **Transport (RECOMMENDED):** Pi connects out to the laptop as a TCP client; laptop is the server. Prefer a wired link (direct Ethernet, or Pi USB gadget Ethernet) over hackathon Wi-Fi, which tends to isolate clients. A Pi-hosted hotspot is the fallback.

## 6. Current focus: accelerometer pipeline (Phase 1)

Camera comes later. First: a clean, timestamped, trustworthy **knee angle stream** from the two accelerometers. Everything else (reps, metrics, UI, fusion) consumes it.

### 6.1 Pi sampler (`pi/sampler.py`)
- Config file (not hardcoded): per sensor name (`thigh`, `shank`), I2C bus, address, rate.
- **First hardware action:** run `i2cdetect -y 1` and confirm both boards appear. Grove ADXL345 boards usually have one fixed address (0x53), so both sensors may collide. If so, give one its own bus (software I2C via the `i2c-gpio` device-tree overlay); the config already supports per-sensor bus/address.
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

**Engine → viewer message (localhost, one per output sample):**
```json
{"type":"angle","t":123999000,"thigh_tilt":12.3,"shank_tilt":47.8,"knee_deg":35.5,"quality":0.97}
```

### 6.3 Laptop ingest (`laptop/ingest/`)
1. asyncio TCP server: validate, check `seq` gaps, track rate/drops/jitter.
2. **Raw logger:** every received message to a timestamped JSONL file, untouched. This is the replay/debug source and lets us retune the algorithms without redoing the exercise.
3. Internal pub/sub bus so logger, preprocessing, and the viewer feed never block each other.

### 6.4 Preprocessing and angle (`laptop/core/`)
1. Resample to a uniform 100 Hz grid on Pi timestamps.
2. Low-pass filter (2nd-order Butterworth ~5 Hz; causal live, zero-phase offline).
3. Quality value 0–1 from |a| vs 1 g.
4. Per-segment tilt via `atan2` of two axes, from a **configurable axis map** (sagittal axis, vertical axis, signs) because mounting orientation is unknown until the sensors are on a leg.
5. Knee angle `θ = tilt_shank − tilt_thigh + offset`.
6. **Zero calibration:** hold the leg straight ~2 s; average to get `offset`. This also removes the need for perfect sensor mounting.
7. Output: `{t, thigh_tilt, shank_tilt, knee_deg, quality}`.

**Angle convention (important — sign bugs are the classic failure):** 0° = full extension, flexion positive.

**Why gravity-only works here:** each sensor mostly reads the gravity vector, so each segment's tilt in the sagittal plane is observable. Yaw is not observable from gravity alone, but the knee is a hinge, so it isn't needed. Hard motion (kicks, jumps) adds acceleration on top of gravity and spikes the reading, which is why the exercise is slow and the quality value exists.

Possible later upgrade: estimate the knee flexion axis from data during a calibration movement so the axis-map config disappears.

### 6.5 Phase 1 milestone (definition of done)
**M1:** the laptop prints live at 100 Hz: thigh tilt, shank tilt, knee angle, quality, plus measured jitter and dropped-sample counts, and saves a raw log. Works with `--fake` first, real Pi second.

### 6.6 Tests
- Angle math at known tilts (0°, 45°, 90°) and the sign convention.
- Resampling with jittered timestamps.
- Gap/drop handling and reconnect.
- Replay of saved JSONL logs through the pipeline.
- Mock generator includes: thigh motion during extension; sensor dropout.

## 7. 3D viewer and camera

### 7.1 Viewer (C++ / GLFW / GLAD / GLM)
- Renders a thigh and a shin (boxes or cylinders) joined at a **hinge at the knee**. The shin rotates about the hinge axis by `knee_deg`; 0° is a straight leg, flexion positive, matching §6.4.
- Build order: (1) window + two-segment leg driven by a **fake slider angle**, (2) receive the localhost angle stream, (3) overlays (big angle readout, target band, rep counter, confidence indicator), (4) optional ImGui/ImPlot history charts, (5) optional camera feed as a textured quad or side window.
- Because the viewer only consumes the §6.2 angle message, it can be built and tested before the sensors work.

### 7.2 Camera role (camera has two jobs; the accelerometers stay primary for angle)
- **Placement, origin and scale (DECIDED idea, details RECOMMENDED):** put **ArUco markers** (OpenCV `cv::aruco` / Python `cv2.aruco`) at marked positions — hip, knee, ankle — or on the thigh and shank bands. Detect them once in a calibration step with the person still. The knee marker gives the 3D model's **origin**. The markers' known printed size gives pixels-to-centimeters, and the hip-to-knee and knee-to-ankle distances give **thigh and shin lengths**, so the model matches the real leg. Prefer ArUco over plain colored tape: it is robust to lighting and each marker has an ID (for example 0 = hip, 1 = knee, 2 = ankle).
- **Angle cross-check (Phase 5):** MediaPipe Pose and/or ArUco-on-bands give a camera knee angle with a per-frame visibility score. Use it to correct slow bias, never as the primary source (see §4 for the error figures). A single dot at the joint is fine for the visual overlay but not for measurement.
- Setup: camera side-on, ~3–5 ft away, fitted clothing, full leg in view.

## 8. Later phases (planned, not started)

**Phase 2 — Reps and metrics**
- Live rep counter: hysteresis state machine. Final numbers: `scipy.signal.find_peaks` on the smoothed (offline) signal with a minimum prominence.
- Per rep: peak extension, peak flexion, ROM (max−min), extension deficit vs. configurable target, duration, peak velocity, smoothness (jerk-based), thigh motion during the rep (compensation flag), confidence.
- Per session: rep count, best/mean ROM, variability.

**Phase 3 — Storage, history, trends**
- SQLite: sessions, reps, calibration, raw-log pointers.
- Trend test: per-session mean rep ROM with a standard error from within-session spread. Between two sessions, call the change meaningful only if `|Δ| > 1.96·sqrt(se₁² + se₂²)`, with a floor equal to the calibration residual. UI labels: improved / declined / within noise.

**Phase 4 — Calibration with known angles**
- Hold 3–4 known angles (e.g., 90°, 60°, 30°, 0°) using an angle block or protractor; fit per-sensor gain and offset by least squares. Show the residual RMSE in the UI as our validity number, and report measured error vs. the anchors in the demo.

**Phase 5 — Camera angle and fusion**
- OpenCV webcam + MediaPipe Pose → sagittal knee angle with a per-frame visibility score.
- Fusion: 1D Kalman filter. Accelerometer angle drives at high rate with adaptive noise (worse when |a| ≠ 1 g); camera angle (~30 Hz) corrects slow bias, noise set by visibility. Post-session smoother for the stored numbers.
- Needs full Pi↔laptop clock sync (handshake + cross-correlation refinement).

**Phase 6 — UI polish and demo**
- Live view: big angle, target band, rep counter, 3D leg, confidence indicator.
- History view: ROM per session with noise band, per-rep table, exportable session report.

## 9. Time plan (about 23 hours remaining at snapshot)

1. Freeze the protocol and angle convention (§6.2, §6.4) so both people can work in parallel.
2. Person A: Pi sampler with `--fake`, then ingest + angle → hit **M1** (fake first, real Pi second).
3. Person B: C++ viewer on a fake slider angle, then on the live localhost stream.
4. Reps and ROM metrics, then known-angle calibration.
5. Camera calibration with ArUco markers (origin + scale), then history/trends.
6. **If time runs short, cut MediaPipe fusion, not the live 3D knee plus per-rep ROM.**

## 10. Repo layout (proposed; supersedes the earlier Pico/Arduino/C++ draft)

```
brace-tracker-makeuc/
├── README.md
├── PROJECT_CONTEXT.md        # this file
├── CMakeLists.txt            # C++ viewer (vcpkg)
├── docs/                     # plan files handed to each builder
├── pi/
│   ├── sampler.py
│   ├── config.example.yaml
│   └── fake_stream.py
├── laptop/
│   ├── ingest/               # tcp server, logger, bus
│   ├── core/                 # resample, filter, tilt, angle, reps, metrics, fusion
│   ├── store/                # sqlite
│   ├── api/                  # localhost publisher (+ FastAPI/WebSocket if web UI)
│   ├── vision/               # ArUco calibration, camera pipeline
│   └── mock/                 # synthetic generators + replay
├── viewer/                   # C++ / GLFW / GLAD / GLM (or ui/ if web is chosen)
└── tests/
```

## 11. Planned plan files (for handing to other chat instances)

1. `00-overview.md` — scope, non-goals, demo script
2. `01-protocol.md` — Pi→laptop message schema, engine→viewer message, sync handshake, angle convention
3. `02-pi-sampler.md` — sensor reader + streamer, self-check, fake mode
4. `03-ingest-and-angle.md` — server, logger, preprocessing, angle, zero calibration
5. `04-engine.md` — reps, metrics, SQLite schema, trends
6. `05-vision.md` — ArUco origin/scale calibration, camera angle, confidence handling
7. `06-ui.md` — viewer, 3D leg, history
8. `07-calibration-validation.md` — known-angle procedure and error reporting
9. `08-mock-and-testing.md` — synthetic data and replay

Each file should end with **acceptance criteria**. Everything except `02-pi-sampler.md` can be built and tested against mock data before the hardware is working.

## 12. Suggested split for two people
- **Person A:** Pi sampler, ingest, preprocessing/angle, reps/metrics, storage.
- **Person B:** viewer, 3D leg, camera/ArUco calibration, history/trend views.
- They meet at the data contracts (§6.2 and the Rep/Fused-state shapes), and the mock generator unblocks Person B immediately.

## 13. Open decisions
1. **UI stack:** C++/OpenGL viewer (current plan, plus ImGui/ImPlot for charts) vs. web dashboard.
2. **Transport:** wired link (recommended) vs. Wi-Fi. Pi as TCP client to the laptop (recommended)?
3. **Sensor chip and addresses:** confirm the chip and whether both boards share an address (`i2cdetect -y 1`).
4. **Camera:** plugged into the laptop (assumed) or the Pi? How many ArUco markers, and where exactly do they go (joints vs. on the bands)?
5. **Sample rate:** 100 Hz assumed; any reason to go higher?
6. **Quantities:** how many accelerometers MLH actually allows (we need 2).
7. **Primary user:** patient doing the exercise, clinician reviewing, or both (default: one app, live patient view + history view).
8. **Target ROM values:** configurable placeholders; who sets them for the demo?

## 14. Known risks
- Both accelerometers may report the same I2C address; if so, give one its own bus (software I2C overlay). The config already supports per-sensor bus/address.
- Mounting orientation of the sensors is unknown until worn => configurable axis map + zero calibration.
- No gyro: fast movement corrupts the angle, so keep the exercise slow and use the quality gate.
- Hackathon Wi-Fi may be unreliable => prefer a wired link (or a Pi hotspot).
- Webcam knee angle can be several degrees off or worse => camera is a cross-check, not the primary source.
- Strap slippage and soft-tissue motion add error => fixed placement, slow movement, per-session calibration.
- Windows C++ dependency setup (GLFW, GLAD, GLM, OpenCV) can eat time => use CMake + vcpkg and test a blank window early.
- This is a prototype; no clinical validation is claimed.
