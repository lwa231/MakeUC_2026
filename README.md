# MakeUC_2026: Knee Brace Range-of-Motion Tracker

A 3D-printed knee brace with two accelerometers and three ArUco tags. A webcam and the
accelerometers are fused into a smooth live knee angle that drives a 3D leg model.

```
Arduino (2x MMA7660) --USB--> serial_bridge.py --> live/leg_filter.py --> build/viewer (3D leg)
                                                     ^ fuses camera + accelerometers, runs calibration
live/leg_panel.py (webcam, ArUco tags, settings) ----+
```

## Setup (once, macOS)

1. **Tools:** Xcode command line tools, [Homebrew](https://brew.sh) and [uv](https://docs.astral.sh/uv/).
   ```bash
   xcode-select --install
   ```
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Python environment.** Homebrew's own Python lacks Tk and blocks pip, so use a uv Python:
   ```bash
   uv venv .venv --python 3.12
   ```
   ```bash
   uv pip install --python .venv/bin/python -r requirements.txt opencv-contrib-python
   ```
3. **Build the 3D viewer.** This installs `glfw` and `glm` with Homebrew if they are missing:
   ```bash
   bash live/build_viewer.sh
   ```
4. **Upload the firmware.** Open `brace_sensors_3.ino` in the Arduino IDE and upload it. Thigh sensor
   goes to the Grove **I2C** port, shin sensor to Grove **D2**. Close the Serial Monitor afterwards (it
   holds the port).
5. **Tags.** Print `aruco_tags_44mm.pdf` at 100% / actual size and check the 100 mm scale bar. Stick
   **ID 1** on the knee cap, **ID 0** on the far end of the thigh bar and **ID 2** on the far end of the
   shin bar, all centred on the bar and facing outward.

## Run

Plug in the Arduino, then from the repo folder:

| Goal | Command |
|---|---|
| Check the sensors first (5 s, no window; expect ~100 Hz and `RESULT: OK`) | `.venv/bin/python live/run_live.py --pretty-check 5` |
| Run with the brace | `.venv/bin/python live/run_live.py` |
| Run without hardware (simulated brace and camera) | `.venv/bin/python live/run_live.py --demo` |
| Also save raw data, and watch it in a second terminal | `.venv/bin/python live/run_live.py --log build/session.csv`, then `python3 live/watch_log.py build/session.csv` |

This opens the 3D viewer and a control panel. On the first run, macOS asks for camera permission:
allow it. To quit, close the viewer window, press ESC, or press Ctrl+C in the terminal.

## Calibrate (each time the brace goes on, ~35 s)

1. Strap the brace on, hinge on the outside of the knee. Pick LEFT or RIGHT in the panel.
2. Sit on a high surface, thigh level and supported, foot off the floor.
3. Press **C** in the viewer (or **Calibrate** in the panel) and follow the prompts:
   **hang** the lower leg loose → **straighten** as far as comfortable → **hang** again. Hold each
   about 2 s. "Hold still..." just restarts that hold.
4. Done when it says "Calibrated: best extension N deg short of straight" and the title shows CAL.

Can't bend to about 90°? Hold any angle, measure it with a goniometer, type it next to **Anchor** in
the panel and click Anchor. Optional, once per brace: **Bench check** in the panel (brace off the leg,
follow the four poses) removes each sensor's factory offset.

## Controls

| Viewer key | Panel | Action |
|---|---|---|
| `C` | Calibrate | Start calibration |
| `[` / `]` | Camera share slider | Rely less / more on the camera (default 75%) |
| `;` / `'` | Smooth / responsive slider | Smoother at rest / faster to follow |
| `V` | Fused / Sensors / Camera | Choose the angle source |
| `Z` | Zero (straight) | Current pose reads 0°, only if the leg is truly straight |
| ← / → | | Orbit the 3D view |

## Tests and tuning

```bash
.venv/bin/python -m unittest live/test_live.py
```
```bash
.venv/bin/python live/eval_fusion.py
```
The second (a couple of minutes) sweeps the camera share and smoothing on simulated data and prints
the best settings.

More detail and troubleshooting (busy port, no data, camera issues): [live/README_LIVE.md](live/README_LIVE.md).
The brace CAD and print files are in [hardware/brace](hardware/brace).
