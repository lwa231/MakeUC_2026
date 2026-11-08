# Brace Tracker - live demo cheat sheet

## How the pieces connect

```
Arduino (2x MMA7660) --USB 115200--> serial_bridge.py --pipe--> live/leg_filter.py --K/M lines--> build/viewer
   (or live/mock_sensors.py with --demo)                         ^  knee_estimator (Kalman + camera)
                                                                 |  calibrate (Hang and Straighten)
live/leg_panel.py (Tk): leg switch, webcam + ArUco, Fusion card -+  via build/leg_state.json{,.cam,.tune}
live/run_live.py  =  launcher: finds the port, builds the viewer if missing, opens the panel, then runs
                     build/viewer --state build/leg_state.json --pi-cmd "<python> live/leg_filter.py ... -- <source>"
```

`leg_filter.py` fuses the two accelerometers with the camera's (late) knee angle and sends the viewer
`K,t_ms,theta,omega,flags,w` lines at ~100 Hz plus `M,level,text` calibration messages. `--demo` uses
`mock_sensors.py`: a simulated brace (6-bit accelerometers, mounting errors) and a laggy, noisy camera
(`--scenario default|fast|slow|badcam|calib`).

## Before the demo (once)

```bash
cd repo
python3 -m pip install -r requirements.txt      # pyserial (run_live.py also does this itself if missing)
bash live/build_viewer.sh                        # builds build/viewer (run_live.py does this itself if missing)
```

## Commands (run from anywhere; shown from `repo/`)

| Goal | Command |
|---|---|
| Real board -> 3D viewer | `python3 live/run_live.py` |
| Check data first (3-5 s, prints Hz + sample lines, no window) | `python3 live/run_live.py --pretty-check 5` |
| No hardware / fallback | `python3 live/run_live.py --demo` |
| Just show the command | `python3 live/run_live.py --dry-run` |
| List ports and whether they are busy | `python3 live/run_live.py --list` |
| Other port | `python3 live/run_live.py --port /dev/cu.usbmodem11401` (Windows: `--port COM5`) |
| Real leg size | `python3 live/run_live.py --viewer-args "--thigh-len 46 --shin-len 43 --thigh-circ 56 --calf-circ 38"` |
| Bridge options | `--bridge-args "--thigh-sensor-cm 18 --shank-sensor-cm 16 --zero-secs 2"` |
| Plain 7-field lines | `--no-position` |
| Save every sample to CSV | `--log session.csv` |
| Webcam + ArUco markers (shown in the panel) | `--camera 1` to pick the camera, `--marker-cm 5`, `--no-camera` to skip |
| Kill an old leftover bridge holding the port | `--kill-stale` |

Expected `--pretty-check` result: about 100 Hz, `mean |a|` near 1 g for both sensors, `RESULT: OK`.

## Viewer keys

| Key | Action |
|---|---|
| **C** | Calibrate (Hang and Straighten, below); prompts appear in the title bar and the panel |
| **Z** | Quick zero: the current pose reads 0 deg. Only if the leg is truly straight |
| **[** / **]** | Camera weight w -/+ 0.05 (0 = accelerometers only, 1 = camera only) |
| **;** / **'** | Responsiveness -/+ 0.1 (low = smoother at rest, high = follows faster) |
| **V** | Cycle source: fused / sensors / camera |
| **SPACE** | Toggle the built-in auto-demo sweep (it hands over to LIVE when data arrives) |
| **R** | Reset angle and camera |
| LEFT / RIGHT | Orbit camera |
| UP / DOWN | Manual knee angle (only with demo sweep off, no live data) |
| Q / A, W / S | Thigh / calf circumference +/- |
| - / = | Overall leg size |
| ESC | Quit |

Title/status shows the knee angle, w, the source, CAL/UNCAL and the latest calibration message.

The panel's **Fusion** card does the same with the mouse: camera share slider, smooth/responsive
slider, source buttons (Fused / Sensors / Camera), Calibrate, Anchor [deg], Bench check, Zero (straight),
and one status line (prompt or result). Settings go to `build/leg_state.json.tune`.

## Calibration (each session, ~35 s)

1. Sit on a chair or table edge, thigh supported and level. Press **C** (or Calibrate).
2. **Hang**: let the lower leg hang loose and stay still (2 s). This is the 90 deg reference.
3. **Straighten** as far as is comfortable and hold (2 s). This finds the bend direction and your
   extension deficit.
4. **Hang again** (2 s): checks the strap did not slip. The result is saved per leg in `live/leg_cal.json`
   and reused for 4 h.

If the knee cannot bend to 90 deg, hold any angle, measure it with a goniometer, type it next to
**Anchor** and press Anchor. **Bench check** (once per brace, `live/brace_cal.json`): with the brace
at 0 deg off the leg, hold it hip end up, hip end down, front edge down, back edge down when prompted;
it measures each sensor's zero-g offset and gain. Neither procedure cares how the boards are turned in
their pockets. "Hold still..." just means the current hold restarted; each step waits up to 30 s. Only a
re-hang more than 5 deg off the first hang ("strap may have slipped") or a timeout ends the run: press C again.
Quit = close the window, ESC, or Ctrl+C in the terminal; `run_live.py` then checks no `serial_bridge.py` is left holding the port.

## Troubleshooting

- **"PORT BUSY ... PID 1234"**: another program has the port. Close the Arduino IDE Serial Monitor/Plotter (or `screen`, a second terminal running the bridge). If it is an old `serial_bridge.py`, run with `--kill-stale`. Two readers on one port split the bytes and give a few Hz of garbage.
- **No data / "NO DATA received"**: use a USB *data* cable; re-upload `brace_sensors_3.ino`; wait ~3 s after plugging in (the board resets when the port opens); run `--list` to see the port name (the mac name changes with the USB port, e.g. `usbmodem1401` vs `usbmodem11401`); pass `--port`.
- **Rate far below 100 Hz**: a second reader on the port, or a flaky cable/hub.
- **A sensor reads ~0 g** (`mean |a|` < 0.5): that sensor is not wired / I2C problem. Thigh = Grove I2C port, shank = Grove D2.
- **Viewer opens but stays in AUTO DEMO**: no valid lines yet. Check the terminal for `[bridge]` messages, and run `--pretty-check`.
- **Knee angle moves the wrong way / wrong axis**: the axis map is compile-time constants in `main.cpp` (`kSagAxis, kVertAxis, kSagSign, kVertSign, kKneeSign`, defaults x / z / +1). Bend the knee and see which axis swings in `--pretty-check`, then change those constants and rebuild with `bash live/build_viewer.sh`. The bridge's position columns have matching flags (`--bridge-args "--sag-axis 1 --vert-axis 2 --knee-sign -1"`); keep both consistent.
- **Angle offset at "straight"**: run the calibration (**C**). **Z** only if the leg is truly straight.
- **Camera on macOS**: first use asks for camera permission; the camera runs in the panel (`leg_panel.py`), so grant it to the terminal app you launch from. Without a camera, fusion falls back to the accelerometers.
- **No panel window**: Homebrew's `python3` has no Tk. Run `run_live.py` with a python.org Python, or use the viewer keys.
- **Viewer binary missing / build fails**: `bash live/build_viewer.sh` needs Homebrew (`glfw`, `glm`) and Xcode command line tools.
- **Always works as a fallback**: `python3 live/run_live.py --demo`.

## Notes

- `run_live.py` is stdlib-only and uses the Python that runs it (`python3` on PATH); packages are installed into that same Python.
- OpenCV (`opencv-contrib-python`, for `cv2.aruco`) is only needed for the camera view; `run_live.py` installs it on demand (unless `--no-install`).
- Honest limit for the demo: MMA7660 steps are 0.047 g (~2.7 deg), no gyro; keep the motion slow.
