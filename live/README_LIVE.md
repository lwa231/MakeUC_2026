# Brace Tracker - live demo cheat sheet

## How the pieces connect

```
Arduino (2x MMA7660) --USB serial 115200--> serial_bridge.py --stdout pipe--> build/viewer (main.cpp)
                                              (+4 position cols)               popen() of --pi-cmd
live/run_live.py  =  launcher: finds the port, checks it is free, builds the viewer if missing,
                     then runs:  build/viewer --pi-cmd "<python> serial_bridge.py --port <port>"
```

The viewer reads `t_ms,tx,ty,tz,sx,sy,sz` from the first 7 fields of each line and ignores the rest.
`--demo` swaps the bridge for `fake_pi.py` (same format, no hardware). None of the existing files are modified.

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
| Webcam + ArUco markers | `--camera` (or `--camera 1`), plus `--bridge-args "--marker-cm 5 --cam-flip-x"` |
| Kill an old leftover bridge holding the port | `--kill-stale` |

Expected `--pretty-check` result: about 100 Hz, `mean |a|` near 1 g for both sensors, `RESULT: OK`.

## Viewer keys

| Key | Action |
|---|---|
| **Z** | Zero-calibrate: hold the leg straight for 2 s (live data only) |
| **SPACE** | Toggle the built-in auto-demo sweep (it hands over to LIVE when data arrives) |
| **R** | Reset angle and camera |
| LEFT / RIGHT | Orbit camera |
| UP / DOWN | Manual knee angle (only with demo sweep off, no live data) |
| Q / A, W / S | Thigh / calf circumference +/- |
| - / = | Overall leg size |
| ESC | Quit |

Title/status shows `LIVE`; if data stops for 0.5 s it holds the last angle and shows SIGNAL LOST.
Quit = close the window, ESC, or Ctrl+C in the terminal; `run_live.py` then checks no `serial_bridge.py` is left holding the port.

## Troubleshooting

- **"PORT BUSY ... PID 1234"**: another program has the port. Close the Arduino IDE Serial Monitor/Plotter (or `screen`, a second terminal running the bridge). If it is an old `serial_bridge.py`, run with `--kill-stale`. Two readers on one port split the bytes and give a few Hz of garbage.
- **No data / "NO DATA received"**: use a USB *data* cable; re-upload `brace_sensors_3.ino`; wait ~3 s after plugging in (the board resets when the port opens); run `--list` to see the port name (the mac name changes with the USB port, e.g. `usbmodem1401` vs `usbmodem11401`); pass `--port`.
- **Rate far below 100 Hz**: a second reader on the port, or a flaky cable/hub.
- **A sensor reads ~0 g** (`mean |a|` < 0.5): that sensor is not wired / I2C problem. Thigh = Grove I2C port, shank = Grove D2.
- **Viewer opens but stays in AUTO DEMO**: no valid lines yet. Check the terminal for `[bridge]` messages, and run `--pretty-check`.
- **Knee angle moves the wrong way / wrong axis**: the axis map is compile-time constants in `main.cpp` (`kSagAxis, kVertAxis, kSagSign, kVertSign, kKneeSign`, defaults x / z / +1). Bend the knee and see which axis swings in `--pretty-check`, then change those constants and rebuild with `bash live/build_viewer.sh`. The bridge's position columns have matching flags (`--bridge-args "--sag-axis 1 --vert-axis 2 --knee-sign -1"`); keep both consistent.
- **Angle offset at "straight"**: press **Z** and hold the leg straight 2 s.
- **Camera mode on macOS**: first use asks for camera permission; the camera runs inside the bridge, so grant it to the terminal app you launch from. If the camera cannot open, the bridge continues without it.
- **Viewer binary missing / build fails**: `bash live/build_viewer.sh` needs Homebrew (`glfw`, `glm`) and Xcode command line tools.
- **Always works as a fallback**: `python3 live/run_live.py --demo`.

## Notes

- `run_live.py` is stdlib-only and uses the Python that runs it (`python3` on PATH); packages are installed into that same Python.
- OpenCV (`opencv-contrib-python`, for `cv2.aruco`) is only needed with `--camera`; `run_live.py` installs it on demand (unless `--no-install`).
- Honest limit for the demo: MMA7660 steps are 0.047 g (~2.7 deg), no gyro; keep the motion slow.
