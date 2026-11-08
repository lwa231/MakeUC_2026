#!/usr/bin/env python3
"""leg_panel.py - Brace Tracker control panel (Tk), runs beside the 3D viewer.

    leg_panel.py --state STATE.json [--leg left|right] [--cam N] [--no-camera] [--marker-cm 5]

* LEFT | RIGHT segmented switch (live, no restart). Writes STATE.json, which
  leg_filter.py re-reads ~10x/s:   {"leg", "invert", "rot180", "chosen"}
* Live camera viewport: the panel owns the webcam (camera_origin.CameraTracker,
  unmodified) and draws every ArUco marker it recognises. Three chips show which
  of the 3 markers are seen: KNEE #1, THIGH END #0, SHIN END #2.
* Sensor stream indicator (reads STATE.json.stats written by leg_filter.py).
* Small fix chips if a leg still looks wrong: Invert bend / Flip shin 180 / Mirror camera.
* Fusion card: camera share (w), responsiveness, source (Fused / Sensors / Camera), Calibrate,
  Anchor, Bench check and Zero, plus leg_filter's calibration status. Writes STATE.json.tune
  {w, resp, src, zero_seq, cal_seq, cal_mode, cal_anchor_deg}; leg_filter polls it.
Per-leg chip settings and the last leg are remembered in live/leg_config.json.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from knee_estimator import RESP_DEFAULT, W_DEFAULT  # noqa: E402
CONFIG = os.path.join(HERE, "leg_config.json")
DEFAULTS = {"left": {"invert": False, "rot180": False},
            "right": {"invert": True, "rot180": False}}   # right = mirrored brace

BG, CARD, TEXT, MUTED = "#0f1419", "#1a212b", "#e6edf3", "#8b98a5"
GREEN, RED, BLUE, AMBER, OFF = "#2ea043", "#f85149", "#58a6ff", "#d29922", "#2b3440"
FONT = "Helvetica Neue"


# ------------------------------------------------------------------ config ----
def load_cfg():
    cfg = {"last_leg": "right", "mirror_cam": False, "assist": True}
    cfg.update({k: dict(v) for k, v in DEFAULTS.items()})
    try:
        with open(CONFIG) as f:
            d = json.load(f)
        for leg in ("left", "right"):
            cfg[leg].update({k: bool(v) for k, v in d.get(leg, {}).items() if k in cfg[leg]})
        if d.get("last_leg") in ("left", "right"):
            cfg["last_leg"] = d["last_leg"]
        cfg["mirror_cam"] = bool(d.get("mirror_cam", False))
        cfg["assist"] = bool(d.get("assist", True))
    except (OSError, ValueError):
        pass
    return cfg


def save_cfg(cfg):
    try:
        with open(CONFIG, "w") as f:
            json.dump(cfg, f, indent=1)
    except OSError:
        pass


def write_state(path, leg, invert, rot180, assist=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"leg": leg, "invert": bool(invert), "rot180": bool(rot180), "assist": bool(assist),
                   "chosen": True}, f)
    os.replace(tmp, path)


def initial_state(path, leg=None):
    """Write the starting state (used by run_live.py before the viewer opens). Returns the leg."""
    cfg = load_cfg()
    leg = leg or cfg["last_leg"]
    write_state(path, leg, cfg[leg]["invert"], cfg[leg]["rot180"])
    return leg


def read_json(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def update_tune(path, bump=None, **changes):
    """Read-modify-write STATE.json.tune atomically; bump names a counter to increment."""
    d = read_json(path)
    d.update(changes)
    if bump:
        d[bump] = int(d.get(bump) or 0) + 1
    tmp = path + ".tmp.panel"
    try:
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, path)
    except OSError:
        pass
    return d


# ------------------------------------------------------------------ camera ----
def ppm_bytes(bgr):
    """Raw PPM for tk.PhotoImage (fast; no PNG encoding)."""
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return b"P6 %d %d 255\n" % (w, h) + rgb.tobytes()


MARKERS = (("knee", "KNEE", 1), ("hip", "THIGH END", 0), ("ankle", "SHIN END", 2))


def render_camera(tracker, frame, fix, out_w):
    """Downscale `frame` to out_w wide and draw the recognised markers. Returns BGR image."""
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    s = out_w / float(w)
    img = cv2.resize(frame, (out_w, int(round(h * s))), interpolation=cv2.INTER_AREA)
    names = {tracker.ids.get(k): lab for k, lab, _ in MARKERS}
    for mid, pts in list(tracker.last_detections.items()):
        p = (np.asarray(pts, dtype="float32").reshape(4, 2) * s).astype("int32")
        cv2.polylines(img, [p.reshape(-1, 1, 2)], True, (80, 220, 100), 3, cv2.LINE_AA)
        cx, cy = p.mean(axis=0).astype(int)
        cv2.circle(img, (cx, cy), 4, (80, 220, 100), -1, cv2.LINE_AA)
        label = "%s #%d" % (names.get(mid, "marker"), mid)
        x0, y0 = int(p[:, 0].min()), int(p[:, 1].min()) - 8
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y0 = max(y0, th + 6)
        cv2.rectangle(img, (x0 - 3, y0 - th - 5), (x0 + tw + 3, y0 + 4), (20, 25, 30), -1)
        cv2.putText(img, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 100), 2, cv2.LINE_AA)
    if fix is not None and fix.knee_px is not None:           # knee -> arm-end lines
        k = (int(fix.knee_px[0] * s), int(fix.knee_px[1] * s))
        for key in ("hip", "ankle"):
            pts = tracker.last_detections.get(tracker.ids.get(key))
            if pts is not None:
                c = np.asarray(pts, dtype="float32").reshape(4, 2).mean(axis=0) * s
                cv2.line(img, k, (int(c[0]), int(c[1])), (255, 170, 60), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------- UI ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--leg", choices=("left", "right"))
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--marker-cm", type=float, default=5.0)
    a = ap.parse_args()

    import tkinter as tk

    cfg = load_cfg()
    leg = a.leg or cfg["last_leg"]
    st = {"leg": leg, "invert": cfg[leg]["invert"], "rot180": cfg[leg]["rot180"], "mirror": cfg["mirror_cam"],
          "assist": True}         # the Fusion card's source buttons replace the old assist chip

    root = tk.Tk()
    root.title("Brace Tracker")
    root.configure(bg=BG)
    root.attributes("-topmost", True)
    root.geometry("+30+40")
    root.resizable(False, False)
    W = 560

    def label(parent, text="", size=12, bold=False, fg=TEXT, bg=BG, **kw):
        return tk.Label(parent, text=text, fg=fg, bg=bg, font=(FONT, size, "bold" if bold else "normal"), **kw)

    # header
    head = tk.Frame(root, bg=BG)
    head.pack(fill="x", padx=18, pady=(16, 8))
    label(head, "BRACE TRACKER", 17, True).pack(side="left")
    stream_dot = label(head, "●", 14, fg=RED)
    stream_dot.pack(side="right")
    stream_txt = label(head, "waiting for sensors", 11, fg=MUTED)
    stream_txt.pack(side="right", padx=(0, 6))

    # leg switch
    seg = tk.Frame(root, bg=CARD, highlightthickness=1, highlightbackground=OFF)
    seg.pack(fill="x", padx=18, pady=(0, 10))
    segs = {}

    def make_seg(name):
        b = tk.Label(seg, text=name.upper() + " LEG", font=(FONT, 15, "bold"), pady=12, cursor="hand2")
        b.pack(side="left", expand=True, fill="both")
        b.bind("<Button-1>", lambda e, n=name: set_leg(n))
        segs[name] = b
    make_seg("left")
    make_seg("right")

    # camera card
    card = tk.Frame(root, bg=CARD, highlightthickness=1, highlightbackground=OFF)
    card.pack(fill="x", padx=18, pady=(0, 10))
    ch = tk.Frame(card, bg=CARD)
    ch.pack(fill="x", padx=12, pady=(10, 6))
    cam_dot = label(ch, "●", 12, fg=MUTED, bg=CARD)
    cam_dot.pack(side="left")
    cam_title = label(ch, "CAMERA  starting…", 12, True, bg=CARD)
    cam_title.pack(side="left", padx=(6, 0))
    fusion_lbl = label(ch, "", 11, True, fg=MUTED, bg=CARD)
    fusion_lbl.pack(side="right", padx=(0, 14))
    cam_btn = label(ch, "switch camera ↻", 11, fg=BLUE, bg=CARD, cursor="hand2")
    cam_btn.pack(side="right")
    VW = W - 36
    vbox = tk.Frame(card, bg="black", width=VW, height=int(VW * 9 / 16))
    vbox.pack(padx=12, pady=(0, 8))
    vbox.pack_propagate(False)
    view = tk.Label(vbox, bg="black", fg=MUTED, font=(FONT, 12), text="starting camera…")
    view.pack(fill="both", expand=True)
    chips = tk.Frame(card, bg=CARD)
    chips.pack(fill="x", padx=12, pady=(0, 10))
    mchips = {}
    for key, lab, mid in MARKERS:
        c = tk.Label(chips, text="%s #%d" % (lab, mid), font=(FONT, 11, "bold"), fg=MUTED, bg=OFF, padx=10, pady=5)
        c.pack(side="left", padx=(0, 6))
        mchips[key] = c
    readout = label(chips, "", 11, fg=MUTED, bg=CARD)
    readout.pack(side="right")

    # fix chips
    fx = tk.Frame(root, bg=BG)
    fx.pack(fill="x", padx=18, pady=(0, 6))
    label(fx, "Options:", 11, fg=MUTED).pack(side="left", padx=(0, 8))
    fchips = {}

    def make_fix(key, text):
        c = tk.Label(fx, text=text, font=(FONT, 11), padx=10, pady=4, cursor="hand2")
        c.pack(side="left", padx=(0, 6))
        c.bind("<Button-1>", lambda e, k=key: toggle(k))
        fchips[key] = c
    make_fix("invert", "Invert bend")
    make_fix("rot180", "Flip shin 180°")
    make_fix("mirror", "Mirror camera")

    # fusion card: how much to trust the camera vs the accelerometers, and calibration
    tune_path = a.state + ".tune"
    tune = read_json(tune_path)
    tune = {"w": float(tune.get("w", W_DEFAULT)), "resp": float(tune.get("resp", RESP_DEFAULT)),
            "src": tune.get("src") or "fused", "t_write": 0.0}
    fc = tk.Frame(root, bg=CARD, highlightthickness=1, highlightbackground=OFF)
    fc.pack(fill="x", padx=18, pady=(4, 10))
    fh = tk.Frame(fc, bg=CARD)
    fh.pack(fill="x", padx=12, pady=(10, 4))
    label(fh, "FUSION", 12, True, bg=CARD).pack(side="left")
    cal_chip = label(fh, "", 10, True, fg="white", bg=OFF, padx=8, pady=2)
    cal_chip.pack(side="right")

    def set_tune(key, value):
        if tune[key] != value:
            tune[key], tune["t_write"] = value, time.monotonic()
            update_tune(tune_path, **{key: value})
        refresh_fusion()

    def on_slide(key, v):
        if abs(v - tune[key]) > 0.024:      # ignore the echo of stats_tick's own set()
            set_tune(key, round(v, 2))

    def slider(key, left, right):
        row = tk.Frame(fc, bg=CARD)
        row.pack(fill="x", padx=12, pady=2)
        title = label(row, "", 11, fg=TEXT, bg=CARD, width=20, anchor="w")
        title.pack(side="left")
        label(row, left, 10, fg=MUTED, bg=CARD).pack(side="left")
        sc = tk.Scale(row, from_=0.0, to=1.0, resolution=0.05, orient="horizontal", showvalue=False,
                      length=220, bg=CARD, troughcolor=OFF, activebackground=BLUE, highlightthickness=0,
                      bd=0, sliderrelief="flat", command=lambda v: on_slide(key, float(v)))
        sc.set(tune[key])
        sc.pack(side="left", padx=6)
        label(row, right, 10, fg=MUTED, bg=CARD).pack(side="left")
        return sc, title
    w_scale, w_title = slider("w", "sensors", "camera")
    r_scale, r_title = slider("resp", "smooth", "responsive")

    def chip(parent, text, cmd, side="left"):
        c = tk.Label(parent, text=text, font=(FONT, 11, "bold"), padx=10, pady=4, cursor="hand2", bg=OFF, fg=TEXT)
        c.pack(side=side, padx=(0, 6))
        c.bind("<Button-1>", lambda e: cmd())
        return c
    srow = tk.Frame(fc, bg=CARD)
    srow.pack(fill="x", padx=12, pady=(6, 4))
    label(srow, "Source", 11, fg=MUTED, bg=CARD, width=8, anchor="w").pack(side="left")
    src_chips = {k: chip(srow, text, lambda k=k: set_tune("src", k))
                 for k, text in (("fused", "Fused"), ("accel", "Sensors"), ("camera", "Camera"))}

    def calibrate(mode):
        extra = {}
        if mode == "anchor":
            try:
                extra["cal_anchor_deg"] = float(anchor_var.get())
            except ValueError:
                return cal_line.config(text="Type the measured knee angle (deg) first, then press Anchor.", fg=AMBER)
        update_tune(tune_path, bump="cal_seq", cal_mode=mode, **extra)
        cal_line.config(text="Starting calibration…", fg=AMBER)
    brow = tk.Frame(fc, bg=CARD)
    brow.pack(fill="x", padx=12, pady=(4, 4))
    chip(brow, "Calibrate", lambda: calibrate("hang")).config(bg=BLUE, fg="white")
    chip(brow, "Anchor", lambda: calibrate("anchor"))
    anchor_var = tk.StringVar()
    tk.Entry(brow, textvariable=anchor_var, width=4, font=(FONT, 11), bg=OFF, fg=TEXT, insertbackground=TEXT,
             relief="flat").pack(side="left")
    label(brow, "deg", 11, fg=MUTED, bg=CARD).pack(side="left", padx=(3, 10))
    chip(brow, "Bench check", lambda: calibrate("bench"))
    chip(brow, "Zero (straight)", lambda: update_tune(tune_path, bump="zero_seq"), side="right")
    cal_line = label(fc, "Calibrate: sit, let the shin hang, then follow the prompts (~35 s).", 11, fg=MUTED,
                     bg=CARD, wraplength=W - 70, justify="left", anchor="w")
    cal_line.pack(fill="x", padx=12, pady=(4, 10))

    def refresh_fusion():
        w_title.config(text="camera share %d%%" % round(100 * tune["w"]))
        r_title.config(text="responsiveness %.2f" % tune["resp"])
        for k, c in src_chips.items():
            on = tune["src"] == k
            c.config(bg=BLUE if on else OFF, fg="white" if on else MUTED)
    refresh_fusion()
    label(root, "Zero only when the leg is truly straight; Calibrate measures your extension too.", 10,
          fg=MUTED).pack(padx=18, pady=(0, 14), anchor="w")

    # ---- behaviour
    def refresh_widgets():
        for n, b in segs.items():
            on = st["leg"] == n
            b.config(bg=GREEN if on else CARD, fg="white" if on else MUTED)
        for k, c in fchips.items():
            on = st[k]
            c.config(bg=BLUE if on else OFF, fg="white" if on else MUTED)

    def push():
        write_state(a.state, st["leg"], st["invert"], st["rot180"])
        cfg["last_leg"] = st["leg"]
        cfg[st["leg"]] = {"invert": st["invert"], "rot180": st["rot180"]}
        cfg["mirror_cam"] = st["mirror"]
        save_cfg(cfg)
        if tracker is not None:
            tracker.flip_x = st["mirror"]
        refresh_widgets()

    def set_leg(name):
        st["leg"] = name
        st["invert"], st["rot180"] = cfg[name]["invert"], cfg[name]["rot180"]
        push()

    def toggle(k):
        st[k] = not st[k]
        push()

    # ---- camera
    tracker = None
    cam = {"idx": a.cam, "photo": None, "next_open": 0.0, "err": None, "fail": 0, "seq": 0}
    use_cam = not a.no_camera

    def open_camera():
        nonlocal tracker
        if tracker is not None:
            try:
                tracker.stop()
            except Exception:
                pass
            tracker = None
        try:
            from camera_origin import CameraTracker   # repo root, unmodified
            t = CameraTracker(cam_index=cam["idx"], marker_cm=a.marker_cm, flip_x=st["mirror"])
        except Exception as e:                          # no cv2 / aruco
            cam["err"] = "OpenCV with aruco not available (%s)" % e
            return
        if t.start(threaded=False):
            tracker = t
            cam["err"] = None
            cam["fail"] = 0
        else:
            cam["err"] = ("Can't open camera %d. Allow Camera access for Terminal in System Settings > "
                          "Privacy & Security > Camera, and run this from Terminal.app." % cam["idx"])

    def switch_cam(_e=None):
        cam["idx"] = (cam["idx"] + 1) % 4
        cam["next_open"] = 0.0
        open_camera()

    cam_btn.bind("<Button-1>", switch_cam)

    def publish_cam(fix):
        """Hand the camera knee angle to leg_filter.py (it fuses it with the accelerometers)."""
        cam["seq"] += 1             # t_cap: capture time on the monotonic clock, so late frames line up
        d = {"knee": None if fix is None else fix.camera_knee_deg,
             "conf": 0.0 if fix is None else float(fix.confidence), "wall": time.time(),
             "t_cap": fix.t if fix is not None and fix.t else time.monotonic(), "seq": cam["seq"]}
        try:
            with open(a.state + ".cam.tmp", "w") as f:
                json.dump(d, f)
            os.replace(a.state + ".cam.tmp", a.state + ".cam")
        except OSError:
            pass

    def set_chip(key, on):
        mchips[key].config(bg=GREEN if on else OFF, fg="white" if on else MUTED)

    def cam_tick():
        now = time.monotonic()
        if not use_cam:
            cam_title.config(text="CAMERA  off")
            view.config(text="camera disabled (--no-camera)", image="")
            return
        if tracker is None:
            if now >= cam["next_open"]:
                open_camera()
                cam["next_open"] = now + 4.0
            if tracker is None:
                cam_dot.config(fg=RED)
                cam_title.config(text="CAMERA  not available")
                view.config(image="", text=cam["err"] or "starting camera…", wraplength=W - 80)
                for k in mchips:
                    set_chip(k, False)
                root.after(500, cam_tick)
                return
        frame, fix = tracker.grab_once()
        if frame is None:
            cam["fail"] += 1
            if cam["fail"] > 60:
                cam["err"] = "Camera opened but sends no frames (permission or busy). Retrying…"
                open_camera()
                cam["fail"] = 0
            root.after(30, cam_tick)
            return
        cam["fail"] = 0
        publish_cam(fix)
        img = render_camera(tracker, frame, fix, VW)
        photo = tk.PhotoImage(data=ppm_bytes(img))
        view.config(image=photo, text="")
        cam["photo"] = photo      # keep a reference
        seen = set(tracker.last_detections.keys())
        for key, _lab, mid in MARKERS:
            set_chip(key, tracker.ids.get(key) in seen)
        n = sum(1 for key, _l, _m in MARKERS if tracker.ids.get(key) in seen)
        cam_dot.config(fg=GREEN if n == 3 else (AMBER if n else RED))
        cam_title.config(text="CAMERA %d  live · %d of 3 markers" % (cam["idx"], n))
        bits = []
        if fix is not None and fix.camera_knee_deg is not None:
            bits.append("knee %.0f°" % abs(fix.camera_knee_deg))
        if fix is not None and fix.px_per_cm:
            bits.append("%.1f px/cm" % fix.px_per_cm)
        readout.config(text="   ".join(bits))
        root.after(15, cam_tick)

    # ---- leg_filter readout (STATE.json.stats, 2 Hz)
    def stats_tick():
        d = read_json(a.state + ".stats")
        if time.time() - d.get("wall", 0) > 1.5:
            d = {}
        live = d.get("rate", 0) > 5
        stream_txt.config(text="sensors %d Hz" % round(d["rate"]) if live else "no sensor data")
        stream_dot.config(fg=GREEN if live else RED)
        ftxt, fcol = {"locked": ("camera fused", GREEN), "waiting": ("camera: waiting", AMBER),
                      "off": ("camera not used", MUTED)}.get(d.get("fusion"), ("", MUTED))
        fusion_lbl.config(text=ftxt, fg=fcol)
        cal = d.get("cal") or {}
        cal_chip.config(text="CALIBRATED" if cal.get("calibrated") else "NOT CALIBRATED",
                        bg=GREEN if cal.get("calibrated") else AMBER)
        if cal.get("state") not in (None, "idle"):
            busy = cal["state"] not in ("done", "failed")
            pct = "  (%d%%)" % round(100 * cal.get("progress", 0)) if busy else ""
            cal_line.config(text=(cal.get("prompt") or "") + pct,
                            fg=AMBER if busy else GREEN if cal["state"] == "done" else RED)
        if d and time.monotonic() - tune["t_write"] > 1.0:   # follow the viewer's keys too
            for key in ("w", "resp", "src"):
                if key in d:
                    tune[key] = d[key]
            w_scale.set(tune["w"])
            r_scale.set(tune["resp"])
            refresh_fusion()
        root.after(500, stats_tick)

    push()
    root.after(200, cam_tick)
    root.after(300, stats_tick)

    def on_close():
        try:
            if tracker is not None:
                tracker.stop()
        finally:
            root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    sys.path.insert(0, REPO)
    root.mainloop()


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    main()
