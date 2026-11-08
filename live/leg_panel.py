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
Per-leg chip settings and the last leg are remembered in live/leg_config.json.
(Planned, not built: Start/Stop, rep counter, more stats.)
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
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
    write_state(path, leg, cfg[leg]["invert"], cfg[leg]["rot180"], cfg["assist"])
    return leg


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
          "assist": cfg["assist"]}

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
    make_fix("assist", "Camera assist")
    make_fix("invert", "Invert bend")
    make_fix("rot180", "Flip shin 180°")
    make_fix("mirror", "Mirror camera")
    label(root, "After changing the leg, hold it straight and press Z in the 3D window.", 10, fg=MUTED
          ).pack(padx=18, pady=(2, 14), anchor="w")

    # ---- behaviour
    def refresh_widgets():
        for n, b in segs.items():
            on = st["leg"] == n
            b.config(bg=GREEN if on else CARD, fg="white" if on else MUTED)
        for k, c in fchips.items():
            on = st[k]
            c.config(bg=BLUE if on else OFF, fg="white" if on else MUTED)

    def push():
        write_state(a.state, st["leg"], st["invert"], st["rot180"], st["assist"])
        cfg["last_leg"] = st["leg"]
        cfg[st["leg"]] = {"invert": st["invert"], "rot180": st["rot180"]}
        cfg["mirror_cam"] = st["mirror"]
        cfg["assist"] = st["assist"]
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
    cam = {"idx": a.cam, "photo": None, "next_open": 0.0, "err": None, "fail": 0}
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
        d = {"knee": None if fix is None else fix.camera_knee_deg,
             "conf": 0.0 if fix is None else float(fix.confidence), "wall": time.time()}
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

    # ---- sensor stream indicator
    def stats_tick():
        txt, col = "no sensor data", RED
        try:
            with open(a.state + ".stats") as f:
                d = json.load(f)
            if time.time() - d.get("wall", 0) < 1.5 and d.get("rate", 0) > 5:
                txt, col = "sensors %d Hz" % round(d["rate"]), GREEN
        except (OSError, ValueError):
            pass
        stream_txt.config(text=txt)
        stream_dot.config(fg=col)
        ftxt, fcol = "assist off", MUTED
        if st["assist"]:
            ftxt, fcol = "assist: waiting for camera", AMBER
            try:
                with open(a.state + ".stats") as f:
                    d = json.load(f)
                if time.time() - d.get("wall", 0) < 1.5 and d.get("fusion") in ("locked", "holding"):
                    off = d.get("offset_deg")
                    ftxt = "assist: %s  %+.0f\u00b0" % ("LOCKED" if d["fusion"] == "locked" else "holding", off or 0)
                    fcol = GREEN if d["fusion"] == "locked" else AMBER
            except (OSError, ValueError):
                pass
        fusion_lbl.config(text=ftxt, fg=fcol)
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
