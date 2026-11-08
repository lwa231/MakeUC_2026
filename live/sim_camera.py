#!/usr/bin/env python3
"""
sim_camera.py -- synthetic-frame simulator + evaluator for camera_origin.CameraTracker.

Renders the 3 ArUco markers of the knee brace (knee id 1 at the hinge, hip-end id 0
at the free end of the thigh arm, ankle-end id 2 at the free end of the shank arm)
for a seated leg-extension pose, through a pinhole camera with yaw/pitch, radial
distortion, per-marker depth/tilt, then degrades the image (defocus, motion blur,
noise, gamma/contrast, JPEG, occlusion) and measures what the tracker reads.

GROUND-TRUTH CONVENTION (camera_origin.py docstring, verified in --selftest)
  leg frame: X = forward (toes), Y = up, knee marker centre = origin.
  thigh arm points knee -> hip along -u(s), u(a) = (cos a, -sin a); s = thigh sway
  (seated: ~horizontal, hip behind the knee).  shank arm: knee -> ankle along u(s+theta).
  theta = 0 straight (shank continues the thigh, pointing forward), 90 = hanging.
  heading phi_thigh = s, phi_shank = s + theta  ->  camera_knee_deg = theta  (exactly,
  sway cancels).  Truth is the TRUE 3D leg-plane angle; yaw/pitch of the view make
  the image-plane angle differ (that is a real error source and is measured).
  flip_x=True: camera on the other side, forward = image-left; the markers are still
  seen un-mirrored (the camera views the side they are stuck on).
  Markers: "down" axis (TL->BL) points proximal->distal along the arm, so the printed
  marker is rotated 90 deg when the arm is horizontal.

Usage:
  python3 live/sim_camera.py                 # full sweep, ~2 min
  python3 live/sim_camera.py --quick         # ~20 s
  python3 live/sim_camera.py --selftest      # convention / zero-error checks
  python3 live/sim_camera.py --factory mymod:make   # compare a modified tracker
        (factory(flip_x=False) -> object with process_frame(img, t) -> CameraFix)
  python3 live/sim_camera.py --dump DIR      # save a few example frames

From other code:
  import sim_camera as sc
  m0 = sc.evaluate(sc.original_factory, sc.make_scenarios(quick=True))
  m1 = sc.evaluate(my_factory,           sc.make_scenarios(quick=True))
  sc.print_report(m0); sc.print_compare(m0, m1)
  sc.evaluate({"orig": f0, "new": f1}, scen)   # renders once, evaluates both
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

IDS = {"knee": 1, "hip": 0, "ankle": 2}
MARKER_CM = 5.0
INNER, BORDER = 300, 50
FULL_CM = MARKER_CM * (INNER + 2 * BORDER) / INNER      # whole PNG incl. white border


def original_factory(flip_x: bool = False):
    from camera_origin import CameraTracker
    return CameraTracker(marker_cm=MARKER_CM, flip_x=flip_x)


def factory_from_file(path):
    """Factory for a CameraTracker defined in another file (e.g. a saved copy of the original)."""
    import importlib.util
    name = "_cam_" + str(abs(hash(path)))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return lambda flip_x=False: mod.CameraTracker(marker_cm=MARKER_CM, flip_x=flip_x)


# --------------------------------------------------------------------------
# parameter containers
# --------------------------------------------------------------------------
@dataclass
class Cam:
    w: int = 1280
    h: int = 720
    hfov: float = 70.0          # deg
    dist: float = 1.0           # m, camera -> leg centre
    yaw: float = 0.0            # deg, view rotation about vertical
    pitch: float = 0.0          # deg, about horizontal
    k1: float = 0.0             # radial distortion (negative = barrel)
    k2: float = 0.0
    flip_x: bool = False        # camera on the other side

    @property
    def f(self):
        return (self.w / 2.0) / math.tan(math.radians(self.hfov) / 2.0)


@dataclass
class Pose:
    theta: float = 45.0         # true knee angle, deg (0 straight, 90 hanging)
    omega: float = 0.0          # shank angular speed, deg/s
    sway: float = 0.0           # thigh heading phi_thigh, deg (0 = horizontal)
    thigh_cm: float = 40.0
    shank_cm: float = 40.0
    dz: Dict[str, float] = field(default_factory=lambda: {"knee": 0., "hip": 0., "ankle": 0.})
    tilt: Dict[str, tuple] = field(default_factory=lambda: {k: (0., 0.) for k in IDS})  # (axis_deg, angle_deg)
    occl: Optional[tuple] = None   # (marker name, covered fraction 0..1)
    shift: tuple = (0.5, 0.5)      # where in the free margin the leg sits (0..1 each)


@dataclass
class Degr:
    light: str = "good"
    gain: float = 1.0
    contrast: float = 1.0
    gamma: float = 1.0
    noise: float = 2.0          # sigma, 0..255 scale
    defocus: float = 0.7        # gaussian sigma px
    exposure_ms: float = 12.0
    jpeg_q: int = 85
    bg_level: float = 130.0
    vignette: float = 0.1
    tint: tuple = (1.0, 1.0, 1.0)


def sample_degr(light: str, rng) -> Degr:
    u = rng.uniform
    if light == "good":
        return Degr("good", gain=u(0.85, 1.15), contrast=u(0.9, 1.1), gamma=u(0.9, 1.1), noise=u(1.0, 3.0),
                    defocus=u(0.4, 1.0), exposure_ms=u(8, 20), jpeg_q=int(u(75, 90)), bg_level=u(110, 160),
                    vignette=u(0.0, 0.2), tint=tuple(u(0.93, 1.07, 3)))
    return Degr("dim", gain=u(0.40, 0.65), contrast=u(0.7, 0.9), gamma=u(1.1, 1.4), noise=u(5.0, 9.0),
                defocus=u(0.7, 1.6), exposure_ms=u(25, 33), jpeg_q=int(u(60, 80)), bg_level=u(100, 150),
                vignette=u(0.2, 0.45), tint=tuple(u(0.85, 1.1, 3)))


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def _u(a_deg):
    a = math.radians(a_deg)
    return np.array([math.cos(a), -math.sin(a)])


def leg_points(pose: Pose):
    """Leg-frame (X fwd, Y up) centres of the three markers + their 'down' (proximal->distal) unit vectors."""
    s = pose.sway
    ut = _u(s)                                  # hip -> knee
    us = _u(s + pose.theta)                     # knee -> ankle
    c = {"knee": np.zeros(2), "hip": -pose.thigh_cm * ut, "ankle": pose.shank_cm * us}
    d = {"knee": ut, "hip": ut, "ankle": us}
    return c, d


def _rodrigues(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def build_scene(pose: Pose, cam: Cam):
    """Return dict: per-marker pixel corners (TL,TR,BR,BL), centre px, depth (cm), plus arm quads and
    projection helpers.  All lengths in cm; camera-ideal frame: x right, y down, z away."""
    sx = -1.0 if cam.flip_x else 1.0
    cxy, dxy = leg_points(pose)
    P = {}      # 3D cam-ideal marker data
    for n in IDS:
        cen = np.array([sx * cxy[n][0], -cxy[n][1], -pose.dz[n]])         # dz>0 = toward camera
        d2 = np.array([sx * dxy[n][0], -dxy[n][1]])                       # image-space down
        r2 = np.array([d2[1], -d2[0]])                                    # image-space right (un-mirrored print)
        r3, d3 = np.array([r2[0], r2[1], 0.0]), np.array([d2[0], d2[1], 0.0])
        ax_deg, tilt_deg = pose.tilt[n]
        if abs(tilt_deg) > 1e-6:
            axis = math.cos(math.radians(ax_deg)) * r3 + math.sin(math.radians(ax_deg)) * d3
            Rm = _rodrigues(axis, math.radians(tilt_deg))
            r3, d3 = Rm @ r3, Rm @ d3
        h = FULL_CM / 2.0
        corners = np.array([cen - h * r3 - h * d3, cen + h * r3 - h * d3,
                            cen + h * r3 + h * d3, cen - h * r3 + h * d3])     # TL TR BR BL (full PNG)
        P[n] = (cen, corners)
    # view rotation about the leg centre (mean of marker centres in-plane)
    C = np.mean([P[n][0] for n in IDS], axis=0) * np.array([1, 1, 0])
    Ry = _rodrigues(np.array([0, 1.0, 0]), math.radians(cam.yaw))
    Rx = _rodrigues(np.array([1.0, 0, 0]), math.radians(cam.pitch))
    R = Ry @ Rx
    dcm = cam.dist * 100.0

    def tf(p):
        return (R @ (np.asarray(p, float) - C).T).T + np.array([0, 0, dcm])

    def proj(q):
        q = np.atleast_2d(q)
        x, y = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
        r2_ = x * x + y * y
        s = 1 + cam.k1 * r2_ + cam.k2 * r2_ * r2_
        return np.stack([cam.f * x * s, cam.f * y * s], 1)

    out = {}
    allpts = []
    for n in IDS:
        cen, corners = P[n]
        out[n] = {"c_cam": tf(cen), "k_cam": tf(corners)}
        out[n]["c_uv"] = proj(out[n]["c_cam"])[0]
        out[n]["k_uv"] = proj(out[n]["k_cam"])
        allpts.append(out[n]["k_uv"])
    allpts = np.vstack(allpts)
    lo, hi = allpts.min(0), allpts.max(0)
    W, H = cam.w, cam.h
    margin = np.array([W, H]) - (hi - lo) - 20.0
    margin = np.maximum(margin, 0.0)
    centre_shift = np.array([W / 2.0, H / 2.0]) - (lo + hi) / 2.0
    off = (np.array(pose.shift) - 0.5) * margin * 0.8
    sh = centre_shift + off
    for n in IDS:
        out[n]["c_uv"] = out[n]["c_uv"] + sh
        out[n]["k_uv"] = out[n]["k_uv"] + sh
    # arm quads (3 cm wide strips on the leg surface) knee->hip, knee->ankle
    arms = {}
    for name, end in (("thigh", "hip"), ("shank", "ankle")):
        a = P["knee"][0].copy()
        b = P[end][0].copy()
        v = b - a
        nrm = np.array([-v[1], v[0], 0.0])
        nrm /= (np.linalg.norm(nrm) + 1e-9)
        w = 1.5
        quad = np.array([a - w * nrm, b - w * nrm, b + w * nrm, a + w * nrm])
        arms[name] = proj(tf(quad)) + sh
    out["arms"] = arms
    out["f"] = cam.f
    out["shift"] = sh
    return out


def truth_for(pose: Pose, cam: Cam, scene) -> dict:
    """Ground truth dict."""
    sx = -1.0 if cam.flip_x else 1.0
    kc = scene["knee"]["c_uv"]

    def rel(n):
        d = scene[n]["c_uv"] - kc
        return (sx * d[0], -d[1])
    hr, ar = rel("hip"), rel("ankle")
    th = math.degrees(math.atan2(hr[1], -hr[0]))
    sh = math.degrees(math.atan2(-ar[1], ar[0]))
    proj_deg = (sh - th + 180.0) % 360.0 - 180.0         # what a perfect centre detector would read
    side = {n: float(np.mean([np.linalg.norm(scene[n]["k_uv"][i] - scene[n]["k_uv"][(i + 1) % 4])
                              for i in range(4)])) * INNER / (INNER + 2 * BORDER) for n in IDS}
    return {"theta": pose.theta, "proj_deg": proj_deg, "omega": pose.omega,
            "marker_px": side, "ppc": side["knee"] / MARKER_CM, "occl": pose.occl,
            "phi_thigh": pose.sway, "phi_shank": pose.sway + pose.theta,
            "uv": {n: scene[n]["c_uv"].copy() for n in IDS}}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
_MARKER_CACHE: Dict[tuple, np.ndarray] = {}
_DICT = None


def _marker_img(mid: int, n: int) -> np.ndarray:
    """Marker PNG as make_markers writes it (300 px + 50 px white border), as float 0..1, resized (area) to n px."""
    global _DICT
    key = (mid, n)
    im = _MARKER_CACHE.get(key)
    if im is None:
        if _DICT is None:
            _DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        base = cv2.aruco.generateImageMarker(_DICT, mid, INNER)
        base = cv2.copyMakeBorder(base, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, value=255)
        if n != base.shape[0]:
            base = cv2.resize(base, (n, n), interpolation=cv2.INTER_AREA)
        im = base.astype(np.float32) / 255.0
        _MARKER_CACHE[key] = im
    return im


def _line_kernel(L: float, ang: float) -> np.ndarray:
    n = int(math.ceil(L)) | 1
    n = max(n, 3)
    k = np.zeros((n, n), np.float32)
    c = n // 2
    dx, dy = math.cos(ang) * L / 2, math.sin(ang) * L / 2
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    s = k.sum()
    return k / s if s > 0 else np.pad(np.ones((1, 1), np.float32), 1)[1:2, 1:2]


def _background(w, h, level, rng):
    small = rng.random((h // 40 + 3, w // 40 + 3)).astype(np.float32)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * 50 - 25
    mid = rng.random((h // 10 + 3, w // 10 + 3)).astype(np.float32)
    bg += cv2.resize(mid, (w, h), interpolation=cv2.INTER_CUBIC) * 14 - 7
    bg += level
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(2, 6))):                      # furniture-ish rectangles
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(w // 10, w // 3)), y0 + int(rng.integers(h // 10, h // 3))
        cv2.rectangle(bg, (x0, y0), (x1, y1), int(np.clip(level + rng.uniform(-45, 45), 10, 245)), -1)
    return bg


def render(pose: Pose, cam: Cam, degr: Degr, rng, scene=None):
    """Render one BGR uint8 frame. Returns (img, truth)."""
    scene = scene or build_scene(pose, cam)
    w, h = cam.w, cam.h
    bg = _background(w, h, degr.bg_level, rng)
    # brace arms (dark grey strips) under the markers
    for name, shade in (("thigh", 55), ("shank", 55)):
        cv2.fillConvexPoly(bg, np.round(scene["arms"][name]).astype(np.int32), shade, cv2.LINE_AA)
    img = bg.astype(np.float32) / 255.0
    ink, paper = 0.07, 0.88                                        # printed black / paper reflectance

    kc = scene["knee"]["c_uv"]
    omega_rad = math.radians(pose.omega)
    for n in ("knee", "hip", "ankle"):
        kuv = scene[n]["k_uv"].astype(np.float64)
        sides = [np.linalg.norm(kuv[i] - kuv[(i + 1) % 4]) for i in range(4)]
        side = float(np.mean(sides))
        # blur: only the shank-end marker moves (arc = omega * exposure * radius)
        depth = scene[n]["c_cam"][2]
        radius_cm = pose.shank_cm if n == "ankle" else 0.0
        blur_px = omega_rad * degr.exposure_ms * 1e-3 * radius_cm * cam.f / depth if n == "ankle" else 0.0
        pad = int(blur_px) + 6
        x0, y0 = np.floor(kuv.min(0)).astype(int) - pad
        x1, y1 = np.ceil(kuv.max(0)).astype(int) + pad
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        npx = int(np.clip(round(side), 8, 400))                    # side of the whole PNG (with border) in px
        m = _marker_img(IDS[n], npx)
        src = np.float32([[0, 0], [npx, 0], [npx, npx], [0, npx]])
        dst = (kuv - np.array([x0, y0])).astype(np.float32)
        Hm = cv2.getPerspectiveTransform(src, dst)
        two = np.dstack([m * (paper - ink) + ink, np.ones_like(m)]).astype(np.float32)
        wp = cv2.warpPerspective(two, Hm, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0))
        if blur_px > 1.2:
            d = scene[n]["c_uv"] - kc
            ang = math.atan2(d[0], -d[1])                        # perpendicular to the radius vector
            wp = cv2.filter2D(wp, -1, _line_kernel(blur_px, ang), borderType=cv2.BORDER_CONSTANT)
        a = wp[..., 1:2]
        roi = img[y0:y1, x0:x1]
        roi[:] = roi * (1 - a[..., 0]) + wp[..., 0]
    # occlusion: skin-coloured patch over (part of) one marker
    if pose.occl is not None:
        n, frac = pose.occl
        kuv = scene[n]["k_uv"]
        lo, hi = kuv.min(0), kuv.max(0)
        x_end = lo[0] + (hi[0] - lo[0]) * frac
        cv2.rectangle(img, (int(lo[0]) - 3, int(lo[1]) - 3), (int(x_end), int(hi[1]) + 3), 0.55, -1)

    # illumination / sensor
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    img *= (1.0 - degr.vignette * rr / 2.0)
    if degr.defocus > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), degr.defocus)
    img = np.clip(img * degr.gain, 0, 1) ** degr.gamma
    img = img * degr.contrast + (1 - degr.contrast) * 0.12
    sig = degr.noise / 255.0
    img = img + rng.normal(0, 1, img.shape).astype(np.float32) * (sig * (0.6 + 0.8 * np.sqrt(np.clip(img, 0, 1))))
    img = np.clip(img, 0, 1)
    bgr = np.dstack([img * degr.tint[2], img * degr.tint[1], img * degr.tint[0]])
    bgr = np.clip(bgr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(degr.jpeg_q)])
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    truth = truth_for(pose, cam, scene)
    truth["blur_px_ankle"] = abs(omega_rad * degr.exposure_ms * 1e-3 * pose.shank_cm * cam.f /
                              scene["ankle"]["c_cam"][2])
    truth["exposure_ms"] = degr.exposure_ms
    truth["jpeg_q"] = degr.jpeg_q
    return bgr, truth


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------
@dataclass
class Scenario:
    res: tuple = (1280, 720)
    dist: float = 1.0
    view: float = 0.0           # |yaw| = view, |pitch| = view/2, random signs
    light: str = "good"
    hfov: float = 70.0
    k1: float = -0.05
    flip_x: bool = False
    occl_prob: float = 0.0
    n: int = 40
    speeds: tuple = (0, 40, 90, 140, 200, 250)
    theta_range: tuple = (0.0, 100.0)
    name: str = ""
    seq: bool = False           # one fixed camera/pose-geometry, leg doing reps over n frames (use fresh=False)
    fps: float = 30.0
    rep_period: float = 2.5     # s per flexion/extension cycle
    seq_range: tuple = (5.0, 95.0)
    mount: str = "leg"          # "leg": markers stuck on a curved leg (tilt ~3 deg sd, depth +-3.5 cm);
                                # "brace": rigid brace (knee cap + flat bars): tilt <= 1.5 deg, depth <= 1 cm

    def key(self):
        return "%dx%d d%.1f v%d %s" % (self.res[0], self.res[1], self.dist, self.view, self.light)


def make_scenarios(quick: bool = False, n: Optional[int] = None, mount: str = "leg") -> List[Scenario]:
    n = n or (10 if quick else 48)
    sc = []
    for res, dist, view, light in itertools.product([(640, 480), (1280, 720)], [0.7, 1.0, 1.5],
                                                    [0, 10, 20], ["good", "dim"]):
        sc.append(Scenario(res=res, dist=dist, view=view, light=light, n=n, mount=mount))
    sc.append(Scenario(res=(1280, 720), dist=1.0, view=10, light="good", occl_prob=1.0, n=max(6, n // 2),
                       name="occlusion", mount=mount))
    return sc


def sample_pose(s: Scenario, k: int, rng) -> Pose:
    u = rng.uniform
    theta = u(*s.theta_range)
    if s.mount == "brace":
        tilt = {nm: (u(0, 360), float(np.clip(rng.normal(0, 0.7), -1.5, 1.5))) for nm in IDS}
        dz = {nm: u(-1.0, 1.0) for nm in IDS}
    else:
        tilt = {nm: (u(0, 360), float(np.clip(rng.normal(0, 3.0), -7, 7))) for nm in IDS}
        dz = {nm: u(-3.5, 3.5) for nm in IDS}
    occl = None
    if s.occl_prob and rng.random() < s.occl_prob:
        occl = (str(rng.choice(["hip", "ankle", "knee"])), float(u(0.7, 1.0)))
    return Pose(theta=theta, omega=float(s.speeds[k % len(s.speeds)]) * float(rng.choice([-1, 1])) * 1.0,
                sway=float(np.clip(rng.normal(0, 2.0), -5, 5)),
                thigh_cm=u(38, 42), shank_cm=u(38, 42), dz=dz, tilt=tilt, occl=occl,
                shift=(u(0, 1), u(0, 1)))


def sample_cam(s: Scenario, rng) -> Cam:
    sg = lambda: float(rng.choice([-1, 1]))
    yaw = s.view * sg() * (rng.uniform(0.5, 1.0) if s.view else 0)
    pitch = s.view * 0.5 * sg() * (rng.uniform(0.5, 1.0) if s.view else 0)
    return Cam(w=s.res[0], h=s.res[1], hfov=s.hfov, dist=s.dist * rng.uniform(0.95, 1.05),
               yaw=yaw, pitch=pitch, k1=s.k1, flip_x=s.flip_x)


def make_sequences(frames: int = 150, mount: str = "leg") -> List[Scenario]:
    """Static-camera sequences (leg doing reps) for testing temporal behaviour; evaluate(..., fresh=False)."""
    out = []
    for res, dist, view, light in itertools.product([(1280, 720), (640, 480)], [1.0, 1.5], [0, 10, 20], ["good", "dim"]):
        if light == "dim" and (res[0] == 640 or dist > 1.0):
            continue
        out.append(Scenario(res=res, dist=dist, view=view, light=light, n=frames, seq=True,
                            name="seq", mount=mount))
    return out


def gen_frames(scenarios, seed=1234):
    """Deterministic generator of (scenario_index, k, img, truth).  truth["t"] = simulated capture time (s)
    for sequence scenarios (frame k at k/fps), else None."""
    for si, s in enumerate(scenarios):
        if s.seq:
            brng = np.random.default_rng([seed, si, 99999])
            cam = sample_cam(s, brng)
            pose0 = sample_pose(replace(s, speeds=(0,)), 0, brng)
        for k in range(s.n):
            rng = np.random.default_rng([seed, si, k])
            if s.seq:
                t = k / s.fps
                lo, hi = s.seq_range
                ph = 2 * math.pi * t / s.rep_period
                theta = lo + (hi - lo) * (1 - math.cos(ph)) / 2
                om = (hi - lo) / 2 * math.sin(ph) * 2 * math.pi / s.rep_period
                pose = replace(pose0, theta=theta, omega=om)
            else:
                t = None
                cam = sample_cam(s, rng)
                pose = sample_pose(s, k, rng)
            degr = sample_degr(s.light, rng)
            img, truth = render(pose, cam, degr, rng)
            truth["omega"] = abs(pose.omega)
            truth["t"] = t
            yield si, k, img, truth


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def cache_frames(scenarios, path, seed=1234):
    """Render once and store losslessly (PNG) so several tracker variants can be evaluated quickly."""
    import pickle
    out = []
    for si, k, img, truth in gen_frames(scenarios, seed):
        out.append((si, k, cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes(), truth))
    with open(path, "wb") as f:
        pickle.dump(out, f)


def load_frames(path):
    """Generator over a cache_frames() file, yielding (si, k, img, truth)."""
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    for si, k, png, truth in data:
        yield si, k, cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), truth


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _det_flags(tr, fix):
    ld = getattr(tr, "last_detections", None)
    if isinstance(ld, dict):
        return tuple(IDS[n] in ld for n in ("knee", "hip", "ankle"))
    return (bool(fix.knee_visible), fix.hip_rel_cm is not None, fix.ankle_rel_cm is not None)


FIELDS = ["si", "k", "knee", "hip", "ankle", "fix", "conf", "err", "err_proj", "ms",
          "theta", "omega", "occl", "ppc", "blur", "res", "dist", "view", "light", "dbg"]


def evaluate(tracker_factory, scenarios, seed=1234, verbose=False, fresh=True, frames=None):
    """Run tracker(s) over the synthetic scenarios.  tracker_factory: callable(flip_x=False) -> tracker,
    or a dict name -> factory (frames rendered once).  fresh=True builds a new tracker per frame (frames of the
    sweep are independent random cameras, so no temporal state may leak between them; use evaluate_sequence for
    temporal behaviour).  Returns metrics dict:
      {"rec": {field: np.array}, "overall": {...}, "groups": {...}} (or dict of those per name)."""
    facs = tracker_factory if isinstance(tracker_factory, dict) else {"_": tracker_factory}
    trackers = {nm: {} for nm in facs}
    recs = {nm: {f: [] for f in FIELDS} for nm in facs}
    t_start = time.time()
    for si, k, img, tr_ in (frames if frames is not None else gen_frames(scenarios, seed)):
        s = scenarios[si]
        for nm, fac in facs.items():
            tkey = si if s.seq else s.flip_x           # sequences: one tracker per sequence
            tr = None if (fresh and not s.seq) else trackers[nm].get(tkey)
            if tr is None:
                tr = trackers[nm][tkey] = fac(flip_x=s.flip_x) if _accepts_flip(fac) else fac()
            t0 = time.perf_counter()
            fix = tr.process_frame(img, tr_["t"] if tr_["t"] is not None else time.monotonic())
            ms = (time.perf_counter() - t0) * 1e3
            kf, hf, af = _det_flags(tr, fix)
            has = fix.camera_knee_deg is not None
            r = recs[nm]
            r["si"].append(si); r["k"].append(k)
            r["knee"].append(kf); r["hip"].append(hf); r["ankle"].append(af)
            r["fix"].append(has)
            r["conf"].append(float(fix.confidence))
            r["err"].append(wrap(fix.camera_knee_deg - tr_["theta"]) if has else np.nan)
            r["err_proj"].append(wrap(fix.camera_knee_deg - tr_["proj_deg"]) if has else np.nan)
            r["ms"].append(ms)
            r["theta"].append(tr_["theta"]); r["omega"].append(tr_["omega"])
            r["occl"].append(tr_["occl"] is not None)
            r["ppc"].append(tr_["ppc"]); r["blur"].append(tr_["blur_px_ankle"])
            r["dbg"].append(dict(getattr(tr, "debug", None) or {}))
            r["res"].append(s.res[0]); r["dist"].append(s.dist); r["view"].append(s.view)
            r["light"].append(s.light)
        if verbose and k == 0:
            print("  scenario %d/%d %s (%.0fs)" % (si + 1, len(scenarios), s.key(), time.time() - t_start),
                  file=sys.stderr, flush=True)
    out = {}
    for nm in facs:
        rec = {f: np.array(v) for f, v in recs[nm].items()}
        out[nm] = {"rec": rec, "overall": summarize(rec, np.ones(len(rec["si"]), bool))}
    return out["_"] if not isinstance(tracker_factory, dict) else out


def _accepts_flip(fac):
    try:
        import inspect
        sig = inspect.signature(fac)
        return "flip_x" in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    except (TypeError, ValueError):
        return False


def summarize(rec, mask) -> dict:
    m = mask
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    clean = m & ~rec["occl"]                       # frames where nothing is occluded
    nc = max(int(clean.sum()), 1)
    fix = rec["fix"].astype(bool)
    err = rec["err"]
    ok = clean & fix
    e = err[ok]
    conf = rec["conf"][clean & fix]
    d = {"n": n,
         "knee": rec["knee"][clean].sum() / nc, "hip": rec["hip"][clean].sum() / nc,
         "ankle": rec["ankle"][clean].sum() / nc, "fix": fix[clean].sum() / nc,
         "lowconf": float((conf < 0.5).mean()) if conf.size else float("nan"),
         "usable": float((fix & (rec["conf"] >= 0.5))[clean].sum() / nc),
         "conf_mean": float(conf.mean()) if conf.size else float("nan"),
         "conf_p5": float(np.percentile(conf, 5)) if conf.size else float("nan"),
         "bias": float(e.mean()) if e.size else float("nan"),
         "sd": float(e.std()) if e.size else float("nan"),
         "p95": float(np.percentile(np.abs(e), 95)) if e.size else float("nan"),
         "max": float(np.abs(e).max()) if e.size else float("nan"),
         "rmse_proj": float(np.sqrt(np.mean(rec["err_proj"][ok] ** 2))) if e.size else float("nan"),
         "ms": float(rec["ms"][m].mean()), "ms_p95": float(np.percentile(rec["ms"][m], 95)),
         # fraction of usable (conf>=0.5) fixes with |err| > 5 deg: the harmful ones
         "bad5": float((np.abs(err[clean & fix & (rec["conf"] >= 0.5)]) > 5).sum() / max(1, (clean & fix & (rec["conf"] >= 0.5)).sum())),
         }
    occ = m & rec["occl"]
    if occ.any():
        d["occl_fix_rate"] = float(fix[occ].mean())
    return d


def group_table(metrics, by):
    """by: record field name or callable(rec)->array of labels. Returns {label: summary}."""
    rec = metrics["rec"]
    lab = by(rec) if callable(by) else rec[by]
    out = {}
    for v in sorted(set(lab.tolist()), key=lambda x: (str(type(x)), x)):
        out[v] = summarize(rec, lab == v)
    return out


def _speed_bin(rec):
    w = rec["omega"]
    return np.where(w < 1, "0", np.where(w <= 100, "1-100", np.where(w <= 200, "101-200", ">200")))


def _theta_bin(rec):
    t = rec["theta"]
    return np.where(t < 25, "00-25", np.where(t < 50, "25-50", np.where(t < 75, "50-75", "75-100")))


def _blur_bin(rec):
    b = rec["blur"]
    return np.where(b < 1.5, "<1.5", np.where(b < 4, "1.5-4", np.where(b < 8, "4-8", ">8")))


def _ppc_bin(rec):
    p = rec["ppc"]
    return np.where(p < 3.5, "<3.5", np.where(p < 5, "3.5-5", np.where(p < 8, "5-8", ">8")))


HDR = ("%-22s %5s | %5s %5s %5s %5s | %5s %5s %5s | %6s %5s %5s %6s | %5s %5s" %
       ("group", "n", "knee%", "hip%", "ank%", "fix%", "usbl%", "lowc%", "cmean", "bias", "sd", "p95", "max", "ms", "bad5%"))


def _row(label, d):
    if d.get("n", 0) == 0:
        return "%-22s (none)" % label
    return ("%-22s %5d | %5.1f %5.1f %5.1f %5.1f | %5.1f %5.1f %5.2f | %6.2f %5.2f %5.2f %6.1f | %5.1f %5.1f" % (
        str(label)[:22], d["n"], 100 * d["knee"], 100 * d["hip"], 100 * d["ankle"], 100 * d["fix"],
        100 * d["usable"], 100 * d["lowconf"] if d["lowconf"] == d["lowconf"] else float("nan"),
        d["conf_mean"], d["bias"], d["sd"], d["p95"], d["max"], d["ms"], 100 * d["bad5"]))


def print_report(metrics, title=""):
    rec = metrics["rec"]
    print("\n=== %s ===  (err in deg vs TRUE 3D knee angle, over non-occluded frames with a fix;"
          " usbl = fix & conf>=0.5; lowc = share of fixes with conf<0.5; bad5 = usable fixes with |err|>5)" % title)
    print(HDR)
    print(_row("OVERALL", metrics["overall"]))
    sections = [("resolution", lambda r: np.array(["%dpx" % x for x in r["res"]])),
                ("distance m", lambda r: np.array(["%.1f m" % x for x in r["dist"]])),
                ("view yaw/pitch", lambda r: np.array(["yaw %d" % x for x in r["view"]])),
                ("light", "light"), ("speed deg/s", _speed_bin), ("knee angle", _theta_bin),
                ("ankle blur px", _blur_bin), ("knee px/cm", _ppc_bin)]
    for name, by in sections:
        print("-- by %s" % name)
        for lab, d in group_table(metrics, by).items():
            print(_row(lab, d))
    print("-- res x dist x light (view pooled)")
    comb = lambda r: np.array(["%d %.1fm %s" % (a, b, c) for a, b, c in zip(r["res"], r["dist"], r["light"])])
    for lab, d in group_table(metrics, comb).items():
        print(_row(lab, d))
    o = metrics["overall"]
    if "occl_fix_rate" in o:
        print("occluded-marker frames: fix produced in %.1f%% (should be ~0)" % (100 * o["occl_fix_rate"]))
    print("time per process_frame: mean %.2f ms, p95 %.2f ms" % (o["ms"], o["ms_p95"]))


def print_compare(m0, m1, names=("A", "B")):
    keys = ["knee", "hip", "ankle", "fix", "usable", "lowconf", "conf_mean", "bias", "sd", "p95", "max", "bad5", "ms"]
    print("\n%-10s %10s %10s" % ("overall", names[0], names[1]))
    for k in keys:
        print("%-10s %10.3f %10.3f" % (k, m0["overall"][k], m1["overall"][k]))


# --------------------------------------------------------------------------
# self test: convention + zero-error checks
# --------------------------------------------------------------------------
def selftest():
    print("selftest: clean render (no noise/blur/tilt/depth/distortion, view 0), tracker vs truth")
    clean = Degr("clean", gain=1.0, contrast=1.0, gamma=1.0, noise=0.0, defocus=0.0, exposure_ms=10,
                 jpeg_q=95, bg_level=130, vignette=0.0)
    bad = 0
    for flip in (False, True):
        tr = original_factory(flip_x=flip)
        worst = 0.0
        for res in ((1280, 720), (640, 480)):
            for dist in (0.7, 1.0):
                for theta in (0, 15, 30, 45, 60, 75, 90, 100):
                    for sway in (-3, 0, 4):
                        pose = Pose(theta=theta, sway=sway, thigh_cm=40, shank_cm=40,
                                    tilt={n: (0., 0.) for n in IDS})
                        cam = Cam(w=res[0], h=res[1], dist=dist, flip_x=flip)
                        img, t = render(pose, cam, clean, np.random.default_rng(1))
                        fix = tr.process_frame(img, 0.0)
                        if fix.camera_knee_deg is None:
                            print("  no fix", res, dist, theta, flip); bad += 1
                            continue
                        e = wrap(fix.camera_knee_deg - theta)
                        worst = max(worst, abs(e))
                        if abs(e) > 0.5:
                            print("  err %.2f at" % e, res, dist, theta, sway, "flip" if flip else "")
                            bad += 1
                        # check sign convention of headings too
                        if abs(wrap(fix.thigh_heading_deg - sway)) > 1.0 or abs(wrap(fix.shank_heading_deg - (sway + theta))) > 1.0:
                            print("  heading mismatch", fix.thigh_heading_deg, fix.shank_heading_deg, sway, theta)
                            bad += 1
        print("  flip_x=%s: worst |err| = %.3f deg" % (flip, worst))
    # sign check with an independent derivation: hanging shank must have positive angle, shank tip BELOW knee
    pose = Pose(theta=90, tilt={n: (0., 0.) for n in IDS})
    sc = build_scene(pose, Cam())
    below = sc["ankle"]["c_uv"][1] > sc["knee"]["c_uv"][1]
    fwd = sc["ankle"]["c_uv"][0]  - sc["knee"]["c_uv"][0]
    straight = build_scene(Pose(theta=0, tilt={n: (0., 0.) for n in IDS}), Cam())
    fwd0 = straight["ankle"]["c_uv"][0] - straight["knee"]["c_uv"][0]
    hipx = straight["hip"]["c_uv"][0] - straight["knee"]["c_uv"][0]
    print("  theta=90: ankle below knee in image: %s;  theta=0: ankle right of knee (+x fwd): %s, hip left: %s"
          % (below, fwd0 > 0, hipx < 0))
    print("  marker 'down' axis check: TL->BL of hip marker (thigh horizontal) points image-right:",
          (lambda k: k[3][0] - k[0][0] > 0 and abs(k[3][1] - k[0][1]) < 1e-3)(build_scene(Pose(theta=0, tilt={n: (0., 0.) for n in IDS}), Cam())["hip"]["k_uv"]))
    # geometry-only floor: projected-centre angle vs truth under yaw/pitch (shows image-plane foreshortening)
    for yaw, pitch in ((10, 5), (20, 10)):
        errs = []
        for theta in range(0, 101, 10):
            cam = Cam(yaw=yaw, pitch=pitch)
            pose = Pose(theta=theta, tilt={n: (0., 0.) for n in IDS})
            tt = truth_for(pose, cam, build_scene(pose, cam))
            errs.append(wrap(tt["proj_deg"] - theta))
        print("  view yaw %d pitch %d: image-plane angle - true angle ranges %.1f .. %.1f deg (inherent, not tracker error)"
              % (yaw, pitch, min(errs), max(errs)))
    print("selftest %s" % ("FAILED (%d problems)" % bad if bad else "OK"))
    return bad


def dump(dirpath, seed=1234):
    os.makedirs(dirpath, exist_ok=True)
    scen = [Scenario(res=(1280, 720), dist=1.0, view=10, light="good", n=2, name="a"),
            Scenario(res=(640, 480), dist=1.5, view=20, light="dim", n=2, name="b"),
            Scenario(res=(640, 480), dist=0.7, view=0, light="good", n=2, name="c")]
    for si, k, img, t in gen_frames(scen, seed):
        p = os.path.join(dirpath, "sim_%d_%d_theta%.0f_w%.0f.png" % (si, k, t["theta"], t["omega"]))
        cv2.imwrite(p, img)
        print("wrote", p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="~20 s run")
    ap.add_argument("-n", type=int, default=None, help="samples per scenario (36 scenarios + occlusion)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mount", choices=["leg", "brace"], default="leg", help="marker mounting model")
    ap.add_argument("--seq", action="store_true", help="static-camera rep sequences (temporal behaviour) instead of the sweep")
    ap.add_argument("--factory", help="module:callable returning a tracker (factory(flip_x=False)); "
                                      "evaluated side by side with the original")
    ap.add_argument("--dump", metavar="DIR", help="write a few example frames and exit")
    a = ap.parse_args(argv)
    if a.selftest:
        return 1 if selftest() else 0
    if a.dump:
        dump(a.dump, a.seed)
        return 0
    scen = make_sequences(60 if a.quick else (a.n or 150), a.mount) if a.seq else make_scenarios(quick=a.quick, n=a.n, mount=a.mount)
    nfr = sum(s.n for s in scen)
    print("sim_camera: %d scenarios, %d frames, seed %d%s" % (len(scen), nfr, a.seed, " (quick)" if a.quick else ""))
    t0 = time.time()
    if a.factory:
        mod, fn = a.factory.split(":")
        newf = getattr(importlib.import_module(mod), fn)
        res = evaluate({"orig": original_factory, "new": newf}, scen, a.seed, verbose=False, fresh=not a.seq)
        print_report(res["orig"], "ORIGINAL")
        print_report(res["new"], "MODIFIED (%s)" % a.factory)
        print_compare(res["orig"], res["new"], ("orig", "new"))
    else:
        res = evaluate(original_factory, scen, a.seed, fresh=not a.seq)
        print_report(res, "CameraTracker (original)")
    print("\ntotal wall time %.1f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
