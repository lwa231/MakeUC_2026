#!/usr/bin/env python3
"""
camera_origin.py -- ArUco side-camera reference for the Brace Tracker.

A webcam looks at the SIDE of the leg (image plane == sagittal plane).  Three
ArUco markers are stuck on the outer side of the leg, facing the camera:

    knee  (id 1)   on the knee joint  -> defines the fixed ORIGIN
    thigh (id 10)  on the thigh band, near the thigh accelerometer
    shank (id 11)  on the shank band, near the shank accelerometer
    (optional extras: hip id 0, ankle id 2 -- reported if seen)

The camera gives the bridge a drift-free absolute position reference:
where the knee is relative to its calibrated origin, where each sensor band
is relative to the knee, and each segment's heading.

FRAME CONVENTION (matches serial_bridge.py)
-------------------------------------------
Sagittal plane, origin = knee marker centre at the reference pose,
+x = forward (toes when leg straight), +y = UP, units = centimetres.
Heading phi (deg) is clockwise-positive with direction vector
u(phi) = (cos phi, -sin phi): thigh points hip->knee along u(phi_thigh),
shank points knee->ankle along u(phi_shank); flexion = phi_shank - phi_thigh.
(Standing straight: both segments point down, phi = 90.  Leg stretched
forward horizontally: phi = 0.)

Image coordinates have y DOWN.  With flip_x=False image-right = forward;
with flip_x=True (camera on the other side) image-left = forward.

MARKER ORIENTATION -> HEADING CONVENTION
----------------------------------------
Each marker is read in its own frame: corners are TL, TR, BR, BL of the
marker as printed (the "top" of the marker is the side that has TL/TR).
The marker's DOWN axis (TL->BL, averaged with TR->BR) is taken to point
ALONG THE SEGMENT, from proximal to distal (hip->knee for the thigh,
knee->ankle for the shank).  Consequences:

  * stick the marker upright (top of the marker towards the hip) on a
    standing leg -> heading 90 deg for both segments (down).
  * heading = atan2(-d_world_y, d_world_x) where d_world is the marker
    down-axis in the world frame (y up, x flipped if flip_x).
  * if you stick a marker on rotated (e.g. 90 deg clockwise), set that
    marker's offset: heading = measured + offset.  Offsets are
    thigh_heading_offset_deg / shank_heading_offset_deg constructor args
    (or --thigh-offset / --shank-offset on the CLI).  Easy way to find the
    offset: pose the leg at a known heading (e.g. standing, 90 deg), read
    the printed heading, and set offset = known - measured.

ASSUMPTIONS / ERROR BUDGET
--------------------------
* The camera looks roughly perpendicular to the leg, so the markers lie
  roughly parallel to the image plane, and all markers are at about the same
  distance.  px_per_cm = mean marker side (px) / marker_cm is then one
  scale for the whole image (knee marker when visible, else the mean of
  the markers seen).
* If a marker plane is tilted by angle theta about an axis in the image
  plane its apparent size shrinks by cos(theta): scale error (1-cos theta)
  = 1.5% at 10 deg, 3.4% at 15 deg, 6.4% at 20 deg, 13% at 30 deg.  Position
  error is that fraction of the distance from the knee marker (e.g. 40 cm
  thigh offset * 3.4% = 1.4 cm at 15 deg).  Tilt also biases headings by a
  similar small amount.  Depth differences between markers (leg is a
  cylinder, the camera is not telecentric) cause the same sort of error.
* Lens distortion is not corrected (no calibration); wide-angle webcams
  distort positions near the image edges -- keep the leg in the central
  part of the frame.
* Camera must not move after calibrate_reference().  Reference is stored in
  pixels; displacement is converted to cm with the current px_per_cm.

Dependencies: opencv-contrib-python (or any OpenCV build with cv2.aruco)
and the standard library only.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

Vec2 = Tuple[float, float]

DEFAULT_DICT = "DICT_4X4_50"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def wrap_deg(a: float) -> float:
    """Wrap an angle to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def _get_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco not available: install opencv-contrib-python "
            "(or opencv-contrib-python-headless)")
    if not hasattr(cv2.aruco, name):
        raise ValueError("unknown ArUco dictionary %r" % name)
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _make_params():
    if hasattr(cv2.aruco, "DetectorParameters"):
        try:
            p = cv2.aruco.DetectorParameters()
        except Exception:
            p = cv2.aruco.DetectorParameters_create()
    else:
        p = cv2.aruco.DetectorParameters_create()
    try:
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    except Exception:
        pass
    return p


def _inv_sqrt2(M):
    """Symmetric inverse square root of a symmetric positive-definite 2x2 matrix."""
    w, v = np.linalg.eigh(M)
    w = np.maximum(w, 1e-9)
    return (v * (1.0 / np.sqrt(w))) @ v.T


CONF_K_ANISO, CONF_K_SPREAD, CONF_K_DEV, CONF_K_SMALL, CONF_SIGMA0 = 17.0, 0.5, 5.5, 6.0, 2.5
CONF_QREF = 0.97          # quality of an ideal frame (sigma ~0.5 deg); divides it out so good frames read ~1.0
VIEW_DEADZONE = 0.03   # ln(axis ratio) of the foreshortening ellipse that is treated as noise (~2% anisotropy)


def _view_W(M, dead=None):
    """W = M^(-1/2), but with the anisotropy soft-thresholded: marker tilt on the curved leg and corner noise
    produce a few % of spurious anisotropy, and 'correcting' that would add error where the view is already
    nearly perpendicular.  M = U diag(e1, e2) U^T (unit det): ln(e1/e2) is shrunk by `dead` before inversion."""
    dead = VIEW_DEADZONE if dead is None else dead
    w, v = np.linalg.eigh(M)
    w = np.maximum(w, 1e-9)
    l = math.log(w[1] / w[0])
    l2 = math.copysign(max(abs(l) - dead, 0.0), l)
    e1, e2 = math.exp(-l2 / 2.0), math.exp(l2 / 2.0)      # eigenvalues (w[0], w[1]) of the shrunk unit-det M
    return (v * (1.0 / np.sqrt([e1, e2]))) @ v.T


def _unit_det(M):
    d = float(np.linalg.det(M))
    return M / math.sqrt(d) if d > 1e-12 else M


def _marker_shape(pts):
    """Foreshortening 'shape' of one marker: S = A A^T with A = [top edge | left edge] (image px),
    normalised to det 1.  A = K R (K = plane->image affine, R = the marker's in-plane rotation), so S = K K^T
    for every marker on the same plane, whatever its rotation.  None if degenerate."""
    tl, tr, br, bl = pts
    top = 0.5 * ((tr - tl) + (br - bl))
    left = 0.5 * ((bl - tl) + (br - tr))
    A = np.column_stack([top, left])
    S = A @ A.T
    d = float(np.linalg.det(S))
    if not (d > 1e-6):
        return None
    return S / math.sqrt(d)


def generate_marker(dictionary, marker_id: int, size_px: int):
    """Render a marker image (new or old OpenCV API)."""
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    return cv2.aruco.drawMarker(dictionary, marker_id, size_px)


# --------------------------------------------------------------------------
# result type
# --------------------------------------------------------------------------
@dataclass
class CameraFix:
    """One camera measurement, in the bridge's sagittal frame
    (origin = reference knee position, +x forward, +y up, cm).

    t                 time.monotonic() of the frame
    knee_visible / thigh_visible / shank_visible   marker seen this frame
    knee_px           knee marker centre (x, y) in image pixels, or None
    px_per_cm         image scale (see module docstring), or None if no marker
    reference_set     True once calibrate_reference()/set_reference() was done
    knee_world_cm     knee marker displacement from the reference pose (cm),
                      None if knee not visible or no reference yet
    thigh_rel_cm      thigh marker centre relative to the knee marker (cm);
    shank_rel_cm      shank marker centre relative to the knee marker (cm);
                      None if that marker or the knee marker is not visible
    thigh_heading_deg / shank_heading_deg
                      segment heading phi (deg, clockwise-positive, wrapped
                      to (-180,180]) from the marker's in-plane rotation, or
                      None if the marker is not visible
    camera_knee_deg   shank_heading - thigh_heading (wrapped), or None
    extra_rel_cm      {"hip": (x,y), "ankle": (x,y)} for optional extra
                      markers seen (relative to knee, needs knee visible)
    hip_rel_cm / ankle_rel_cm
                      the ARM-END markers (default IDs 0 = free end of the thigh arm, 2 = free end
                      of the shank arm): where each arm ends, relative to the knee origin (cm)
    thigh_len_cm / shank_len_cm
                      arm lengths as the camera sees them = distance from the knee marker centre to
                      the arm-end marker centre (cm); None if either marker is missing. Put the
                      end markers' centres exactly at the ends you want measured.
    confidence        0..1 =  count_score * (0.6 + 0.4*size_consistency) * recency
                        count_score = 0.4*knee + 0.3*thigh + 0.3*shank
                                      (1 if seen, else 0)
                        size_consistency = 1 - ((max-min)/mean of the seen
                                      required markers' side lengths) / 0.3,
                                      clipped to [0,1] (1 if <2 markers)
                        recency = 1 at process time; CameraTracker.latest()
                                  scales it by 1 - 0.5*age/max_age.
    """
    t: float
    knee_visible: bool = False
    thigh_visible: bool = False
    shank_visible: bool = False
    knee_px: Optional[Vec2] = None
    px_per_cm: Optional[float] = None
    reference_set: bool = False
    knee_world_cm: Optional[Vec2] = None
    thigh_rel_cm: Optional[Vec2] = None
    shank_rel_cm: Optional[Vec2] = None
    thigh_heading_deg: Optional[float] = None
    shank_heading_deg: Optional[float] = None
    camera_knee_deg: Optional[float] = None
    extra_rel_cm: Dict[str, Vec2] = field(default_factory=dict)
    confidence: float = 0.0
    hip_rel_cm: Optional[Vec2] = None
    ankle_rel_cm: Optional[Vec2] = None
    thigh_len_cm: Optional[float] = None
    shank_len_cm: Optional[float] = None

    def format_line(self) -> str:
        def v(p):
            return "  --  " if p is None else "(%6.1f,%6.1f)" % p

        def a(x):
            return "  -- " if x is None else "%6.1f" % x
        return ("conf=%.2f k%s t%s s%s ppc=%s knee_w=%s thigh_rel=%s shank_rel=%s "
                "ph_t=%s ph_s=%s flex=%s arms(thigh/shank)=%s/%s cm" % (
                    self.confidence,
                    "+" if self.knee_visible else "-",
                    "+" if self.thigh_visible else "-",
                    "+" if self.shank_visible else "-",
                    "  -- " if self.px_per_cm is None else "%5.1f" % self.px_per_cm,
                    v(self.knee_world_cm), v(self.thigh_rel_cm), v(self.shank_rel_cm),
                    a(self.thigh_heading_deg), a(self.shank_heading_deg),
                    a(self.camera_knee_deg),
                    a(self.thigh_len_cm).strip(), a(self.shank_len_cm).strip()))


# --------------------------------------------------------------------------
# tracker
# --------------------------------------------------------------------------
class CameraTracker:
    """Background-thread ArUco tracker.

    Typical use:
        cam = CameraTracker(cam_index=0, marker_cm=5.0)
        if cam.start():
            cam.calibrate_reference(2.0)     # hold the leg still, knee visible
            ...
            fix = cam.latest()               # None if stale/no camera
        cam.stop()

    For offline use / tests: CameraTracker().process_frame(img).
    """

    def __init__(self, cam_index: int = 0, marker_cm: float = 5.0,
                 dictionary: str = DEFAULT_DICT,
                 knee_id: int = 1, thigh_id: Optional[int] = None, shank_id: Optional[int] = None,
                 hip_id: Optional[int] = 0, ankle_id: Optional[int] = 2,
                 flip_x: bool = False,
                 thigh_heading_offset_deg: float = 0.0,
                 shank_heading_offset_deg: float = 0.0,
                 max_age_s: float = 0.2,
                 width: Optional[int] = None, height: Optional[int] = None,
                 fps: Optional[float] = None):
        self.cam_index = cam_index
        self.marker_cm = float(marker_cm)
        self.dictionary_name = dictionary
        # thigh / shank = OPTIONAL markers next to each sensor (None = not used; -1 never matches).
        # The default 3-marker set is knee (1) + the free end of each arm (hip 0, ankle 2).
        self.ids = {"knee": knee_id, "thigh": -1 if thigh_id is None else thigh_id,
                    "shank": -1 if shank_id is None else shank_id}
        if hip_id is not None:
            self.ids["hip"] = hip_id
        if ankle_id is not None:
            self.ids["ankle"] = ankle_id
        self.flip_x = bool(flip_x)
        self.heading_offset_deg = {"thigh": float(thigh_heading_offset_deg),
                                   "shank": float(shank_heading_offset_deg)}
        self.max_age_s = float(max_age_s)
        self.width, self.height, self.fps = width, height, fps

        self._dict = _get_dictionary(dictionary)
        self._params = _make_params()
        self._detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)

        self._lock = threading.Lock()
        self._latest: Optional[CameraFix] = None
        self._ref_px: Optional[Vec2] = None
        self._cal_samples: Optional[list] = None
        self._len_hist = {"thigh": deque(maxlen=150), "shank": deque(maxlen=150)}  # recent arm lengths, cm
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cap = None
        # view (foreshortening) correction state, see _view_matrix()
        self.view_tau_s = 1.5            # EMA time constant of the smoothed foreshortening matrix
        self.hold_thigh_s = 1.0          # hold the last thigh heading this long if only the hip-end marker is lost
        self._view_M = None              # smoothed unit-det 2x2 (K K^T), image px
        self._view_t = None
        self._view_jumps = 0
        self._thigh_hold = None          # (t, thigh heading deg)
        self.debug: dict = {}            # per-frame internals (for the simulator / tuning)
        self.error: Optional[str] = None
        self.last_frame = None
        self.last_detections: Dict[int, object] = {}   # id -> 4x2 corners (for drawing)

    def arm_lengths_cm(self, min_samples: int = 10):
        """(thigh_arm_cm, shank_arm_cm): median knee-marker -> arm-end-marker distance over the
        last ~1.5 s of frames (median ignores the odd bad detection). None where the end marker
        has not been seen enough."""
        out = []
        with self._lock:
            for name in ("thigh", "shank"):
                h = sorted(self._len_hist[name])
                out.append(h[len(h) // 2] if len(h) >= min_samples else None)
        return tuple(out)

    def viewer_args(self) -> Optional[str]:
        """The main.cpp options matching the measured arms, e.g. '--thigh-len 44.8 --shin-len 41.9'."""
        t, s = self.arm_lengths_cm()
        parts = []
        if t is not None:
            parts.append("--thigh-len %.1f" % t)
        if s is not None:
            parts.append("--shin-len %.1f" % s)
        return " ".join(parts) or None

    # ---- reference / origin ------------------------------------------------
    @property
    def reference_px(self) -> Optional[Vec2]:
        with self._lock:
            return self._ref_px

    def set_reference(self, knee_px: Optional[Vec2]) -> None:
        """Set (or clear with None) the origin directly, in image pixels."""
        with self._lock:
            self._ref_px = None if knee_px is None else (float(knee_px[0]), float(knee_px[1]))

    def calibrate_reference(self, seconds: float = 2.0, min_samples: int = 5, pump=None) -> Optional[Vec2]:
        """Average the knee marker pixel position for `seconds` (leg still,
        knee marker visible) and store it as the origin.  Frames must be
        arriving (start() running, or another thread calling process_frame), or pass
        pump=tracker.grab_once to pull frames on the calling thread while waiting.
        Returns the reference px (x, y), or None if fewer than `min_samples`
        frames showed the knee marker (reference is then left unchanged)."""
        with self._lock:
            self._cal_samples = []
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end:
            if pump is not None:
                pump()
            else:
                time.sleep(0.01)
        with self._lock:
            samples, self._cal_samples = self._cal_samples, None
            if samples is None or len(samples) < min_samples:
                return None
            ref = (sum(s[0] for s in samples) / len(samples),
                   sum(s[1] for s in samples) / len(samples))
            self._ref_px = ref
            return ref

    # ---- core processing ---------------------------------------------------
    def _detect(self, gray):
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dict, parameters=self._params)
        return corners, ids

    def _view_matrix(self, found, t):
        """Foreshortening matrix M (unit det, = K K^T up to scale) from the markers seen, smoothed over time
        (the camera does not move relative to the leg plane).  Returns (W, info): W = M^(-1/2) maps image-plane
        vectors to leg-plane vectors up to a ROTATION (W K = rotation), and a common rotation leaves
        shank_heading - thigh_heading unchanged.  W is symmetric, i.e. the stretch with the smallest rotation,
        so individual headings stay close to the image-plane ones (offset = rotation part of K, a few deg at most).
        info: dict(n, dev, jump) -- n markers used, dev = worst marker/mean shape disagreement, jump flag."""
        items = [(v[1], v[3]) for v in found.values() if v[3] is not None]
        info = {"n": len(items), "dev": 0.0, "jump": False}
        if not items:
            M = self._view_M
            return (_view_W(M) if M is not None else np.eye(2)), info
        wts = np.array([w for w, _ in items], float)
        Ss = [S for _, S in items]

        def wmean(idx):
            Mx = sum(wts[i] * Ss[i] for i in idx) / sum(wts[i] for i in idx)
            return _unit_det(Mx)
        idx = list(range(len(items)))
        M = wmean(idx)
        Wm = _inv_sqrt2(M)
        devs = [float(np.linalg.norm(Wm @ S @ Wm - np.eye(2))) for S in Ss]
        if len(items) >= 3 and max(devs) > 0.2:          # one marker disagrees strongly (tilted / misdetected)
            idx.remove(int(np.argmax(devs)))
            M = wmean(idx)
            Wm = _inv_sqrt2(M)
            devs = [float(np.linalg.norm(Wm @ Ss[i] @ Wm - np.eye(2))) for i in idx]
        info["dev"] = max(devs) if devs else 0.0
        if self._view_M is None:
            self._view_M, self._view_t, self._view_jumps = M, t, 0
        else:
            Wo = _inv_sqrt2(self._view_M)
            jump = float(np.linalg.norm(Wo @ M @ Wo - np.eye(2)))
            if jump > 0.3:                                # camera moved (or garbage): keep old M, reset if persistent
                info["jump"] = True
                self._view_jumps += 1
                if self._view_jumps >= 10:
                    self._view_M, self._view_t, self._view_jumps = M, t, 0
            else:
                self._view_jumps = 0
                dt = min(max(t - (self._view_t if self._view_t is not None else t), 0.0), 1.0)
                a = 1.0 - math.exp(-dt / max(self.view_tau_s, 1e-3))
                if len(items) == 1:
                    a *= 0.3                              # one marker is a weaker estimate
                self._view_M = _unit_det((1.0 - a) * self._view_M + a * M)
                self._view_t = t
        w_, _ = np.linalg.eigh(self._view_M)
        info["aniso"] = abs(math.log(max(w_[1], 1e-9) / max(w_[0], 1e-9)))     # ln axis-ratio^2 of the view ellipse
        return _view_W(self._view_M), info

    def process_frame(self, img, t: Optional[float] = None) -> CameraFix:
        """Detect markers in one BGR/gray numpy image and return a CameraFix.
        Never raises on a bad frame (returns an empty fix, sets .error)."""
        if t is None:
            t = time.monotonic()
        try:
            return self._process(img, t)
        except Exception as e:  # defensive: bad frame must not kill the thread
            self.error = "process_frame: %r" % (e,)
            with self._lock:
                ref = self._ref_px
            return CameraFix(t=t, reference_set=ref is not None)

    def _process(self, img, t: float) -> CameraFix:
        with self._lock:
            ref = self._ref_px
        if img is None or getattr(img, "size", 0) == 0:
            return CameraFix(t=t, reference_set=ref is not None)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        corners, ids = self._detect(gray)

        # id -> (centre(x,y) px, side px, down-axis vector px)
        found: Dict[int, tuple] = {}
        det_draw: Dict[int, object] = {}
        if ids is not None:
            for c, mid in zip(corners, ids.flatten()):
                mid = int(mid)
                pts = c.reshape(4, 2).astype("float64")
                tl, tr, br, bl = pts
                centre = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
                side = 0.25 * (math.dist(tl, tr) + math.dist(tr, br)
                               + math.dist(br, bl) + math.dist(bl, tl))
                down = (0.5 * ((bl[0] - tl[0]) + (br[0] - tr[0])),
                        0.5 * ((bl[1] - tl[1]) + (br[1] - tr[1])))
                if mid in found and found[mid][1] >= side:
                    continue           # duplicate id: keep the larger one
                found[mid] = (centre, side, down, _marker_shape(pts))
                det_draw[mid] = pts
        self.last_detections = det_draw

        seen = {name: found.get(i) for name, i in self.ids.items()}
        fix = CameraFix(t=t, reference_set=ref is not None)
        fix.knee_visible = seen["knee"] is not None
        fix.thigh_visible = seen["thigh"] is not None
        fix.shank_visible = seen["shank"] is not None

        # scale: knee marker if visible, else mean of markers seen
        if seen["knee"] is not None:
            side_px = seen["knee"][1]
        else:
            sides = [v[1] for v in seen.values() if v is not None]
            side_px = sum(sides) / len(sides) if sides else None
        if side_px is not None and self.marker_cm > 0:
            fix.px_per_cm = side_px / self.marker_cm
        ppc = fix.px_per_cm

        sx = -1.0 if self.flip_x else 1.0

        def to_world_vec(dx_px, dy_px):
            return (sx * dx_px, -dy_px)

        kc = None
        if fix.knee_visible:
            kc = seen["knee"][0]
            fix.knee_px = kc
            with self._lock:
                if self._cal_samples is not None:
                    self._cal_samples.append(kc)
            if ref is not None and ppc:
                wx, wy = to_world_vec(kc[0] - ref[0], kc[1] - ref[1])
                fix.knee_world_cm = (wx / ppc, wy / ppc)

        def rel(name):
            v = seen.get(name)
            if v is None or kc is None or not ppc:
                return None
            wx, wy = to_world_vec(v[0][0] - kc[0], v[0][1] - kc[1])
            return (wx / ppc, wy / ppc)

        def heading(name):
            v = seen.get(name)
            if v is None:
                return None
            wx, wy = to_world_vec(*v[2])           # marker down-axis, world
            return wrap_deg(math.degrees(math.atan2(-wy, wx))
                            + self.heading_offset_deg[name])

        fix.thigh_rel_cm = rel("thigh")
        fix.shank_rel_cm = rel("shank")
        fix.thigh_heading_deg = heading("thigh")
        fix.shank_heading_deg = heading("shank")
        # Arm-end markers give the segment direction directly: knee -> hip end is -u(phi_thigh),
        # knee -> ankle end is +u(phi_shank). The ~40 cm lever arm makes this much more accurate than the
        # rotation of one small marker, so it wins when available (needs no per-marker offset).
        # The arm vectors are corrected for view foreshortening first (W: image px -> leg-plane px up to a common
        # rotation, applied in image coordinates, before the flip/y-up conversion).  rel_cm / len_cm stay unrectified.
        W, vinfo = self._view_matrix(found, t)

        def rect_rel(name):
            v = seen.get(name)
            if v is None or kc is None or not ppc:
                return None
            ix, iy = W @ np.array([v[0][0] - kc[0], v[0][1] - kc[1]])
            wx, wy = to_world_vec(float(ix), float(iy))
            return (wx / ppc, wy / ppc)

        hr, ar = rect_rel("hip"), rect_rel("ankle")
        held = False
        if hr is not None and math.hypot(*hr) > 1.0:
            fix.thigh_heading_deg = wrap_deg(math.degrees(math.atan2(hr[1], -hr[0])))
            self._thigh_hold = (t, fix.thigh_heading_deg)
        if ar is not None and math.hypot(*ar) > 1.0:
            fix.shank_heading_deg = wrap_deg(math.degrees(math.atan2(-ar[1], ar[0])))
        if (fix.thigh_heading_deg is None and seen.get("hip") is None and "hip" in self.ids
                and fix.shank_heading_deg is not None and ar is not None and self._thigh_hold is not None
                and 0.0 <= t - self._thigh_hold[0] <= self.hold_thigh_s):
            fix.thigh_heading_deg = self._thigh_hold[1]      # seated extension: the thigh hardly moves
            held = True
        if fix.thigh_heading_deg is not None and fix.shank_heading_deg is not None:
            fix.camera_knee_deg = wrap_deg(fix.shank_heading_deg - fix.thigh_heading_deg)
        for name in ("hip", "ankle"):
            r = rel(name)
            if r is not None:
                fix.extra_rel_cm[name] = r
        fix.hip_rel_cm = fix.extra_rel_cm.get("hip")
        fix.ankle_rel_cm = fix.extra_rel_cm.get("ankle")
        if fix.hip_rel_cm is not None:
            fix.thigh_len_cm = math.hypot(*fix.hip_rel_cm)
        if fix.ankle_rel_cm is not None:
            fix.shank_len_cm = math.hypot(*fix.ankle_rel_cm)
        with self._lock:
            if fix.thigh_len_cm is not None:
                self._len_hist["thigh"].append(fix.thigh_len_cm)
            if fix.shank_len_cm is not None:
                self._len_hist["shank"].append(fix.shank_len_cm)

        # confidence: 0..1 ~ presence score * quality, quality = 1/(1 + sigma^2/2.5^2) where sigma (deg) is the
        # expected knee-angle error predicted from what the frame shows (coefficients fitted on tools/sim_camera.py
        # renders; the error is dominated by view foreshortening, not by corner noise):
        #   anisotropy of the view ellipse (the uncorrected part of the foreshortening grows with it),
        #   marker-size spread (depth / tilt / blur / mis-detection), shape disagreement between markers,
        #   small markers (noisy corners), a held (not measured) thigh heading.
        thigh_seen = seen["thigh"] is not None or seen.get("hip") is not None
        shank_seen = seen["shank"] is not None or seen.get("ankle") is not None
        score = 0.4 * fix.knee_visible + 0.3 * thigh_seen + 0.3 * shank_seen + (0.18 if held else 0.0)
        req_sides = [seen[n][1] for n in ("knee", "thigh", "shank", "hip", "ankle")
                     if seen.get(n) is not None]
        size_cons = 1.0
        if len(req_sides) >= 2:
            mean = sum(req_sides) / len(req_sides)
            spread = (max(req_sides) - min(req_sides)) / mean if mean > 0 else 1.0
            size_cons = min(1.0, max(0.0, 1.0 - spread / 0.3))
        end_sides = [seen[n][1] for n in ("knee", "hip", "ankle") if seen.get(n) is not None]
        min_side = min(end_sides) if end_sides else 0.0
        aniso, dev = vinfo.get("aniso", 0.0), vinfo.get("dev", 0.0)
        delta = 0.0           # how much the view correction changed this frame's knee angle (deg)
        hr0, ar0 = rel("hip"), rel("ankle")
        if hr0 is not None and ar0 is not None and hr is not None and ar is not None:
            k0 = math.degrees(math.atan2(-ar0[1], ar0[0])) - math.degrees(math.atan2(hr0[1], -hr0[0]))
            k1 = math.degrees(math.atan2(-ar[1], ar[0])) - math.degrees(math.atan2(hr[1], -hr[0]))
            delta = abs(wrap_deg(k1 - k0))
        var = (0.35 + (CONF_K_ANISO * aniso) ** 2 + (CONF_K_SPREAD * (1.0 - size_cons)) ** 2
               + (CONF_K_DEV * dev) ** 2 + (CONF_K_SMALL / max(min_side, 1.0)) ** 2 + (0.5 if held else 0.0))
        quality = 1.0 / (1.0 + var / CONF_SIGMA0 ** 2)
        fix.confidence = float(min(1.0, score * quality / CONF_QREF))
        self.debug = {"min_side": min_side, "spread": 1.0 - size_cons, "dev": dev, "aniso": aniso,
                      "n_view": vinfo["n"], "jump": vinfo["jump"], "held": held, "delta": delta, "score": score,
                      "sigma_deg": math.sqrt(var)}

        with self._lock:
            self._latest = fix
        return fix

    # ---- thread ------------------------------------------------------------
    def grab_once(self):
        """Read one frame on the CALLING thread and process it (for the --show / --snapshot
        modes, which must run on the main thread on macOS). Returns (frame, fix) or (None, None)."""
        try:
            ok, frame = self._cap.read()
        except Exception as e:
            self.error = "capture: %r" % (e,)
            return None, None
        if not ok or frame is None:
            return None, None
        self.last_frame = frame
        return frame, self.process_frame(frame, time.monotonic())

    def start(self, threaded: bool = True) -> bool:
        """Open the camera and (if threaded) start the capture thread.  Returns True on
        success.  If the camera cannot be opened, prints a message to stderr,
        sets .error, and returns False (latest() stays None).  threaded=False only opens
        the camera; the caller then pulls frames itself with grab_once()."""
        if self._thread is not None and self._thread.is_alive():
            return True
        cap, ok = None, False
        try:
            cap = cv2.VideoCapture(self.cam_index)
            if self.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.fps:
                cap.set(cv2.CAP_PROP_FPS, self.fps)
            ok = cap.isOpened()
        except Exception as e:
            self.error = "camera open failed: %r" % (e,)
        if not ok:
            if cap is not None:
                cap.release()
            self.error = self.error or ("cannot open camera index %s" % self.cam_index)
            print("[camera_origin] ERROR: %s (camera tracking disabled; "
                  "on macOS check System Settings > Privacy > Camera)" % self.error,
                  file=sys.stderr)
            return False
        self.error = None
        self._cap = cap
        self._stop.clear()
        if not threaded:
            return True
        self._thread = threading.Thread(target=self._run, name="CameraTracker", daemon=True)
        self._thread.start()
        return True

    def _run(self):
        bad = 0
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    bad += 1
                    if bad == 50:
                        self.error = "camera returned no frames"
                        print("[camera_origin] WARNING: camera returns no frames",
                              file=sys.stderr)
                    time.sleep(0.02)
                    continue
                bad = 0
                self.last_frame = frame
                self.process_frame(frame, time.monotonic())
            except Exception as e:  # never die on a bad frame
                self.error = "capture loop: %r" % (e,)
                time.sleep(0.02)
        try:
            self._cap.release()
        except Exception:
            pass

    def stop(self) -> None:
        self._stop.set()
        th = self._thread
        if th is not None and th is not threading.current_thread():
            th.join(timeout=2.0)
        elif th is None and getattr(self, "_cap", None) is not None:
            try:
                self._cap.release()   # unthreaded mode: nobody else will
            except Exception:
                pass
        self._thread = None

    def latest(self, max_age_s: Optional[float] = None) -> Optional[CameraFix]:
        """Most recent CameraFix, or None if there is none or it is older than
        max_age_s (default self.max_age_s).  confidence is scaled by recency
        (1 - 0.5*age/max_age)."""
        if max_age_s is None:
            max_age_s = self.max_age_s
        with self._lock:
            fix = self._latest
        if fix is None:
            return None
        age = max(0.0, time.monotonic() - fix.t)
        if age > max_age_s:
            return None
        rec = 1.0 - 0.5 * age / max_age_s if max_age_s > 0 else 1.0
        return replace(fix, confidence=fix.confidence * rec,
                       extra_rel_cm=dict(fix.extra_rel_cm))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def make_markers(out_dir: str, tracker: CameraTracker) -> None:
    os.makedirs(out_dir, exist_ok=True)
    inner, border = 300, 50
    for name in ("knee", "thigh", "shank", "hip", "ankle"):   # hip / ankle = free ends of the two arms
        if name not in tracker.ids or tracker.ids[name] < 0:
            continue
        mid = tracker.ids[name]
        img = generate_marker(tracker._dict, mid, inner)
        img = cv2.copyMakeBorder(img, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=255)
        path = os.path.join(out_dir, "marker_%s_id%d.png" % (name, mid))
        cv2.imwrite(path, img)
        print("wrote", path)
    full = tracker.marker_cm * (inner + 2 * border) / inner
    print("\nPRINTING: print each PNG with scaling OFF ('actual size') so the BLACK square\n"
          "(the marker, not the white border) measures exactly %.2f cm per side.\n"
          "With the 400x400 px PNGs that means the whole image, white border included,\n"
          "is %.2f cm square.  Keep the white border; mount flat, matte, upright\n"
          "(top of the marker towards the hip), facing the camera.\n"
          "Run the tracker with --marker-cm set to the measured size."
          % (tracker.marker_cm, full))


def list_cameras(max_index: int = 8) -> None:
    found = False
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found = True
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("camera %d: %dx%d" % (i, w, h))
        cap.release()
    if not found:
        print("no cameras found (indices 0..%d)" % (max_index - 1))


def draw_overlay(img, tracker: CameraTracker, fix: CameraFix):
    for mid, pts in list(tracker.last_detections.items()):
        cv2.polylines(img, [pts.astype("int32").reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        p = pts[0]
        cv2.putText(img, str(mid), (int(p[0]), int(p[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # knee -> arm-end lines, so you can see the origin and where each arm ends
    if fix.knee_px is not None and tracker.last_detections:
        k = (int(fix.knee_px[0]), int(fix.knee_px[1]))
        for name in ("hip", "ankle"):
            mid = tracker.ids.get(name)
            pts = tracker.last_detections.get(mid) if mid is not None else None
            if pts is not None:
                c = pts.mean(axis=0)
                cv2.line(img, k, (int(c[0]), int(c[1])), (255, 160, 0), 2)
                L = fix.thigh_len_cm if name == "hip" else fix.shank_len_cm
                if L is not None:
                    cv2.putText(img, "%.1f cm" % L, ((k[0] + int(c[0])) // 2, (k[1] + int(c[1])) // 2 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)
    ref = tracker.reference_px
    s = fix.format_line()
    cut = s.find(" ph_t=")
    lines = [s[:cut], s[cut + 1:]] if cut > 0 else [s]      # two lines so it fits the frame
    va = tracker.viewer_args()
    if va:
        lines.append("viewer: " + va)
    if ref is not None:
        r = (int(ref[0]), int(ref[1]))
        cv2.drawMarker(img, r, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(img, "origin", (r[0] + 8, r[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
        if fix.knee_px is not None:
            cv2.line(img, r, (int(fix.knee_px[0]), int(fix.knee_px[1])), (0, 0, 255), 1)
    else:
        lines.append("no origin set: press 'c' to calibrate (hold still)")
    for i, s in enumerate(lines):
        cv2.putText(img, s, (10, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3)
        cv2.putText(img, s, (10, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ArUco side-camera origin tracker for Brace Tracker")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--marker-cm", type=float, default=5.0, help="marker side length (cm)")
    ap.add_argument("--dict", default=DEFAULT_DICT, help="ArUco dictionary name")
    ap.add_argument("--knee-id", type=int, default=1)
    ap.add_argument("--thigh-id", type=int, default=None, help="optional marker next to the thigh sensor (e.g. 10)")
    ap.add_argument("--shank-id", type=int, default=None, help="optional marker next to the shank sensor (e.g. 11)")
    ap.add_argument("--hip-id", type=int, default=0, help="marker at the free end of the THIGH arm")
    ap.add_argument("--ankle-id", type=int, default=2, help="marker at the free end of the SHANK arm")
    ap.add_argument("--flip-x", action="store_true",
                    help="image-LEFT is forward (camera on the other side)")
    ap.add_argument("--thigh-offset", type=float, default=0.0,
                    help="deg added to thigh marker heading")
    ap.add_argument("--shank-offset", type=float, default=0.0,
                    help="deg added to shank marker heading")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--show", action="store_true", help="preview window (q quits, c recalibrates)")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="print a line per fix at ~10 Hz")
    ap.add_argument("--calibrate", type=float, default=0.0, metavar="SEC",
                    help="calibrate origin for SEC seconds at startup")
    ap.add_argument("--image", metavar="PATH", help="process a still image and print the fix")
    ap.add_argument("--make-markers", metavar="DIR", help="write printable marker PNGs")
    ap.add_argument("--snapshot", metavar="PATH",
                    help="grab one frame after a short warm-up, save it as PATH (.jpg) plus an annotated "
                         "PATH-annotated.jpg, print the fix, and exit (no window needed)")
    a = ap.parse_args(argv)

    if a.list_cameras:
        list_cameras()
        return 0

    tr = CameraTracker(cam_index=a.cam, marker_cm=a.marker_cm, dictionary=a.dict,
                       knee_id=a.knee_id, thigh_id=a.thigh_id, shank_id=a.shank_id,
                       hip_id=a.hip_id, ankle_id=a.ankle_id,
                       flip_x=a.flip_x, thigh_heading_offset_deg=a.thigh_offset,
                       shank_heading_offset_deg=a.shank_offset,
                       width=a.width, height=a.height)

    if a.make_markers:
        make_markers(a.make_markers, tr)
        return 0

    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            print("cannot read image", a.image, file=sys.stderr)
            return 1
        fix = tr.process_frame(img)
        if fix.knee_px is not None:
            # a single still has no reference; use it as the origin so
            # knee_world_cm prints (0, 0)
            tr.set_reference(fix.knee_px)
            fix = tr.process_frame(img)
        print(fix)
        print(fix.format_line())
        return 0

    if a.snapshot:
        if not tr.start(threaded=False):
            return 2
        try:
            frame, fix, t0 = None, None, time.monotonic()
            n = 0
            while time.monotonic() - t0 < 8.0:        # first frames are dark while exposure settles
                f, x = tr.grab_once()
                if f is not None:
                    n += 1
                    frame, fix = f, x
                    if n >= 20 and time.monotonic() - t0 > 1.0:
                        break
                else:
                    time.sleep(0.05)
            if frame is None:
                print("no frame arrived within 8 s (camera busy, or not allowed to capture?)", file=sys.stderr)
                return 4
            h, w = frame.shape[:2]
            print("grabbed %d frames, last is %dx%d" % (n, w, h))
            raw_path = a.snapshot
            base, ext = os.path.splitext(raw_path)
            ann_path = base + "-annotated" + (ext or ".jpg")
            cv2.imwrite(raw_path, frame)
            cv2.imwrite(ann_path, draw_overlay(frame.copy(), tr, fix))
            print(fix.format_line())
            print("detected marker ids:", sorted(tr.last_detections.keys()) or "none")
            print("saved", raw_path, "and", ann_path)
            return 0
        finally:
            tr.stop()

    if not (a.show or a.do_print):
        ap.print_help()
        return 0

    if a.show:
        # Window + capture on the MAIN thread (macOS needs that for windows; also avoids silent
        # stalls of a background capture thread).
        if not tr.start(threaded=False):
            return 2
        print("camera open; waiting for frames (the first can take a few seconds). q quits, c recalibrates.")
        t_start, got, last_status = time.monotonic(), 0, 0.0
        try:
            if a.calibrate > 0:
                print("calibrating origin for %.1f s ..." % a.calibrate)
                print("reference px:", tr.calibrate_reference(a.calibrate, pump=tr.grab_once))
            while True:
                frame, f = tr.grab_once()
                now = time.monotonic()
                if frame is None:
                    if now - last_status > 3.0:
                        last_status = now
                        print("... no frame yet (%.0f s)" % (now - t_start), flush=True)
                    time.sleep(0.02)
                    continue
                got += 1
                if got == 1:
                    print("first frame: %dx%d" % (frame.shape[1], frame.shape[0]), flush=True)
                try:
                    cv2.imshow("camera_origin", draw_overlay(frame.copy(), tr, f))
                except cv2.error as e:
                    print("cannot show window (headless OpenCV?):", e, file=sys.stderr)
                    return 3
                if a.do_print and now - last_status >= 0.5:
                    last_status = now
                    print(f.format_line(), flush=True)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("c"):
                    print("calibrating 2 s, hold still ...", flush=True)
                    print("reference px:", tr.calibrate_reference(2.0, pump=tr.grab_once), flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            tr.stop()
            cv2.destroyAllWindows()
        return 0

    if not tr.start():
        return 2
    try:
        if a.calibrate > 0:
            print("calibrating origin for %.1f s ..." % a.calibrate)
            print("reference px:", tr.calibrate_reference(a.calibrate))
        last_print = 0.0
        while True:
            now = time.monotonic()
            if a.do_print and now - last_print >= 0.1:
                last_print = now
                f = tr.latest()
                print("no fix" if f is None else f.format_line(), flush=True)
            if a.show:
                frame = tr.last_frame
                f = tr.latest(1.0)
                if frame is not None and f is not None:
                    try:
                        cv2.imshow("camera_origin", draw_overlay(frame.copy(), tr, f))
                    except cv2.error as e:
                        print("cannot show window (headless OpenCV?):", e, file=sys.stderr)
                        return 3
                k = cv2.waitKey(15) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("c"):
                    print("calibrating 2 s, hold still ...")
                    print("reference px:", tr.calibrate_reference(2.0))
            else:
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        tr.stop()
        if a.show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
