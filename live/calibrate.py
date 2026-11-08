"""calibrate.py - knee brace calibration. Python 3.9+ standard library only.

leg_filter feeds a CalSession every RAW accelerometer sample (t host seconds, 6 floats in g, thigh xyz
then shank xyz, before invert/rot180); the session applies the bench correction itself.

  hang   Hang and Straighten (~35 s): hang still (knee = 90 deg) -> straighten as far as comfortable
         (measures the extension deficit and the bend sign) -> hang again (strap-slip check).
  anchor hold any angle the clinician measured; k = sign * r - anchor_deg (sign from the last hang run).
  bench  brace at 0 deg on the hinge, 4 orientations -> per-axis zero-g offset and gain (once per brace).

On success the display angle is theta = wrap(sign * r - k_deg), where r = raw_knee(apply_bench(bench, v6))
(same convention as knee_estimator.raw_knee); sign -1 is the same as "invert" (negate x of both sensors).
Holds are judged on 0.25 s block means: single samples sit at the 6-bit noise floor, and averaging the
VECTORS before taking angles lets the noise dither past the 2.7 deg quantization step.
"""
import json
import math
import os
import time
from collections import deque

BLOCK_S = 0.25          # samples are averaged per block; stillness is judged on block means
WINDOW_BLOCKS = 6       # 1.5 s rolling still window
HOLD_BLOCKS = 8         # 2 s of vectors averaged per hold
GAP_S = 0.2             # a gap in the data this long restarts the window
TIMEOUT_S = 15.0        # per step
AXIS_SD_G = 0.035       # still: per-axis sd of the block means
ANGLE_SD_DEG = 1.0      # still: knee-angle sd of the block means
SHIN_MAX_DEG = 12.0     # hanging shin may be this far from vertical
BEND_MIN_DEG = 25.0     # straighten must change the knee at least this much
BEND_NOTE_DEG = 8.0     # a still hold bent more than this (but < 25) gets the "straighten further" note
REHANG_GATE_DEG = 10.0  # a re-hang hold must be this close to the first hang to count
SLIP_MAX_DEG = 2.5      # re-hang vs hang
EXT_MIN_DEG = -12.0     # more hyperextension than this is implausible
OUT_OF_PLANE_G = 0.25   # y (hinge axis) reading above this: brace twisted
POSE_MIN_G = 0.7        # bench: the pose's gravity axis must read at least this
GAIN_RANGE = (0.8, 1.25)
COUNTS_PER_G = 21.33    # MMA7660
STALE_S = 4 * 3600.0    # a leg calibration older than this should be redone

STEPS = {"hang": ("hang", "straight", "rehang"), "anchor": ("hold",),
         "bench": ("pose1", "pose2", "pose3", "pose4")}
PROMPTS = {
    "hang": "Sit back, let your lower leg hang loose and stay still.",
    "straight": "Slowly straighten as far as is comfortable and hold.",
    "rehang": "Let it hang loose again.",
    "hold": "Hold the knee at %.0f deg and stay still.",
    "pose1": "Bench check: brace at 0 deg, stand it on end, hip end UP, hold still.",
    "pose2": "Hip end DOWN, hold still.",
    "pose3": "Lay it on its FRONT edge, hold still.",
    "pose4": "Now the BACK edge down, hold still.",
}
MOVED = "Leg moved, hold still."
BEND = "Couldn't see the knee bend, straighten further."
TWISTED = "Brace twisted or toes turned out."
TIMEOUT_HINT = {"straight": BEND, "rehang": "Re-hang didn't match: strap may have slipped, tighten and redo."}


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def tilt(x, z):
    return math.degrees(math.atan2(x, z))


def raw_knee(v6):
    return wrap(tilt(v6[3], v6[5]) - tilt(v6[0], v6[2]))


def apply_bench(bench, v6):
    """Remove the bench check's zero-g offset and gain from a raw 6-vector (no-op without a bench)."""
    if not bench:
        return v6
    return tuple((v - o) / g for v, o, g in zip(v6, bench["offset"], bench["gain"]))


def _mean(vs):
    vs = list(vs)
    return tuple(sum(v[j] for v in vs) / len(vs) for j in range(6))


def _sd(xs):
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def _still(means):
    if any(_sd([m[j] for m in means]) >= AXIS_SD_G for j in range(6)):
        return False
    r0 = raw_knee(means[0])
    return _sd([wrap(raw_knee(m) - r0) for m in means]) < ANGLE_SD_DEG


class CalSession:
    """One calibration run. Public: state, prompt (what to do now, or the final text), message (last
    check message, "" if none), level ("info"|"warn"|"ok"), progress (0..1), result (dict on success)."""

    def __init__(self, mode="hang", anchor_deg=None, bench=None, sign=1):
        if mode == "anchor" and anchor_deg is None:
            raise ValueError("anchor mode needs anchor_deg")
        self.mode, self.anchor_deg, self.sign = mode, anchor_deg, 1 if sign >= 0 else -1
        self.bench = None if mode == "bench" else bench     # the bench check measures raw values
        self.steps = STEPS[mode]
        self.state = self.steps[0]
        self.prompt, self.message, self.result = self._step_prompt(), "", None
        self.holds = {}                                     # step -> mean 6-vector of its hold
        self._warn = ""
        self._blocks = deque(maxlen=HOLD_BLOCKS)
        self._sum, self._n = [0.0] * 6, 0
        self._t_block = self._t_last = self._t_step = None
        self._run = 0                                       # consecutive still blocks in this hold

    @property
    def done(self):
        return self.state in ("done", "failed")

    @property
    def level(self):
        return "ok" if self.state == "done" else "warn" if self.state == "failed" or self.message else "info"

    @property
    def progress(self):
        if self.state == "done":
            return 1.0
        i = self.steps.index(self.state) if self.state in self.steps else 0
        return min(1.0, (i + self._run / HOLD_BLOCKS) / len(self.steps))

    def feed(self, t, v6):
        """One raw sample. Returns True when state, prompt or message changed (print an M line)."""
        if self.done:
            return False
        before = (self.state, self.prompt, self.message)
        v6 = apply_bench(self.bench, v6)
        if self._t_step is None:
            self._t_step = t
        if self._t_last is not None and t - self._t_last > GAP_S:
            self._blocks.clear()
            self._sum, self._n, self._t_block, self._run = [0.0] * 6, 0, None, 0
        self._t_last = t
        if self._t_block is None:
            self._t_block = t
        elif t - self._t_block >= BLOCK_S and self._n:
            self._end_block(t)
            self._t_block = t
        self._n += 1
        for j in range(6):
            self._sum[j] += v6[j]
        if not self.done and t - self._t_step > TIMEOUT_S:
            self._fail("Timed out: " + (self.message or TIMEOUT_HINT.get(self.state, MOVED)))
        return (self.state, self.prompt, self.message) != before

    # ---- holds ----
    def _end_block(self, t):
        self._blocks.append(tuple(s / self._n for s in self._sum))
        self._sum, self._n = [0.0] * 6, 0
        if len(self._blocks) < WINDOW_BLOCKS:
            return
        window = list(self._blocks)[-WINDOW_BLOCKS:]
        if not _still(window):
            if self._run:
                self.message = MOVED
            self._run = 0
            return
        ok, note = self._gate(_mean(window))
        if not ok:
            self._run = 0
            if note:
                self.message = note
            return
        if not self._run:
            self.message = self._warn                     # a new hold clears "Leg moved"
        self._run = self._run + 1 if self._run else WINDOW_BLOCKS
        if self._run >= HOLD_BLOCKS:      # every block in the deque is now part of a still window
            hold = _mean(self._blocks)
            self._blocks.clear()
            self._run = 0
            self._capture(hold, t)

    def _gate(self, m):
        """Is this still window the pose the current step waits for? (ok, note or None)"""
        if self.state == "straight":
            bend = abs(wrap(raw_knee(m) - raw_knee(self.holds["hang"])))
            return bend >= BEND_MIN_DEG, BEND if BEND_NOTE_DEG < bend < BEND_MIN_DEG else None
        if self.state == "rehang":
            return abs(wrap(raw_knee(m) - raw_knee(self.holds["hang"]))) < REHANG_GATE_DEG, None
        if self.state.startswith("pose"):
            i = int(self.state[-1])
            axis = 0 if i <= 2 else 2                       # x (along the limb) or z (front/back)
            prev = self.holds.get("pose%d" % (i - 1)) if i in (2, 4) else None
            for j in (axis, axis + 3):
                if abs(m[j]) < POSE_MIN_G or (prev and m[j] * prev[j] > 0):
                    return False, None                      # wrong pose, or not turned over yet
        return True, None

    def _capture(self, hold, t):
        self.holds[self.state] = hold
        if self.state in ("hang", "straight", "hold") and max(abs(hold[1]), abs(hold[4])) > OUT_OF_PLANE_G:
            self._warn = TWISTED                            # a warning only: the y axis is not used
        err = getattr(self, "_after_" + self.state, lambda h: None)(hold)
        if err:
            self._fail(err)
        elif not self.done:
            self.state = self.steps[self.steps.index(self.state) + 1]
            self.prompt, self.message, self._t_step = self._step_prompt(), self._warn, t

    # ---- per-step checks and solvers; each returns a failure message or None ----
    def _after_hang(self, h):
        shin = math.degrees(math.atan2(abs(h[5]), abs(h[3])))     # hanging shin: gravity along x
        if shin > SHIN_MAX_DEG:
            return ("Shin reads %.0f deg from vertical: use Anchor mode if your knee can't bend "
                    "that far." % shin)
        return None

    def _after_straight(self, h):
        r_hang, r_str = raw_knee(self.holds["hang"]), raw_knee(h)
        self.sign = 1 if wrap(r_hang - r_str) > 0 else -1          # flexion reads positive
        self._k = wrap(self.sign * r_hang - 90.0)
        self._ext = wrap(self.sign * r_str - self._k)
        if self._ext < EXT_MIN_DEG:
            return ("%.0f deg past straight is implausible: check the seat and that the shin hung "
                    "freely." % -self._ext)
        return None

    def _after_rehang(self, h):
        rep = abs(wrap(raw_knee(h) - raw_knee(self.holds["hang"])))
        if rep > SLIP_MAX_DEG:
            return "Re-hang differs by %.1f deg: strap may have slipped, tighten and redo." % rep
        ext = "%.0f deg %s straight" % (abs(self._ext), "short of" if self._ext >= 0 else "past")
        self._finish(dict(sign=self.sign, k_deg=round(self._k, 2), ext_deg=round(self._ext, 2),
                          repeat_deg=round(rep, 2)),
                     "Calibrated: best extension %s, hang repeat %.1f deg." % (ext, rep))
        return None

    def _after_hold(self, h):
        k = wrap(self.sign * raw_knee(h) - self.anchor_deg)
        self._finish(dict(sign=self.sign, k_deg=round(k, 2), ext_deg=None, repeat_deg=None),
                     "Calibrated at the anchor angle (%.0f deg)." % self.anchor_deg)
        return None

    def _after_pose4(self, _):
        up, down, front, back = (self.holds["pose%d" % i] for i in (1, 2, 3, 4))
        offset, gain = [], []
        for j in range(6):
            if j % 3 == 1:              # out-of-plane axis: ~0 g in every pose
                offset.append(sum(p[j] for p in (up, down, front, back)) / 4.0)
                gain.append(1.0)
                continue
            a, b = (up, down) if j % 3 == 0 else (front, back)
            offset.append((a[j] + b[j]) / 2.0)
            # in-plane gain: the axis's swing over both pose pairs (exact for a sensor tilted in the plane)
            gain.append(math.hypot((up[j] - down[j]) / 2.0, (front[j] - back[j]) / 2.0))
        if not all(GAIN_RANGE[0] <= g <= GAIN_RANGE[1] for g in gain):
            return "Bench poses unclear, redo the Bench check."
        worst = max(abs(o) for o in offset) * COUNTS_PER_G
        self._finish(dict(offset=[round(o, 4) for o in offset], gain=[round(g, 4) for g in gain]),
                     "Bench check done: largest zero-g offset %.1f counts." % worst)
        return None

    # ---- state ----
    def _step_prompt(self):
        p = PROMPTS[self.state]
        return p % self.anchor_deg if self.state == "hold" else p

    def _finish(self, result, text):
        if self._warn:
            text += " " + self._warn
        result.update(text=text, t=time.time())
        self.state, self.prompt, self.message, self.result = "done", text, self._warn, result

    def _fail(self, msg):
        self.state, self.prompt, self.message = "failed", msg, msg


def load(path):
    """Parsed JSON object from path, or None when missing or unreadable."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def save(path, d):
    """Write JSON atomically (tmp + os.replace)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def is_fresh(leg_cal):
    """A per-leg record ({sign, k_deg, ext_deg, t}) younger than STALE_S."""
    return bool(leg_cal) and time.time() - float(leg_cal.get("t", 0.0)) < STALE_S
