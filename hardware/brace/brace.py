"""Parametric hinged knee-brace side bars (CadQuery).

FULLY PRINTED: no screws, nuts or bought hardware. Two flat side bars joined at the knee by a
printed axle (part of the shin bar) and a printed snap clip.
The bars sit in two stacked layers (shin bar against the leg, thigh bar on top),
so they swing past each other through any flexion angle. Each bar carries one
Grove 3-Axis Digital Accelerometer (+-16g, 20x20 mm) in a keyed pocket so the
sensor axes are always parallel to the bar (held by two printed push-pins), and is held on the
leg by two fabric straps through edge slots, tightened with printed tri-glide sliders. The thigh
bar ends in a deck with snap pegs for a Raspberry Pi 4. Printed cable clips replace zip ties.

Assembly frame (used for all modelling):
  origin = hinge axis, on the leg-facing face of the shin bar
  +Z     = outward, away from the leg
  +Y     = toward the hip (thigh bar points +Y; shin bar points -Y when the leg is straight)
  +X     = anterior (front of the leg) for a RIGHT-leg brace worn on the outer side
Knee flexion turns the shin bar clockwise seen from +Z (right leg); "left" mirrors X.

Printed on 7 Nov 2026 (MakeUC): right leg, thigh bar WITHOUT the Pi deck, i.e. Params(pi_mount_enabled=False);
see out/right_thigh_nodeck.step and the out/fast_*_pla.gcode.3mf print files.

Run:  .venv/bin/python brace.py      -> out/{right,left}_{thigh,shin}.stl / .step
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cadquery as cq


# --------------------------------------------------------------------------- params
@dataclass
class Params:
    side: str = "right"  # "right" or "left" leg (brace on the outer side of the knee)

    # Bars
    bar_t: float = 5.0  # thickness of each bar (Z)
    bar_w: float = 26.0  # body width
    thigh_len: float = 200.0  # hinge center -> bar tip (the Pi deck extends past it)
    shin_len: float = 198.0
    outline_fillet: float = 3.0  # rounding of every corner of the outline (vertical edges)
    top_fillet: float = 1.2  # outer-face perimeter
    bottom_chamfer: float = 1.0  # leg-face perimeter: softer on the skin, prints as a 45 deg chamfer

    # Hinge: a 12 mm axle printed as part of the shin bar; the thigh bar spins on it and a printed
    # C-clip in a groove keeps it on. Nothing to buy, nothing to adjust.
    disc_r: float = 18.0  # hinge disc radius (both bars)
    axle_d: float = 12.0
    axle_base_fillet: float = 1.0  # where the axle meets the bearing ring (stress relief)
    axle_bore_clear: float = 0.4  # thigh bore = axle_d + this (running fit after FDM shrink)
    bore_chamfer: float = 1.3  # bottom of the thigh bore, clears the axle's base fillet
    clip_t: float = 1.6  # C-clip thickness (8 layers)
    groove_depth: float = 1.0  # radial
    groove_gap: float = 0.1  # end play between the thigh bar and the clip
    groove_clear: float = 0.25  # groove width = clip_t + this
    axle_tip: float = 4.0  # axle above the groove: the knee marker cap pushes onto it
    clip_od: float = 17.0
    clip_id_interf: float = 0.2  # clip ID = groove diameter - this (snug)
    clip_mouth: float = 8.2  # opening; spreads to the 10 mm groove diameter when snapped on
    bearing_ring_od: float = 20.0  # raised ring on the shin disc; the thigh disc rides on it
    bearing_ring_h: float = 0.6
    neck_fillet: float = 9.0  # big fillet where each bar meets its hinge disc (main crack risk)

    # Optional hyperextension stop (off by default: see README)
    stop_enabled: bool = False
    hyperext_deg: float = 15.0
    stop_clear: float = 1.2
    stop_depth: float = 5.0
    stop_reach: float = 27.0

    # Straps: ~1" fabric strips. Each strap station has a slot on the front and back edge of the
    # bar, running ALONG the bar so the strap passes through flat while it wraps the leg. Each strap
    # end loops through a slot and back through a printed tri-glide slider (no sewing, adjustable).
    strap_w: float = 25.4
    strap_slot_extra: float = 3.0  # slot length = strap_w + this
    strap_slot_h: float = 4.0  # slot opening across the bar (strap thickness ~2-3 mm + slack)
    strap_wall: float = 4.0  # material outside each slot
    strap_slot_chamfer: float = 0.5  # edge break on the slot mouths, both faces
    thigh_strap_pos: tuple = (75.0, 160.0)  # distance from hinge center
    shin_strap_pos: tuple = (75.0, 160.0)

    # Sensor: Seeed Grove 3-Axis Digital Accelerometer (+-16g), SKU 101020054, PCB v1.3
    # 20.0 x 20.0 mm, 1.6 mm thick; two M2 (dia 2.2) holes at the midpoints of the two edges
    # adjacent to the connector edge, each in an R1.92 ear (23.9 mm across the ears);
    # an R~2.4 notch centred on the edge opposite the connector.
    sensor_pos: float = 118.0  # hinge center -> pocket center
    pcb: float = 20.0
    pcb_hole_span: float = 20.0
    pcb_ear_r: float = 1.95
    sensor_clear: float = 0.25  # per side
    pocket_depth: float = 1.2  # of the 1.6 mm PCB, 0.4 mm stands proud
    pocket_relief: float = 0.5  # extra depth under the middle of the board
    key_enabled: bool = True  # bump that fits the PCB notch, so the board only fits one way round
    key_r: float = 1.9
    screw_pilot_d: float = 1.85  # blind hole for the printed push-pin (tight press fit)
    pilot_floor: float = 0.8  # pilots stop this far above the leg face (blind)
    pin_shank_d: float = 2.0  # through the PCB's 2.2 mm hole into the pilot
    pin_head_d: float = 4.2
    pin_head_t: float = 1.0
    sensor_ear_w: float = 32.0
    sensor_ear_len: float = 28.0
    conn_notch_w: float = 13.0  # clearance for the Grove plug on the connector edge (+Y)
    conn_notch_len: float = 6.0

    # Cable clip anchors (pairs of slots straddling the cable), distance from hinge center
    thigh_cable_ties: tuple = (40.0, 96.0, 150.0)  # 150: strain relief for the thigh plug
    shin_cable_ties: tuple = (40.0, 96.0)
    tie_slot_len: float = 4.2  # along the bar
    tie_slot_w: float = 3.0  # across the bar (clip leg 1.4 + room to flex past its barb)
    tie_gap: float = 9.0  # center spacing of the two slots in a pair
    tie_channel_depth: float = 1.2  # recess on the leg face: the clip barbs sit flush in it
    tie_channel_margin: float = 1.2  # channel extends this far past the slots' outer edges
    cclip_leg: float = 1.4
    cclip_barb: float = 1.4  # outward from the leg; hooks 0.6 mm past the slot edge
    cclip_band: float = 3.8  # clip thickness along the bar (fits the 4.2 slot)
    cclip_arch_h: float = 6.5  # room under the arch for two stacked Grove cables
    cclip_t: float = 1.6

    # Protractor ticks on the thigh disc + pointer line on the shin bar (bench calibration)
    ticks_enabled: bool = True
    tick_max: int = 150
    tick_w: float = 0.8  # two 0.4 mm lines
    tick_depth: float = 0.4
    tick_major_len: float = 5.0  # every 30 deg
    tick_minor_len: float = 2.8

    # Knee ArUco marker cap (pushes onto the tip of the hinge axle)
    marker_size: float = 40.0  # printed ArUco square incl. its black border
    cap_size: float = 46.0
    cap_plate_t: float = 3.0
    cap_boss_d: float = 18.0
    cap_bore_clear: float = 0.5  # pocket = axle_d + this; crush ribs give the grip
    cap_rib_r: float = 0.6
    cap_rib_proud: float = 0.45
    cap_pocket_depth: float = 3.6  # keeps the cap's rim clear of the C-clip

    # Raspberry Pi 4 deck: the thigh bar widens past the upper strap into a deck with four snap pegs.
    # Pi 4: 85 x 56 mm board, M2.5 (2.7 mm) holes on a 58 x 49 mm pattern, 3.5 mm in from one end
    # and 23.5 mm from the USB end.
    pi_mount_enabled: bool = True
    deck_y0: float = 176.0  # deck starts just past the upper thigh strap (strap edge at 172.7)
    deck_len: float = 100.0
    deck_w: float = 66.0
    pi_center_y: float = 222.0  # centre of the hole pattern; USB end toward the hip (board 189.5-274.5)
    pi_hole_x: float = 49.0
    pi_hole_y: float = 58.0
    standoff_d: float = 6.5
    standoff_h: float = 5.0  # Pi underside clearance above the deck
    peg_d: float = 2.4  # into the Pi's 2.7 mm hole
    peg_barb_d: float = 3.1
    peg_barb_h: float = 1.4
    peg_slot_w: float = 0.8  # split so the two halves flex
    pcb_t_pi: float = 1.6

    # Tri-glide strap slider (two per strap)
    tg_slot_len: float = 27.5  # fabric width 25.4 + slack
    tg_slot_w: float = 4.5  # fits a doubled ~2 mm fabric; raise for thicker fabric
    tg_bar: float = 3.5
    tg_t: float = 3.0

    # Markings (debossed on the outer face)
    label_depth: float = 0.6
    label_size: float = 7.0
    label_pos: float = 75.0  # centered between the first pair of strap slots
    arrow_pos: float = 48.0

    @property
    def s(self) -> int:
        return 1 if self.side == "right" else -1

    @property
    def strap_slot_len(self) -> float:
        return self.strap_w + self.strap_slot_extra

    @property
    def ear_w(self) -> float:
        return self.bar_w + 2 * (self.strap_slot_h + self.strap_wall)

    @property
    def strap_ear_len(self) -> float:
        return self.strap_slot_len + 8.0

    @property
    def thigh_z0(self) -> float:
        return self.bar_t + self.bearing_ring_h


# --------------------------------------------------------------------------- slab
def _outline(p: Params, length: float, d: int, strap_pos, extra_rects=()) -> cq.Sketch:
    """Bar outline in XY (sharp corners; rounded later in 3D). d=+1 -> along +Y, d=-1 -> -Y."""
    end_r = p.bar_w / 2
    body = length - end_r
    sk = (
        cq.Sketch()
        .push([(0, d * body / 2)]).rect(p.bar_w, body).reset()
        .push([(0, 0)]).circle(p.disc_r).reset()
        .push([(0, d * body)]).circle(end_r).reset()
        .push([(0, d * p.sensor_pos)]).rect(p.sensor_ear_w, p.sensor_ear_len).reset()
    )
    for pos in strap_pos:
        sk = sk.push([(0, d * pos)]).rect(p.ear_w, p.strap_ear_len).reset()
    for (cx, cy), w, h, ang in extra_rects:
        sk = sk.push([(cx, cy)]).rect(w, h, angle=ang).reset()
    return sk.clean()


def _slab(p: Params, length: float, d: int, strap_pos, z0: float, extra_rects=()) -> cq.Workplane:
    slab = cq.Workplane("XY").workplane(offset=z0).placeSketch(_outline(p, length, d, strap_pos, extra_rects)).extrude(p.bar_t)
    # Round the outline's corners (vertical edges). Skip the tangent seam where the straight edge
    # meets the round tip; the sensor ear only steps out (sensor_ear_w - bar_w)/2, so its corners
    # get a smaller radius that fits twice in that step.
    tip_y = d * (length - p.bar_w / 2)
    ear_y = d * p.sensor_pos

    neck_y = d * math.sqrt(p.disc_r**2 - (p.bar_w / 2) ** 2)  # where the bar edge meets the disc

    def corners(wp, kind: str):
        out = []
        for e in wp.edges("|Z").vals():
            c = e.Center()
            if abs(c.y - tip_y) < 1e-3:
                continue
            if abs(c.y - neck_y) < 1e-3 and abs(abs(c.x) - p.bar_w / 2) < 1e-3:
                k = "neck"
            elif abs(abs(c.y - ear_y) - p.sensor_ear_len / 2) < 1e-3:
                k = "sensor"
            else:
                k = "other"
            if k == kind:
                out.append(e)
        return out

    slab = slab.newObject(corners(slab, "neck")).fillet(p.neck_fillet)
    slab = slab.newObject(corners(slab, "other")).fillet(p.outline_fillet)
    small = min(p.outline_fillet, 0.45 * (p.sensor_ear_w - p.bar_w) / 2)
    slab = slab.newObject(corners(slab, "sensor")).fillet(small)
    slab = slab.faces("<Z").edges().chamfer(p.bottom_chamfer)
    slab = slab.faces(">Z").edges().fillet(p.top_fillet)
    return slab


# --------------------------------------------------------------------------- features
def _stadium(x, y, len_x, len_y, z0=-50.0, h=100.0) -> cq.Workplane:
    wp = cq.Workplane("XY").workplane(offset=z0).center(x, y)
    wp = wp.slot2D(len_x, len_y, 0) if len_x >= len_y else wp.slot2D(len_y, len_x, 90)
    return wp.extrude(h)


def _strap_slots(bar, p: Params, d: int, strap_pos):
    x = p.bar_w / 2 + p.strap_slot_h / 2
    for pos in strap_pos:
        for sx in (-1, 1):
            cutter = (
                cq.Workplane("XY").workplane(offset=-50).center(sx * x, d * pos)
                .rect(p.strap_slot_h, p.strap_slot_len).extrude(100)
                .edges("|Z").fillet(1.0)
            )
            bar = bar.cut(cutter)
    return bar


def _chamfer_strap_slots(bar, p: Params, d: int, strap_pos, z_leg: float):
    """Break the top and bottom edges of each strap slot so the strap isn't chewed."""
    x = p.bar_w / 2 + p.strap_slot_h / 2
    hx, hy = p.strap_slot_h / 2 + 0.05, p.strap_slot_len / 2 + 0.05
    for pos in strap_pos:
        for sx in (-1, 1):
            cx, cy = sx * x, d * pos
            for z in (z_leg, z_leg + p.bar_t):
                sel = cq.selectors.BoxSelector((cx - hx, cy - hy, z - 0.05), (cx + hx, cy + hy, z + 0.05))
                bar = bar.edges(sel).chamfer(p.strap_slot_chamfer)
    return bar


def _cable_ties(bar, p: Params, d: int, positions, z_leg: float):
    for pos in positions:
        for sx in (-1, 1):
            bar = bar.cut(_stadium(sx * p.tie_gap / 2, d * pos, p.tie_slot_w, p.tie_slot_len))
        cw = p.tie_gap + p.tie_slot_w + 2 * p.tie_channel_margin
        channel = _stadium(0, d * pos, cw, p.tie_slot_len, z0=z_leg - 1, h=p.tie_channel_depth + 1)
        bar = bar.cut(channel)
    return bar


def _sensor_mount(bar, p: Params, z_leg: float, z_top: float, d: int):
    """Keyed pocket for the Grove PCB. The connector edge faces +Y (toward the hip) on both bars,
    so both sensors share one orientation when the leg is straight."""
    yc = d * p.sensor_pos
    a = p.pcb + 2 * p.sensor_clear
    ear_r = p.pcb_ear_r + p.sensor_clear
    hx = p.pcb_hole_span / 2  # holes sit at x = +-10 (edges adjacent to the connector edge)
    z_floor = z_top - p.pocket_depth

    pocket = cq.Workplane("XY").workplane(offset=z_floor).center(0, yc).rect(a, a).extrude(p.pocket_depth + 1)
    pocket = pocket.edges("|Z").fillet(0.5)
    for sx in (-1, 1):
        pocket = pocket.union(
            cq.Workplane("XY").workplane(offset=z_floor).center(sx * hx, yc).circle(ear_r).extrude(p.pocket_depth + 1)
        )
    bar = bar.cut(pocket)

    relief = (
        cq.Workplane("XY").workplane(offset=z_floor - p.pocket_relief)
        .center(0, yc).rect(a - 5, a - 5).extrude(p.pocket_relief + 0.01)
    )
    bar = bar.cut(relief)

    notch = (
        cq.Workplane("XY").workplane(offset=z_floor)
        .center(0, yc + a / 2 + p.conn_notch_len / 2 - 0.5).rect(p.conn_notch_w, p.conn_notch_len)
        .extrude(p.pocket_depth + 1)
    )
    bar = bar.cut(notch)

    if p.key_enabled:  # sits in the PCB's notch on the edge opposite the connector (-Y side)
        key = (
            cq.Workplane("XY").workplane(offset=z_floor - 0.01)
            .center(0, yc - p.pcb / 2 - p.sensor_clear).circle(p.key_r).extrude(p.pocket_depth + 0.01)
        )
        bar = bar.union(key)

    for sx in (-1, 1):  # blind M2 pilots: never reach the leg face
        pilot = (
            cq.Workplane("XY").workplane(offset=z_leg + p.pilot_floor)
            .center(sx * hx, yc).circle(p.screw_pilot_d / 2).extrude(z_top - z_leg)
        )
        bar = bar.cut(pilot)
    return bar


def _label(bar, p: Params, text: str, z_top: float, y: float):
    cutter = (
        cq.Workplane("XY").workplane(offset=z_top - p.label_depth)
        .center(0, y).transformed(rotate=(0, 0, 90))
        .text(text, p.label_size, p.label_depth + 1, kind="bold", halign="center", valign="center")
    )
    return bar.cut(cutter)


def _arrow(bar, p: Params, z_top: float, y: float):
    """Debossed triangle pointing anterior (+X right leg, -X left leg)."""
    s = p.s
    tri = (
        cq.Workplane("XY").workplane(offset=z_top - p.label_depth)
        .polyline([(s * 8.0, y), (s * 1.0, y + 4.5), (s * 1.0, y - 4.5)]).close()
        .extrude(p.label_depth + 1)
    )
    return bar.cut(tri)


def _stop_geometry(p: Params):
    """Optional hyperextension stop: a pillar on a lobe of the shin disc that meets the thigh bar's
    anterior edge when the shin swings `hyperext_deg` past straight. Laid out at the stop pose,
    then rotated back into the shin's rest frame. Returns (pillar polygon, lobe rect)."""
    s = p.s
    half = p.bar_w / 2
    r_in = p.disc_r + p.stop_clear
    x0, x1 = half, half + p.stop_depth
    arc = []
    for i in range(11):
        x = x0 + (x1 - x0) * i / 10
        arc.append((x, math.sqrt(max(r_in**2 - x**2, 0))))
    poly = [(x0, p.stop_reach), *arc, (x1, p.stop_reach)]
    a_deg = -p.hyperext_deg * s
    ca, sa = math.cos(math.radians(a_deg)), math.sin(math.radians(a_deg))

    def to_rest(x, y):
        x = s * x
        return (x * ca - y * sa, x * sa + y * ca)

    poly_r = [to_rest(x, y) for x, y in poly]
    if s < 0:
        poly_r = poly_r[::-1]
    m = 1.5
    lx0, lx1, ly0, ly1 = x0 - m, x1 + m, 3.0, p.stop_reach + m
    lobe = (to_rest((lx0 + lx1) / 2, (ly0 + ly1) / 2), lx1 - lx0, ly1 - ly0, a_deg)
    return poly_r, lobe


def _stop_pillar(p: Params) -> cq.Workplane:
    poly_r, _ = _stop_geometry(p)
    z_top = p.thigh_z0 + p.bar_t
    z_bot = p.bar_t - 1.0
    pillar = cq.Workplane("XY").workplane(offset=z_bot).polyline(poly_r).close().extrude(z_top - z_bot)
    pillar = pillar.edges("|Z").fillet(0.6)
    return pillar.faces(">Z").edges().chamfer(0.5)


# --------------------------------------------------------------------------- parts
def shin_bar(p: Params) -> cq.Workplane:
    """Shin bar: leg-side layer, z in [0, bar_t], points -Y."""
    t = p.bar_t
    extra = [_stop_geometry(p)[1]] if p.stop_enabled else []
    bar = _slab(p, p.shin_len, -1, p.shin_strap_pos, 0.0, extra)
    bar = bar.union(cq.Workplane("XY").workplane(offset=t - 0.01).circle(p.bearing_ring_od / 2).extrude(p.bearing_ring_h + 0.01))
    if p.stop_enabled:
        bar = bar.union(_stop_pillar(p))
    bar = _axle(bar, p)
    bar = _strap_slots(bar, p, -1, p.shin_strap_pos)
    bar = _chamfer_strap_slots(bar, p, -1, p.shin_strap_pos, 0.0)
    bar = _cable_ties(bar, p, -1, p.shin_cable_ties, 0.0)
    bar = _sensor_mount(bar, p, 0.0, t, -1)
    bar = _label(bar, p, ("R" if p.s > 0 else "L") + " SHIN", t, -p.label_pos)
    bar = _arrow(bar, p, t, -p.arrow_pos)
    if p.ticks_enabled:
        bar = _shin_pointer(bar, p, t)
    return bar


def thigh_bar(p: Params) -> cq.Workplane:
    """Thigh bar: outer layer, z in [thigh_z0, thigh_z0 + bar_t], points +Y."""
    z0, t = p.thigh_z0, p.bar_t
    deck = [((0, p.deck_y0 + p.deck_len / 2), p.deck_w, p.deck_len, 0)] if p.pi_mount_enabled else []
    bar = _slab(p, p.thigh_len, +1, p.thigh_strap_pos, z0, deck)
    bore = p.axle_d + p.axle_bore_clear
    bar = bar.cut(cq.Workplane("XY").circle(bore / 2).extrude(100, both=True))
    c = p.bore_chamfer  # conical lead-in at the bottom of the bore (clears the axle's base fillet)
    bar = bar.cut(cq.Workplane("XY").add(cq.Solid.makeCone(bore / 2 + c, bore / 2, c, pnt=cq.Vector(0, 0, z0 - 0.01))))
    bar = _strap_slots(bar, p, +1, p.thigh_strap_pos)
    bar = _chamfer_strap_slots(bar, p, +1, p.thigh_strap_pos, z0)
    bar = _cable_ties(bar, p, +1, p.thigh_cable_ties, z0)
    bar = _sensor_mount(bar, p, z0, z0 + t, +1)
    bar = _label(bar, p, ("R" if p.s > 0 else "L") + " THIGH", z0 + t, p.label_pos)
    bar = _arrow(bar, p, z0 + t, p.arrow_pos)
    if p.ticks_enabled:
        bar = _hinge_ticks(bar, p, z0 + t)
    if p.pi_mount_enabled:
        bar = _pi_deck_features(bar, p, z0 + t)
    return bar


def _axle_levels(p: Params):
    thigh_top = p.thigh_z0 + p.bar_t
    g0 = thigh_top + p.groove_gap  # clip sits on the thigh bar's top with this end play
    g1 = g0 + p.clip_t + p.groove_clear
    top = g1 + p.axle_tip
    return g0, g1, top


def _axle(bar, p: Params):
    """12 mm axle rising from the shin bar's bearing ring, with a groove for the C-clip."""
    g0, g1, top = _axle_levels(p)
    z_ring = p.bar_t + p.bearing_ring_h
    axle = cq.Workplane("XY").workplane(offset=z_ring - 0.01).circle(p.axle_d / 2).extrude(top - z_ring + 0.01)
    axle = axle.faces(">Z").edges().chamfer(0.8)
    bar = bar.union(axle)
    r = p.axle_d / 2
    f = p.axle_base_fillet  # 45 deg collar at the axle's base (stress relief)
    bar = bar.union(cq.Workplane("XY").add(cq.Solid.makeCone(r + f, r - 0.01, f, pnt=cq.Vector(0, 0, z_ring - 0.01))))
    groove = (
        cq.Workplane("XY").workplane(offset=g0).circle(r + 1).circle(r - p.groove_depth).extrude(g1 - g0)
    )
    return bar.cut(groove)


def _pi_deck_features(bar, p: Params, z_top: float):
    """Four standoffs with split, barbed snap pegs for the Pi 4, and a window under the board."""
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * p.pi_hole_x / 2, p.pi_center_y + sy * p.pi_hole_y / 2
            z_so = z_top + p.standoff_h
            so = cq.Workplane("XY").workplane(offset=z_top - 0.01).center(x, y).circle(p.standoff_d / 2).extrude(p.standoff_h + 0.01)
            peg = cq.Workplane("XY").workplane(offset=z_so - 0.01).center(x, y).circle(p.peg_d / 2).extrude(p.pcb_t_pi + 0.2 + 0.01)
            barb = cq.Workplane("XY").add(
                cq.Solid.makeCone(p.peg_barb_d / 2, p.peg_d / 2 - 0.25, p.peg_barb_h, pnt=cq.Vector(x, y, z_so + p.pcb_t_pi + 0.2))
            )
            bar = bar.union(so).union(peg).union(barb)
            slot = (
                cq.Workplane("XY").workplane(offset=z_so - 0.6).center(x, y)
                .rect(p.peg_slot_w, p.peg_barb_d + 1).extrude(p.pcb_t_pi + p.peg_barb_h + 2)
            )
            bar = bar.cut(slot)
    win = (
        cq.Workplane("XY").workplane(offset=-50).center(0, p.pi_center_y + 2)
        .rect(p.pi_hole_x - 16, p.pi_hole_y - 20).extrude(100).edges("|Z").fillet(4.0)
    )
    return bar.cut(win)


def axle_clip(p: Params) -> cq.Workplane:
    """C-clip for the axle groove, printed flat (flexes in the strong direction)."""
    groove_d = p.axle_d - 2 * p.groove_depth
    ri = (groove_d - p.clip_id_interf) / 2
    ro = p.clip_od / 2
    ring = cq.Workplane("XY").circle(ro).circle(ri).extrude(p.clip_t)
    # mouth: a slot from the centre out through one side, flared for a lead-in
    mouth = cq.Workplane("XY").workplane(offset=-1).polyline(
        [(-p.clip_mouth / 2, 0), (p.clip_mouth / 2, 0), (p.clip_mouth / 2 + 1.6, -ro - 1), (-p.clip_mouth / 2 - 1.6, -ro - 1)]
    ).close().extrude(p.clip_t + 2)
    clip = ring.cut(mouth)
    tab = cq.Workplane("XY").center(0, ro + 1.5).rect(6.0, 5.0).extrude(p.clip_t).edges("|Z").fillet(1.5)
    clip = clip.union(tab)
    hole = cq.Workplane("XY").center(0, ro + 2.0).circle(1.2).extrude(10, both=True)  # pick hole for removal
    return clip.cut(hole).edges("|Z").fillet(0.3)


def sensor_pin(p: Params) -> cq.Workplane:
    """Push-pin through the Grove board's mounting hole into the blind pilot (printed head-down)."""
    shank = p.pcb + 0  # placeholder to keep the name meaning clear below
    depth = (p.bar_t - p.pocket_depth) - p.pilot_floor  # pilot depth below the pocket floor
    shank_l = 1.6 + depth - 0.3
    head = cq.Workplane("XY").circle(p.pin_head_d / 2).extrude(p.pin_head_t)
    body = cq.Workplane("XY").workplane(offset=p.pin_head_t - 0.01).circle(p.pin_shank_d / 2).extrude(shank_l + 0.01)
    pin = head.union(body).faces(">Z").edges().chamfer(0.3)
    return pin


def cable_clip(p: Params) -> cq.Workplane:
    """Snap staple for a pair of cable-anchor slots. Modelled in the bar's cross-section (x = across
    the bar, y = height from the leg face), extruded along the bar; printed lying flat like this."""
    t, h, L = p.bar_t, p.cclip_arch_h, p.cclip_leg
    xo = p.tie_gap / 2 + L / 2  # leg outer face sits at the slot's outer half
    xi = xo - L
    yb = 0.15  # barb bottom, just above the leg face (inside the channel)
    yh = p.tie_channel_depth - 0.05  # barb top: hooks against the channel ceiling
    top_in = t + h
    top_out = top_in + p.cclip_t
    half = [
        (xi, yb + 0.6), (xi, top_in), (0, top_in),
    ]
    right = [(0, top_out), (xo, top_out), (xo, yh), (xo + p.cclip_barb, yh), (xi + 0.2, yb)]
    pts = [(-x, y) for (x, y) in reversed(right)] + [(-xi, yb + 0.6), (-xi, top_in), (xi, top_in), (xi, yb + 0.6)] + right
    # build as a single closed outline: left barb -> left leg -> arch -> right leg -> right barb
    outline = [
        (-(xi + 0.2), yb), (-(xo + p.cclip_barb), yh), (-xo, yh), (-xo, top_out), (xo, top_out), (xo, yh),
        (xo + p.cclip_barb, yh), (xi + 0.2, yb), (xi, yb + 0.6), (xi, top_in), (-xi, top_in), (-xi, yb + 0.6),
    ]
    return cq.Workplane("XY").polyline(outline).close().extrude(p.cclip_band)


def tri_glide(p: Params) -> cq.Workplane:
    """Three-bar strap slider (printed flat). The centre bar stands 0.6 mm proud to bite the fabric."""
    L = p.tg_slot_len + 2 * p.tg_bar
    W = 3 * p.tg_bar + 2 * p.tg_slot_w
    body = cq.Workplane("XY").rect(L, W).extrude(p.tg_t).edges("|Z").fillet(2.0)
    for sy in (-1, 1):
        yc = sy * (p.tg_bar / 2 + p.tg_slot_w / 2)
        body = body.cut(cq.Workplane("XY").workplane(offset=-1).center(0, yc).rect(p.tg_slot_len, p.tg_slot_w).extrude(p.tg_t + 2).edges("|Z").fillet(1.0))
    ridge = cq.Workplane("XY").workplane(offset=p.tg_t - 0.01).rect(p.tg_slot_len, p.tg_bar - 1.0).extrude(0.61)
    body = body.union(ridge)
    return body.faces("<Z").edges().chamfer(0.3)


def _hinge_ticks(bar, p: Params, z_top: float):
    """Protractor on the thigh disc: a tick wherever the shin bar's centerline points at
    0, 10, 20 ... deg of flexion (long ticks every 30 deg). Read against the shin pointer line."""
    for deg in range(0, p.tick_max + 1, 10):
        phi = math.radians(-90.0 - p.s * deg)  # shin centerline direction at this flexion
        r0 = p.disc_r - (p.tick_major_len if deg % 30 == 0 else p.tick_minor_len)
        r1 = p.disc_r + 1.0  # run out through the rim so the tick reaches the edge
        mid = (r0 + r1) / 2
        cutter = (
            cq.Workplane("XY").workplane(offset=z_top - p.tick_depth)
            .center(mid * math.cos(phi), mid * math.sin(phi))
            .transformed(rotate=(0, 0, math.degrees(phi)))
            .rect(r1 - r0, p.tick_w).extrude(p.tick_depth + 1)
        )
        bar = bar.cut(cutter)
    return bar


def _shin_pointer(bar, p: Params, z_top: float):
    """Index line on the shin bar's centerline, just outside the thigh disc rim."""
    r0, r1 = p.disc_r + 0.6, p.disc_r + 6.0
    cutter = (
        cq.Workplane("XY").workplane(offset=z_top - p.tick_depth)
        .center(0, -(r0 + r1) / 2).rect(p.tick_w, r1 - r0).extrude(p.tick_depth + 1)
    )
    return bar.cut(cutter)


def knee_marker_cap(p: Params) -> cq.Workplane:
    """Flat plate that pushes onto the tip of the hinge axle and carries the ArUco knee marker.
    The axle sits over the lateral femoral epicondyle, so the marker centre = knee landmark.
    Modelled marker-face down (print orientation): z=0 is the marker face."""
    a, t = p.cap_size, p.cap_plate_t
    plate = cq.Workplane("XY").rect(a, a).extrude(t).edges("|Z").fillet(2.0)
    plate = plate.faces("<Z").edges().chamfer(0.4)
    h = p.cap_pocket_depth
    boss = cq.Workplane("XY").workplane(offset=t - 0.01).circle(p.cap_boss_d / 2).extrude(h + 0.01)
    cap = plate.union(boss)
    rp = (p.axle_d + p.cap_bore_clear) / 2
    cap = cap.cut(cq.Workplane("XY").workplane(offset=t).circle(rp).extrude(h + 1))
    for ang in (90, 210, 330):  # crush ribs grip the axle tip
        a_r = math.radians(ang)
        rc = rp + p.cap_rib_r - p.cap_rib_proud
        rib = cq.Workplane("XY").workplane(offset=t - 0.01).center(rc * math.cos(a_r), rc * math.sin(a_r)).circle(p.cap_rib_r).extrude(h - 0.8)
        cap = cap.union(rib)
    # The marker face (z=0, on the bed) is left flat: glue the printed marker on, centred.
    return cap


def build(p: Params) -> dict[str, cq.Workplane]:
    parts = {"shin": shin_bar(p), "thigh": thigh_bar(p)}
    for name, wp in parts.items():
        solids = wp.solids().vals()
        if len(solids) != 1 or not solids[0].isValid():
            raise RuntimeError(f"{p.side} {name}: expected one valid solid, got {len(solids)}")
    return parts


def assembly_extras(p: Params) -> dict[str, cq.Workplane]:
    """Small printed parts positioned on the RIGHT-leg brace in the assembly frame (shin at 0 deg)."""
    g0, g1, top = _axle_levels(p)
    out = {}
    out["axle_clip"] = axle_clip(p).translate((0, 0, g0 + p.groove_clear / 2))
    cap = knee_marker_cap(p).rotate((0, 0, 0), (1, 0, 0), 180).translate((0, 0, top + p.cap_plate_t))
    out["knee_marker_cap"] = cap
    pins = []
    for d, z_top in ((-1, p.bar_t), (+1, p.thigh_z0 + p.bar_t)):
        for sx in (-1, 1):
            pin = sensor_pin(p).rotate((0, 0, 0), (1, 0, 0), 180).translate(
                (sx * p.pcb_hole_span / 2, d * p.sensor_pos, z_top + 0.4 + p.pin_head_t))
            pins.append(pin)
    allpins = pins[0]
    for q in pins[1:]:
        allpins = allpins.union(q)
    out["sensor_pins"] = allpins
    clips = []
    for d, z_leg, positions in ((-1, 0.0, p.shin_cable_ties), (+1, p.thigh_z0, p.thigh_cable_ties)):
        for pos in positions:
            c = cable_clip(p).translate((0, 0, -p.cclip_band / 2)).rotate((0, 0, 0), (1, 0, 0), 90)
            clips.append(c.translate((0, d * pos, z_leg)))
    allclips = clips[0]
    for q in clips[1:]:
        allclips = allclips.union(q)
    out["cable_clips"] = allclips
    return out


# --------------------------------------------------------------------------- export
def print_orientation(name: str, wp: cq.Workplane, p: Params) -> cq.Workplane:
    """Leg-facing face down on the bed, centered on the origin."""
    z = p.thigh_z0 if name == "thigh" else 0.0  # the leg face sits exactly at this height
    bb = wp.val().BoundingBox()
    return wp.translate((-(bb.xmin + bb.xmax) / 2, -(bb.ymin + bb.ymax) / 2, -z))


def export_all(out: Path, sides=("right", "left"), **overrides) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for side in sides:
        p = Params(side=side, **overrides)
        for name, wp in build(p).items():
            stem = f"{side}_{name}"
            cq.exporters.export(wp, str(out / f"{stem}_assembly_frame.stl"), tolerance=0.02, angularTolerance=0.1)
            pw = print_orientation(name, wp, p)
            cq.exporters.export(pw, str(out / f"{stem}.stl"), tolerance=0.02, angularTolerance=0.1)
            cq.exporters.export(pw, str(out / f"{stem}.step"))
            written[stem] = pw
    p0 = Params(**overrides)
    singles = {
        "knee_marker_cap": knee_marker_cap(p0),
        "axle_clip": axle_clip(p0),
        "sensor_pin": sensor_pin(p0),
        "cable_clip": cable_clip(p0),
        "tri_glide": tri_glide(p0),
    }
    for name, wp in singles.items():
        if len(wp.solids().vals()) != 1 or not wp.val().isValid():
            raise RuntimeError(f"{name}: expected one valid solid")
        bb = wp.val().BoundingBox()
        wp = wp.translate((-(bb.xmin + bb.xmax) / 2, -(bb.ymin + bb.ymax) / 2, -bb.zmin))
        cq.exporters.export(wp, str(out / f"{name}.stl"), tolerance=0.02, angularTolerance=0.1)
        cq.exporters.export(wp, str(out / f"{name}.step"))
        written[name] = wp
    for name, wp in assembly_extras(p0).items():  # small parts placed on the right-leg brace
        cq.exporters.export(wp, str(out / f"{name}_assembly_frame.stl"), tolerance=0.02, angularTolerance=0.1)
    return written


if __name__ == "__main__":
    for k, v in export_all(Path(__file__).parent / "out").items():
        bb = v.val().BoundingBox()
        print(f"{k:12s} {bb.xlen:6.1f} x {bb.ylen:6.1f} x {bb.zlen:5.1f} mm   {v.val().Volume() / 1000:5.1f} cm3")
