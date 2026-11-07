#!/usr/bin/env python3
"""fake_pi.py - stand-in for the Pi. Prints raw accelerometer readings to stdout.

One line per sample, 100 Hz, comma-separated, in g:

    t_ms,thigh_x,thigh_y,thigh_z,shank_x,shank_y,shank_z

Gravity convention (what the C++ math expects): a sensor tilted by angle a in the
sagittal plane reads (sin a, 0, cos a). Knee angle = tilt(shank) - tilt(thigh).
Fake motion: knee swings 90 -> 0 -> 90 deg every 4 s (a seated leg extension).

To use the real Pi, replace this file with anything that prints lines in the same
format. Flush every line, or the C++ side will not see data until a buffer fills.
"""
import math
import random
import sys
import time

RATE_HZ = 100


def reading(tilt_deg):
    r = math.radians(tilt_deg)
    n = lambda: random.gauss(0, 0.01)  # sensor noise, in g
    return math.sin(r) + n(), n(), math.cos(r) + n()


t0 = time.monotonic()
next_t = t0
while True:
    t = time.monotonic() - t0
    knee = 45.0 * (1.0 + math.cos(2.0 * math.pi * t / 4.0))      # 90 -> 0 -> 90
    thigh_tilt = 2.0 * math.sin(2.0 * math.pi * t / 7.0)          # slight thigh sway
    row = (int(t * 1000),) + reading(thigh_tilt) + reading(thigh_tilt + knee)
    try:
        print("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % row, flush=True)
    except BrokenPipeError:  # C++ side closed the pipe
        break
    next_t += 1.0 / RATE_HZ
    time.sleep(max(0.0, next_t - time.monotonic()))
