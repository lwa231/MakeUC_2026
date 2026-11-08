#!/usr/bin/env python3
"""Print the sensor log readably at 10 Hz:  python3 live/watch_log.py build/session.csv"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knee_estimator import raw_knee, tilt  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "build/session.csv"
while True:
    try:
        f = open(path)
        break
    except OSError:
        time.sleep(0.5)
f.seek(0, 2)
print("   t_ms |  thigh x     y     z  |a| | shank x     y     z  |a| | thigh  shank  raw knee (deg)")
n = 0
while True:
    line = f.readline()
    if not line:
        time.sleep(0.01)
        continue
    try:
        t, tx, ty, tz, sx, sy, sz = (float(v) for v in line.split(",")[:7])
    except ValueError:
        continue
    n += 1
    if n % 10:
        continue
    th, sh, knee = tilt(tx, ty), tilt(sx, sy), raw_knee((tx, ty, tz, sx, sy, sz))
    print("%7.0f | %+5.2f %+5.2f %+5.2f %4.2f | %+5.2f %+5.2f %+5.2f %4.2f | %+6.1f %+6.1f %+7.1f" % (
        t, tx, ty, tz, math.sqrt(tx * tx + ty * ty + tz * tz), sx, sy, sz,
        math.sqrt(sx * sx + sy * sy + sz * sz), th, sh, knee), flush=True)
