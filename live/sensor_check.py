#!/usr/bin/env python3
"""sensor_check.py - ask the Arduino for its own sensor self-check (the 'C' command).

Prints the board's '# thigh ... ok/FAIL' lines. Use it when the bridge connects
but no samples arrive. Close the Arduino Serial Monitor first (one reader per port).

    python3 live/sensor_check.py [--port /dev/cu.usbmodemXXXX]
"""
import argparse
import glob
import sys
import time

import serial


def read_for(s, secs):
    end, buf = time.time() + secs, b""
    while time.time() < end:
        buf += s.read(512)
    return buf.decode(errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    a = ap.parse_args()
    port = a.port or next(iter(sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*"))), None)
    if not port:
        sys.exit("No Arduino serial port found (is it plugged in with a data cable?)")
    try:
        s = serial.Serial(port, 115200, timeout=0.3)
    except (serial.SerialException, OSError) as e:
        sys.exit("Cannot open %s: %s\nClose the Serial Monitor / any other bridge (lsof %s)." % (port, e, port))
    print("port:", port)
    read_for(s, 2.5)  # Arduino resets when the port opens
    s.write(b"C")
    out = read_for(s, 2.0)
    s.close()
    lines = [l for l in out.splitlines() if l.startswith("#")]
    print("\n".join(lines) if lines else "(no reply - wrong sketch or port?)")
    bad = [l for l in lines if "FAIL" in l]
    print("\nRESULT:", "SENSOR PROBLEM - reseat Grove cables / power" if bad else "sensors OK")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
