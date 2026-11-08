#!/usr/bin/env python3
"""sim_arduino.py - a software stand-in for brace_sensors_3.ino (no hardware needed).

Creates a pseudo-terminal and behaves like the Arduino on the far end of the USB cable:
the FIRST line on stdout is the path of the pty slave (e.g. /dev/ttys012). Give that path to
the bridge as its serial port:

    python3 live/sim_arduino.py                       # prints /dev/ttysNNN, then runs until Ctrl+C
    python3 serial_bridge.py --port /dev/ttysNNN      # in another terminal

Firmware behaviour reproduced (see brace_sensors_3.ino):
  * 115200-baud style text protocol, lines end in CRLF (Arduino println).
  * Boot banner + self-check '#' status lines, streaming ON at power-up.
  * ~100 Hz sample lines  t_ms,thigh_x,thigh_y,thigh_z,shank_x,shank_y,shank_z  (g, 4 decimals,
    exact printG() integer formatting: 1 count = 0.046875 g printed as 0.0468 / 0.0937 / ...).
  * Commands (single characters): S = start streaming and reset t_ms to 0 (prints '# streaming'),
    X = stop ('# stopped'), C = self-check, I = bus scan. Other bytes are ignored.
  * MMA7660 6-bit quantisation: 21.33 counts/g, range -32..31 counts.
  * If a sensor is missing (--fail) the stream is suppressed, as in the firmware.

Motion (same convention as fake_pi.py): a seated leg extension, knee 90 -> 0 -> 90 deg every
4 s, slight thigh sway; a sensor tilted by angle a in the sagittal plane reads (sin a, 0, cos a)
plus noise. Physical motion keeps running when S resets t_ms.

Stdlib only. Ctrl+C / SIGTERM / --duration end it cleanly.
"""
import argparse
import math
import os
import random
import select
import signal
import sys
import time
import tty

COUNTS_PER_G = 64.0 / 3.0          # 21.33 counts per g
NAMES = ("thigh", "shank")
BANNER = "# brace_sensors ready (MMA7660). Commands: S=start X=stop C=check I=scan"


def millis():
    return time.monotonic_ns() // 1_000_000


def print_g(count):
    """Port of printG() in the firmware: integer math, 4 decimals, truncation toward zero."""
    v = abs(count) * 46875 // 100
    return "%s%d.%04d" % ("-" if count < 0 else "", v // 10000, v % 10000)


class Sim:
    def __init__(self, args):
        self.a = args
        self.rng = random.Random(args.seed)
        self.master, self.slave = os.openpty()
        tty.setraw(self.slave)                      # no echo, no CR/LF translation
        os.set_blocking(self.master, False)
        self.name = os.ttyname(self.slave)
        self.out = bytearray()
        self.streaming = True
        self.sensor_ok = [n != args.fail for n in NAMES]
        self.t_start = time.monotonic()
        self.t0_ms = millis()
        self.next_t = time.monotonic()
        self.period = 1.0 / args.rate
        self.cmdlog = open(args.cmd_log, "a", buffering=1) if args.cmd_log else None
        self.stop = False
        self.n_sent = 0

    # ------------------------------------------------------------ serial out ----
    def println(self, text):
        self.out += text.encode("ascii") + b"\r\n"

    def pump(self):
        """Push pending bytes into the pty without ever blocking; drop whole lines if backed up."""
        while self.out:
            try:
                n = os.write(self.master, self.out)
            except BlockingIOError:
                break
            except OSError:
                break
            del self.out[:n]

    # ------------------------------------------------------------- commands ----
    def self_check(self):
        for i, name in enumerate(NAMES):
            where = "(I2C port)" if i == 0 else "(D2/D3 soft I2C)"
            if self.sensor_ok[i]:
                self.println("# %s %s: ok (MMA7660 at 0x4C)" % (name, where))
            else:
                self.println("# %s %s: FAIL, no answer at 0x4C (check cable/port)" % (name, where))
        if not all(self.sensor_ok):
            self.scan()

    def scan(self):
        for i, label in enumerate(("# scan I2C port (thigh):", "# scan D2/D3 soft I2C (shank):")):
            self.println(label + (" 0x4C" if self.sensor_ok[i] else " nothing (check cable and port)"))
        self.println("# expected: 0x4C on both buses (MMA7660)")

    def command(self, c):
        if self.cmdlog:
            self.cmdlog.write("%.3f %s\n" % (time.monotonic() - self.t_start, c))
        if c == "S":
            self.streaming = True
            self.t0_ms = millis()
            self.next_t = time.monotonic()
            self.println("# streaming")
        elif c == "X":
            self.streaming = False
            self.println("# stopped")
        elif c == "C":
            self.self_check()
        elif c == "I":
            self.scan()

    def read_commands(self):
        try:
            data = os.read(self.master, 256)
        except (BlockingIOError, OSError):          # nothing yet / no peer open yet
            return
        for b in data:
            self.command(chr(b))

    # --------------------------------------------------------------- motion ----
    def sample(self):
        t = time.monotonic() - self.t_start
        knee = 45.0 * (1.0 + math.cos(2.0 * math.pi * t / self.a.knee_period))   # 90 -> 0 -> 90
        thigh = 2.0 * math.sin(2.0 * math.pi * t / 7.0)                           # slight sway
        counts = []
        for tilt in (thigh, thigh + knee):
            r = math.radians(tilt)
            for g in (math.sin(r), 0.0, math.cos(r)):
                g += self.rng.gauss(0.0, self.a.noise)
                counts.append(max(-32, min(31, int(round(g * COUNTS_PER_G)))))
        return "%d,%s" % (millis() - self.t0_ms, ",".join(print_g(c) for c in counts))

    # ----------------------------------------------------------------- main ----
    def run(self):
        self.println(BANNER)
        self.self_check()
        self.t0_ms = millis()
        self.next_t = time.monotonic()
        t_end = time.monotonic() + self.a.duration if self.a.duration > 0 else None
        while not self.stop:
            now = time.monotonic()
            if t_end is not None and now >= t_end:
                break
            live = self.streaming and all(self.sensor_ok) and not self.a.mute
            wait = max(0.0, self.next_t - now) if live else 0.05
            select.select([self.master], [], [], min(wait, 0.05))
            self.read_commands()
            self.pump()
            now = time.monotonic()
            if self.streaming and all(self.sensor_ok) and not self.a.mute and now >= self.next_t:
                self.next_t += self.period
                if now - self.next_t > self.period:          # fell behind: resync (firmware does too)
                    self.next_t = now + self.period
                if not self.out:                              # tty queue full / reader absent: drop the sample (a UART
                    # holds no backlog either), so a flush by the reader really discards all stale data
                    self.println(self.sample())
                    self.n_sent += 1
                self.pump()

    def close(self):
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass
        if self.cmdlog:
            self.cmdlog.close()


def main():
    ap = argparse.ArgumentParser(description="Simulated brace_sensors_3.ino on a pty (prints the pty path first).")
    ap.add_argument("--rate", type=float, default=100.0, help="samples per second (default 100, like the firmware)")
    ap.add_argument("--duration", type=float, default=0.0, help="exit after this many seconds (0 = run until killed)")
    ap.add_argument("--noise", type=float, default=0.01, help="sensor noise sigma in g before quantisation (default 0.01)")
    ap.add_argument("--knee-period", type=float, default=4.0, help="seconds per 90->0->90 rep (default 4)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (default: random)")
    ap.add_argument("--cmd-log", help="append every command byte received ('time char') to this file")
    ap.add_argument("--fail", choices=NAMES, help="pretend this sensor is missing (stream suppressed, like the firmware)")
    ap.add_argument("--mute", action="store_true", help="never stream, even after S (a hung board; for testing the bridge)")
    args = ap.parse_args()
    if args.rate <= 0:
        ap.error("--rate must be > 0")

    sim = Sim(args)
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, lambda *_: setattr(sim, "stop", True))
    try:
        print(sim.name, flush=True)                  # first line: the port for --port
    except BrokenPipeError:
        sim.close()
        return 1
    try:
        sim.run()
    finally:
        sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
