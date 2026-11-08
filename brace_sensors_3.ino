// brace_sensors.ino - two Grove MMA7660 accelerometers -> USB serial, ~100 Hz.
// (Grove "3-Axis Digital Accelerometer +/-1.5g", I2C address 0x4C.)
//
// Wiring (Grove Base Shield; any Arduino works with the same pins):
//   THIGH sensor -> any Grove "I2C" port     (hardware I2C, Wire)
//   SHANK sensor -> Grove port "D2"          (software I2C: D2 = SCL, D3 = SDA)
// Both boards have the same fixed address (0x4C), so they need separate buses.
// Without a shield: SHANK yellow wire -> D2, white -> D3, red -> 5V, black -> GND.
//
// Output, 115200 baud, one line per sample (the format main.cpp / fake_pi.py use):
//     t_ms,thigh_x,thigh_y,thigh_z,shank_x,shank_y,shank_z      (values in g)
// Status lines start with '#' and are ignored by the viewer and the bridge.
//
// Commands (send one character):
//   S  start streaming (also resets t_ms to 0)     X  stop streaming
//   C  re-run the sensor self-check                I  scan both buses
// Streaming is ON at power-up, so the Serial Monitor shows data immediately.
//
// Note: the MMA7660 is a 6-bit sensor: 1 count = 0.047 g (about 2.7 degrees of
// tilt near level). The viewer's low-pass filter smooths this, but expect a few
// degrees of graininess in the knee angle.

#include <Wire.h>

const uint8_t SENSOR_ADDR = 0x4C;
const uint8_t SOFT_SCL = 2;            // Grove D2 port: yellow wire
const uint8_t SOFT_SDA = 3;            // Grove D2 port: white wire
const unsigned long PERIOD_US = 10000; // 100 Hz polling (sensor updates at 120 Hz)
const long BAUD = 115200;

// MMA7660 registers
const uint8_t REG_XOUT = 0x00;  // XOUT, YOUT, ZOUT follow at 0x01, 0x02
const uint8_t REG_INTSU = 0x06;
const uint8_t REG_MODE = 0x07;
const uint8_t REG_SR = 0x08;
const uint8_t ALERT_BIT = 0x40; // set while the sensor is updating: read again

const char* const NAMES[2] = {"thigh", "shank"};
const bool ON_SOFT_BUS[2] = {false, true};

bool sensorOk[2] = {false, false};
bool streaming = true;
unsigned long t0Ms = 0, nextUs = 0, lastRetryMs = 0;

// ------------------------------------------------------------ software I2C ---
// Open-drain: a line is driven LOW, or released and pulled HIGH.
static void sdaHigh() { pinMode(SOFT_SDA, INPUT_PULLUP); }
static void sdaLow()  { digitalWrite(SOFT_SDA, LOW); pinMode(SOFT_SDA, OUTPUT); }
static void sclLow()  { digitalWrite(SOFT_SCL, LOW); pinMode(SOFT_SCL, OUTPUT); }
static void sclHigh() {
  pinMode(SOFT_SCL, INPUT_PULLUP);
  unsigned long t = micros();
  while (!digitalRead(SOFT_SCL) && micros() - t < 1000) {}  // allow clock stretching
}
static void halfBit() { delayMicroseconds(20); }

static void softStart() { sdaHigh(); sclHigh(); halfBit(); sdaLow(); halfBit(); sclLow(); }
static void softStop()  { sdaLow(); halfBit(); sclHigh(); halfBit(); sdaHigh(); halfBit(); }

static bool softWriteByte(uint8_t b) {
  for (int8_t i = 7; i >= 0; --i) {
    if (b & (1 << i)) sdaHigh(); else sdaLow();
    halfBit(); sclHigh(); halfBit(); sclLow();
  }
  sdaHigh(); halfBit(); sclHigh(); halfBit();
  bool ack = !digitalRead(SOFT_SDA);
  sclLow();
  return ack;
}

static uint8_t softReadByte(bool ack) {
  uint8_t b = 0;
  sdaHigh();
  for (uint8_t i = 0; i < 8; ++i) {
    halfBit(); sclHigh(); halfBit();
    b = (b << 1) | (digitalRead(SOFT_SDA) ? 1 : 0);
    sclLow();
  }
  if (ack) sdaLow(); else sdaHigh();
  halfBit(); sclHigh(); halfBit(); sclLow();
  sdaHigh();
  return b;
}

// --------------------------------------------------- register access, both buses
static bool regWrite(bool soft, uint8_t reg, uint8_t val) {
  if (!soft) {
    Wire.beginTransmission(SENSOR_ADDR);
    Wire.write(reg);
    Wire.write(val);
    return Wire.endTransmission() == 0;
  }
  softStart();
  bool ok = softWriteByte(SENSOR_ADDR << 1) && softWriteByte(reg) && softWriteByte(val);
  softStop();
  return ok;
}

static bool regRead(bool soft, uint8_t reg, uint8_t* buf, uint8_t n) {
  if (!soft) {
    Wire.beginTransmission(SENSOR_ADDR);
    Wire.write(reg);
    if (Wire.endTransmission(false) != 0) return false;  // repeated start
    if (Wire.requestFrom(SENSOR_ADDR, n) != n) return false;
    for (uint8_t i = 0; i < n; ++i) buf[i] = Wire.read();
    return true;
  }
  softStart();
  bool ok = softWriteByte(SENSOR_ADDR << 1) && softWriteByte(reg);
  if (ok) {
    softStart();  // repeated start
    ok = softWriteByte((SENSOR_ADDR << 1) | 1);
    if (ok)
      for (uint8_t i = 0; i < n; ++i) buf[i] = softReadByte(i + 1 < n);
  }
  softStop();
  return ok;
}

static bool probe(bool soft, uint8_t addr) {
  if (!soft) {
    Wire.beginTransmission(addr);
    return Wire.endTransmission() == 0;
  }
  softStart();
  bool ack = softWriteByte(addr << 1);
  softStop();
  return ack;
}

// ------------------------------------------------------------------ scan ----
static void scanBuses() {
  for (uint8_t b = 0; b < 2; ++b) {
    bool soft = (b == 1);
    Serial.print(soft ? F("# scan D2/D3 soft I2C (shank):") : F("# scan I2C port (thigh):"));
    uint8_t found = 0;
    for (uint8_t a = 0x08; a < 0x78; ++a) {
      if (probe(soft, a)) {
        Serial.print(F(" 0x"));
        if (a < 16) Serial.write('0');
        Serial.print(a, HEX);
        ++found;
      }
    }
    Serial.println(found ? F("") : F(" nothing (check cable and port)"));
  }
  Serial.println(F("# expected: 0x4C on both buses (MMA7660)"));
}

// --------------------------------------------------------------- MMA7660 ----
static bool initSensor(uint8_t idx, bool verbose) {
  bool soft = ON_SOFT_BUS[idx];
  bool ok = probe(soft, SENSOR_ADDR)
         && regWrite(soft, REG_MODE, 0x00)    // standby (required to configure)
         && regWrite(soft, REG_INTSU, 0x00)   // no interrupts
         && regWrite(soft, REG_SR, 0x00)      // 120 samples/s
         && regWrite(soft, REG_MODE, 0x01);   // active
  if (verbose || ok) {
    Serial.print(F("# ")); Serial.print(NAMES[idx]);
    Serial.print(soft ? F(" (D2/D3 soft I2C): ") : F(" (I2C port): "));
    Serial.println(ok ? F("ok (MMA7660 at 0x4C)") : F("FAIL, no answer at 0x4C (check cable/port)"));
  }
  return ok;
}

static void selfCheck() {
  for (uint8_t i = 0; i < 2; ++i) sensorOk[i] = initSensor(i, true);
  if (!sensorOk[0] || !sensorOk[1]) scanBuses();
}

// Reads X, Y, Z as signed 6-bit counts (21.33 counts per g).
static bool readXYZ(uint8_t idx, int8_t* out) {
  bool soft = ON_SOFT_BUS[idx];
  uint8_t b[3];
  for (uint8_t attempt = 0; attempt < 4; ++attempt) {
    if (!regRead(soft, REG_XOUT, b, 3)) return false;
    if ((b[0] | b[1] | b[2]) & ALERT_BIT) continue;  // mid-update: read again
    for (uint8_t k = 0; k < 3; ++k) {
      int8_t v = b[k] & 0x3F;
      if (v & 0x20) v -= 64;  // sign-extend 6-bit two's complement
      out[k] = v;
    }
    return true;
  }
  return false;
}

// Prints a count as g with 4 decimals (1 count = 0.046875 g), integer math only.
static void printG(int8_t count) {
  long v = (long)count * 46875 / 100;  // units of 0.0001 g
  if (v < 0) { Serial.write('-'); v = -v; }
  Serial.print(v / 10000);
  Serial.write('.');
  long f = v % 10000;
  if (f < 1000) Serial.write('0');
  if (f < 100) Serial.write('0');
  if (f < 10) Serial.write('0');
  Serial.print(f);
}

// ------------------------------------------------------------- main -------
void setup() {
  Serial.begin(BAUD);
  unsigned long w = millis();
  while (!Serial && millis() - w < 2000) {}  // boards with native USB
  Wire.begin();
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(3000, true);  // never hang on a stuck bus
#endif
  sdaHigh();
  pinMode(SOFT_SCL, INPUT_PULLUP);
  Serial.println(F("# brace_sensors ready (MMA7660). Commands: S=start X=stop C=check I=scan"));
  selfCheck();
  t0Ms = millis();
  nextUs = micros();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'S') { streaming = true; t0Ms = millis(); nextUs = micros(); Serial.println(F("# streaming")); }
    else if (c == 'X') { streaming = false; Serial.println(F("# stopped")); }
    else if (c == 'C') selfCheck();
    else if (c == 'I') scanBuses();
  }

  // Retry any missing sensor once a second (quietly; reports when it comes back).
  if ((!sensorOk[0] || !sensorOk[1]) && millis() - lastRetryMs > 1000) {
    lastRetryMs = millis();
    for (uint8_t i = 0; i < 2; ++i)
      if (!sensorOk[i]) sensorOk[i] = initSensor(i, false);
  }

  if (!streaming || !sensorOk[0] || !sensorOk[1]) return;

  unsigned long now = micros();
  if ((long)(now - nextUs) < 0) return;  // not time yet
  nextUs += PERIOD_US;
  if ((long)(now - nextUs) > (long)PERIOD_US) nextUs = now + PERIOD_US;  // fell behind: resync

  int8_t v[6];
  for (uint8_t i = 0; i < 2; ++i) {
    if (!readXYZ(i, v + 3 * i)) {
      Serial.print(F("# read error on ")); Serial.println(NAMES[i]);
      sensorOk[i] = false;
      return;
    }
  }

  Serial.print(millis() - t0Ms);
  for (uint8_t k = 0; k < 6; ++k) { Serial.write(','); printG(v[k]); }
  Serial.println();
}
