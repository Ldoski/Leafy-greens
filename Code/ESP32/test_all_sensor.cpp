#include <Arduino.h>
#include <Wire.h>
#include "DHT.h"
#include "Adafruit_SGP30.h"
#include <Preferences.h>

// ---------------- Pins ----------------
#define DHTPIN   4
#define DHTTYPE  DHT22
DHT dht(DHTPIN, DHTTYPE);

Adafruit_SGP30 sgp;

const int MQ3_PIN = 34;

// ---------------- MQ3 constants (from your node) ----------------
const float MQ3_VCC        = 5.0f;
const float ADC_REF        = 3.3f;
const float ADC_MAX        = 4095.0f;
const float MQ3_RL         = 1000.0f;
const float DIVIDER_TOP    = 56000.0f;
const float DIVIDER_BOTTOM = 100000.0f;
const float DIVIDER_RATIO  = DIVIDER_BOTTOM / (DIVIDER_TOP + DIVIDER_BOTTOM);

// ---------------- NVS (shared with your node) ----------------
// Same namespace/key the node reads, so calibrating here updates the
// value the node will load on its next boot. One source of truth.
Preferences prefs;
const char* MQ3_NS  = "mq3";
const char* MQ3_KEY = "r0";
float MQ3_R0 = 20000.0f;   // overwritten from NVS at boot if present

// ---------------- Timing ----------------
const uint32_t TICK_MS    = 1000;   // 1Hz master tick (SGP30 needs this)
const uint32_t WARMUP_MS  = 15000;  // SGP30 fixed-output boot window

uint32_t lastTick = 0;
uint32_t startMs  = 0;
uint32_t tick     = 0;

float heldT = NAN, heldH = NAN;

// ---------- MQ3 read (identical math to your node) ----------
float mq3_Rs() {
  int adc = analogRead(MQ3_PIN);
  float vAdc = (adc / ADC_MAX) * ADC_REF;        // node's manual scaling
  float vOut = vAdc / DIVIDER_RATIO;             // back up to 0..5V
  if (vOut < 0.01f || vOut >= MQ3_VCC) return NAN;
  return MQ3_RL * ((MQ3_VCC - vOut) / vOut);
}

// ---------- Calibration: average Rs in clean air -> R0 -> NVS ----------
// Triggered by sending 'c' over serial. Blocks ~60s while sampling.
// Keeps the SGP30 serviced at 1Hz during the wait so its baseline
// algorithm isn't disturbed by the calibration pause.
void runCalibration() {
  Serial.println("\n=== MQ3 CALIBRATION ===");
  Serial.println("Sensor MUST be in clean reference air. Starting in 3s...");
  delay(3000);

  const int NUM_SAMPLES = 30;
  float rsSum = 0.0f;
  int valid = 0;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    float rs = mq3_Rs();   // same function the live loop uses
    if (!isnan(rs) && rs > 0.0f) {
      rsSum += rs; valid++;
      Serial.printf("  sample %2d/%d  Rs=%.0f ohm\n", i + 1, NUM_SAMPLES, rs);
    } else {
      Serial.printf("  sample %2d/%d  INVALID (V out of range)\n", i + 1, NUM_SAMPLES);
    }
    // 2s spacing, but keep SGP30 alive at 1Hz during the gap.
    for (int s = 0; s < 2; s++) { sgp.IAQmeasure(); delay(1000); }
  }

  if (valid == 0) {
    Serial.println("ERROR: no valid samples, R0 not changed.\n");
    return;
  }

  float R0 = rsSum / valid;
  prefs.begin(MQ3_NS, false);     // RW
  prefs.putFloat(MQ3_KEY, R0);
  prefs.end();
  MQ3_R0 = R0;                    // update live value so ppm uses it now

  Serial.printf("New R0 = %.0f ohm (from %d samples), saved to NVS %s/%s\n",
                R0, valid, MQ3_NS, MQ3_KEY);
  Serial.println("Live ppm now uses this R0. Your node will pick it up on next boot.\n");

  lastTick = millis();           // avoid a burst when the normal loop resumes
}

void setup() {
  Serial.begin(115200);
  delay(500);
  startMs = millis();

  // ADC config MUST match between calibration and live reads, or the
  // saved R0 won't correspond to the Rs you compute later. 11dB is the
  // ESP32 default but set it explicitly so there's no ambiguity.
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  dht.begin();
  Wire.begin();

  if (!sgp.begin()) {
    Serial.println("SGP30 not found. Check SDA/SCL, 3.3V, pull-ups.");
    while (true) delay(1000);
  }
  Serial.print("SGP30 serial: 0x");
  Serial.print(sgp.serialnumber[0], HEX);
  Serial.print(sgp.serialnumber[1], HEX);
  Serial.println(sgp.serialnumber[2], HEX);

  // Load R0 from NVS, same as the node (read-only).
  prefs.begin(MQ3_NS, true);
  MQ3_R0 = prefs.getFloat(MQ3_KEY, 20000.0f);
  prefs.end();
  Serial.printf("Loaded MQ3_R0 from NVS = %.0f ohm%s\n",
                MQ3_R0, (MQ3_R0 == 20000.0f) ? " (fallback, likely uncalibrated)" : "");

  delay(2000);   // let the DHT settle before its first read
  Serial.println("ALL-SENSOR test start. Send 'c' to (re)calibrate MQ3 R0.");
  Serial.println("t(s)\tT(C)\tRH(%)\teCO2\tTVOC\tRs(ohm)\tppm\t| flags (D=dht S=sgp M=mq3)");
}

void loop() {
  // Calibration trigger: any 'c'/'C' on serial starts the routine.
  if (Serial.available()) {
    char ch = Serial.read();
    if (ch == 'c' || ch == 'C') runCalibration();
  }

  if (millis() - lastTick < TICK_MS) return;
  lastTick = millis();
  tick++;

  bool warming = (millis() - startMs) < WARMUP_MS;

  // ---- DHT: only every 2nd tick (2s hardware floor) ----
  bool dhtFresh = (tick % 2 == 1);
  bool dhtOkThis = false;
  if (dhtFresh) {
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    bool sane = !isnan(t) && !isnan(h) && t > -40 && t < 80 && h >= 0 && h <= 100;
    if (sane) { heldT = t; heldH = h; dhtOkThis = true; }
  }

  // ---- SGP30: every tick, IAQmeasure only (matches node) ----
  bool sgpOk = sgp.IAQmeasure();

  // ---- MQ3: every tick ----
  float rs = mq3_Rs();
  bool mqOk = !isnan(rs) && rs > 0;
  float ppm = NAN;
  if (mqOk) ppm = powf(10.0f, ((log10f(rs / MQ3_R0) - 0.35f) / -0.47f));

  // ---- Print combined row ----
  char dFlag = dhtFresh ? (dhtOkThis ? 'o' : 'x') : '-';
  char sFlag = sgpOk ? (warming ? 'w' : 'o') : 'x';
  char mFlag = mqOk ? 'o' : 'x';

  Serial.printf("%.0f\t%.1f\t%.1f\t%u\t%u\t%.0f\t%.2f\t| D:%c S:%c M:%c\n",
                millis()/1000.0,
                heldT, heldH,
                sgpOk ? sgp.eCO2 : 0, sgpOk ? sgp.TVOC : 0,
                mqOk ? rs : 0.0f, mqOk ? ppm : 0.0f,
                dFlag, sFlag, mFlag);
}