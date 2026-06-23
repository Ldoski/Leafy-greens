#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <Wire.h>
#include "DHT.h"
#include "Adafruit_SGP30.h"
#include <Preferences.h>

// --------- General config ---------
const uint32_t CYCLE_SECONDS      = 900;  // total cycle length (active + sleep) - 15 minutes
const uint8_t  SGP_WARMUP_SECONDS = 45;   // SGP30 IAQmeasure() calls during warmup

const bool NODE_ENABLE_HUMAN_OUTPUT  = true;
const bool NODE_DEBUG_PRINT_TIMING   = true;

// --------- Sensors ---------
#define DHTPIN   4
#define DHTTYPE  DHT22
DHT dht(DHTPIN, DHTTYPE);

Adafruit_SGP30 sgp;

const int MQ3_PIN = 34;

// MQ3 + divider config
const float MQ3_VCC        = 5.0f;
const float ADC_REF        = 3.3f;
const float ADC_MAX        = 4095.0f;
const float MQ3_RL         = 1000.0f;

const float DIVIDER_TOP    = 56000.0f;   // 56k between MQ3 and node
const float DIVIDER_BOTTOM = 100000.0f;  // 100k between node and GND
const float DIVIDER_RATIO  = DIVIDER_BOTTOM / (DIVIDER_TOP + DIVIDER_BOTTOM);

// Preferences (NVS)
Preferences prefs;
float MQ3_R0 = 20000.0f;   // default fallback if not found
const char* PREFS_NAMESPACE = "mq3";
const char* PREFS_KEY       = "r0";

// SGP30 baseline keys
const char* SGP_NAMESPACE   = "sgp30";
const char* SGP_ECO2_KEY    = "eco2_base";
const char* SGP_TVOC_KEY    = "tvoc_base";

// --------- ESP-NOW / MAC ---------
uint8_t masterMac[] = {0xCC, 0xDB, 0xA7, 0x98, 0xD2, 0xD0}; // Master MAC

typedef struct __attribute__((packed)) {
  float    temp;
  float    hum;
  uint16_t tvoc;
  uint16_t eco2;
  float    mq3_ppm;
  uint32_t timestamp;  // millis() at time of reading
} SensorPacket;

// Timing correction packet from master
typedef struct __attribute__((packed)) {
  int32_t adjustmentMs;  // positive = sleep longer, negative = sleep shorter
} TimingPacket;

SensorPacket pkt;
TimingPacket timingCorrection;
bool receivedTimingCorrection = false;

unsigned long startMs = 0;
int32_t sleepAdjustmentMs = 0;  // accumulated adjustment from master

// --------- MQ3 helpers ---------
float MQ3_getResistance() {
  int adc = analogRead(MQ3_PIN);

  float v_adc = (adc / ADC_MAX) * ADC_REF;   // voltage at ESP32 ADC pin (0–3.3 V)
  float v_out = v_adc / DIVIDER_RATIO;      // reconstructed MQ3 output voltage (0–5 V)

  if (v_out < 0.01f || v_out >= MQ3_VCC) {
    return NAN;
  }

  float v_ratio = (MQ3_VCC - v_out) / v_out;
  float Rs = MQ3_RL * v_ratio;

  return Rs;
}

// ESP-NOW send callback (optional debug)
void onDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  if (NODE_DEBUG_PRINT_TIMING) {
    Serial.print("[Node] Packet send status: ");
    Serial.println(status == ESP_NOW_SEND_SUCCESS ? "SUCCESS" : "FAILED");
  }
}

// ESP-NOW receive callback for timing corrections from master
void onDataRecv(const uint8_t *mac, const uint8_t *data, int len) {
  if (len == sizeof(TimingPacket)) {
    memcpy(&timingCorrection, data, sizeof(TimingPacket));
    receivedTimingCorrection = true;
    
    Serial.print("[Node] Received timing correction from master: ");
    Serial.print(timingCorrection.adjustmentMs);
    Serial.println(" ms");
  }
}

void goToSleepForRemainingCycle() {
  unsigned long activeMs = millis() - startMs;
  long remainingMs = (long)CYCLE_SECONDS * 1000L - (long)activeMs;
  
  // Apply accumulated timing correction from master
  remainingMs += sleepAdjustmentMs;
  
  if (remainingMs < 100) remainingMs = 100;

  if (NODE_DEBUG_PRINT_TIMING) {
    Serial.println("\n===== NODE SLEEP CALCULATION =====");
    Serial.print("Active time this cycle: ");
    Serial.print(activeMs);
    Serial.println(" ms");
    
    if (sleepAdjustmentMs != 0) {
      Serial.print("Timing adjustment applied: ");
      Serial.print(sleepAdjustmentMs);
      Serial.println(" ms");
    }
    
    Serial.print("Final sleep duration: ");
    Serial.print(remainingMs);
    Serial.println(" ms");
    Serial.println("==================================");
  }

  uint64_t sleepUs = (uint64_t)remainingMs * 1000ULL;
  Serial.println("\n[Node] Going to deep sleep...\n");
  esp_sleep_enable_timer_wakeup(sleepUs);
  delay(50);
  esp_deep_sleep_start();
}

void setup() {
  Serial.begin(115200);
  delay(500);
  startMs = millis();

  // Energy optimizations
  btStop();                    // Disable Bluetooth - not used
  setCpuFrequencyMhz(160);     // Reduce CPU from 240MHz to 160MHz

  Serial.println("NODE WAKEUP");

  // Load MQ3_R0 from NVS
  prefs.begin(PREFS_NAMESPACE, true);  // read-only
  MQ3_R0 = prefs.getFloat(PREFS_KEY, 20000.0f);
  prefs.end();

  Serial.print("Node: loaded MQ3_R0 from NVS (");
  Serial.print(PREFS_NAMESPACE);
  Serial.print("/");
  Serial.print(PREFS_KEY);
  Serial.print(") = ");
  Serial.print(MQ3_R0, 2);
  Serial.println(" ohms");

  analogReadResolution(12);

  // Init sensors
  dht.begin();
  Wire.begin();

  if (!sgp.begin()) {
    Serial.println("SGP30 init failed on node!");
  } else {
    // Restore SGP30 baseline from NVS
    prefs.begin(SGP_NAMESPACE, true);  // read-only
    bool hasEco2 = prefs.isKey(SGP_ECO2_KEY);
    bool hasTvoc = prefs.isKey(SGP_TVOC_KEY);
    uint16_t eco2_base = hasEco2 ? prefs.getUShort(SGP_ECO2_KEY, 0) : 0;
    uint16_t tvoc_base = hasTvoc ? prefs.getUShort(SGP_TVOC_KEY, 0) : 0;
    prefs.end();
    
    if (hasEco2 && hasTvoc && eco2_base != 0 && tvoc_base != 0
        && eco2_base <= 60000 && tvoc_base <= 60000) {
      sgp.setIAQBaseline(eco2_base, tvoc_base);
      Serial.print("SGP30 baseline restored: eCO2=");
      Serial.print(eco2_base);
      Serial.print(", TVOC=");
      Serial.println(tvoc_base);
    } else {
      Serial.println("No SGP30 baseline found in NVS - starting fresh");
    }
  }

  // SGP30 warmup
  for (uint8_t i = 0; i < SGP_WARMUP_SECONDS; i++) {
    if (sgp.IAQmeasure()) {
      pkt.tvoc = sgp.TVOC;
      pkt.eco2 = sgp.eCO2;
    }
    delay(1000);
  }

  // DHT22
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  pkt.temp = isnan(t) ? NAN : t;
  pkt.hum  = isnan(h) ? NAN : h;

  // MQ3 PPM
  float rs = MQ3_getResistance();
  float ppm = NAN;
  if (!isnan(rs) && rs > 0.0f && MQ3_R0 > 0.0f) {
    float ratio     = rs / MQ3_R0;
    float log_ratio = log10f(ratio);
    ppm = powf(10.0f, ((log_ratio - 0.35f) / -0.47f));
  }
  pkt.mq3_ppm = (isnan(ppm) || isinf(ppm)) ? NAN : ppm;
  pkt.timestamp = millis();

  // Save SGP30 baseline to NVS for next cycle
  // Guard: SGP30 returns ~65528 (0xFFF8) as its initial unlearned default.
  // Only save once the sensor has converged to plausible values (<= 60000).
  uint16_t eco2_base, tvoc_base;
  if (sgp.getIAQBaseline(&eco2_base, &tvoc_base)) {
    if (eco2_base <= 60000 && tvoc_base <= 60000) {
      prefs.begin(SGP_NAMESPACE, false);  // read-write
      prefs.putUShort(SGP_ECO2_KEY, eco2_base);
      prefs.putUShort(SGP_TVOC_KEY, tvoc_base);
      prefs.end();
      Serial.print("SGP30 baseline saved: eCO2=");
      Serial.print(eco2_base);
      Serial.print(", TVOC=");
      Serial.println(tvoc_base);
    } else {
      Serial.print("SGP30 baseline not saved (sensor still initialising): eCO2=");
      Serial.print(eco2_base);
      Serial.print(", TVOC=");
      Serial.println(tvoc_base);
    }
  }

  if (NODE_ENABLE_HUMAN_OUTPUT) {
    Serial.println("[node1]");
    Serial.print("  Temp: "); Serial.print(pkt.temp, 2);    Serial.println(" °C");
    Serial.print("  Hum : "); Serial.print(pkt.hum, 2);     Serial.println(" %RH");
    Serial.print("  TVOC: "); Serial.print(pkt.tvoc);       Serial.println(" ppb");
    Serial.print("  eCO2: "); Serial.print(pkt.eco2);       Serial.println(" ppm");
    Serial.print("  MQ3 : "); Serial.print(pkt.mq3_ppm, 3); Serial.println(" ppm");
  }

  // ESP-NOW init & send
  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed on node, going to sleep anyway");
    goToSleepForRemainingCycle();
    return;
  }

  esp_now_register_send_cb(onDataSent);
  esp_now_register_recv_cb(onDataRecv);  // Listen for timing corrections

  esp_now_peer_info_t peerInfo{};
  memcpy(peerInfo.peer_addr, masterMac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add master as peer");
    goToSleepForRemainingCycle();
    return;
  }

  esp_err_t res = esp_now_send(masterMac, (uint8_t*)&pkt, sizeof(SensorPacket));
  if (res != ESP_OK) {
    Serial.print("[Node] esp_now_send error: ");
    Serial.println(res);
  } else {
    Serial.println("[Node] Sensor packet sent via ESP-NOW");
  }

  // Wait for timing correction from master (with timeout)
  Serial.println("[Node] Waiting for timing correction from master...");
  unsigned long waitStart = millis();
  while (!receivedTimingCorrection && (millis() - waitStart < 2000)) {
    delay(10);
  }
  
  if (receivedTimingCorrection) {
    sleepAdjustmentMs = timingCorrection.adjustmentMs;
    Serial.print("[Node] Timing correction received and will be applied: ");
    Serial.print(sleepAdjustmentMs);
    Serial.println(" ms");
  } else {
    Serial.println("[Node] No timing correction received (timeout)");
    sleepAdjustmentMs = 0;  // No adjustment
  }

  delay(100);  // give radio time to finish

  goToSleepForRemainingCycle();
}

void loop() {
  // never used; all work done in setup() per wake
}