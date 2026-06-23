#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <Wire.h>
#include "DHT.h"
#include "Adafruit_SGP30.h"
#include <Preferences.h>
#include <ArduinoJson.h>
#include <aifes.h>
#include "aifes_live.h"
#include "mould_prediction_dataset_nir.h"

// --------- UART Configuration ---------
// UART2 pins for Raspberry Pi communication
#define UART_RX_PIN 16  // GPIO16 - connect to Pi TX (Pin 8)
#define UART_TX_PIN 17  // GPIO17 - connect to Pi RX (Pin 10)
#define UART_BAUD 115200

// --------- General config ---------
const uint32_t CYCLE_SECONDS          = 30;      // total cycle length (active + sleep) - 15 minutes
const uint8_t  SGP_WARMUP_SECONDS     = 15;       // SGP30 warmup calls
const uint32_t MASTER_LISTEN_WINDOW_MS = 90000;  // 90 s listen window (after warmup)

const bool DEBUG_PRINT_LOCAL_MAC      = false;
const bool DEBUG_PRINT_TIMING         = true;
const bool DEBUG_PRINT_MAC_EVENTS     = true;
const bool DEBUG_NO_SLEEP             = false;

const bool ENABLE_HUMAN_OUTPUT        = true;
const bool ENABLE_CSV_OUTPUT          = false;

// --------- Sensors ---------
#define DHTPIN   4
#define DHTTYPE  DHT22
DHT dht(DHTPIN, DHTTYPE);

Adafruit_SGP30 sgp;

const int MQ3_PIN = 34;

// MQ3 + divider config (same as node)
const float MQ3_VCC        = 5.0f;
const float ADC_REF        = 3.3f;
const float ADC_MAX        = 4095.0f;
const float MQ3_RL         = 1000.0f;

const float DIVIDER_TOP    = 56000.0f;   // 56k between MQ3 and node
const float DIVIDER_BOTTOM = 100000.0f;  // 100k between node and GND
const float DIVIDER_RATIO  = DIVIDER_BOTTOM / (DIVIDER_TOP + DIVIDER_BOTTOM);

// Preferences (NVS) for MQ3 R0
Preferences prefs;
float MQ3_R0 = 20000.0f;
const char* PREFS_NAMESPACE = "mq3";
const char* PREFS_KEY       = "r0";

// --------- Packet format ---------
typedef struct __attribute__((packed)) {
  float    temp;
  float    hum;
  uint16_t tvoc;
  uint16_t eco2;
  float    mq3_ppm;
  uint32_t timestamp;  // millis() at time of reading
} SensorPacket;

// Timing correction packet to send to nodes
typedef struct __attribute__((packed)) {
  int32_t adjustmentMs;  // positive = sleep longer, negative = sleep shorter
} TimingPacket;

SensorPacket masterPkt;
SensorPacket node1Pkt;
SensorPacket node2Pkt;

bool node1ReceivedThisCycle = false;
bool node2ReceivedThisCycle = false;

// Timing tracking
unsigned long node1ArrivalTime = 0;
unsigned long node2ArrivalTime = 0;

// --------- Node MAC addresses ---------
uint8_t node1Mac[] = { 0xE0, 0x8C, 0xFE, 0x2D, 0xD8, 0x60 }; // Node 1 MAC: E0:8C:FE:2D:D8:60
uint8_t node2Mac[] = { 0xE0, 0x8C, 0xFE, 0x5E, 0x1B, 0x8C }; // Node 2 MAC: E0:8C:FE:5E:1B:8C

// --------- Timing ---------
unsigned long startMs      = 0;
unsigned long listenStartMs = 0;  // Reference time = master wake
unsigned long warmupEndMs = 0;    // Time when warmup completes

// RTC memory to persist cycle count and delta TVOC across deep sleep
RTC_DATA_ATTR int   cycleNumber      = 0;
RTC_DATA_ATTR float prev_node1_tvoc  = 0.0f;
RTC_DATA_ATTR float prev_node2_tvoc  = 0.0f;

// --------- MQ3 helper ---------
float MQ3_getResistance() {
  int adc = analogRead(MQ3_PIN);

  float v_adc = (adc / ADC_MAX) * ADC_REF;   // voltage at ESP32 ADC pin
  float v_out = v_adc / DIVIDER_RATIO;      // reconstructed MQ3 output voltage

  if (v_out < 0.01f || v_out >= MQ3_VCC) {
    return NAN;
  }

  float v_ratio = (MQ3_VCC - v_out) / v_out;
  float Rs = MQ3_RL * v_ratio;

  return Rs;
}

// --------- Utility functions ---------
bool macEquals(const uint8_t *a, const uint8_t *b) {
  for (int i = 0; i < 6; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

void printMac(const uint8_t *mac) {
  for (int i = 0; i < 6; i++) {
    if (mac[i] < 16) Serial.print("0");
    Serial.print(mac[i], HEX);
    if (i < 5) Serial.print(":");
  }
}

void printHuman(const char *label, const SensorPacket &p) {
  Serial.print("[");
  Serial.print(label);
  Serial.println("]");
  Serial.print("  Temp: "); Serial.print(p.temp, 2);    Serial.println(" °C");
  Serial.print("  Hum : "); Serial.print(p.hum, 2);     Serial.println(" %RH");
  Serial.print("  TVOC: "); Serial.print(p.tvoc);       Serial.println(" ppb");
  Serial.print("  eCO2: "); Serial.print(p.eco2);       Serial.println(" ppm");
  Serial.print("  MQ3 : "); Serial.print(p.mq3_ppm, 3); Serial.println(" ppm");

  Serial.flush();
}

void printCSV(const char *label, const SensorPacket &p) {
  Serial.print(label); Serial.print(",");
  Serial.print(millis()); Serial.print(",");
  Serial.print(p.temp, 2); Serial.print(",");
  Serial.print(p.hum, 2);  Serial.print(",");
  Serial.print(p.tvoc);    Serial.print(",");
  Serial.print(p.eco2);    Serial.print(",");
  Serial.println(p.mq3_ppm, 3);
}

// --------- Timing Correction Logic ---------
void sendTimingCorrection(const uint8_t *nodeMac, unsigned long arrivalTime) {
  const long TARGET_ARRIVAL_MS = 50000;
  long targetArrivalTime = (long)listenStartMs + TARGET_ARRIVAL_MS;
  long error = (long)arrivalTime - targetArrivalTime;
  long arrivalTimeMs = (long)arrivalTime - (long)listenStartMs;
  int32_t correction = 0;

  if (arrivalTimeMs < 45000) {
    long distanceToTarget = TARGET_ARRIVAL_MS - arrivalTimeMs;
    correction = distanceToTarget + 2000;
    if (correction < 5000) correction = 5000;
    Serial.println("!!! EMERGENCY CORRECTION - NODE ARRIVED DURING WARMUP !!!");
    Serial.print("Arrived at: "); Serial.print(arrivalTimeMs); Serial.println(" ms");
    Serial.print("Target: "); Serial.print(TARGET_ARRIVAL_MS); Serial.println(" ms");
    Serial.print("Pushing to: "); Serial.print(TARGET_ARRIVAL_MS + 2000); Serial.println(" ms");
  } else {
    float gain;
    if (abs(error) > 10000) {
      gain = 0.5;
    } else if (abs(error) > 5000) {
      gain = 0.45;
    } else {
      gain = 0.4;
    }
    correction = (int32_t)(-error * gain);
    if (correction > 10000) correction = 10000;
    if (correction < -10000) correction = -10000;
  }

  TimingPacket timingPkt;
  timingPkt.adjustmentMs = correction;

  if (DEBUG_PRINT_TIMING) {
    Serial.println("\n----- TIMING CORRECTION -----");
    Serial.print("Arrival time: "); Serial.print(arrivalTimeMs); Serial.println(" ms into window");
    Serial.print("Node arrival error: ");
    Serial.print(error);
    Serial.println(" ms");
    Serial.print("Sending correction: ");
    Serial.print(correction);
    Serial.print(" ms (");
    if (correction > 0) {
      Serial.print("sleep ");
      Serial.print(correction);
      Serial.println(" ms longer)");
    } else if (correction < 0) {
      Serial.print("sleep ");
      Serial.print(-correction);
      Serial.println(" ms shorter)");
    } else {
      Serial.println("no adjustment)");
    }
    Serial.println("----------------------------\n");
  }

  esp_err_t result = esp_now_send(nodeMac, (uint8_t*)&timingPkt, sizeof(TimingPacket));
  if (result != ESP_OK && DEBUG_PRINT_TIMING) {
    Serial.print("[Master] Failed to send timing correction, error: ");
    Serial.println(result);
  }
}

// --------- ESP-NOW callbacks ---------
void onDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  if (DEBUG_PRINT_TIMING) {
    Serial.print("[Master] Timing correction send status: ");
    Serial.println(status == ESP_NOW_SEND_SUCCESS ? "SUCCESS" : "FAILED");
  }
}

void onDataRecv(const uint8_t *mac, const uint8_t *data, int len) {
  if (DEBUG_PRINT_MAC_EVENTS) {
    Serial.print("ESP-NOW packet received from MAC ");
    printMac(mac);
    Serial.print(", len = ");
    Serial.println(len);
  }

  if (len != sizeof(SensorPacket)) {
    if (DEBUG_PRINT_MAC_EVENTS) {
      Serial.println("  -> Unexpected packet size, ignoring.");
    }
    return;
  }

  if (macEquals(mac, node1Mac)) {
    node1ReceivedThisCycle = true;
    node1ArrivalTime = millis();
    memcpy(&node1Pkt, data, sizeof(SensorPacket));

    if (DEBUG_PRINT_TIMING && listenStartMs != 0) {
      unsigned long dt = node1ArrivalTime - listenStartMs;
      float dtSec = dt / 1000.0f;
      Serial.println("\n========== NODE 1 ARRIVAL ==========");
      Serial.print("Time since listen start: ");
      Serial.print(dtSec, 3);
      Serial.println(" s");
      Serial.print("Absolute arrival time: ");
      Serial.print(node1ArrivalTime);
      Serial.println(" ms");
      if (node2ReceivedThisCycle) {
        long timeDiff = (long)node1ArrivalTime - (long)node2ArrivalTime;
        Serial.print("Time difference from Node 2: ");
        Serial.print(timeDiff);
        Serial.println(" ms");
      }
      Serial.println("===================================\n");
    }

    if (ENABLE_HUMAN_OUTPUT) printHuman("node1", node1Pkt);
    if (ENABLE_CSV_OUTPUT)   printCSV("node1", node1Pkt);
    sendTimingCorrection(node1Mac, node1ArrivalTime);

  } else if (macEquals(mac, node2Mac)) {
    node2ReceivedThisCycle = true;
    node2ArrivalTime = millis();
    memcpy(&node2Pkt, data, sizeof(SensorPacket));

    if (DEBUG_PRINT_TIMING && listenStartMs != 0) {
      unsigned long dt = node2ArrivalTime - listenStartMs;
      float dtSec = dt / 1000.0f;
      Serial.println("\n========== NODE 2 ARRIVAL ==========");
      Serial.print("Time since listen start: ");
      Serial.print(dtSec, 3);
      Serial.println(" s");
      Serial.print("Absolute arrival time: ");
      Serial.print(node2ArrivalTime);
      Serial.println(" ms");
      if (node1ReceivedThisCycle) {
        long timeDiff = (long)node2ArrivalTime - (long)node1ArrivalTime;
        Serial.print("Time difference from Node 1: ");
        Serial.print(timeDiff);
        Serial.println(" ms");
      }
      Serial.println("===================================\n");
    }

    if (ENABLE_HUMAN_OUTPUT) printHuman("node2", node2Pkt);
    if (ENABLE_CSV_OUTPUT)   printCSV("node2", node2Pkt);
    sendTimingCorrection(node2Mac, node2ArrivalTime);

  } else if (DEBUG_PRINT_MAC_EVENTS) {
    Serial.println("  -> Unknown MAC, ignoring.");
  }
}

// --------- Send Data via UART to Raspberry Pi ---------
void sendDataToRaspberryPi(uint8_t freshness_class) {
  Serial.println("\n=== Sending data to Raspberry Pi via UART ===");

  StaticJsonDocument<900> doc;
  doc["cycle"] = cycleNumber;
  doc["timestamp"] = millis();

  // Freshness prediction from AIfES
  doc["freshness_class"] = freshness_class;
  doc["freshness_label"] = freshness_class == 0 ? "Fresh" :
                           freshness_class == 1 ? "Aging" :
                           freshness_class == 255 ? "unknown" : "Degraded";

  // Master data
  JsonObject master = doc.createNestedObject("master");
  master["temp"] = masterPkt.temp;
  master["hum"] = masterPkt.hum;
  master["tvoc"] = masterPkt.tvoc;
  master["eco2"] = masterPkt.eco2;
  master["mq3_ppm"] = masterPkt.mq3_ppm;
  master["received"] = true;

  // Node1 data
  JsonObject node1 = doc.createNestedObject("node1");
  if (node1ReceivedThisCycle) {
    node1["temp"] = node1Pkt.temp;
    node1["hum"] = node1Pkt.hum;
    node1["tvoc"] = node1Pkt.tvoc;
    node1["eco2"] = node1Pkt.eco2;
    node1["mq3_ppm"] = node1Pkt.mq3_ppm;
    node1["received"] = true;
  } else {
    node1["received"] = false;
  }

  // Node2 data
  JsonObject node2 = doc.createNestedObject("node2");
  if (node2ReceivedThisCycle) {
    node2["temp"] = node2Pkt.temp;
    node2["hum"] = node2Pkt.hum;
    node2["tvoc"] = node2Pkt.tvoc;
    node2["eco2"] = node2Pkt.eco2;
    node2["mq3_ppm"] = node2Pkt.mq3_ppm;
    node2["received"] = true;
  } else {
    node2["received"] = false;
  }

  // Serialize and send
  String output;
  serializeJson(doc, output);
  Serial2.println(output);  // Send to Pi via UART2
  Serial2.flush();

  Serial.println("Data sent to Raspberry Pi");
  if (DEBUG_PRINT_TIMING) {
    Serial.print("JSON size: ");
    Serial.print(output.length());
    Serial.println(" bytes");
  }
}

// --------- ESP-NOW init ---------
void initEspNowMaster() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  if (DEBUG_PRINT_LOCAL_MAC) {
    Serial.print("MASTER WiFi MAC: ");
    Serial.println(WiFi.macAddress());
  }

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed on master!");
    return;
  }

  esp_now_register_recv_cb(onDataRecv);
  esp_now_register_send_cb(onDataSent);

  esp_now_peer_info_t peerInfo{};
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  memcpy(peerInfo.peer_addr, node1Mac, 6);
  esp_now_add_peer(&peerInfo);

  memcpy(peerInfo.peer_addr, node2Mac, 6);
  esp_now_add_peer(&peerInfo);
}

// --------- Sleep handling ---------
void goToSleepForRemainingCycle() {
  unsigned long activeMs = millis() - startMs;
  float activeSec = activeMs / 1000.0f;

  long remainingMs = (long)CYCLE_SECONDS * 1000L - (long)activeMs;
  if (remainingMs < 100) remainingMs = 100;

  float remainingSec = remainingMs / 1000.0f;

  if (DEBUG_PRINT_TIMING) {
    Serial.print("Master active time this cycle (s): ");
    Serial.println(activeSec, 3);
    Serial.print("Master going to sleep for (s): ");
    Serial.println(remainingSec, 3);
  }

  if (DEBUG_NO_SLEEP) {
    Serial.println("DEBUG_NO_SLEEP enabled: Master will stay awake and keep listening.");
    while (true) delay(1000);
  } else {
    uint64_t sleepUs = (uint64_t)remainingMs * 1000ULL;
    Serial.println("\n");
    Serial.println("###############################################################");
    Serial.println("#                      CYCLE END - SLEEPING                   #");
    Serial.println("###############################################################");
    Serial.println("Master going to deep sleep...");
    esp_sleep_enable_timer_wakeup(sleepUs);
    delay(50);
    esp_deep_sleep_start();
  }
}

// --------- setup / loop ---------
void setup() {
  Serial.begin(115200);
  delay(500);
  startMs = millis();
  listenStartMs = millis();
  cycleNumber++;

  btStop();
  setCpuFrequencyMhz(160);

  Serial.println("\n\n");
  Serial.println("###############################################################");
  Serial.println("#                      NEW CYCLE START                        #");
  Serial.println("###############################################################");
  Serial.printf("MASTER WAKEUP - Cycle #%d\n", cycleNumber);

  // Load MQ3_R0 from NVS
  prefs.begin(PREFS_NAMESPACE, true);
  MQ3_R0 = prefs.getFloat(PREFS_KEY, 20000.0f);
  prefs.end();

  Serial.print("Master: loaded MQ3_R0 from NVS (");
  Serial.print(PREFS_NAMESPACE);
  Serial.print("/");
  Serial.print(PREFS_KEY);
  Serial.print(") = ");
  Serial.print(MQ3_R0, 2);
  Serial.println(" ohms");

  Serial2.begin(UART_BAUD, SERIAL_8N1, UART_RX_PIN, UART_TX_PIN);
  Serial.println("UART2 initialized for Raspberry Pi (GPIO16 RX, GPIO17 TX)");

  analogReadResolution(12);

  dht.begin();
  Wire.begin();

  if (!sgp.begin()) {
    Serial.println("SGP30 init failed on master!");
  }

  initEspNowMaster();
  aifes_build();

  if (ENABLE_CSV_OUTPUT) {
    Serial.println("label,millis,temp,hum,tvoc,eco2,mq3_ppm");
  }

  // SGP30 warmup
  for (uint8_t i = 0; i < SGP_WARMUP_SECONDS; i++) {
    if (sgp.IAQmeasure()) {
      masterPkt.tvoc = sgp.TVOC;
      masterPkt.eco2 = sgp.eCO2;
    }
    delay(1000);
  }

  warmupEndMs = millis();

  // DHT22
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  masterPkt.temp = isnan(t) ? NAN : t;
  masterPkt.hum  = isnan(h) ? NAN : h;

  // MQ3 PPM
  float rs = MQ3_getResistance();
  float ppm = NAN;
  if (!isnan(rs) && rs > 0.0f && MQ3_R0 > 0.0f) {
    float ratio     = rs / MQ3_R0;
    float log_ratio = log10f(ratio);
    ppm = powf(10.0f, ((log_ratio - 0.35f) / -0.47f));
  }
  masterPkt.mq3_ppm = (isnan(ppm) || isinf(ppm)) ? NAN : ppm;

  if (ENABLE_HUMAN_OUTPUT) printHuman("master", masterPkt);
  if (ENABLE_CSV_OUTPUT)   printCSV("master", masterPkt);

  Serial.println("Master listening for node packets...");

  while (true) {
    unsigned long elapsed = millis() - warmupEndMs;

    if (node1ReceivedThisCycle && node2ReceivedThisCycle) {
      Serial.println("Both node1 and node2 packets received, ending listen early.");
      break;
    }

    if (elapsed >= MASTER_LISTEN_WINDOW_MS) {
      Serial.println("Listen window expired, ending listen.");
      break;
    }

    delay(10);
  }

  delay(100);

  // Print summary
  if (DEBUG_PRINT_TIMING) {
    Serial.println("\n============================================");
    Serial.println("   LISTEN WINDOW COMPLETE - SUMMARY");
    Serial.println("============================================");
    int count = (node1ReceivedThisCycle ? 1 : 0) + (node2ReceivedThisCycle ? 1 : 0);
    Serial.print("Nodes received: ");
    Serial.print(count);
    Serial.println("/2");
    Serial.print("Node 1: ");
    Serial.println(node1ReceivedThisCycle ? "RECEIVED" : "MISSING");
    Serial.print("Node 2: ");
    Serial.println(node2ReceivedThisCycle ? "RECEIVED" : "MISSING");
    if (node1ReceivedThisCycle && node2ReceivedThisCycle) {
      long timeDiff = abs((long)node1ArrivalTime - (long)node2ArrivalTime);
      Serial.print("Sync offset: ");
      Serial.print(timeDiff);
      Serial.println(" ms");
    }
    Serial.println("============================================\n");
  }

  // Run AIfES inference if both nodes reported
  uint8_t freshness_class = 255;
  if (node1ReceivedThisCycle && node2ReceivedThisCycle) {
    float raw[10] = {
      node1Pkt.temp,  node1Pkt.hum,  (float)node1Pkt.tvoc,  node1Pkt.mq3_ppm,
      node2Pkt.temp,  node2Pkt.hum,  (float)node2Pkt.tvoc,  node2Pkt.mq3_ppm,
      (float)node1Pkt.tvoc - prev_node1_tvoc,
      (float)node2Pkt.tvoc - prev_node2_tvoc
    };
    for (int i = 0; i < 10; i++)
      raw[i] = constrain((raw[i] - NIR_FEAT_MIN[i]) / NIR_FEAT_RANGE[i], 0.0f, 1.0f);

    freshness_class = aifes_predict(raw);

    prev_node1_tvoc = (float)node1Pkt.tvoc;
    prev_node2_tvoc = (float)node2Pkt.tvoc;

    const char* labels[] = {"Fresh", "Aging", "Degraded"};
    Serial.println("\n========== FRESHNESS PREDICTION ==========");
    Serial.printf("  Class : %d (%s)\n", freshness_class,
        freshness_class < 3 ? labels[freshness_class] : "error");
    Serial.printf("  Scores: Fresh=%.3f  Aging=%.3f  Degraded=%.3f\n",
        _nir_out[0], _nir_out[1], _nir_out[2]);
    Serial.println("==========================================\n");
  } else {
    Serial.println("[AIfES] Skipped — not all nodes reported this cycle.");
  }

  sendDataToRaspberryPi(freshness_class);
  delay(100);

  goToSleepForRemainingCycle();
}

void loop() {
  // never used; all work is done in setup() per wake
}
