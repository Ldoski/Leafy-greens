#include <Arduino.h>
#include <Preferences.h>

// --------- Hardware config ---------
const int MQ3_PIN = 34;   // ADC pin on ESP32

// MQ3 + divider config
const float MQ3_VCC        = 5.0f;     // Sensor supply
const float ADC_REF        = 3.3f;     // ESP32 ADC reference
const float ADC_MAX        = 4095.0f;  // 12-bit ADC
const float MQ3_RL         = 1000.0f;  // 1k load resistor to GND (sensor side)

// Divider: MQ3 A0 -> 56k -> node -> 100k -> GND, node -> ADC (via 1k)
const float DIVIDER_TOP    = 56000.0f;   // 56k between MQ3 and node
const float DIVIDER_BOTTOM = 100000.0f;  // 100k between node and GND
const float DIVIDER_RATIO  = DIVIDER_BOTTOM / (DIVIDER_TOP + DIVIDER_BOTTOM);
// V_node = V_out * DIVIDER_RATIO

// NVS config
Preferences prefs;
const char* PREFS_NAMESPACE = "mq3";
const char* PREFS_KEY       = "r0";

// --------- MQ3 helpers ---------
float MQ3_getResistance() {
  int adc = analogRead(MQ3_PIN);

  float v_adc = (adc / ADC_MAX) * ADC_REF;      // voltage at ESP32 ADC pin
  float v_out = v_adc / DIVIDER_RATIO;          // reconstructed MQ3 output voltage (before divider)

  // Sanity check: avoid nonsense and division by zero
  if (v_out < 0.01f || v_out >= MQ3_VCC) {
    return NAN;
  }

  // Voltage divider between MQ3 internal Rs and RL to ground:
  // V_out = Vcc * (RL / (Rs + RL))  =>  Rs = RL * (Vcc - V_out) / V_out
  float v_ratio = (MQ3_VCC - v_out) / v_out;
  float Rs = MQ3_RL * v_ratio;

  return Rs;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("MQ3 Calibration Sketch");
  Serial.println("----------------------");
  Serial.println("Make sure the sensor is in clean reference air.");
  Serial.println("Starting calibration...\n");
  
  analogReadResolution(12);  // ensure 12-bit ADC (0-4095)
  analogSetAttenuation(ADC_11db);  // Set ADC attenuation to 11dB (0-3.3V range)

  const int NUM_SAMPLES = 30;
  float rsSum = 0.0f;
  int validCount = 0;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    // Read raw ADC and show voltages for debugging
    int adc = analogRead(MQ3_PIN);
    float v_adc = (adc / ADC_MAX) * ADC_REF;
    float v_out = v_adc / DIVIDER_RATIO;
    
    Serial.print("Sample ");
    Serial.print(i + 1);
    Serial.print("/");
    Serial.print(NUM_SAMPLES);
    Serial.print(": ADC=");
    Serial.print(adc);
    Serial.print(", V_adc=");
    Serial.print(v_adc, 3);
    Serial.print("V, V_out=");
    Serial.print(v_out, 3);
    Serial.print("V");
    
    float rs = MQ3_getResistance();

    if (!isnan(rs) && rs > 0.0f) {
      rsSum += rs;
      validCount++;
      Serial.print(" -> Rs = ");
      Serial.print(rs, 2);
      Serial.println(" ohms");
    } else {
      Serial.println(" -> INVALID (out of range)");
    }

    delay(2000);  // 2 seconds between samples for stability
  }

  if (validCount == 0) {
    Serial.println("ERROR: No valid MQ3 samples collected; cannot compute R0.");
  } else {
    float R0 = rsSum / validCount;

    Serial.println();
    Serial.print("Calculated MQ3 R0 from ");
    Serial.print(validCount);
    Serial.print(" samples: ");
    Serial.print(R0, 2);
    Serial.println(" ohms");

    prefs.begin(PREFS_NAMESPACE, false);  // RW mode
    prefs.putFloat(PREFS_KEY, R0);
    prefs.end();

    Serial.print("Saved MQ3_R0 to NVS (");
    Serial.print(PREFS_NAMESPACE);
    Serial.print("/");
    Serial.print(PREFS_KEY);
    Serial.print(") = ");
    Serial.print(R0, 2);
    Serial.println(" ohms");

    Serial.println();
    Serial.println("Calibration complete.");
  }
}

void loop() {
  // Nothing; single-run calibrator
}