#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_AS7341.h>
#include <ArduinoJson.h>

#define SERIAL_BAUD 115200

#define SDA_PIN 4
#define SCL_PIN 3

#define SENSOR_ATIME 29
#define SENSOR_ASTEP 29999
#define SENSOR_GAIN  AS7341_GAIN_256X

Adafruit_AS7341 as7341;
bool sensor_ok = false;

void take_reading();

void setup() {
  Serial.begin(SERIAL_BAUD);

  delay(2000);

  Serial.println("{\"status\":\"booting\",\"firmware\":\"as7341_pi_controlled\"}");

  Wire.begin(SDA_PIN, SCL_PIN);

  if (!as7341.begin()) {
    Serial.println("{\"status\":\"error\",\"msg\":\"AS7341 not found\"}");
    sensor_ok = false;
    return;
  }

  as7341.setATIME(SENSOR_ATIME);
  as7341.setASTEP(SENSOR_ASTEP);
  as7341.setGain(SENSOR_GAIN);

  sensor_ok = true;

  Serial.println("{\"status\":\"ready\",\"sensor\":\"AS7341\"}");
}

void loop() {

  // ONLY respond when Raspberry Pi requests data
  if (Serial.available()) {

    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "READ") {

      if (sensor_ok) {
        take_reading();
      }
      else {
        Serial.println("{\"status\":\"error\",\"msg\":\"sensor not initialised\"}");
      }
    }
  }
}

void take_reading() {

  if (!as7341.readAllChannels()) {
    Serial.println("{\"status\":\"error\",\"msg\":\"readAllChannels failed\"}");
    return;
  }

  JsonDocument doc;

  doc["sensor"] = "AS7341";

  doc["f1_415nm"] = as7341.getChannel(AS7341_CHANNEL_415nm_F1);
  doc["f2_445nm"] = as7341.getChannel(AS7341_CHANNEL_445nm_F2);
  doc["f3_480nm"] = as7341.getChannel(AS7341_CHANNEL_480nm_F3);
  doc["f4_515nm"] = as7341.getChannel(AS7341_CHANNEL_515nm_F4);
  doc["f5_555nm"] = as7341.getChannel(AS7341_CHANNEL_555nm_F5);
  doc["f6_590nm"] = as7341.getChannel(AS7341_CHANNEL_590nm_F6);
  doc["f7_630nm"] = as7341.getChannel(AS7341_CHANNEL_630nm_F7);
  doc["f8_680nm"] = as7341.getChannel(AS7341_CHANNEL_680nm_F8);

  doc["clear"] = as7341.getChannel(AS7341_CHANNEL_CLEAR);
  doc["nir"]   = as7341.getChannel(AS7341_CHANNEL_NIR);

  doc["gain"]  = 256;
  doc["atime"] = SENSOR_ATIME;
  doc["astep"] = SENSOR_ASTEP;

  serializeJson(doc, Serial);
  Serial.println();
}