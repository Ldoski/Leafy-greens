#include <Arduino.h>
#include <nvs_flash.h>

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n=================================");
  Serial.println("     NVS ERASE UTILITY");
  Serial.println("=================================\n");
  
  Serial.println("Erasing all NVS partitions...");
  
  esp_err_t err = nvs_flash_erase();
  if (err == ESP_OK) {
    Serial.println("✓ NVS erased successfully!");
    
    err = nvs_flash_init();
    if (err == ESP_OK) {
      Serial.println("✓ NVS reinitialized successfully!");
      Serial.println("\n=================================");
      Serial.println("  NVS CLEARED - READY TO USE");
      Serial.println("=================================\n");
      Serial.println("You can now upload your main code.");
    } else {
      Serial.print("✗ NVS init failed with error: ");
      Serial.println(err);
    }
  } else {
    Serial.print("✗ NVS erase failed with error: ");
    Serial.println(err);
  }
}

void loop() {
  delay(1000);
}