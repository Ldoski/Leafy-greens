# ESP32 Firmware — Setup and Run Guide

This folder contains all firmware for the SmartShelfLife leafy greens system.
There are four physical nodes — three standard ESP32 dev boards and one ESP32-C3 for the AS7341 sensor PCB.

---

## Node Overview

| Node | Board | Role | Code to flash |
|---|---|---|---|
| Node 0 | ESP32 dev board | Master — aggregates data, logs to SD, runs inference | `master_node.cpp` |
| Node 1 | ESP32 dev board | Slave — reads sensors, sends to master via ESP-NOW | `slave_node.cpp` |
| Node 2 | ESP32 dev board | Slave — reads sensors, sends to master via ESP-NOW | `slave_node.cpp` |
| Node 3 | ESP32-C3 Dev Module | AS7341 spectral sensor board | `AS7341.cpp` |

Nodes 0, 1, and 2 each carry: DHT22 (temp/humidity), SGP30 (TVOC/eCO2), MQ3 (alcohol gas).
Node 3 carries only the AS7341 11-channel multispectral sensor.

---

## Setup Order for Each New ESP32 (Nodes 0, 1, 2)

Follow this order every time you set up a node from scratch:

### Step 1 — Test all sensors first
Flash `test_all_sensors` to confirm the DHT22, SGP30, and MQ3 are all wired and responding correctly before doing anything else. Fix any hardware issues before proceeding.

```
pio run -e test_all_sensors --target upload
```

Open Serial Monitor at **115200 baud**. You should see live readings from all three sensors. If any sensor reads 0 or NaN, check the wiring.

### Step 2 — Erase NVS
Flash `nvs_erase` to wipe any leftover MQ3 calibration values stored in flash from a previous run. Skip this only if you are sure the NVS is already clean.

```
pio run -e nvs_erase --target upload
```

Open Serial Monitor at **115200 baud**. Wait for "NVS CLEARED — READY TO USE" before proceeding.

### Step 3 — Calibrate MQ3
Place the ESP32 in **clean open air** (not inside the chamber, not near the spinach). Flash `mq3_calibration` and let it take 30 samples over ~60 seconds. It measures the MQ3 sensor resistance in clean air (R0) and saves it to NVS. This value is used by the main firmware to convert raw ADC readings into ppm.

```
pio run -e mq3_calibration --target upload
```

Open Serial Monitor at **115200 baud**. Wait for "Calibration complete." before proceeding. The R0 value is printed and saved automatically.

### Step 4 — Flash main firmware
Flash `slave_node` for Nodes 1 and 2, or `master_node` for Node 0.

```
# For Node 0 (master):
pio run -e master_node --target upload

# For Node 1 or Node 2 (slave):
pio run -e slave_node --target upload
```

---

## AS7341 Spectral Sensor Board (Node 3)

**This board uses an ESP32-C3 Dev Module, not a standard ESP32.**

Before uploading, change the board target in PlatformIO:
- In `platformio.ini`, the `as7341_test` environment uses `board = esp32dev` by default for the project — for this board you must select **ESP32-C3 Dev Module** in your IDE or change the env board setting.

In VS Code with PlatformIO: click the board selector at the bottom of the screen and choose **Espressif ESP32-C3 Dev Module** before uploading.

```
pio run -e as7341_test --target upload
```

Open Serial Monitor at **115200 baud**. The board waits for a "READ" command from the Raspberry Pi over UART and responds with all 11 channel values as a CSV line.

---

## ML Benchmark Sketches

These are for thesis measurements only — not part of normal data collection.

| Environment | What it does |
|---|---|
| `aifes_inference_nir` | Runs AIfES float32 3-class inference on Batch 4 test set, 100 repeats. Reports accuracy and timing. Connect PPK2 between BENCHMARK START and END markers. |
| `tflm_inference_nir` | Same but with TFLite Micro INT8. Runs on same Batch 4 test set. |
| `tinyol_benchmark` | Fine-tunes the output layer on Batch 5 (378 samples, 10 epochs) then evaluates on Batch 4. Reports accuracy before and after. |

Flash these from the relevant PlatformIO environment. All use GPIO2 (built-in LED) as the PPK2 timing window: LED ON = benchmark running, LED OFF = done.

---

## Utility Sketches

| File | What it does |
|---|---|
| `nvs_eraser.cpp` | Erases all NVS flash storage. Use before MQ3 calibration on a new or recycled board. |
| `mq3_calibration_code.cpp` | Takes 30 samples in clean air, computes MQ3 R0, saves to NVS. Must be done before flashing slave/master. |
| `test_all_sensors.cpp` | Reads DHT22, SGP30, and MQ3 every 5 seconds and prints to serial. Use to confirm all sensors are wired correctly before setup. |

---

## Serial Monitor

All sketches use **115200 baud**. In PlatformIO:
```
pio device monitor --baud 115200
```
Or use the serial monitor button in VS Code.

---

## Pin Reference (Nodes 0, 1, 2)

| Sensor | Interface | Pin |
|---|---|---|
| DHT22 | Digital | GPIO 4 |
| SGP30 | I2C (address 0x58) | SDA/SCL default |
| MQ3 | ADC | GPIO 34 |
| Master UART TX to Pi | UART2 | GPIO 17 |
| Master UART RX from Pi | UART2 | GPIO 16 |
