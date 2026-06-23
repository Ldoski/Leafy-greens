# SmartShelfLife — Leafy Greens Freshness Sensing

Masters thesis project, Smart Systems Engineering, Hanze University of Applied Sciences.

An ESP32-based multi-node sensor system that monitors spinach freshness in real time using gas sensors, environmental sensors, and an AS7341 multispectral sensor. A 3-class freshness classifier (Fresh / Aging / Degraded) runs directly on the ESP32 using AIfES. TinyOL enables on-device fine-tuning without any PC connection.

---

## System Overview

```
[ Node 1 (inside) ]──┐
[ Node 2 (inside) ]──┤  ESP-NOW  ──►  [ Master Node ]──UART──►  [ Raspberry Pi ]
                      │                      │                         │
                      └──────────────────────┘                   Polls AS7341
                                                                  Captures timelapse
                                                                  Logs CSV data
[ AS7341 ESP32-C3 ]──USB──────────────────────────────────────────►  Pi
```

Each sensor node carries: DHT22 (temperature, humidity), SGP30 (TVOC, eCO2), MQ3 (ethanol).
The AS7341 reads 11 spectral channels (415–855 nm) including NIR at 855 nm.

The master node aggregates readings from both slave nodes via ESP-NOW, runs the AIfES freshness classifier on the combined 10-feature vector, and sends all data to the Pi over UART every 15 minutes.

NIR at 855 nm is used as a labeling signal only. It is not a model input.

---

## Repository Structure

```
Code/
  ESP32/              ESP32 firmware (PlatformIO)
    platformio.ini    Build environments: master_node, slave_node, aifes, tflm, tinyol
    master_node.cpp        Master node firmware — data aggregation, AIfES inference, UART output
    master_node_live.cpp   Live inference variant with real-time classification output
    slave_node.cpp         Slave node firmware — sensor reading, ESP-NOW broadcast
    AS7341.cpp             AS7341 multispectral node firmware — on-demand READ via USB serial
    README.md

  ML/                 Python ML pipeline (run in order)
    aifes/
      prepare_dataset_NIR.py    Step 1 — load batches, label, normalise, export CSVs
      train_model_NIR.py        Step 2 — train AIfES + TFLite Micro model, export .h files
    tinyol/
      train_tinyol_backbone.py  Step 3 — train weak backbone for TinyOL fine-tuning
    README.md

  Raspberry_pi/       Pi data collection service
    uart_data_collector.py      Receives ESP32 UART, polls AS7341, captures timelapse
    README.md

  Dashboard/          Browser-based data explorer
    imu.html          Open directly in browser — no server required
    README.md

  Analysis/           Pipeline analysis notebook
    batch_analysis.ipynb        Data quality, exclusions, correlation, feature distributions
    README.md

Experiments/          Publication figure notebooks and ground test data
  experiment_analysis.ipynb     NIR time-series, spoilage markers, spectral channels
  test_analysis.ipynb           AS7341 ground test analysis (5 conditions)
  ground_test/                  Ground test CSVs (natural light, dark, movement, 40°C, 88% humidity)
  Ground_Test_Protocol_v3.pdf   Ground test protocol document
  README.md
```

---

## ML Pipeline

Three scripts in dependency order:

```
python Code/ML/aifes/prepare_dataset_NIR.py     # generates train/test CSVs
python Code/ML/aifes/train_model_NIR.py         # trains AIfES + TFLite Micro models
python Code/ML/tinyol/train_tinyol_backbone.py  # trains weak TinyOL backbone
```

Input data lives in `sensor_data/project2_data/Lidl_batches/` (not committed — too large).

**10 input features:** node1_temp, node1_hum, node1_tvoc, node1_mq3_ppm, node2_temp, node2_hum, node2_tvoc, node2_mq3_ppm, delta_node1_tvoc, delta_node2_tvoc.

**Batches:** Train = 1 + 3 (670 samples). Test = 4 (317 samples). Held (TinyOL) = 5 (378 samples). Batch 2 excluded (Node 1 communication failure).

---

## Results

### AIfES Float32 — on-device inference (EXP-010)

| Metric | Value |
|---|---|
| Architecture | Input(10) → Dense(16, ReLU) → Dense(3, Softmax) |
| Parameters | 227 floats |
| Overall accuracy | 67.8% (215 / 317) |
| Inference time | 74.7 µs |

Per-class: Fresh 75.9%, Aging 1.5%, Degraded 100.0%. Aging class collapse is a labeling boundary issue, not a framework issue confirmed present in both AIfES and TFLite Micro.

### TinyOL On-Device Fine-Tuning (EXP-011)

| Metric | Value |
|---|---|
| Frozen params | 176 (hidden layer) |
| Trainable params | 51 (output layer) |
| Accuracy before fine-tuning | 21.8% |
| Accuracy after fine-tuning | 45.7% |
| Time per update | 24.9 µs |
| Total training time (10 epochs, 378 samples) | 94.25 ms |
| Heap leak | 0 B |

### TFLite Micro INT8 (EXP-012)

| Metric | Value |
|---|---|
| Model size | 2,688 bytes |
| Overall accuracy | 66.9% |
| Inference time | 304.2 µs |

AIfES is 4× faster than TFLite Micro on ESP32 because INT8 ops run in software; AIfES float32 benefits from CMSIS-optimised dot products.

---

## Dependencies

**Python (ML + analysis):**
```
tensorflow, numpy, pandas, matplotlib, scipy
```

**Python (Raspberry Pi):**
```
pyserial, picamera2, lgpio  (or RPi.GPIO on Pi 4)
```

**ESP32 (PlatformIO):** libraries are managed by PlatformIO via `platformio.ini`. Build with `pio run -e <environment>`.

---

## Related

- [Thesis-Edge-AI](https://github.com/Ciaranm1999/Thesis-Edge-AI) — companion repo: TinyOL on-device learning for strawberry mould detection on ESP32.
