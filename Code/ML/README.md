# ML Pipeline

Three scripts in dependency order. Run them top to bottom.

```
1. aifes/prepare_dataset_NIR.py
2. aifes/train_model_NIR.py
3. tinyol/train_tinyol_backbone.py
```

---

## Scripts

### 1. `aifes/prepare_dataset_NIR.py`

Reads the raw batch CSVs, applies all exclusions (eCO2 dropped, TVOC saturated values masked, null MQ3 rows removed, SGP30 dropout rows flagged), runs per-batch NIR tertile labeling, computes delta-TVOC features, normalises, and writes `train_nir.csv` and `test_nir.csv` to `aifes/output_nir/`.

Train: Batches 1 + 3 (670 samples). Test: Batch 4 (317 samples). Batch 5 (378 samples) is written separately as the held/TinyOL set.

### 2. `aifes/train_model_NIR.py`

Reads the CSVs from step 1. Trains the 3-class freshness model (Input(10) → Dense(16, ReLU) → Dense(3, Softmax), 227 weights). Exports:
- `aifes/output_nir/aifes_weights_nir.h` — AIfES float32 weight array for the ESP32
- `tflite/tflm_model_nir.h` — TFLite Micro INT8 model for the ESP32

Results: AIfES 67.8% accuracy, 74.7 µs/inference. TFLite Micro 66.9% accuracy, 304.2 µs/inference.

### 3. `tinyol/train_tinyol_backbone.py`

Trains a deliberately weak backbone (Batch 1 only) so TinyOL has room to improve on-device. The output layer is left trainable; the hidden layer is frozen during on-device fine-tuning. Exports `tinyol/tinyol_weights.h`.

The backbone is intentionally undertrained — if it were trained on all available batches it would already generalise well and TinyOL would have nothing to learn.

---

## Features (10 inputs)

| Feature | Source |
|---|---|
| node1_temp, node1_hum | DHT22, Node 1 |
| node1_tvoc | SGP30, Node 1 |
| node1_mq3_ppm | MQ3, Node 1 |
| node2_temp, node2_hum | DHT22, Node 2 |
| node2_tvoc | SGP30, Node 2 |
| node2_mq3_ppm | MQ3, Node 2 |
| delta_node1_tvoc | Per-cycle TVOC change, Node 1 |
| delta_node2_tvoc | Per-cycle TVOC change, Node 2 |

Master node excluded — Spearman |r| < 0.05 with NIR label (ambient air, no spoilage signal). eCO2 excluded — derived from TVOC by SGP30 firmware, saturates within hours.

## Labels (NIR 855 nm, not an input feature)

Labels are assigned per batch using the AS7341 NIR channel. The NIR range for each batch is split into tertiles: Fresh (0), Aging (1), Degraded (2). NIR is used only for labeling — it is not fed to the model.

## Outputs

| File | Written by | Used by |
|---|---|---|
| `aifes/output_nir/train_nir.csv` | prepare_dataset | train_model, train_backbone |
| `aifes/output_nir/test_nir.csv` | prepare_dataset | train_model, train_backbone |
| `aifes/output_nir/aifes_weights_nir.h` | train_model | ESP32 (aifes_inference_nir env) |
| `tflite/tflm_model_nir.h` | train_model | ESP32 (tflm_inference_nir env) |
| `tinyol/tinyol_weights.h` | train_backbone | ESP32 (tinyol benchmark) |

The generated `.h` files and CSVs are excluded from git (see `.gitignore`) — re-run the scripts to regenerate them.
