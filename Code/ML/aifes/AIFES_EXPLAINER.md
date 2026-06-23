# AIfES — Spinach Freshness Classifier

## What This Folder Contains

This folder holds everything needed to train and run the AIfES 3-class spinach freshness classifier on an ESP32.

| File | Purpose |
|---|---|
| `prepare_dataset_NIR.py` | Loads sensor + spectral CSVs, assigns NIR freshness labels, exports C headers |
| `train_model_NIR.py` | Trains the Keras model, exports AIfES weights and TFLite model |
| `aifes_inference_nir.cpp` | ESP32 benchmark — measures inference accuracy and speed |
| `output_nir/aifes_weights_nir.h` | 227 trained float32 weights ready for the ESP32 |
| `output_nir/mould_prediction_dataset_nir.h` | Batch 4 test set (317 samples) as C arrays |
| `output_nir/held_out_dataset.h` | Batch 5 held-out set (378 samples) used by TinyOL |
| `output_nir/combined_training_dataset_nir.h` | Full training set (Batches 1+3) as C arrays |
| `output_nir/train_nir.csv` / `test_nir.csv` | Train/test splits as CSV |
| `output_nir/training_report_nir.json` | Last training run results |
| `output_nir/dataset_stats_nir.json` | Dataset statistics and normalisation parameters |

---

## The Model

**Architecture:** `Input(10) → Dense(16, ReLU) → Dense(3, Softmax)`

**Parameters:** 227 float32 weights
- W1: 10×16 = 160 weights
- B1: 16 biases
- W2: 16×3 = 48 weights
- B2: 3 biases

**Input features (10):**

| Feature | Node |
|---|---|
| Temperature | Node 1, Node 2 |
| Humidity | Node 1, Node 2 |
| TVOC (SGP30) | Node 1, Node 2 |
| MQ3 gas (ppm) | Node 1, Node 2 |
| Delta TVOC | Node 1, Node 2 |

**Output classes:** Fresh (0), Aging (1), Degraded (2) — argmax of softmax probabilities.

**Labels:** Derived from the AS7341 NIR 855 nm channel using per-batch tertile thresholds. The NIR channel is not a model input — it is used only to assign labels during dataset preparation.

---

## How to Regenerate Everything

**Step 1 — Prepare dataset**
```bash
python prepare_dataset_NIR.py
```
Outputs into `output_nir/`: train/test CSVs, `mould_prediction_dataset_nir.h`, `held_out_dataset.h`, `combined_training_dataset_nir.h`, `dataset_stats_nir.json`.

**Step 2 — Train model**
```bash
python train_model_NIR.py
```
Outputs into `output_nir/`: `aifes_weights_nir.h`, `training_report_nir.json`.
Also outputs into `../tflite/`: `tflm_model_nir.h` (INT8 quantised TFLite flatbuffer).

**Step 3 — Flash and benchmark**

In the `ESP32/` folder:
```bash
pio run -e aifes_inference_nir --target upload
```

---

## Benchmark Results (Batch 4 test set, 317 samples, 100 repeats)

| Metric | Value |
|---|---|
| Overall accuracy | 67.8% |
| Inference time | 74.7 µs |
| Fresh recall | 75.9% (110/145) |
| Aging recall | 1.5% (1/68) |
| Degraded recall | 100% (104/104) |

**Confusion matrix:**

|  | Pred Fresh | Pred Aging | Pred Degraded |
|---|---|---|---|
| Actual Fresh (145) | 110 | 0 | 35 |
| Actual Aging (68) | 2 | 1 | 65 |
| Actual Degraded (104) | 0 | 0 | 104 |

The Aging class collapse (1.5% recall) is a data problem — the middle NIR tertile boundary sits where the NIR distribution is densest, making it hard to separate from Degraded. Both AIfES and TFLite show the same pattern, confirming it is not a model issue.

---

## Training Setup

- **Train batches:** 1 and 3 (670 samples combined)
- **Test batch:** 4 (317 samples, held out)
- **Held-out batch:** 5 (378 samples, used for TinyOL fine-tuning only)
- **Normalisation:** min-max, fitted on Batches 1+3 only
- **Class weights:** inverse frequency (Fresh, Aging, Degraded)
- **Loss:** sparse categorical cross-entropy
- **Early stopping:** patience=20 on val_accuracy
