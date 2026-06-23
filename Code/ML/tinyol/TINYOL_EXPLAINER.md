# TinyOL — On-Device Fine-Tuning

## What This Folder Contains

| File | Purpose |
|---|---|
| `train_tinyol_backbone.py` | PC script — trains the frozen backbone on Batches 1+3 |
| `tinyol_weights.h` | 227 weights from the backbone, ready for the ESP32 |
| `tinyol_benchmark.cpp` | ESP32 benchmark — fine-tunes on Batch 5, reports accuracy before/after |

---

## The Problem TinyOL Solves

You train a model on a PC and deploy it to the ESP32. When a new batch of spinach arrives with slightly different conditions, the model's accuracy may drop. Normally you would need to retrain on a PC and re-flash the device.

TinyOL lets the ESP32 adapt to new data on-device — no PC, no USB cable, no cloud.

---

## How It Works

The network is split into two parts:

**Frozen backbone** — `Input(10) → Dense(16, ReLU)` — 176 parameters (W1 + B1)
This layer learns general patterns from the sensor data. It is trained on the PC and never changed on the device.

**Trainable output layer** — `Dense(3, Softmax)` — 51 parameters (W2 + B2)
This is the part that adapts. Only 51 of the 227 total parameters are updated on-device via SGD.

During fine-tuning, the ESP32 runs a forward pass, computes the cross-entropy gradient for the output layer only, and nudges the 51 weights slightly in the direction that reduces the error. The backbone is never touched.

**Why only the output layer?** Backpropagating through the full network requires storing intermediate activations and computing gradients through every layer — expensive in RAM and CPU time. Updating only the output layer (51 parameters) is fast enough to run on a microcontroller between sensor readings.

---

## Results (Batch 4 test set, 317 samples)

| Stage | Accuracy |
|---|---|
| Before fine-tuning (frozen backbone only) | 21.8% |
| After fine-tuning on Batch 5 (378 samples, 10 epochs) | 45.7% |
| Gain | +23.9 percentage points |

**Fine-tuning details:**
- Fine-tuning data: Batch 5 held-out set (378 samples: 67 Fresh, 24 Aging, 287 Degraded)
- Epochs: 10 (3,780 total weight updates)
- Total fine-tuning time: 94.25 ms
- Time per update: 24.9 µs

The post-fine-tuning accuracy (45.7%) remains below the full AIfES result (67.8%) because only 51 parameters are updated and the Batch 5 fine-tuning set is heavily class-imbalanced (287 Degraded vs 24 Aging). With a more balanced held-out set the improvement would be larger.

---

## Three-Way Comparison in This Project

| Approach | Accuracy | Time | Adapts on-device? |
|---|---|---|---|
| AIfES (float32 inference) | 67.8% | 74.7 µs/inference | No |
| TFLite Micro (INT8 inference) | 66.9% | 304.2 µs/inference | No |
| TinyOL (on-device fine-tuning) | 21.8% → 45.7% | 24.9 µs/update | Yes |

AIfES is the best fit for inference accuracy and speed. TinyOL is the right choice when the device needs to adapt to new data without any PC involvement.

---

## How to Run

**Step 1 — Train the backbone (PC)**
```bash
python train_tinyol_backbone.py
```
Outputs `tinyol_weights.h` into this folder.

**Step 2 — Flash and benchmark (ESP32)**

In the `ESP32/` folder:
```bash
pio run -e tinyol_benchmark --target upload
```

The benchmark loads the frozen backbone from `tinyol_weights.h`, evaluates on Batch 4, fine-tunes on Batch 5, then evaluates on Batch 4 again and prints accuracy before and after.

---

## Weight Layout

The 227 weights are stored flat in `tinyol_weights.h`:

```
Indices [0–159]   — W1 (10×16, backbone input→hidden weights)
Indices [160–175] — B1 (16 hidden biases)
Indices [176–223] — W2 (16×3, output weights)
Indices [224–226] — B2 (3 output biases)
```

On the ESP32, `tinyol_benchmark.cpp` copies W2 and B2 into separate mutable arrays at startup. Only those arrays are updated during fine-tuning.
