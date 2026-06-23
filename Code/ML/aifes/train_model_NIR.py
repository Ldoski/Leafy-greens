"""
train_model_NIR.py
==================
Trains a 3-class freshness prediction model using NIR-derived labels.
Read from output_nir/train_nir.csv and test_nir.csv produced by prepare_dataset_NIR.py.

Exports:
  aifes_weights_nir.h   -- float32 weight arrays for AIfES (3-class output)
  training_report_nir.json

Run AFTER prepare_dataset_NIR.py.

Changes from train_model.py are printed at the end of every run.
"""

import json
import os
import sys
from pathlib import Path

# Force UTF-8 on Windows so box-drawing / arrows print correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUT_DIR     = SCRIPT_DIR / "output_nir"
ESP32_DIR   = OUT_DIR   # aifes weights go into output_nir/
TFLITE_DIR  = SCRIPT_DIR.parent / "tflite"
TFLITE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = OUT_DIR / "train_nir.csv"
TEST_CSV  = OUT_DIR / "test_nir.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FEATURES   = 10
HIDDEN_SIZE  = 16
N_CLASSES    = 3                  # Fresh=0, Aging=1, Degraded=2  [CHANGED: was 1]
EPOCHS       = 200
BATCH_SIZE   = 32
LEARNING_RATE = 0.001
CLASS_NAMES  = ["Fresh", "Aging", "Degraded"]

NORM_COLS = [
    "node1_temp_norm", "node1_hum_norm", "node1_tvoc_norm", "node1_mq3_ppm_norm",
    "node2_temp_norm", "node2_hum_norm", "node2_tvoc_norm", "node2_mq3_ppm_norm",
    "delta_node1_tvoc_norm", "delta_node2_tvoc_norm",
]

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("FRESHNESS PREDICTION MODEL TRAINING — NIR 3-CLASS")
print("=" * 60)

if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"Run prepare_dataset_NIR.py first. Missing: {TRAIN_CSV}")

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

X_train = train_df[NORM_COLS].values.astype(np.float32)
y_train = train_df["label"].values.astype(np.int32)
X_test  = test_df[NORM_COLS].values.astype(np.float32)
y_test  = test_df["label"].values.astype(np.int32)

print(f"\nLoaded:")
print(f"  Train : {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Test  : {X_test.shape[0]} samples")
print(f"\nTrain class balance:")
for i, name in enumerate(CLASS_NAMES):
    n = int((y_train == i).sum())
    print(f"  {name:<10}: {n:>4}  ({100*n/len(y_train):.0f}%)")
print(f"Test class balance:")
for i, name in enumerate(CLASS_NAMES):
    n = int((y_test == i).sum())
    print(f"  {name:<10}: {n:>4}  ({100*n/len(y_test):.0f}%)")

# ---------------------------------------------------------------------------
# 2. Build model
# [CHANGED] Output: Dense(3, softmax) instead of Dense(1, sigmoid)
# [CHANGED] Loss:   sparse_categorical_crossentropy instead of binary_crossentropy
# ---------------------------------------------------------------------------
print(f"\nBuilding model:")
print(f"  Input({N_FEATURES}) -> Dense({HIDDEN_SIZE}, ReLU) -> Dense({N_CLASSES}, Softmax)")

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.layers.Input(shape=(N_FEATURES,)),
    keras.layers.Dense(HIDDEN_SIZE, activation="relu",    name="hidden"),
    keras.layers.Dense(N_CLASSES,   activation="softmax", name="output"),  # [CHANGED]
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",   # [CHANGED]
    metrics=["accuracy"],
)
model.summary()

# ---------------------------------------------------------------------------
# Validation split: hold out the last training batch for val.
# If only one batch is available, fall back to a random 80/20 split.
# ---------------------------------------------------------------------------
train_batches = train_df["batch"].unique().tolist()

if len(train_batches) > 1:
    val_batch = sorted(train_batches)[-1]
    val_mask  = train_df["batch"] == val_batch
    X_tr  = train_df.loc[~val_mask, NORM_COLS].values.astype(np.float32)
    y_tr  = train_df.loc[~val_mask, "label"].values.astype(np.int32)
    X_val = train_df.loc[ val_mask, NORM_COLS].values.astype(np.float32)
    y_val = train_df.loc[ val_mask, "label"].values.astype(np.int32)
    split_desc = f"held out batch '{val_batch}'"
else:
    val_batch = train_batches[0]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_df))
    split = int(0.8 * len(idx))
    tr_idx, val_idx = idx[:split], idx[split:]
    X_tr  = train_df.iloc[tr_idx][NORM_COLS].values.astype(np.float32)
    y_tr  = train_df.iloc[tr_idx]["label"].values.astype(np.int32)
    X_val = train_df.iloc[val_idx][NORM_COLS].values.astype(np.float32)
    y_val = train_df.iloc[val_idx]["label"].values.astype(np.int32)
    split_desc = f"random 80/20 split (only one training batch)"

# [CHANGED] Class weights now cover 3 classes (inverse frequency)
class_counts = np.bincount(y_tr, minlength=N_CLASSES)
class_weight = {
    i: len(y_tr) / (N_CLASSES * max(class_counts[i], 1))
    for i in range(N_CLASSES)
}

print(f"\nValidation split: {split_desc}")
print(f"  Fit  : {len(X_tr)} samples")
print(f"  Val  : {len(X_val)} samples")
print(f"  Class weights: { {CLASS_NAMES[i]: f'{w:.2f}' for i, w in class_weight.items()} }")

# ---------------------------------------------------------------------------
# Feature-class separation diagnostic
# Prints per-class mean for each normalised feature so we can see if the
# sensor signals actually differ between Fresh / Aging / Degraded.
# ---------------------------------------------------------------------------
print("\nFeature means per class (TRAIN set — normalised 0-1):")
print(f"  {'Feature':<28}", end="")
for n in CLASS_NAMES:
    print(f"  {n:>10}", end="")
print()
for col in NORM_COLS:
    print(f"  {col:<28}", end="")
    for i in range(N_CLASSES):
        mask = train_df["label"] == i
        val = train_df.loc[mask, col].mean()
        print(f"  {val:>10.3f}", end="")
    print()

print("\nFeature means per class (TEST set):")
print(f"  {'Feature':<28}", end="")
for n in CLASS_NAMES:
    print(f"  {n:>10}", end="")
print()
for col in NORM_COLS:
    print(f"  {col:<28}", end="")
    for i in range(N_CLASSES):
        mask = test_df["label"] == i
        val = test_df.loc[mask, col].mean()
        print(f"  {val:>10.3f}", end="")
    print()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=20, restore_best_weights=True
)

print(f"\nTraining for up to {EPOCHS} epochs (early stop on val_accuracy, patience=20)...")
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1,
)

# ---------------------------------------------------------------------------
# 3. Evaluate
# [CHANGED] Per-class precision / recall / F1 instead of threshold sweep
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_prob    = model.predict(X_test, verbose=0)          # shape (N, 3)
y_pred    = np.argmax(y_prob, axis=1)

print(f"\n  Overall accuracy : {acc*100:.1f}%")
print(f"  Loss             : {loss:.4f}")

print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
header = f"  {'':>12}" + "".join(f"  Pred {n:<10}" for n in CLASS_NAMES)
print(header)
cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for true, pred in zip(y_test, y_pred):
    cm[true][pred] += 1

per_class = {}
for i, name in enumerate(CLASS_NAMES):
    row_str = f"  {f'Actual {name}':<12}" + "".join(f"  {cm[i][j]:>15}" for j in range(N_CLASSES))
    print(row_str)
    tp = cm[i][i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    prec   = tp / max(tp + fp, 1)
    rec    = tp / max(tp + fn, 1)
    f1     = 2 * prec * rec / max(prec + rec, 1e-9)
    per_class[name] = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                       "support": int(cm[i, :].sum())}

print(f"\n  Per-class metrics:")
print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print(f"  {'-'*52}")
for name, m in per_class.items():
    print(f"  {name:<12} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10}")

macro_f1 = np.mean([m["f1"] for m in per_class.values()])
print(f"\n  Macro F1: {macro_f1:.3f}")

# ---------------------------------------------------------------------------
# 4. Export AIfES weights header (float32)
# [CHANGED] Output layer is now (16,3) instead of (16,1) — 3 output neurons
# ---------------------------------------------------------------------------
print("\nExporting AIfES weights header...")

hidden_layer = model.get_layer("hidden")
output_layer = model.get_layer("output")

W1, B1 = hidden_layer.get_weights()   # (10, 16), (16,)
W2, B2 = output_layer.get_weights()   # (16, 3),  (3,)   [CHANGED: was (16,1)]

flat_weights = np.concatenate([
    W1.flatten().astype(np.float32),   # 160 floats
    B1.flatten().astype(np.float32),   # 16 floats
    W2.flatten().astype(np.float32),   # 48 floats  [CHANGED: was 16]
    B2.flatten().astype(np.float32),   # 3 floats   [CHANGED: was 1]
])
n_weights = len(flat_weights)
vals = ", ".join(f"{v:.8f}f" for v in flat_weights)

aifes_lines = [
    "/*",
    " * aifes_weights_nir.h",
    " * Auto-generated by train_model_NIR.py",
    " *",
    " * Pre-trained weights for AIfES 3-class freshness inference on ESP32.",
    " * Architecture: Input(10) -> Dense(16, ReLU) -> Dense(3, Softmax)",
    " * Classes: 0=Fresh  1=Aging  2=Degraded",
    " *",
    f" * Test accuracy: {acc*100:.1f}%  |  Macro F1: {macro_f1:.3f}",
    " *",
    " * Weight layout for AIFES_E_inference_fnn_f32:",
    f" *   W1(10x16) + B1(16) + W2(16x3) + B2(3)",
    f" *   Total: {n_weights} floats",
    " */",
    "",
    "#pragma once",
    "",
    f"#define AIFES_NIR_INPUT_SIZE   {N_FEATURES}",
    f"#define AIFES_NIR_HIDDEN_SIZE  {HIDDEN_SIZE}",
    f"#define AIFES_NIR_OUTPUT_SIZE  {N_CLASSES}",
    f"#define AIFES_NIR_N_WEIGHTS    {n_weights}",
    "",
    "/* Class index: 0=Fresh  1=Aging  2=Degraded */",
    f"/* Layout: W1[10*16] | B1[16] | W2[16*3] | B2[3] */",
    f"static float aifes_nir_flat_weights[{n_weights}] = {{",
    f"  {vals}",
    "};",
]

aifes_path = ESP32_DIR / "aifes_weights_nir.h"
with open(aifes_path, "w") as f:
    f.write("\n".join(aifes_lines) + "\n")
print(f"  Saved: {aifes_path}")

# ---------------------------------------------------------------------------
# 5. Export TFLite INT8 quantised model header (tflm_model_nir.h)
# Same pipeline as Thesis-Edge-AI train_model.py — only label-related diffs:
#   N_CLASSES=3, Softmax output, variable names use _nir suffix.
# ---------------------------------------------------------------------------
print("\nExporting TFLite INT8 quantised model...")

def representative_dataset():
    for i in range(len(X_train)):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32
# tflm_esp32 does not support per-channel quantization for Dense layers
converter._experimental_disable_per_channel_quantization_for_dense_layers = True

tflite_model     = converter.convert()
tflite_size_kb   = len(tflite_model) / 1024
print(f"  TFLite model size: {tflite_size_kb:.1f} KB")

hex_vals = ", ".join(f"0x{b:02x}" for b in tflite_model)
n_bytes  = len(tflite_model)

tflm_header = f"""/*
 * tflm_model_nir.h
 * Auto-generated by train_model_NIR.py
 *
 * INT8 quantised TFLite flatbuffer for TF Lite Micro inference on ESP32.
 * Architecture: Input(10) -> Dense(16, ReLU) -> Dense(3, Softmax)
 * Labels: 0=Fresh  1=Aging  2=Degraded
 *
 * Test accuracy: {acc*100:.1f}%  |  Macro F1: {macro_f1:.3f}
 * Model size: {tflite_size_kb:.1f} KB ({n_bytes} bytes)
 *
 * Include this file and pass g_tflm_model_nir to the EloquentTinyML interpreter.
 */

#pragma once
#include <stdint.h>

const unsigned char g_tflm_model_nir[] = {{
  {hex_vals}
}};
const unsigned int g_tflm_model_nir_len = {n_bytes};
"""

tflm_path = TFLITE_DIR / "tflm_model_nir.h"
with open(tflm_path, "w") as f:
    f.write(tflm_header)
print(f"  Saved: {tflm_path}")

# ---------------------------------------------------------------------------
# 6. Save training report
# ---------------------------------------------------------------------------
report = {
    "label_system":       "NIR 3-class (Fresh=0, Aging=1, Degraded=2)",
    "architecture":       f"Input({N_FEATURES}) -> Dense({HIDDEN_SIZE}, ReLU) -> Dense({N_CLASSES}, Softmax)",
    "epochs_trained":     len(history.history.get("loss", [])),
    "test_loss":          float(loss),
    "test_accuracy":      float(acc),
    "macro_f1":           float(macro_f1),
    "per_class_metrics":  per_class,
    "confusion_matrix":   cm.tolist(),
    "class_names":        CLASS_NAMES,
}
report_path = OUT_DIR / "training_report_nir.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"  Saved: {report_path}")

# ---------------------------------------------------------------------------
# Changes report — printed every run so the diff stays visible
# ---------------------------------------------------------------------------
CHANGES = """
╔══════════════════════════════════════════════════════════════╗
║            CHANGES FROM ORIGINAL PIPELINE                    ║
╠══════════════════════════════════════════════════════════════╣
║  prepare_dataset.py  →  prepare_dataset_NIR.py               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  LABELS                                                      ║
║  Original : binary (0=no-mould, 1=mould)                     ║
║             derived from mould_start timestamps in CSV       ║
║  NIR      : 3-class (0=Fresh, 1=Aging, 2=Degraded)          ║
║             derived from AS7341 NIR channel tertile zones    ║
║             per-batch, direction-aware, 7-pt rolling smooth  ║
║                                                              ║
║  DATA SOURCE                                                 ║
║  Original : all_batches_ml_curated.csv (pre-merged)          ║
║  NIR      : per-batch Lidl_sensors_testN.csv +               ║
║             Lidl_multispectral_testN.csv (joined on time)    ║
║                                                              ║
║  TRAIN/TEST SPLIT                                            ║
║  Original : Batches 1-4 train / Batch 5 test                 ║
║  NIR      : Batches 1-2 train / Batch 3 test                 ║
║             (only 3 batches have NIR data currently)         ║
║                                                              ║
║  NULL HANDLING                                               ║
║  Original : drop rows where node1 OR node2 MQ3 is null       ║
║  NIR      : drop only where node2 MQ3 is null                ║
║             node1 nulls imputed by per-batch median          ║
║             (node1 disconnects intermittently across batches) ║
║                                                              ║
║  LIGHTING ARTIFACTS                                          ║
║  Original : not applicable                                   ║
║  NIR      : clear channel > 1.6× median flagged and          ║
║             excluded from NIR zone computation               ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  train_model.py  →  train_model_NIR.py                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OUTPUT LAYER                                                ║
║  Original : Dense(1, sigmoid)  → single 0-1 probability      ║
║  NIR      : Dense(3, softmax)  → 3 class probabilities       ║
║                                                              ║
║  LOSS FUNCTION                                               ║
║  Original : binary_crossentropy                              ║
║  NIR      : sparse_categorical_crossentropy                  ║
║             ('sparse' = integer labels, not one-hot)         ║
║                                                              ║
║  LABEL DTYPE                                                 ║
║  Original : float32  (needed for binary CE)                  ║
║  NIR      : int32    (needed for sparse CE)                  ║
║                                                              ║
║  CLASS WEIGHTS                                               ║
║  Original : 2-class inverse frequency {0: 1.0, 1: ratio}     ║
║  NIR      : 3-class inverse frequency across Fresh/Aging/    ║
║             Degraded                                         ║
║                                                              ║
║  EVALUATION                                                  ║
║  Original : threshold sweep (0.1→0.9) to maximise F1        ║
║             single confusion matrix (TP/TN/FP/FN)           ║
║  NIR      : argmax prediction, no threshold needed           ║
║             per-class precision/recall/F1 + macro F1         ║
║             3×3 confusion matrix                             ║
║                                                              ║
║  WEIGHT LAYOUT                                               ║
║  Original : W1(10×16) B1(16) W2(16×1) B2(1) = 193 floats    ║
║  NIR      : W1(10×16) B1(16) W2(16×3) B2(3) = 227 floats    ║
║                                                              ║
║  OUTPUT FILES                                                ║
║  Original : aifes_weights.h  /  training_report.json        ║
║  NIR      : aifes_weights_nir.h  /  training_report_nir.json ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

print(CHANGES)

print("=" * 60)
print("DONE")
print(f"  Epochs trained : {len(history.history.get('loss', []))}")
print(f"  Test accuracy  : {acc*100:.1f}%")
print(f"  Macro F1       : {macro_f1:.3f}")
print(f"  Weights saved  : {aifes_path}")
print(f"  Report saved   : {report_path}")
print("=" * 60)
