"""
train_tinyol_backbone.py
========================
Trains a WEAK backbone for TinyOL on-device fine-tuning experiments.
Adapted for the Leafy Greens NIR 3-class pipeline (Fresh / Aging / Degraded).

RATIONALE (weak backbone):
  A backbone trained on all available batches already generalises well to
  new data, leaving no accuracy gap for TinyOL to close. To demonstrate
  that TinyOL genuinely helps, this script trains the backbone on Batch 1
  ONLY. Batch 2 is used for early-stopping validation. Batch 3 is the
  held-out evaluation set (never seen during backbone training).

  TinyOL then fine-tunes the output layer on new incoming batches (Batch 4+)
  entirely on the ESP32, with no PC connection.

Data split:
  Backbone fit (Batch 1)  : gradient updates during PC training
  Backbone val (Batch 2)  : early stopping only, no gradient updates
  Evaluation  (Batch 3)   : held-out test, never seen in any training

Architecture (matches NIR AIfES model):
  Input(10) -> Dense(16, ReLU) [FROZEN in TinyOL] -> Dense(3, Softmax) [TRAINABLE]
  Total weights: 160 + 16 + 48 + 3 = 227 floats

Output:
  tinyol_weights.h  -- 227-float weight array for tinyol_benchmark.cpp
"""

import os
from pathlib import Path

os.add_dll_directory("C:/Windows/System32")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent                        # Code/ML/tinyol/
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent            # Leafy_Greens_Project/
DATA_DIR   = PROJECT_ROOT / "Code" / "ML" / "aifes" / "output_nir"
OUT_DIR    = SCRIPT_DIR                                   # write next to this script

TRAIN_CSV = DATA_DIR / "train_nir.csv"
TEST_CSV  = DATA_DIR / "test_nir.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FEATURES  = 10
HIDDEN_SIZE = 16
OUTPUT_SIZE = 3       # Fresh=0, Aging=1, Degraded=2
EPOCHS      = 200
BATCH_SIZE  = 32
LEARNING_RATE = 0.001

NORM_COLS = [
    "node1_temp_norm", "node1_hum_norm", "node1_tvoc_norm", "node1_mq3_ppm_norm",
    "node2_temp_norm", "node2_hum_norm", "node2_tvoc_norm", "node2_mq3_ppm_norm",
    "delta_node1_tvoc_norm", "delta_node2_tvoc_norm",
]

CLASS_NAMES = {0: "Fresh", 1: "Aging", 2: "Degraded"}

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("TINYOL BACKBONE TRAINING — NIR 3-class (Batch 1 only)")
print("=" * 60)

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"Missing: {TRAIN_CSV}\n"
        f"Run prepare_dataset_NIR.py first to generate the normalised CSVs."
    )

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"\nLoaded train: {len(train_df)} rows | test: {len(test_df)} rows")
print(f"Train batches: {sorted(train_df['batch'].unique())}")
print(f"Test batches : {sorted(test_df['batch'].unique())}")

# ---------------------------------------------------------------------------
# 2. Split — Batch 1 fit, Batch 2 val, Batch 3 is test set
# ---------------------------------------------------------------------------
fit_mask = train_df["batch"] == "batch1"
val_mask = train_df["batch"] == "batch2"

X_fit = train_df.loc[fit_mask, NORM_COLS].values.astype(np.float32)
y_fit = train_df.loc[fit_mask, "label"].values.astype(np.int32)
X_val = train_df.loc[val_mask, NORM_COLS].values.astype(np.float32)
y_val = train_df.loc[val_mask, "label"].values.astype(np.int32)
X_test = test_df[NORM_COLS].values.astype(np.float32)
y_test = test_df["label"].values.astype(np.int32)

print(f"\nData split:")
print(f"  Backbone fit  (Batch 1 only) : {len(X_fit)} samples")
for c, name in CLASS_NAMES.items():
    print(f"    {name}: {int((y_fit == c).sum())}")

print(f"  Backbone val  (Batch 2)      : {len(X_val)} samples  [early stop only]")
print(f"  Evaluation    (Batch 3)      : {len(X_test)} samples  [never seen in training]")

if len(X_fit) == 0:
    raise RuntimeError("No Batch 1 data found — check batch column values in train_nir.csv")

# ---------------------------------------------------------------------------
# 3. Class weights (inverse frequency, corrects for class imbalance in Batch 1)
# ---------------------------------------------------------------------------
n_total = len(y_fit)
class_weight = {}
for c in range(OUTPUT_SIZE):
    n_c = int((y_fit == c).sum())
    class_weight[c] = n_total / (OUTPUT_SIZE * max(n_c, 1))

print(f"\nClass weights (from Batch 1 distribution):")
for c, name in CLASS_NAMES.items():
    print(f"  {name}: {class_weight[c]:.3f}")

# ---------------------------------------------------------------------------
# 4. Build and train backbone
# ---------------------------------------------------------------------------
print(f"\nBuilding model: Input({N_FEATURES}) -> Dense({HIDDEN_SIZE}, ReLU) -> Dense({OUTPUT_SIZE}, Softmax)")

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.layers.Input(shape=(N_FEATURES,)),
    keras.layers.Dense(HIDDEN_SIZE, activation="relu",    name="hidden"),
    keras.layers.Dense(OUTPUT_SIZE,  activation="softmax", name="output"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=20, restore_best_weights=True
)

print(f"\nTraining for up to {EPOCHS} epochs (early stop on val_accuracy, patience=20)...")
history = model.fit(
    X_fit, y_fit,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1,
)

epochs_trained = len(history.history["loss"])
print(f"\nStopped at epoch {epochs_trained}")

# ---------------------------------------------------------------------------
# 5. Evaluate on Batch 3 (held-out)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("EVALUATION ON BATCH 3 (held-out — never seen in training)")
print("=" * 60)

y_pred_probs = model.predict(X_test, verbose=0)   # shape (N, 3)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = float((y_pred == y_test).mean())
print(f"\nOverall accuracy: {accuracy*100:.1f}%")

print("\nPer-class breakdown:")
print(f"  {'Class':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
print(f"  {'-'*42}")
for c, name in CLASS_NAMES.items():
    mask = y_test == c
    if mask.sum() == 0:
        continue
    acc_c = float((y_pred[mask] == c).mean())
    print(f"  {name:<12} {int((y_pred[mask]==c).sum()):>8} {int(mask.sum()):>8} {acc_c*100:>9.1f}%")

print("\nConfusion matrix (rows=actual, cols=predicted):")
print(f"  {'':12}", end="")
for name in CLASS_NAMES.values():
    print(f"  {name:>10}", end="")
print()
for r, r_name in CLASS_NAMES.items():
    print(f"  {r_name:<12}", end="")
    for c in range(OUTPUT_SIZE):
        count = int(((y_pred == c) & (y_test == r)).sum())
        print(f"  {count:>10}", end="")
    print()

# ---------------------------------------------------------------------------
# 6. Export tinyol_weights.h (227 floats — same layout as aifes_weights_nir.h)
# ---------------------------------------------------------------------------
print("\nExporting tinyol_weights.h...")

W1, B1 = model.get_layer("hidden").get_weights()   # (10,16), (16,)
W2, B2 = model.get_layer("output").get_weights()   # (16,3),  (3,)

flat_weights = np.concatenate([
    W1.flatten().astype(np.float32),
    B1.flatten().astype(np.float32),
    W2.flatten().astype(np.float32),
    B2.flatten().astype(np.float32),
])
n_weights = len(flat_weights)
vals = ", ".join(f"{v:.8f}f" for v in flat_weights)

print(f"  Weight count: {n_weights} floats")
print(f"    W1({N_FEATURES}x{HIDDEN_SIZE}={N_FEATURES*HIDDEN_SIZE}) + "
      f"B1({HIDDEN_SIZE}) + "
      f"W2({HIDDEN_SIZE}x{OUTPUT_SIZE}={HIDDEN_SIZE*OUTPUT_SIZE}) + "
      f"B2({OUTPUT_SIZE}) = {n_weights}")

header = f"""\
/*
 * tinyol_weights.h
 * Auto-generated by train_tinyol_backbone.py
 *
 * Weak backbone weights for TinyOL on-device fine-tuning.
 * Architecture: Input({N_FEATURES}) -> Dense({HIDDEN_SIZE}, ReLU) -> Dense({OUTPUT_SIZE}, Softmax)
 *
 * TRAINING SPLIT (weak backbone — Batch 1 only):
 *   Backbone fit : Batch 1 ONLY
 *   Backbone val : Batch 2       (early stopping only)
 *   Withheld     : Batch 3       (evaluation, never seen in training)
 *   TinyOL adapt : Batch 4+      (on-device, future batches)
 *
 * Baseline accuracy on Batch 3: {accuracy*100:.1f}%
 * Epochs trained: {epochs_trained}
 *
 * Weight layout: W1[{N_FEATURES}*{HIDDEN_SIZE}] | B1[{HIDDEN_SIZE}] | W2[{HIDDEN_SIZE}*{OUTPUT_SIZE}] | B2[{OUTPUT_SIZE}]
 * Total: {n_weights} floats
 *
 * Classes: 0=Fresh, 1=Aging, 2=Degraded
 */

#pragma once

#define AIFES_NIR_INPUT_SIZE   {N_FEATURES}
#define AIFES_NIR_HIDDEN_SIZE  {HIDDEN_SIZE}
#define AIFES_NIR_OUTPUT_SIZE  {OUTPUT_SIZE}
#define AIFES_NIR_N_WEIGHTS    {n_weights}

static float aifes_flat_weights[{n_weights}] = {{
  {vals}
}};
"""

out_path = OUT_DIR / "tinyol_weights.h"
out_path.write_text(header)
print(f"\nSaved: {out_path}")

print("\n" + "=" * 60)
print("DONE")
print(f"  Baseline accuracy (Batch 3): {accuracy*100:.1f}%")
print(f"  TinyOL will fine-tune output layer on Batch 4+ (on-device)")
print(f"  Weight file: {out_path}")
print("=" * 60)
