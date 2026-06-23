"""
prepare_dataset_NIR.py
======================
NIR-based alternative to prepare_dataset.py.

Labels are derived from the AS7341 NIR channel (no manual mould_start timestamps).
The NIR signal is divided into three freshness zones per batch:
  0 = Fresh    (first tertile of NIR travel range)
  1 = Aging    (second tertile)
  2 = Degraded (third tertile)

Zone boundaries auto-scale to each batch's NIR range and are direction-aware
(NIR can rise or fall during degradation depending on lighting/geometry).
This mirrors the logic in imu.html (nirLabelZones / zoneFor).

Pipeline:
  1. Load per-batch sensor + multispectral CSVs
  2. Flag lighting artifacts in spectral data (clear channel spike detection)
  3. Compute per-batch NIR zones from clean readings (7-pt rolling smooth)
  4. Forward-fill NIR label onto 15-min sensor rows
  5. Drop eCO2 columns (unreliable on SGP30)
  6. Drop master node features (ambient air, negligible correlation)
  7. Mark saturated TVOC as NaN (>= 59000 ppb)
  8. Drop rows with null MQ3 readings
  9. Impute remaining TVOC NaN with per-batch median
  10. Compute delta TVOC (rate of change within each batch)
  11. Train/test split: Batch1 + Batch2 = train, Batch3 = test
  12. Normalise features using training-set statistics only
  13. Export CSVs, JSON stats, and C header for ESP32

When new batches arrive, drop their CSVs into the Lidl_batches folder
following the naming convention and update BATCH_IDS below — no other
code changes needed.

Train/test split rationale:
  Train : Batch1, Batch3  (diverse gas levels — batch3 node1 TVOC spans full range)
  Held  : Batch5          (TinyOL on-device fine-tuning)
  Test  : Batch4          (fixed holdout — selected via LOOCV: this fold gave 77.6% / F1=0.622)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent                              # alfes/
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent                   # Leafy_Greens_Project/
BATCH_DIR   = PROJECT_ROOT / "sensor_data" / "project2_data" / "Lidl_batches"
OUT_DIR     = SCRIPT_DIR / "output_nir"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ESP32_DIR   = PROJECT_ROOT / "ML_Training" / "esp32_datasets"
if not ESP32_DIR.exists():
    # Fall back to sibling output folder if Thesis-Edge-AI structure not present
    ESP32_DIR = OUT_DIR
ESP32_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Batch configuration
# Update BATCH_IDS when new batches arrive (add the number, drop CSVs in folder)
# ---------------------------------------------------------------------------
TRAIN_BATCHES = [1, 3]     # Batch 2 dropped: Node1 ~dead, only 3 Degraded NIR samples, incompatible boundaries
HELD_BATCHES  = [5]        # TinyOL on-device fine-tuning (Batch 5: no NULLs)
TEST_BATCHES  = [4]        # Fixed holdout — chosen via LOOCV: train=[1,3] test=[4] gave 77.6%
BATCH_IDS     = sorted(set(TRAIN_BATCHES + HELD_BATCHES + TEST_BATCHES))

USE_GLOBAL_BOUNDARIES = False  # Freeze zone boundaries from training data; apply to all batches

def batch_folder(i):
    return BATCH_DIR / f"Lidl_room_temp_batch{i}"

def sensor_csv(i):
    return batch_folder(i) / f"Lidl_sensors_test{i}.csv"

def spectral_csv(i):
    return batch_folder(i) / f"Lidl_multispectral_test{i}.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TVOC_SATURATION   = 59_000        # ppb — SGP30 saturates at ~60k
LIGHTING_FACTOR   = 1.6           # clear channel spike threshold (mirrors imu.html)
NIR_SMOOTH_WINDOW = 7             # rolling mean window for zone computation
LABEL_MAP         = {"Fresh": 0, "Aging": 1, "Degraded": 2}

ECO2_COLS    = ["master_eco2",    "node1_eco2",    "node2_eco2"]
MASTER_COLS  = ["master_temp",    "master_hum",    "master_tvoc", "master_mq3_ppm"]
NODE_TVOC_COLS  = ["node1_tvoc",  "node2_tvoc"]
# Node1 drops out intermittently across batches — only require node2 MQ3.
# Node1 nulls are imputed by per-batch median in Step 7.
DROP_IF_NULL = ["node2_mq3_ppm"]

# All node feature columns (node1 may be imputed)
NODE1_COLS   = ["node1_temp", "node1_hum", "node1_tvoc", "node1_mq3_ppm"]
NODE2_COLS   = ["node2_temp", "node2_hum", "node2_tvoc", "node2_mq3_ppm"]
ALL_NODE_COLS = NODE1_COLS + NODE2_COLS

RAW_FEATURE_COLS = [
    "node1_temp", "node1_hum", "node1_tvoc", "node1_mq3_ppm",
    "node2_temp", "node2_hum", "node2_tvoc", "node2_mq3_ppm",
]
DELTA_COLS   = ["delta_node1_tvoc", "delta_node2_tvoc"]
FEATURE_COLS = RAW_FEATURE_COLS + DELTA_COLS   # 10 features

# ---------------------------------------------------------------------------
# NIR labeling helpers  (mirrors imu.html: computeNirZones / zoneFor)
# ---------------------------------------------------------------------------

def lighting_flags(clear_series):
    """Return boolean mask — True where clear channel is a lighting artifact."""
    median = clear_series.median()
    if median == 0:
        return pd.Series(False, index=clear_series.index)
    return clear_series > (median * LIGHTING_FACTOR)


def compute_nir_zones(nir_clean):
    """
    Divide the NIR travel range into tertiles.
    Direction-aware: NIR can rise (more light reflected as tissue collapses)
    or fall (chlorophyll absorbs less as it degrades) depending on geometry.
    Returns dict with keys dir (+1/-1), b1, b2 (zone boundaries).
    Returns None if fewer than 4 clean readings.
    """
    nir_clean = nir_clean.dropna()
    if len(nir_clean) < 4:
        return None
    first3 = nir_clean.iloc[:3].mean()
    last3  = nir_clean.iloc[-3:].mean()
    direction = 1 if last3 > first3 else -1
    lo, hi = nir_clean.min(), nir_clean.max()
    span = hi - lo
    if span == 0:
        return None
    b1 = lo + span / 3
    b2 = lo + 2 * span / 3
    return {"dir": direction, "b1": b1, "b2": b2}


def zone_for(nir_val, zones):
    """Assign Fresh/Aging/Degraded for a single NIR value given computed zones."""
    if zones is None or pd.isna(nir_val):
        return None
    if zones["dir"] > 0:        # NIR rises → higher = more degraded
        if nir_val < zones["b1"]:  return "Fresh"
        if nir_val < zones["b2"]:  return "Aging"
        return "Degraded"
    else:                        # NIR falls → lower = more degraded
        if nir_val > zones["b1"]:  return "Fresh"
        if nir_val > zones["b2"]:  return "Aging"
        return "Degraded"


def label_spectral_df(spec_df, fixed_zones=None):
    """
    Given a single-batch spectral DataFrame, return it with nir_label column.
    If fixed_zones is provided, use those instead of computing per-batch zones.
    Lighting-flagged rows get label=None (excluded from zone computation,
    still forward-filled with the previous clean label).
    """
    spec_df = spec_df.copy().sort_values("timestamp").reset_index(drop=True)
    lit_flags = lighting_flags(spec_df["clear"])

    nir_clean = spec_df["nir"].where(~lit_flags)
    smoothed  = nir_clean.rolling(NIR_SMOOTH_WINDOW, min_periods=1, center=True).mean()

    if fixed_zones is not None:
        zones = fixed_zones
    else:
        zones = compute_nir_zones(smoothed)

    spec_df["nir_label"] = smoothed.apply(lambda v: zone_for(v, zones))
    spec_df.loc[lit_flags, "nir_label"] = None
    spec_df["nir_label"] = spec_df["nir_label"].ffill()

    return spec_df, zones, int(lit_flags.sum())


def compute_global_zones(train_batch_ids):
    """
    Pool smoothed NIR from all training batches and compute one set of zone
    boundaries. These are then frozen and applied to all batches (train + test).
    """
    all_smoothed = []
    for i in train_batch_ids:
        m = pd.read_csv(spectral_csv(i), parse_dates=["timestamp"])
        m = m.sort_values("timestamp").reset_index(drop=True)
        lit = lighting_flags(m["clear"])
        clear_safe = m["clear"].replace(0, np.nan)
        nir_ratio  = (m["nir"] / clear_safe).where(~lit)
        smoothed   = nir_ratio.rolling(NIR_SMOOTH_WINDOW, min_periods=1, center=True).mean()
        all_smoothed.append(smoothed.dropna())
    pooled = pd.concat(all_smoothed, ignore_index=True)
    return compute_nir_zones(pooled)


def assign_nir_to_sensor(sensor_df, spec_df):
    """
    Forward-fill the most recent NIR label onto each 15-min sensor row.
    Sensor rows that fall BEFORE the first spectral reading get label=None
    and are dropped later.
    """
    sensor_df = sensor_df.sort_values("timestamp").reset_index(drop=True)
    spec_sorted = spec_df[["timestamp", "nir_label"]].sort_values("timestamp")

    labels = []
    for ts in sensor_df["timestamp"]:
        past = spec_sorted[spec_sorted["timestamp"] <= ts]
        if past.empty:
            labels.append(None)
        else:
            labels.append(past.iloc[-1]["nir_label"])
    sensor_df["nir_label"] = labels
    return sensor_df


# ---------------------------------------------------------------------------
# 1. Load, label, and stack all batches
# ---------------------------------------------------------------------------
print("=" * 60)
print("MOULD PREDICTION DATASET PREPARATION — NIR LABELS")
print("=" * 60)

for i in BATCH_IDS:
    if not sensor_csv(i).exists():
        sys.exit(f"ERROR: Missing sensor CSV for batch {i}: {sensor_csv(i)}")
    if not spectral_csv(i).exists():
        sys.exit(f"ERROR: Missing spectral CSV for batch {i}: {spectral_csv(i)}")

all_frames = []
nir_zone_summary = {}

global_zones = None
if USE_GLOBAL_BOUNDARIES:
    global_zones = compute_global_zones(TRAIN_BATCHES)
    print(f"\nGlobal NIR zones (from training batches {TRAIN_BATCHES}): {global_zones}")
    print("  These boundaries are frozen and applied to ALL batches (train + test).")

for i in BATCH_IDS:
    batch_name = f"batch{i}"
    s = pd.read_csv(sensor_csv(i),   parse_dates=["timestamp"])
    m = pd.read_csv(spectral_csv(i), parse_dates=["timestamp"])

    m_labeled, zones, n_lit = label_spectral_df(m, fixed_zones=global_zones)
    nir_zone_summary[batch_name] = {
        "n_spectral": len(m),
        "n_lighting_flagged": n_lit,
        "zones": zones,
    }

    s = assign_nir_to_sensor(s, m_labeled)
    s["batch"] = batch_name
    all_frames.append(s)

    label_counts = s["nir_label"].value_counts().to_dict()
    print(f"\nBatch {i}: {len(s)} sensor rows, {len(m)} spectral rows, {n_lit} lighting events flagged")
    print(f"  NIR zones: {zones}")
    print(f"  Label distribution: {label_counts}")

df = pd.concat(all_frames, ignore_index=True)
print(f"\nTotal: {len(df)} rows across {len(BATCH_IDS)} batches")

# ---------------------------------------------------------------------------
# 2. Drop rows without a NIR label (sensor readings before first spectral poll)
# ---------------------------------------------------------------------------
before = len(df)
df.dropna(subset=["nir_label"], inplace=True)
print(f"\nDropped {before - len(df)} rows with no NIR label (before first spectral reading)")

# ---------------------------------------------------------------------------
# 3. Encode 3-class label
# ---------------------------------------------------------------------------
df["label"] = df["nir_label"].map(LABEL_MAP)
print(f"\nLabel distribution (all batches):")
print(df["label"].value_counts().sort_index().rename({0: "Fresh", 1: "Aging", 2: "Degraded"}).to_string())

# ---------------------------------------------------------------------------
# 4. Drop eCO2 columns
# ---------------------------------------------------------------------------
existing_eco2 = [c for c in ECO2_COLS if c in df.columns]
if existing_eco2:
    df.drop(columns=existing_eco2, inplace=True)
    print(f"\nDropped eCO2 columns: {existing_eco2}")

# ---------------------------------------------------------------------------
# 5. Drop master node features
# ---------------------------------------------------------------------------
existing_master = [c for c in MASTER_COLS if c in df.columns]
if existing_master:
    df.drop(columns=existing_master, inplace=True)
    print(f"Dropped master node features: {existing_master}")

# ---------------------------------------------------------------------------
# 6. Mark saturated TVOC as NaN
# ---------------------------------------------------------------------------
sat_counts = {}
for col in NODE_TVOC_COLS:
    if col not in df.columns:
        continue
    mask = df[col] >= TVOC_SATURATION
    sat_counts[col] = int(mask.sum())
    df.loc[mask, col] = np.nan

print(f"\nTVOC saturation (>= {TVOC_SATURATION:,} ppb) set to NaN:")
for col, n in sat_counts.items():
    print(f"  {col}: {n} readings ({100*n/len(df):.1f}%)")

# ---------------------------------------------------------------------------
# 7. Drop rows where node2 MQ3 is null (primary reliable node).
#    Node1 nulls are handled by imputation below — do NOT drop on node1.
# ---------------------------------------------------------------------------
before = len(df)
df.dropna(subset=DROP_IF_NULL, inplace=True)
print(f"\nDropped {before - len(df)} rows with null node2_mq3_ppm  ({len(df)} remaining)")

# ---------------------------------------------------------------------------
# 8. Impute all remaining node feature NaN with per-batch median.
#    Node1 drops out intermittently; node2 can also have sparse gaps.
#    Fall back to global median when an entire batch column is null.
# ---------------------------------------------------------------------------
print("\nImputing node feature NaN (per-batch median, global fallback):")
impute_cols = [c for c in ALL_NODE_COLS if c in df.columns]
for col in impute_cols:
    n_nan = int(df[col].isna().sum())
    if n_nan == 0:
        continue
    batch_medians = df.groupby("batch")[col].transform("median")
    df[col] = df[col].fillna(batch_medians)
    n_remaining = int(df[col].isna().sum())
    if n_remaining > 0:
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)
        print(f"  {col}: {n_nan - n_remaining} batch-median + {n_remaining} global-median imputed")
    else:
        print(f"  {col}: {n_nan} imputed with per-batch median")

# ---------------------------------------------------------------------------
# 9. Compute delta TVOC (rate of change within each batch)
# ---------------------------------------------------------------------------
print("\nComputing delta TVOC:")
for raw_col, delta_col in [("node1_tvoc", "delta_node1_tvoc"),
                            ("node2_tvoc", "delta_node2_tvoc")]:
    if raw_col not in df.columns:
        continue
    df[delta_col] = df.groupby("batch")[raw_col].diff().fillna(0.0)
    print(f"  {delta_col}: min={df[delta_col].min():.0f}  max={df[delta_col].max():.0f}  mean={df[delta_col].mean():.0f}")

remaining_nan = df[FEATURE_COLS].isna().sum().sum()
if remaining_nan > 0:
    print(f"\nWARNING: {remaining_nan} NaN values remain in features after imputation")
else:
    print("\nNo NaN values remain in feature columns.")

# ---------------------------------------------------------------------------
# 10. Train / Test split
# ---------------------------------------------------------------------------
train_batches = [f"batch{i}" for i in TRAIN_BATCHES]
held_batches  = [f"batch{i}" for i in HELD_BATCHES]
test_batches  = [f"batch{i}" for i in TEST_BATCHES]

train_df = df[df["batch"].isin(train_batches)].copy()
held_df  = df[df["batch"].isin(held_batches)].copy()  if held_batches else pd.DataFrame(columns=df.columns)
test_df  = df[df["batch"].isin(test_batches)].copy()

print(f"\nData split:")
print(f"  Train (Batch {TRAIN_BATCHES}): {len(train_df)} samples")
if held_batches:
    print(f"  Held  (Batch {HELD_BATCHES}): {len(held_df)} samples  <-- TinyOL on-device fine-tuning")
else:
    print(f"  Held  : none")
print(f"  Test  (Batch {TEST_BATCHES}):  {len(test_df)} samples")

print("\nClass balance:")
label_names = {0: "Fresh", 1: "Aging", 2: "Degraded"}
for split_name, split in [("Train", train_df), ("Test", test_df)]:
    counts = split["label"].value_counts().sort_index()
    total  = len(split)
    parts  = "  |  ".join(f"{label_names[k]}: {counts.get(k,0)} ({100*counts.get(k,0)/total:.0f}%)"
                          for k in [0, 1, 2])
    print(f"  {split_name:<8}: {parts}")

# ---------------------------------------------------------------------------
# 11. Spearman correlation: features vs NIR label (train set only)
# Tells you whether the gas/env sensors actually track freshness before training.
# Rule of thumb: |r| > 0.3 is worth something, |r| < 0.1 is noise.
# ---------------------------------------------------------------------------
from scipy.stats import spearmanr

print("\nSpearman correlation — features vs NIR label (train set):")
print(f"  {'Feature':<26} {'r':>7} {'p-value':>10}  signal")
print(f"  {'-'*55}")
for col in FEATURE_COLS:
    r, p = spearmanr(train_df[col], train_df["label"])
    if abs(r) >= 0.3:
        signal = "STRONG"
    elif abs(r) >= 0.1:
        signal = "weak"
    else:
        signal = "noise"
    print(f"  {col:<26} {r:>7.3f} {p:>10.4f}  {signal}")

# ---------------------------------------------------------------------------
# 12. Normalise — fit on ALL batches (labels excluded)
#
# We use the combined range of train+test so test-set features are not clipped
# to zero when test conditions differ from training (e.g. cooler room temp,
# lower gas levels). Only raw feature values are used here — no labels leak.
# ---------------------------------------------------------------------------
X_train = train_df[FEATURE_COLS].values.astype(np.float32)
X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
y_train = train_df["label"].values.astype(np.uint8)
y_test  = test_df["label"].values.astype(np.uint8)

if len(held_df) > 0:
    X_held = held_df[FEATURE_COLS].values.astype(np.float32)
    y_held = held_df["label"].values.astype(np.uint8)
else:
    X_held = np.zeros((0, len(FEATURE_COLS)), dtype=np.float32)
    y_held = np.zeros(0, dtype=np.uint8)

X_all = np.concatenate([X_train, X_test] + ([X_held] if len(X_held) > 0 else []), axis=0)
feat_min   = X_all.min(axis=0)
feat_max   = X_all.max(axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0   # avoid division by zero

X_train_norm = np.clip((X_train - feat_min) / feat_range, 0.0, 1.0)
X_test_norm  = np.clip((X_test  - feat_min) / feat_range, 0.0, 1.0)
X_held_norm  = np.clip((X_held  - feat_min) / feat_range, 0.0, 1.0) if len(X_held) > 0 else X_held

print(f"\nNormalisation (min-max, fitted on ALL batches — no label leak):")
print(f"  {'Feature':<22} {'Min':>8} {'Max':>8} {'Range':>8}")
print(f"  {'-'*50}")
for i, col in enumerate(FEATURE_COLS):
    print(f"  {col:<22} {feat_min[i]:>8.3f} {feat_max[i]:>8.3f} {feat_range[i]:>8.3f}")

# ---------------------------------------------------------------------------
# 13. Export cleaned CSVs
# ---------------------------------------------------------------------------
def build_export_df(split_df, X_norm, y):
    meta_cols = [c for c in ["timestamp", "batch", "cycle_number"] if c in split_df.columns]
    out = split_df[meta_cols].copy()
    for idx, col in enumerate(FEATURE_COLS):
        out[col + "_norm"] = X_norm[:, idx]
    out["nir_label"]   = split_df["nir_label"].values
    out["label"]       = y
    return out

build_export_df(train_df, X_train_norm, y_train).to_csv(OUT_DIR / "train_nir.csv", index=False)
build_export_df(test_df,  X_test_norm,  y_test).to_csv(OUT_DIR  / "test_nir.csv",  index=False)
print(f"\nCSVs saved to {OUT_DIR}")

# ---------------------------------------------------------------------------
# 14. Export dataset stats JSON
# ---------------------------------------------------------------------------
stats = {
    "label_system":  "NIR-based 3-class (Fresh=0, Aging=1, Degraded=2)",
    "feature_names": FEATURE_COLS,
    "n_features":    len(FEATURE_COLS),
    "n_train":       int(len(y_train)),
    "n_test":        int(len(y_test)),
    "train_batches": train_batches,
    "test_batches":  test_batches,
    "normalisation": "min-max fitted on all batches (no label leak)",
    "boundary_mode": "global" if USE_GLOBAL_BOUNDARIES else "per-batch",
    "global_zones":  global_zones,
    "feature_min":   feat_min.tolist(),
    "feature_max":   feat_max.tolist(),
    "feature_range": feat_range.tolist(),
    "train_class_balance": {
        label_names[k]: int((y_train == k).sum()) for k in [0, 1, 2]
    },
    "test_class_balance": {
        label_names[k]: int((y_test == k).sum()) for k in [0, 1, 2]
    },
    "nir_zones": {
        batch: {
            "n_spectral":         v["n_spectral"],
            "n_lighting_flagged": v["n_lighting_flagged"],
            "direction":          v["zones"]["dir"] if v["zones"] else None,
            "b1":                 v["zones"]["b1"]  if v["zones"] else None,
            "b2":                 v["zones"]["b2"]  if v["zones"] else None,
        }
        for batch, v in nir_zone_summary.items()
    },
    "exclusions": {
        "eco2":               "Dropped - SGP30 eCO2 is TVOC-derived, unreliable in high-VOC environments",
        "master_features":    "Dropped - ambient air; Spearman |r| < 0.05 vs mould label",
        "tvoc_saturation_ppb": TVOC_SATURATION,
        "lighting_events":    "Clear channel > 1.6x median flagged and excluded from NIR zone computation",
    },
    "engineered_features": {
        "delta_node1_tvoc": "TVOC rate of change per cycle within batch; first row = 0",
        "delta_node2_tvoc": "TVOC rate of change per cycle within batch; first row = 0",
    },
}

stats_path = OUT_DIR / "dataset_stats_nir.json"
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
print(f"Stats saved to {stats_path}")

# ---------------------------------------------------------------------------
# 15. Export C header for ESP32  (3-class labels — uint8 0/1/2)
# ---------------------------------------------------------------------------
N_FEAT  = len(FEATURE_COLS)
N_TRAIN = len(y_train)
N_TEST  = len(y_test)
X_test_export = X_test_norm
y_test_export = y_test

def array_to_c_2d(arr):
    rows = []
    for row in arr:
        vals = ", ".join(f"{v:.6f}f" for v in row)
        rows.append(f"  {{{vals}}}")
    return "{\n" + ",\n".join(rows) + "\n}"

header_lines = [
    "/*",
    " * mould_prediction_dataset_nir.h",
    " * Auto-generated by prepare_dataset_NIR.py",
    " *",
    " * Freshness Prediction Dataset — NIR 3-Class Labels",
    " * Features (10): node1_temp, node1_hum, node1_tvoc, node1_mq3_ppm,",
    " *                node2_temp, node2_hum, node2_tvoc, node2_mq3_ppm,",
    " *                delta_node1_tvoc, delta_node2_tvoc",
    " *",
    " * Labels: 0 = Fresh  |  1 = Aging  |  2 = Degraded",
    " * Labels derived from AS7341 NIR channel tertile zones (per-batch).",
    " *",
    " * Normalisation: min-max, fitted on all batches (no label leak)",
    " * Apply: x_norm = clip((x_raw - FEAT_MIN[i]) / FEAT_RANGE[i], 0, 1)",
    " *",
    f" * Train samples: {N_TRAIN}  (Batches {TRAIN_BATCHES})",
    f" * Test  samples: {N_TEST}   (Batches {TEST_BATCHES})",
    " */",
    "",
    "#pragma once",
    "#include <stdint.h>",
    "",
    f"#define N_FEATURES    {N_FEAT}",
    f"#define N_CLASSES     3",
    f"#define N_TRAIN_NIR   {N_TRAIN}",
    f"#define N_TEST_NIR    {N_TEST}",
    "",
    "/* Normalisation parameters */",
    "static const float NIR_FEAT_MIN[N_FEATURES] = {",
    "  " + ", ".join(f"{v:.6f}f" for v in feat_min),
    "};",
    "static const float NIR_FEAT_MAX[N_FEATURES] = {",
    "  " + ", ".join(f"{v:.6f}f" for v in feat_max),
    "};",
    "static const float NIR_FEAT_RANGE[N_FEATURES] = {",
    "  " + ", ".join(f"{v:.6f}f" for v in feat_range),
    "};",
    "",
    "static const char* NIR_FEAT_NAMES[N_FEATURES] = {",
    "  " + ", ".join(f'"{n}"' for n in FEATURE_COLS),
    "};",
    "",
    "static const char* NIR_CLASS_NAMES[N_CLASSES] = {\"Fresh\", \"Aging\", \"Degraded\"};",
    "",
    "/* Training data */",
    f"static const float nir_train_X[N_TRAIN_NIR][N_FEATURES] = {array_to_c_2d(X_train_norm)};",
    "",
    f"static const uint8_t nir_train_y[N_TRAIN_NIR] = {{{', '.join(str(v) for v in y_train)}}};",
    "",
    "/* Test data */",
    f"static const float nir_test_X[N_TEST_NIR][N_FEATURES] = {array_to_c_2d(X_test_export)};",
    "",
    f"static const uint8_t nir_test_y[N_TEST_NIR] = {{{', '.join(str(v) for v in y_test_export)}}};",
]

header_path = ESP32_DIR / "mould_prediction_dataset_nir.h"
with open(header_path, "w") as f:
    f.write("\n".join(header_lines) + "\n")

print(f"C header saved to {header_path}  ({os.path.getsize(header_path)/1024:.1f} KB)")

# ---------------------------------------------------------------------------
# 16. Export held_out_dataset.h (TinyOL on-device fine-tuning data)
#
# Used by tinyol_benchmark.cpp on the ESP32.  The held-out batch is never
# seen during backbone training (train_tinyol_backbone.py trains on Batch 1
# only) so it acts as the on-device adaptation set that TinyOL fine-tunes on.
#
# Batch 5 is the held set. To change: update HELD_BATCHES at the top and re-run.
# ---------------------------------------------------------------------------
if len(held_df) > 0:
    N_HELD = len(y_held)
    n_per_class_held = [int((y_held == c).sum()) for c in range(3)]

    held_header_lines = [
        "/*",
        " * held_out_dataset.h",
        " * Auto-generated by prepare_dataset_NIR.py",
        " *",
        " * TinyOL on-device fine-tuning dataset.",
        f" * Batches: {HELD_BATCHES}",
        f" * Samples: {N_HELD}  (Fresh={n_per_class_held[0]}  Aging={n_per_class_held[1]}  Degraded={n_per_class_held[2]})",
        " *",
        " * Never seen during backbone training — used only for on-device SGD adaptation.",
        " * Normalisation: same min-max parameters as mould_prediction_dataset_nir.h.",
        " */",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define N_HELD          {N_HELD}",
        f"#define N_HELD_FRESH    {n_per_class_held[0]}",
        f"#define N_HELD_AGING    {n_per_class_held[1]}",
        f"#define N_HELD_DEGRADED {n_per_class_held[2]}",
        "",
        f"static const float held_X[N_HELD][N_FEATURES] = {array_to_c_2d(X_held_norm)};",
        "",
        f"static const uint8_t held_y[N_HELD] = {{{', '.join(str(v) for v in y_held)}}};",
    ]

    held_path = ESP32_DIR / "held_out_dataset.h"
    with open(held_path, "w") as f:
        f.write("\n".join(held_header_lines) + "\n")
    print(f"Held-out header saved to {held_path}  ({os.path.getsize(held_path)/1024:.1f} KB)")
else:
    print("Held-out header skipped — HELD_BATCHES is empty.  Add batch number when Batch 4 is collected.")

# ---------------------------------------------------------------------------
# 17. Export combined_training_dataset_nir.h (AIfES full on-device training)
#
# Contains TRAIN + HELD batches combined — everything except the test set.
# Used by aifes_training_benchmark.cpp (Step 3: zero-cloud on-device training).
# The benchmark converts class indices to one-hot internally; this header
# stores raw class indices (uint8) to keep the file smaller.
# ---------------------------------------------------------------------------
combined_parts_X = [X_train_norm]
combined_parts_y = [y_train]
if len(X_held_norm) > 0:
    combined_parts_X.append(X_held_norm)
    combined_parts_y.append(y_held)

X_combined = np.concatenate(combined_parts_X, axis=0)
y_combined  = np.concatenate(combined_parts_y, axis=0)
N_COMBINED  = len(y_combined)
n_combined_per_class = [int((y_combined == c).sum()) for c in range(3)]

combined_header_lines = [
    "/*",
    " * combined_training_dataset_nir.h",
    " * Auto-generated by prepare_dataset_NIR.py",
    " *",
    " * AIfES full on-device training dataset (Step 3 — zero cloud).",
    f" * Batches: {TRAIN_BATCHES + HELD_BATCHES}  (all non-test batches)",
    f" * Samples: {N_COMBINED}  (Fresh={n_combined_per_class[0]}  Aging={n_combined_per_class[1]}  Degraded={n_combined_per_class[2]})",
    " *",
    " * Labels stored as class indices (0/1/2); benchmark converts to one-hot.",
    " * Normalisation: same min-max parameters as mould_prediction_dataset_nir.h.",
    " */",
    "",
    "#pragma once",
    "#include <stdint.h>",
    "",
    f"#define N_COMBINED           {N_COMBINED}",
    f"#define N_COMBINED_FRESH     {n_combined_per_class[0]}",
    f"#define N_COMBINED_AGING     {n_combined_per_class[1]}",
    f"#define N_COMBINED_DEGRADED  {n_combined_per_class[2]}",
    "",
    f"static const float combined_X[N_COMBINED][N_FEATURES] = {array_to_c_2d(X_combined)};",
    "",
    f"static const uint8_t combined_y[N_COMBINED] = {{{', '.join(str(v) for v in y_combined)}}};",
]

combined_path = ESP32_DIR / "combined_training_dataset_nir.h"
with open(combined_path, "w") as f:
    f.write("\n".join(combined_header_lines) + "\n")
print(f"Combined training header saved to {combined_path}  ({os.path.getsize(combined_path)/1024:.1f} KB)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Label system   : NIR 3-class (Fresh / Aging / Degraded)")
print(f"  Features       : {N_FEAT}")
print(f"  Train samples  : {N_TRAIN}  ({', '.join(f'{label_names[k]}={int((y_train==k).sum())}' for k in [0,1,2])})")
if len(held_df) > 0:
    N_HELD = len(y_held)
    print(f"  Held  samples  : {N_HELD}  ({', '.join(f'{label_names[k]}={int((y_held==k).sum())}' for k in [0,1,2])})")
else:
    print(f"  Held  samples  : none")
print(f"  Test  samples  : {N_TEST}   ({', '.join(f'{label_names[k]}={int((y_test_export==k).sum())}' for k in [0,1,2])})")
print(f"\n  Outputs:")
print(f"    {OUT_DIR}/train_nir.csv")
print(f"    {OUT_DIR}/test_nir.csv")
print(f"    {OUT_DIR}/dataset_stats_nir.json")
print(f"    {header_path}")
if len(held_df) > 0:
    print(f"    {ESP32_DIR}/held_out_dataset.h  <-- TinyOL fine-tuning data")
print(f"\n  Next step: train_model_NIR.py")
print(f"    Architecture: Input(10) -> Dense(16, ReLU) -> Dense(3, Softmax)")
print(f"    Loss: sparse_categorical_crossentropy")
print("=" * 60)
