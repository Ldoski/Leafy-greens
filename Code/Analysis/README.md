# Batch Analysis Notebook

`batch_analysis.ipynb` is the exploratory analysis notebook for the SmartShelfLife spinach freshness project. It covers data quality, NIR labeling, spectral behaviour, feature distributions, and feature-label correlation across all experimental batches. It does not train models — that is handled by `ML/aifes/train_model_NIR.py`.

---

## What It Analyses

| Section | What it does |
|---|---|
| **1. Load and validate** | Reads all four batch CSVs, runs NIR zone labeling (matches `prepare_dataset_NIR.py` exactly), and prints a per-batch overview table. |
| **2. Data quality** | Heatmaps showing missing-value % and TVOC saturation % per batch and per sensor column. eCO2 columns are greyed out (excluded). |
| **3. eCO2 exclusion** | Time-series of eCO2 from all three nodes across all batches. Shows why it is excluded: saturates at the SGP30 hardware ceiling within hours. |
| **4. TVOC saturation** | TVOC time-series with the 59,000 ppb saturation ceiling marked. Saturated readings are dotted; valid readings are solid. |
| **5-8. Sensor time series** | One subplot per batch for temperature, humidity, TVOC, and ethanol (MQ3). Background shading shows the current NIR label zone (green=Fresh, amber=Aging, red=Degraded). Master node shown in grey — it measures ambient air and is excluded from ML. |
| **9. Distributions by class** | Boxplots of all 10 ML features grouped by NIR class. Separability between boxes is a quick read on how useful each feature is. |
| **10. Pearson correlation** | Correlation matrix (all features + label) plus a bar chart of each feature's linear correlation with the NIR label. Computed on the training set (Batches 1 and 3). |
| **11. Train / Held / Test split** | Sample counts per class for each split. Stacked bar chart shows how batches contribute within each split. |
| **12. Density distributions** | Histogram density plots for all 10 features across all three classes — complements the boxplots by showing distribution shape. |
| **12b. Lighting artifacts** | AS7341 clear channel per batch with artifact threshold (1.6x median). Flagged readings shown in red. |
| **13. NIR zones** | Raw and smoothed NIR per batch, with b1/b2 tertile boundaries marked. Colour bands show which readings fell into which class. |
| **14. Spectral channels** | All 8 visible AS7341 bands (415–680 nm), NIR, and Clear — all batches overlaid. Clean readings only. |
| **15. Spearman correlation** | Rank-based correlation of each feature vs the NIR label. More appropriate than Pearson for the ordinal 0/1/2 label. Computed on Batches 1 and 3 only. |

---

## Batch Split

| Batch | Role | Notes |
|---|---|---|
| Batch 1 | Train | ~102 h |
| Batch 2 | Excluded | Node 1 communication failure; folder exists on disk but is empty |
| Batch 3 | Train | ~70 h |
| Batch 4 | Test | Held out during all training decisions |
| Batch 5 | Held (TinyOL) | Used for on-device fine-tuning only; never seen by backbone |
| Batch 6 | In progress | ASTEP=29,999 (higher integration time); not included here |

---

## NIR Labeling Logic

Labels are assigned inside the notebook using the same logic as `prepare_dataset_NIR.py` so outputs match the training pipeline exactly.

1. Lighting artifacts are flagged: any AS7341 reading where the clear channel exceeds 1.6x the per-batch median is excluded.
2. NIR is smoothed with a 7-point rolling mean (center-aligned).
3. Direction is determined by comparing the first 3 and last 3 smoothed readings. For spinach, NIR rises with degradation.
4. The smoothed NIR range is split into tertiles to form three zones: Fresh (closest to baseline), Aging (middle), Degraded (furthest from baseline).
5. Each sensor row is assigned the label of the most recent spectral reading at or before its timestamp.

NIR is used as a label source only. It is not an ML input feature.

---

## Data Exclusions

These exclusions are applied in cell-13 before any analysis that uses `df_clean`:

| Exclusion | Reason |
|---|---|
| eCO2 (all nodes) | Derived from TVOC by SGP30 firmware; saturates within hours; no independent signal |
| TVOC >= 59,000 ppb | SGP30 hardware saturation ceiling; readings above this are clipped to NaN |
| Rows with null Node 2 MQ3 | Node 2 ethanol is a key freshness indicator; rows without it are unusable |
| SGP30 dropout rows | Sensor returned 0 ppb TVOC on multiple nodes due to I2C bus timing conflict. These rows are kept in `df_clean` (removing them would break time-series continuity) but the Spearman section uses `notna()` masks so they do not corrupt the correlation. |

---

## Output Figures

All figures are saved to `figures/` next to this notebook.

| File | Section |
|---|---|
| `01_data_quality.png` | Missing values and saturation heatmaps |
| `02_eco2_exclusion.png` | eCO2 time-series |
| `03_tvoc_saturation.png` | TVOC with saturation threshold |
| `04_temp.png` | Temperature time-series |
| `05_humidity.png` | Humidity time-series |
| `06_tvoc.png` | TVOC time-series (saturated masked) |
| `07_ethanol.png` | MQ3 ethanol time-series |
| `08_distributions_by_class.png` | Boxplots by NIR class |
| `09_pearson_correlation.png` | Pearson correlation matrix + bar chart |
| `10_dataset_split.png` | Train / Held / Test stacked bar |
| `11_lighting_artifacts.png` | Clear channel artifact detection |
| `11b_feature_distributions_density.png` | Density histograms by class |
| `12_nir_zones.png` | NIR signal and zone boundaries |
| `13_spectral_channels.png` | All AS7341 visible bands |
| `14_nir_clear.png` | NIR and Clear channels overlaid |
| `15_spearman_correlation.png` | Spearman r bar chart |

---

## Running It

Open in Jupyter and run all cells top to bottom. Cell 1 sets all constants — if you change `TRAIN_BATCHES`, `HELD_BATCHES`, or `TEST_BATCHES` there, the rest of the notebook updates automatically.

The notebook expects the batch data to be under:
```
Leafy_Greens_Project/sensor_data/project2_data/Lidl_batches/Lidl_room_temp_batch{N}/
```

Each batch folder needs two CSVs:
- `Lidl_sensors_test{N}.csv` — environmental sensor readings with a `timestamp` column
- `Lidl_multispectral_test{N}.csv` — AS7341 readings with `timestamp`, `nir`, and `clear` columns
