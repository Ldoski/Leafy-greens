# Experiments

This folder contains two analysis notebooks and the ground test dataset.

| File | What it is |
|---|---|
| `experiment_analysis.ipynb` | Publication figures for the freshness detection system (NIR, spoilage markers, spectral channels) |
| `test_analysis.ipynb` | AS7341 sensor ground test analysis (5 environmental conditions) |
| `ground_test/` | Ground test CSV data 5 conditions, 11 readings each |
| `Ground_Test_Protocol_v3.pdf` | Protocol document for the ground tests |

---

## experiment_analysis.ipynb

Generates the publication-quality figures for the SmartShelfLife spinach freshness project. It focuses on visual outputs: NIR spoilage signals, spectral channel behaviour, gas and environmental sensors, and the sensor-vs-visual spoilage comparison that is the headline result of the paper. All figures export as PNG, PDF, and SVG.

This is separate from `Code/Analysis/batch_analysis.ipynb`, which covers pipeline decisions (data quality, feature correlation, exclusion justification). This notebook is for what goes in the thesis.

---

## What It Produces

| Section | Figures | What it shows |
|---|---|---|
| **S1** | `S1_nir_batch{N}.png/pdf/svg` | NIR 855 nm over time per batch, with Fresh/Aging/Degraded zone shading |
| **S1b** | `S1b_nir_grid_all_batches.*` | All 4 batches in a 2×2 grid for side-by-side comparison |
| **S1c** | `S1c_nir_spoilage_marker.*` | NIR vs visual spoilage threshold the key early-detection result. Shows how many hours before or after visual inspection the sensor crossed into Degraded |
| **S2** | `S2_nir_all_batches.*` | All batches overlaid on raw time axis |
| **S3** | `S3_nir_normalised.*` | All batches overlaid on normalised time axis (0–100% of run) |
| **S4** | `S4_channels_batch{N}.*` | All spectral channels (555–855 nm) over time per batch on one axis |
| **S5** | `S5_grid_batch{N}.*` | 2×3 subplot grid per batch, one channel per plot with CV annotation |
| **S6** | `S6_gas_batch{N}.*` | TVOC, eCO2, and MQ3 over time per batch |
| **S7** | `S7_temp_hum_batch{N}.*` | Temperature and humidity per batch verifies stable conditions |
| **S8** | `S8_nir_tvoc_batch{N}.*` | NIR and TVOC on dual y-axes checks whether gas and optical signals track together |
| **S9** | *(table only)* | Summary statistics per spectral channel per batch (mean, SD, CV%, range) |

All figures are saved to `plots/spinach/`.

---

## Key Result S1c

Section S1c is the headline figure. It overlays the visual spoilage timestamp (from timelapse review) on the NIR time-series for each batch and annotates how many hours the sensor led or lagged visual inspection. Run the notebook to see the computed values the lead/lag depends on the NIR threshold crossing point which varies with smoothing method.

Batch 5 is the only batch where the sensor detected spoilage before visual inspection. The Batch 5 visual spoilage timestamp is estimated (no timelapse images were captured for that batch; value is the average ratio from Batches 1, 3, 4).

---

## NIR Zone Boundaries

Zones are computed per batch as simple quantile tertiles (no rolling smooth, no direction detection simpler than `batch_analysis.ipynb` which mirrors the pipeline exactly).

| Batch | Fresh | Aging | Degraded |
|---|---|---|---|
| Batch 1 | ≤ 80 | 80–92 | > 92 |
| Batch 3 | ≤ 57 | 57–65 | > 65 |
| Batch 4 | ≤ 66 | 66–80 | > 80 |
| Batch 5 | ≤ 80 | 80–84 | > 84 |

---

## Style

All figures use `font.family = Courier New`, `font.size = 22`, bold weights, and `savefig.dpi = 200`. This matches the thesis template. Each batch has a fixed colour and marker:

| Batch | Colour | Marker |
|---|---|---|
| Batch 1 | Purple `#9671bd` | Circle |
| Batch 3 | Teal `#77b5b6` | Square |
| Batch 4 | Amber `#e8a838` | Diamond |
| Batch 5 | Red `#e05c5c` | Triangle |

---

## Data Sources

Reads directly from:
```
Leafy_Greens_Project/sensor_data/project2_data/Lidl_batches/Lidl_room_temp_batch{N}/
  Lidl_sensors_test{N}.csv
  Lidl_multispectral_test{N}.csv
```

Batches used: 1, 3, 4, 5. Batch 2 excluded (Node 1 hardware failure). Batch 6 not included (still running at time of writing, different ASTEP setting).

---

## test_analysis.ipynb

AS7341 sensor characterisation under 5 controlled conditions. Data lives in `ground_test/thesis_data/ground_test/`.

| Run | Condition | Folder |
|---|---|---|
| 1 | Natural light | `ground_test_natural_light/` |
| 2 | Dark | `ground_test_dark/` |
| 3 | Light + movement | `ground_test_light_movement/` |
| 4 | Temperature 40°C | `ground_test_temp_40degC/` |
| 5 | Humidity 88% | `ground_test_humidity_88%/` |

Each run has 11 spectral readings. The notebook plots NIR and the four high-signal channels (590, 630, 680, 855 nm) across all conditions and computes CV% and drift% per channel. Figures are saved to `plots/ground_test/`.
