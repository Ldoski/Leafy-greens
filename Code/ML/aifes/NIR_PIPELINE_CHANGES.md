# NIR Pipeline — Design Decisions

Key decisions made when building the NIR 3-class labelling pipeline.

---

## Labelling

Labels are derived from the AS7341 NIR 855 nm channel, not from manual timestamps or visual inspection.

Each batch is split into three equal-frequency bins based on that batch's own NIR range:
- Bottom third → Fresh (0)
- Middle third → Aging (1)
- Top third → Degraded (2)

**Why per-batch tertiles?** Spinach lots, initial freshness, and ambient conditions vary batch to batch, shifting the absolute NIR baseline. A single global threshold would mislabel large parts of the dataset.

**Lighting artefact filtering:** Spectral readings where the AS7341 `clear` channel exceeds 1.6× its per-batch median are flagged as contaminated by external light and excluded from zone boundary computation. They still receive a forward-filled label.

---

## Joining Sensor and Spectral Data

The gas/temperature sensors (MQ3, SGP30, DHT22) log every 15 minutes. The AS7341 logs every 53 minutes. They are joined by forward-filling the most recent spectral reading onto each sensor row.

The 53-minute spectral interval was chosen to be coprime with the 60-minute camera cycle so the two never align and the camera flash never contaminates a spectral reading.

---

## Null Handling

SGP30 TVOC returned zeros on multiple nodes due to a timing conflict on the shared I2C bus. Affected rows were kept in training rather than dropped, because removing them would break time-series continuity. They add noise to the decision boundary.

Node 2 MQ3 null → row dropped. Node 1 MQ3 null → imputed with per-batch median (Node 1 had intermittent disconnects up to 74% null in some batches; dropping on Node 1 would remove all Degraded rows from the test set).

---

## Train / Test Split

| Set | Batches | Samples |
|---|---|---|
| Train | 1, 3 | 670 |
| Test | 4 | 317 |
| Held-out (TinyOL) | 5 | 378 |

Leave-one-batch-out cross-validation across Batches 1, 3, and 4 selected this split.

---

## Adding a New Batch

1. Drop `Lidl_sensors_testN.csv` and `Lidl_multispectral_testN.csv` into `sensor_data/project2_data/Lidl_batches/Lidl_room_temp_batchN/`
2. In `prepare_dataset_NIR.py`, update the batch config:
   ```python
   BATCH_IDS     = [1, 3, 4, 5, 6]   # add new batch number
   TRAIN_BATCHES = [1, 3]
   TEST_BATCHES  = [4]
   HELD_BATCHES  = [5]
   ```
3. Re-run `prepare_dataset_NIR.py` then `train_model_NIR.py`.
