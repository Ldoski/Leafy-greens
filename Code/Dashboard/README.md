# Sensor Trial Dashboard

A standalone HTML dashboard for exploring and visualising data from the SmartShelfLife spinach freshness monitoring system. Open `imu.html` in any browser no server, no install, no internet required after the page loads (Chart.js is pulled from a CDN on first open).

---

## What It Shows

The dashboard has 8 tabs, each covering a different view of the trial data.

| Tab | What it does |
|---|---|
| **Overview** | Combined timeline of all sensor families, normalised to 0–1 so shape can be compared on one axis. Also shows KPI tiles: trial duration, reading count, median interval. |
| **Temp & Humidity** | Temperature and humidity from all 3 nodes (Node 1 inside, Node 2 inside, Master/Reference outside) on overlaid line charts. |
| **VOC & MQ3 & CO2** | Gas sensor readings: TVOC (ppb), MQ3 alcohol (ppm), eCO2 (ppm). All three nodes overlaid. Trend stats (start→end change, rate per hour) shown below each chart. |
| **Node Comparison** | Delta charts: each inside node minus the outside reference. Strips ambient drift so only the chamber effect is visible. A delta near zero means the node is not separating from ambient. |
| **Data Quality** | Sampling interval consistency chart, per-sensor noise ratios, and a reliability table (coverage %, noise ratio, spike count, keep/caution/drop verdict). Useful for identifying SGP30 dropout rows before training. |
| **Multispectral** | AS7341 spectral data: all 8 visible channels (415–680 nm) and a focus view on the red/green region (555–680 nm). Lighting-spike events flagged automatically from the clear channel. |
| **NIR Labeling** | Core labeling tab. Shows raw NIR with direction-aware freshness label zones (Fresh / Aging / Degraded) as colour bands, plus a 7-point rolling mean. Also compares raw NIR % change vs NIR/Red ratio % change to justify the labeling choice. |
| **Images** | Load timelapse photos. Each image is aligned to the nearest NIR reading by filename timestamp. A scrubber lets you step through the trial with the NIR label shown alongside each photo. Contact sheet with colour-coded borders (green=Fresh, amber=Aging, red=Degraded). |

---

## How to Use

### Loading your data

The dashboard opens with built-in sample data so you can see all tabs working immediately. To load real data, use the buttons in the top bar.

**Load CSV** drag or select a CSV export from the trial. The dashboard auto-detects whether it is sensor data or multispectral data by looking at the column headers:
- If headers contain `415` or `f1`–`f8` with `nm` treated as AS7341 spectral data, opens Multispectral tab.
- Otherwise treated as environmental sensor data (temp, hum, TVOC, eCO2, MQ3), opens Overview tab.

Column name matching is flexible. Node columns are matched on patterns like `node1`, `n1_`, `node_1`; sensor columns on `tvoc`/`voc`, `mq3`/`alcohol`, `temp`/`temperature`, `humid`/`hum`, `eco2`/`co2`. The `master` or `ref` node maps to the outside reference.

**Load Images** select one or more timelapse photos. Filenames must contain a timestamp in one of these formats for alignment to work:
- `capture_20260604_135110.jpg` (YYYYMMDD_HHMMSS)
- `capture_2026-06-04_13-51-10.jpg` (YYYY-MM-DD HH-MM-SS)

Images are never uploaded anywhere they stay in your browser.

**Reset to sample** returns to the built-in synthetic dataset.

### Exporting charts

Hover over any chart card and a small **↓ PNG** button appears in the top-right corner. Click it to save that chart as a PNG at 2× resolution.

---

## Node Layout

| Node | Role | Sensors |
|---|---|---|
| Node 1 (inside) | Inside the spinach chamber | DHT22 (temp/hum), SGP30 (TVOC/eCO2), MQ3 (alcohol) |
| Node 2 (inside) | Inside the spinach chamber | DHT22 (temp/hum), SGP30 (TVOC/eCO2), MQ3 (alcohol) |
| Master / Reference (outside) | Outside the chamber, ambient | DHT22 (temp/hum), SGP30 (TVOC/eCO2), MQ3 (alcohol) |
| AS7341 board | Pointed at spinach | 11-channel multispectral (415–850 nm + NIR + Clear) |

---

## NIR Labeling Logic

The **NIR Labeling** tab uses the AS7341 NIR channel (~855 nm) alone as the freshness label source. No red channel, no ratio.

Labels are assigned by direction-aware tertiles:
- The dashboard measures the total travel of NIR from the start-of-trial baseline to the end-state value and determines whether NIR rises or falls with degradation (for spinach it rises).
- The travel range is split into thirds: Fresh (closest to baseline), Aging (middle third), Degraded (furthest from baseline).
- A 7-point rolling mean is overlaid to smooth noise; the raw NIR is also shown.

The comparison chart below (NIR % change vs NIR/Red ratio % change) shows that both signals track the same degradation direction and timing dgssgedfsdg this is the justification for using raw NIR alone rather than the ratio.
