# Raspberry Pi Data Collection

`uart_data_collector.py` is the data acquisition service that runs on the Pi throughout each batch trial. It collects sensor readings from the ESP32 network, polls the AS7341 spectral sensor, captures timelapse images, and writes everything to CSV.

Run it with `TEST_ID` set before starting a new batch — this controls the output folder name.

```bash
TEST_ID=Lidl_room_temp_batch6 python3 uart_data_collector.py
```

Or as a systemd service (recommended for unattended trials):

```bash
sudo systemd-run --unit=data_collector \
  -E TEST_ID=Lidl_room_temp_batch6 \
  python3 /path/to/uart_data_collector.py
```

---

## What It Does

Three concurrent threads plus the main loop:

| Thread / Loop | What it does | Interval |
|---|---|---|
| Main loop | Reads JSON from ESP32 Master via UART (GPIO pins), writes to `Lidl_sensors_test.csv` | Every ~15 min (set on ESP32 side) |
| AS7341 thread | Opens `/dev/ttyUSB0`, sends `READ` command to the AS7341 ESP32-C3, saves response to `Lidl_multispectral_test.csv` | Every 53 min |
| Camera thread | Fires LED, triggers autofocus, captures JPEG via Picamera2, saves to `images/` | Every 60 min |

The camera adapts its autofocus delay based on measured exposure time and gain — 3 s in good light, 6 s in low light.

---

## Wiring

### ESP32 Master → Pi (UART)

| ESP32 pin | Pi pin | Signal |
|---|---|---|
| GPIO17 (TX2) | Pin 10 (GPIO15 / RXD) | Data from ESP32 to Pi |
| GPIO16 (RX2) | Pin 8 (GPIO14 / TXD) | Data from Pi to ESP32 |
| GND | GND | Common ground |

### AS7341 ESP32-C3 → Pi (USB)

USB cable only. Appears as `/dev/ttyUSB0`. The Pi opens and closes the port on each poll cycle — the USB connection causes the ESP32-C3 to reset, which is intentional (it sends a `ready` JSON before accepting commands).

### LED

GPIO pin 17 (BCM). Fires before each camera capture, off after. Uses `lgpio` on Pi 5 (inverted logic — 0 = ON, 1 = OFF) or `RPi.GPIO` on older Pi models. Falls back gracefully if GPIO init fails.

---

## Output Files

All output goes under `~/project2_data/`:

```
~/project2_data/
  Lidl_batches/
    <TEST_ID>/
      Lidl_sensors_test.csv       ← ESP32 master/node1/node2 readings
      Lidl_multispectral_test.csv ← AS7341 spectral readings
  images/
    <TEST_ID>/
      capture_YYYYMMDD_HHMMSS.jpg
  uart_data_collector.log
```

The image filenames use `capture_YYYYMMDD_HHMMSS.jpg` — this format is required by the Dashboard image loader to align photos to NIR readings.

---

## CSV Columns

**`Lidl_sensors_test.csv`**

`timestamp, cycle_number, master_temp, master_hum, master_tvoc, master_eco2, master_mq3_ppm, node1_temp, node1_hum, node1_tvoc, node1_eco2, node1_mq3_ppm, node2_temp, node2_hum, node2_tvoc, node2_eco2, node2_mq3_ppm`

Missing node data (if a node did not respond in that cycle) is written as `N/A`.

**`Lidl_multispectral_test.csv`**

`timestamp, f1_415nm, f2_445nm, f3_480nm, f4_515nm, f5_555nm, f6_590nm, f7_630nm, f8_680nm, clear, nir, gain, atime, astep`

---

## Dependencies

```bash
pip install pyserial picamera2
# lgpio (Pi 5) or RPi.GPIO (Pi 4 and older) — install whichever matches your hardware
pip install lgpio
```

Serial console must be disabled on the Pi (`raspi-config` → Interface Options → Serial Port → disable login shell, enable hardware port).
