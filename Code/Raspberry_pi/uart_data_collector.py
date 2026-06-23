#!/usr/bin/env python3
"""
UART Data Collection Service for Raspberry Pi
- Receives sensor data from ESP32 Master via UART (GPIO pins)
- Polls AS7341 NIR sensor via USB serial (/dev/ttyUSB0) every 59 minutes
- Saves ESP32 node data (master/node1/node2) to unified CSV
- Saves AS7341 NIR data to a separate CSV
- Captures camera images with LED every 60 minutes
- Runs as a background service

Wiring (ESP32 Master <-> Pi):
  ESP32 GPIO17 (TX2) -> Pi Pin 10 (GPIO15/RXD)
  ESP32 GPIO16 (RX2) -> Pi Pin 8  (GPIO14/TXD)
  ESP32 GND          -> Pi GND

AS7341 ESP32 connects via USB cable -> /dev/ttyUSB0
"""

import serial
import json
import csv
import os
from datetime import datetime
from pathlib import Path
import time
import threading
import logging
from picamera2 import Picamera2

try:
    import lgpio as GPIO
    GPIO_CHIP = 4   # Pi 5 uses chip 4
    USE_LGPIO = True
except ImportError:
    import RPi.GPIO as GPIO
    USE_LGPIO = False

# ==================== Configuration ====================

# ESP32 Master (UART via GPIO pins)
SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 115200

# AS7341 NIR sensor (USB cable)
AS7341_PORT  = "/dev/ttyUSB0"
AS7341_BAUD_RATE = 115200
AS7341_INTERVAL_SECONDS = 53 * 60   # 53 minutes

# Camera
CAMERA_INTERVAL_SECONDS = 60 * 60   # 60 minutes

# GPIO
LED_PIN     = 17
gpio_handle = None

# Test / Batch ID
TEST_ID = os.environ.get("TEST_ID", "Lidl_room_temp_batch")

# Directoriesd
BASE_DIR  = Path.home() / "project2_data"
DATA_DIR  = BASE_DIR / "Lidl_batches" / TEST_ID
IMAGE_DIR = BASE_DIR / "images"      / TEST_ID

# Log file
LOG_FILE = BASE_DIR / "uart_data_collector.log"

# CSV filenames
CSV_FILENAME       = "Lidl_sensors_test.csv"
AS7341_CSV_FILENAME = "Lidl_multispectral_test.csv"


# ==================== Logging ====================
def setup_logging():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


# ==================== Directories & GPIO ====================
def setup_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Test/Batch ID  : {TEST_ID}")
    logger.info(f"Data directory : {DATA_DIR}")
    logger.info(f"Image directory: {IMAGE_DIR}")


def setup_gpio():
    global gpio_handle
    try:
        if USE_LGPIO:
            gpio_handle = GPIO.gpiochip_open(GPIO_CHIP)
            GPIO.gpio_claim_output(gpio_handle, LED_PIN, 1)  # 1 = OFF (inverted)
            logger.info(f"GPIO {LED_PIN} configured (lgpio, inverted logic)")
        else:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(LED_PIN, GPIO.OUT)
            GPIO.output(LED_PIN, GPIO.LOW)
            logger.info(f"GPIO {LED_PIN} configured (RPi.GPIO)")
        led_off()
        logger.info("LED initialised to OFF")
    except Exception as e:
        logger.warning(f"GPIO setup failed: {e}. LED control disabled.")
        gpio_handle = None


def led_on():
    try:
        if USE_LGPIO and gpio_handle is not None:
            GPIO.gpio_write(gpio_handle, LED_PIN, 0)   # 0 = ON
        elif not USE_LGPIO:
            GPIO.output(LED_PIN, GPIO.HIGH)
    except Exception as e:
        logger.warning(f"LED on failed: {e}")


def led_off():
    try:
        if USE_LGPIO and gpio_handle is not None:
            GPIO.gpio_write(gpio_handle, LED_PIN, 1)   # 1 = OFF
        elif not USE_LGPIO:
            GPIO.output(LED_PIN, GPIO.LOW)
    except Exception as e:
        logger.warning(f"LED off failed: {e}")


# ==================== Camera ====================
class CameraController:
    def __init__(self):
        self.picam2       = None
        self.initialized  = False

    def initialize(self):
        try:
            if not self.initialized:
                self.picam2 = Picamera2(camera_num=0)
                config = self.picam2.create_still_configuration()
                self.picam2.configure(config)
                self.picam2.set_controls({"AfMode": 2})   # Continuous autofocus
                self.picam2.start()
                time.sleep(2)
                self.initialized = True
                logger.info("Camera initialised with continuous autofocus")
        except Exception as e:
            logger.error(f"Camera init failed: {e}")
            self.initialized = False

    def capture_with_led(self):
        try:
            if not self.initialized:
                self.initialize()
            if not self.initialized:
                logger.error("Camera not initialised, skipping capture")
                return False

            led_on()
            logger.info("LED ON - analysing lighting...")
            time.sleep(0.5)

            # Determine focus delay based on lighting
            try:
                meta          = self.picam2.capture_metadata()
                exposure_time = meta.get('ExposureTime', 0)
                analogue_gain = meta.get('AnalogueGain', 1.0)
                is_low_light  = (exposure_time > 20000) or (analogue_gain > 4.0)
                focus_delay   = 6 if is_low_light else 3
                logger.info(
                    f"{'Low light' if is_low_light else 'Good light'} "
                    f"(Exp: {exposure_time/1000:.1f}ms, Gain: {analogue_gain:.2f}x) "
                    f"- focus delay {focus_delay}s"
                )
            except Exception as e:
                focus_delay = 5
                logger.warning(f"Could not detect lighting: {e} - using {focus_delay}s delay")

            # Trigger autofocus
            try:
                self.picam2.set_controls({"AfTrigger": 0})
                logger.info("Autofocus triggered")
            except Exception as e:
                logger.warning(f"Autofocus trigger failed: {e}")

            time.sleep(focus_delay)

            # Log AF state
            try:
                meta = self.picam2.capture_metadata()
                if 'AfState' in meta:
                    states = {0: "Idle", 1: "Scanning", 2: "Focused", 3: "Failed"}
                    logger.info(f"AF State: {states.get(meta['AfState'], 'Unknown')}")
                if 'LensPosition' in meta:
                    logger.info(f"Lens Position: {meta['LensPosition']:.2f}")
            except Exception:
                pass

            # Capture
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = IMAGE_DIR / f"capture_{timestamp}.jpg"
            self.picam2.capture_file(str(filename))

            led_off()
            logger.info("LED OFF - capture complete")

            if filename.exists():
                logger.info(f"Captured: {filename.name} ({filename.stat().st_size / 1024:.2f} KB)")
                return True
            else:
                logger.error("Capture file not created")
                return False

        except Exception as e:
            try:
                led_off()
            except Exception:
                pass
            logger.error(f"Camera capture failed: {e}")
            try:
                self.cleanup()
                time.sleep(2)
                self.initialize()
            except Exception as reinit_err:
                logger.error(f"Camera reinit failed: {reinit_err}")
            return False

    def cleanup(self):
        try:
            if self.initialized and self.picam2:
                self.picam2.stop()
                self.initialized = False
                logger.info("Camera stopped")
        except Exception as e:
            logger.error(f"Camera cleanup error: {e}")


camera = CameraController()


# ==================== Data Saving ====================
def save_unified_data(data):
    """Save ESP32 master/node1/node2 data to unified CSV."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_file  = DATA_DIR / CSV_FILENAME
        exists    = csv_file.exists()

        cycle  = data.get('cycle', 'N/A')
        master = data.get('master', {})
        node1  = data.get('node1',  {})
        node2  = data.get('node2',  {})

        def val(node, key):
            return node.get(key, 'N/A') if node.get('received') else 'N/A'

        row = {
            'timestamp':    timestamp,
            'cycle_number': cycle,
            'master_temp':     val(master, 'temp'),
            'master_hum':      val(master, 'hum'),
            'master_tvoc':     val(master, 'tvoc'),
            'master_eco2':     val(master, 'eco2'),
            'master_mq3_ppm':  val(master, 'mq3_ppm'),
            'node1_temp':      val(node1,  'temp'),
            'node1_hum':       val(node1,  'hum'),
            'node1_tvoc':      val(node1,  'tvoc'),
            'node1_eco2':      val(node1,  'eco2'),
            'node1_mq3_ppm':   val(node1,  'mq3_ppm'),
            'node2_temp':      val(node2,  'temp'),
            'node2_hum':       val(node2,  'hum'),
            'node2_tvoc':      val(node2,  'tvoc'),
            'node2_eco2':      val(node2,  'eco2'),
            'node2_mq3_ppm':   val(node2,  'mq3_ppm'),
        }

        fieldnames = [
            'timestamp', 'cycle_number',
            'master_temp', 'master_hum', 'master_tvoc', 'master_eco2', 'master_mq3_ppm',
            'node1_temp',  'node1_hum',  'node1_tvoc',  'node1_eco2',  'node1_mq3_ppm',
            'node2_temp',  'node2_hum',  'node2_tvoc',  'node2_eco2',  'node2_mq3_ppm',
        ]

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
                logger.info(f"Created CSV: {CSV_FILENAME}")
            writer.writerow(row)

        logger.info(
            f"Cycle {cycle}: saved - Master={master.get('received')}, "
            f"Node1={node1.get('received')}, Node2={node2.get('received')}"
        )
        if master.get('received'):
            logger.info(
                f"  Master: T={master.get('temp')}°C, "
                f"TVOC={master.get('tvoc')}, MQ3={master.get('mq3_ppm')}"
            )

    except Exception as e:
        logger.error(f"Failed to save unified data: {e}")


def save_as7341_data(data):
    """Save AS7341 NIR reading to its own CSV."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_file  = DATA_DIR / AS7341_CSV_FILENAME
        exists    = csv_file.exists()

        row = {
            'timestamp': timestamp,

            'f1_415nm': data.get('f1_415nm', 'N/A'),
            'f2_445nm': data.get('f2_445nm', 'N/A'),
            'f3_480nm': data.get('f3_480nm', 'N/A'),
            'f4_515nm': data.get('f4_515nm', 'N/A'),
            'f5_555nm': data.get('f5_555nm', 'N/A'),
            'f6_590nm': data.get('f6_590nm', 'N/A'),
            'f7_630nm': data.get('f7_630nm', 'N/A'),
            'f8_680nm': data.get('f8_680nm', 'N/A'),

            'clear': data.get('clear', 'N/A'),
            'nir':   data.get('nir', 'N/A'),

            'gain':  data.get('gain', 'N/A'),
            'atime': data.get('atime', 'N/A'),
            'astep': data.get('astep', 'N/A'),
        }

        fieldnames = [
            'timestamp',

            'f1_415nm',
            'f2_445nm',
            'f3_480nm',
            'f4_515nm',
            'f5_555nm',
            'f6_590nm',
            'f7_630nm',
            'f8_680nm',

            'clear',
            'nir',

            'gain',
            'atime',
            'astep'
        ]

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
                logger.info(f"Created CSV: {AS7341_CSV_FILENAME}")
            writer.writerow(row)

        logger.info(
            f"AS7341 saved: "
            f"F1={row['f1_415nm']}, "
            f"F8={row['f8_680nm']}, "
            f"NIR={row['nir']}"
        )

    except Exception as e:
        logger.error(f"Failed to save AS7341 data: {e}")


# ==================== AS7341 Polling ====================
def _as7341_send_read(ser):
    """Send READ over an already-open serial port and save the response."""
    ser.reset_input_buffer()
    ser.write(b"READ\n")
    logger.info("AS7341: sent READ command")

    for _ in range(5):
        line = ser.readline().decode('utf-8', errors='replace').strip()
        if not line:
            logger.warning("AS7341: no response (timeout)")
            return
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.debug(f"AS7341: skipping non-JSON line: {line!r}")
            continue

        if data.get('sensor') == 'AS7341' and 'nir' in data:
            save_as7341_data(data)
            return
        elif data.get('status') == 'error':
            logger.error(f"AS7341 error: {data.get('msg')}")
            return
        else:
            logger.info(f"AS7341 status (skipping): {line}")

    logger.warning("AS7341: no sensor data after 5 lines")


def as7341_polling_thread():
    logger.info(f"AS7341 polling thread started - interval {AS7341_INTERVAL_SECONDS}s")
    
    while True:
        ser = None
        try:
            logger.info("AS7341: opening port (ESP32 will reset)...")
            ser = serial.Serial()
            ser.port     = AS7341_PORT
            ser.baudrate = AS7341_BAUD_RATE
            ser.bytesize = serial.EIGHTBITS
            ser.parity   = serial.PARITY_NONE
            ser.stopbits = serial.STOPBITS_ONE
            ser.timeout  = 15   # readAllChannels() with ASTEP=29999 takes ~5s (2 passes x 2.5s)
            ser.rts      = False
            ser.dtr      = False
            ser.open()

            logger.info("AS7341: waiting for ESP32 ready signal (up to 20s)...")
            ready    = False
            deadline = time.time() + 20
            while time.time() < deadline:
                line = ser.readline().decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get('status') == 'ready':
                        logger.info("AS7341: ESP32 is ready")
                        ready = True
                        break
                    elif msg.get('status') == 'error':
                        logger.error(f"AS7341 startup error: {msg.get('msg')}")
                        break
                except json.JSONDecodeError:
                    logger.debug(f"AS7341 boot line: {line}")

            if not ready:
                logger.warning("AS7341: ready signal not seen — proceeding anyway")

            time.sleep(1)
            ser.reset_input_buffer()

            _as7341_send_read(ser)
            logger.info(f"AS7341: next reading in {AS7341_INTERVAL_SECONDS}s")

            while True:
                time.sleep(AS7341_INTERVAL_SECONDS)
                _as7341_send_read(ser)

        except serial.SerialException as e:
            logger.error(f"AS7341 serial error: {e}")
        except Exception as e:
            logger.error(f"AS7341 thread error: {e}")
        finally:
            if ser and ser.is_open:
                ser.close()
                logger.info("AS7341: port closed")

        logger.info("AS7341: reconnecting in 30s...")
        time.sleep(30)


# ==================== UART Reading (ESP32 Master) ====================
def read_uart_data(ser):
    """Continuously read JSON data sent by the ESP32 Master."""
    logger.info("Starting UART data collection from ESP32 Master...")

    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if not line:
                    continue

                logger.debug(f"UART received: {line[:100]}")

                try:
                    data = json.loads(line)
                    if 'master' in data and 'node1' in data and 'node2' in data:
                        save_unified_data(data)
                    else:
                        logger.warning(f"Unexpected UART data keys: {list(data.keys())}")
                except json.JSONDecodeError as e:
                    logger.error(f"UART JSON decode error: {e} | raw line: {line!r}")

        except Exception as e:
            logger.error(f"UART read error: {e}")
            time.sleep(1)


# ==================== Camera Timer Thread ====================
def camera_timer_thread():
    """Capture an image every 60 minutes. Also captures immediately on startup."""
    logger.info(f"Camera timer started - interval {CAMERA_INTERVAL_SECONDS}s")

    logger.info("Camera: taking initial photo on startup...")
    time.sleep(5)
    try:
        camera.capture_with_led()
    except Exception as e:
        logger.error(f"Initial camera capture error: {e}")

    logger.info(f"Camera: next capture in {CAMERA_INTERVAL_SECONDS}s")

    while True:
        try:
            time.sleep(CAMERA_INTERVAL_SECONDS)
            camera.capture_with_led()
        except Exception as e:
            logger.error(f"Camera timer error: {e}")
            try:
                led_off()
            except Exception:
                pass
            time.sleep(60)


# ==================== Main ====================
def main():
    logger.info("=" * 60)
    logger.info("  Thesis UART Data Collection Service Starting")
    logger.info("=" * 60)

    setup_directories()
    setup_gpio()
    camera.initialize()

    # Camera thread - every 60 minutes
    camera_thread = threading.Thread(target=camera_timer_thread, daemon=True)
    camera_thread.start()
    logger.info("Camera timer thread started")

    # AS7341 thread - every 59 minutes
    as7341_thread = threading.Thread(target=as7341_polling_thread, daemon=True)
    as7341_thread.start()
    logger.info("AS7341 polling thread started")

    # Main UART loop (ESP32 Master)
    try:
        logger.info(f"Opening serial port {SERIAL_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        logger.info("Serial port opened, waiting for ESP32 data...")
        read_uart_data(ser)

    except serial.SerialException as e:
        logger.error(f"Failed to open serial port {SERIAL_PORT}: {e}")
        logger.error("Check: serial console disabled, ESP32 wired correctly (TX->RX, RX->TX, GND->GND)")

    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")

    except Exception as e:
        logger.error(f"Fatal error: {e}")

    finally:
        logger.info("Cleaning up...")
        camera.cleanup()
        if USE_LGPIO and gpio_handle is not None:
            GPIO.gpiochip_close(gpio_handle)
        elif not USE_LGPIO:
            GPIO.cleanup()
        if 'ser' in locals() and ser.is_open:
            ser.close()
        logger.info("Service stopped")


if __name__ == "__main__":
    main()