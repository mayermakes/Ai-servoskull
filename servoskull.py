#!/usr/bin/env python3

"""Based on the Example module for Hailo Detection."""

import argparse

import cv2
import threading
import time
import math
import queue

from pi5neo import Pi5Neo

from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import Hailo
from libcamera import Transform

# Position (x, y) of the current object of interest (closest object).
# Will be updated each frame by draw_objects. None when no detections.
Object_of_interest_pos = None

# Servo configuration (adjust pins to your wiring)
PAN_GPIO = 17    # GPIO pin for pan servo (BCM numbering)
TILT_GPIO = 27   # GPIO pin for tilt servo (BCM numbering)
JAW_GPIO = 22    # GPIO pin for jaw servo (BCM numbering)

# Control gains and smoothing
PAN_K = 0.9    # degrees per pixel (proportional) — reduced for slower response
TILT_K = 0.9    # degrees per pixel (proportional) — reduced for slower response
SMOOTHING = 0.05 # interpolation factor (0..1), smaller = slower/smoother movement

# Deadzone (in pixels) around the image center where small errors are ignored
CENTER_DEADZONE = 50

# Deadzone hysteresis (pixels)
DEADZONE_HYSTERESIS = 10

# When the detected center is within EXTREME_MARGIN_RATIO of the image edge
# the servo will be driven to its mechanical extreme (min/max angle).
EXTREME_MARGIN_RATIO = 0.02

# Default servo update period in milliseconds (can be overridden via CLI)
SERVO_UPDATE_MS_DEFAULT = 60.0

# Jaw control
JAW_OPEN = 175.0  # degrees for open jaw
JAW_CLOSED = 5.0 # degrees for closed jaw

# NeoPixel LED ring configuration for Pi5Neo
LED_SPI_DEV = '/dev/spidev0.0'  # SPI device for Pi5Neo
LED_COUNT = 12                   # Number of LEDs in the ring
LED_SPEED = 800              # SPI clock frequency
# Colors in RGB format (note: Pi5Neo uses RGB order)
LED_COLOR_OFF = (0, 0, 0)
LED_COLOR_GREEN = (0, 255, 0)  # For person detection
LED_COLOR_RED = (255, 0, 0)    # For laptop/keyboard detection
LED_UPDATE_INTERVAL = 0.5     # seconds between actual LED hardware updates


class NeoPixelRing:
    """Threaded controller for a NeoPixel ring using Pi5Neo.

    Public API matches previous class: show_detection(class, confidence), clear(), cleanup().
    Internally a worker thread processes requests so LED updates run independently.
    """

    def __init__(self, spi_dev=LED_SPI_DEV, num_pixels=LED_COUNT, speed=LED_SPEED):
        self._pixels = Pi5Neo(spi_dev, num_pixels, speed)
        self.num_pixels = num_pixels
        self.active_leds = 0
        self.current_color = LED_COLOR_OFF

        # Queue of pending operations: ('show', label, confidence) or ('clear',)
        self._q = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _apply_buffer(self, label, confidence):
        # choose color
        if label == 'person':
            color = LED_COLOR_GREEN
        elif label in ('laptop', 'keyboard'):
            color = LED_COLOR_RED
        else:
            color = LED_COLOR_OFF

        num_leds = int(round(max(0.0, min(1.0, confidence)) * self.num_pixels))
        for i in range(num_leds):
            self._pixels.set_led_color(i, *color)
        for i in range(num_leds, self.num_pixels):
            self._pixels.set_led_color(i, *LED_COLOR_OFF)

        # commit immediately (worker thread handles timing)
        self._pixels.update_strip()
        self.active_leds = num_leds
        self.current_color = color

    def _worker(self):
        # worker loop drains the queue and applies the most recent command
        while self._running:
            try:
                cmd = self._q.get(timeout=0.25)
            except queue.Empty:
                continue

            # Drain to the latest command to avoid backlog.
            latest = cmd
            while True:
                try:
                    latest = self._q.get_nowait()
                except queue.Empty:
                    break

            if not self._running:
                break

            if latest[0] == 'clear':
                # clear buffer and update strip
                for i in range(self.num_pixels):
                    self._pixels.set_led_color(i, *LED_COLOR_OFF)
                self._pixels.update_strip()
                self.active_leds = 0
                self.current_color = LED_COLOR_OFF
            elif latest[0] == 'show':
                _, label, confidence = latest
                self._apply_buffer(label, confidence)

    def show_detection(self, detected_class, confidence):
        # enqueue show request
        try:
            self._q.put_nowait(('show', detected_class, confidence))
        except Exception:
            pass

    def clear(self):
        try:
            self._q.put_nowait(('clear',))
        except Exception:
            pass

    def cleanup(self):
        # stop worker and clear strip
        self._running = False
        try:
            # wake worker
            self._q.put_nowait(('clear',))
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # final clear
        for i in range(self.num_pixels):
            self._pixels.set_led_color(i, *LED_COLOR_OFF)
        self._pixels.update_strip()


class ServoController:
    """Background controller for three servos (pan, tilt, jaw) with smoothing.
    Uses RPi.GPIO as the servo backend. If RPi.GPIO is not available the
    controller becomes a no-op but still runs so the rest of the program works.
    """

    def __init__(self, pan_pin=PAN_GPIO, tilt_pin=TILT_GPIO, jaw_pin=JAW_GPIO,
                 min_angle=0, max_angle=180, smoothing=SMOOTHING, update_period=0.06):
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        self.jaw_pin = jaw_pin
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.smoothing = smoothing
        self.update_period = update_period

        self._use_rpigpio = False
        self._rpigpio_pwm = {}

        # Current and target angles
        self.current_pan = 90.0
        self.current_tilt = 90.0
        self.current_jaw = JAW_CLOSED
        self.target_pan = 90.0
        self.target_tilt = 90.0
        self.target_jaw = JAW_CLOSED

        self._running = False
        self._thread = None
        # Hysteresis state for deadzone per axis
        self.pan_in_deadzone = False
        self.tilt_in_deadzone = False
        # Last written integer-degree positions to avoid rapid tiny writes
        self.last_written_pan = None
        self.last_written_tilt = None
        self.last_written_jaw = None

        # Use RPi.GPIO PWM as the servo backend (fallback-only approach).
        try:
            import RPi.GPIO as GPIO

            self._GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pan_pin, GPIO.OUT)
            GPIO.setup(self.tilt_pin, GPIO.OUT)
            GPIO.setup(self.jaw_pin, GPIO.OUT)
            # 50Hz for standard servos
            self._rpigpio_pwm[self.pan_pin] = GPIO.PWM(self.pan_pin, 50)
            self._rpigpio_pwm[self.tilt_pin] = GPIO.PWM(self.tilt_pin, 50)
            self._rpigpio_pwm[self.jaw_pin] = GPIO.PWM(self.jaw_pin, 50)
            self._rpigpio_pwm[self.pan_pin].start(self._angle_to_duty(self.current_pan))
            self._rpigpio_pwm[self.tilt_pin].start(self._angle_to_duty(self.current_tilt))
            self._rpigpio_pwm[self.jaw_pin].start(self._angle_to_duty(self.current_jaw))
            self._use_rpigpio = True
        except Exception:
            # No servo library available; controller will be a no-op.
            self._use_rpigpio = False

    def _angle_to_pulse(self, angle):
        """Convert 0..180 angle to pigpio pulse width in microseconds.

        Typical servos use ~500..2500us.
        """
        angle = max(self.min_angle, min(self.max_angle, angle))
        return int(500 + (angle / 180.0) * 2000)

    def _angle_to_duty(self, angle):
        """Convert 0..180 angle to RPi.GPIO PWM duty cycle (for 50Hz).

        Duty cycle percent: ~2.5 (0 deg) .. 12.5 (180 deg)
        """
        angle = max(self.min_angle, min(self.max_angle, angle))
        return 2.5 + (angle / 180.0) * 10.0

    def _write_servo(self, pin, angle):
        if self._use_rpigpio:
            duty = self._angle_to_duty(angle)
            self._rpigpio_pwm[pin].ChangeDutyCycle(duty)
        else:
            # No-op if no backend
            pass

    def start(self, frame_size=(1280, 960)):
        self.frame_w, self.frame_h = frame_size
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        # Stop PWM cleanly (RPi.GPIO)
        if self._use_rpigpio:
            try:
                self._rpigpio_pwm[self.pan_pin].stop()
                self._rpigpio_pwm[self.tilt_pin].stop()
                self._rpigpio_pwm[self.jaw_pin].stop()
                self._GPIO.cleanup()
            except Exception:
                pass

    def _run(self):
        global Object_of_interest_pos
        # Loop and interpolate current angles towards targets computed from the
        # Object_of_interest_pos variable.
        while self._running:
            pos = Object_of_interest_pos
            # Compute pixel error relative to frame center
            if pos is None:
                # If no object, center the servos slowly
                dx = 0
                dy = 0
                desired_pan = 90.0
                desired_tilt = 90.0
                cx = self.frame_w / 2.0
                cy = self.frame_h / 2.0
            else:
                cx, cy = pos

                # Compute pixel error relative to center
                dx = (cx - (self.frame_w / 2.0))
                dy = (cy - (self.frame_h / 2.0))

            # Determine extreme thresholds in pixels
            extreme_x_left = self.frame_w * EXTREME_MARGIN_RATIO
            extreme_x_right = self.frame_w * (1.0 - EXTREME_MARGIN_RATIO)
            extreme_y_top = self.frame_h * EXTREME_MARGIN_RATIO
            extreme_y_bottom = self.frame_h * (1.0 - EXTREME_MARGIN_RATIO)

            # Hysteresis-enabled deadzone behavior and extremes handling
            # Pan axis
            if cx <= extreme_x_left:
                desired_pan = self.min_angle
                self.pan_in_deadzone = False
            elif cx >= extreme_x_right:
                desired_pan = self.max_angle
                self.pan_in_deadzone = False
            else:
                # Enter/leave deadzone using hysteresis thresholds
                if self.pan_in_deadzone:
                    if abs(dx) > (CENTER_DEADZONE + DEADZONE_HYSTERESIS):
                        self.pan_in_deadzone = False
                else:
                    if abs(dx) < (CENTER_DEADZONE - DEADZONE_HYSTERESIS):
                        self.pan_in_deadzone = True

                if self.pan_in_deadzone:
                    desired_pan = self.current_pan
                else:
                    desired_pan = 90.0 - dx * PAN_K

            # Tilt axis
            if cy <= extreme_y_top:
                desired_tilt = self.min_angle
                self.tilt_in_deadzone = False
            elif cy >= extreme_y_bottom:
                desired_tilt = self.max_angle
                self.tilt_in_deadzone = False
            else:
                if self.tilt_in_deadzone:
                    if abs(dy) > (CENTER_DEADZONE + DEADZONE_HYSTERESIS):
                        self.tilt_in_deadzone = False
                else:
                    if abs(dy) < (CENTER_DEADZONE - DEADZONE_HYSTERESIS):
                        self.tilt_in_deadzone = True

                if self.tilt_in_deadzone:
                    desired_tilt = self.current_tilt
                else:
                    desired_tilt = 90.0 + dy * TILT_K

            # Clamp
            desired_pan = max(self.min_angle, min(self.max_angle, desired_pan))
            desired_tilt = max(self.min_angle, min(self.max_angle, desired_tilt))

            # Smooth (interpolate)
            self.current_pan += (desired_pan - self.current_pan) * self.smoothing
            self.current_tilt += (desired_tilt - self.current_tilt) * self.smoothing

            # Set jaw position based on whether we have a target
            desired_jaw = JAW_OPEN if pos is not None else JAW_CLOSED
            # Smooth jaw
            self.current_jaw += (desired_jaw - self.current_jaw) * self.smoothing

            # Quantize to full-degree increments before writing
            q_pan = int(round(self.current_pan))
            q_tilt = int(round(self.current_tilt))
            q_jaw = int(round(self.current_jaw))

            # Write only when integer degree changed (reduces twitching)
            if self.last_written_pan is None or q_pan != self.last_written_pan:
                self._write_servo(self.pan_pin, q_pan)
                self.last_written_pan = q_pan

            if self.last_written_tilt is None or q_tilt != self.last_written_tilt:
                self._write_servo(self.tilt_pin, q_tilt)
                self.last_written_tilt = q_tilt

            if self.last_written_jaw is None or q_jaw != self.last_written_jaw:
                self._write_servo(self.jaw_pin, q_jaw)
                self.last_written_jaw = q_jaw

            time.sleep(self.update_period)


def print_debug_state(pos, servo, led_ring):
    """Print current system state for debugging."""
    # Object position
    pos_str = f"Object: {pos if pos else 'None'}"
    
    # Servo angles
    servo_str = f"Servos: pan={servo.current_pan:.1f}° tilt={servo.current_tilt:.1f}° jaw={servo.current_jaw:.1f}°"
    
    # LED state
    if led_ring and led_ring.active_leds > 0:
        color_name = "GREEN" if led_ring.current_color == LED_COLOR_GREEN else "RED"
        led_str = f"LEDs: {led_ring.active_leds}/{led_ring.num_pixels} {color_name}"
    else:
        led_str = "LEDs: OFF"
    
    # Print all state on one line, overwriting previous output
    print(f"\r{pos_str} | {servo_str} | {led_str}", end="", flush=True)

def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    """Extract detections from the HailoRT-postprocess output."""
    results = []
    for class_id, detections in enumerate(hailo_output):
        for detection in detections:
            score = detection[4]
            if score >= threshold:
                y0, x0, y1, x1 = detection[:4]
                bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                results.append([class_names[class_id], bbox, score])
    return results


def draw_objects(request):
    global Object_of_interest_pos, detections, led_ring
    # read latest detections (may be None)
    current_detections = detections or []
    # Default to None each frame; will set to (x, y) of chosen object if any.
    Object_of_interest_pos = None

    # Update LED ring based on detections (robust matching)
    if led_ring is not None:
        best_detection = None
        best_conf = 0.0
        best_label = None
        for det in current_detections:
            class_name, _, conf = det
            if not isinstance(class_name, str):
                continue
            name = class_name.lower()
            # match person, laptop or keyboard by substring to be tolerant
            if ('person' in name) or ('laptop' in name) or ('keyboard' in name):
                if conf > best_conf:
                    best_conf = conf
                    best_detection = det
                    if 'person' in name:
                        best_label = 'person'
                    elif 'laptop' in name:
                        best_label = 'laptop'
                    elif 'keyboard' in name:
                        best_label = 'keyboard'

        if best_detection is not None and best_label is not None:
            # clamp confidence
            conf = max(0.0, min(1.0, float(best_conf)))
            led_ring.show_detection(best_label, conf)
        else:
            led_ring.clear()

    if current_detections:
        # Choose the "closest" object as the one with the largest bounding-box area.
        # (Assumption: larger apparent area == closer object.)
        best_idx = None
        best_area = -1
        centers = []
        with MappedArray(request, "main") as m:
            # First draw all bounding boxes and labels as before, and compute centers/areas.
            for i, (class_name, bbox, score) in enumerate(current_detections):
                x0, y0, x1, y1 = bbox
                label = f"{class_name} %{int(score * 100)}"
                cv2.rectangle(m.array, (x0, y0), (x1, y1), (0, 255, 0, 0), 2)
                cv2.putText(m.array, label, (x0 + 5, y0 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 0), 1, cv2.LINE_AA)

                w = x1 - x0
                h = y1 - y0
                area = w * h
                cx = int(x0 + w / 2)
                cy = int(y0 + h / 2)
                centers.append((cx, cy, area))

                if area > best_area:
                    best_area = area
                    best_idx = i

            # If we found a best object, draw a red filled dot at its center and
            # record the position in Object_of_interest_pos.
            if best_idx is not None and 0 <= best_idx < len(centers):
                cx, cy, _ = centers[best_idx]
                # Red in BGR with 4 channels (XRGB8888) -> (0,0,255,0)
                cv2.circle(m.array, (cx, cy), 6, (0, 0, 255, 0), -1, cv2.LINE_AA)
                Object_of_interest_pos = (cx, cy)

if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Detection Example")
    parser.add_argument("-m", "--model", help="Path for the HEF model.",
                        default="/usr/share/hailo-models/yolov8s_h8l.hef")
    parser.add_argument("-l", "--labels", default="coco.txt",
                        help="Path to a text file containing labels.")
    parser.add_argument("-s", "--score_thresh", type=float, default=0.5,
                        help="Score threshold, must be a float between 0 and 1.")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug output (servo angles, object position, LED state)")
    parser.add_argument("--servo-update-ms", type=float, default=SERVO_UPDATE_MS_DEFAULT,
                        help="Servo update period in milliseconds (minimum 10 ms).")
    args = parser.parse_args()

    # Get the Hailo model, the input size it wants, and the size of our preview stream.
    with Hailo(args.model) as hailo:
        model_h, model_w, _ = hailo.get_input_shape()
        video_w, video_h = 1280, 960

        # Load class names from the labels file
        with open(args.labels, 'r', encoding="utf-8") as f:
            class_names = f.read().splitlines()

        # Initialize LED ring
        global led_ring
        try:
            led_ring = NeoPixelRing()
        except Exception as e:
            print(f"Warning: Could not initialize LED ring: {e}")
            led_ring = None

        # The list of detected objects to draw.
        detections = None

        # Configure and start Picamera2.
        with Picamera2() as picam2:
            main = {'size': (video_w, video_h), 'format': 'XRGB8888'}
            lores = {'size': (model_w, model_h), 'format': 'RGB888'}
            controls = {'FrameRate': 30}
            config = picam2.create_preview_configuration(main, lores=lores, controls=controls,transform=Transform(hflip=1, vflip=1))
            picam2.configure(config)

            picam2.start_preview(Preview.QTGL, x=0, y=0, width=video_w, height=video_h)
            picam2.start()
            picam2.pre_callback = draw_objects

            # Start servo controller (runs in background thread)
            update_period_s = max(0.01, args.servo_update_ms / 1000.0)
            servo = ServoController(pan_pin=PAN_GPIO, tilt_pin=TILT_GPIO, jaw_pin=JAW_GPIO,
                                     update_period=update_period_s)
            servo.start(frame_size=(video_w, video_h))

            try:
                # Process each low resolution camera frame.
                while True:
                    frame = picam2.capture_array('lores')

                    # Run inference on the preprocessed frame
                    results = hailo.run(frame)

                    # Extract detections from the inference results
                    detections = extract_detections(results, video_w, video_h, class_names, args.score_thresh)
                    
                    # Print debug info if enabled
                    if args.debug:
                        print_debug_state(Object_of_interest_pos, servo, led_ring)
            except KeyboardInterrupt:
                # Allow clean exit with Ctrl-C
                pass
            finally:
                # Stop servo controller and LED ring cleanly
                try:
                    servo.stop()
                except Exception:
                    pass
                    
                if led_ring is not None:
                    try:
                        led_ring.cleanup()
                    except Exception:
                        pass
