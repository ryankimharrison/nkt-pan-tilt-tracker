"""
main.py — Entry point for the pan-tilt tracker.

Thread layout
-------------
  Thread 1 (capture)    : grabs camera frames into a shared buffer as fast as possible
  Thread 2 (processing) : detect → track → PID → serial send
  Thread 3 (web server) : Flask server reads latest annotated frame for MJPEG streaming
  Main thread           : OpenCV display window + trackbars + keyboard input
                          (cv2.imshow MUST run on the main thread on Windows)

Keyboard shortcuts (OpenCV window)
-----------------------------------
  W / S   — Tilt up / down  (manual jog, hold to keep moving)
  A / D   — Pan  left / right
  T / M   — Toggle tracking ON / OFF (de-energizes motors when off)
  Space   — Fire (hold; works in any fire mode)
  F       — Toggle fire mode (AUTO ↔ MANUAL)
  N       — Force fire HIGH (hold; bypasses fire mode)
  E       — Emergency stop (motors stop + de-energize immediately)
  R       — Re-center (force target re-acquisition)
  Z       — Zero position counters (after manually repositioning rig)
  Q / Esc — Quit

Trackbars (OpenCV window, labeled by axis)
------------------------------------------
  Pan  Kp / Kd / MaxVel
  Tilt Kp / Kd / MaxVel
  Deadzone / EMA Alpha / Target-Lost Frames / Idle Timeout Frames
"""

from __future__ import annotations

import atexit
import collections
import signal
import sys
import threading
import time

import cv2
import numpy as np

import config
from tracker import Tracker
from controller import MotionController
from serial_comm import SerialComm, steps_to_degrees, degrees_to_steps
from web_server import WebServer
from utils.fps_counter import FPSCounter
import utils.frame_annotator as annotator
from psl_analyzer import PSLAnalyzer, draw_psl_overlay
from hand_head_controller import HandHeadController


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self):
        self.frame_lock      = threading.Lock()
        self.latest_frame    = None
        self.latest_frame_time = 0.0

        self.annotated_lock  = threading.Lock()
        self.annotated_frame = None

        self.status_lock     = threading.Lock()
        self.status: dict    = {}

        self.params_lock     = threading.Lock()
        self.pending_params: dict | None = None

        self.stop_event          = threading.Event()
        self.re_center_event     = threading.Event()
        self.estop_event         = threading.Event()
        self.motor_toggle_event  = threading.Event()
        self.cycle_target_event  = threading.Event()  # set by Tab key to cycle target slot
        self.shutdown_home_event = threading.Event()  # set by shutdown button — return to 0,0 then exit

        self.motor_enabled  = False
        self.pin12_active   = False  # True when firing condition is met
        self.pin12_manual   = False  # True when N key forces pin 12 HIGH (always fires)
        self.spacebar_held  = False  # True when spacebar is held (fires in manual mode)

        self.fire_mode           = config.FIRE_MODE  # "off" / "auto" / "manual"

        self.tracking_enabled   = True           # False = motors held still, no PID

        # (pan_vel, tilt_vel) set by keyboard jog; None = let PID run
        self.manual_jog: tuple[float, float] | None = None

        self.control_source = "pid"  # "pid" / "hand" / "head"

        self.tilt_upper_calibrated = False  # True after user presses C
        self.tilt_lower_calibrated = False  # True after user presses X
        self.limits_calibrated     = False  # True after user presses L (all limits set)

        self.est_distance_m = 0.0  # estimated distance to target (meters)

        # Stored calibration offsets so we can re-send after Arduino reset
        self.cal_pan_upper:  int | None = None
        self.cal_pan_lower:  int | None = None
        self.cal_tilt_upper: int | None = None
        self.cal_tilt_lower: int | None = None


shared = SharedState()

_serial:     SerialComm       | None = None
_camera:     cv2.VideoCapture | None = None
_web_server: WebServer        | None = None
_hand_head:  HandHeadController      = HandHeadController()


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

def _return_to_home():
    """Drive both axes back to position 0,0, then shut down."""
    if not (_serial and _serial.connected):
        print("[Main] No serial — skipping return-to-home.")
        _shutdown("shutdown (no serial)")
        return

    print("[Main] Returning to home (0,0) ...")
    _serial.enable_motors()
    _HOME_SPEED = 60.0   # deg/sec — moderate homing speed
    _THRESHOLD  = 20     # steps — close enough to call it home

    for attempt in range(200):  # max ~10 seconds at 20 Hz
        info = _serial.query_positions()
        if info is None:
            print("[Main] Position query failed during homing.")
            break

        pan_pos  = info["pan_position"]
        tilt_pos = info["tilt_position"]

        pan_done  = abs(pan_pos)  <= _THRESHOLD
        tilt_done = abs(tilt_pos) <= _THRESHOLD

        if pan_done and tilt_done:
            print(f"[Main] Home reached (pan={pan_pos}, tilt={tilt_pos}).")
            break

        # Drive toward zero: negative position → positive velocity, and vice versa
        pan_vel  = 0.0 if pan_done  else (-_HOME_SPEED if pan_pos > 0 else _HOME_SPEED)
        tilt_vel = 0.0 if tilt_done else (-_HOME_SPEED if tilt_pos > 0 else _HOME_SPEED)
        _serial.send_velocity(pan_vel, tilt_vel)
        time.sleep(0.05)

    _serial.send_velocity(0.0, 0.0)
    time.sleep(0.05)
    _shutdown("shutdown — homed")


def _shutdown(reason: str = ""):
    if shared.stop_event.is_set():
        return
    if reason:
        print(f"\n[Main] Shutdown: {reason}")
    shared.stop_event.set()

    if _serial and _serial.connected:
        print("[Main] Sending emergency stop to Arduino ...")
        _serial.emergency_stop()
        time.sleep(0.1)

    if _camera and _camera.isOpened():
        _camera.release()

    cv2.destroyAllWindows()
    print("[Main] Shutdown complete.")


def _signal_handler(sig, frame):
    _shutdown("signal received")
    sys.exit(0)


atexit.register(_shutdown, "atexit")
signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Thread 1 — Camera capture
# ---------------------------------------------------------------------------

def _capture_thread(cap: cv2.VideoCapture, fps_counter: FPSCounter):
    print("[Capture] Thread started.")
    while not shared.stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue
        fps_counter.tick()
        with shared.frame_lock:
            shared.latest_frame = frame
            shared.latest_frame_time = time.perf_counter()
    print("[Capture] Thread stopped.")


# ---------------------------------------------------------------------------
# Annotation thread — draws overlays off the critical processing path
# ---------------------------------------------------------------------------

_annotation_queue: collections.deque = collections.deque(maxlen=1)


def _annotation_thread(psl: "PSLAnalyzer"):
    """Reads annotation jobs from a single-slot deque and writes to shared.annotated_frame."""
    print("[Annotation] Thread started.")
    while not shared.stop_event.is_set():
        try:
            job = _annotation_queue.popleft()
        except IndexError:
            time.sleep(0.002)
            continue

        try:
            ann = annotator.annotate(**job["annotate_kwargs"])
            draw_psl_overlay(ann, psl.get_result(), analyzer=psl)

            # Pin 12 fire indicator
            p12 = job.get("pin12")
            if p12:
                fh, fw = ann.shape[:2]
                label, colour = p12["label"], p12["colour"]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
                tx = (fw - tw) // 2
                ty = fh - 22
                cv2.rectangle(ann, (tx - 10, ty - th - 6),
                              (tx + tw + 10, ty + 6), (0, 0, 0), -1)
                cv2.rectangle(ann, (tx - 10, ty - th - 6),
                              (tx + tw + 10, ty + 6), colour, 2)
                cv2.putText(ann, label, (tx, ty),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, colour, 2, cv2.LINE_AA)

            with shared.annotated_lock:
                shared.annotated_frame = ann
        except Exception as _e:
            import traceback
            print(f"[Annotation] Error: {_e}")
            traceback.print_exc()
            with shared.annotated_lock:
                shared.annotated_frame = job["annotate_kwargs"]["frame"].copy()

    print("[Annotation] Thread stopped.")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dynamic deadzone
# ---------------------------------------------------------------------------

def _dynamic_deadzone(target, frame_h: int) -> float:
    """
    Return a normalized deadzone value that scales linearly with how close
    the tracked person is to the camera.

    Distance proxy: bounding-box height as a fraction of frame height.
      - Small box (person far away)  → deadzone = DYNAMIC_DEADZONE_MIN_PX
      - Large box (person very close) → deadzone = DYNAMIC_DEADZONE_MAX_PX

    The pixel value is then converted to the normalized [-1, +1] error space
    using the frame half-height as the reference dimension.
    """
    min_px = float(config.DYNAMIC_DEADZONE_MIN_PX)
    max_px = float(config.DYNAMIC_DEADZONE_MAX_PX)

    if target is None:
        dz_px = min_px
    else:
        box_h = target.y2 - target.y1
        # fraction of frame height the bounding box occupies (0 = far, 1 = fills frame)
        frac  = max(0.0, min(1.0, box_h / frame_h))
        dz_px = min_px + frac * (max_px - min_px)

    # Normalize: half-height is the reference (tilt error = 1.0 at frame edge)
    return dz_px / (frame_h / 2.0)


# Distance estimation state (bbox height + heavy EMA + rate limiting)
_dist_ema  = 0.0
_dist_time = 0.0

# ---------------------------------------------------------------------------
# Thread 2 — Processing loop (detection → tracking → PID → serial)
# ---------------------------------------------------------------------------

def _processing_thread(
    tracker:    Tracker,
    controller: MotionController,
    serial:     SerialComm,
    fps_capture:   FPSCounter,
    fps_inference: FPSCounter,
    fps_loop:      FPSCounter,
    psl:        "PSLAnalyzer",
):
    print("[Processing] Thread started.")
    centered_frames = 0
    _was_idle = False      # tracks the previous frame's idle state
    _center_deenergized = False  # True when de-energized due to centre-idle
    _pin12_state        = False  # last known state of Arduino pin 12
    _last_fire_source   = "auto" # "auto" | "space" — label for the FIRE banner
    # Latency-compensating prediction state
    _prev_aim_cx    = 0.0
    _prev_aim_cy    = 0.0
    _prev_detect_t  = 0.0              # perf_counter time of previous detection
    _vel_px_x       = 0.0              # EMA-smoothed velocity in px/sec
    _vel_px_y       = 0.0
    _had_target     = False
    _vel_stable_cnt = 0                # frames of stable velocity
    _cal_check_ctr  = 0                # calibration watchdog frame counter

    # Hand/head position control state
    _hh_pos_initialized = False
    _hh_pan_pos_deg   = 0.0     # estimated pan position (raw degrees)
    _hh_tilt_pos_deg  = 0.0     # estimated tilt position (raw degrees)
    _hh_pan_center    = 0.0     # calibrated center (raw degrees)
    _hh_tilt_center   = 0.0
    _hh_pan_limit_lo  = None    # limits relative to center
    _hh_pan_limit_hi  = None
    _hh_tilt_limit_lo = None
    _hh_tilt_limit_hi = None
    _hh_last_time     = 0.0     # for velocity integration

    while not shared.stop_event.is_set():

        # ── control events from UI / keyboard ──
        if shared.estop_event.is_set():
            shared.estop_event.clear()
            serial.emergency_stop()
            shared.motor_enabled = False
            controller.reset()

        if shared.shutdown_home_event.is_set():
            shared.shutdown_home_event.clear()
            threading.Thread(target=_return_to_home, daemon=True, name="HomingShutdown").start()
            break

        if shared.re_center_event.is_set():
            shared.re_center_event.clear()
            tracker.reset_target()
            controller.reset()

        if shared.motor_toggle_event.is_set():
            shared.motor_toggle_event.clear()
            # tracking_enabled already flipped by the display thread;
            # here we just handle the hardware side.
            if not shared.tracking_enabled and shared.motor_enabled:
                serial.disable_motors()
                shared.motor_enabled = False
                controller.reset()
            print(f"[Processing] Tracking: {'ON' if shared.tracking_enabled else 'OFF'}")

        # ── apply pending param updates from trackbars / Flask ──
        with shared.params_lock:
            pending = shared.pending_params
            shared.pending_params = None
        if pending:
            controller.update_params(**pending)

        # ── grab latest frame (single source: display shows exactly what detector sees) ──
        with shared.frame_lock:
            frame = shared.latest_frame
            _frame_time = shared.latest_frame_time
        if frame is None:
            time.sleep(0.001)
            continue

        # ── software brightness normalization ──
        if config.AUTO_BRIGHTNESS_TARGET > 0:
            _mean = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]
            if _mean > 1.0:
                _scale = config.AUTO_BRIGHTNESS_TARGET / _mean
                # Gamma-style scale: gentle in highlights, stronger in shadows
                frame = cv2.convertScaleAbs(frame, alpha=_scale, beta=0)

        h, w = frame.shape[:2]

        # ── detection + tracking (single model.track() call) ──
        track_result = tracker.update(frame, (h, w), cycle_target_event=shared.cycle_target_event)
        fps_inference.tick()
        tracks  = track_result.tracks
        is_idle = track_result.is_idle

        # When tracking is disabled OR limits not calibrated, suppress target so PID stays zero
        target = track_result.target if (shared.tracking_enabled and shared.limits_calibrated) else None

        # ── idle de-energize (fire only on the transition into idle) ──
        if (is_idle or not shared.tracking_enabled) and not _was_idle:
            if shared.motor_enabled:
                serial.disable_motors()
                shared.motor_enabled = False
            controller.reset()
            if is_idle:
                print("[Processing] Idle timeout — motors de-energized.")
        _was_idle = is_idle or not shared.tracking_enabled

        # ── Calibration watchdog: re-send limits if Arduino reset ──
        _cal_check_ctr += 1
        if (_cal_check_ctr % 300 == 0              # every ~5 sec at 60fps
                and shared.limits_calibrated
                and shared.cal_pan_upper is not None
                and serial.connected):
            _q = serial.query_positions()
            if _q and not _q.get("pan_calibrated", True):
                # Arduino lost calibration (likely reset) — re-send limits
                print("[Processing] Arduino lost calibration — re-sending limits ...")
                serial.calibrate_from_center(
                    shared.cal_pan_upper, shared.cal_pan_lower,
                    shared.cal_tilt_upper, shared.cal_tilt_lower)
                print("[Processing] Limits restored.")

        # ── PID ──
        pan_error = tilt_error = pan_vel = tilt_vel = 0.0

        if target is not None and shared.tracking_enabled:
            # ── Latency-compensating prediction ──
            _now_t = time.perf_counter()
            if _had_target:
                dt_detect = _now_t - _prev_detect_t
                if dt_detect > 0.001:
                    raw_vx = (target.aim_cx - _prev_aim_cx) / dt_detect
                    raw_vy = (target.aim_cy - _prev_aim_cy) / dt_detect
                    _a = config.PREDICTION_VEL_ALPHA
                    _vel_px_x = _a * raw_vx + (1 - _a) * _vel_px_x
                    _vel_px_y = _a * raw_vy + (1 - _a) * _vel_px_y
                    _vel_stable_cnt += 1
            else:
                _vel_px_x = 0.0
                _vel_px_y = 0.0
                _vel_stable_cnt = 0
            _prev_aim_cx   = target.aim_cx
            _prev_aim_cy   = target.aim_cy
            _prev_detect_t = _now_t
            _had_target    = True

            # Predict forward by measured pipeline latency
            aim_cx = target.aim_cx
            aim_cy = target.aim_cy
            if config.PREDICTION_ENABLED and _vel_stable_cnt >= config.PREDICTION_MIN_FRAMES:
                frame_age = max(0.0, _now_t - _frame_time) if _frame_time > 0 else 0.0
                predict_sec = frame_age + config.PREDICTION_EXTRA_MS / 1000.0
                predict_sec = min(predict_sec, config.PREDICTION_MAX_SEC)
                aim_cx += _vel_px_x * predict_sec
                aim_cy += _vel_px_y * predict_sec

            pan_error  = (aim_cx - w / 2.0) / (w / 2.0)
            tilt_error = (aim_cy - h / 2.0) / (h / 2.0)

            # Distance-adaptive gain: boost error when target is far (small box)
            # Soft rolloff above 4x to prevent gain discontinuity at distance
            _bbox_h_px = target.y2 - target.y1
            if config.DISTANCE_GAIN_REF_HEIGHT > 0.0:
                _bbox_h_frac = _bbox_h_px / h
                if _bbox_h_frac > 0.01:
                    _raw_scale = config.DISTANCE_GAIN_REF_HEIGHT / _bbox_h_frac
                    if _raw_scale <= 4.0:
                        _dist_scale = _raw_scale
                    else:
                        _dist_scale = 4.0 + (_raw_scale - 4.0) ** 0.5
                    _dist_scale = min(_dist_scale, config.DISTANCE_GAIN_MAX)
                    pan_error   *= _dist_scale
                    tilt_error  *= _dist_scale

            # Distance estimate: bbox height + heavy EMA + rate limiting
            global _dist_ema, _dist_time
            if _bbox_h_px > 10:
                _raw_dist = (config.DISTANCE_PERSON_HEIGHT_M * h) / _bbox_h_px
                _now_d = time.perf_counter()
                if _dist_ema <= 0:
                    _dist_ema = _raw_dist
                    _dist_time = _now_d
                else:
                    # Rate limit: clamp raw to max slew rate before EMA
                    _dt_d = _now_d - _dist_time
                    if _dt_d > 0:
                        _max_delta = config.DISTANCE_MAX_RATE_MPS * _dt_d
                        _clamped = max(_dist_ema - _max_delta,
                                       min(_dist_ema + _max_delta, _raw_dist))
                    else:
                        _clamped = _raw_dist
                    # Heavy EMA on top of rate-limited value
                    _a = config.DISTANCE_EMA_ALPHA
                    _dist_ema = _a * _clamped + (1 - _a) * _dist_ema
                    _dist_time = _now_d
                shared.est_distance_m = _dist_ema

            # Dynamic deadzone: scales with target proximity
            if config.DYNAMIC_DEADZONE:
                _dz = _dynamic_deadzone(target, h)
                controller.pan.deadzone = _dz
                controller.tilt.deadzone = _dz

            pan_vel, tilt_vel = controller.compute(pan_error, tilt_error)

            # Hysteresis re-enable: after a centre-idle de-energize, only wake motors
            # once error exceeds the centering threshold, preventing click cycling.
            _ct = config.CENTERING_THRESHOLD
            if not shared.motor_enabled:
                should_enable = (
                    not _center_deenergized
                    or abs(pan_error)  > _ct * 2.0
                    or abs(tilt_error) > _ct * 2.0
                )
                if should_enable:
                    serial.enable_motors()
                    shared.motor_enabled = True
                    _center_deenergized = False
                    print("[Processing] Target acquired — motors enabled.")
                else:
                    pan_vel = tilt_vel = 0.0

            if abs(pan_error) < _ct and abs(tilt_error) < _ct:
                centered_frames += 1
            else:
                centered_frames = 0

            if config.CENTER_IDLE_FRAMES > 0 and centered_frames > config.CENTER_IDLE_FRAMES:
                # Target centred long enough — de-energize to relieve driver heat.
                # Hysteresis above ensures we won't immediately re-enable next frame.
                if shared.motor_enabled:
                    serial.disable_motors()
                    shared.motor_enabled = False
                    _center_deenergized = True
                controller.reset()
                pan_vel = tilt_vel = 0.0
        else:
            controller.reset()
            centered_frames = 0
            _center_deenergized = False
            _had_target = False
            _vel_px_x = 0.0
            _vel_px_y = 0.0
            _vel_stable_cnt = 0

        # ── manual jog overrides PID; hand/head overrides PID too ──
        jog = shared.manual_jog
        if jog is not None:
            pan_vel, tilt_vel = jog
            if not shared.motor_enabled:
                serial.enable_motors()
                shared.motor_enabled = True
        elif shared.control_source in ("hand", "head"):
            hh = _hand_head.get_result()
            if hh is not None:
                if not shared.motor_enabled:
                    serial.enable_motors()
                    shared.motor_enabled = True

                # One-time init: query Arduino for position + limits
                # Convert everything to NATURAL frame (positive pan=right,
                # positive tilt=down as camera sees it).  Arduino reports
                # in motor frame, which is inverted when *_INVERT is True.
                if not _hh_pos_initialized:
                    _hh_pos_initialized = True
                    _hh_last_time = time.monotonic()
                    pos = serial.query_positions()
                    if pos is not None:
                        _hh_pan_pos_deg = steps_to_degrees(
                            pos["pan_position"],
                            config.PAN_GEAR_RATIO, config.PAN_MICROSTEP_MULTIPLIER)
                        _hh_tilt_pos_deg = steps_to_degrees(
                            pos["tilt_position"],
                            config.TILT_GEAR_RATIO, config.TILT_MICROSTEP_MULTIPLIER)
                        # Convert motor frame → natural frame
                        if config.PAN_INVERT:
                            _hh_pan_pos_deg = -_hh_pan_pos_deg
                        if config.TILT_INVERT:
                            _hh_tilt_pos_deg = -_hh_tilt_pos_deg

                        if pos["pan_calibrated"]:
                            lo = steps_to_degrees(pos["pan_lower_limit"],
                                config.PAN_GEAR_RATIO, config.PAN_MICROSTEP_MULTIPLIER)
                            hi = steps_to_degrees(pos["pan_upper_limit"],
                                config.PAN_GEAR_RATIO, config.PAN_MICROSTEP_MULTIPLIER)
                            if config.PAN_INVERT:
                                lo, hi = -hi, -lo
                            _hh_pan_center = (lo + hi) / 2.0
                            _hh_pan_limit_lo = lo - _hh_pan_center
                            _hh_pan_limit_hi = hi - _hh_pan_center
                        if pos["tilt_upper_calibrated"] and pos["tilt_lower_calibrated"]:
                            lo = steps_to_degrees(pos["tilt_lower_limit"],
                                config.TILT_GEAR_RATIO, config.TILT_MICROSTEP_MULTIPLIER)
                            hi = steps_to_degrees(pos["tilt_upper_limit"],
                                config.TILT_GEAR_RATIO, config.TILT_MICROSTEP_MULTIPLIER)
                            if config.TILT_INVERT:
                                lo, hi = -hi, -lo
                            _hh_tilt_center = (lo + hi) / 2.0
                            _hh_tilt_limit_lo = lo - _hh_tilt_center
                            _hh_tilt_limit_hi = hi - _hh_tilt_center

                # Position relative to calibrated center (natural frame)
                pan_rel  = _hh_pan_pos_deg  - _hh_pan_center
                tilt_rel = _hh_tilt_pos_deg - _hh_tilt_center

                # Map normalized [-1, 1] to target degrees (relative to center)
                target_pan_deg  = hh[0] * config.HAND_CONTROL_PAN_RANGE
                target_tilt_deg = hh[1] * config.HAND_CONTROL_TILT_RANGE

                # Clamp target to calibrated limits
                if _hh_pan_limit_lo is not None and _hh_pan_limit_hi is not None:
                    target_pan_deg = max(_hh_pan_limit_lo,
                                         min(_hh_pan_limit_hi, target_pan_deg))
                if _hh_tilt_limit_lo is not None and _hh_tilt_limit_hi is not None:
                    target_tilt_deg = max(_hh_tilt_limit_lo,
                                          min(_hh_tilt_limit_hi, target_tilt_deg))

                # P-controller: velocity = error * Kp (natural frame)
                # Tilt disabled for hand/head — pan only
                pan_vel  = (target_pan_deg  - pan_rel)  * config.HAND_CONTROL_KP
                tilt_vel = 0.0

                # Clamp velocity
                max_v = config.HAND_CONTROL_MAX_VEL
                pan_vel  = max(-max_v, min(max_v, pan_vel))
                tilt_vel = max(-max_v, min(max_v, tilt_vel))

                # Integrate velocity to estimate position (natural frame)
                _now_hh = time.monotonic()
                dt = _now_hh - _hh_last_time
                _hh_last_time = _now_hh
                if dt > 0 and dt < 0.2:
                    _hh_pan_pos_deg  += pan_vel  * dt
                    _hh_tilt_pos_deg += tilt_vel * dt

                # Velocity is in natural frame — send_velocity handles
                # PAN_INVERT / TILT_INVERT internally, no pre-negation needed.
        else:
            # Not in hand/head mode — reset so we re-query on next entry
            _hh_pos_initialized = False

        # ── send velocity ──
        if shared.motor_enabled:
            serial.send_velocity(pan_vel, tilt_vel)

        # ── fire zone: crosshair inside tracked person's bounding box ──
        _in_fire_zone = False
        if target is not None:
            cx_frame = w / 2.0
            cy_frame = h / 2.0
            _in_fire_zone = (target.x1 <= cx_frame <= target.x2
                             and target.y1 <= cy_frame <= target.y2)

        # ── pin 12: fire signal ──
        # Auto:      HIGH while crosshair is within person's bounding box.
        # Spacebar:  HIGH while held (MANUAL mode only).
        # N key:     HIGH while held (any mode, force override).

        # Auto: hold HIGH while in fire zone
        _auto_on = shared.fire_mode == "auto" and _in_fire_zone
        if _auto_on:
            _last_fire_source = "auto"

        # Spacebar: hold HIGH while held (manual mode only)
        _sp_on = shared.spacebar_held and shared.fire_mode == "manual"
        if _sp_on:
            _last_fire_source = "space"

        # N key: hold HIGH while held (any mode)
        _n_on = shared.pin12_manual

        _want_pin12 = _auto_on or _sp_on or _n_on
        if _want_pin12 != _pin12_state:
            serial.set_signal(_want_pin12)
            _pin12_state = _want_pin12
            shared.pin12_active = _want_pin12

        # ── steps/sec for display ──
        pan_sps  = (pan_vel  * config.PAN_GEAR_RATIO  * config.STEPS_PER_REV * config.PAN_MICROSTEP_MULTIPLIER)  / 360.0
        tilt_sps = (tilt_vel * config.TILT_GEAR_RATIO * config.STEPS_PER_REV * config.TILT_MICROSTEP_MULTIPLIER) / 360.0

        # ── PSL: feed raw frame (isolated thread does the heavy lifting) ──
        psl.push_frame(frame)

        # ── push annotation job to background thread (off critical path) ──
        params = controller.get_params()
        _ann_job: dict = {
            "annotate_kwargs": {
                "frame": frame,
                "tracks": tracks,
                "target": target,
                "pan_error": pan_error,
                "tilt_error": tilt_error,
                "pan_vel": pan_vel,
                "tilt_vel": tilt_vel,
                "deadzone": controller.pan.deadzone,
                "fps_capture": fps_capture.get(),
                "fps_inference": fps_inference.get(),
                "fps_loop": fps_loop.get(),
                "person_count": len(tracks),
                "motor_enabled": shared.motor_enabled,
                "is_idle": is_idle,
                "pan_steps_sec": pan_sps,
                "tilt_steps_sec": tilt_sps,
                "params": params,
                "fire_mode": shared.fire_mode,
                "is_firing": _pin12_state,
                "tracking_enabled": shared.tracking_enabled,
                "in_fire_zone": _in_fire_zone,
                "locked_slot": tracker._locked_slot,
                "tilt_upper_calibrated": shared.tilt_upper_calibrated,
                "tilt_lower_calibrated": shared.tilt_lower_calibrated,
                "limits_calibrated": shared.limits_calibrated,
                "control_source": shared.control_source,
                "est_distance_m": getattr(shared, 'est_distance_m', 0.0),
            },
        }

        if _pin12_state:
            if shared.pin12_manual:
                _ann_job["pin12"] = {"label": "FIRE  [N-OVERRIDE]", "colour": (0, 165, 255)}
            elif _last_fire_source == "space":
                _ann_job["pin12"] = {"label": "FIRE  [\u25a0SPACE]", "colour": (0, 200, 255)}
            else:
                _ann_job["pin12"] = {"label": "FIRE  [AUTO]", "colour": (0, 0, 220)}
        _annotation_queue.append(_ann_job)

        # ── status for Flask UI ──
        with shared.status_lock:
            shared.status = {
                "fps_capture":    round(fps_capture.get(),    1),
                "fps_inference":  round(fps_inference.get(),  1),
                "fps_loop":       round(fps_loop.get(),       1),
                "target_id":      target.track_id if target else None,
                "pan_error":      round(pan_error,  4),
                "tilt_error":     round(tilt_error, 4),
                "pan_vel":        round(pan_vel,    2),
                "tilt_vel":       round(tilt_vel,   2),
                "pan_steps_sec":  round(pan_sps,    1),
                "tilt_steps_sec": round(tilt_sps,   1),
                "motor_enabled":  shared.motor_enabled,
                "is_idle":        is_idle,
                "person_count":   len(tracks),
                "fire_mode":              shared.fire_mode,
                "tracking_enabled":       shared.tracking_enabled,
                "is_firing":              _pin12_state,
                "in_fire_zone":           _in_fire_zone,
                "tilt_upper_calibrated":  shared.tilt_upper_calibrated,
                "tilt_lower_calibrated":  shared.tilt_lower_calibrated,
                "est_distance_m":         round(getattr(shared, 'est_distance_m', 0.0), 2),
                **params,
            }

        fps_loop.tick()

    print("[Processing] Thread stopped.")


# ---------------------------------------------------------------------------
# Main thread — OpenCV display window + trackbars + keyboard
# ---------------------------------------------------------------------------

_WINDOW = "Pan-Tilt Tracker"

# Trackbar integer ranges and scale factors (int_val * scale = float_param)
# Labels kept very short (≤6 chars) so OpenCV doesn't truncate.
# Pan Gain: /100 → 0.10–3.00 (1.0 = no change)
_TB = {
    # name     : (min,  max, scale,  param_key)
    "P-Kp"     : (0,   1000, 1.0,    "pan_kp"),
    "P-Kd"     : (0,     50, 0.1,    "pan_kd"),
    "P-Gain"   : (10,   300, 0.01,   "pan_gain"),   # 0.10–3.00
    "P-Vel"    : (10,  1000, 1.0,    "max_pan_velocity"),
    "T-Kp"     : (0,   1000, 1.0,    "tilt_kp"),
    "T-Kd"     : (0,     50, 0.1,    "tilt_kd"),
    "T-Vel"    : (10,   500, 1.0,    "max_tilt_velocity"),
    "Curve"    : (5,     30, 0.1,    "curve_exponent"),
    "EMA"      : (1,    100, 0.01,   "ema_alpha"),
    "DZ"       : (1,     50, 1.0,    None),   # /1000 → CENTERING_THRESHOLD
    "Lost"     : (5,    120, 1.0,    None),
    "Idle"     : (10,   300, 1.0,    None),
    "Lead"     : (0,     40, 0.5,    None),   # 0–20.0 frames ahead
    "LeadAcc"  : (0,     30, 0.5,    None),   # 0–15.0 accel gain
}


def _float_default(key: str) -> int:
    """Return the initial integer trackbar position from config."""
    defaults = {
        "pan_kp":            config.PAN_KP,
        "pan_kd":            config.PAN_KD,
        "max_pan_velocity":  config.MAX_PAN_VELOCITY,
        "tilt_kp":           config.TILT_KP,
        "tilt_kd":           config.TILT_KD,
        "max_tilt_velocity": config.MAX_TILT_VELOCITY,
        "centering_threshold": config.CENTERING_THRESHOLD,
        "ema_alpha":         config.EMA_ALPHA,
    }
    return int(defaults.get(key, 0))


def _make_trackbar_cb(tb_name: str, scale: float, param_key: str | None, controller: MotionController):
    """Return an OpenCV trackbar onChange callback."""
    def cb(int_val: int):
        float_val = int_val * scale
        if param_key is not None:
            with shared.params_lock:
                if shared.pending_params is None:
                    shared.pending_params = {}
                shared.pending_params[param_key] = float_val
            controller.update_params(**{param_key: float_val})
        else:
            # Integer / config-only params (no controller key)
            if tb_name == "Lost":
                config.TARGET_LOST_FRAMES = max(1, int(int_val))
            elif tb_name == "Idle":
                config.IDLE_TIMEOUT_FRAMES = max(1, int(int_val))
            elif tb_name == "DZ":
                # Repurposed: controls centering threshold (normalized).
                # Slider range 1–50, value / 1000 → 0.001–0.050
                config.CENTERING_THRESHOLD = max(0.001, int_val / 1000.0)
            elif tb_name == "Lead":
                config.LEAD_GAIN = max(0.0, int_val * 0.5)
            elif tb_name == "LeadAcc":
                config.LEAD_ACCEL_GAIN = max(0.0, int_val * 0.5)
    return cb


_CTRL_WINDOW = "Controls"

# Legend for each control (shown in panel below trackbars)
_CTRL_LEGEND = {
    "P-Kp":   "Pan proportional gain (higher = stronger correction)",
    "P-Kd":   "Pan derivative / damping",
    "P-Gain": "Pan output multiplier (1.0=normal, 2.0=2x speed)",
    "P-Vel":  "Pan max velocity deg/sec",
    "T-Kp":   "Tilt proportional gain",
    "T-Kd":   "Tilt derivative / damping",
    "T-Vel":  "Tilt max velocity deg/sec",
    "Curve":  "Gain curve (1.0=linear, 1.3=softer near center)",
    "EMA":    "Smoothing (higher=smoother, more lag)",
    "DZ":     "Center tolerance (1-50 = 0.001-0.050 normalized, smaller=tighter)",
    "Lost":   "Frames with no detections before target cleared",
    "Idle":   "Frames with no detections before motors off",
    "Lead":   "Lead gain — frames ahead to aim (velocity prediction)",
    "LeadAcc":"Lead accel — extra prediction from acceleration",
}


def _build_legend_image() -> np.ndarray:
    """Build a panel image with control descriptions."""
    h, w = 230, 880
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # dark gray background
    y = 18
    for tb_name, desc in _CTRL_LEGEND.items():
        cv2.putText(img, f"{tb_name}:", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
        cv2.putText(img, desc, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y += 15
    return img


def _setup_trackbars(controller: MotionController):
    """
    Create all trackbars in the separate Controls window.
    A 1-pixel-tall dummy image is shown so OpenCV has something to display;
    the trackbars are what actually matter.
    """
    init_ints = {
        "P-Kp"     : int(config.PAN_KP),
        "P-Kd"     : int(config.PAN_KD / 0.1),
        "P-Gain"   : 100,  # 1.0 = no change
        "P-Vel"    : int(config.MAX_PAN_VELOCITY),
        "T-Kp"     : int(config.TILT_KP),
        "T-Kd"     : int(config.TILT_KD / 0.1),
        "T-Vel"    : int(config.MAX_TILT_VELOCITY),
        "Curve"    : int(config.GAIN_CURVE_EXPONENT / 0.1),
        "EMA"      : int(config.EMA_ALPHA / 0.01),
        "DZ"       : int(config.CENTERING_THRESHOLD * 1000),
        "Lost"     : config.TARGET_LOST_FRAMES,
        "Idle"     : config.IDLE_TIMEOUT_FRAMES,
        "Lead"     : int(config.LEAD_GAIN / 0.5),
        "LeadAcc"  : int(config.LEAD_ACCEL_GAIN / 0.5),
    }
    # Controls window — trackbars + legend panel below
    cv2.namedWindow(_CTRL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_CTRL_WINDOW, 900, 420)

    for tb_name, (tb_min, tb_max, scale, param_key) in _TB.items():
        init_val = max(tb_min, min(tb_max, init_ints[tb_name]))
        cb = _make_trackbar_cb(tb_name, scale, param_key, controller)
        cv2.createTrackbar(tb_name, _CTRL_WINDOW, init_val, tb_max, cb)


def _display_loop(controller: MotionController):
    """
    Main-thread display loop.
    Camera window = exact frame size (WINDOW_AUTOSIZE, no trackbars).
    Trackbars live in a separate Controls window.
    """
    # Camera window — AUTOSIZE so it's exactly the frame resolution, no scaling
    cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)

    # Controls window with all trackbars
    _setup_trackbars(controller)

    # Show the controls window with legend panel (descriptions for each control)
    _ctrl_img = _build_legend_image()
    cv2.imshow(_CTRL_WINDOW, _ctrl_img)

    print(f"[Display] Windows open. Keyboard (click camera window first): "
          f"WASD=jog  E=estop  M=motor  R=re-center  Q/Esc=quit")

    _prev_key = 255  # track previous key to detect key-down edges
    _jog_last_time = 0.0  # monotonic timestamp of last WASD keypress
    _JOG_HOLD_GRACE = 0.10  # seconds — bridge gaps between OS key-repeat events
    _jog_active = False  # True while jogging (grace period still open)
    _HOLD_GRACE = 0.12  # seconds — grace period for N / spacebar hold
    _n_last_time = 0.0
    _sp_last_time = 0.0

    while not shared.stop_event.is_set():
        with shared.annotated_lock:
            frame = shared.annotated_frame

        # Display = same frame that was passed to YOLO (with annotations overlaid)
        if frame is not None:
            cv2.imshow(_WINDOW, frame)
        cv2.imshow(_CTRL_WINDOW, _ctrl_img)  # keep legend visible

        key = cv2.waitKey(16) & 0xFF  # ~60 fps poll
        _key_down = (key != _prev_key)  # True only on the first frame a key appears

        # ── N hold: unconditional pin 12 override (HIGH while held) ──
        # Uses grace period to bridge gaps between OS key-repeat events.
        _now_hold = time.monotonic()
        if key == ord('n') or key == ord('N'):
            _n_last_time = _now_hold
        shared.pin12_manual = (_now_hold - _n_last_time) < _HOLD_GRACE

        # ── Spacebar hold: fire in MANUAL mode ──
        if key == ord(' '):
            _sp_last_time = _now_hold
        shared.spacebar_held = (_now_hold - _sp_last_time) < _HOLD_GRACE

        # ── WASD manual jog (hold key = continuous movement) ──
        # Send velocity directly from this thread for instant response.
        # Use a grace period to bridge gaps between OS key-repeat events
        # so the motors don't see rapid on/off/on/off jitter.
        _JOG = config.MANUAL_JOG_VELOCITY
        _now_jog = time.monotonic()
        _wasd = False
        if   key == ord('d') or key == ord('D'):
            shared.manual_jog = ( _JOG,   0.0);  _wasd = True
        elif key == ord('a') or key == ord('A'):
            shared.manual_jog = (-_JOG,   0.0);  _wasd = True
        elif key == ord('w') or key == ord('W'):
            shared.manual_jog = (  0.0, -_JOG);  _wasd = True  # tilt up = negative error
        elif key == ord('s') or key == ord('S'):
            shared.manual_jog = (  0.0,  _JOG);  _wasd = True

        if _wasd:
            _jog_last_time = _now_jog
            _jog_active = True
            # Send immediately — don't wait for processing loop
            if _serial and _serial.connected:
                if not shared.motor_enabled:
                    _serial.enable_motors()
                    shared.motor_enabled = True
                _serial.send_velocity(shared.manual_jog[0], shared.manual_jog[1])

        if not _wasd:
            if _key_down and (key == ord('f') or key == ord('F')):
                _cycle = {"off": "auto", "auto": "manual", "manual": "off"}
                shared.fire_mode = _cycle.get(shared.fire_mode, "off")
                print(f"[Display] Fire mode: {shared.fire_mode.upper()}")
            elif _key_down and key == 9:  # Tab — cycle to next target slot
                shared.cycle_target_event.set()
            elif _key_down and (key == ord('t') or key == ord('T') or key == ord('m') or key == ord('M')):
                shared.tracking_enabled = not shared.tracking_enabled
                shared.motor_toggle_event.set()  # notify processing thread
                print(f"[Display] Tracking: {'ON' if shared.tracking_enabled else 'OFF'}")
            elif _key_down and (key == ord('h') or key == ord('H')):
                _hh_cycle = {"pid": "hand", "hand": "head", "head": "pid"}
                shared.control_source = _hh_cycle.get(shared.control_source, "pid")
                _hand_head.set_mode("off" if shared.control_source == "pid" else shared.control_source)
                print(f"[Display] Control: {shared.control_source.upper()}")

        # Clear jog only after grace period expires (bridges OS key-repeat gaps)
        if _jog_active and not _wasd and (_now_jog - _jog_last_time) > _JOG_HOLD_GRACE:
            shared.manual_jog = None
            _jog_active = False

        if key == ord('q') or key == 27:
            shared.manual_jog = None
            print("[Display] Quit key pressed.")
            _shutdown("quit key")
            break
        elif key == ord('e') or key == ord('E'):
            shared.manual_jog = None
            print("[Display] Emergency stop.")
            shared.estop_event.set()
        elif key == ord('r') or key == ord('R'):
            print("[Display] Re-center.")
            shared.re_center_event.set()
        elif key == ord('z') or key == ord('Z'):
            if _serial and _serial.connected:
                _serial.zero_position()
                shared.tilt_upper_calibrated = False
                shared.tilt_lower_calibrated = False
                shared.limits_calibrated = False
                print("[Display] Position counters zeroed. All limits cleared.")
            else:
                print("[Display] Z key: no serial connection.")
        elif _key_down and (key == ord('c') or key == ord('C')):
            if _serial and _serial.connected:
                if _serial.calibrate_tilt_limit():
                    shared.tilt_upper_calibrated = True
                    print("[Display] Tilt UPPER limit calibrated.")
                else:
                    print("[Display] Tilt upper calibration failed (no reply).")
            else:
                print("[Display] C key: no serial connection.")
        elif _key_down and (key == ord('x') or key == ord('X')):
            if _serial and _serial.connected:
                if _serial.calibrate_tilt_lower_limit():
                    shared.tilt_lower_calibrated = True
                    print("[Display] Tilt LOWER limit calibrated.")
                else:
                    print("[Display] Tilt lower calibration failed (no reply).")
            else:
                print("[Display] X key: no serial connection.")
        elif _key_down and (key == ord('i') or key == ord('I')):
            if _serial and _serial.connected:
                info = _serial.query_positions()
                if info:
                    print(f"[Position Info]  "
                          f"Pan: {info['pan_position']}  limits [{info['pan_lower_limit']}, {info['pan_upper_limit']}] ({'SET' if info['pan_calibrated'] else 'NOT SET'})  |  "
                          f"Tilt: {info['tilt_position']}  upper={info['tilt_upper_limit']} ({'SET' if info['tilt_upper_calibrated'] else 'NOT SET'})  "
                          f"lower={info['tilt_lower_limit']} ({'SET' if info['tilt_lower_calibrated'] else 'NOT SET'})")
                else:
                    print("[Display] Position query failed (no reply).")
            else:
                print("[Display] I key: no serial connection.")
        elif _key_down and (key == ord('l') or key == ord('L')):
            # One-button calibration: point away from you (center), press L
            # Pan: ±2662 steps (±110°), Tilt: +978 upper, -2898 lower
            if _serial and _serial.connected:
                if _serial.calibrate_from_center(2662, -2662, 978, -2898):
                    shared.tilt_upper_calibrated = True
                    shared.tilt_lower_calibrated = True
                    shared.limits_calibrated = True
                    # Store offsets so processing thread can re-send after Arduino reset
                    shared.cal_pan_upper  = 2662
                    shared.cal_pan_lower  = -2662
                    shared.cal_tilt_upper = 978
                    shared.cal_tilt_lower = -2898
                    print("[Display] All limits set from center — Pan: ±2662 (±110°), Tilt: +978/-2898")
                else:
                    print("[Display] Calibration from center failed (no reply).")
            else:
                print("[Display] L key: no serial connection.")

        _prev_key = key

        # Exit if main camera window is closed
        try:
            if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                print("[Display] Window closed.")
                _shutdown("window closed")
                break
        except cv2.error:
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# External camera auto-detection
# ---------------------------------------------------------------------------

# USB Vendor IDs (VIDs) of chips that are almost exclusively used in
# integrated/built-in laptop webcams.  Any camera whose InstanceId contains
# one of these VIDs is classified as internal.
_INTERNAL_VIDS = {
    "VID_30C9",  # Generalplus / Sunplus2  — HP, Acer, Asus integrated cams
    "VID_04F2",  # Chicony Electronics     — Lenovo, HP ThinkPad
    "VID_0408",  # Quanta Computer         — HP, Dell OEM
    "VID_05CA",  # Ricoh
    "VID_05C8",  # Cheng Uei / FoxConn
    "VID_2232",  # Silicon Motion
    "VID_064E",  # Sunplus Innovation
    "VID_0C45",  # Microdia
    "VID_13D3",  # IMC Networks
    "VID_0457",  # Silicon Integrated Systems
    "VID_0BDA",  # Realtek (integrated USB cam variant)
    "VID_174F",  # Syntek
    "VID_0AC8",  # Vimicro
    "VID_1BCF",  # Suyin
}

# Name fragments that also indicate a built-in camera (case-insensitive).
# Used as a fallback when VID matching is inconclusive.
_INTERNAL_NAME_KW = [
    "integrated", "built-in", "builtin", "internal",
    "ir camera", "windows hello", "facetime",
    "wide vision",                  # HP Wide Vision HD Camera
    "truevision", "hd camera",      # generic HP integrated names
    "bison", "chicony", "suyin",
    "syntek", "alcor", "azurewave",
    "quanta", "realtek webcam",
]


def _query_pnp_cameras() -> list[dict]:
    """
    Return a list of dicts {name, instance_id} for all OK cameras,
    sorted by InstanceId (which matches DirectShow enumeration order).
    """
    import subprocess
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-PnpDevice -Class Camera -Status OK "
                "| Sort-Object InstanceId "
                "| Select-Object FriendlyName, InstanceId "
                "| ConvertTo-Csv -NoTypeInformation",
            ],
            capture_output=True, text=True, timeout=6,
        )
        lines = [l.strip().strip('"') for l in result.stdout.splitlines() if l.strip()]
        # lines[0] is the CSV header "FriendlyName","InstanceId"
        devices = []
        for line in lines[1:]:
            parts = [p.strip().strip('"') for p in line.split('","')]
            if len(parts) >= 2:
                devices.append({"name": parts[0], "instance_id": parts[1]})
        return devices
    except Exception:
        return []


def _is_internal(device: dict) -> bool:
    """Return True if the camera device appears to be a built-in/integrated camera."""
    iid_upper = device["instance_id"].upper()
    name_lower = device["name"].lower()

    # VID check (most reliable)
    for vid in _INTERNAL_VIDS:
        if vid in iid_upper:
            return True

    # Name keyword fallback
    for kw in _INTERNAL_NAME_KW:
        if kw in name_lower:
            return True

    return False


def _find_cameras() -> tuple[int, int | None]:
    """
    Identify which DirectShow index is the external USB cam (SVPRO) and which
    is the internal laptop webcam.

    Returns (external_idx, internal_idx).  internal_idx may be None if the
    built-in webcam is disabled or not found.

    Strategy: open each camera at 1280x720 MJPG, read 10 frames, and measure
    actual FPS.  The SVPRO delivers ~60 fps; the built-in webcam tops out at
    ~30 fps regardless of what it reports via CAP_PROP_FPS.
    """
    import time as _time

    # Log PnP info for debugging only
    devices = _query_pnp_cameras()
    if devices:
        print("[Camera] PnP camera list (info only):")
        for i, d in enumerate(devices):
            tag = "INTERNAL" if _is_internal(d) else "EXTERNAL"
            print(f"  [{i}] {tag}: {d['name']}  ({d['instance_id']})")

    # --- Measure actual FPS for each working index ---
    target_w   = config.CAMERA_WIDTH
    target_h   = config.CAMERA_HEIGHT
    target_fps = config.CAMERA_FPS
    results = []   # list of (idx, measured_fps)

    print(f"[Camera] Probing indices 0-7 at {target_w}x{target_h} MJPG ...")
    for idx in range(8):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  target_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        cap.set(cv2.CAP_PROP_FPS,          target_fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Warm up
        for _ in range(3):
            cap.read()
        # Time 10 frames
        t0 = _time.perf_counter()
        good = 0
        for _ in range(10):
            ret, _ = cap.read()
            if ret:
                good += 1
        elapsed = _time.perf_counter() - t0
        cap.release()
        if good >= 5:
            measured = good / elapsed if elapsed > 0 else 0
            results.append((idx, measured))
            print(f"  [{idx}] measured {measured:.1f} fps ({good} frames in {elapsed:.2f}s)")

    if not results:
        raise RuntimeError(
            "No cameras found. Plug in the USB camera, or set CAMERA_INDEX "
            "to an integer in config.py."
        )

    # --- Pick the SVPRO: highest actual FPS (should be ~60, webcam ~30) ---
    results.sort(key=lambda x: x[1], reverse=True)
    external_idx = results[0][0]
    internal_idx = results[1][0] if len(results) > 1 else None

    print(f"[Camera] Selected — Turret (external): index {external_idx} "
          f"({results[0][1]:.0f} fps), "
          f"Head tracking (internal): index {internal_idx}"
          + (f" ({results[1][1]:.0f} fps)" if internal_idx is not None else ""))
    return external_idx, internal_idx


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    global _serial, _camera, _web_server

    print("[Main] Pan-Tilt Tracker starting ...")

    # ── serial ──
    _serial = SerialComm()

    # ── resolve camera indices ──
    internal_cam_index = None
    if config.CAMERA_INDEX == "auto_external":
        try:
            cam_index, internal_cam_index = _find_cameras()
        except RuntimeError as exc:
            print(f"[Main] ERROR: {exc}")
            sys.exit(1)
    else:
        cam_index = int(config.CAMERA_INDEX)
        print(f"[Main] Using camera index {cam_index} (from config).")

    # ── open camera ──
    _camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    _camera.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    _camera.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)
    _camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Hardware auto-exposure (DirectShow: 0.75 = auto, 0.25 = manual)
    if config.CAMERA_AUTO_EXPOSURE:
        _camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        print("[Main] Hardware auto-exposure requested (driver may ignore).")

    if not _camera.isOpened():
        print(f"[Main] ERROR: Could not open camera index {cam_index}")
        sys.exit(1)

    actual_w   = int(_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = _camera.get(cv2.CAP_PROP_FPS)
    print(f"[Main] Camera opened (index {cam_index}): {actual_w}×{actual_h} @ {actual_fps:.0f}fps")

    # ── subsystems ──
    tracker    = Tracker()
    controller = MotionController()
    psl        = PSLAnalyzer()
    psl.start()
    _hand_head.set_camera_index(internal_cam_index, exclude=cam_index)
    _hand_head.start()

    fps_capture   = FPSCounter(window=60)
    fps_inference = FPSCounter(window=60)
    fps_loop      = FPSCounter(window=60)

    # ── Thread 3: Flask web server (start() creates its own daemon thread) ──
    _web_server = WebServer(shared, controller)
    _web_server.start()

    # ── Thread 1: Camera capture ──
    cap_thread = threading.Thread(
        target=_capture_thread,
        args=(_camera, fps_capture),
        daemon=True,
        name="Capture",
    )
    cap_thread.start()

    # ── Wait for first frame ──
    print("[Main] Waiting for first frame ...")
    for _ in range(200):
        with shared.frame_lock:
            if shared.latest_frame is not None:
                break
        time.sleep(0.05)
    else:
        print("[Main] ERROR: No frames received.")
        _shutdown("no frames")
        sys.exit(1)

    print(f"[Main] Ready.  Flask UI: http://localhost:{config.WEB_SERVER_PORT}")

    # ── Thread 2: Processing (background) ──
    proc_thread = threading.Thread(
        target=_processing_thread,
        args=(tracker, controller, _serial, fps_capture, fps_inference, fps_loop, psl),
        daemon=True,
        name="Processing",
    )
    proc_thread.start()

    # ── Thread 3: Annotation (background, off critical path) ──
    ann_thread = threading.Thread(
        target=_annotation_thread,
        args=(psl,),
        daemon=True,
        name="Annotation",
    )
    ann_thread.start()

    # ── Main thread: OpenCV display + trackbars + keyboard ──
    _display_loop(controller)


if __name__ == "__main__":
    main()
