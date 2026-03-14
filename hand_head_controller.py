"""
hand_head_controller.py — Control tracker via hand position or head direction.

Runs MediaPipe hand/face detection on the built-in laptop webcam in a
background daemon thread.  The processing thread polls get_result() each
frame and uses it as jog velocity when hand or head mode is active.

Opens a separate OpenCV preview window showing the webcam feed with
landmarks and trackbars for tuning speed, deadzone, and smoothing.

Modes:
  "off"  — detector idle, camera closed, window hidden
  "hand" — track palm center, map to jog velocity
  "head" — track head yaw/pitch, map to jog velocity
"""

from __future__ import annotations

import os
import threading
import time
import math
import cv2
import numpy as np

try:
    import mediapipe as mp
    _vision = mp.tasks.vision
    _BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = _vision.HandLandmarker
    HandLandmarkerOptions = _vision.HandLandmarkerOptions
    HandLandmarkerResult = _vision.HandLandmarkerResult
    FaceLandmarker = _vision.FaceLandmarker
    FaceLandmarkerOptions = _vision.FaceLandmarkerOptions
    FaceLandmarkerResult = _vision.FaceLandmarkerResult
    RunningMode = _vision.RunningMode
    _MP_AVAILABLE = True
except (ImportError, AttributeError):
    _MP_AVAILABLE = False

import config

# Model file paths (next to this script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HAND_MODEL = os.path.join(_SCRIPT_DIR, "hand_landmarker.task")
_FACE_MODEL = os.path.join(_SCRIPT_DIR, "face_landmarker.task")

_WIN_NAME = "Hand/Head Control"


class HandHeadController:
    """Background hand/head detector on the built-in webcam."""

    def __init__(self, exclude_index: int | None = None):
        self._mode = "off"          # "off" | "hand" | "head"
        self._lock = threading.Lock()
        self._result: tuple[float, float] | None = None  # (pan_norm, tilt_norm) in [-1, 1]
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cap: cv2.VideoCapture | None = None
        self._exclude_index = exclude_index  # tracker camera index to avoid
        self._cam_index: int | None = None   # specific internal cam index (set by main)
        self._window_created = False

        # EMA state
        self._ema_x = 0.0
        self._ema_y = 0.0

        # MediaPipe detectors (lazy init in thread)
        self._hand_landmarker: HandLandmarker | None = None
        self._face_landmarker: FaceLandmarker | None = None

        # Latest detection results (set by callbacks)
        self._hand_result: HandLandmarkerResult | None = None
        self._face_result: FaceLandmarkerResult | None = None
        self._hand_result_lock = threading.Lock()
        self._face_result_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_camera_index(self, cam_idx: int | None, exclude: int | None = None):
        """Set the internal webcam index directly, and the turret index to avoid."""
        self._cam_index = cam_idx
        self._exclude_index = exclude

    def start(self):
        if not _MP_AVAILABLE:
            print("[HandHead] mediapipe not installed — hand/head control disabled.")
            return
        if not os.path.isfile(_HAND_MODEL):
            print(f"[HandHead] Missing model: {_HAND_MODEL} — hand control disabled.")
            return
        if not os.path.isfile(_FACE_MODEL):
            print(f"[HandHead] Missing model: {_FACE_MODEL} — head control disabled.")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="HandHeadCtrl")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._close_camera()
        self._destroy_window()

    def set_mode(self, mode: str):
        with self._lock:
            self._mode = mode
            self._result = None
            self._ema_x = 0.0
            self._ema_y = 0.0

    def get_mode(self) -> str:
        with self._lock:
            return self._mode

    def get_result(self) -> tuple[float, float] | None:
        with self._lock:
            return self._result

    # ------------------------------------------------------------------
    # Trackbar callbacks (called from OpenCV GUI thread = this thread)
    # ------------------------------------------------------------------

    @staticmethod
    def _on_pan_range(val):
        config.HAND_CONTROL_PAN_RANGE = max(5, val)

    @staticmethod
    def _on_tilt_range(val):
        config.HAND_CONTROL_TILT_RANGE = max(5, val)

    @staticmethod
    def _on_kp(val):
        config.HAND_CONTROL_KP = max(1, val) / 10.0  # slider 10-200 → 1.0-20.0

    @staticmethod
    def _on_max_vel(val):
        config.HAND_CONTROL_MAX_VEL = max(10, val)

    @staticmethod
    def _on_deadzone(val):
        config.HAND_CONTROL_DEADZONE = val / 100.0  # slider 0-50 → 0.00-0.50

    @staticmethod
    def _on_ema(val):
        config.HAND_CONTROL_EMA = max(1, val) / 100.0  # slider 1-100 → 0.01-1.00

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self):
        print("[HandHead] Thread started.")
        interval = 1.0 / max(1, config.HAND_CONTROL_FPS)
        frame_ts = 0  # monotonic timestamp counter for mediapipe

        while not self._stop.is_set():
            mode = self.get_mode()

            if mode == "off":
                self._close_camera()
                self._destroy_window()
                with self._lock:
                    self._result = None
                time.sleep(0.1)
                continue

            # Ensure camera is open
            if self._cap is None or not self._cap.isOpened():
                if not self._open_camera():
                    time.sleep(1.0)
                    continue

            # Ensure detector is initialized
            if mode == "hand" and self._hand_landmarker is None:
                self._init_hands()
            elif mode == "head" and self._face_landmarker is None:
                self._init_face()

            # Ensure window exists
            if not self._window_created:
                self._create_window()

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Flip horizontally for mirror view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts += 33  # ~30fps timestamp in ms

            raw_x, raw_y = None, None

            if mode == "hand" and self._hand_landmarker is not None:
                self._hand_landmarker.detect_async(mp_image, frame_ts)
                raw_x, raw_y, frame = self._process_hand(frame)
            elif mode == "head" and self._face_landmarker is not None:
                self._face_landmarker.detect_async(mp_image, frame_ts)
                raw_x, raw_y, frame = self._process_head(frame)

            if raw_x is not None and raw_y is not None:
                # Apply deadzone
                dz = config.HAND_CONTROL_DEADZONE
                if abs(raw_x) < dz:
                    raw_x = 0.0
                if abs(raw_y) < dz:
                    raw_y = 0.0

                # EMA smoothing
                alpha = config.HAND_CONTROL_EMA
                self._ema_x = alpha * raw_x + (1 - alpha) * self._ema_x
                self._ema_y = alpha * raw_y + (1 - alpha) * self._ema_y

                with self._lock:
                    self._result = (self._ema_x, self._ema_y)
            else:
                # No detection — decay to zero
                self._ema_x *= 0.8
                self._ema_y *= 0.8
                with self._lock:
                    if abs(self._ema_x) < 0.01 and abs(self._ema_y) < 0.01:
                        self._result = None
                    else:
                        self._result = (self._ema_x, self._ema_y)

            # Draw HUD on preview frame
            self._draw_hud(frame, mode, raw_x, raw_y)

            # Show preview
            cv2.imshow(_WIN_NAME, frame)
            cv2.waitKey(1)

            time.sleep(interval)

        self._close_camera()
        self._release_detectors()
        self._destroy_window()
        print("[HandHead] Thread stopped.")

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _open_camera(self) -> bool:
        # Use the specific internal cam index if set by main, otherwise scan
        indices_to_try = []
        if self._cam_index is not None:
            indices_to_try.append(self._cam_index)
        indices_to_try.append(config.HAND_CONTROL_CAM_INDEX)
        for i in range(8):
            if i not in indices_to_try:
                indices_to_try.append(i)

        for idx in indices_to_try:
            if idx == self._exclude_index:
                continue  # never use the turret camera
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            ret, _ = cap.read()
            if ret:
                self._cap = cap
                print(f"[HandHead] Opened internal webcam at index {idx} (640x480).")
                return True
            cap.release()

        print("[HandHead] Could not open any camera for hand/head control.")
        return False

    def _close_camera(self):
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Window
    # ------------------------------------------------------------------

    def _create_window(self):
        cv2.namedWindow(_WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Pan Range", _WIN_NAME,
                           int(config.HAND_CONTROL_PAN_RANGE), 180, self._on_pan_range)
        cv2.createTrackbar("Tilt Range", _WIN_NAME,
                           int(config.HAND_CONTROL_TILT_RANGE), 90, self._on_tilt_range)
        cv2.createTrackbar("Kp x10", _WIN_NAME,
                           int(config.HAND_CONTROL_KP * 10), 200, self._on_kp)
        cv2.createTrackbar("Max Vel", _WIN_NAME,
                           int(config.HAND_CONTROL_MAX_VEL), 500, self._on_max_vel)
        cv2.createTrackbar("Deadzone%", _WIN_NAME,
                           int(config.HAND_CONTROL_DEADZONE * 100), 50, self._on_deadzone)
        cv2.createTrackbar("Smooth%", _WIN_NAME,
                           int(config.HAND_CONTROL_EMA * 100), 100, self._on_ema)
        self._window_created = True
        print("[HandHead] Preview window created.")

    def _destroy_window(self):
        if self._window_created:
            try:
                cv2.destroyWindow(_WIN_NAME)
            except Exception:
                pass
            self._window_created = False

    # ------------------------------------------------------------------
    # MediaPipe init (new Tasks API)
    # ------------------------------------------------------------------

    def _hand_result_callback(self, result: HandLandmarkerResult, image: mp.Image, timestamp_ms: int):
        with self._hand_result_lock:
            self._hand_result = result

    def _face_result_callback(self, result: FaceLandmarkerResult, image: mp.Image, timestamp_ms: int):
        with self._face_result_lock:
            self._face_result = result

    def _init_hands(self):
        options = HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=_HAND_MODEL),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._hand_result_callback,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(options)
        print("[HandHead] Hand landmarker initialized (Tasks API).")

    def _init_face(self):
        options = FaceLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=_FACE_MODEL),
            running_mode=RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._face_result_callback,
        )
        self._face_landmarker = FaceLandmarker.create_from_options(options)
        print("[HandHead] Face landmarker initialized (Tasks API).")

    def _release_detectors(self):
        if self._hand_landmarker is not None:
            self._hand_landmarker.close()
            self._hand_landmarker = None
        if self._face_landmarker is not None:
            self._face_landmarker.close()
            self._face_landmarker = None

    # ------------------------------------------------------------------
    # Detection + drawing
    # ------------------------------------------------------------------

    def _process_hand(self, frame: np.ndarray) -> tuple[float | None, float | None, np.ndarray]:
        """Get latest hand result, draw landmarks, return normalized palm offset."""
        with self._hand_result_lock:
            result = self._hand_result

        if result is None or not result.hand_landmarks:
            return None, None, frame

        hand = result.hand_landmarks[0]
        h, w = frame.shape[:2]

        # Draw all hand landmarks + connections
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # Draw connections
        _HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),     # thumb
            (0,5),(5,6),(6,7),(7,8),     # index
            (5,9),(9,10),(10,11),(11,12), # middle
            (9,13),(13,14),(14,15),(15,16), # ring
            (13,17),(17,18),(18,19),(19,20), # pinky
            (0,17),
        ]
        for a, b in _HAND_CONNECTIONS:
            if a < len(hand) and b < len(hand):
                p1 = (int(hand[a].x * w), int(hand[a].y * h))
                p2 = (int(hand[b].x * w), int(hand[b].y * h))
                cv2.line(frame, p1, p2, (0, 200, 0), 2)

        # Landmark 9 = middle finger MCP (palm center)
        palm = hand[9]
        palm_px = (int(palm.x * w), int(palm.y * h))
        cv2.circle(frame, palm_px, 10, (0, 0, 255), 3)

        # Normalize to [-1, 1] — center of frame = (0, 0)
        # Note: frame is already flipped, so x is mirrored
        x_norm = (palm.x - 0.5) * 2.0
        y_norm = (palm.y - 0.5) * 2.0
        return x_norm, y_norm, frame

    def _process_head(self, frame: np.ndarray) -> tuple[float | None, float | None, np.ndarray]:
        """Get latest face result, draw key landmarks, return normalized yaw/pitch."""
        with self._face_result_lock:
            result = self._face_result

        if result is None or not result.face_landmarks:
            return None, None, frame

        face = result.face_landmarks[0]
        h, w = frame.shape[:2]

        # Draw face oval (subset of landmarks for a clean look)
        _FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                       323, 361, 288, 397, 365, 379, 378, 400, 377,
                       152, 148, 176, 149, 150, 136, 172, 58, 132,
                       93, 234, 127, 162, 21, 54, 103, 67, 109]
        for i in range(len(_FACE_OVAL)):
            a = _FACE_OVAL[i]
            b = _FACE_OVAL[(i + 1) % len(_FACE_OVAL)]
            if a < len(face) and b < len(face):
                p1 = (int(face[a].x * w), int(face[a].y * h))
                p2 = (int(face[b].x * w), int(face[b].y * h))
                cv2.line(frame, p1, p2, (0, 200, 200), 1)

        # Key landmarks
        nose     = face[1]
        l_cheek  = face[234]
        r_cheek  = face[454]
        forehead = face[10]
        chin     = face[152]

        # Draw key points
        for lm, color in [(nose, (0,0,255)), (l_cheek, (255,0,0)),
                           (r_cheek, (255,0,0)), (forehead, (0,255,255)),
                           (chin, (0,255,255))]:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, color, -1)

        # Yaw: nose position relative to cheek midpoint (works great for L/R)
        cheek_mid_x = (l_cheek.x + r_cheek.x) / 2.0
        cheek_span = abs(r_cheek.x - l_cheek.x)
        if cheek_span < 0.01:
            return None, None, frame
        yaw = (nose.x - cheek_mid_x) / (cheek_span * 0.5)
        yaw = max(-1.0, min(1.0, yaw * 2.0))

        # Pitch: face center Y position in frame (0.0 = top, 1.0 = bottom)
        # When you look up your face rises in frame; when you look down it drops.
        # Much stronger signal than nose-relative-to-chin which barely changes.
        face_center_y = (forehead.y + chin.y) / 2.0
        # Map so 0.5 (center of frame) = 0, edges = ±1
        pitch = (face_center_y - 0.5) * 2.0
        pitch = max(-1.0, min(1.0, pitch))

        # Draw direction arrow from nose
        nose_px = (int(nose.x * w), int(nose.y * h))
        arrow_end = (int(nose_px[0] + yaw * 60), int(nose_px[1] + pitch * 60))
        cv2.arrowedLine(frame, nose_px, arrow_end, (0, 255, 0), 3, tipLength=0.3)

        return yaw, pitch, frame

    # ------------------------------------------------------------------
    # HUD overlay
    # ------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, mode: str, raw_x, raw_y):
        h, w = frame.shape[:2]

        # Mode label
        label = f"Mode: {mode.upper()}"
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        # Crosshair at center
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

        # Deadzone circle
        dz_px = int(config.HAND_CONTROL_DEADZONE * min(w, h))
        cv2.circle(frame, (cx, cy), dz_px, (100, 100, 100), 1)

        # Output values
        with self._lock:
            res = self._result
        if res is not None:
            target_pan  = res[0] * config.HAND_CONTROL_PAN_RANGE
            target_tilt = res[1] * config.HAND_CONTROL_TILT_RANGE
            cv2.putText(frame, f"Pan: {target_pan:+.0f} deg", (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Tilt: {target_tilt:+.0f} deg", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Direction arrow from center
            arrow_x = int(cx + res[0] * 80)
            arrow_y = int(cy + res[1] * 80)
            cv2.arrowedLine(frame, (cx, cy), (arrow_x, arrow_y),
                            (0, 200, 255), 2, tipLength=0.3)
        else:
            cv2.putText(frame, "No detection", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
