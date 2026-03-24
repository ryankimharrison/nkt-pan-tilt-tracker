"""
tracker.py — YOLOv8 + ByteTrack multi-person tracker with target selection.

The Tracker owns its own YOLO model so detection and tracking happen
in a single model.track() call (persist=True keeps IDs stable across frames).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os
import numpy as np
import config

# Prefer a local bytetrack.yaml so we can tune track_buffer and other params
# without modifying the ultralytics package.
_LOCAL_BYTETRACK = os.path.join(os.path.dirname(__file__), "bytetrack.yaml")
_BYTETRACK_CFG   = _LOCAL_BYTETRACK if os.path.isfile(_LOCAL_BYTETRACK) else "bytetrack.yaml"


# COCO pose keypoint indices (0-based)
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_HIP       = 11
RIGHT_HIP      = 12
KEYPOINT_CONF_THRESH = 0.3
# Minimum average keypoint confidence to use keypoint-based aim point
# Below this, fall back to bbox center (more stable when joints are noisy)
KEYPOINT_AIM_MIN_CONF = 0.5


@dataclass
class Track:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    # Aim point: between shoulders when visible, else bbox center
    aim_cx: float = 0.0
    aim_cy: float = 0.0
    # Pose keypoints: list of (x, y, conf) for 17 COCO points, or None
    keypoints: list[tuple[float, float, float]] | None = None
    # Global person ID from ReID system (persistent across cameras/re-entry)
    global_id: int = -1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class TrackResult:
    tracks: list[Track] = field(default_factory=list)
    target: Track | None = None
    is_idle: bool = False
    released_lock: bool = False  # True when tracker gave up on locked target


class Tracker:
    """
    Wraps ultralytics YOLO.track() for combined detection + ByteTrack.

    Target selection: always one person when visible.
      - Keep current target if still in frame.
      - Otherwise pick by highest confidence, then closest to centre.
      - Idle after IDLE_TIMEOUT_FRAMES with no detections.
    """

    def __init__(self):
        from ultralytics import YOLO
        print(f"[Tracker] Loading model: {config.YOLO_MODEL_PATH}")
        self._model = YOLO(config.YOLO_MODEL_PATH)
        print("[Tracker] Model loaded.")

        self._target_id:      int | None = None
        self._lost_counter:   int = 0
        self._idle_counter:   int = 0
        self._no_det_counter: int = 0

        # Manual slot lock: None = auto-select; int = left-to-right slot index
        self._locked_slot:    int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        frame_shape: tuple[int, int],
        cycle_target_event=None,
    ) -> TrackResult:
        """Run detection + tracking on a BGR frame.

        cycle_target_event: optional threading.Event that, when set, advances
        the locked slot to the next person (left-to-right ordering).
        """
        h, w = frame_shape

        # Use lower conf for YOLO to let ByteTrack see marginal detections.
        # Filter in _parse_results: tracked targets keep at lower threshold (hysteresis).
        _keep_conf = config.CONFIDENCE_THRESHOLD * 0.6  # ~0.24 — keep threshold
        results = self._model.track(
            frame,
            classes=[0],                         # person only
            conf=_keep_conf,                     # low threshold — ByteTrack handles persistence
            persist=True,                        # maintains IDs across frames
            tracker=_BYTETRACK_CFG,
            imgsz=config.CAMERA_WIDTH,           # run inference at full camera resolution
            verbose=False,
        )

        tracks = self._parse_results(results, tracked_id=self._target_id)

        if not tracks:
            self._no_det_counter += 1
        else:
            self._no_det_counter = 0

        is_idle = self._no_det_counter >= config.IDLE_TIMEOUT_FRAMES

        # Handle Tab cycle before target selection so the new slot is used this frame
        if cycle_target_event is not None and cycle_target_event.is_set():
            cycle_target_event.clear()
            self._cycle_target(tracks)

        target, released = self._select_target(tracks, w, h)
        return TrackResult(tracks=tracks, target=target, is_idle=is_idle, released_lock=released)

    def reset_target(self):
        """Force re-acquisition on next frame."""
        self._target_id    = None
        self._locked_slot  = None
        self._lost_counter = 0

    def unlock_target(self):
        """Release manual lock and return to auto-selection."""
        self._locked_slot = None
        self._target_id   = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_results(self, results, tracked_id: int | None = None) -> list[Track]:
        tracks: list[Track] = []
        _acquire_conf = config.CONFIDENCE_THRESHOLD        # higher to acquire new target
        _keep_conf    = config.CONFIDENCE_THRESHOLD * 0.6  # lower to keep existing target
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes
            if boxes.id is None:
                continue
            kpts = getattr(r, "keypoints", None)
            for i in range(len(boxes)):
                tid  = int(boxes.id[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                # Hysteresis: already-tracked target uses lower threshold
                _min_conf = _keep_conf if (tracked_id is not None and tid == tracked_id) else _acquire_conf
                if conf < _min_conf:
                    continue
                # Default: bbox center (always stable)
                aim_cx = (x1 + x2) / 2.0
                aim_cy = (y1 + y2) / 2.0

                # Only use keypoint-based aim if enough keypoints are confident
                if kpts is not None and kpts.conf is not None and i < len(kpts.conf):
                    _kp_confs = kpts.conf[i]
                    # Average confidence of torso keypoints (shoulders + hips)
                    _torso_idxs = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
                    _torso_confs = [float(_kp_confs[j]) for j in _torso_idxs]
                    _avg_torso_conf = sum(_torso_confs) / len(_torso_confs)

                    if _avg_torso_conf >= KEYPOINT_AIM_MIN_CONF:
                        _kp_aim = self._shoulder_midpoint(kpts, i)
                        if _kp_aim[0] is not None:
                            aim_cx, aim_cy = _kp_aim
                kp_list = self._extract_keypoints(kpts, i) if kpts is not None else None
                tracks.append(Track(tid, x1, y1, x2, y2, conf, aim_cx=aim_cx, aim_cy=aim_cy, keypoints=kp_list))
        return tracks

    def _extract_keypoints(self, keypoints, idx: int) -> list[tuple[float, float, float]] | None:
        """Extract (x, y, conf) for each of 17 COCO keypoints."""
        if keypoints is None or keypoints.xy is None or idx >= len(keypoints.xy):
            return None
        xy = keypoints.xy[idx]
        conf = keypoints.conf[idx] if keypoints.conf is not None else None
        kp_list: list[tuple[float, float, float]] = []
        for j in range(17):
            x, y = float(xy[j][0]), float(xy[j][1])
            c = float(conf[j]) if conf is not None else 1.0
            kp_list.append((x, y, c))
        return kp_list

    def _shoulder_midpoint(self, keypoints, idx: int) -> tuple[float | None, float | None]:
        """Return (cx, cy) at torso center (between shoulders and hips).
        Falls back to shoulder midpoint if hips not visible."""
        if keypoints is None or keypoints.xy is None or idx >= len(keypoints.xy):
            return (None, None)
        xy = keypoints.xy[idx]  # (17, 2)
        conf = keypoints.conf[idx] if keypoints.conf is not None else None
        ls = xy[LEFT_SHOULDER]
        rs = xy[RIGHT_SHOULDER]
        lh = xy[LEFT_HIP]
        rh = xy[RIGHT_HIP]
        if conf is None:
            return (None, None)

        # Check which keypoints are visible
        ls_ok = conf[LEFT_SHOULDER] >= KEYPOINT_CONF_THRESH
        rs_ok = conf[RIGHT_SHOULDER] >= KEYPOINT_CONF_THRESH
        lh_ok = conf[LEFT_HIP] >= KEYPOINT_CONF_THRESH
        rh_ok = conf[RIGHT_HIP] >= KEYPOINT_CONF_THRESH

        if not ls_ok and not rs_ok:
            return (None, None)

        # Shoulder center
        if ls_ok and rs_ok:
            sx = (float(ls[0]) + float(rs[0])) / 2.0
            sy = (float(ls[1]) + float(rs[1])) / 2.0
        elif ls_ok:
            sx, sy = float(ls[0]), float(ls[1])
        else:
            sx, sy = float(rs[0]), float(rs[1])

        # Hip center (if visible)
        if lh_ok and rh_ok:
            hx = (float(lh[0]) + float(rh[0])) / 2.0
            hy = (float(lh[1]) + float(rh[1])) / 2.0
        elif lh_ok:
            hx, hy = float(lh[0]), float(lh[1])
        elif rh_ok:
            hx, hy = float(rh[0]), float(rh[1])
        else:
            # No hips visible — use shoulder midpoint only
            return (sx, sy)

        # Torso center = midpoint between shoulder center and hip center
        return ((sx + hx) / 2.0, (sy + hy) / 2.0)
        # No conf — assume visible if non-zero
        return ((float(ls[0]) + float(rs[0])) / 2.0, (float(ls[1]) + float(rs[1])) / 2.0)

    @staticmethod
    def _ordered_tracks(tracks: list[Track]) -> list[Track]:
        """Return tracks sorted left-to-right by aim_cx (stable slot ordering)."""
        return sorted(tracks, key=lambda t: t.aim_cx)

    def _cycle_target(self, tracks: list[Track]) -> None:
        """Advance _locked_slot to the next position, wrapping around.

        Called when the Tab key event fires.  If no one is visible yet,
        do nothing — there's nothing to cycle to.
        """
        if not tracks:
            return
        n = len(tracks)
        if self._locked_slot is None:
            self._locked_slot = 0
        else:
            self._locked_slot = (self._locked_slot + 1) % n
        # Reset auto-tracking state so the new selection takes effect cleanly
        self._target_id   = None
        self._lost_counter = 0
        ordered = self._ordered_tracks(tracks)
        print(f"[Tracker] Slot {self._locked_slot} locked → track {ordered[self._locked_slot].track_id}")

    def _select_target(
        self, tracks: list[Track], w: int, h: int
    ) -> tuple[Track | None, bool]:
        """Return (target, released_lock).

        When _locked_slot is set, track the person at that left-to-right slot.
        When no lock, auto-select closest to center (most stable default).
        """
        if not tracks:
            self._lost_counter += 1
            if self._locked_slot is not None:
                # Locked person has left the frame — wait before giving up
                if self._lost_counter >= config.TARGET_LOST_FRAMES:
                    print("[Tracker] Locked target lost — reverting to auto.")
                    self._locked_slot = None
                    self._target_id   = None
                    return (None, True)
                return (None, False)
            if self._lost_counter >= config.TARGET_LOST_FRAMES:
                self._target_id = None
            return (None, False)

        self._lost_counter = 0
        ordered = self._ordered_tracks(tracks)

        # ── Manual slot lock ──
        if self._locked_slot is not None:
            # Clamp slot if people left the frame since Tab was pressed
            slot = min(self._locked_slot, len(ordered) - 1)
            target = ordered[slot]
            self._target_id = target.track_id
            return (target, False)

        # ── Auto-select: closest to frame center, then highest confidence ──
        cx, cy = w / 2.0, h / 2.0
        best = min(tracks, key=lambda t: (
            (t.aim_cx - cx) ** 2 + (t.aim_cy - cy) ** 2,
            -t.conf,
        ))
        self._target_id = best.track_id
        return (best, False)
