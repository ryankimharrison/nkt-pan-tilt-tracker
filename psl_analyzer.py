"""
psl_analyzer.py — Isolated PSL (attractiveness metric) analyzer.

Runs in its own daemon thread. Accepts raw BGR frames via push_frame(),
processes them with MediaPipe FaceMesh, and exposes results via get_result().
No imports from the rest of this project — completely standalone.

Usage
-----
    analyzer = PSLAnalyzer()
    analyzer.start()

    # each frame (from any thread):
    analyzer.push_frame(bgr_frame)

    # in display loop:
    draw_psl_overlay(frame, analyzer.get_result())
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# Face-landmarker model (478 points, float16, ~1 MB).
# Downloaded automatically on first run alongside this script.
_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "face_landmarker.task")


def _ensure_model() -> str:
    """Download the face-landmarker model if it isn't already on disk."""
    if not os.path.exists(_MODEL_PATH):
        print(f"[PSL] Downloading face-landmarker model to {_MODEL_PATH} …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[PSL] Model downloaded.")
    return _MODEL_PATH


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PSLResult:
    """All computed metrics for a single frame."""
    psl_score:     Optional[float] = None   # 1.0 – 10.0 composite
    symmetry:      Optional[float] = None   # 0 – 100 %
    canthal_tilt:  Optional[float] = None   # degrees, + = outer corner higher
    eye_spacing:   Optional[float] = None   # IPD / face_width  (ideal ≈ 0.46)
    fwhr:          Optional[float] = None   # face width / mid-face height (ideal ≈ 2.0)
    thirds:        Optional[tuple] = None   # (upper%, mid%, lower%)  ideal ≈ 33/33/33
    jawline_score: Optional[float] = None   # 0 – 100
    landmarks:     Optional[np.ndarray] = field(default=None, repr=False)
    age:           float = 0.0              # seconds since last update


# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe FaceMesh 468-point model)
# ---------------------------------------------------------------------------

_CHIN           = 152
_FOREHEAD       = 10
_L_CHEEK        = 234
_R_CHEEK        = 454
_RE_OUTER       = 33     # right eye lateral canthus
_RE_INNER       = 133    # right eye medial canthus
_LE_INNER       = 362    # left  eye medial canthus
_LE_OUTER       = 263    # left  eye lateral canthus
_RE_TOP         = 159
_RE_BOT         = 145
_LE_TOP         = 386
_LE_BOT         = 374
_R_BROW         = 70
_L_BROW         = 300
_NOSE_TIP       = 4
_UPPER_LIP      = 13
_LOWER_LIP      = 14
_R_MOUTH        = 61
_L_MOUTH        = 291
_R_JAW          = 172
_L_JAW          = 397

_MESH_DRAW_IDS = [
    _RE_OUTER, _RE_INNER, _LE_INNER, _LE_OUTER,
    _NOSE_TIP, _CHIN, _L_CHEEK, _R_CHEEK,
    _R_MOUTH, _L_MOUTH, _R_BROW, _L_BROW,
    _R_JAW, _L_JAW, _FOREHEAD,
]


# ---------------------------------------------------------------------------
# Frontality gate + temporal smoother
# ---------------------------------------------------------------------------

# Only accept measurements taken within this many degrees of straight-on.
# Nose-to-cheek ratio method: 0 = perfectly frontal, 1 = fully sideways.
# 0.25 ≈ ~30° yaw — reliable region for symmetric metrics.
_MAX_TURN = 0.25

# EMA alpha for blending new accepted measurements into the stable result.
# Low value = slow, stable display.  0.12 ≈ ~8-frame lag at 4 fps update rate.
_SMOOTH_ALPHA = 0.12


def _frontality_ok(pts: np.ndarray) -> bool:
    """Return True if the face is close enough to frontal for reliable metrics."""
    nose_x = float(pts[_NOSE_TIP][0])
    l_x    = float(pts[_L_CHEEK][0])
    r_x    = float(pts[_R_CHEEK][0])
    face_w = abs(l_x - r_x)
    if face_w < 1.0:
        return False
    # Nose should sit near the mid-point of the two cheekbones.
    ratio = (nose_x - min(l_x, r_x)) / face_w   # 0.5 = perfectly frontal
    turn  = abs(ratio - 0.5) * 2.0               # 0 = frontal, 1 = sideways
    return turn < _MAX_TURN


def _smooth_result(old: "PSLResult", new: "PSLResult", alpha: float) -> "PSLResult":
    """EMA-blend *new* into *old*. Landmarks always come from *new* for drawing."""
    def _b(o, n):
        if o is None: return n
        if n is None: return o
        return o * (1.0 - alpha) + n * alpha

    r = PSLResult()
    r.landmarks     = new.landmarks          # always fresh for drawing
    r.age           = new.age
    r.symmetry      = _b(old.symmetry,      new.symmetry)
    r.canthal_tilt  = _b(old.canthal_tilt,  new.canthal_tilt)
    r.eye_spacing   = _b(old.eye_spacing,   new.eye_spacing)
    r.fwhr          = _b(old.fwhr,          new.fwhr)
    r.jawline_score = _b(old.jawline_score, new.jawline_score)
    if old.thirds and new.thirds:
        r.thirds = tuple(_b(o, n) for o, n in zip(old.thirds, new.thirds))
    else:
        r.thirds = new.thirds if new.thirds else old.thirds
    r.psl_score = _b(old.psl_score, new.psl_score)
    return r


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _calc_symmetry(pts: np.ndarray) -> float:
    """Bilateral symmetry score 0–100. Compares left/right landmark distances
    from the vertical midline through the nose."""
    mid_x = float(pts[_NOSE_TIP][0])
    pairs = [
        (_L_CHEEK,  _R_CHEEK),
        (_RE_OUTER, _LE_OUTER),
        (_RE_INNER, _LE_INNER),
        (_R_BROW,   _L_BROW),
        (_R_JAW,    _L_JAW),
        (_R_MOUTH,  _L_MOUTH),
    ]
    diffs = []
    for l, r in pairs:
        dl = abs(float(pts[l][0]) - mid_x)
        dr = abs(float(pts[r][0]) - mid_x)
        mx = max(dl, dr)
        if mx > 0:
            diffs.append(abs(dl - dr) / mx)
    if not diffs:
        return 50.0
    score = (1.0 - min(float(np.mean(diffs)) * 4.0, 1.0)) * 100.0
    return float(np.clip(score, 0.0, 100.0))


def _calc_canthal_tilt(pts: np.ndarray) -> float:
    """Canthal tilt in degrees. Positive = outer eye corner higher (attractive)."""
    # Right eye: inner canthus to outer canthus
    ri, ro = pts[_RE_INNER], pts[_RE_OUTER]
    r_angle = math.degrees(math.atan2(float(ri[1] - ro[1]), float(ro[0] - ri[0])))

    # Left eye: mirror so sign convention matches
    li, lo = pts[_LE_INNER], pts[_LE_OUTER]
    l_angle = math.degrees(math.atan2(float(li[1] - lo[1]), float(li[0] - lo[0])))

    return (r_angle + l_angle) / 2.0


def _calc_eye_spacing(pts: np.ndarray) -> float:
    """Inter-pupillary distance / face width.  Ideal ≈ 0.46."""
    face_w = _dist(pts[_L_CHEEK], pts[_R_CHEEK])
    if face_w < 1.0:
        return 0.46
    r_pupil = (pts[_RE_OUTER] + pts[_RE_INNER]) / 2.0
    l_pupil = (pts[_LE_OUTER] + pts[_LE_INNER]) / 2.0
    ipd = _dist(r_pupil, l_pupil)
    return float(np.clip(ipd / face_w, 0.0, 1.0))


def _calc_fwhr(pts: np.ndarray) -> float:
    """Facial width-height ratio (cheekbone width / brow-to-upper-lip height).
    Higher = wider face.  Research suggests ~1.9–2.1 for attractive males."""
    face_w = _dist(pts[_L_CHEEK], pts[_R_CHEEK])
    brow_mid_y = (float(pts[_R_BROW][1]) + float(pts[_L_BROW][1])) / 2.0
    lip_y      = float(pts[_UPPER_LIP][1])
    face_h_mid = abs(lip_y - brow_mid_y)
    if face_h_mid < 1.0:
        return 2.0
    return float(np.clip(face_w / face_h_mid, 0.5, 4.0))


def _calc_thirds(pts: np.ndarray) -> tuple[float, float, float]:
    """Facial thirds as percentages (upper, mid, lower). Ideal ≈ 33/33/33."""
    top   = float(pts[_FOREHEAD][1])
    brow  = (float(pts[_R_BROW][1]) + float(pts[_L_BROW][1])) / 2.0
    nose  = float(pts[_NOSE_TIP][1])
    chin  = float(pts[_CHIN][1])
    total = chin - top
    if total <= 0:
        return (33.3, 33.3, 33.3)
    upper = (brow - top)  / total * 100.0
    mid   = (nose - brow) / total * 100.0
    lower = (chin - nose) / total * 100.0
    return (upper, mid, lower)


def _calc_jawline(pts: np.ndarray) -> float:
    """Jawline sharpness proxy 0–100 based on gonial angle and jaw-to-chin ratio."""
    # Gonial angle: angle at jaw corner between chin and cheek
    chin   = pts[_CHIN]
    r_jaw  = pts[_R_JAW]
    l_jaw  = pts[_L_JAW]
    r_cheek = pts[_R_CHEEK]
    l_cheek = pts[_L_CHEEK]

    def _angle_at(vertex, a, b):
        va = a - vertex
        vb = b - vertex
        cos_t = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
        return math.degrees(math.acos(float(np.clip(cos_t, -1, 1))))

    r_angle = _angle_at(r_jaw, chin, r_cheek)
    l_angle = _angle_at(l_jaw, chin, l_cheek)
    avg_angle = (r_angle + l_angle) / 2.0
    # Sharper jaw ≈ smaller angle.  Ideal ~120–130°.
    score = float(np.clip((155.0 - avg_angle) / 35.0 * 100.0, 0.0, 100.0))
    return score


def _calc_psl_score(r: PSLResult) -> float:
    """Composite PSL score 1–10 from individual metrics."""
    score = 5.0

    if r.symmetry is not None:
        # 90% → +2, 70% → 0, 50% → -2
        score += float(np.clip((r.symmetry - 70.0) / 10.0, -2.0, 2.0))

    if r.canthal_tilt is not None:
        # +5° → +1.5, 0° → 0, −3° → −1
        score += float(np.clip(r.canthal_tilt / 3.5, -1.5, 1.5))

    if r.eye_spacing is not None:
        # Ideal 0.46; penalty for deviation
        score += float(np.clip(0.5 - abs(r.eye_spacing - 0.46) * 8.0, -1.0, 0.5))

    if r.fwhr is not None:
        # Ideal 2.0; score peaks there
        score += float(np.clip(0.5 - abs(r.fwhr - 2.0) * 1.5, -0.5, 0.5))

    if r.thirds is not None:
        upper, mid, lower = r.thirds
        dev = (abs(upper - 33.3) + abs(mid - 33.3) + abs(lower - 33.3)) / 3.0
        score += float(np.clip(0.5 - dev / 12.0, -0.5, 0.5))

    if r.jawline_score is not None:
        score += float(np.clip((r.jawline_score - 50.0) / 50.0, -1.0, 1.0))

    return float(np.clip(score, 1.0, 10.0))


# ---------------------------------------------------------------------------
# Analyzer thread
# ---------------------------------------------------------------------------

class PSLAnalyzer:
    """
    Completely isolated face-metric analyzer.

    Start it, push frames in, read results out.
    No coupling to the rest of the project.
    """

    # Max frames-per-second the analyzer will actually process.
    # Keeps the PSL thread from competing with YOLO for CPU cores.
    ANALYSIS_FPS = 4.0

    def __init__(self, max_queue: int = 1):
        self._q:          queue.Queue    = queue.Queue(maxsize=max_queue)
        self._result:     PSLResult      = PSLResult()
        self._lock:       threading.Lock = threading.Lock()
        self._stop:       threading.Event = threading.Event()
        self._thread:     Optional[threading.Thread] = None
        self._last_push:  float = 0.0   # monotonic time of last accepted push

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not _MP_AVAILABLE:
            print("[PSL] mediapipe not installed. Run:  pip install mediapipe")
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="psl-analyzer")
        self._thread.start()
        print("[PSL] Analyzer started.")

    def stop(self) -> None:
        self._stop.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def push_frame(self, frame: np.ndarray) -> None:
        """Non-blocking, rate-limited frame push.

        Rate-limits to ANALYSIS_FPS so the PSL thread never competes heavily
        with YOLO.  The copy is only made when the frame will actually be used.
        """
        if not self.is_running():
            return
        now = time.monotonic()
        if (now - self._last_push) < 1.0 / self.ANALYSIS_FPS:
            return                          # too soon — drop this frame cheaply
        if self._q.full():
            return                          # PSL still busy — drop without copying
        try:
            # np.ascontiguousarray ensures cv2.cvtColor won't fail on buffer views
            self._q.put_nowait(np.ascontiguousarray(frame))
            self._last_push = now
        except queue.Full:
            pass

    def get_result(self) -> PSLResult:
        with self._lock:
            return self._result

    # ------------------------------------------------------------------
    def _run(self) -> None:
        print("[PSL] Thread running — initialising FaceLandmarker ...")
        try:
            model_path = _ensure_model()
            base_opts  = _mp_python.BaseOptions(model_asset_path=model_path)
            options    = _mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=_mp_vision.RunningMode.IMAGE,
            )
            landmarker = _mp_vision.FaceLandmarker.create_from_options(options)
            print("[PSL] FaceLandmarker ready.")
        except Exception as exc:
            print(f"[PSL] ERROR initialising FaceLandmarker: {exc}")
            return

        stable: PSLResult = PSLResult()   # last accepted + smoothed result

        try:
            while not self._stop.is_set():
                try:
                    frame = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    h, w   = frame.shape[:2]
                    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    out    = landmarker.detect(mp_img)
                except Exception as exc:
                    print(f"[PSL] Frame processing error: {exc}")
                    continue

                if not out.face_landmarks:
                    # No face — clear result but keep stable landmarks for fade-out
                    with self._lock:
                        self._result = PSLResult()
                    stable = PSLResult()
                    continue

                try:
                    lms = out.face_landmarks[0]
                    pts = np.array([[lm.x * w, lm.y * h] for lm in lms],
                                   dtype=np.float32)

                    # Always update landmarks so the face mesh overlay stays live
                    # even when we're not accepting new metric measurements.
                    with self._lock:
                        if self._result.psl_score is not None:
                            self._result.landmarks = pts
                            self._result.age       = time.monotonic()

                    # Frontality gate: skip metric update if face is too turned
                    if not _frontality_ok(pts):
                        continue

                    raw = PSLResult()
                    raw.landmarks     = pts
                    raw.symmetry      = _calc_symmetry(pts)
                    raw.canthal_tilt  = _calc_canthal_tilt(pts)
                    raw.eye_spacing   = _calc_eye_spacing(pts)
                    raw.fwhr          = _calc_fwhr(pts)
                    raw.thirds        = _calc_thirds(pts)
                    raw.jawline_score = _calc_jawline(pts)
                    raw.psl_score     = _calc_psl_score(raw)
                    raw.age           = time.monotonic()

                    # Smooth new measurement into the stable result
                    stable = _smooth_result(stable, raw, _SMOOTH_ALPHA)
                    stable.landmarks = pts          # always current for drawing
                    stable.age       = raw.age

                    with self._lock:
                        self._result = stable
                except Exception as exc:
                    print(f"[PSL] Metric calculation error: {exc}")
        finally:
            landmarker.close()
            print("[PSL] Thread exited.")


# ---------------------------------------------------------------------------
# Overlay renderer (call from display thread)
# ---------------------------------------------------------------------------

_PANEL_W  = 230
_PANEL_H  = 235
_PAD      = 12
_BAR_W    = 80

def _score_color(value: float, lo: float, hi: float) -> tuple:
    """Return BGR color: green at hi, yellow at mid, red at lo."""
    t = float(np.clip((value - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
    if t >= 0.5:
        r = int((1.0 - t) * 2 * 255)
        return (0, 255, r)
    else:
        g = int(t * 2 * 255)
        return (0, g, 255)


def _bar(img, x, y, value, lo, hi, width=_BAR_W, height=8):
    t = float(np.clip((value - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + width, y + height), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y), (x + int(width * t), y + height), _score_color(value, lo, hi), -1)


def _blend_roi(frame: np.ndarray, x: int, y: int, panel: np.ndarray, alpha: float) -> None:
    """Blend *panel* into *frame* at (x, y) without touching the rest of the frame."""
    ph, pw = panel.shape[:2]
    fh, fw = frame.shape[:2]
    # Clamp to frame bounds
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + pw, fw), min(y + ph, fh)
    if x2 <= x1 or y2 <= y1:
        return
    px1, py1 = x1 - x, y1 - y
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
    roi = frame[y1:y2, x1:x2]
    cv2.addWeighted(panel[py1:py2, px1:px2], alpha, roi, 1.0 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def draw_psl_overlay(frame: np.ndarray, result: PSLResult,
                     analyzer: "PSLAnalyzer | None" = None) -> np.ndarray:
    """
    Draw PSL metrics panel anchored above the detected face and return *frame*.
    The panel follows the person as they move.
    Safe to call every display frame; handles None / stale results gracefully.
    """
    h, w = frame.shape[:2]

    # Always show a small status chip in the bottom-left so you can confirm
    # whether the PSL thread is alive and whether a face is currently seen.
    if analyzer is not None:
        if not analyzer.is_running():
            status_txt, status_col = "PSL: stopped", (0, 0, 200)
        elif result is None or result.psl_score is None:
            status_txt, status_col = "PSL: no face", (0, 165, 255)
        else:
            age_s = time.monotonic() - result.age
            if age_s > 1.5:
                status_txt, status_col = "PSL: stale", (0, 165, 255)
            else:
                status_txt, status_col = f"PSL: {result.psl_score:.1f}", (0, 220, 80)
        # Legend panel is ~42 px tall; place chip just above it
        _chip_y = h - 42 - 8
        cv2.putText(frame, status_txt, (10, _chip_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),   2, cv2.LINE_AA)
        cv2.putText(frame, status_txt, (10, _chip_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_col,   1, cv2.LINE_AA)

    if result is None or result.psl_score is None or result.landmarks is None:
        return frame

    # Fade out stale results (> 1.5 s old)
    age = time.monotonic() - result.age
    if age > 1.5:
        return frame
    alpha_fade = float(np.clip(1.0 - (age - 0.5), 0.0, 1.0))

    h, w = frame.shape[:2]
    pts = result.landmarks

    # ------------------------------------------------------------------
    # Derive face bounding box from landmarks
    # ------------------------------------------------------------------
    fx1 = int(np.min(pts[:, 0]))
    fy1 = int(np.min(pts[:, 1]))
    fx2 = int(np.max(pts[:, 0]))
    fy2 = int(np.max(pts[:, 1]))
    fcx = (fx1 + fx2) // 2   # face horizontal centre

    # ------------------------------------------------------------------
    # Draw landmark dots + eye canthal-tilt lines on the face
    # ------------------------------------------------------------------
    for idx in _MESH_DRAW_IDS:
        p = (int(pts[idx][0]), int(pts[idx][1]))
        cv2.circle(frame, p, 2, (0, 210, 180), -1, cv2.LINE_AA)
    for inner, outer in [(_RE_INNER, _RE_OUTER), (_LE_INNER, _LE_OUTER)]:
        p1 = (int(pts[inner][0]), int(pts[inner][1]))
        p2 = (int(pts[outer][0]), int(pts[outer][1]))
        cv2.line(frame, p1, p2, (0, 210, 180), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Build rows for the metrics panel
    # ------------------------------------------------------------------
    score = result.psl_score
    sc    = _score_color(score, 3.0, 8.5)
    tier  = "NPC"
    if score >= 9.0:   tier = "CHAD"
    elif score >= 7.5: tier = "HIGH T"
    elif score >= 6.0: tier = "ABOVE AVG"
    elif score >= 4.5: tier = "AVG"
    elif score >= 3.0: tier = "BELOW AVG"

    rows = []
    if result.symmetry is not None:
        rows.append(("Symmetry",  f"{result.symmetry:.0f}%",
                     result.symmetry, 60.0, 98.0))
    if result.canthal_tilt is not None:
        ct = result.canthal_tilt
        rows.append(("Canthal",   f"{ct:+.1f}\u00b0",
                     ct + 5.0, 0.0, 10.0))
    if result.eye_spacing is not None:
        es = result.eye_spacing
        rows.append(("Eye space", f"{es:.3f}",
                     100.0 - abs(es - 0.46) * 600, 0.0, 100.0))
    if result.fwhr is not None:
        rows.append(("fWHR",      f"{result.fwhr:.2f}",
                     100.0 - abs(result.fwhr - 2.0) * 80, 0.0, 100.0))
    if result.jawline_score is not None:
        rows.append(("Jawline",   f"{result.jawline_score:.0f}",
                     result.jawline_score, 30.0, 90.0))
    if result.thirds is not None:
        u, m, lo_ = result.thirds
        rows.append(("Thirds",    f"{u:.0f}/{m:.0f}/{lo_:.0f}",
                     100.0 - (abs(u-33.3)+abs(m-33.3)+abs(lo_-33.3)), 60.0, 100.0))

    # ------------------------------------------------------------------
    # Panel dimensions (dynamic: grows with row count)
    # ------------------------------------------------------------------
    HEADER_H  = 62   # score + tier label + divider
    ROW_H     = 26
    panel_h   = HEADER_H + len(rows) * ROW_H + _PAD
    panel_w   = _PANEL_W

    # Position: centred on face, sitting just above the forehead with a 6px gap
    gap  = 6
    px   = fcx - panel_w // 2
    py   = fy1 - panel_h - gap

    # If it goes off the top, drop it below the chin instead
    if py < 4:
        py = fy2 + gap
    # Clamp horizontally; clamp vertically so the panel stays above the legend bar
    _legend_h = 42  # height reserved by the bottom key legend
    px = int(np.clip(px, 4, w - panel_w - 4))
    py = int(np.clip(py, 4, h - panel_h - _legend_h - 4))

    # ------------------------------------------------------------------
    # Connector line: panel bottom-centre → forehead landmark
    # ------------------------------------------------------------------
    line_top    = (fcx, fy1)
    line_bottom = (px + panel_w // 2, py + panel_h)
    if py > fy2:   # panel is below face
        line_top    = (fcx, fy2)
        line_bottom = (px + panel_w // 2, py)
    cv2.line(frame, line_bottom, line_top, (0, 210, 180), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Semi-transparent panel background (ROI-only blend)
    # ------------------------------------------------------------------
    bg = np.full((panel_h, panel_w, 3), (15, 15, 15), dtype=np.uint8)
    cv2.rectangle(bg, (0, 0), (panel_w - 1, panel_h - 1), (60, 60, 60), 1)
    _blend_roi(frame, px, py, bg, 0.78 * alpha_fade)

    # ------------------------------------------------------------------
    # Header: score + tier label
    # ------------------------------------------------------------------
    cv2.putText(frame, f"PSL  {score:.1f}",
                (px + _PAD, py + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.82, sc, 2, cv2.LINE_AA)
    cv2.putText(frame, tier,
                (px + _PAD, py + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, sc, 1, cv2.LINE_AA)
    cv2.line(frame,
             (px + _PAD, py + 56),
             (px + panel_w - _PAD, py + 56),
             (60, 60, 60), 1)

    # ------------------------------------------------------------------
    # Metric rows
    # ------------------------------------------------------------------
    ry = py + HEADER_H
    bar_w = panel_w - _PAD * 2
    for lbl, val, bv, blo, bhi in rows:
        cv2.putText(frame, lbl,  (px + _PAD,      ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, val,  (px + 92,         ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (230, 230, 230), 1, cv2.LINE_AA)
        _bar(frame, px + _PAD, ry + 3, bv, blo, bhi, width=bar_w, height=4)
        ry += ROW_H

    return frame
