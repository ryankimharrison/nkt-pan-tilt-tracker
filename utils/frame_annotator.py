"""utils/frame_annotator.py — Tactical HUD overlay with SF/Segoe UI font."""

from __future__ import annotations
import cv2
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import config

# COCO pose skeleton connections (0-indexed)
_COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
]
_KEYPOINT_CONF_THRESH = 0.3

# Colours (BGR for cv2, RGB for PIL)
_ORANGE       = (0, 107, 255)
_ORANGE_DIM   = (0,  80, 200)
_WHITE        = (255, 255, 255)
_GRAY         = (180, 180, 180)
_GRAY_DIM     = (120, 120, 120)
_BLACK        = (0,   0,    0)
_RED          = (0,   0,   255)
_SUCCESS      = (94,  197,  34)
_WARNING      = (11,  158, 245)
_GREEN        = (0,   255,  0)
_CYAN         = (255, 200,  0)   # light blue in BGR

# RGB versions for PIL (swap B and R)
def _rgb(bgr): return (bgr[2], bgr[1], bgr[0])

_ORANGE_RGB   = _rgb(_ORANGE)
_ORANGE_DIM_RGB = _rgb(_ORANGE_DIM)
_WHITE_RGB    = _rgb(_WHITE)
_GRAY_RGB     = _rgb(_GRAY)
_GRAY_DIM_RGB = _rgb(_GRAY_DIM)
_BLACK_RGB    = _rgb(_BLACK)
_RED_RGB      = _rgb(_RED)
_SUCCESS_RGB  = _rgb(_SUCCESS)
_WARNING_RGB  = _rgb(_WARNING)
_CYAN_RGB     = _rgb(_CYAN)

# ── Font loading (SF Pro → Segoe UI → Arial → default) ──
_FONT_NAMES = [
    "GoogleSans-Regular.ttf", "Google Sans Regular.ttf",
    "ProductSans-Regular.ttf",
    "SF-Pro-Display-Medium.otf", "SFProDisplay-Medium.otf",
    "segoeui.ttf", "arial.ttf",
]
_FONT_NAMES_BOLD = [
    "GoogleSans-Bold.ttf", "Google Sans Bold.ttf",
    "ProductSans-Bold.ttf",
    "SF-Pro-Display-Bold.otf", "SFProDisplay-Bold.otf",
    "segoeuib.ttf", "arialbd.ttf",
]

import os as _os
_FONTS_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "fonts")

def _load_font(size, bold=False):
    names = _FONT_NAMES_BOLD if bold else _FONT_NAMES
    for name in names:
        # Try local fonts/ directory first, then system fonts
        for path in [_os.path.join(_FONTS_DIR, name), name]:
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue
    return ImageFont.load_default()

# Cache fonts at common sizes
_F12    = _load_font(12)
_F13    = _load_font(13)
_F14    = _load_font(14)
_F14B   = _load_font(14, bold=True)
_F16    = _load_font(16)
_F16B   = _load_font(16, bold=True)
_F20B   = _load_font(20, bold=True)
_F28B   = _load_font(28, bold=True)

# cv2 font fallback for geometric labels
_CV_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_hud_bg(img, x, y, w, h, alpha=0.6):
    """Semi-transparent dark panel."""
    y1, y2 = max(0, y), min(img.shape[0], y + h)
    x1, x2 = max(0, x), min(img.shape[1], x + w)
    if y2 <= y1 or x2 <= x1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, (20, 20, 20), dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def _dashed_rect(img, x1, y1, x2, y2, color, thick=1, dash=8):
    pts = [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]
    for ax, ay, bx, by in pts:
        dx, dy = bx - ax, by - ay
        length = max(1, int((dx**2 + dy**2) ** 0.5))
        steps  = max(1, length // (dash * 2))
        for s in range(steps):
            t0 = s * 2 * dash / length
            t1 = min(1.0, (s * 2 * dash + dash) / length)
            p0 = (int(ax + dx * t0), int(ay + dy * t0))
            p1 = (int(ax + dx * t1), int(ay + dy * t1))
            cv2.line(img, p0, p1, color, thick, cv2.LINE_AA)


def _draw_pose(img, keypoints, color, is_target):
    if not keypoints or len(keypoints) < 17:
        return
    pts = [(int(k[0]), int(k[1]), k[2]) for k in keypoints]
    line_thick = 2 if is_target else 1
    dot_rad = 4 if is_target else 2
    for (i, j) in _COCO_SKELETON:
        if i >= len(pts) or j >= len(pts):
            continue
        x1, y1, c1 = pts[i]
        x2, y2, c2 = pts[j]
        if c1 >= _KEYPOINT_CONF_THRESH and c2 >= _KEYPOINT_CONF_THRESH:
            cv2.line(img, (x1, y1), (x2, y2), color, line_thick, cv2.LINE_AA)
    for x, y, c in pts:
        if c >= _KEYPOINT_CONF_THRESH:
            cv2.circle(img, (x, y), dot_rad, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), dot_rad, _BLACK, 1, cv2.LINE_AA)


def _corner_accents(img, x1, y1, x2, y2, color, length=14, thick=2):
    corners = [
        ((x1, y1), ( 1,  0), ( 0,  1)),
        ((x2, y1), (-1,  0), ( 0,  1)),
        ((x1, y2), ( 1,  0), ( 0, -1)),
        ((x2, y2), (-1,  0), ( 0, -1)),
    ]
    for (cx, cy), (hx, hy), (vx, vy) in corners:
        cv2.line(img, (cx, cy), (cx + hx * length, cy + hy * length), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx + vx * length, cy + vy * length), color, thick, cv2.LINE_AA)


def _draw_frame_border(img, w, h):
    """Thin double border with corner brackets."""
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), _ORANGE, 1)
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), _ORANGE_DIM, 1)
    L = 35
    for (cx, cy, hx, vy) in [(2, 2, 1, 1), (w-3, 2, -1, 1), (2, h-3, 1, -1), (w-3, h-3, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + hx * L, cy), _ORANGE, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + vy * L), _ORANGE, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# PIL text rendering helpers
# ---------------------------------------------------------------------------

def _pil_draw_shadow_text(draw, x, y, text, font, fill_rgb):
    """Draw text with black shadow for readability."""
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 200))
    draw.text((x, y), text, font=font, fill=(*fill_rgb, 255))


def _draw_text_block(draw, lines, x, y, line_h, font, bold_font=None):
    """Draw a block of (text, color_rgb) lines. Returns total height."""
    for i, (txt, color_rgb, is_header) in enumerate(lines):
        if txt:
            f = bold_font if (is_header and bold_font) else font
            _pil_draw_shadow_text(draw, x, y + i * line_h, txt, f, color_rgb)
    return len(lines) * line_h


# ---------------------------------------------------------------------------
# Telemetry graph ring buffers (module-level, persist across frames)
# ---------------------------------------------------------------------------

_GRAPH_LEN = 120  # ~2 seconds at 60fps

_buf_pan_vel   = deque(maxlen=_GRAPH_LEN)
_buf_tilt_vel  = deque(maxlen=_GRAPH_LEN)
_buf_pan_acc   = deque(maxlen=_GRAPH_LEN)
_buf_tilt_acc  = deque(maxlen=_GRAPH_LEN)
_prev_pan_vel  = 0.0
_prev_tilt_vel = 0.0


def _draw_graph(img, x, y, gw, gh, buf1, buf2, color1, color2, y_floor=1.0):
    """Draw a single telemetry graph panel with two traces."""
    _draw_hud_bg(img, x, y, gw, gh, alpha=0.7)
    cy = y + gh // 2
    # Zero line
    cv2.line(img, (x + 1, cy), (x + gw - 1, cy), _GRAY_DIM, 1, cv2.LINE_AA)
    # Faint quarter lines
    cv2.line(img, (x + 1, y + gh // 4), (x + gw - 1, y + gh // 4), (40, 40, 40), 1)
    cv2.line(img, (x + 1, y + 3 * gh // 4), (x + gw - 1, y + 3 * gh // 4), (40, 40, 40), 1)
    # Border
    cv2.rectangle(img, (x, y), (x + gw, y + gh), _ORANGE_DIM, 1)

    # Auto-scale Y from both buffers
    y_max = y_floor
    for buf in (buf1, buf2):
        if buf:
            m = max(abs(v) for v in buf)
            if m > y_max:
                y_max = m
    y_max *= 1.1  # 10% headroom

    half = (gh - 2) / 2.0

    # Draw traces
    for buf, color in ((buf1, color1), (buf2, color2)):
        n = len(buf)
        if n < 2:
            continue
        # Map buffer values to pixel coordinates, right-aligned
        step = max(1.0, (gw - 2) / (_GRAPH_LEN - 1))
        x_start = x + gw - 1 - int((n - 1) * step)
        pts = []
        for i, val in enumerate(buf):
            px = x_start + int(i * step)
            py = int(cy - (val / y_max) * half)
            py = max(y + 1, min(y + gh - 1, py))
            pts.append((px, py))
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main annotate function
# ---------------------------------------------------------------------------

def annotate(
    frame: np.ndarray,
    tracks,
    target,
    pan_error:    float,
    tilt_error:   float,
    pan_vel:      float,
    tilt_vel:     float,
    deadzone:     float,
    fps_capture:  float,
    fps_inference: float,
    fps_loop:     float,
    person_count: int,
    motor_enabled: bool,
    is_idle:      bool,
    pan_steps_sec:  float = 0.0,
    tilt_steps_sec: float = 0.0,
    params: dict | None = None,
    fire_mode:        str   = "auto",
    is_firing:        bool  = False,
    tracking_enabled: bool  = True,
    fire_zone:        float = 0.10,
    in_fire_zone:     bool  = False,
    locked_slot:      int | None = None,
    tilt_upper_calibrated: bool = False,
    tilt_lower_calibrated: bool = False,
    limits_calibrated:     bool = False,
    control_source:        str  = "pid",
    est_distance_m:        float = 0.0,
    waypoints_world:       list | None = None,
    waypoint_current_idx:  int  = -1,
    waypoint_mode:         bool = False,
    turret_pan_deg:        float = 0.0,
    turret_tilt_deg:       float = 0.0,
    cal_active:            bool = False,
    cal_click_px:          tuple | None = None,
    ballistic_lead_px:     tuple = (0.0, 0.0),
    ballistic_active:      bool = False,
    vel_px:                tuple = (0.0, 0.0),
    predict_px:            tuple = (0.0, 0.0),
    using_narrow_cam:      bool = False,
    reid_names:            dict = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cx_frame = w // 2 + config.CROSSHAIR_OFFSET_X
    cy_frame = h // 2 + config.CROSSHAIR_OFFSET_Y
    p = params or {}

    # ── cv2 geometric drawing (fast, stays on numpy) ──

    # Pose keypoints
    for t in tracks:
        if t.keypoints:
            is_tgt = target is not None and t.track_id == target.track_id
            _draw_pose(out, t.keypoints, _GREEN if is_tgt else _GRAY, is_tgt)

    # Bounding boxes
    for t in tracks:
        x1, y1, x2, y2 = int(t.x1), int(t.y1), int(t.x2), int(t.y2)
        is_tgt = target is not None and t.track_id == target.track_id
        _rn = reid_names or {}
        if is_tgt:
            cv2.rectangle(out, (x1, y1), (x2, y2), _GREEN, 2)
            _corner_accents(out, x1, y1, x2, y2, _GREEN)
            _name = _rn.get(t.global_id, f"P{t.global_id}" if t.global_id >= 0 else f"ID {t.track_id}")
            label = f"{_name}  {t.conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, _CV_FONT, 0.45, 1)
            cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 4, y1), _GREEN, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4), _CV_FONT, 0.45, _BLACK, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), _GRAY, 1)
            _name = _rn.get(t.global_id, f"P{t.global_id}" if t.global_id >= 0 else str(t.track_id))
            cv2.putText(out, _name, (x1 + 2, y1 + 14), _CV_FONT, 0.45, _GRAY, 1, cv2.LINE_AA)

    # Tactical crosshair reticle
    _cx, _cy = cx_frame, cy_frame
    # Outer ring (faint)
    cv2.circle(out, (_cx, _cy), 60, _ORANGE_DIM, 1, cv2.LINE_AA)
    # Inner ring
    cv2.circle(out, (_cx, _cy), 30, _ORANGE, 1, cv2.LINE_AA)
    # Long gapped cross lines — outer segments (far from center)
    cv2.line(out, (_cx - 90, _cy), (_cx - 60, _cy), _ORANGE_DIM, 1, cv2.LINE_AA)
    cv2.line(out, (_cx + 60, _cy), (_cx + 90, _cy), _ORANGE_DIM, 1, cv2.LINE_AA)
    cv2.line(out, (_cx, _cy - 90), (_cx, _cy - 60), _ORANGE_DIM, 1, cv2.LINE_AA)
    cv2.line(out, (_cx, _cy + 60), (_cx, _cy + 90), _ORANGE_DIM, 1, cv2.LINE_AA)
    # Mid cross lines — between rings
    cv2.line(out, (_cx - 55, _cy), (_cx - 30, _cy), _ORANGE, 2, cv2.LINE_AA)
    cv2.line(out, (_cx + 30, _cy), (_cx + 55, _cy), _ORANGE, 2, cv2.LINE_AA)
    cv2.line(out, (_cx, _cy - 55), (_cx, _cy - 30), _ORANGE, 2, cv2.LINE_AA)
    cv2.line(out, (_cx, _cy + 30), (_cx, _cy + 55), _ORANGE, 2, cv2.LINE_AA)
    # Inner cross — close to center
    cv2.line(out, (_cx - 18, _cy), (_cx - 6, _cy), _ORANGE, 2, cv2.LINE_AA)
    cv2.line(out, (_cx + 6, _cy), (_cx + 18, _cy), _ORANGE, 2, cv2.LINE_AA)
    cv2.line(out, (_cx, _cy - 18), (_cx, _cy - 6), _ORANGE, 2, cv2.LINE_AA)
    cv2.line(out, (_cx, _cy + 6), (_cx, _cy + 18), _ORANGE, 2, cv2.LINE_AA)
    # Tick marks on outer ring (cardinal, 4px perpendicular stubs)
    cv2.line(out, (_cx - 60, _cy - 4), (_cx - 60, _cy + 4), _ORANGE, 1, cv2.LINE_AA)
    cv2.line(out, (_cx + 60, _cy - 4), (_cx + 60, _cy + 4), _ORANGE, 1, cv2.LINE_AA)
    cv2.line(out, (_cx - 4, _cy - 60), (_cx + 4, _cy - 60), _ORANGE, 1, cv2.LINE_AA)
    cv2.line(out, (_cx - 4, _cy + 60), (_cx + 4, _cy + 60), _ORANGE, 1, cv2.LINE_AA)
    # Diagonal tick marks on outer ring (45°, at r=60)
    _d = int(60 * 0.7071)  # 60 * cos(45°)
    for _sx, _sy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
        _dx, _dy = _cx + _sx * _d, _cy + _sy * _d
        cv2.line(out, (_dx - _sy * 3, _dy + _sx * 3),
                 (_dx + _sy * 3, _dy - _sx * 3), _ORANGE, 1, cv2.LINE_AA)
    # Center dot
    cv2.circle(out, (_cx, _cy), 3, _ORANGE, -1, cv2.LINE_AA)
    cv2.circle(out, (_cx, _cy), 3, _BLACK, 1, cv2.LINE_AA)

    # Ballistic lead reticle — shows where darts will be aimed
    if ballistic_active and config.BALLISTIC_SHOW_RETICLE:
        _bl_x = int(_cx + ballistic_lead_px[0])
        _bl_y = int(_cy + ballistic_lead_px[1])
        _bl_x = max(10, min(w - 10, _bl_x))
        _bl_y = max(10, min(h - 10, _bl_y))
        # Lead reticle: diamond shape in cyan
        _LEAD_CLR = (255, 255, 0)  # cyan BGR
        _sz = 12
        pts = np.array([
            [_bl_x, _bl_y - _sz], [_bl_x + _sz, _bl_y],
            [_bl_x, _bl_y + _sz], [_bl_x - _sz, _bl_y],
        ], dtype=np.int32)
        cv2.polylines(out, [pts], True, _LEAD_CLR, 2, cv2.LINE_AA)
        cv2.circle(out, (_bl_x, _bl_y), 3, _LEAD_CLR, -1, cv2.LINE_AA)
        # Line from crosshair center to lead point
        cv2.line(out, (_cx, _cy), (_bl_x, _bl_y), _LEAD_CLR, 1, cv2.LINE_AA)
        # Label
        cv2.putText(out, "LEAD", (_bl_x + 14, _bl_y - 4),
                    _CV_FONT, 0.4, _LEAD_CLR, 1, cv2.LINE_AA)

    # Fire zone + deadzone
    if target is not None:
        dz_cx, dz_cy = int(target.aim_cx), int(target.aim_cy)
    else:
        dz_cx, dz_cy = cx_frame, cy_frame

    # Fire zone = target bounding box (drawn as dashed rect)
    if target is not None:
        fz_color = _RED if in_fire_zone else _ORANGE_DIM
        _dashed_rect(out, int(target.x1), int(target.y1),
                     int(target.x2), int(target.y2), fz_color, thick=2, dash=6)

    if deadzone >= 0.005:
        dz_pw = int(deadzone * (w / 2))
        dz_ph = int(deadzone * (h / 2))
        _dashed_rect(out, dz_cx - dz_pw, dz_cy - dz_ph,
                     dz_cx + dz_pw, dz_cy + dz_ph, _ORANGE_DIM, thick=1)

    # Target dot + error arrow
    if target is not None:
        tcx, tcy = int(target.aim_cx), int(target.aim_cy)
        cv2.circle(out, (tcx, tcy), 6, _BLACK, -1)
        cv2.circle(out, (tcx, tcy), 5, _RED,   -1)
        cv2.circle(out, (tcx, tcy), 6, _BLACK,  1)
        cv2.arrowedLine(out, (cx_frame, cy_frame), (tcx, tcy), _ORANGE, 1,
                        cv2.LINE_AA, tipLength=0.15)

    # Velocity arrow — shows direction and speed of target motion
    if target is not None:
        _vx, _vy = vel_px
        _speed_px = (_vx ** 2 + _vy ** 2) ** 0.5
        if _speed_px > 30:  # only draw when target is visibly moving
            tcx, tcy = int(target.aim_cx), int(target.aim_cy)
            # Scale: 0.1 seconds of motion = arrow length
            _arrow_scale = 0.1
            _ax = int(tcx + _vx * _arrow_scale)
            _ay = int(tcy + _vy * _arrow_scale)
            # Clamp to frame
            _ax = max(5, min(w - 5, _ax))
            _ay = max(5, min(h - 5, _ay))
            # Yellow arrow from target center in direction of motion
            _VEL_CLR = (0, 255, 255)  # yellow BGR
            cv2.arrowedLine(out, (tcx, tcy), (_ax, _ay), _VEL_CLR, 2,
                            cv2.LINE_AA, tipLength=0.25)
            # Speed label near arrow tip
            _spd_label = f"{_speed_px:.0f} px/s"
            cv2.putText(out, _spd_label, (_ax + 8, _ay - 4),
                        _CV_FONT, 0.35, _VEL_CLR, 1, cv2.LINE_AA)

        # Prediction compensation marker — small + where the aim is shifted to
        _px, _py = predict_px
        if abs(_px) > 1 or abs(_py) > 1:
            _pred_x = int(cx_frame + _px)
            _pred_y = int(cy_frame + _py)
            _pred_x = max(5, min(w - 5, _pred_x))
            _pred_y = max(5, min(h - 5, _pred_y))
            # Small green + marker showing prediction shift
            _PRED_CLR = (0, 255, 100)  # bright green
            cv2.drawMarker(out, (_pred_x, _pred_y), _PRED_CLR,
                           cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
            _comp_ms = config.PREDICTION_EXTRA_MS + 16  # approx total
            cv2.putText(out, f"PRED +{_comp_ms:.0f}ms",
                        (_pred_x + 8, _pred_y + 4),
                        _CV_FONT, 0.3, _PRED_CLR, 1, cv2.LINE_AA)

    # Waypoint dots — project from world angles to current pixel positions
    _wp_px_list = []
    if waypoints_world:
        _ppx = config.CAMERA_PAN_DEG_PER_PX
        _tpx = config.CAMERA_TILT_DEG_PER_PX
        for i, (wp_pan, wp_tilt) in enumerate(waypoints_world):
            # Offset from current turret position → pixel offset from center
            _d_pan  = wp_pan  - turret_pan_deg
            _d_tilt = wp_tilt - turret_tilt_deg
            # Invert back to pixel space if axis is inverted
            if config.PAN_INVERT:
                _d_pan = -_d_pan
            if config.TILT_INVERT:
                _d_tilt = -_d_tilt
            px_x = int(w / 2.0 + _d_pan  / _ppx) if _ppx > 0 else cx_frame
            px_y = int(h / 2.0 + _d_tilt / _tpx) if _tpx > 0 else cy_frame
            _wp_px_list.append((px_x, px_y))
            # Only draw if on-screen
            if 0 <= px_x < w and 0 <= px_y < h:
                if i < waypoint_current_idx:
                    color = (0, 200, 0)      # green = done
                elif i == waypoint_current_idx:
                    color = (0, 165, 255)    # orange = current target
                else:
                    color = (0, 0, 255)      # red = pending
                cv2.circle(out, (px_x, px_y), 10, _BLACK, -1)
                cv2.circle(out, (px_x, px_y), 9, color, -1)
                cv2.circle(out, (px_x, px_y), 10, _BLACK, 1)
                cv2.putText(out, str(i + 1), (px_x - 5, px_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        # Connecting lines
        for i in range(len(_wp_px_list) - 1):
            cv2.line(out, _wp_px_list[i], _wp_px_list[i + 1], (0, 0, 180), 1, cv2.LINE_AA)

    # Calibration crosshair on clicked point
    if cal_active and cal_click_px is not None:
        cx_cal, cy_cal = cal_click_px
        cv2.drawMarker(out, (cx_cal, cy_cal), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)

    # Frame border
    _draw_frame_border(out, w, h)

    # ── Telemetry graphs (bottom-right) ──
    global _prev_pan_vel, _prev_tilt_vel
    pan_acc  = pan_vel  - _prev_pan_vel
    tilt_acc = tilt_vel - _prev_tilt_vel
    _prev_pan_vel  = pan_vel
    _prev_tilt_vel = tilt_vel
    _buf_pan_vel.append(pan_vel)
    _buf_tilt_vel.append(tilt_vel)
    _buf_pan_acc.append(pan_acc)
    _buf_tilt_acc.append(tilt_acc)

    _gw, _gh = 250, 64
    _gx = w - _gw - 14
    _gy_base = h - 38 - 40 - 2 * (_gh + 4)  # above bottom legend (38px) + biometric (40px)
    _draw_graph(out, _gx, _gy_base,           _gw, _gh, _buf_pan_vel,  _buf_tilt_vel,  _ORANGE, _CYAN, y_floor=5.0)
    _draw_graph(out, _gx, _gy_base + _gh + 4, _gw, _gh, _buf_pan_acc, _buf_tilt_acc, _ORANGE, _CYAN, y_floor=2.0)

    # HUD background panels (drawn before PIL text)
    panel_top = 50
    # Left panel
    left_panel_h = 20 * 18 + 16
    _draw_hud_bg(out, 10, panel_top, 280, left_panel_h)
    cv2.line(out, (10, panel_top), (10, panel_top + left_panel_h), _ORANGE, 2)

    # Right panel
    right_panel_h = 20 * 17 + 16
    _draw_hud_bg(out, w - 290, panel_top, 280, right_panel_h)
    cv2.line(out, (w - 11, panel_top), (w - 11, panel_top + right_panel_h), _ORANGE, 2)

    # Title bar bg — two-line title, large bold font
    title_l1 = "Autonomous Inference-Powered Spatiotemporal Kinematic"
    title_l2 = "Pan-Tilt Stepper Actuated Gelatinous Projectile Dispersion System"
    bbox1 = _F16B.getbbox(title_l1)
    bbox2 = _F16B.getbbox(title_l2)
    tw1 = bbox1[2] - bbox1[0] if bbox1 else 500
    tw2 = bbox2[2] - bbox2[0] if bbox2 else 500
    tw = max(tw1, tw2)
    tx1 = (w - tw1) // 2
    tx2 = (w - tw2) // 2
    tx_bg = (w - tw) // 2
    title_h = 44
    _draw_hud_bg(out, tx_bg - 14, 2, tw + 28, title_h, alpha=0.8)
    cv2.line(out, (tx_bg - 14, 2), (tx_bg + tw + 14, 2), _ORANGE, 2)
    cv2.line(out, (tx_bg - 14, 2 + title_h), (tx_bg + tw + 14, 2 + title_h), _ORANGE, 1)

    # Bottom legend bg
    leg_h = 36
    leg_y0 = h - leg_h - 2
    _draw_hud_bg(out, 2, leg_y0, w - 4, leg_h + 2, alpha=0.75)
    cv2.line(out, (2, leg_y0), (w - 3, leg_y0), _ORANGE, 1)

    # Biometric scan bg
    bio_y = h - 72
    _draw_hud_bg(out, 12, bio_y - 4, 320, 24, alpha=0.65)

    # TRACKING DISABLED banner (cv2 for speed)
    if not tracking_enabled:
        ban = "TRACKING DISABLED"
        (bw, bh), _ = cv2.getTextSize(ban, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        bx = (w - bw) // 2
        by = 68
        cv2.rectangle(out, (bx - 12, by - bh - 6), (bx + bw + 12, by + 6), _BLACK, -1)
        cv2.rectangle(out, (bx - 12, by - bh - 6), (bx + bw + 12, by + 6), _RED, 2)
        cv2.putText(out, ban, (bx, by), cv2.FONT_HERSHEY_DUPLEX, 0.8, _RED, 2, cv2.LINE_AA)

    # ENGAGING banner
    if is_firing:
        ftxt = "ENGAGING"
        (fw, fh), _ = cv2.getTextSize(ftxt, cv2.FONT_HERSHEY_DUPLEX, 1.0, 3)
        fx = (w - fw) // 2
        fy = 68 if tracking_enabled else 103
        cv2.rectangle(out, (fx - 16, fy - fh - 8), (fx + fw + 16, fy + 8), _RED, -1)
        cv2.rectangle(out, (fx - 16, fy - fh - 8), (fx + fw + 16, fy + 8), _ORANGE, 2)
        cv2.putText(out, ftxt, (fx, fy), cv2.FONT_HERSHEY_DUPLEX, 1.0, _WHITE_RGB, 3, cv2.LINE_AA)

    # CALIBRATE LIMITS warning banner
    if not limits_calibrated:
        cal_txt = "CALIBRATE LIMITS [L]"
        (cw, ch), _ = cv2.getTextSize(cal_txt, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        cal_x = (w - cw) // 2
        cal_y_base = 68
        if not tracking_enabled:
            cal_y_base = 95
        if is_firing:
            cal_y_base = 135 if not tracking_enabled else 103
        cv2.rectangle(out, (cal_x - 14, cal_y_base - ch - 8),
                      (cal_x + cw + 14, cal_y_base + 8), _BLACK, -1)
        cv2.rectangle(out, (cal_x - 14, cal_y_base - ch - 8),
                      (cal_x + cw + 14, cal_y_base + 8), _WARNING, 2)
        cv2.putText(out, cal_txt, (cal_x, cal_y_base),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, _WARNING, 2, cv2.LINE_AA)

    # ── PIL text pass: convert to PIL, draw all text, convert back ──
    pil_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Title (two-line, bold, centered)
    _pil_draw_shadow_text(draw, tx1, 5, title_l1, _F16B, _ORANGE_RGB)
    _pil_draw_shadow_text(draw, tx2, 25, title_l2, _F16B, _ORANGE_RGB)

    # ── Left HUD — all status data ──
    state_str = "IDLE" if is_idle else ("OFF" if not tracking_enabled else "TRACKING")
    state_clr = _GRAY_RGB if is_idle else (_RED_RGB if not tracking_enabled else _ORANGE_RGB)
    motor_str = "ON" if motor_enabled else "OFF"

    fire_clr = _RED_RGB if is_firing else (_GRAY_RGB if fire_mode == "off" else (_ORANGE_RGB if fire_mode == "auto" else _WARNING_RGB))
    people_str = f"People: {person_count}  {'[auto]' if locked_slot is None else f'[locked #{locked_slot}]'}"
    people_clr = _ORANGE_RGB if locked_slot is not None else _WHITE_RGB

    lx, ly, lh = 18, 54, 20
    left_data = [
        ("SENSOR STATUS",                      _ORANGE_RGB,  True),
        ("",                                    _WHITE_RGB,   False),
        (f"Capture  {fps_capture:.1f} fps",     _WHITE_RGB,   False),
        (f"Inference {fps_inference:.1f} fps",   _WHITE_RGB,   False),
        (f"Loop     {fps_loop:.1f} fps",         _WHITE_RGB,   False),
        ("",                                    _WHITE_RGB,   False),
        (f"State:  {state_str}",                state_clr,    False),
        (f"Motor:  {motor_str}",                _SUCCESS_RGB if motor_enabled else _RED_RGB, False),
        (f"Fire:   {fire_mode.upper()}",         fire_clr,     False),
        (f"Input:  {control_source.upper()}",    _CYAN_RGB if control_source != "pid" else _WHITE_RGB, False),
        (people_str,                             people_clr,   False),
        ("",                                    _WHITE_RGB,   False),
        (f"Pan  err {pan_error:+.3f}",           _ORANGE_RGB,  False),
        (f"Tilt err {tilt_error:+.3f}",          _ORANGE_RGB,  False),
        (f"Pan  vel {pan_vel:+.1f} dps",         _WHITE_RGB,   False),
        (f"Tilt vel {tilt_vel:+.1f} dps",        _WHITE_RGB,   False),
        (f"Pan  sps {pan_steps_sec:+.0f}",       _GRAY_DIM_RGB, False),
        (f"Tilt sps {tilt_steps_sec:+.0f}",      _GRAY_DIM_RGB, False),
        ("",                                    _WHITE_RGB,   False),
        (f"Limits:    {'CALIBRATED' if limits_calibrated else 'NOT SET'}",
         _SUCCESS_RGB if limits_calibrated else _WARNING_RGB, False),
    ]
    _draw_text_block(draw, left_data, lx, ly, lh, _F14, _F14B)

    # ── Right HUD — params + tracking + status ──
    rx_base = w - 282
    ry = 54
    right_data = [
        ("ACTUATOR DIAGNOSTICS",                                         _ORANGE_RGB,  True),
        ("",                                                             _WHITE_RGB,   False),
        (f"Kp={p.get('pan_kp',0):.1f}  Kd={p.get('pan_kd',0):.1f}  G={p.get('pan_gain',1):.2f}", _WHITE_RGB, False),
        (f"Pan MaxVel  {p.get('max_pan_velocity',0):.0f} dps",           _WHITE_RGB,   False),
        (f"Kp={p.get('tilt_kp',0):.1f}  Kd={p.get('tilt_kd',0):.1f}",   _WHITE_RGB,   False),
        (f"Tilt MaxVel {p.get('max_tilt_velocity',0):.0f} dps",          _WHITE_RGB,   False),
        ("",                                                             _WHITE_RGB,   False),
        ("TRACKING STATUS",                                              _ORANGE_RGB,  True),
        (f"Target:   {'ID ' + str(target.track_id) if target else 'NONE'}",
         _WHITE_RGB if target else _GRAY_RGB, False),
        ("Algorithm: Predictive AI",                                     _WHITE_RGB,   False),
        (f"Deadzone:  {deadzone:.3f}",                                    _WHITE_RGB,   False),
        (f"EMA:       {p.get('ema_alpha',0):.2f}",                       _WHITE_RGB,   False),
        (f"Distance:  {est_distance_m:.1f}m [{('CLOSE' if est_distance_m < config.DISTANCE_CLOSE_M else ('FAR' if est_distance_m > config.DISTANCE_FAR_M else 'MEDIUM')) if est_distance_m > 0 else 'N/A'}]",
         (_SUCCESS_RGB if est_distance_m > 0 and est_distance_m < config.DISTANCE_CLOSE_M else (_RED_RGB if est_distance_m > config.DISTANCE_FAR_M else _WARNING_RGB)) if est_distance_m > 0 else _GRAY_RGB, False),
        (f"Waypoint:  {'EXEC ' + str(waypoint_current_idx + 1) + '/' + str(len(waypoints_world)) if waypoints_world and waypoint_current_idx >= 0 else (str(len(waypoints_world)) + ' placed' if waypoints_world else 'OFF')}" if waypoint_mode else "",
         _ORANGE_RGB if waypoint_mode else _WHITE_RGB, False),
        ("",                                                             _WHITE_RGB,   False),
        ("LOCK STATUS",                                                  _ORANGE_RGB,  True),
        (f"System:  {'TRACKING' if tracking_enabled and not is_idle else ('STANDBY' if is_idle else 'OFFLINE')}",
         _SUCCESS_RGB if (tracking_enabled and not is_idle) else (_GRAY_RGB if is_idle else _RED_RGB), False),
        (f"Motor:   {'ENABLED' if motor_enabled else 'DISABLED'}",
         _SUCCESS_RGB if motor_enabled else _RED_RGB, False),
        (f"Camera:  {'NARROW/IR' if using_narrow_cam else 'WIDE 120°'}",
         (0, 200, 255) if using_narrow_cam else _SUCCESS_RGB, False),
        (f"Safety:  ENABLED",                                            _SUCCESS_RGB, False),
    ]
    _draw_text_block(draw, right_data, rx_base, ry, lh, _F14, _F14B)

    # ── Biometric scan ──
    if target is not None:
        _rn2 = reid_names or {}
        _bio_name = _rn2.get(target.global_id, f"P{target.global_id}" if target.global_id >= 0 else f"ID {target.track_id}")
        bio_txt = f"BIOMETRIC SCAN: LOCKED [{_bio_name}]"
        bio_clr = _SUCCESS_RGB
    else:
        bio_txt = "BIOMETRIC SCAN: NO TARGET"
        bio_clr = _RED_RGB
    _pil_draw_shadow_text(draw, 20, bio_y, bio_txt, _F16B, bio_clr)

    # ── Graph labels ──
    _glx = _gx + 3
    _pil_draw_shadow_text(draw, _glx, _gy_base + 1,           "VEL", _F12, _ORANGE_RGB)
    _pil_draw_shadow_text(draw, _glx, _gy_base + _gh + 5,     "ACC", _F12, _ORANGE_RGB)
    # Legend: pan=orange, tilt=cyan
    _leg_lx = _gx + _gw - 70
    _pil_draw_shadow_text(draw, _leg_lx,      _gy_base - 14, "PAN", _F12, _ORANGE_RGB)
    _pil_draw_shadow_text(draw, _leg_lx + 36, _gy_base - 14, "TILT", _F12, (0, 200, 255))

    # ── Bottom legend ──
    leg1 = "[W/A/S/D] Move Camera   [T] Toggle Tracking   [Tab] Next Target   [F] Fire Mode   [P] Ballistic Lead   [L] Set Boundaries   [I] System Info"
    leg2 = "[Space] Manual Fire   [N] Force Engage   [H] Hand/Head   [G] Scan   [U] Name Person   [E] E-Stop   [R] Re-center   [V] Waypoint   [Q] Exit"
    draw.text((16, leg_y0 + 4), leg1, font=_F12, fill=(*_GRAY_RGB, 220))
    draw.text((16, leg_y0 + 19), leg2, font=_F12, fill=(*_GRAY_RGB, 220))

    # ── Composite PIL overlay onto frame ──
    pil_img = Image.alpha_composite(pil_img, overlay)
    out = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    return out