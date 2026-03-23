# =============================================================
#  config.py — All tunable parameters for the pan-tilt tracker
# =============================================================

# --------------- Camera ---------------
# "auto_external" = scan for the first non-built-in USB camera (recommended).
# Set to an integer (0, 1, 2 ...) to pin a specific index and skip detection.
CAMERA_INDEX  = "auto_external"

# If set, _find_external_camera() matches this VID&PID string first (case-insensitive)
# before falling back to heuristics.  This is the SVPRO OV2710 USB camera.
PREFERRED_CAMERA_VIDPID = "VID_32E4&PID_9230"
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 60

# --------------- Serial ---------------
# "auto" = scan ports for Arduino Mega VID/PID, or set explicitly e.g. "COM3"
SERIAL_PORT = "auto"
SERIAL_BAUD = 115200

# --------------- Detection ---------------
# Use pose model for shoulder-based aiming (yolov8n-pose.pt).
# Falls back to bbox center if shoulders not visible.
YOLO_MODEL_PATH        = "yolov8n-pose.pt"
CONFIDENCE_THRESHOLD   = 0.5

# --------------- Gearing ---------------
PAN_GEAR_RATIO   = 98.0 / 18.0   # ≈ 5.4444 — motor revs per camera rev
TILT_GEAR_RATIO  = 20.0           # worm gear
STEPS_PER_REV    = 200            # NEMA 17, 1.8° step

# Set each to match the physical MODE jumpers on the DRV8825 for that axis:
#   All LOW (no jumpers) -> 1  (full step)
#   M0 HIGH              -> 2  (half step)
#   M0+M1 HIGH           -> 8  (1/8 step)
#   M2 HIGH              -> 16 (1/16 step)
PAN_MICROSTEP_MULTIPLIER  = 8   # M0+M1 pulled HIGH
TILT_MICROSTEP_MULTIPLIER = 1   # no jumpers yet

# Invert motor direction if the axis moves the wrong way.
# Change to True if the axis drives away from centre instead of toward it.
PAN_INVERT  = False
TILT_INVERT = True

# Set to False to send zero tilt velocity every frame (axis physically disabled)
TILT_ENABLED = True

# Swap pan/tilt: set True only if motors are wired to opposite drivers.
SWAP_PAN_TILT = False

# --------------- PD defaults ---------------
# Pan axis
PAN_KP  = 250.0
PAN_KD  = 4.0

# Tilt axis
TILT_KP = 600.0
TILT_KD = 4.0

# Maximum output velocity (deg/sec) before steps/sec conversion
MAX_PAN_VELOCITY  = 1200.0
MAX_TILT_VELOCITY = 300.0

# --------------- Deadzone ---------------
# Point-target mode: deadzone is effectively zero so the controller always
# drives toward the exact red dot (person center), not just to the zone edge.
# A tiny non-zero value avoids floating-point noise at perfect centre.
DEADZONE = 0.001
DEADZONE_CENTER_PULL = 0.0
DYNAMIC_DEADZONE = True           # scales deadzone with target proximity

# Threshold (normalized) used for "centered" detection and motor hysteresis.
# Replaces the old deadzone-based check. ~0.01 = 1% of frame half-width (~6px on 1280).
CENTERING_THRESHOLD = 0.01

# Dynamic deadzone config kept for reference but unused when DYNAMIC_DEADZONE=False.
DYNAMIC_DEADZONE_MIN_PX  = 4
DYNAMIC_DEADZONE_MAX_PX  = 10

# --------------- Gain curve ---------------
# 1.0 = linear: velocity ∝ error, so arrow length correlates to commanded speed.
# Approach decelerates naturally as the crosshairs get closer (error shrinks).
GAIN_CURVE_EXPONENT = 1.0

# --------------- Smoothing ---------------
# Exponential moving average on PID output (0 = no smoothing, 1 = frozen)
# Higher = faster response; lower = more smoothing/lag.
# Lower values give a natural acceleration ramp; 0.6–0.7 balances speed and smoothness.
EMA_ALPHA = 0.85

# --------------- Target management ---------------
# Frames without seeing the tracked target before switching to nearest person
TARGET_LOST_FRAMES = 30

# Frames with no detections at all before de-energizing motors (idle state)
IDLE_TIMEOUT_FRAMES = 30

# Frames the target must stay inside the deadzone before de-energizing (0 = disabled)
# Set to 0 for "taut string" feel: motors stay energized the entire time a person
# is visible so there is zero re-enable lag when they start moving.
# Trade-off: drivers run warmer — add heatsinks if sessions are > ~5 min.
CENTER_IDLE_FRAMES = 0

# --------------- Web server ---------------
WEB_SERVER_PORT = 5000
WEB_SERVER_HOST = "0.0.0.0"

# --------------- Firing ---------------
# "off"    — firing disabled entirely (safe default)
# "auto"   — pin 12 goes HIGH automatically when crosshair enters the fire zone
# "manual" — pin 12 goes HIGH only while Spacebar is held (target must be present)
# Press F at runtime to cycle: off → auto → manual → off
FIRE_MODE = "off"

# Fire zone: how many shoulder-widths (±) count as "on target".
# 1.0 = exactly one shoulder-width radius from centre → fires when aim is within
# the person's shoulder span. Increase for looser trigger, decrease for tighter.
FIRE_ZONE_SHOULDER_SCALE = 1.0

# Normalized fallback fire-zone half-width used when shoulder keypoints aren't visible.
# 0.10 ≈ 10% of frame half-width (roughly 64 px on a 1280-wide frame).
FIRE_ZONE_FALLBACK = 0.10

# Duration (seconds) of each fire pulse sent to pin 12.
# The pin goes HIGH for this long, then returns LOW.
# Note: relay may need ≥0.25s to trigger; reduce if mechanism misfires.
FIRE_PULSE_DURATION = 0.2

# Minimum seconds between successive auto-fire pulses (0.2 = 5 shots/sec).
FIRE_AUTO_INTERVAL = 0.2

# --------------- Distance-adaptive gain ---------------
# When the target is far away (small bounding box), the normalised error is
# inherently small, causing sluggish tracking.  This multiplies the error by
# a scale factor that increases as the person gets farther away, so the PID
# controller sees a consistent "effective" error regardless of distance.
#
# DISTANCE_GAIN_REF_HEIGHT: bounding-box height (fraction of frame) at which
#   the multiplier equals 1.0.  0.35 ≈ person filling ~35% of frame height.
# DISTANCE_GAIN_MAX: cap on the multiplier (prevents runaway on tiny detections).
# Set DISTANCE_GAIN_REF_HEIGHT = 0.0 to disable.
DISTANCE_GAIN_REF_HEIGHT = 0.35
DISTANCE_GAIN_MAX        = 8.0

# --------------- Distance estimation ---------------
# Bbox-height based with heavy temporal smoothing + rate limiting.
# Immune to pose changes (turning, squatting) via slow EMA + max slew rate.
DISTANCE_PERSON_HEIGHT_M  = 1.7    # assumed average person height
DISTANCE_EMA_ALPHA        = 0.15   # heavy smoothing — rejects pose-induced spikes
DISTANCE_MAX_RATE_MPS     = 1.5    # max distance change (m/s) — nobody walks faster
DISTANCE_CLOSE_M          = 3.0    # < 3m = CLOSE
DISTANCE_FAR_M            = 8.0    # > 8m = FAR, between = MEDIUM

# --------------- Predictive lead ---------------
# When a target is moving, add a lead offset to the aim point so the
# tracker anticipates where the person will be, rather than always lagging.
# LEAD_GAIN: how many frames ahead to aim (scales with target velocity).
#   0.0 = disabled (pure reactive), 3.0 = lead by ~3 frames of target motion.
# LEAD_EMA: smoothing on the velocity estimate (0.0–1.0, higher = less noise).
SNAP_THRESHOLD  = 0.08   # normalized error above this → bypass EMA, instant snap

LEAD_GAIN       = 0.0    # (legacy, unused) frames ahead to aim
LEAD_ACCEL_GAIN = 0.0    # (legacy, unused)
LEAD_EMA        = 0.4    # (legacy, unused)

# --------------- Latency-compensating prediction ---------------
PREDICTION_ENABLED    = True
PREDICTION_VEL_ALPHA  = 0.7    # velocity EMA (higher = more responsive, noisier)
PREDICTION_EXTRA_MS   = 40     # assumed inference latency added to measured frame age
PREDICTION_MAX_SEC    = 0.15   # cap prediction horizon (prevents runaway overshoot)
PREDICTION_MIN_FRAMES = 3      # frames of stable velocity before prediction kicks in

# --------------- Auto-brightness ---------------
# Hardware: ask the camera driver to manage exposure automatically.
# Most USB cameras on Windows (DirectShow) respond to this.
CAMERA_AUTO_EXPOSURE = True

# Software fallback: normalize each frame so mean luminance hits this target.
# 0 = disabled.  100–120 works well for YOLO pose detection indoors.
AUTO_BRIGHTNESS_TARGET = 110  # 0-255; set to 0 to disable software normalization

# --------------- Misc ---------------
# Delay (ms) after motor re-enable before sending velocity commands
MOTOR_ENABLE_DELAY_MS = 20

# Manual jog speed (deg/sec) used when WASD keys are held
MANUAL_JOG_VELOCITY = 80.0

# --------------- Hand/Head control ---------------
# Control the tracker using hand position or head direction via the built-in webcam.
# Toggle with H key: PID → Hand → Head → PID
#
# Position-based: the tracker mirrors the absolute head/hand direction.
# Head center or hand center → turret center.  Full range maps [-1,1] to ±RANGE degrees.
HAND_CONTROL_PAN_RANGE  = 30.0   # ±degrees: hand/head full-left/right maps to this angle
HAND_CONTROL_TILT_RANGE = 60.0   # ±degrees: hand/head full-up/down maps to this angle
HAND_CONTROL_KP         = 6.0    # proportional gain for position→velocity (higher = faster)
HAND_CONTROL_MAX_VEL    = 300.0  # max velocity (deg/sec) when driving to target position
HAND_CONTROL_DEADZONE   = 0.12   # normalized deadzone — ignore small movements near center
HAND_CONTROL_EMA        = 0.5    # smoothing on hand/head position (0–1, higher = faster)
HAND_CONTROL_CAM_INDEX  = 0      # built-in webcam index (0 = first/internal camera)
HAND_CONTROL_FPS        = 15     # detection rate limit (Hz)
