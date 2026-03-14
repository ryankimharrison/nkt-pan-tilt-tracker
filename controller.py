"""
controller.py — Dual-axis PD motion controller.

Each axis (pan, tilt) has an independent PD instance.

Input  : normalized error in [-1.0, +1.0]
Output : target angular velocity in degrees/second

The controller is resolution-independent because the error is always
normalized to frame dimensions before being fed in.
"""

from __future__ import annotations

import time
import threading
import config


class PIDAxis:
    """
    Single-axis PD controller with:
    - Deadzone (error below threshold → treated as zero)
    - Output velocity clamping
    - Exponential moving average (EMA) smoothing on output
    """

    def __init__(
        self,
        kp: float,
        kd: float,
        max_velocity: float,
        ema_alpha: float,
        deadzone: float,
        curve_exponent: float = 1.0,
        deadzone_center_pull: float = 0.0,
    ):
        self._lock = threading.Lock()

        self.kp                   = kp
        self.kd                   = kd
        self.max_velocity         = max_velocity
        self.ema_alpha            = ema_alpha
        self.deadzone             = deadzone
        self.curve_exponent       = curve_exponent
        self.deadzone_center_pull = deadzone_center_pull

        self._prev_error = 0.0
        self._prev_time  = time.perf_counter()
        self._ema_output = 0.0

    def reset(self):
        with self._lock:
            self._prev_error = 0.0
            self._prev_time  = time.perf_counter()
            self._ema_output = 0.0

    def compute(self, error: float) -> float:
        """
        Compute PD output for the given normalized error.

        Parameters
        ----------
        error : float
            Normalized error in [-1.0, +1.0].

        Returns
        -------
        float
            Target angular velocity in degrees/second.
        """
        now = time.perf_counter()
        with self._lock:
            dt = now - self._prev_time
            self._prev_time = now

            # Apply deadzone (soft: gentle pull toward center when inside)
            if abs(error) < self.deadzone:
                if self.deadzone_center_pull > 0:
                    error = error * self.deadzone_center_pull
                else:
                    error = 0.0

            # Non-linear gain curve: gentle near centre, snappy at frame edge.
            # error^1 = linear; error^2 = quadratic snap; values 1.5–2.5 feel best.
            if error != 0.0 and self.curve_exponent != 1.0:
                sign  = 1.0 if error > 0.0 else -1.0
                error = sign * (abs(error) ** self.curve_exponent)

            # Proportional
            p_term = self.kp * error

            # Derivative — skip if dt is too small (avoids catastrophic spikes
            # from timer resolution limits; Windows time.monotonic can return dt=0)
            if dt >= 0.002:   # require at least 2 ms between samples
                d_term = self.kd * (error - self._prev_error) / dt
            else:
                d_term = 0.0
            self._prev_error = error

            # Raw PD output (deg/sec)
            raw = p_term + d_term

            # Clamp to max velocity
            raw = max(-self.max_velocity, min(self.max_velocity, raw))

            # EMA smoothing — bypass when error is large for instant snap
            if abs(error) > config.SNAP_THRESHOLD:
                alpha = 1.0   # no smoothing — instant response
            else:
                alpha = self.ema_alpha
            self._ema_output = alpha * raw + (1.0 - alpha) * self._ema_output

            return self._ema_output

    def update_gains(
        self,
        kp: float | None = None,
        kd: float | None = None,
        max_velocity: float | None = None,
        ema_alpha: float | None = None,
        deadzone: float | None = None,
        curve_exponent: float | None = None,
        deadzone_center_pull: float | None = None,
    ):
        """Update one or more gains at runtime (called by sliders / Flask)."""
        with self._lock:
            if kp                   is not None: self.kp                   = kp
            if kd                   is not None: self.kd                   = kd
            if max_velocity         is not None: self.max_velocity         = max_velocity
            if ema_alpha            is not None: self.ema_alpha            = ema_alpha
            if deadzone             is not None: self.deadzone             = deadzone
            if curve_exponent       is not None: self.curve_exponent       = curve_exponent
            if deadzone_center_pull is not None: self.deadzone_center_pull = deadzone_center_pull


class MotionController:
    """
    Dual-axis (pan + tilt) PD controller.

    Usage
    -----
    ctrl = MotionController()

    # Each frame:
    pan_dps, tilt_dps = ctrl.compute(pan_error, tilt_error)

    # From slider callback:
    ctrl.update_params(pan_kp=90.0, deadzone=0.03)
    """

    def __init__(self):
        self.pan_gain  = 1.0   # multiplier on pan output (slider: 0.1–2.0)
        self.tilt_gain = 1.0   # multiplier on tilt output
        self.pan = PIDAxis(
            kp=config.PAN_KP,
            kd=config.PAN_KD,
            max_velocity=config.MAX_PAN_VELOCITY,
            ema_alpha=config.EMA_ALPHA,
            deadzone=config.DEADZONE,
            curve_exponent=config.GAIN_CURVE_EXPONENT,
            deadzone_center_pull=config.DEADZONE_CENTER_PULL,
        )
        self.tilt = PIDAxis(
            kp=config.TILT_KP,
            kd=config.TILT_KD,
            max_velocity=config.MAX_TILT_VELOCITY,
            ema_alpha=config.EMA_ALPHA,
            deadzone=config.DEADZONE,
            curve_exponent=config.GAIN_CURVE_EXPONENT,
            deadzone_center_pull=config.DEADZONE_CENTER_PULL,
        )

    def compute(self, pan_error: float, tilt_error: float) -> tuple[float, float]:
        pan_vel  = self.pan.compute(pan_error)  * self.pan_gain
        tilt_vel = self.tilt.compute(tilt_error) * self.tilt_gain
        return pan_vel, tilt_vel

    def reset(self):
        self.pan.reset()
        self.tilt.reset()

    def update_params(
        self,
        pan_kp: float | None            = None,
        pan_kd: float | None            = None,
        tilt_kp: float | None           = None,
        tilt_kd: float | None           = None,
        max_pan_velocity: float | None  = None,
        max_tilt_velocity: float | None = None,
        pan_gain: float | None           = None,
        tilt_gain: float | None         = None,
        ema_alpha: float | None         = None,
        deadzone: float | None          = None,
        curve_exponent: float | None    = None,
        deadzone_center_pull: float | None = None,
    ):
        """Update any subset of parameters on either/both axes."""
        if pan_gain  is not None: self.pan_gain  = max(0.1, min(3.0, pan_gain))
        if tilt_gain is not None: self.tilt_gain = max(0.1, min(3.0, tilt_gain))
        self.pan.update_gains(
            kp=pan_kp, kd=pan_kd,
            max_velocity=max_pan_velocity,
            ema_alpha=ema_alpha,
            deadzone=deadzone,
            curve_exponent=curve_exponent,
            deadzone_center_pull=deadzone_center_pull,
        )
        self.tilt.update_gains(
            kp=tilt_kp, kd=tilt_kd,
            max_velocity=max_tilt_velocity,
            ema_alpha=ema_alpha,
            deadzone=deadzone,
            curve_exponent=curve_exponent,
            deadzone_center_pull=deadzone_center_pull,
        )

    def get_params(self) -> dict:
        """Return current parameters as a dict (for Flask status endpoint)."""
        return {
            "pan_kp":            self.pan.kp,
            "pan_kd":            self.pan.kd,
            "pan_gain":          self.pan_gain,
            "tilt_kp":           self.tilt.kp,
            "tilt_kd":           self.tilt.kd,
            "tilt_gain":         self.tilt_gain,
            "max_pan_velocity":  self.pan.max_velocity,
            "max_tilt_velocity": self.tilt.max_velocity,
            "ema_alpha":         self.pan.ema_alpha,
            "deadzone":          self.pan.deadzone,
        }
