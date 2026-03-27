"""
serial_comm.py — Thread-safe Arduino serial communication.

Velocity commands (V) are fire-and-forget.
Enable (E), stop (S), and ping (P) commands block and wait for a reply.
"""

import threading
import time
import serial
import serial.tools.list_ports
import config


# USB VID:PID for Arduino Mega 2560 (official and CH340 clones)
_ARDUINO_VIDS = {0x2341, 0x1A86, 0x0403}
_ARDUINO_MEGA_PID = {0x0010, 0x7523, 0x6001}


def _steps_per_sec(deg_per_sec: float, gear_ratio: float, microstep: int) -> float:
    """Convert angular velocity (deg/sec) to stepper steps/sec."""
    return (deg_per_sec * gear_ratio * config.STEPS_PER_REV * microstep) / 360.0


def steps_to_degrees(steps: int, gear_ratio: float, microstep: int) -> float:
    """Convert step count to output-shaft degrees."""
    return (steps * 360.0) / (gear_ratio * config.STEPS_PER_REV * microstep)


def degrees_to_steps(degrees: float, gear_ratio: float, microstep: int) -> float:
    """Convert output-shaft degrees to step count."""
    return (degrees * gear_ratio * config.STEPS_PER_REV * microstep) / 360.0


def _auto_detect_port() -> str | None:
    """Scan serial ports and return the first one that looks like an Arduino."""
    for port_info in serial.tools.list_ports.comports():
        vid = port_info.vid
        pid = port_info.pid
        if vid in _ARDUINO_VIDS or (vid is not None and pid in _ARDUINO_MEGA_PID):
            return port_info.device
    # Fallback: return the first available port (user can override in config)
    ports = list(serial.tools.list_ports.comports())
    if ports:
        return ports[0].device
    return None


class SerialComm:
    """
    Manages serial communication with the Arduino.

    Public API
    ----------
    send_velocity(pan_dps, tilt_dps)   — fire and forget
    enable_motors()                     — blocks for OK
    disable_motors()                    — blocks for OK
    emergency_stop()                    — blocks for OK
    ping()                              — blocks for PONG, returns True/False
    close()
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ser: serial.Serial | None = None
        self._connected = False
        self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self):
        port = config.SERIAL_PORT
        if port == "auto":
            port = _auto_detect_port()
            if port is None:
                print("[SerialComm] WARNING: No Arduino found. Running in simulation mode.")
                return

        try:
            self._ser = serial.Serial(
                port=port,
                baudrate=config.SERIAL_BAUD,
                timeout=1.0,
            )
            time.sleep(2.0)  # Arduino resets on serial connect; wait for bootloader
            self._ser.reset_input_buffer()
            self._connected = True
            print(f"[SerialComm] Connected to Arduino on {port}")
        except serial.SerialException as exc:
            print(f"[SerialComm] ERROR: Could not open {port}: {exc}")
            self._ser = None
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Internal send / receive
    # ------------------------------------------------------------------

    def _send_raw(self, data: bytes):
        """Write bytes to serial (must be called with lock held)."""
        if self._ser and self._connected:
            try:
                self._ser.write(data)
            except serial.SerialException as exc:
                print(f"[SerialComm] Write error: {exc}")
                self._connected = False

    def _wait_reply(self, expected: bytes, timeout: float = 0.3) -> bool:
        """
        Block until expected reply arrives or timeout.
        Must be called with lock held.
        Default timeout reduced to 0.3s to prevent display freezes.
        """
        if not (self._ser and self._connected):
            return False
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            try:
                chunk = self._ser.read(self._ser.in_waiting or 1)
                buf += chunk
                if expected in buf:
                    return True
            except serial.SerialException as exc:
                print(f"[SerialComm] Read error: {exc}")
                self._connected = False
                return False
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_velocity(self, pan_dps: float, tilt_dps: float):
        """
        Convert deg/sec to steps/sec and send a V command.
        Fire-and-forget — does not wait for acknowledgment.
        """
        pan_sps  = _steps_per_sec(pan_dps,  config.PAN_GEAR_RATIO,  config.PAN_MICROSTEP_MULTIPLIER)
        tilt_sps = _steps_per_sec(tilt_dps, config.TILT_GEAR_RATIO, config.TILT_MICROSTEP_MULTIPLIER)
        if config.PAN_INVERT:   pan_sps  = -pan_sps
        if config.TILT_INVERT:  tilt_sps = -tilt_sps
        if not config.TILT_ENABLED: tilt_sps = 0.0
        if getattr(config, "SWAP_PAN_TILT", False):
            pan_sps, tilt_sps = tilt_sps, pan_sps
        cmd = f"V{pan_sps:.2f},{tilt_sps:.2f}\n".encode()
        with self._lock:
            self._send_raw(cmd)

    def enable_motors(self) -> bool:
        """Send E0 (fire-and-forget for minimal latency)."""
        with self._lock:
            self._send_raw(b"E0\n")
        return True

    def disable_motors(self) -> bool:
        """Send E1 and wait for OK. Returns True on success."""
        with self._lock:
            self._send_raw(b"E1\n")
            ok = self._wait_reply(b"OK")
        return ok

    def emergency_stop(self) -> bool:
        """Send S (stop + de-energize) and wait for OK."""
        with self._lock:
            self._send_raw(b"S\n")
            ok = self._wait_reply(b"OK")
        return ok

    def ping(self) -> bool:
        """Send P and wait for PONG. Returns True if alive."""
        with self._lock:
            self._send_raw(b"P\n")
            ok = self._wait_reply(b"PONG")
        return ok

    def zero_position(self) -> bool:
        """Send Z — resets the Arduino position counters to 0."""
        with self._lock:
            self._send_raw(b"Z\n")
            ok = self._wait_reply(b"OK")
        return ok

    def calibrate_tilt_limit(self) -> bool:
        """Send C — mark current tilt position as upper limit."""
        with self._lock:
            self._send_raw(b"C\n")
            ok = self._wait_reply(b"OK")
        return ok

    def calibrate_tilt_lower_limit(self) -> bool:
        """Send B — mark current tilt position as lower limit."""
        with self._lock:
            self._send_raw(b"B\n")
            ok = self._wait_reply(b"OK")
        return ok

    def query_positions(self) -> dict | None:
        """Send Q — returns dict with pan/tilt positions, limits, and cal flags."""
        with self._lock:
            self._send_raw(b"Q\n")
            if not (self._ser and self._connected):
                return None
            deadline = time.time() + 1.0
            buf = b""
            while time.time() < deadline:
                try:
                    chunk = self._ser.read(self._ser.in_waiting or 1)
                    buf += chunk
                    if b"\n" in buf:
                        for line in buf.split(b"\n"):
                            line = line.strip()
                            if line.startswith(b"T"):
                                parts = line[1:].split(b",")
                                if len(parts) == 9:
                                    return {
                                        "tilt_position": int(parts[0]),
                                        "tilt_upper_limit": int(parts[1]),
                                        "tilt_upper_calibrated": parts[2] == b"1",
                                        "tilt_lower_limit": int(parts[3]),
                                        "tilt_lower_calibrated": parts[4] == b"1",
                                        "pan_position": int(parts[5]),
                                        "pan_upper_limit": int(parts[6]),
                                        "pan_lower_limit": int(parts[7]),
                                        "pan_calibrated": parts[8] == b"1",
                                    }
                except serial.SerialException:
                    self._connected = False
                    return None
        return None

    def calibrate_from_center(self, pan_upper: int, pan_lower: int,
                              tilt_upper: int, tilt_lower: int) -> bool:
        """Send L — set pan + tilt limits relative to current positions."""
        cmd = f"L{pan_upper},{pan_lower},{tilt_upper},{tilt_lower}\n".encode()
        with self._lock:
            self._send_raw(cmd)
            ok = self._wait_reply(b"OK")
        return ok

    def set_signal(self, high: bool) -> None:
        """Send G1 / G0 — drive pin 12 HIGH or LOW.  Fire-and-forget."""
        cmd = b"G1\n" if high else b"G0\n"
        with self._lock:
            self._send_raw(cmd)

    def close(self):
        """Graceful shutdown: stop, disable, close port."""
        if self._ser and self._connected:
            try:
                with self._lock:
                    self._send_raw(b"S\n")
                    time.sleep(0.1)
                    self._send_raw(b"E1\n")
                    time.sleep(0.1)
            except Exception:
                pass
            try:
                self._ser.close()
            except Exception:
                pass
        self._connected = False
        print("[SerialComm] Serial port closed.")
