"""
web_server.py — Flask debug web UI.

Endpoints:
  GET  /             — HTML dashboard
  GET  /video_feed   — MJPEG stream of the annotated frame
  GET  /status       — JSON status snapshot
  POST /params       — Update PD gains / tracking params
  POST /control      — E-STOP, re-center, toggle motor
"""

from __future__ import annotations
import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
import config


class WebServer:
    def __init__(self, shared, controller):
        self._shared     = shared
        self._controller = controller
        self._app        = Flask(__name__, template_folder="templates",
                                 static_folder="static")
        self._thread: threading.Thread | None = None
        self._register_routes()

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _register_routes(self):
        app    = self._app
        shared = self._shared
        ctrl   = self._controller

        @app.route("/")
        def index():
            params = ctrl.get_params()
            return render_template("index.html", params=params,
                                   port=config.WEB_SERVER_PORT,
                                   lead_gain=config.LEAD_GAIN,
                                   lead_accel_gain=config.LEAD_ACCEL_GAIN)

        @app.route("/video_feed")
        def video_feed():
            return Response(self._gen_frames(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.route("/status")
        def status():
            with shared.status_lock:
                return jsonify(shared.status)

        @app.route("/params", methods=["POST"])
        def params():
            data = request.get_json(force=True)
            kwargs = {}
            _float_fields = [
                "pan_kp", "pan_kd",
                "tilt_kp", "tilt_kd",
                "max_pan_velocity", "max_tilt_velocity",
                "ema_alpha", "deadzone",
            ]
            _int_fields = ["target_lost_frames", "idle_timeout_frames"]

            for f in _float_fields:
                if f in data:
                    kwargs[f] = float(data[f])

            if "target_lost_frames" in data:
                config.TARGET_LOST_FRAMES = max(1, int(data["target_lost_frames"]))
            if "idle_timeout_frames" in data:
                config.IDLE_TIMEOUT_FRAMES = max(1, int(data["idle_timeout_frames"]))
            if "lead_gain" in data:
                config.LEAD_GAIN = max(0.0, float(data["lead_gain"]))
            if "lead_accel_gain" in data:
                config.LEAD_ACCEL_GAIN = max(0.0, float(data["lead_accel_gain"]))

            if kwargs:
                ctrl.update_params(**kwargs)
                with shared.params_lock:
                    if shared.pending_params is None:
                        shared.pending_params = {}
                    shared.pending_params.update(kwargs)

            return jsonify({"ok": True})

        @app.route("/control", methods=["POST"])
        def control():
            data   = request.get_json(force=True)
            action = data.get("action", "")
            if action == "estop":
                shared.estop_event.set()
            elif action == "recenter":
                shared.re_center_event.set()
            elif action == "toggle_motor":
                shared.motor_toggle_event.set()
            elif action == "shutdown":
                shared.shutdown_home_event.set()
            return jsonify({"ok": True})

    # ------------------------------------------------------------------
    # MJPEG generator
    # ------------------------------------------------------------------

    def _gen_frames(self):
        while True:
            with self._shared.annotated_lock:
                frame = self._shared.annotated_frame

            if frame is None:
                time.sleep(0.05)
                continue

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
            time.sleep(1.0 / 30.0)

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(
            target=lambda: self._app.run(
                host=config.WEB_SERVER_HOST,
                port=config.WEB_SERVER_PORT,
                debug=False,
                use_reloader=False,
            ),
            daemon=True,
            name="WebServer",
        )
        self._thread.start()
        print(f"[WebServer] Listening on http://{config.WEB_SERVER_HOST}:{config.WEB_SERVER_PORT}")
