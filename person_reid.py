# person_reid.py — Person Re-Identification using OSNet embeddings
#
# Provides cross-camera and re-entry person matching by computing
# 512-dim appearance embeddings from body crops.
# OSNet-x0.25 architecture implemented from scratch (no torchreid dependency).

import threading
import time
import json
import os
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    print("[ReID] WARNING: torch not installed — ReID disabled")


_EMBED_DIM = 512
_INPUT_SIZE = (256, 128)  # H, W — standard person ReID input

# Matching thresholds
MATCH_THRESHOLD = 0.45      # cosine similarity above this = same person
GALLERY_MAX_AGE = 300.0     # seconds before a gallery entry expires (5 min)
GALLERY_UPDATE_INTERVAL = 0.5  # seconds between embedding updates for tracked person
GALLERY_MAX_SIZE = 20       # max people in gallery
NAMES_FILE = "person_names.json"  # persists names across sessions

# Weights file — MSMT17 combineall (best cross-domain generalization)
_WEIGHTS_FILE = "osnet_x0_25_msmt17.pth"


# ======================================================================
# OSNet-x0.25 Architecture (Zhou et al., "Omni-Scale Feature Learning
# for Person Re-Identification")
# ======================================================================

if _HAS_TORCH:

    class _ConvLayer(nn.Module):
        """Conv2d + BatchNorm + ReLU."""
        def __init__(self, in_c, out_c, kernel, stride=1, padding=0, groups=1):
            super().__init__()
            self.conv = nn.Conv2d(in_c, out_c, kernel, stride=stride,
                                  padding=padding, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_c)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class _Conv1x1(nn.Module):
        """1x1 Conv + BN + ReLU."""
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv = nn.Conv2d(in_c, out_c, 1, bias=False)
            self.bn = nn.BatchNorm2d(out_c)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class _Conv1x1Linear(nn.Module):
        """1x1 Conv + BN (no ReLU)."""
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv = nn.Conv2d(in_c, out_c, 1, bias=False)
            self.bn = nn.BatchNorm2d(out_c)

        def forward(self, x):
            return self.bn(self.conv(x))

    class _LightConv3x3(nn.Module):
        """Lightweight 3x3: pointwise 1x1 + depthwise 3x3 + BN + ReLU."""
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
            self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False)
            self.bn = nn.BatchNorm2d(out_c)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return self.relu(self.bn(x))

    class _ChannelGate(nn.Module):
        """SE-style channel attention: GAP -> FC -> ReLU -> FC -> Sigmoid."""
        def __init__(self, channels, reduction=16):
            super().__init__()
            mid = max(1, channels // reduction)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(channels, mid, 1, bias=True)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Conv2d(mid, channels, 1, bias=True)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            g = self.gap(x)
            g = self.relu(self.fc1(g))
            g = self.sigmoid(self.fc2(g))
            return x * g

    class _OSBlock(nn.Module):
        """Omni-Scale block with 4 parallel multi-scale streams + channel gating."""
        def __init__(self, in_c, out_c, bottleneck_reduction=4, gate_reduction=16):
            super().__init__()
            mid = out_c // bottleneck_reduction
            self.conv1 = _Conv1x1(in_c, mid)
            # 4 parallel streams with increasing receptive field
            self.conv2a = nn.Sequential(_LightConv3x3(mid, mid))
            self.conv2b = nn.Sequential(_LightConv3x3(mid, mid), _LightConv3x3(mid, mid))
            self.conv2c = nn.Sequential(_LightConv3x3(mid, mid), _LightConv3x3(mid, mid),
                                        _LightConv3x3(mid, mid))
            self.conv2d = nn.Sequential(_LightConv3x3(mid, mid), _LightConv3x3(mid, mid),
                                        _LightConv3x3(mid, mid), _LightConv3x3(mid, mid))
            self.gate_a = _ChannelGate(mid, gate_reduction)
            self.gate_b = _ChannelGate(mid, gate_reduction)
            self.gate_c = _ChannelGate(mid, gate_reduction)
            self.gate_d = _ChannelGate(mid, gate_reduction)
            self.conv3 = _Conv1x1Linear(mid, out_c)
            self.downsample = _Conv1x1Linear(in_c, out_c) if in_c != out_c else None
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            identity = x
            x1 = self.conv1(x)
            x2 = (self.gate_a(self.conv2a(x1)) +
                  self.gate_b(self.conv2b(x1)) +
                  self.gate_c(self.conv2c(x1)) +
                  self.gate_d(self.conv2d(x1)))
            x3 = self.conv3(x2)
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.relu(x3 + identity)

    class OSNet_x0_25(nn.Module):
        """OSNet with 0.25x width multiplier. Channels: [16, 64, 96, 128], feature_dim=512."""
        def __init__(self, num_classes=1, feature_dim=512):
            super().__init__()
            channels = [16, 64, 96, 128]

            # Stem
            self.conv1 = _ConvLayer(3, channels[0], 7, stride=2, padding=3)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

            # Stage 1: 2x OSBlock(16->64) + transition
            self.conv2 = nn.Sequential(
                _OSBlock(channels[0], channels[1]),
                _OSBlock(channels[1], channels[1]),
            )
            self.pool2 = nn.Sequential(
                _Conv1x1(channels[1], channels[1]),
                nn.AvgPool2d(2, stride=2),
            )

            # Stage 2: 2x OSBlock(64->96) + transition
            self.conv3 = nn.Sequential(
                _OSBlock(channels[1], channels[2]),
                _OSBlock(channels[2], channels[2]),
            )
            self.pool3 = nn.Sequential(
                _Conv1x1(channels[2], channels[2]),
                nn.AvgPool2d(2, stride=2),
            )

            # Stage 3: 2x OSBlock(96->128), no transition
            self.conv4 = nn.Sequential(
                _OSBlock(channels[2], channels[3]),
                _OSBlock(channels[3], channels[3]),
            )

            self.conv5 = _Conv1x1(channels[3], channels[3])
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)

            # FC embedding layer
            self.fc = nn.Sequential(
                nn.Linear(channels[3], feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
            )

            # Classifier (unused at inference)
            self.classifier = nn.Linear(feature_dim, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.global_avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x


# ======================================================================
# PersonReID manager
# ======================================================================

class PersonReID:
    """
    Manages a gallery of person appearance embeddings for cross-camera
    and re-entry matching.

    Usage:
        reid = PersonReID()
        reid.start()  # loads model in background

        # When a new person is detected:
        embedding = reid.compute_embedding(frame, bbox)
        match_id = reid.match(embedding)  # returns existing ID or None

        # Register/update:
        reid.register(track_id, embedding)
        reid.update(track_id, embedding)
    """

    def __init__(self):
        self._model = None
        self._device = None
        self._ready = False
        self._use_histogram = False  # fallback if OSNet fails
        self._lock = threading.RLock()

        # Gallery: {person_id: {"embeddings": [list of np arrays], "last_seen": time, "last_update": time}}
        self._gallery: dict[int, dict] = {}
        self._next_global_id = 1  # monotonically increasing global person ID

        # Person names: {global_id: "name"} — persisted to JSON
        self._names: dict[int, str] = {}
        # Saved embeddings for named people: {name: [list of embeddings as lists]}
        self._saved_profiles: dict[str, list] = {}
        self._load_names()

    @property
    def ready(self) -> bool:
        return self._ready

    def start(self):
        """Load model in background thread."""
        if not _HAS_TORCH:
            return
        t = threading.Thread(target=self._load_model, daemon=True, name="ReID-Init")
        t.start()

    def _load_model(self):
        """Load OSNet-x0.25 from local weights, or fall back to histogram."""
        _weights_path = os.path.join(os.path.dirname(__file__), _WEIGHTS_FILE)
        # Also check the old filename
        _alt_path = os.path.join(os.path.dirname(__file__), "osnet_x0_25.pth")

        weights_file = None
        if os.path.exists(_weights_path):
            weights_file = _weights_path
        elif os.path.exists(_alt_path):
            weights_file = _alt_path

        if weights_file is not None:
            try:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = OSNet_x0_25()

                # Load state dict, stripping "module." prefix if present
                state = torch.load(weights_file, map_location=self._device, weights_only=True)
                if "state_dict" in state:
                    state = state["state_dict"]
                clean = {}
                for k, v in state.items():
                    k = k.replace("module.", "")
                    clean[k] = v

                # Ignore classifier weights (different num_classes)
                clean = {k: v for k, v in clean.items()
                         if not k.startswith("classifier")}

                model.load_state_dict(clean, strict=False)
                model.eval()
                model.to(self._device)

                self._model = model
                self._use_histogram = False
                self._ready = True
                print(f"[ReID] OSNet-x0.25 loaded on {self._device} "
                      f"({sum(p.numel() for p in model.parameters()):,} params)")
            except Exception as e:
                print(f"[ReID] OSNet load failed: {e}")
                import traceback
                traceback.print_exc()
                self._model = None
                self._use_histogram = True
                self._ready = True
        else:
            print(f"[ReID] No OSNet weights found ({_WEIGHTS_FILE}) — using color histogram ReID")
            self._model = None
            self._use_histogram = True
            self._ready = True

    def compute_embedding(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        """
        Compute an embedding from a person crop.
        Uses OSNet (512-dim) if available, falls back to color histogram (180-dim).
        Returns normalized embedding vector or None if failed.
        """
        if not self._ready:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp and validate
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 10 or y2 - y1 < 20:
            return None

        crop = frame[y1:y2, x1:x2]

        if self._use_histogram or self._model is None:
            return self._histogram_embedding(crop)

        # OSNet embedding
        crop_resized = cv2.resize(crop, (_INPUT_SIZE[1], _INPUT_SIZE[0]))
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        crop_norm = crop_rgb.astype(np.float32) / 255.0
        crop_norm = (crop_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        tensor = torch.from_numpy(crop_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self._device)

        with torch.no_grad():
            features = self._model(tensor)

        embedding = features.cpu().numpy().flatten()
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    @staticmethod
    def _histogram_embedding(crop: np.ndarray) -> np.ndarray | None:
        """Fallback: HSV color histogram embedding (180 bins for hue, normalized)."""
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Use middle 60% of crop (torso region, skip head/feet)
        ch, cw = hsv.shape[:2]
        y_start = int(ch * 0.2)
        y_end = int(ch * 0.8)
        torso = hsv[y_start:y_end, :]
        # Hue histogram (clothing color)
        h_hist = cv2.calcHist([torso], [0], None, [90], [0, 180]).flatten()
        # Saturation histogram (color intensity)
        s_hist = cv2.calcHist([torso], [1], None, [45], [0, 256]).flatten()
        # Value histogram (brightness)
        v_hist = cv2.calcHist([torso], [2], None, [45], [0, 256]).flatten()
        embedding = np.concatenate([h_hist, s_hist, v_hist])  # 180-dim
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def match(self, embedding: np.ndarray) -> int | None:
        """
        Find the best matching person in the gallery OR saved profiles.
        Returns the person's global ID if match found, None otherwise.
        If matched against a saved profile, creates a gallery entry and assigns the name.
        """
        if embedding is None:
            return None

        self._cleanup_gallery()

        best_id = None
        best_sim = MATCH_THRESHOLD

        with self._lock:
            # Check active gallery
            for pid, entry in self._gallery.items():
                if not entry["embeddings"]:
                    continue
                gallery_emb = np.mean(entry["embeddings"], axis=0)
                sim = float(np.dot(embedding, gallery_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid

        # Check saved profiles (named people from previous sessions)
        if best_id is None:
            for name, saved_embs in self._saved_profiles.items():
                if not saved_embs:
                    continue
                saved_avg = np.mean([np.array(e) for e in saved_embs], axis=0)
                norm = np.linalg.norm(saved_avg)
                if norm > 0:
                    saved_avg = saved_avg / norm
                sim = float(np.dot(embedding, saved_avg))
                if sim > best_sim:
                    best_sim = sim
                    # Create a gallery entry for this saved person
                    with self._lock:
                        global_id = self._next_global_id
                        self._next_global_id += 1
                        self._gallery[global_id] = {
                            "embeddings": [embedding],
                            "last_seen": time.time(),
                            "last_update": time.time(),
                            "byte_track_ids": set(),
                        }
                        self._names[global_id] = name
                    best_id = global_id
                    print(f"[ReID] Recognized saved person: {name} (sim={sim:.2f})")

        return best_id

    def register(self, track_id: int, embedding: np.ndarray) -> int:
        """
        Register a new person or update existing.
        If track_id maps to a known gallery entry, updates it.
        Otherwise creates a new gallery entry.
        Returns the global person ID.
        """
        if embedding is None:
            return track_id

        with self._lock:
            # Check if this track_id is already in gallery
            if track_id in self._gallery:
                self._gallery[track_id]["embeddings"].append(embedding)
                # Keep only last 10 embeddings
                if len(self._gallery[track_id]["embeddings"]) > 10:
                    self._gallery[track_id]["embeddings"] = self._gallery[track_id]["embeddings"][-10:]
                self._gallery[track_id]["last_seen"] = time.time()
                self._gallery[track_id]["last_update"] = time.time()
                return track_id

            # Try to match against gallery first (re-entry detection)
            match_id = self.match(embedding)
            if match_id is not None:
                self._gallery[match_id]["embeddings"].append(embedding)
                if len(self._gallery[match_id]["embeddings"]) > 10:
                    self._gallery[match_id]["embeddings"] = self._gallery[match_id]["embeddings"][-10:]
                self._gallery[match_id]["last_seen"] = time.time()
                self._gallery[match_id]["byte_track_ids"].add(track_id)
                print(f"[ReID] Re-identified! ByteTrack #{track_id} → global P{match_id}")
                return match_id

            # New person — create gallery entry
            global_id = self._next_global_id
            self._next_global_id += 1
            print(f"[ReID] New person: ByteTrack #{track_id} → global P{global_id} (gallery={len(self._gallery)+1})")
            self._gallery[global_id] = {
                "embeddings": [embedding],
                "last_seen": time.time(),
                "last_update": time.time(),
                "byte_track_ids": {track_id},  # track which ByteTrack IDs map to this person
            }
            return global_id

    def update(self, global_id: int, embedding: np.ndarray):
        """Update an existing gallery entry with a fresh embedding."""
        if embedding is None:
            return
        with self._lock:
            if global_id in self._gallery:
                now = time.time()
                if now - self._gallery[global_id]["last_update"] >= GALLERY_UPDATE_INTERVAL:
                    self._gallery[global_id]["embeddings"].append(embedding)
                    if len(self._gallery[global_id]["embeddings"]) > 10:
                        self._gallery[global_id]["embeddings"] = self._gallery[global_id]["embeddings"][-10:]
                    self._gallery[global_id]["last_update"] = now
                self._gallery[global_id]["last_seen"] = now

    def get_gallery_size(self) -> int:
        """Return number of people in gallery."""
        return len(self._gallery)

    def _cleanup_gallery(self):
        """Remove stale entries from gallery."""
        now = time.time()
        with self._lock:
            expired = [pid for pid, entry in self._gallery.items()
                       if now - entry["last_seen"] > GALLERY_MAX_AGE]
            for pid in expired:
                del self._gallery[pid]

            # Enforce max size — remove oldest (but never remove named people)
            if len(self._gallery) > GALLERY_MAX_SIZE:
                sorted_entries = sorted(self._gallery.items(), key=lambda x: x[1]["last_seen"])
                for pid, _ in sorted_entries[:len(self._gallery) - GALLERY_MAX_SIZE]:
                    if pid not in self._names:  # don't expire named people
                        del self._gallery[pid]

    # ------------------------------------------------------------------
    # Name management
    # ------------------------------------------------------------------

    def set_name(self, global_id: int, name: str):
        """Assign a name to a person. If the name already exists in saved profiles,
        merge the current embeddings with the saved ones (same person, better model)."""
        name = name.strip().upper()
        if not name:
            return
        with self._lock:
            self._names[global_id] = name

            # Get current embeddings for this person
            new_embs = []
            if global_id in self._gallery:
                new_embs = [e.tolist() for e in self._gallery[global_id]["embeddings"]]

            if name in self._saved_profiles:
                # Name already exists — merge embeddings (same person seen again)
                existing = self._saved_profiles[name]
                merged = existing + new_embs
                # Keep last 20 embeddings (more data = better matching)
                if len(merged) > 20:
                    merged = merged[-20:]
                self._saved_profiles[name] = merged
                print(f"[ReID] Merged person {global_id} with existing profile '{name}' "
                      f"({len(existing)} + {len(new_embs)} = {len(merged)} embeddings)")
            else:
                # New name — save embeddings
                self._saved_profiles[name] = new_embs
                print(f"[ReID] Named person {global_id} as '{name}' — saved {len(new_embs)} embeddings")

        self._save_names()

    def get_name(self, global_id: int) -> str | None:
        """Get the name of a person by global ID. Returns None if unnamed."""
        return self._names.get(global_id)

    def get_display_name(self, global_id: int) -> str:
        """Get display string: name if set, otherwise P{id}."""
        name = self._names.get(global_id)
        if name:
            return name
        return f"P{global_id}" if global_id >= 0 else "?"

    def _load_names(self):
        """Load saved names and embeddings from JSON file."""
        if not os.path.exists(NAMES_FILE):
            return
        try:
            with open(NAMES_FILE, "r") as f:
                data = json.load(f)
            self._saved_profiles = data.get("profiles", {})
            print(f"[ReID] Loaded {len(self._saved_profiles)} saved person profiles")
        except Exception as e:
            print(f"[ReID] Failed to load names: {e}")

    def _save_names(self):
        """Save names and embeddings to JSON file."""
        try:
            data = {
                "profiles": self._saved_profiles,
            }
            with open(NAMES_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ReID] Failed to save names: {e}")
