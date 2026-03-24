# person_reid.py — Person Re-Identification using OSNet embeddings
#
# Provides cross-camera and re-entry person matching by computing
# 512-dim appearance embeddings from body crops.

import threading
import time
import json
import os
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    print("[ReID] WARNING: torch not installed — ReID disabled")


_EMBED_DIM = 512
_INPUT_SIZE = (256, 128)  # H, W — standard person ReID input

# Matching thresholds
MATCH_THRESHOLD = 0.55      # cosine similarity above this = same person
GALLERY_MAX_AGE = 30.0      # seconds before a gallery entry expires
GALLERY_UPDATE_INTERVAL = 0.5  # seconds between embedding updates for tracked person
GALLERY_MAX_SIZE = 20       # max people in gallery
NAMES_FILE = "person_names.json"  # persists names across sessions


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
        self._lock = threading.Lock()

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
        """Load OSNet via torch hub (no torchreid dependency)."""
        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Load OSNet from KaiyangZhou's repo via torch hub
            self._model = torch.hub.load(
                "KaiyangZhou/deep-person-reid",
                "osnet_x0_25",
                pretrained=True,
            )
            self._model = self._model.to(self._device)
            self._model.eval()
            # Remove classifier head — we only want features
            if hasattr(self._model, "classifier"):
                self._model.classifier = nn.Identity()
            # Warm up
            _dummy = torch.zeros(1, 3, _INPUT_SIZE[0], _INPUT_SIZE[1]).to(self._device)
            with torch.no_grad():
                self._model(_dummy)
            self._ready = True
            print(f"[ReID] OSNet loaded on {self._device}")
        except Exception as e:
            print(f"[ReID] torch.hub load failed: {e}")
            print("[ReID] Falling back to color histogram ReID")
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
                return match_id

            # New person — create gallery entry
            global_id = self._next_global_id
            self._next_global_id += 1
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
        """Assign a name to a person. Saves their embeddings for future recognition."""
        name = name.strip().upper()
        if not name:
            return
        with self._lock:
            self._names[global_id] = name
            # Save their current embeddings to the profile
            if global_id in self._gallery:
                embs = self._gallery[global_id]["embeddings"]
                self._saved_profiles[name] = [e.tolist() for e in embs]
        self._save_names()
        print(f"[ReID] Named person {global_id} as '{name}' — saved profile with embeddings")

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
