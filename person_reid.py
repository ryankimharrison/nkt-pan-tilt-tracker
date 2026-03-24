# person_reid.py — Person Re-Identification using OSNet embeddings
#
# Provides cross-camera and re-entry person matching by computing
# 512-dim appearance embeddings from body crops.

import threading
import time
import numpy as np
import cv2

try:
    import torch
    import torchreid
    _HAS_TORCHREID = True
except ImportError:
    _HAS_TORCHREID = False
    print("[ReID] WARNING: torchreid not installed — ReID disabled")


# OSNet-x0_25: ~2MB model, very fast (~2-3ms per crop on GPU)
_MODEL_NAME = "osnet_x0_25"
_EMBED_DIM = 512
_INPUT_SIZE = (256, 128)  # H, W — standard person ReID input

# Matching thresholds
MATCH_THRESHOLD = 0.55      # cosine similarity above this = same person
GALLERY_MAX_AGE = 30.0      # seconds before a gallery entry expires
GALLERY_UPDATE_INTERVAL = 0.5  # seconds between embedding updates for tracked person
GALLERY_MAX_SIZE = 20       # max people in gallery


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
        self._lock = threading.Lock()

        # Gallery: {person_id: {"embeddings": [list of np arrays], "last_seen": time, "last_update": time}}
        self._gallery: dict[int, dict] = {}
        self._next_global_id = 1  # monotonically increasing global person ID

    @property
    def ready(self) -> bool:
        return self._ready

    def start(self):
        """Load model in background thread."""
        if not _HAS_TORCHREID:
            return
        t = threading.Thread(target=self._load_model, daemon=True, name="ReID-Init")
        t.start()

    def _load_model(self):
        """Load OSNet model (called from background thread)."""
        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = torchreid.models.build_model(
                name=_MODEL_NAME,
                num_classes=1,  # not used for feature extraction
                pretrained=True,
            )
            self._model = self._model.to(self._device)
            self._model.eval()
            # Warm up
            _dummy = torch.zeros(1, 3, _INPUT_SIZE[0], _INPUT_SIZE[1]).to(self._device)
            with torch.no_grad():
                self._model(_dummy)
            self._ready = True
            print(f"[ReID] OSNet ({_MODEL_NAME}) loaded on {self._device}")
        except Exception as e:
            print(f"[ReID] Failed to load model: {e}")
            self._ready = False

    def compute_embedding(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        """
        Compute a 512-dim embedding from a person crop.
        bbox = (x1, y1, x2, y2) in pixel coordinates.
        Returns normalized embedding vector or None if failed.
        """
        if not self._ready or self._model is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp and validate
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 10 or y2 - y1 < 20:
            return None

        # Crop and resize
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (_INPUT_SIZE[1], _INPUT_SIZE[0]))  # W, H
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Normalize to ImageNet stats
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        # To tensor: (H, W, C) → (1, C, H, W)
        tensor = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0).float().to(self._device)

        with torch.no_grad():
            features = self._model(tensor)

        embedding = features.cpu().numpy().flatten()
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def match(self, embedding: np.ndarray) -> int | None:
        """
        Find the best matching person in the gallery.
        Returns the person's global ID if match found, None otherwise.
        """
        if embedding is None:
            return None

        self._cleanup_gallery()

        best_id = None
        best_sim = MATCH_THRESHOLD

        with self._lock:
            for pid, entry in self._gallery.items():
                # Compare against average of stored embeddings
                if not entry["embeddings"]:
                    continue
                gallery_emb = np.mean(entry["embeddings"], axis=0)
                sim = float(np.dot(embedding, gallery_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid

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

            # Enforce max size — remove oldest
            if len(self._gallery) > GALLERY_MAX_SIZE:
                sorted_entries = sorted(self._gallery.items(), key=lambda x: x[1]["last_seen"])
                for pid, _ in sorted_entries[:len(self._gallery) - GALLERY_MAX_SIZE]:
                    del self._gallery[pid]
