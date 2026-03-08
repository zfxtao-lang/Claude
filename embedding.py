"""
Vector embedding + lightweight numpy-based vector store.

Uses Alibaba text-embedding-v3 (768 dimensions) for embeddings.
Stores vectors in a numpy .npy file + metadata in SQLite.
Designed for 2C2G servers: ~30MB memory for 10K chunks.
"""
import logging
import os
import threading
import time

import numpy as np
import requests

from config import DB_PATH

logger = logging.getLogger(__name__)

# ---------- Configuration ----------
EMBEDDING_API_KEY = os.getenv("ALIBABA_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
VECTOR_FILE = os.getenv("VECTOR_FILE", "vectors.npy")
VECTOR_IDS_FILE = os.getenv("VECTOR_IDS_FILE", "vectors_ids.npy")
CHUNK_SIZE = int(os.getenv("VECTOR_CHUNK_ROUNDS", "4"))  # rounds per chunk

# ---------- Embedding API ----------

def get_embedding(text: str) -> np.ndarray | None:
    """Call Alibaba text-embedding-v3 API, return 768-dim vector or None."""
    if not EMBEDDING_API_KEY:
        logger.warning("[Vector] ALIBABA_API_KEY not set, embedding disabled")
        return None

    # Truncate to ~8000 chars (API limit is ~8K tokens, Chinese ~1 char = 1-2 tokens)
    text = text[:8000].strip()
    if not text:
        return None

    try:
        resp = requests.post(
            f"{EMBEDDING_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": text,
                "dimensions": EMBEDDING_DIMENSIONS,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error(f"[Vector] embedding API error {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        vec = data["data"][0]["embedding"]
        return np.array(vec, dtype=np.float32)

    except Exception:
        logger.error("[Vector] embedding API call failed", exc_info=True)
        return None


def get_embeddings_batch(texts: list[str]) -> list[np.ndarray | None]:
    """Batch embedding call. Alibaba supports up to 6 texts per call."""
    if not EMBEDDING_API_KEY or not texts:
        return [None] * len(texts)

    results = [None] * len(texts)
    batch_size = 6  # Alibaba API limit

    for start in range(0, len(texts), batch_size):
        batch = [t[:8000].strip() for t in texts[start:start + batch_size]]
        batch = [t if t else " " for t in batch]  # API rejects empty strings

        try:
            resp = requests.post(
                f"{EMBEDDING_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": batch,
                    "dimensions": EMBEDDING_DIMENSIONS,
                },
                timeout=60,
            )
            if resp.status_code != 200:
                logger.error(f"[Vector] batch embedding error {resp.status_code}: "
                             f"{resp.text[:200]}")
                continue

            data = resp.json()
            for item in data["data"]:
                idx = item["index"]
                results[start + idx] = np.array(item["embedding"], dtype=np.float32)

        except Exception:
            logger.error("[Vector] batch embedding failed", exc_info=True)

    return results


# ---------- Vector Store (numpy-based) ----------

class VectorStore:
    """
    Lightweight vector store backed by numpy arrays + SQLite metadata.
    Thread-safe. Lazy-loads vectors on first search.

    Memory usage: 10K chunks × 768 dims × 4 bytes = ~30MB
    """

    def __init__(self):
        self._vectors: np.ndarray | None = None  # shape: (N, 768)
        self._chunk_ids: np.ndarray | None = None  # shape: (N,) int64
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self):
        """Load vectors from disk if available."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if os.path.exists(VECTOR_FILE) and os.path.exists(VECTOR_IDS_FILE):
                try:
                    self._vectors = np.load(VECTOR_FILE)
                    self._chunk_ids = np.load(VECTOR_IDS_FILE)
                    logger.info(f"[Vector] loaded {len(self._chunk_ids)} vectors from disk")
                except Exception:
                    logger.error("[Vector] failed to load vectors from disk", exc_info=True)
                    self._vectors = None
                    self._chunk_ids = None
            else:
                logger.info("[Vector] no vector files found, starting empty")
            self._loaded = True

    def add(self, chunk_id: int, vector: np.ndarray):
        """Add a single vector. Thread-safe."""
        self._load()
        with self._lock:
            vec = vector.reshape(1, -1)
            cid = np.array([chunk_id], dtype=np.int64)
            if self._vectors is None:
                self._vectors = vec
                self._chunk_ids = cid
            else:
                self._vectors = np.vstack([self._vectors, vec])
                self._chunk_ids = np.concatenate([self._chunk_ids, cid])

    def save(self):
        """Persist vectors to disk."""
        with self._lock:
            if self._vectors is not None and len(self._vectors) > 0:
                np.save(VECTOR_FILE, self._vectors)
                np.save(VECTOR_IDS_FILE, self._chunk_ids)
                logger.info(f"[Vector] saved {len(self._chunk_ids)} vectors to disk")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Cosine similarity search. Returns [(chunk_id, score), ...].
        """
        self._load()
        if self._vectors is None or len(self._vectors) == 0:
            return []

        with self._lock:
            # Normalize for cosine similarity
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-9
            normed = self._vectors / norms

            scores = normed @ query_norm  # (N,)
            top_indices = np.argsort(scores)[::-1][:top_k]

            return [(int(self._chunk_ids[i]), float(scores[i])) for i in top_indices]

    @property
    def size(self) -> int:
        self._load()
        return len(self._chunk_ids) if self._chunk_ids is not None else 0

    def rebuild(self, vectors: np.ndarray, chunk_ids: np.ndarray):
        """Replace all vectors (used during full rebuild)."""
        with self._lock:
            self._vectors = vectors
            self._chunk_ids = chunk_ids
            self._loaded = True
        self.save()


# Global instance
vector_store = VectorStore()
