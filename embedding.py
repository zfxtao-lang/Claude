"""
Vector embedding + lightweight numpy-based vector store.

Uses Alibaba text-embedding-v3 (768 dimensions) for embeddings.
Stores vectors in a numpy .npy file + metadata in SQLite.
Designed for 2C2G servers: ~30MB memory for 10K chunks.
"""
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict

import numpy as np
import requests

from config import DB_PATH

logger = logging.getLogger(__name__)

# ---------- Query Embedding Cache ----------
_QUERY_CACHE_MAX = 128  # max cached query embeddings
_query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_query_cache_lock = threading.Lock()

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


# ---------- Query Embedding (cached + 3s timeout) ----------

QUERY_EMBEDDING_TIMEOUT = int(os.getenv("QUERY_EMBEDDING_TIMEOUT", "3"))


def get_embedding_for_query(text: str) -> np.ndarray | None:
    """
    Get embedding for a user query with:
    1. In-memory LRU cache — same question won't hit API twice
    2. 3-second timeout — fail fast, let caller fall back to LIKE search

    For pre-computed chunk/card embeddings, use get_embedding() or
    get_embeddings_batch() instead (those use longer timeouts for batch jobs).
    """
    if not EMBEDDING_API_KEY or not text:
        return None

    text = text[:8000].strip()
    if not text:
        return None

    # --- Cache lookup ---
    cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
    with _query_cache_lock:
        if cache_key in _query_cache:
            _query_cache.move_to_end(cache_key)
            logger.info("[Vector] query embedding cache HIT")
            return _query_cache[cache_key].copy()

    # --- API call with short timeout ---
    t0 = time.time()
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
            timeout=QUERY_EMBEDDING_TIMEOUT,
        )
        elapsed = time.time() - t0
        if resp.status_code != 200:
            logger.error(f"[Vector] query embedding error {resp.status_code} "
                         f"({elapsed:.1f}s): {resp.text[:200]}")
            return None

        data = resp.json()
        vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
        logger.info(f"[Vector] query embedding OK ({elapsed:.1f}s), caching")

        # --- Store in cache ---
        with _query_cache_lock:
            _query_cache[cache_key] = vec.copy()
            while len(_query_cache) > _QUERY_CACHE_MAX:
                _query_cache.popitem(last=False)

        return vec

    except requests.exceptions.Timeout:
        elapsed = time.time() - t0
        logger.warning(f"[Vector] query embedding TIMEOUT after {elapsed:.1f}s "
                       f"(limit={QUERY_EMBEDDING_TIMEOUT}s), will fall back to LIKE")
        return None
    except Exception:
        elapsed = time.time() - t0
        logger.error(f"[Vector] query embedding failed ({elapsed:.1f}s)", exc_info=True)
        return None


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


CARD_VECTOR_FILE = os.getenv("CARD_VECTOR_FILE", "card_vectors.npy")
CARD_VECTOR_IDS_FILE = os.getenv("CARD_VECTOR_IDS_FILE", "card_vectors_ids.npy")


class CardVectorStore(VectorStore):
    """Vector store specifically for memory cards. Same engine, separate files."""

    def __init__(self):
        super().__init__()

    def _load(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if os.path.exists(CARD_VECTOR_FILE) and os.path.exists(CARD_VECTOR_IDS_FILE):
                try:
                    self._vectors = np.load(CARD_VECTOR_FILE)
                    self._chunk_ids = np.load(CARD_VECTOR_IDS_FILE)
                    logger.info(f"[CardVector] loaded {len(self._chunk_ids)} card vectors from disk")
                except Exception:
                    logger.error("[CardVector] failed to load card vectors", exc_info=True)
                    self._vectors = None
                    self._chunk_ids = None
            else:
                logger.info("[CardVector] no card vector files found, starting empty")
            self._loaded = True

    def save(self):
        with self._lock:
            if self._vectors is not None and len(self._vectors) > 0:
                np.save(CARD_VECTOR_FILE, self._vectors)
                np.save(CARD_VECTOR_IDS_FILE, self._chunk_ids)
                logger.info(f"[CardVector] saved {len(self._chunk_ids)} card vectors to disk")


# Global instances
vector_store = VectorStore()
card_vector_store = CardVectorStore()
