"""
SQLite database layer - WAL mode, FTS5 with jieba, backup, dedup
Uses a background thread for writes to avoid blocking the main request thread.
"""
import atexit
import json
import logging
import os
import queue
import re
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timedelta

import jieba

from config import DB_PATH, DB_BACKUP_DIR, DB_BACKUP_KEEP_DAYS

logger = logging.getLogger(__name__)

# ---------- Async Write Queue ----------
_write_queue: queue.Queue = queue.Queue()
_writer_thread: threading.Thread | None = None
_stop_event = threading.Event()


def jieba_tokenize(text: str) -> str:
    """Segment Chinese text with jieba for FTS5 indexing."""
    words = jieba.cut_for_search(text)
    return " ".join(w.strip() for w in words if w.strip())


def get_db() -> sqlite3.Connection:
    """Get a database connection with WAL mode enabled."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    """Create tables if not exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            model TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            model TEXT,
            provider TEXT,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conv
            ON messages(conversation_id, created_at);

        -- FTS5 table for full-text search (tokenized by jieba externally)
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            conversation_id,
            content,
            tokenize='unicode61'
        );

        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT PRIMARY KEY,
            summary TEXT,
            message_count INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS weekly_summary (
            week TEXT PRIMARY KEY,
            summary TEXT,
            message_count INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def _write_worker():
    """Background thread: consume write queue and persist to SQLite."""
    while not _stop_event.is_set():
        try:
            task = _write_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            _do_save_message(**task)
        except Exception as e:
            logger.error(f"DB write failed: {e}")
        finally:
            _write_queue.task_done()


def start_writer():
    """Start the background writer thread."""
    global _writer_thread
    if _writer_thread is None or not _writer_thread.is_alive():
        _writer_thread = threading.Thread(target=_write_worker, daemon=True)
        _writer_thread.start()


def stop_writer():
    """Gracefully stop the writer thread and flush remaining items."""
    _stop_event.set()
    if _writer_thread:
        _writer_thread.join(timeout=10)


atexit.register(stop_writer)


def save_message(conversation_id: str, role: str, content: str,
                 model: str = "", provider: str = "",
                 tokens_in: int = 0, tokens_out: int = 0):
    """Enqueue a message for async writing (non-blocking)."""
    _write_queue.put({
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "model": model,
        "provider": provider,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    })


def _do_save_message(conversation_id: str, role: str, content: str,
                     model: str = "", provider: str = "",
                     tokens_in: int = 0, tokens_out: int = 0):
    """Actually persist a message to SQLite (called from writer thread)."""
    conn = get_db()
    try:
        # Ensure conversation exists
        conn.execute(
            "INSERT OR IGNORE INTO conversations (id, title, model) VALUES (?, ?, ?)",
            (conversation_id, "", model)
        )
        # Check for duplicate: same conv + role + content within last 5 seconds
        recent = conn.execute(
            """SELECT id FROM messages
               WHERE conversation_id = ? AND role = ? AND content = ?
               AND created_at > datetime('now', '-5 seconds')
               LIMIT 1""",
            (conversation_id, role, content)
        ).fetchone()
        if recent:
            return  # skip duplicate

        conn.execute(
            """INSERT INTO messages
               (conversation_id, role, content, model, provider, tokens_in, tokens_out)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (conversation_id, role, content, model, provider, tokens_in, tokens_out)
        )
        conn.execute(
            "UPDATE conversations SET updated_at = datetime('now'), model = ? WHERE id = ?",
            (model, conversation_id)
        )

        # Index in FTS5 (jieba tokenized)
        tokenized = jieba_tokenize(content) if content else ""
        if tokenized:
            conn.execute(
                "INSERT INTO messages_fts (conversation_id, content) VALUES (?, ?)",
                (conversation_id, tokenized)
            )
        conn.commit()
    finally:
        conn.close()


def search_history(query: str, limit: int = 5, max_chars: int = 4000) -> list[dict]:
    """Search message history using LIKE (fallback) + FTS5 jieba."""
    conn = get_db()
    results = []
    try:
        # FTS5 search with jieba tokenization
        tokenized_query = jieba_tokenize(query)
        if tokenized_query:
            fts_rows = conn.execute(
                """SELECT m.role, m.content, m.created_at, m.conversation_id
                   FROM messages_fts f
                   JOIN messages m ON m.conversation_id = f.conversation_id
                        AND m.content LIKE '%' || ? || '%'
                   WHERE messages_fts MATCH ?
                   ORDER BY m.created_at DESC
                   LIMIT ?""",
                (query[:20], tokenized_query, limit)
            ).fetchall()
            for row in fts_rows:
                results.append(dict(row))

        # Fallback: LIKE search if FTS didn't find enough
        if len(results) < limit:
            like_rows = conn.execute(
                """SELECT role, content, created_at, conversation_id
                   FROM messages
                   WHERE content LIKE ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (f"%{query}%", limit - len(results))
            ).fetchall()
            seen = {(r["conversation_id"], r["created_at"]) for r in results}
            for row in like_rows:
                d = dict(row)
                if (d["conversation_id"], d["created_at"]) not in seen:
                    results.append(d)

        # Trim total chars
        trimmed = []
        total = 0
        for r in results:
            c = r.get("content", "") or ""
            if total + len(c) > max_chars:
                break
            trimmed.append(r)
            total += len(c)
        return trimmed
    finally:
        conn.close()


def get_recent_messages(conversation_id: str, limit: int = 10) -> list[dict]:
    """Get recent messages for a conversation (from our DB, not Kelivo)."""
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT role, content, created_at FROM messages
               WHERE conversation_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (conversation_id, limit)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def backup_database():
    """Backup chats.db, keep last N days."""
    os.makedirs(DB_BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(DB_BACKUP_DIR, f"chats_{timestamp}.db")
    shutil.copy2(DB_PATH, dest)

    # Clean old backups
    cutoff = datetime.now() - timedelta(days=DB_BACKUP_KEEP_DAYS)
    for f in os.listdir(DB_BACKUP_DIR):
        fpath = os.path.join(DB_BACKUP_DIR, f)
        if os.path.isfile(fpath):
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime < cutoff:
                os.remove(fpath)
    return dest
