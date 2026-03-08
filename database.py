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


def _sanitize_fts5_query(tokenized: str) -> str:
    """
    Escape a jieba-tokenized string so it is safe for FTS5 MATCH.

    FTS5 operators/special chars: " * ~ - ( ) OR AND NOT NEAR
    Strategy: wrap every token in double-quotes so FTS5 treats it as a literal.
    Tokens that themselves contain a double-quote get that quote doubled (FTS5 escaping).
    Skip tokens that are pure punctuation / empty.
    """
    tokens = tokenized.split()
    safe = []
    for t in tokens:
        # Skip pure punctuation / whitespace tokens
        if not t or re.fullmatch(r'[\s\W]+', t):
            continue
        # Double any internal double-quotes, then wrap in quotes
        escaped = t.replace('"', '""')
        safe.append(f'"{escaped}"')
    return " ".join(safe)


def get_db() -> sqlite3.Connection:
    """Get a database connection with WAL mode enabled."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Get column names for a table."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row["name"] for row in rows]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Check if a table exists."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def init_db():
    """Create tables if not exist, migrate legacy chat_history if found."""
    conn = get_db()

    # --- Create new schema tables ---
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

    # --- Migrate legacy chat_history table if it exists ---
    if _table_exists(conn, "chat_history"):
        _migrate_chat_history(conn)

    conn.close()


def _migrate_chat_history(conn: sqlite3.Connection):
    """
    Migrate data from legacy chat_history table into the new messages table.
    Handles various old column layouts gracefully. Never deletes old data.
    """
    # Check if already migrated (messages table has data)
    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    old_count = conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0]
    if old_count == 0:
        return
    if msg_count >= old_count:
        logger.info(f"Migration already done ({msg_count} messages exist, "
                     f"{old_count} in chat_history). Skipping.")
        return

    logger.info(f"Migrating {old_count} records from chat_history -> messages ...")

    old_cols = _get_table_columns(conn, "chat_history")
    logger.info(f"chat_history columns: {old_cols}")

    # Build SELECT mapping: figure out what the old table has
    has_conversation_id = "conversation_id" in old_cols
    has_role = "role" in old_cols
    has_content = "content" in old_cols
    has_message = "message" in old_cols  # some old schemas use "message" instead
    has_model = "model" in old_cols
    has_provider = "provider" in old_cols
    has_tokens_in = "tokens_in" in old_cols
    has_tokens_out = "tokens_out" in old_cols
    has_created_at = "created_at" in old_cols
    has_timestamp = "timestamp" in old_cols  # another common variant

    # Content column: prefer "content", fall back to "message"
    content_col = "content" if has_content else ("message" if has_message else None)
    if not content_col:
        logger.warning("chat_history has no 'content' or 'message' column. "
                        "Cannot migrate.")
        return

    # Time column
    time_col = "created_at" if has_created_at else ("timestamp" if has_timestamp else None)

    # Read all old records
    rows = conn.execute(f"SELECT * FROM chat_history ORDER BY rowid").fetchall()

    migrated = 0
    for row in rows:
        row_dict = dict(row)

        content = row_dict.get(content_col, "") or ""
        if not content.strip():
            continue

        role = row_dict.get("role", "user") if has_role else "user"
        conv_id = (row_dict.get("conversation_id", "") or "") if has_conversation_id else ""
        model = (row_dict.get("model", "") or "") if has_model else ""
        provider = (row_dict.get("provider", "") or "") if has_provider else ""
        tokens_in = row_dict.get("tokens_in", 0) if has_tokens_in else 0
        tokens_out = row_dict.get("tokens_out", 0) if has_tokens_out else 0

        # Determine created_at
        created_at = None
        if time_col:
            created_at = row_dict.get(time_col)

        # Generate conversation_id from date if missing
        if not conv_id:
            if created_at and len(str(created_at)) >= 10:
                date_part = str(created_at)[:10]  # "2025-03-01"
                conv_id = f"legacy-{date_part}"
            else:
                conv_id = "legacy-unknown"

        # Ensure conversation exists
        conn.execute(
            "INSERT OR IGNORE INTO conversations (id, title, model) VALUES (?, ?, ?)",
            (conv_id, "", model)
        )

        # Insert into messages
        if created_at:
            conn.execute(
                """INSERT INTO messages
                   (conversation_id, role, content, model, provider,
                    tokens_in, tokens_out, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (conv_id, role, content, model, provider,
                 tokens_in or 0, tokens_out or 0, created_at)
            )
        else:
            conn.execute(
                """INSERT INTO messages
                   (conversation_id, role, content, model, provider,
                    tokens_in, tokens_out)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (conv_id, role, content, model, provider,
                 tokens_in or 0, tokens_out or 0)
            )

        # Index in FTS5
        tokenized = jieba_tokenize(content)
        if tokenized:
            conn.execute(
                "INSERT INTO messages_fts (conversation_id, content) VALUES (?, ?)",
                (conv_id, tokenized)
            )
        migrated += 1

    conn.commit()

    # Rename old table to backup (keep data, stop future migration attempts)
    conn.execute("ALTER TABLE chat_history RENAME TO chat_history_backup")
    conn.commit()
    logger.info(f"Migration complete: {migrated}/{old_count} records migrated. "
                 f"Old table renamed to chat_history_backup.")


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


def _build_like_conditions(keywords: list[str], column: str = "content") -> tuple[str, list[str]]:
    """
    Build a WHERE clause that requires ALL keywords to appear (AND logic).
    Returns (sql_fragment, params).
    E.g. keywords=["机器","学习"] -> "content LIKE ? AND content LIKE ?", ["%机器%", "%学习%"]
    """
    if not keywords:
        return "1=1", []
    clauses = [f"{column} LIKE ?" for _ in keywords]
    params = [f"%{kw}%" for kw in keywords]
    return " AND ".join(clauses), params


def search_history(query: str, limit: int = 5, max_chars: int = 4000) -> list[dict]:
    """
    Search message history.
    Priority: jieba tokenized multi-keyword LIKE (works for Chinese),
    then FTS5 as fallback (works for English/indexed content).
    """
    conn = get_db()
    results = []
    seen = set()

    def _add_rows(rows):
        for row in rows:
            d = dict(row)
            key = (d["conversation_id"], d["created_at"])
            if key not in seen:
                seen.add(key)
                results.append(d)

    try:
        # --- Phase 1: jieba-tokenized multi-keyword LIKE (Chinese-friendly) ---
        keywords = [w for w in jieba_tokenize(query).split() if len(w) >= 1]
        # Always include the original query as a keyword for exact-phrase matching
        all_keywords = list(dict.fromkeys([query] + keywords))  # dedup, preserve order

        if all_keywords:
            # First try exact query match
            exact_rows = conn.execute(
                """SELECT role, content, created_at, conversation_id
                   FROM messages WHERE content LIKE ?
                   ORDER BY created_at DESC LIMIT ?""",
                (f"%{query}%", limit)
            ).fetchall()
            _add_rows(exact_rows)

            # Then try multi-keyword AND match (jieba tokens)
            if len(results) < limit and len(keywords) > 1:
                where_sql, params = _build_like_conditions(keywords)
                kw_rows = conn.execute(
                    f"""SELECT role, content, created_at, conversation_id
                        FROM messages WHERE {where_sql}
                        ORDER BY created_at DESC LIMIT ?""",
                    params + [limit - len(results)]
                ).fetchall()
                _add_rows(kw_rows)

            # Then try individual keywords (broader recall)
            if len(results) < limit and len(keywords) > 1:
                for kw in keywords:
                    if len(results) >= limit:
                        break
                    if len(kw) < 2:
                        continue  # skip single-char tokens for noise reduction
                    single_rows = conn.execute(
                        """SELECT role, content, created_at, conversation_id
                           FROM messages WHERE content LIKE ?
                           ORDER BY created_at DESC LIMIT ?""",
                        (f"%{kw}%", limit - len(results))
                    ).fetchall()
                    _add_rows(single_rows)

        # --- Phase 2: FTS5 fallback (useful for English / already-indexed content) ---
        if len(results) < limit:
            tokenized_query = jieba_tokenize(query)
            safe_fts_query = _sanitize_fts5_query(tokenized_query) if tokenized_query else ""
            if safe_fts_query:
                try:
                    fts_rows = conn.execute(
                        """SELECT m.role, m.content, m.created_at, m.conversation_id
                           FROM messages_fts f
                           JOIN messages m ON m.conversation_id = f.conversation_id
                           WHERE messages_fts MATCH ?
                           ORDER BY m.created_at DESC
                           LIMIT ?""",
                        (safe_fts_query, limit - len(results))
                    ).fetchall()
                    _add_rows(fts_rows)
                except Exception:
                    logger.warning("FTS5 query failed", exc_info=True)

        # --- Phase 3: legacy backup table ---
        if len(results) < limit and _table_exists(conn, "chat_history_backup"):
            try:
                backup_cols = _get_table_columns(conn, "chat_history_backup")
                c_col = "content" if "content" in backup_cols else (
                    "message" if "message" in backup_cols else None)
                t_col = "created_at" if "created_at" in backup_cols else (
                    "timestamp" if "timestamp" in backup_cols else None)
                r_col = "role" if "role" in backup_cols else None
                if c_col:
                    legacy_sql = f"""SELECT
                        {f'{r_col}' if r_col else "'user'"} as role,
                        {c_col} as content,
                        {t_col if t_col else "'unknown'"} as created_at,
                        'legacy' as conversation_id
                        FROM chat_history_backup
                        WHERE {c_col} LIKE ?
                        ORDER BY rowid DESC LIMIT ?"""
                    legacy_rows = conn.execute(
                        legacy_sql, (f"%{query}%", limit - len(results))
                    ).fetchall()
                    _add_rows(legacy_rows)
            except Exception as e:
                logger.debug(f"Legacy search failed (ok to ignore): {e}")

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
