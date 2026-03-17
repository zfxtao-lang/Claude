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

        CREATE INDEX IF NOT EXISTS idx_messages_created_at
            ON messages(created_at);

        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            conversation_id,
            content,
            tokenize='unicode61'
        );

        -- Legacy only: kept for rollback/history reference. New summary pipeline
        -- uses memory_cards + rolling summaries instead of writing these tables.
        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT PRIMARY KEY,
            summary TEXT,
            message_count INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Legacy only: do not use as an active summary source.
        CREATE TABLE IF NOT EXISTS weekly_summary (
            week TEXT PRIMARY KEY,
            summary TEXT,
            message_count INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Vector search: conversation chunks for embedding
        CREATE TABLE IF NOT EXISTS vector_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            content TEXT NOT NULL,
            msg_id_start INTEGER NOT NULL,
            msg_id_end INTEGER NOT NULL,
            round_count INTEGER DEFAULT 0,
            has_embedding INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON vector_chunks(has_embedding);
        CREATE INDEX IF NOT EXISTS idx_chunks_msg_range
            ON vector_chunks(msg_id_end);

        -- Memory cards: AI-generated daily summaries for efficient retrieval
        CREATE TABLE IF NOT EXISTS memory_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            summary TEXT NOT NULL,
            tags TEXT DEFAULT '',
            conversation_ids TEXT DEFAULT '',
            msg_id_start INTEGER DEFAULT 0,
            msg_id_end INTEGER DEFAULT 0,
            has_embedding INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_cards_date
            ON memory_cards(date);
        CREATE INDEX IF NOT EXISTS idx_cards_embedding
            ON memory_cards(has_embedding);

        CREATE TABLE IF NOT EXISTS memory_slices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL DEFAULT '',
            conversation_ids TEXT NOT NULL DEFAULT '',
            slice_index INTEGER NOT NULL DEFAULT 0,
            msg_id_start INTEGER NOT NULL,
            msg_id_end INTEGER NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0,
            summary TEXT NOT NULL,
            tags TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            source_long_memory_id INTEGER,
            has_embedding INTEGER NOT NULL DEFAULT 0,
            meta_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_memory_slices_status
            ON memory_slices(status, id);
        CREATE INDEX IF NOT EXISTS idx_memory_slices_msg_range
            ON memory_slices(msg_id_start, msg_id_end);
        CREATE INDEX IF NOT EXISTS idx_memory_slices_source_long_memory
            ON memory_slices(source_long_memory_id);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_slices_msg_range
            ON memory_slices(msg_id_start, msg_id_end);

        CREATE TABLE IF NOT EXISTS long_term_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_type TEXT NOT NULL DEFAULT 'summary',
            title TEXT DEFAULT '',
            content TEXT NOT NULL,
            fact_weight REAL NOT NULL DEFAULT 1.0,
            half_life_days REAL NOT NULL DEFAULT 30.0,
            hits INTEGER NOT NULL DEFAULT 0,
            last_hit_at TEXT,
            source_slice_start_id INTEGER DEFAULT 0,
            source_slice_end_id INTEGER DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
            has_embedding INTEGER NOT NULL DEFAULT 0,
            meta_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_long_term_memories_status
            ON long_term_memories(status, id);
        CREATE INDEX IF NOT EXISTS idx_long_term_memories_embedding
            ON long_term_memories(has_embedding);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_long_term_memories_source_range
            ON long_term_memories(source_slice_start_id, source_slice_end_id);

        CREATE TABLE IF NOT EXISTS long_term_memory_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            long_term_memory_id INTEGER NOT NULL,
            memory_slice_id INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_long_term_memory_sources_long
            ON long_term_memory_sources(long_term_memory_id);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_long_term_memory_sources_pair
            ON long_term_memory_sources(long_term_memory_id, memory_slice_id);

        CREATE TABLE IF NOT EXISTS diary_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT NOT NULL,
            title TEXT DEFAULT '',
            content TEXT NOT NULL,
            mood_score INTEGER NOT NULL DEFAULT 5,
            mood_label TEXT DEFAULT '',
            source_msg_id_start INTEGER DEFAULT 0,
            source_msg_id_end INTEGER DEFAULT 0,
            source_slice_ids TEXT DEFAULT '',
            worker_run_id INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_diary_entries_date
            ON diary_entries(entry_date DESC, id DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_diary_entries_entry_date
            ON diary_entries(entry_date);

        CREATE TABLE IF NOT EXISTS persona_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            persona_json TEXT NOT NULL DEFAULT '{}',
            relationship_json TEXT NOT NULL DEFAULT '{}',
            summary TEXT DEFAULT '',
            source_diary_id INTEGER,
            worker_run_id INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_persona_snapshots_date
            ON persona_snapshots(snapshot_date DESC, id DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_persona_snapshots_diary
            ON persona_snapshots(snapshot_date, source_diary_id);

        CREATE TABLE IF NOT EXISTS pending_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_type TEXT NOT NULL,
            source_snapshot_id INTEGER,
            proposed_payload TEXT NOT NULL DEFAULT '{}',
            diff_summary TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            review_note TEXT DEFAULT '',
            edited_payload TEXT DEFAULT '',
            approved_payload TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            reviewed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_pending_reviews_status
            ON pending_reviews(status, id DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_reviews_snapshot_type
            ON pending_reviews(review_type, source_snapshot_id);

        CREATE TABLE IF NOT EXISTS active_profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            profile_json TEXT NOT NULL DEFAULT '{}',
            relationship_json TEXT NOT NULL DEFAULT '{}',
            source_review_id INTEGER,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS review_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pending_review_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            before_payload TEXT DEFAULT '',
            after_payload TEXT DEFAULT '',
            note TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_review_history_queue
            ON review_history(pending_review_id, id DESC);

        CREATE TABLE IF NOT EXISTS worker_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_name TEXT NOT NULL DEFAULT 'memory_worker',
            run_mode TEXT NOT NULL DEFAULT 'manual',
            status TEXT NOT NULL DEFAULT 'queued',
            phase TEXT NOT NULL DEFAULT 'queued',
            message TEXT DEFAULT '',
            started_at TEXT DEFAULT (datetime('now')),
            completed_at TEXT,
            token_input INTEGER NOT NULL DEFAULT 0,
            token_output INTEGER NOT NULL DEFAULT 0,
            token_total INTEGER NOT NULL DEFAULT 0,
            result_json TEXT DEFAULT '{}',
            error TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_worker_runs_started
            ON worker_runs(started_at DESC, id DESC);
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


def search_history(query: str, limit: int = 5, max_chars: int = 4000,
                   exclude_recent: int = 50) -> list[dict]:
    """
    Search message history.
    Priority: jieba tokenized multi-keyword LIKE (works for Chinese),
    then FTS5 as fallback (works for English/indexed content).

    exclude_recent: skip the N most recent messages to avoid polluting
    results with the current conversation (e.g. model saying "I don't remember").
    Uses rowid instead of time because migrated data may have inaccurate timestamps.
    """
    conn = get_db()
    results = []
    seen = set()

    # Find the rowid cutoff: exclude the most recent N messages
    try:
        max_id_row = conn.execute("SELECT MAX(id) FROM messages").fetchone()
        max_id = max_id_row[0] if max_id_row and max_id_row[0] else 0
    except Exception:
        max_id = 0
    id_cutoff = max(0, max_id - exclude_recent)
    recency_filter = "id <= ?"

    logger.info(f"[Memory] search_history called: query='{query[:80]}', "
                f"limit={limit}, max_id={max_id}, id_cutoff={id_cutoff} "
                f"(excluding latest {exclude_recent} msgs)")

    def _add_rows(rows):
        for row in rows:
            d = dict(row)
            key = d.get("id") or (d["conversation_id"], d["created_at"], d["content"])
            if key not in seen:
                seen.add(key)
                results.append(d)

    try:
        # --- Phase 1: jieba-tokenized multi-keyword LIKE (Chinese-friendly) ---
        keywords = [w for w in jieba_tokenize(query).split() if len(w) >= 1]
        # Always include the original query as a keyword for exact-phrase matching
        all_keywords = list(dict.fromkeys([query] + keywords))  # dedup, preserve order

        if all_keywords:
            logger.info(f"[Memory] keywords: {keywords[:10]}")

            # First try exact query match
            exact_rows = conn.execute(
                f"""SELECT id, role, content, created_at, conversation_id
                    FROM messages WHERE content LIKE ? AND {recency_filter}
                    ORDER BY id DESC LIMIT ?""",
                (f"%{query}%", id_cutoff, limit)
            ).fetchall()
            _add_rows(exact_rows)
            logger.info(f"[Memory] phase1-exact: {len(exact_rows)} rows, total={len(results)}")

            # Then try multi-keyword AND match (jieba tokens)
            if len(results) < limit and len(keywords) > 1:
                where_sql, params = _build_like_conditions(keywords)
                kw_rows = conn.execute(
                    f"""SELECT id, role, content, created_at, conversation_id
                        FROM messages WHERE {where_sql} AND {recency_filter}
                        ORDER BY id DESC LIMIT ?""",
                    params + [id_cutoff, limit - len(results)]
                ).fetchall()
                _add_rows(kw_rows)
                logger.info(f"[Memory] phase1-multi-kw: {len(kw_rows)} rows, total={len(results)}")

            # Then try individual keywords (broader recall)
            if len(results) < limit and len(keywords) > 1:
                for kw in keywords:
                    if len(results) >= limit:
                        break
                    if len(kw) < 2:
                        continue  # skip single-char tokens for noise reduction
                    single_rows = conn.execute(
                        f"""SELECT id, role, content, created_at, conversation_id
                            FROM messages WHERE content LIKE ? AND {recency_filter}
                            ORDER BY id DESC LIMIT ?""",
                        (f"%{kw}%", id_cutoff, limit - len(results))
                    ).fetchall()
                    _add_rows(single_rows)
                logger.info(f"[Memory] phase1-single-kw: total={len(results)}")

        # --- Phase 2: FTS5 fallback (useful for English / already-indexed content) ---
        if len(results) < limit:
            tokenized_query = jieba_tokenize(query)
            safe_fts_query = _sanitize_fts5_query(tokenized_query) if tokenized_query else ""
            if safe_fts_query:
                try:
                    fts_rows = conn.execute(
                        f"""SELECT m.id, m.role, m.content, m.created_at, m.conversation_id
                            FROM messages_fts f
                            JOIN messages m ON m.id = f.rowid
                            WHERE messages_fts MATCH ? AND m.{recency_filter}
                            ORDER BY m.id DESC
                            LIMIT ?""",
                        (safe_fts_query, id_cutoff, limit - len(results))
                    ).fetchall()
                    _add_rows(fts_rows)
                    logger.info(f"[Memory] phase2-fts5: {len(fts_rows)} rows, total={len(results)}")
                except Exception:
                    logger.warning("[Memory] phase2-fts5 failed", exc_info=True)

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
                        legacy_sql, [f"%{query}%", limit - len(results)]
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

        if trimmed:
            dates = [r.get("created_at", "?")[:10] for r in trimmed]
            logger.info(f"[Memory] search_history returning {len(trimmed)} results "
                        f"({total} chars), dates: {dates}")
        else:
            logger.info(f"[Memory] search_history returning 0 results "
                        f"(pre-trim had {len(results)})")
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


# ---------- Vector Chunking ----------

CHUNK_ROUNDS = int(os.getenv("VECTOR_CHUNK_ROUNDS", "4"))  # user+assistant pairs per chunk


def build_pending_chunks() -> int:
    """
    Scan messages table for new messages not yet chunked.
    Groups by conversation_id, creates chunks of CHUNK_ROUNDS rounds each.
    Returns number of new chunks created.
    """
    conn = get_db()
    try:
        # Find the highest msg_id already chunked
        row = conn.execute("SELECT MAX(msg_id_end) FROM vector_chunks").fetchone()
        last_chunked_id = row[0] if row and row[0] else 0

        # Get all un-chunked messages ordered by id
        rows = conn.execute(
            """SELECT id, conversation_id, role, content, created_at
               FROM messages WHERE id > ? ORDER BY id""",
            (last_chunked_id,)
        ).fetchall()

        if not rows:
            return 0

        # Group by conversation_id, preserving order
        from collections import OrderedDict
        conv_msgs: dict[str, list[dict]] = OrderedDict()
        for r in rows:
            d = dict(r)
            conv_id = d["conversation_id"]
            if conv_id not in conv_msgs:
                conv_msgs[conv_id] = []
            conv_msgs[conv_id].append(d)

        created = 0
        for conv_id, msgs in conv_msgs.items():
            # Count rounds: a round = one user message (assistant may follow)
            rounds = []
            current_round = []
            for m in msgs:
                current_round.append(m)
                if m["role"] == "assistant":
                    rounds.append(current_round)
                    current_round = []
            # Handle dangling user message (no assistant reply yet)
            if current_round:
                if len(rounds) >= CHUNK_ROUNDS:
                    pass  # enough rounds already, leave dangling for next time
                elif rounds:
                    # Not enough full rounds, but we have some complete rounds.
                    # Include them (don't skip entire conversation).
                    pass
                else:
                    # Only a dangling user message, no complete rounds at all.
                    # Still include it as a chunk so no message is lost.
                    rounds.append(current_round)
                    current_round = []

            # If we only have incomplete rounds (no assistant replies),
            # include the dangling messages as a round so they get chunked
            if not rounds and current_round:
                rounds.append(current_round)
                current_round = []

            if not rounds:
                continue

            # Create chunks of CHUNK_ROUNDS rounds
            for i in range(0, len(rounds), CHUNK_ROUNDS):
                batch = rounds[i:i + CHUNK_ROUNDS]
                # Always include the last batch even if small (don't skip any messages)

                all_msgs_in_chunk = [m for rnd in batch for m in rnd]
                chunk_text = "\n".join(
                    f"{m['role']}: {m['content']}"
                    for m in all_msgs_in_chunk
                    if m.get("content")
                )
                if not chunk_text.strip():
                    continue

                msg_id_start = all_msgs_in_chunk[0]["id"]
                msg_id_end = all_msgs_in_chunk[-1]["id"]

                conn.execute(
                    """INSERT INTO vector_chunks
                       (conversation_id, content, msg_id_start, msg_id_end, round_count)
                       VALUES (?, ?, ?, ?, ?)""",
                    (conv_id, chunk_text, msg_id_start, msg_id_end, len(batch))
                )
                created += 1

        conn.commit()
        logger.info(f"[Vector] created {created} new chunks from {len(rows)} messages")
        return created
    finally:
        conn.close()


def get_unembedded_chunks(limit: int = 50) -> list[dict]:
    """Get chunks that don't have embeddings yet."""
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT id, conversation_id, content, msg_id_start, msg_id_end, created_at
               FROM vector_chunks WHERE has_embedding = 0
               ORDER BY id LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def mark_chunks_embedded(chunk_ids: list[int]):
    """Mark chunks as having embeddings."""
    if not chunk_ids:
        return
    conn = get_db()
    try:
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(
            f"UPDATE vector_chunks SET has_embedding = 1 WHERE id IN ({placeholders})",
            chunk_ids
        )
        conn.commit()
    finally:
        conn.close()


def get_chunks_by_ids(chunk_ids: list[int]) -> list[dict]:
    """Fetch chunk content by IDs (for displaying search results)."""
    if not chunk_ids:
        return []
    conn = get_db()
    try:
        placeholders = ",".join("?" * len(chunk_ids))
        rows = conn.execute(
            f"""SELECT id, conversation_id, content, msg_id_start, msg_id_end, created_at
                FROM vector_chunks WHERE id IN ({placeholders})""",
            chunk_ids
        ).fetchall()
        # Return in the order requested
        id_to_row = {dict(r)["id"]: dict(r) for r in rows}
        return [id_to_row[cid] for cid in chunk_ids if cid in id_to_row]
    finally:
        conn.close()


def get_neighbor_chunks(chunk_ids: list[int], window: int = 1) -> list[dict]:
    """
    Given a list of matched chunk IDs, also fetch neighboring chunks
    from the same conversation (±window chunks by id order).

    This expands recall: if chunk 5 matched, also return chunks 4 and 6
    from the same conversation, giving complete topic context.

    Returns all chunks (original + neighbors), deduplicated, sorted by
    (conversation_id, msg_id_start).
    """
    if not chunk_ids:
        return []
    conn = get_db()
    try:
        # Get conversation_ids and id ranges for the matched chunks
        placeholders = ",".join("?" * len(chunk_ids))
        matched = conn.execute(
            f"""SELECT id, conversation_id FROM vector_chunks
                WHERE id IN ({placeholders})""",
            chunk_ids
        ).fetchall()

        # For each matched chunk, find neighbors in the same conversation
        all_ids = set(chunk_ids)
        for row in matched:
            conv_id = row["conversation_id"]
            chunk_id = row["id"]
            neighbors = conn.execute(
                """SELECT id FROM vector_chunks
                   WHERE conversation_id = ?
                   AND id BETWEEN ? AND ?
                   ORDER BY id""",
                (conv_id, chunk_id - window, chunk_id + window)
            ).fetchall()
            for n in neighbors:
                all_ids.add(n["id"])

        # Fetch all chunks (matched + neighbors)
        all_placeholders = ",".join("?" * len(all_ids))
        rows = conn.execute(
            f"""SELECT id, conversation_id, content, msg_id_start, msg_id_end, created_at
                FROM vector_chunks WHERE id IN ({all_placeholders})
                ORDER BY conversation_id, msg_id_start""",
            list(all_ids)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_chunks_for_rebuild() -> list[dict]:
    """Get all chunks for full vector rebuild."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, content FROM vector_chunks ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------- Memory Cards ----------

def get_messages_by_date(date: str) -> list[dict]:
    """Get all messages for a specific date (YYYY-MM-DD)."""
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT id, conversation_id, role, content, model, created_at
               FROM messages
               WHERE date(created_at) = ?
               ORDER BY id""",
            (date,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_dates_without_cards(start_date: str = None, end_date: str = None) -> list[str]:
    """Get dates that have messages but no memory cards yet."""
    conn = get_db()
    try:
        sql = """
            SELECT DISTINCT date(created_at) as d
            FROM messages
            WHERE date(created_at) NOT IN (SELECT DISTINCT date FROM memory_cards)
              AND content IS NOT NULL AND content != ''
        """
        params = []
        if start_date:
            sql += " AND date(created_at) >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date(created_at) <= ?"
            params.append(end_date)
        sql += " ORDER BY d"
        rows = conn.execute(sql, params).fetchall()
        return [r[0] for r in rows if r[0]]
    finally:
        conn.close()


def get_all_message_dates(start_date: str = None, end_date: str = None) -> list[str]:
    """Get all distinct message dates that have non-empty content."""
    conn = get_db()
    try:
        sql = """
            SELECT DISTINCT date(created_at) as d
            FROM messages
            WHERE content IS NOT NULL AND content != ''
        """
        params = []
        if start_date:
            sql += " AND date(created_at) >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date(created_at) <= ?"
            params.append(end_date)
        sql += " ORDER BY d"
        rows = conn.execute(sql, params).fetchall()
        return [r[0] for r in rows if r[0]]
    finally:
        conn.close()


def save_memory_card(date: str, summary: str, tags: str = "",
                     conversation_ids: str = "",
                     msg_id_start: int = 0, msg_id_end: int = 0) -> int:
    """Save a memory card, returns the card ID."""
    conn = get_db()
    try:
        cursor = conn.execute(
            """INSERT INTO memory_cards
               (date, summary, tags, conversation_ids, msg_id_start, msg_id_end)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (date, summary, tags, conversation_ids, msg_id_start, msg_id_end)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_cards_by_date(date: str) -> list[dict]:
    """Get all memory cards for a specific date."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM memory_cards WHERE date = ? ORDER BY id",
            (date,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_unembedded_cards(limit: int = 50) -> list[dict]:
    """Get cards that don't have embeddings yet."""
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT id, summary FROM memory_cards
               WHERE has_embedding = 0 ORDER BY id LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def mark_cards_embedded(card_ids: list[int]):
    """Mark cards as having embeddings."""
    if not card_ids:
        return
    conn = get_db()
    try:
        placeholders = ",".join("?" * len(card_ids))
        conn.execute(
            f"UPDATE memory_cards SET has_embedding = 1 WHERE id IN ({placeholders})",
            card_ids
        )
        conn.commit()
    finally:
        conn.close()


def get_cards_by_ids(card_ids: list[int]) -> list[dict]:
    """Fetch memory cards by IDs."""
    if not card_ids:
        return []
    conn = get_db()
    try:
        placeholders = ",".join("?" * len(card_ids))
        rows = conn.execute(
            f"SELECT * FROM memory_cards WHERE id IN ({placeholders})",
            card_ids
        ).fetchall()
        id_to_row = {dict(r)["id"]: dict(r) for r in rows}
        return [id_to_row[cid] for cid in card_ids if cid in id_to_row]
    finally:
        conn.close()


def get_all_cards() -> list[dict]:
    """Get all memory cards (for full rebuild)."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT id, summary FROM memory_cards ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_cards_full() -> list[dict]:
    """Get full memory card records ordered by date then id."""
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT id, date, summary, tags, conversation_ids,
                      msg_id_start, msg_id_end, has_embedding, created_at
               FROM memory_cards
               ORDER BY date, id"""
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def export_memory_cards_backup(label: str | None = None) -> str:
    """Export current memory_cards records to a JSON backup file."""
    os.makedirs(DB_BACKUP_DIR, exist_ok=True)
    stamp = label or datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(DB_BACKUP_DIR, f"memory_cards_{stamp}.json")
    cards = get_all_cards_full()
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
    return dest


def replace_all_memory_cards(cards: list[dict]) -> list[dict]:
    """
    Replace the entire memory_cards table with the provided dataset.
    Each card may include an explicit id/has_embedding/created_at.
    """
    conn = get_db()
    try:
        conn.execute("DELETE FROM memory_cards")
        for idx, card in enumerate(cards, start=1):
            card_id = int(card.get("id") or idx)
            has_embedding = int(card.get("has_embedding", 0))
            created_at = card.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                """INSERT INTO memory_cards
                   (id, date, summary, tags, conversation_ids,
                    msg_id_start, msg_id_end, has_embedding, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    card_id,
                    card["date"],
                    card["summary"],
                    card.get("tags", ""),
                    card.get("conversation_ids", ""),
                    int(card.get("msg_id_start", 0)),
                    int(card.get("msg_id_end", 0)),
                    has_embedding,
                    created_at,
                )
            )
        conn.commit()
        rows = conn.execute(
            """SELECT id, date, summary, tags, conversation_ids,
                      msg_id_start, msg_id_end, has_embedding, created_at
               FROM memory_cards
               ORDER BY date, id"""
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_recent_cards(days: int = 3) -> list[dict]:
    """Get memory cards from the last N days (unconditional, for session context)."""
    conn = get_db()
    try:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = conn.execute(
            """SELECT * FROM memory_cards
               WHERE date >= ?
               ORDER BY date DESC""",
            (cutoff,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_recent_cross_window_messages(limit: int = 30,
                                     exclude_conversation_id: str | None = None) -> list[dict]:
    """
    Get recent messages from today for cross-window context.
    Returns the last N messages from today, across all conversation windows.
    """
    conn = get_db()
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        sql = """SELECT role, content, created_at, conversation_id
                 FROM messages
                 WHERE created_at >= ?
                   AND content IS NOT NULL AND content != ''"""
        params: list = [today]
        if exclude_conversation_id:
            sql += " AND conversation_id != ?"
            params.append(exclude_conversation_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def get_card_count() -> dict:
    """Get memory card statistics."""
    conn = get_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM memory_cards").fetchone()[0]
        embedded = conn.execute(
            "SELECT COUNT(*) FROM memory_cards WHERE has_embedding = 1"
        ).fetchone()[0]
        dates = conn.execute(
            "SELECT COUNT(DISTINCT date) FROM memory_cards"
        ).fetchone()[0]
        return {"total": total, "embedded": embedded, "dates": dates}
    finally:
        conn.close()


def _json_dump(value) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def _row_to_dict(row):
    return dict(row) if row is not None else None


# ---------- New Memory Architecture ----------

def get_last_sliced_message_id() -> int:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(msg_id_end), 0) AS max_id FROM memory_slices"
        ).fetchone()
        return int(row["max_id"] if row else 0)
    finally:
        conn.close()


def get_next_memory_slice_index() -> int:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(slice_index), 0) AS max_idx FROM memory_slices"
        ).fetchone()
        return int((row["max_idx"] if row else 0) or 0) + 1
    finally:
        conn.close()


def get_unsliced_messages(limit: int = 500) -> list[dict]:
    conn = get_db()
    try:
        last_id = get_last_sliced_message_id()
        rows = conn.execute(
            """SELECT id, conversation_id, role, content, model, provider, created_at
               FROM messages
               WHERE id > ?
                 AND content IS NOT NULL AND trim(content) != ''
               ORDER BY id
               LIMIT ?""",
            (last_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_messages_by_range(msg_id_start: int, msg_id_end: int) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT id, conversation_id, role, content, model, provider, created_at
               FROM messages
               WHERE id BETWEEN ? AND ?
               ORDER BY id""",
            (msg_id_start, msg_id_end)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_memory_slices_overlapping_range(msg_id_start: int, msg_id_end: int,
                                        limit: int = 10) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT * FROM memory_slices
               WHERE status = 'active'
                 AND msg_id_end >= ?
                 AND msg_id_start <= ?
               ORDER BY msg_id_end DESC
               LIMIT ?""",
            (msg_id_start, msg_id_end, limit)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def save_memory_slice(conversation_id: str, slice_index: int, msg_id_start: int,
                      msg_id_end: int, message_count: int, summary: str,
                      tags: str = "", conversation_ids: str = "",
                      status: str = "active", source_long_memory_id: int | None = None,
                      has_embedding: int = 0, meta_json=None) -> int:
    conn = get_db()
    try:
        existing = conn.execute(
            "SELECT id FROM memory_slices WHERE msg_id_start = ? AND msg_id_end = ?",
            (msg_id_start, msg_id_end)
        ).fetchone()
        if existing:
            return int(existing["id"])
        cursor = conn.execute(
            """INSERT INTO memory_slices
               (conversation_id, conversation_ids, slice_index, msg_id_start, msg_id_end,
                message_count, summary, tags, status, source_long_memory_id,
                has_embedding, meta_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                conversation_id,
                conversation_ids,
                slice_index,
                msg_id_start,
                msg_id_end,
                message_count,
                summary,
                tags,
                status,
                source_long_memory_id,
                has_embedding,
                _json_dump(meta_json),
            )
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_memory_slice(slice_id: int) -> dict | None:
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM memory_slices WHERE id = ?", (slice_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def list_memory_slices(status: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    conn = get_db()
    try:
        params: list = []
        sql = "SELECT * FROM memory_slices"
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def count_memory_slices(status: str | None = None) -> int:
    conn = get_db()
    try:
        if status:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM memory_slices WHERE status = ?", (status,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS c FROM memory_slices").fetchone()
        return int(row["c"] if row else 0)
    finally:
        conn.close()


def update_memory_slice(slice_id: int, **updates) -> dict | None:
    allowed = {
        "summary", "tags", "status", "source_long_memory_id",
        "has_embedding", "meta_json",
    }
    fields = []
    params = []
    for key, value in updates.items():
        if key not in allowed:
            continue
        fields.append(f"{key} = ?")
        if key == "meta_json":
            params.append(_json_dump(value))
        else:
            params.append(value)
    if not fields:
        return get_memory_slice(slice_id)
    params.extend([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), slice_id])
    conn = get_db()
    try:
        conn.execute(
            f"UPDATE memory_slices SET {', '.join(fields)}, updated_at = ? WHERE id = ?",
            params
        )
        conn.commit()
        row = conn.execute("SELECT * FROM memory_slices WHERE id = ?", (slice_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def delete_memory_slice(slice_id: int):
    conn = get_db()
    try:
        conn.execute("DELETE FROM memory_slices WHERE id = ?", (slice_id,))
        conn.commit()
    finally:
        conn.close()


def get_recent_active_slices(limit: int = 4) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT * FROM memory_slices
               WHERE status = 'active'
               ORDER BY id DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def get_compactable_slice_groups(group_size: int = 4) -> list[list[dict]]:
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT * FROM memory_slices
               WHERE status = 'active'
               ORDER BY id ASC"""
        ).fetchall()
        slices = [dict(r) for r in rows]
        return [slices[i:i + group_size] for i in range(0, len(slices), group_size)
                if len(slices[i:i + group_size]) == group_size]
    finally:
        conn.close()


def mark_memory_slices_compacted(slice_ids: list[int], long_memory_id: int):
    if not slice_ids:
        return
    conn = get_db()
    try:
        placeholders = ",".join("?" * len(slice_ids))
        conn.execute(
            f"""UPDATE memory_slices
                SET status = 'compacted',
                    source_long_memory_id = ?,
                    updated_at = datetime('now')
                WHERE id IN ({placeholders})""",
            [long_memory_id, *slice_ids]
        )
        conn.commit()
    finally:
        conn.close()


def save_long_term_memory(memory_type: str, title: str, content: str,
                          fact_weight: float = 1.0, half_life_days: float = 30.0,
                          source_slice_start_id: int = 0, source_slice_end_id: int = 0,
                          status: str = "active", has_embedding: int = 0,
                          meta_json=None) -> int:
    conn = get_db()
    try:
        existing = conn.execute(
            """SELECT id FROM long_term_memories
               WHERE source_slice_start_id = ? AND source_slice_end_id = ?""",
            (source_slice_start_id, source_slice_end_id)
        ).fetchone()
        if existing:
            return int(existing["id"])
        cursor = conn.execute(
            """INSERT INTO long_term_memories
               (memory_type, title, content, fact_weight, half_life_days,
                source_slice_start_id, source_slice_end_id, status, has_embedding,
                meta_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                memory_type,
                title,
                content,
                fact_weight,
                half_life_days,
                source_slice_start_id,
                source_slice_end_id,
                status,
                has_embedding,
                _json_dump(meta_json),
            )
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def link_long_term_memory_sources(long_memory_id: int, slice_ids: list[int]):
    if not slice_ids:
        return
    conn = get_db()
    try:
        conn.executemany(
            """INSERT OR IGNORE INTO long_term_memory_sources (long_term_memory_id, memory_slice_id)
               VALUES (?, ?)""",
            [(long_memory_id, slice_id) for slice_id in slice_ids]
        )
        conn.commit()
    finally:
        conn.close()


def get_long_term_memory(memory_id: int) -> dict | None:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM long_term_memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_long_term_memories_by_ids(memory_ids: list[int]) -> list[dict]:
    if not memory_ids:
        return []
    conn = get_db()
    try:
        placeholders = ",".join("?" * len(memory_ids))
        rows = conn.execute(
            f"SELECT * FROM long_term_memories WHERE id IN ({placeholders})",
            memory_ids
        ).fetchall()
        row_map = {dict(r)["id"]: dict(r) for r in rows}
        return [row_map[mid] for mid in memory_ids if mid in row_map]
    finally:
        conn.close()


def list_long_term_memories(status: str | None = None, limit: int = 50,
                            offset: int = 0) -> list[dict]:
    conn = get_db()
    try:
        params: list = []
        sql = "SELECT * FROM long_term_memories"
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def count_long_term_memories(status: str | None = None) -> int:
    conn = get_db()
    try:
        if status:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM long_term_memories WHERE status = ?", (status,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS c FROM long_term_memories").fetchone()
        return int(row["c"] if row else 0)
    finally:
        conn.close()


def update_long_term_memory(memory_id: int, **updates) -> dict | None:
    allowed = {
        "memory_type", "title", "content", "fact_weight", "half_life_days",
        "hits", "last_hit_at", "status", "has_embedding", "meta_json",
        "source_slice_start_id", "source_slice_end_id",
    }
    fields = []
    params = []
    for key, value in updates.items():
        if key not in allowed:
            continue
        fields.append(f"{key} = ?")
        if key == "meta_json":
            params.append(_json_dump(value))
        else:
            params.append(value)
    if not fields:
        return get_long_term_memory(memory_id)
    params.extend([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), memory_id])
    conn = get_db()
    try:
        conn.execute(
            f"UPDATE long_term_memories SET {', '.join(fields)}, updated_at = ? WHERE id = ?",
            params
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM long_term_memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def delete_long_term_memory(memory_id: int):
    conn = get_db()
    try:
        conn.execute("DELETE FROM long_term_memory_sources WHERE long_term_memory_id = ?", (memory_id,))
        conn.execute("DELETE FROM long_term_memories WHERE id = ?", (memory_id,))
        conn.commit()
    finally:
        conn.close()


def get_unembedded_long_term_memories(limit: int = 50) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT id, content FROM long_term_memories
               WHERE has_embedding = 0 AND status = 'active'
               ORDER BY id LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def mark_long_term_memories_embedded(memory_ids: list[int]):
    if not memory_ids:
        return
    conn = get_db()
    try:
        placeholders = ",".join("?" * len(memory_ids))
        conn.execute(
            f"""UPDATE long_term_memories
                SET has_embedding = 1, updated_at = datetime('now')
                WHERE id IN ({placeholders})""",
            memory_ids
        )
        conn.commit()
    finally:
        conn.close()


def increment_long_term_memory_hits(memory_ids: list[int]):
    if not memory_ids:
        return
    conn = get_db()
    try:
        placeholders = ",".join("?" * len(memory_ids))
        conn.execute(
            f"""UPDATE long_term_memories
                SET hits = hits + 1,
                    last_hit_at = datetime('now'),
                    updated_at = datetime('now')
                WHERE id IN ({placeholders})""",
            memory_ids
        )
        conn.commit()
    finally:
        conn.close()


def save_diary_entry(entry_date: str, title: str, content: str, mood_score: int = 5,
                     mood_label: str = "", source_msg_id_start: int = 0, source_msg_id_end: int = 0,
                     source_slice_ids: str = "", worker_run_id: int | None = None) -> int:
    conn = get_db()
    try:
        existing = conn.execute(
            "SELECT id FROM diary_entries WHERE entry_date = ?",
            (entry_date,)
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE diary_entries
                   SET title = ?, content = ?, mood_score = ?, mood_label = ?,
                       source_msg_id_start = ?, source_msg_id_end = ?, source_slice_ids = ?,
                       worker_run_id = ?, updated_at = datetime('now')
                   WHERE id = ?""",
                (
                    title, content, mood_score, mood_label, source_msg_id_start,
                    source_msg_id_end, source_slice_ids, worker_run_id, int(existing["id"])
                )
            )
            conn.commit()
            return int(existing["id"])
        cursor = conn.execute(
            """INSERT INTO diary_entries
               (entry_date, title, content, mood_score, mood_label, source_msg_id_start, source_msg_id_end,
                source_slice_ids, worker_run_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                entry_date,
                title,
                content,
                mood_score,
                mood_label,
                source_msg_id_start,
                source_msg_id_end,
                source_slice_ids,
                worker_run_id,
            )
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_diary_entry(entry_id: int) -> dict | None:
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM diary_entries WHERE id = ?", (entry_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def list_diary_entries(limit: int = 50, offset: int = 0) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT * FROM diary_entries
               ORDER BY entry_date DESC, id DESC
               LIMIT ? OFFSET ?""",
            (limit, offset)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_diary_entry(entry_id: int, **updates) -> dict | None:
    allowed = {"title", "content", "mood_score", "mood_label", "source_slice_ids"}
    fields = []
    params = []
    for key, value in updates.items():
        if key in allowed:
            fields.append(f"{key} = ?")
            params.append(value)
    if not fields:
        return get_diary_entry(entry_id)
    params.extend([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), entry_id])
    conn = get_db()
    try:
        conn.execute(
            f"UPDATE diary_entries SET {', '.join(fields)}, updated_at = ? WHERE id = ?",
            params
        )
        conn.commit()
        row = conn.execute("SELECT * FROM diary_entries WHERE id = ?", (entry_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def delete_diary_entry(entry_id: int):
    conn = get_db()
    try:
        conn.execute("DELETE FROM diary_entries WHERE id = ?", (entry_id,))
        conn.commit()
    finally:
        conn.close()


def save_persona_snapshot(snapshot_date: str, persona_json, relationship_json,
                          summary: str = "", source_diary_id: int | None = None,
                          worker_run_id: int | None = None) -> int:
    conn = get_db()
    try:
        existing = conn.execute(
            """SELECT id FROM persona_snapshots
               WHERE snapshot_date = ? AND source_diary_id IS ?""",
            (snapshot_date, source_diary_id)
        ).fetchone()
        if existing:
            return int(existing["id"])
        cursor = conn.execute(
            """INSERT INTO persona_snapshots
               (snapshot_date, persona_json, relationship_json, summary, source_diary_id, worker_run_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                snapshot_date,
                _json_dump(persona_json),
                _json_dump(relationship_json),
                summary,
                source_diary_id,
                worker_run_id,
            )
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_persona_snapshots(limit: int = 20, offset: int = 0) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT * FROM persona_snapshots
               ORDER BY snapshot_date DESC, id DESC
               LIMIT ? OFFSET ?""",
            (limit, offset)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_pending_review(review_type: str, proposed_payload, diff_summary: str = "",
                        source_snapshot_id: int | None = None, status: str = "pending") -> int:
    conn = get_db()
    try:
        existing = conn.execute(
            """SELECT id FROM pending_reviews
               WHERE review_type = ? AND source_snapshot_id IS ?""",
            (review_type, source_snapshot_id)
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE pending_reviews
                   SET proposed_payload = ?, diff_summary = ?, status = ?,
                       review_note = '', edited_payload = '', approved_payload = '',
                       reviewed_at = NULL
                   WHERE id = ?""",
                (
                    _json_dump(proposed_payload),
                    diff_summary,
                    status,
                    int(existing["id"]),
                )
            )
            conn.commit()
            return int(existing["id"])
        cursor = conn.execute(
            """INSERT INTO pending_reviews
               (review_type, source_snapshot_id, proposed_payload, diff_summary, status)
               VALUES (?, ?, ?, ?, ?)""",
            (
                review_type,
                source_snapshot_id,
                _json_dump(proposed_payload),
                diff_summary,
                status,
            )
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_pending_review(review_id: int) -> dict | None:
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM pending_reviews WHERE id = ?", (review_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def list_pending_reviews(status: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    conn = get_db()
    try:
        params: list = []
        sql = "SELECT * FROM pending_reviews"
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def append_review_history(pending_review_id: int, action: str, before_payload=None,
                          after_payload=None, note: str = ""):
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO review_history
               (pending_review_id, action, before_payload, after_payload, note)
               VALUES (?, ?, ?, ?, ?)""",
            (
                pending_review_id,
                action,
                _json_dump(before_payload) if before_payload is not None else "",
                _json_dump(after_payload) if after_payload is not None else "",
                note,
            )
        )
        conn.commit()
    finally:
        conn.close()


def update_pending_review(review_id: int, status: str, review_note: str = "",
                          approved_payload=None, edited_payload=None) -> dict | None:
    conn = get_db()
    try:
        conn.execute(
            """UPDATE pending_reviews
               SET status = ?, review_note = ?, approved_payload = ?, edited_payload = ?,
                   reviewed_at = datetime('now')
               WHERE id = ?""",
            (
                status,
                review_note,
                _json_dump(approved_payload) if approved_payload is not None else "",
                _json_dump(edited_payload) if edited_payload is not None else "",
                review_id,
            )
        )
        conn.commit()
        row = conn.execute("SELECT * FROM pending_reviews WHERE id = ?", (review_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def save_review_item(review_type: str, proposed_payload, diff_summary: str = "",
                     source_snapshot_id: int | None = None, status: str = "pending") -> int:
    return save_pending_review(
        review_type=review_type,
        proposed_payload=proposed_payload,
        diff_summary=diff_summary,
        source_snapshot_id=source_snapshot_id,
        status=status,
    )


def get_review_item(review_id: int) -> dict | None:
    return get_pending_review(review_id)


def list_review_items(status: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    return list_pending_reviews(status=status, limit=limit, offset=offset)


def update_review_item(review_id: int, status: str, review_note: str = "",
                       approved_payload=None, edited_payload=None) -> dict | None:
    return update_pending_review(
        review_id=review_id,
        status=status,
        review_note=review_note,
        approved_payload=approved_payload,
        edited_payload=edited_payload,
    )


def get_active_profile() -> dict:
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM active_profile WHERE id = 1").fetchone()
        if row is None:
            return {
                "id": 1,
                "profile_json": "{}",
                "relationship_json": "{}",
                "source_review_id": None,
                "updated_at": None,
            }
        return dict(row)
    finally:
        conn.close()


def upsert_active_profile(profile_json, relationship_json, source_review_id: int | None = None):
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO active_profile (id, profile_json, relationship_json, source_review_id, updated_at)
               VALUES (1, ?, ?, ?, datetime('now'))
               ON CONFLICT(id) DO UPDATE SET
                 profile_json = excluded.profile_json,
                 relationship_json = excluded.relationship_json,
                 source_review_id = excluded.source_review_id,
                 updated_at = datetime('now')""",
            (
                _json_dump(profile_json),
                _json_dump(relationship_json),
                source_review_id,
            )
        )
        conn.commit()
    finally:
        conn.close()


def save_worker_run(worker_name: str = "memory_worker", run_mode: str = "manual",
                    status: str = "queued", phase: str = "queued",
                    message: str = "") -> int:
    conn = get_db()
    try:
        cursor = conn.execute(
            """INSERT INTO worker_runs
               (worker_name, run_mode, status, phase, message)
               VALUES (?, ?, ?, ?, ?)""",
            (worker_name, run_mode, status, phase, message)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def update_worker_run(run_id: int, **updates) -> dict | None:
    allowed = {
        "status", "phase", "message", "completed_at",
        "token_input", "token_output", "token_total", "result_json", "error",
    }
    fields = []
    params = []
    for key, value in updates.items():
        if key not in allowed:
            continue
        fields.append(f"{key} = ?")
        if key == "result_json":
            params.append(_json_dump(value))
        else:
            params.append(value)
    if not fields:
        return get_worker_run(run_id)
    params.append(run_id)
    conn = get_db()
    try:
        conn.execute(
            f"UPDATE worker_runs SET {', '.join(fields)} WHERE id = ?",
            params
        )
        conn.commit()
        row = conn.execute("SELECT * FROM worker_runs WHERE id = ?", (run_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_worker_run(run_id: int) -> dict | None:
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM worker_runs WHERE id = ?", (run_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def list_worker_runs(limit: int = 20, offset: int = 0) -> list[dict]:
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT * FROM worker_runs
               ORDER BY id DESC
               LIMIT ? OFFSET ?""",
            (limit, offset)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_new_memory_stats() -> dict:
    conn = get_db()
    try:
        slice_total = conn.execute("SELECT COUNT(*) AS c FROM memory_slices").fetchone()["c"]
        slice_active = conn.execute(
            "SELECT COUNT(*) AS c FROM memory_slices WHERE status = 'active'"
        ).fetchone()["c"]
        long_total = conn.execute("SELECT COUNT(*) AS c FROM long_term_memories").fetchone()["c"]
        long_embedded = conn.execute(
            "SELECT COUNT(*) AS c FROM long_term_memories WHERE has_embedding = 1"
        ).fetchone()["c"]
        diaries = conn.execute("SELECT COUNT(*) AS c FROM diary_entries").fetchone()["c"]
        reviews_pending = conn.execute(
            "SELECT COUNT(*) AS c FROM pending_reviews WHERE status = 'pending'"
        ).fetchone()["c"]
        return {
            "slice_total": int(slice_total),
            "slice_active": int(slice_active),
            "long_term_total": int(long_total),
            "long_term_embedded": int(long_embedded),
            "diary_total": int(diaries),
            "reviews_pending": int(reviews_pending),
        }
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
