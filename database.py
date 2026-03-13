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
            logger.info(f"[Memory] keywords: {keywords[:10]}")

            # First try exact query match
            exact_rows = conn.execute(
                f"""SELECT role, content, created_at, conversation_id
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
                    f"""SELECT role, content, created_at, conversation_id
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
                        f"""SELECT role, content, created_at, conversation_id
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
                        f"""SELECT m.role, m.content, m.created_at, m.conversation_id
                            FROM messages_fts f
                            JOIN messages m ON m.conversation_id = f.conversation_id
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


def get_recent_cross_window_messages(limit: int = 30) -> list[dict]:
    """
    Get recent messages from today for cross-window context.
    Returns the last N messages from today, across all conversation windows.
    """
    conn = get_db()
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        rows = conn.execute(
            """SELECT role, content, created_at, conversation_id
               FROM messages
               WHERE created_at >= ?
                 AND content IS NOT NULL AND content != ''
               ORDER BY id DESC LIMIT ?""",
            (today, limit)
        ).fetchall()
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
