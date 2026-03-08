"""
AI Chat Gateway - handles Kelivo/OpenAI-compatible clients

Fixes applied:
  1. Kelivo sends full history -> gateway does NOT append its own history (no duplication)
  2. Only stores latest user + assistant message (not Kelivo's bundled history)
  3. Filters out Kelivo hidden system messages (Memory Tool, etc.)
  4. Logs raw API response for debugging empty replies
  5. Supports both streaming and non-streaming modes
  6. memory_context placed in user message, not system
  7. Provider adapter unifies response format across OpenAI/Anthropic/DeepSeek
  8. Notion content cached with TTL
  9. Health check endpoint
  10. Rate limiting
"""
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from functools import wraps

import requests
from flask import Flask, Response, jsonify, request, stream_with_context

import config
from config import (
    API_MAX_RETRIES,
    API_RETRY_BACKOFF,
    CORS_ALLOWED_ORIGINS,
    GATEWAY_AUTH_TOKEN,
    HISTORY_SEARCH_LIMIT,
    MAX_HISTORY_CHARS,
    MAX_NOTION_CHARS,
    PROVIDERS,
    RATE_LIMIT_RPD,
    RATE_LIMIT_RPM,
    get_provider_for_model,
    load_system_prompt,
    reload_providers,
)
from database import (
    backup_database, build_pending_chunks, get_chunks_by_ids, get_neighbor_chunks,
    get_unembedded_chunks, init_db, mark_chunks_embedded, save_message,
    search_history, start_writer,
)
from embedding import (
    get_embedding, get_embeddings_batch, vector_store,
)
from notion_cache import get_notion_content, invalidate_cache

# ---------- App Setup ----------
app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

init_db()
start_writer()  # Start async DB write thread


# ---------- Background Embedding Worker ----------
def _embedding_worker():
    """Background thread: chunk new messages → embed → store vectors."""
    import time as _time
    logger.info("[Vector] background embedding worker started")
    while True:
        try:
            # 1. Create chunks from new messages
            new_chunks = build_pending_chunks()

            # 2. Embed un-embedded chunks
            pending = get_unembedded_chunks(limit=30)
            if pending:
                texts = [c["content"] for c in pending]
                vectors = get_embeddings_batch(texts)

                embedded_ids = []
                for chunk, vec in zip(pending, vectors):
                    if vec is not None:
                        vector_store.add(chunk["id"], vec)
                        embedded_ids.append(chunk["id"])

                if embedded_ids:
                    mark_chunks_embedded(embedded_ids)
                    vector_store.save()
                    logger.info(f"[Vector] embedded {len(embedded_ids)} chunks, "
                                f"store size: {vector_store.size}")

        except Exception:
            logger.error("[Vector] embedding worker error", exc_info=True)

        _time.sleep(60)  # run every 60 seconds


_embedding_thread = threading.Thread(target=_embedding_worker, daemon=True)
_embedding_thread.start()


# ---------- Vector Search ----------
VECTOR_MIN_SCORE = float(os.getenv("VECTOR_MIN_SCORE", "0.2"))
VECTOR_NEIGHBOR_WINDOW = int(os.getenv("VECTOR_NEIGHBOR_WINDOW", "1"))


def vector_search_memories(query: str, top_k: int = 5,
                           min_score: float = None) -> list[dict]:
    """
    Search memories using vector similarity + context expansion.

    1. Find top_k most similar chunks
    2. For each matched chunk, also pull ±VECTOR_NEIGHBOR_WINDOW neighboring
       chunks from the same conversation (restores full topic context)
    3. Merge and deduplicate, sorted by conversation → time order

    Returns list of chunk dicts with content.
    """
    if min_score is None:
        min_score = VECTOR_MIN_SCORE

    if vector_store.size == 0:
        logger.info("[Vector] store empty, skipping vector search")
        return []

    query_vec = get_embedding(query)
    if query_vec is None:
        logger.warning("[Vector] failed to embed query, skipping vector search")
        return []

    results = vector_store.search(query_vec, top_k=top_k)
    # Filter by minimum score
    good_results = [(cid, score) for cid, score in results if score >= min_score]

    if not good_results:
        best = f"{results[0][1]:.3f}" if results else "N/A"
        logger.info(f"[Vector] no results above min_score={min_score} (best: {best})")
        return []

    matched_ids = [cid for cid, _ in good_results]
    scores = {cid: score for cid, score in good_results}

    logger.info(f"[Vector] matched {len(matched_ids)} chunks, "
                f"scores: {[f'{s:.3f}' for _, s in good_results]}")

    # Context expansion: pull in neighboring chunks from same conversations
    if VECTOR_NEIGHBOR_WINDOW > 0:
        chunks = get_neighbor_chunks(matched_ids, window=VECTOR_NEIGHBOR_WINDOW)
        expanded = len(chunks) - len(matched_ids)
        if expanded > 0:
            logger.info(f"[Vector] context expansion: +{expanded} neighbor chunks "
                        f"(window=±{VECTOR_NEIGHBOR_WINDOW}), total={len(chunks)}")
    else:
        chunks = get_chunks_by_ids(matched_ids)

    # Attach scores (neighbors get a "context" marker)
    for c in chunks:
        if c["id"] in scores:
            c["score"] = scores[c["id"]]
            c["match_type"] = "direct"
        else:
            c["score"] = 0.0
            c["match_type"] = "context"

    return chunks


# ---------- Rate Limiter (in-memory, simple) ----------
_rate_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(key: str = "global") -> tuple[bool, str]:
    """Returns (allowed, error_message)."""
    now = time.time()
    timestamps = _rate_store[key]
    # Clean old entries
    _rate_store[key] = [t for t in timestamps if t > now - 86400]
    timestamps = _rate_store[key]

    rpm_count = sum(1 for t in timestamps if t > now - 60)
    rpd_count = len(timestamps)

    if rpm_count >= RATE_LIMIT_RPM:
        return False, f"Rate limit exceeded: {RATE_LIMIT_RPM} requests/minute"
    if rpd_count >= RATE_LIMIT_RPD:
        return False, f"Rate limit exceeded: {RATE_LIMIT_RPD} requests/day"
    return True, ""


# ---------- Auth Middleware ----------
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not GATEWAY_AUTH_TOKEN:
            return f(*args, **kwargs)
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else ""
        if token != GATEWAY_AUTH_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ---------- CORS ----------
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    if "*" in CORS_ALLOWED_ORIGINS or origin in CORS_ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin or "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ---------- Health Check ----------
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring."""
    provider_status = {}
    for name, cfg in config.PROVIDERS.items():
        provider_status[name] = {
            "configured": bool(cfg.get("api_key")),
            "prefixes": cfg.get("prefixes", []),
        }
    return jsonify({
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "providers": provider_status,
    })


# ---------- Message Filtering (Kelivo cleanup) ----------

# Regex to strip Kelivo's injected "tool guide" preamble from user/system messages.
# Matches blocks like "你是一个无状态的大模型...工具说明...使用指南..." up to the
# first real user content or end of text.  Keeps actual tool_call parameters intact.
_KELIVO_TOOL_GUIDE_PATTERNS = [
    # "你是一个无状态的大模型..." block (greedy up to a blank line or end)
    re.compile(
        r'你是一个无状态的.*?(?=\n\n|\Z)',
        re.DOTALL,
    ),
    # "## 工具使用指南" / "## Tool Guide" style markdown sections
    re.compile(
        r'#{1,3}\s*(?:工具使用指南|工具说明|Tool Guide|Tool Instructions).*?(?=\n#{1,3}\s|\Z)',
        re.DOTALL,
    ),
    # "以下是你可以使用的工具" / "Available tools:" intro paragraphs
    re.compile(
        r'(?:以下是你可以使用的工具|以下是可用的工具|Available tools)[：:].*?(?=\n\n|\Z)',
        re.DOTALL,
    ),
    # Generic "你没有记忆能力" / "你无法记住之前的对话" anti-memory statements
    re.compile(
        r'(?:你没有记忆能力|你无法记住|你不具备记忆|每次对话都是全新的|你没有任何关于用户的记忆).*?(?=\n|\Z)',
        re.DOTALL,
    ),
]


def _strip_kelivo_tool_guide(text: str) -> str:
    """Remove Kelivo's injected tool-guide preamble from message content."""
    result = text
    for pat in _KELIVO_TOOL_GUIDE_PATTERNS:
        result = pat.sub('', result)
    # Collapse leftover blank lines
    result = re.sub(r'\n{3,}', '\n\n', result).strip()
    return result


def filter_kelivo_messages(messages: list[dict]) -> list[dict]:
    """
    Filter out Kelivo's hidden system messages and strip tool-guide preambles:
    - Remove Memory Tool injections (entire message)
    - Strip "你是一个无状态的大模型..." tool guides from message content
    - Remove empty content messages
    - Skip duplicate system prompts (keep only our own)
    """
    filtered = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle content that's a list (multimodal messages with images)
        if isinstance(content, list):
            # Strip tool guide from text parts
            new_parts = []
            for part in content:
                if part.get("type") == "text" and part.get("text"):
                    cleaned_text = _strip_kelivo_tool_guide(part["text"])
                    if cleaned_text:
                        new_parts.append({**part, "text": cleaned_text})
                    # drop empty text parts
                else:
                    new_parts.append(part)
            if new_parts:
                filtered.append({**msg, "content": new_parts})
            continue

        content_str = str(content).strip() if content else ""

        # Skip empty messages
        if not content_str:
            continue

        # Skip Kelivo Memory Tool hidden messages (entire message)
        if role == "system" and any(kw in content_str for kw in [
            "Memory Tool", "memory_tool", "MEMORY:", "<memory>",
            "User preferences:", "Previous context:"
        ]):
            logger.info("Filtered out Kelivo Memory Tool message")
            continue

        # Strip tool-guide preamble from content
        cleaned_content = _strip_kelivo_tool_guide(content_str)
        if not cleaned_content:
            logger.info(f"Filtered out Kelivo tool-guide-only {role} message")
            continue

        filtered.append({**msg, "content": cleaned_content})
    return filtered


def extract_latest_user_message(messages: list[dict]) -> str:
    """Get the last user message content (text only)."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text from multimodal content
                texts = [p.get("text", "") for p in content if p.get("type") == "text"]
                return " ".join(texts).strip()
            return str(content).strip()
    return ""


def _extract_msg_text(msg: dict) -> str:
    """Extract plain text from a message (handles both str and multimodal list)."""
    content = msg.get("content", "")
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if p.get("type") == "text"]
        return " ".join(texts).strip()
    return str(content).strip() if content else ""


def extract_search_query(messages: list[dict], max_msgs: int = 5) -> str:
    """
    Build a search query from the recent conversation context.

    Short user messages like "不是这个呢" / "对" / "嗯" are useless for search.
    Strategy:
      1. Collect the last max_msgs user+assistant messages
      2. Use jieba to extract keywords from all of them
      3. Deduplicate and return as a combined query string

    This gives the search engine enough context even when the latest
    message is just "不是这个".
    """
    from database import jieba_tokenize

    # Collect recent messages (user + assistant only)
    recent_texts = []
    count = 0
    for msg in reversed(messages):
        if msg.get("role") not in ("user", "assistant"):
            continue
        text = _extract_msg_text(msg)
        if not text:
            continue
        recent_texts.append(text)
        count += 1
        if count >= max_msgs:
            break

    if not recent_texts:
        return ""

    # The latest user message
    latest = recent_texts[0] if recent_texts else ""

    # If the latest message is already substantial (>10 chars), use it directly
    # combined with a bit of context from the previous message
    if len(latest) > 10:
        # Still add one prior message for better context
        context = " ".join(recent_texts[:2])
    else:
        # Short message - pull in more context
        context = " ".join(recent_texts)

    # Tokenize and deduplicate keywords, keep order
    tokens = jieba_tokenize(context).split()
    seen = set()
    keywords = []
    for t in tokens:
        if len(t) < 2:
            continue  # skip single chars (的, 了, 是, ...)
        if t not in seen:
            seen.add(t)
            keywords.append(t)

    # Cap at ~15 keywords to avoid overly broad queries
    query = " ".join(keywords[:15])
    logger.info(f"[Memory] search_query extracted: '{query}' "
                f"(from {len(recent_texts)} recent msgs, latest={len(latest)} chars)")
    return query


# ---------- Provider Call (all OpenAI-compatible) ----------
def call_provider(provider_cfg: dict, messages: list[dict],
                  model: str, stream: bool, **kwargs) -> requests.Response:
    """
    Call any OpenAI-compatible API.
    Works for OpenRouter, DeepSeek, Zhipu, Alibaba - they all use /chat/completions.
    """
    url = f"{provider_cfg['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {provider_cfg['api_key']}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "stream": stream,
        **kwargs,
    }
    return requests.post(
        url, headers=headers, json=body,
        timeout=provider_cfg["timeout"],
        stream=stream,
    )


# ---------- Build Messages ----------
def build_messages(incoming_messages: list[dict], model: str) -> list[dict]:
    """
    Build the final message list:
    - system(persona) → system(Notion) → system(memories) → Kelivo messages
    - Memories placed BEFORE today's chat so model reads them first
    - Excludes recent 24h from search to avoid self-pollution
    """
    system_prompt = load_system_prompt()
    notion_content = get_notion_content()

    # Filter out Kelivo junk
    cleaned = filter_kelivo_messages(incoming_messages)

    # Message structure (top-down, model reads in this order):
    #   1. system(persona)           — highest weight
    #   2. system(Notion core memory)
    #   3. system(retrieved memories) — model sees old memories BEFORE today's chat
    #   4. Kelivo user/assistant messages (today's conversation)
    final_messages = []

    # --- 1. Persona ---
    if system_prompt:
        final_messages.append({"role": "system", "content": system_prompt})

    # --- 2. Notion knowledge base ---
    if notion_content:
        truncated_notion = notion_content[:MAX_NOTION_CHARS]
        if len(notion_content) > MAX_NOTION_CHARS:
            truncated_notion += "\n...(truncated)"
        final_messages.append({
            "role": "system",
            "content": f"[Core Memory from Knowledge Base]\n{truncated_notion}",
        })

    # --- 3. Retrieved memories (placed BEFORE today's chat) ---
    # Strategy: vector search (semantic) primary, LIKE search (keyword) fallback
    memory_inject_idx = len(final_messages)
    search_query = extract_search_query(cleaned)
    memory_lines = []

    if search_query:
        # Primary: vector search (semantic + context expansion)
        vector_chunks = vector_search_memories(search_query, top_k=HISTORY_SEARCH_LIMIT)
        if vector_chunks:
            # Group chunks by conversation for coherent presentation
            from collections import OrderedDict
            conv_groups: dict[str, list[dict]] = OrderedDict()
            for vc in vector_chunks:
                cid = vc.get("conversation_id", "?")
                if cid not in conv_groups:
                    conv_groups[cid] = []
                conv_groups[cid].append(vc)

            for conv_id, chunks in conv_groups.items():
                # Sort chunks within conversation by message order
                chunks.sort(key=lambda c: c.get("msg_id_start", 0))
                # Combine chunks from same conversation into one memory block
                parts = []
                for vc in chunks:
                    content = (vc.get("content", "") or "")[:600]
                    parts.append(content)
                date = chunks[0].get("created_at", "")[:10]
                best_score = max(c.get("score", 0) for c in chunks)
                combined = "\n".join(parts)
                memory_lines.append(f"[{date} 相关度:{best_score:.0%}]\n{combined}")

            direct = sum(1 for vc in vector_chunks if vc.get("match_type") == "direct")
            context = sum(1 for vc in vector_chunks if vc.get("match_type") == "context")
            logger.info(f"[Memory] vector search: {direct} direct + {context} context "
                        f"= {len(vector_chunks)} chunks, {len(conv_groups)} conversations")

        # Fallback: LIKE keyword search (catches what vectors miss)
        if len(memory_lines) < HISTORY_SEARCH_LIMIT:
            remaining = HISTORY_SEARCH_LIMIT - len(memory_lines)
            like_results = search_history(search_query, remaining, MAX_HISTORY_CHARS)
            logger.info(f"[Memory] LIKE fallback: {len(like_results)} results")
            for h in like_results:
                date = h.get("created_at", "")[:10]
                role_label = "淘淘" if h.get("role") == "user" else "你"
                snippet = (h.get("content", "") or "")[:200]
                memory_lines.append(f"[{date}] {role_label}: {snippet}")

        if memory_lines:
            memory_text = (
                "【你和淘淘的真实对话记忆】\n"
                "以下是你和淘淘之前真实发生过的对话，是你们共同的记忆。\n"
                "回答时如果与这些记忆相关，必须自然地引用这些内容，"
                "像真的记得一样，不要说\"我不记得\"或\"我没有记忆\"。\n\n"
                + "\n\n".join(memory_lines)
            )
            final_messages.append({
                "role": "user",
                "content": memory_text,
            })
            logger.info(f"[Memory] injected {len(memory_lines)} memories at position "
                        f"{memory_inject_idx} (before Kelivo messages)")
        else:
            logger.info(f"[Memory] no results from vector or LIKE search "
                        f"for query: '{search_query[:60]}'")
    else:
        logger.info("[Memory] no search_query extracted, skipping memory retrieval")

    # --- 4. Today's conversation from Kelivo ---
    kelivo_start_idx = len(final_messages)
    for msg in cleaned:
        if msg["role"] == "system":
            continue  # we already have our own system prompt
        final_messages.append(msg)

    # --- Final structure log ---
    roles_summary = [f"{i}:{m['role']}" for i, m in enumerate(final_messages)]
    logger.info(f"[Memory] build_messages final: {len(final_messages)} msgs, "
                f"memory@{memory_inject_idx} kelivo@{kelivo_start_idx} | "
                f"{' '.join(roles_summary)}")

    return final_messages


# ---------- Kelivo Internal Request Detection ----------
_KELIVO_INTERNAL_PATTERNS = [
    "Generate or update a brief summary",
    "generate or update a brief summary",
    "Generate a brief summary",
    "Update the summary",
    "Summarize the conversation",
    "summarize the conversation",
    "Create a title for this conversation",
    "create a title for this conversation",
]


def _is_kelivo_summary_request(messages: list[dict]) -> bool:
    """
    Detect Kelivo's internal housekeeping requests (summary generation, title
    generation, etc.) that should NOT go through the normal chat pipeline.
    """
    if not messages:
        return False
    # Check the last message (usually a system or user message with the instruction)
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        content_str = str(content) if content else ""
        if any(pat in content_str for pat in _KELIVO_INTERNAL_PATTERNS):
            return True
    return False


# ---------- Main Chat Endpoint ----------
@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
@require_auth
def chat_completions():
    if request.method == "OPTIONS":
        return "", 204

    # Rate limit (per-IP)
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
    client_ip = client_ip.split(",")[0].strip()  # first IP if behind proxy
    allowed, err_msg = _check_rate_limit(client_ip)
    if not allowed:
        return jsonify({"error": {"message": err_msg, "type": "rate_limit_error"}}), 429
    _rate_store[client_ip].append(time.time())

    data = request.get_json(force=True)
    model = data.get("model", "gpt-4o")
    stream = data.get("stream", False)
    incoming_messages = data.get("messages", [])

    logger.info(f"Request: model={model}, stream={stream}, "
                f"messages_count={len(incoming_messages)}")

    # Intercept Kelivo internal summary requests - don't waste API calls
    if _is_kelivo_summary_request(incoming_messages):
        logger.info("Intercepted Kelivo summary request, returning empty summary")
        return jsonify({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    # Route to provider
    provider_cfg = get_provider_for_model(model)
    if not provider_cfg:
        return jsonify({"error": {"message": f"Unknown model: {model}",
                                   "type": "invalid_request_error"}}), 400

    provider_name = provider_cfg["provider"]
    messages = build_messages(incoming_messages, model)

    # Extract user message for storage
    user_text = extract_latest_user_message(incoming_messages)
    conversation_id = data.get("conversation_id", str(uuid.uuid4()))

    # Save user message (only the latest, not Kelivo history)
    if user_text:
        save_message(conversation_id, "user", user_text, model, provider_name)

    # Extra params to pass through
    extra = {}
    for key in ["temperature", "max_tokens", "top_p", "frequency_penalty",
                "presence_penalty", "stop"]:
        if key in data:
            extra[key] = data[key]

    # Call API with retry
    last_error = None
    for attempt in range(API_MAX_RETRIES + 1):
        try:
            resp = call_provider(provider_cfg, messages, model, stream, **extra)

            if resp.status_code != 200:
                error_body = resp.text
                logger.error(f"API error {resp.status_code}: {error_body}")
                if resp.status_code >= 500 and attempt < API_MAX_RETRIES:
                    time.sleep(API_RETRY_BACKOFF * (2 ** attempt))
                    continue
                return jsonify({"error": {"message": error_body,
                                          "type": "api_error"}}), resp.status_code
            break
        except requests.Timeout:
            logger.error(f"API timeout (attempt {attempt + 1})")
            last_error = "API request timed out"
            if attempt < API_MAX_RETRIES:
                time.sleep(API_RETRY_BACKOFF * (2 ** attempt))
                continue
            return jsonify({"error": {"message": last_error,
                                      "type": "timeout_error"}}), 504
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            last_error = str(e)
            if attempt < API_MAX_RETRIES:
                time.sleep(API_RETRY_BACKOFF * (2 ** attempt))
                continue
            return jsonify({"error": {"message": last_error,
                                      "type": "api_error"}}), 502

    # ---------- Streaming response ----------
    if stream:
        def generate():
            assistant_text = []
            try:
                # Force UTF-8 decoding — OpenRouter may not set charset in headers,
                # causing requests to default to latin-1 and garble Chinese text
                resp.encoding = "utf-8"
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    # All providers are OpenAI-compatible: pass through SSE
                    yield line + "\n\n" if not line.endswith("\n\n") else line
                    # Collect text for DB storage
                    if line.startswith("data: ") and "[DONE]" not in line:
                        try:
                            chunk_data = json.loads(line[6:])
                            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                assistant_text.append(delta["content"])
                        except (json.JSONDecodeError, IndexError):
                            pass
            finally:
                # Save assistant response
                full_text = "".join(assistant_text)
                if full_text.strip():
                    save_message(conversation_id, "assistant", full_text,
                                model, provider_name)
                else:
                    logger.warning("Empty assistant response in stream")

        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---------- Non-streaming response ----------
    # Force UTF-8 before parsing JSON — same OpenRouter encoding issue
    resp.encoding = "utf-8"
    raw_json = resp.json()
    logger.info(f"Raw API response keys: {list(raw_json.keys())}")
    result = raw_json

    # Extract and save assistant reply
    assistant_content = ""
    choices = result.get("choices", [])
    if choices:
        assistant_content = choices[0].get("message", {}).get("content", "")

    if assistant_content:
        tokens_in = result.get("usage", {}).get("prompt_tokens", 0)
        tokens_out = result.get("usage", {}).get("completion_tokens", 0)
        save_message(conversation_id, "assistant", assistant_content,
                     model, provider_name, tokens_in, tokens_out)
    else:
        logger.warning(f"Empty assistant response. Raw: {json.dumps(raw_json)[:500]}")

    return jsonify(result)


# ---------- Model List ----------
@app.route("/v1/models", methods=["GET"])
@require_auth
def list_models():
    """Return available providers and their supported prefixes."""
    providers = []
    for name, cfg in config.PROVIDERS.items():
        if not cfg.get("api_key"):
            continue
        providers.append({
            "provider": name,
            "prefixes": cfg.get("prefixes", []),
            "base_url": cfg.get("base_url", ""),
        })
    return jsonify({"object": "list", "data": providers})


# ---------- Notion Cache Management ----------
@app.route("/admin/notion/refresh", methods=["POST"])
@require_auth
def refresh_notion():
    """Force refresh Notion cache."""
    invalidate_cache()
    content = get_notion_content(force_refresh=True)
    return jsonify({"status": "refreshed", "content_length": len(content)})


# ---------- Database Backup ----------
@app.route("/admin/backup", methods=["POST"])
@require_auth
def trigger_backup():
    """Manually trigger database backup."""
    path = backup_database()
    return jsonify({"status": "ok", "backup_path": path})


# ---------- Stats ----------
@app.route("/admin/stats", methods=["GET"])
@require_auth
def stats():
    """Basic usage stats."""
    from database import get_db
    conn = get_db()
    try:
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        today = time.strftime("%Y-%m-%d")
        today_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE created_at >= ?", (today,)
        ).fetchone()[0]
        return jsonify({
            "total_messages": msg_count,
            "total_conversations": conv_count,
            "today_messages": today_count,
        })
    finally:
        conn.close()


# ---------- Provider Hot-Reload ----------
@app.route("/admin/providers/reload", methods=["POST"])
@require_auth
def reload_providers_endpoint():
    """Hot-reload providers.json without restarting the gateway."""
    new_providers = reload_providers()
    config.PROVIDERS = new_providers
    # Update module-level reference
    import gateway
    summary = {
        name: {"prefixes": cfg.get("prefixes", []),
               "has_key": bool(cfg.get("api_key"))}
        for name, cfg in new_providers.items()
    }
    return jsonify({"status": "reloaded", "providers": summary})


# ---------- Vector Admin ----------
@app.route("/admin/vectors/status", methods=["GET"])
@require_auth
def vectors_status():
    """Show vector store status."""
    from database import get_db
    conn = get_db()
    try:
        total_chunks = conn.execute("SELECT COUNT(*) FROM vector_chunks").fetchone()[0]
        embedded = conn.execute(
            "SELECT COUNT(*) FROM vector_chunks WHERE has_embedding = 1"
        ).fetchone()[0]
        pending = total_chunks - embedded
        total_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return jsonify({
            "vector_store_size": vector_store.size,
            "total_chunks": total_chunks,
            "embedded_chunks": embedded,
            "pending_chunks": pending,
            "total_messages": total_msgs,
        })
    finally:
        conn.close()


@app.route("/admin/vectors/rebuild", methods=["POST"])
@require_auth
def vectors_rebuild():
    """
    Full rebuild: re-chunk all messages + re-embed everything.
    Runs in background, returns immediately.
    """
    from database import get_all_chunks_for_rebuild, get_db
    import numpy as _np

    def _do_rebuild():
        try:
            # Step 1: Re-chunk all messages from scratch
            conn = get_db()
            try:
                conn.execute("DELETE FROM vector_chunks")
                conn.commit()
            finally:
                conn.close()

            logger.info("[Vector] rebuild: cleared old chunks, re-chunking...")
            n_chunks = build_pending_chunks()
            logger.info(f"[Vector] rebuild: created {n_chunks} chunks")

            # Step 2: Embed all chunks
            all_chunks = get_all_chunks_for_rebuild()
            if not all_chunks:
                logger.info("[Vector] rebuild: no chunks to embed")
                return

            all_vecs = []
            all_ids = []
            batch_size = 6

            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                texts = [c["content"] for c in batch]
                vectors = get_embeddings_batch(texts)
                for chunk, vec in zip(batch, vectors):
                    if vec is not None:
                        all_vecs.append(vec)
                        all_ids.append(chunk["id"])
                # Rate limiting: ~10 calls/sec max
                if i + batch_size < len(all_chunks):
                    time.sleep(0.2)

            if all_vecs:
                vec_array = _np.stack(all_vecs)
                id_array = _np.array(all_ids, dtype=_np.int64)
                vector_store.rebuild(vec_array, id_array)
                mark_chunks_embedded(all_ids)
                logger.info(f"[Vector] rebuild complete: {len(all_vecs)} vectors")
            else:
                logger.warning("[Vector] rebuild: no vectors produced")

        except Exception:
            logger.error("[Vector] rebuild failed", exc_info=True)

    t = threading.Thread(target=_do_rebuild, daemon=True)
    t.start()

    return jsonify({
        "status": "rebuild_started",
        "message": "Rebuilding vectors in background. Check /admin/vectors/status for progress."
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
