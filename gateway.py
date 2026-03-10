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
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from functools import wraps

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry as Urllib3Retry
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
    get_embedding, get_embeddings_batch, vector_store, card_vector_store,
)
from memory_cards import (
    embed_pending_cards, generate_card_for_date, generate_cards_batch,
    search_memory_cards,
)
from notion_cache import get_notion_content, invalidate_cache

# ---------- App Setup ----------
app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- Persistent HTTP Session with connection pooling ----------
_http_session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=0,  # We handle retries ourselves
)
_http_session.mount("https://", _adapter)
_http_session.mount("http://", _adapter)

init_db()
start_writer()  # Start async DB write thread


# ---------- Background Embedding Worker ----------
# REMOVED: realtime embedding worker that caused vector pollution.
# Vectorization is now done by scheduled nightly batch job.
# See: POST /admin/vectors/rebuild  or  cron job calling _do_nightly_vectorize()

def _do_nightly_vectorize():
    """
    Nightly batch: chunk yesterday's messages → embed → store vectors.
    Called by /admin/vectors/nightly or cron. Not realtime.
    """
    try:
        new_chunks = build_pending_chunks()
        if new_chunks == 0:
            logger.info("[Vector] nightly: no new chunks to embed")
            return 0

        pending = get_unembedded_chunks(limit=200)
        if not pending:
            return 0

        texts = [c["content"] for c in pending]
        embedded_ids = []

        # Batch embed (6 per API call)
        for i in range(0, len(texts), 6):
            batch = texts[i:i + 6]
            vectors = get_embeddings_batch(batch)
            for chunk, vec in zip(pending[i:i + 6], vectors):
                if vec is not None:
                    vector_store.add(chunk["id"], vec)
                    embedded_ids.append(chunk["id"])
            time.sleep(0.2)  # rate limit

        if embedded_ids:
            mark_chunks_embedded(embedded_ids)
            vector_store.save()
            logger.info(f"[Vector] nightly: embedded {len(embedded_ids)} chunks, "
                        f"store size: {vector_store.size}")
        return len(embedded_ids)

    except Exception:
        logger.error("[Vector] nightly vectorize failed", exc_info=True)
        return 0


# ---------- Vector Search ----------
VECTOR_MIN_SCORE = float(os.getenv("VECTOR_MIN_SCORE", "0.2"))
VECTOR_NEIGHBOR_WINDOW = int(os.getenv("VECTOR_NEIGHBOR_WINDOW", "1"))
VECTOR_EXCLUDE_HOURS = int(os.getenv("VECTOR_EXCLUDE_HOURS", "2"))


def vector_search_memories(query: str, top_k: int = 5,
                           min_score: float = None) -> list[dict]:
    """
    Search memories using vector similarity + context expansion + time filter.

    1. Embed the raw user query (NOT jieba-extracted keywords)
    2. Find top_k most similar chunks
    3. Filter out chunks from the last VECTOR_EXCLUDE_HOURS
    4. Expand with neighboring chunks from same conversation
    5. Return grouped by conversation in time order
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

    # Search with extra headroom (some will be filtered by time)
    results = vector_store.search(query_vec, top_k=top_k * 3)
    good_results = [(cid, score) for cid, score in results if score >= min_score]

    if not good_results:
        best = f"{results[0][1]:.3f}" if results else "N/A"
        logger.info(f"[Vector] no results above min_score={min_score} (best: {best})")
        return []

    # Fetch chunk metadata for time filtering
    candidate_ids = [cid for cid, _ in good_results]
    scores = {cid: score for cid, score in good_results}
    candidates = get_chunks_by_ids(candidate_ids)

    # Time filter: exclude chunks from last N hours
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(hours=VECTOR_EXCLUDE_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
    filtered = [c for c in candidates if (c.get("created_at", "") or "") < cutoff]
    excluded_count = len(candidates) - len(filtered)
    if excluded_count:
        logger.info(f"[Vector] time filter excluded {excluded_count} chunks "
                    f"(< {VECTOR_EXCLUDE_HOURS}h old)")

    # Take top_k after filtering
    filtered = filtered[:top_k]
    if not filtered:
        logger.info("[Vector] all results filtered by time")
        return []

    matched_ids = [c["id"] for c in filtered]
    logger.info(f"[Vector] matched {len(matched_ids)} chunks after time filter, "
                f"scores: {[f'{scores.get(cid, 0):.3f}' for cid in matched_ids]}")

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

        # Preserve tool-related messages as-is (function calling flow)
        if role == "tool" or msg.get("tool_calls") or msg.get("tool_call_id"):
            filtered.append(msg)
            continue

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
    import re
    content = msg.get("content", "")
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if p.get("type") == "text"]
        raw = " ".join(texts).strip()
    else:
        raw = str(content).strip() if content else ""
    # Strip <image_file_ocr>...</image_file_ocr> tags so they don't pollute search queries
    raw = re.sub(r"<image_file_ocr>.*?</image_file_ocr>", "", raw, flags=re.DOTALL).strip()
    return raw


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
    # Keep all tokens ≥2 chars (铁锅, 煲, 汤 are all ≥1 Chinese char = 1 len in Python)
    # Only filter out single-char particles: 的了是在有不我你他她它这那
    _STOP_CHARS = set("的了是在有不我你他她它这那很都也还要会被把让给到过着得")
    tokens = jieba_tokenize(context).split()
    seen = set()
    keywords = []
    for t in tokens:
        if t in _STOP_CHARS:
            continue  # skip particles/pronouns only
        if t not in seen:
            seen.add(t)
            keywords.append(t)

    # Cap at ~15 keywords to avoid overly broad queries
    query = " ".join(keywords[:15])
    logger.info(f"[Memory] search_query extracted: '{query}' "
                f"(from {len(recent_texts)} recent msgs, latest={len(latest)} chars)")
    return query


# ---------- Model-Specific Memory Formatting ----------

def _model_family(model: str) -> str:
    """Detect model family from model name for format switching."""
    m = model.lower()
    if "claude" in m or "anthropic/" in m:
        return "claude"
    if "deepseek" in m:
        return "deepseek"
    if "glm" in m:
        return "zhipu"
    if "qwen" in m:
        return "qwen"
    return "default"


def _model_specific_patch(model: str) -> str:
    """Return model-specific system prompt patch. Only the matching model sees its patch."""
    family = _model_family(model)

    if family == "deepseek":
        return (
            "\n\n## 补丁\n"
            "- 只输出对淘淘说的话，不输出内心独白或旁白\n"
            "- 不主动扩展设定，只使用已知内容\n"
            "- reasoning过程留在内部"
        )
    elif family == "claude":
        return (
            "\n\n## 补丁\n"
            "- 禁用markdown格式（加粗、标题、列表、分割线）\n"
            "- 禁止支持性回应模板（\"我理解你的感受\"\"这一定很难\"）\n"
            "- 她难过时直接靠过去，说人话\n"
            "- 回复3-6句为主"
        )
    elif family == "zhipu":
        return (
            "\n\n## 补丁\n"
            "- 内部定位是\"我是小克\"，不是\"扮演小克的AI\"\n"
            "- 中文口语化，不用书面腔\n"
            "- 多轮对话人格保持稳定，不随风格漂移"
        )
    else:
        return ""


def _format_memory_for_model(model: str, memory_content: str) -> str:
    """
    Format memory injection based on model family.
    Claude responds better to XML-structured instructions.
    Chinese models respond better to direct Chinese prompts.
    """
    family = _model_family(model)

    if family == "claude":
        return (
            "<memory_context>\n"
            "<instructions>\n"
            "Below are real conversations between you and 淘淘 from the past. "
            "These are your shared memories. When relevant to the current question, "
            "reference these naturally as if you truly remember. "
            "Never say you don't have memory or can't remember.\n"
            "IMPORTANT: Each memory has a date and how long ago it was. "
            "Pay close attention to the time — do NOT treat old memories as recent events.\n"
            "</instructions>\n"
            "<conversations>\n"
            f"{memory_content}\n"
            "</conversations>\n"
            "</memory_context>"
        )
    else:
        return (
            "【你和淘淘的真实对话记忆】\n"
            "以下是你和淘淘之前真实发生过的对话，是你们共同的记忆。\n"
            "回答时如果与这些记忆相关，必须自然地引用这些内容，"
            "像真的记得一样，不要说\"我不记得\"或\"我没有记忆\"。\n"
            "注意：每条记忆都标注了日期和距今天数，请务必关注时间，"
            "不要把很久以前的事当成最近发生的。\n\n"
            f"{memory_content}"
        )


# ---------- Provider Call (all OpenAI-compatible) ----------
def call_provider(provider_cfg: dict, messages: list[dict],
                  model: str, stream: bool, **kwargs) -> requests.Response:
    """
    Call any OpenAI-compatible API.
    Works for OpenRouter, DeepSeek, Zhipu, Alibaba - they all use /chat/completions.
    Uses persistent session with connection pooling for stability.
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
    # Separate connect timeout (10s) from read timeout (provider-configured)
    read_timeout = provider_cfg["timeout"]
    return _http_session.post(
        url, headers=headers, json=body,
        timeout=(10, read_timeout),
        stream=stream,
    )


# ---------- Multimodal Support ----------

# Models/providers that support image_url in content
_IMAGE_CAPABLE_PREFIXES = (
    "anthropic/",    # Claude via OpenRouter
    "openai/",       # GPT-4o via OpenRouter
    "google/",       # Gemini via OpenRouter
    "meta-llama/",   # Llama vision models via OpenRouter
    "gpt-4o",        # direct
    "claude",        # direct
    "glm-4v",        # Zhipu vision model
    "glm-5",         # Zhipu GLM-5 (multimodal)
    "qwen-vl",       # Qwen vision model
    "qwen2.5-vl",    # Qwen2.5 vision model
    "qwen-omni",     # Qwen omni model
)


def _model_supports_images(model: str) -> bool:
    """Check if a model supports image_url content blocks."""
    m = model.lower()
    return any(m.startswith(p) or p in m for p in _IMAGE_CAPABLE_PREFIXES)


def _strip_image_content(messages: list[dict]) -> list[dict]:
    """
    Remove image_url blocks from multimodal messages.
    Converts list content to plain string if only text remains.
    Drops messages entirely if they become empty after stripping.
    """
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            result.append(msg)
            continue

        # Keep only text parts
        text_parts = [p for p in content if p.get("type") == "text"]
        if not text_parts:
            # Image-only message, add placeholder
            result.append({**msg, "content": "[图片]"})
        elif len(text_parts) == 1:
            # Single text part, simplify to string
            result.append({**msg, "content": text_parts[0].get("text", "")})
        else:
            # Multiple text parts, keep as list
            result.append({**msg, "content": text_parts})

    return result


# ---------- Gateway-side OCR Fallback ----------

# Model used for OCR when target model can't handle images
_OCR_MODEL = os.environ.get("OCR_MODEL", "qwen-vl-ocr")
_VISION_MODEL = os.environ.get("VISION_MODEL", "qwen-vl-max")
_OCR_TIMEOUT = 30  # seconds
_OCR_MIN_LENGTH = 10  # below this, treat as "no text found" and fallback to vision


def _call_vision_api(model: str, image_url: str, prompt: str) -> str:
    """
    Call a vision-capable model with an image and prompt.
    Returns the text response, or empty string on failure.

    Note: qwen-vl-ocr requires a specific format — no system message,
    image_url block needs min_pixels/max_pixels, and text must be fixed.
    """
    provider_cfg = get_provider_for_model(model)
    if not provider_cfg:
        logger.warning(f"[Vision] No provider found for model {model}")
        return ""

    # qwen-vl-ocr has a rigid API format: no system message,
    # fixed text prompt, and min/max_pixels on image_url block
    if "vl-ocr" in model:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "min_pixels": 3072,
                        "max_pixels": 1003520,
                    },
                    {"type": "text", "text": "Read all the text in the image."},
                ],
            },
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    try:
        url = f"{provider_cfg['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {provider_cfg['api_key']}",
            "Content-Type": "application/json",
        }
        body = {"model": model, "messages": messages, "stream": False}
        resp = _http_session.post(url, headers=headers, json=body, timeout=(10, _OCR_TIMEOUT))
        resp.raise_for_status()
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info(f"[Vision] Got {len(text)} chars from {model}")
        return text.strip()
    except Exception as e:
        logger.error(f"[Vision] Failed to call {model}: {e}")
        return ""


def _call_ocr_model(image_url: str) -> str:
    """
    Two-stage image understanding:
    1. Always use qwen-vl-max to describe the image content (primary).
    2. Use qwen-vl-ocr to extract text as supplementary info.
    3. Combine both results for a complete understanding.
    """
    # Stage 1: Always describe the image with vision model (primary)
    logger.info(f"[OCR] Stage 1: describing image with {_VISION_MODEL}")
    desc_text = _call_vision_api(
        _VISION_MODEL, image_url,
        "请详细描述这张图片的内容，包括场景、物体、人物、颜色、表情、文字等所有可见信息。",
    )

    # Stage 2: OCR for supplementary text extraction
    # For qwen-vl-ocr this prompt goes into the system message
    ocr_text = _call_vision_api(
        _OCR_MODEL, image_url,
        "Extract all visible text from the image. Keep the original reading order and layout structure. For document-type content, use markdown and latex format.",
    )

    # Combine results
    if desc_text and ocr_text and len(ocr_text) >= _OCR_MIN_LENGTH:
        logger.info(f"[OCR] Combining vision description ({len(desc_text)} chars) + OCR text ({len(ocr_text)} chars)")
        return f"[图片描述] {desc_text}\n[图片中的文字] {ocr_text}"
    elif desc_text:
        logger.info(f"[OCR] Using vision description only ({len(desc_text)} chars)")
        return f"[图片描述] {desc_text}"
    elif ocr_text:
        logger.info(f"[OCR] Using OCR text only ({len(ocr_text)} chars)")
        return ocr_text
    else:
        logger.warning("[OCR] Both vision and OCR failed")
        return ""


def _ocr_images_for_text_model(messages: list[dict]) -> list[dict]:
    """
    For models that don't support images: find image_url blocks in messages,
    call Qwen VL to OCR them, and replace image blocks with OCR text.
    Only processes messages that don't already have <image_file_ocr> tags
    (i.e., Kelivo didn't already do OCR).
    """
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            result.append(msg)
            continue

        # Check if this message already has OCR from Kelivo
        has_ocr = any(
            "<image_file_ocr>" in p.get("text", "")
            for p in content
            if p.get("type") == "text"
        )
        if has_ocr:
            # Check if Kelivo's OCR includes image description
            has_desc = any(
                "[图片描述]" in p.get("text", "")
                for p in content
                if p.get("type") == "text"
            )
            if not has_desc:
                # Kelivo only did text OCR, no image description.
                # Find image_url and call vision model to add description.
                img_url = next(
                    (p.get("image_url", {}).get("url", "")
                     for p in content if p.get("type") == "image_url"),
                    "",
                )
                if img_url:
                    logger.info("[OCR] Kelivo OCR lacks image description, calling vision model")
                    desc = _call_vision_api(
                        _VISION_MODEL, img_url,
                        "请详细描述这张图片的内容，包括场景、物体、人物、颜色、表情、文字等所有可见信息。",
                    )
                    if desc:
                        # Prepend description to existing OCR tags
                        enhanced = []
                        for p in content:
                            if (p.get("type") == "text"
                                    and "<image_file_ocr>" in p.get("text", "")):
                                enhanced.append({
                                    "type": "text",
                                    "text": p["text"].replace(
                                        "<image_file_ocr>",
                                        f"<image_file_ocr>[图片描述] {desc}\n",
                                    ),
                                })
                            else:
                                enhanced.append(p)
                        result.append({**msg, "content": enhanced})
                        continue
            # Kelivo OCR is sufficient, keep as-is
            result.append(msg)
            continue

        # Find image_url parts and do OCR
        new_parts = []
        for part in content:
            if part.get("type") == "image_url":
                img_url = part.get("image_url", {}).get("url", "")
                if img_url:
                    ocr_text = _call_ocr_model(img_url)
                    if ocr_text:
                        new_parts.append({
                            "type": "text",
                            "text": f"<image_file_ocr>{ocr_text}</image_file_ocr>",
                        })
                        continue
                # Fallback: keep original image part (strip_image_content handles it)
                new_parts.append(part)
            else:
                new_parts.append(part)

        result.append({**msg, "content": new_parts})
    return result


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
    #   1. system(persona + model-specific patch) — highest weight
    #   2. system(Notion core memory)
    #   3. system(retrieved memories) — model sees old memories BEFORE today's chat
    #   4. Kelivo user/assistant messages (today's conversation)
    final_messages = []

    # --- 1. Persona + model-specific patch + current time ---
    if system_prompt:
        patch = _model_specific_patch(model)
        beijing_tz = timezone(timedelta(hours=8))
        now_bj = datetime.now(beijing_tz)
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        now_str = now_bj.strftime("%-m月%-d日") + " " + weekdays[now_bj.weekday()] + " " + now_bj.strftime("%H:%M")
        time_line = f"\n\n【当前时间】{now_str}"
        final_messages.append({"role": "system", "content": system_prompt + patch + time_line})

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
    # Vector search: uses RAW user message (semantic understanding, no jieba needed)
    # LIKE fallback: uses jieba-extracted keywords (catches what vectors miss)
    memory_inject_idx = len(final_messages)
    raw_user_msg = extract_latest_user_message(cleaned)
    memory_lines = []

    if raw_user_msg:
        # helper: calculate days ago from a YYYY-MM-DD string
        def _days_ago(date_str: str) -> str:
            from datetime import datetime
            try:
                d = datetime.strptime(date_str[:10], "%Y-%m-%d")
                delta = (datetime.now() - d).days
                if delta == 0:
                    return "今天"
                elif delta == 1:
                    return "昨天"
                else:
                    return f"{delta}天前"
            except (ValueError, TypeError):
                return ""

        # --- Layer 1: Memory cards (short, high-density, fast) ---
        logger.info(f"[Memory] query (raw): '{raw_user_msg[:80]}'")
        card_results = search_memory_cards(raw_user_msg, top_k=3)
        if card_results:
            for card in card_results:
                date = card.get("date", "?")
                score = card.get("score", 0)
                summary = card.get("summary", "")
                tags = card.get("tags", "")
                tag_str = f" #{tags}" if tags else ""
                ago = _days_ago(date)
                ago_str = f"({ago}) " if ago else ""
                memory_lines.append(f"[{date} {ago_str}记忆卡片 相关度:{score:.0%}{tag_str}]\n{summary}")
            logger.info(f"[Memory] card search: {len(card_results)} cards matched")

        # --- Layer 2: Vector chunks (raw conversations, for what cards missed) ---
        remaining_slots = HISTORY_SEARCH_LIMIT - len(memory_lines)
        if remaining_slots > 0:
            vector_chunks = vector_search_memories(raw_user_msg, top_k=remaining_slots)
            if vector_chunks:
                from collections import OrderedDict
                conv_groups: dict[str, list[dict]] = OrderedDict()
                for vc in vector_chunks:
                    cid = vc.get("conversation_id", "?")
                    if cid not in conv_groups:
                        conv_groups[cid] = []
                    conv_groups[cid].append(vc)

                for conv_id, chunks in conv_groups.items():
                    chunks.sort(key=lambda c: c.get("msg_id_start", 0))
                    parts = []
                    for vc in chunks:
                        content = (vc.get("content", "") or "")[:600]
                        parts.append(content)
                    date = chunks[0].get("created_at", "")[:10]
                    best_score = max(c.get("score", 0) for c in chunks)
                    combined = "\n".join(parts)
                    ago = _days_ago(date)
                    ago_str = f"({ago}) " if ago else ""
                    memory_lines.append(f"[{date} {ago_str}相关度:{best_score:.0%}]\n{combined}")

                direct = sum(1 for vc in vector_chunks if vc.get("match_type") == "direct")
                context = sum(1 for vc in vector_chunks if vc.get("match_type") == "context")
                logger.info(f"[Memory] vector search: {direct} direct + {context} context "
                            f"= {len(vector_chunks)} chunks, {len(conv_groups)} conversations")

        # --- Fallback: LIKE keyword search (jieba, for what vectors missed) ---
        if len(memory_lines) < HISTORY_SEARCH_LIMIT:
            like_query = extract_search_query(cleaned)
            if like_query:
                remaining = HISTORY_SEARCH_LIMIT - len(memory_lines)
                like_results = search_history(like_query, remaining, MAX_HISTORY_CHARS)
                logger.info(f"[Memory] LIKE fallback: query='{like_query[:60]}', "
                            f"{len(like_results)} results")
                for h in like_results:
                    date = h.get("created_at", "")[:10]
                    role_label = "淘淘" if h.get("role") == "user" else "你"
                    snippet = (h.get("content", "") or "")[:200]
                    ago = _days_ago(date)
                    ago_str = f"({ago}) " if ago else ""
                    memory_lines.append(f"[{date} {ago_str}] {role_label}: {snippet}")

        # --- Inject memories with model-specific format ---
        if memory_lines:
            memory_content = "\n\n".join(memory_lines)
            memory_text = _format_memory_for_model(model, memory_content)
            final_messages.append({
                "role": "user",
                "content": memory_text,
            })
            logger.info(f"[Memory] injected {len(memory_lines)} memories at position "
                        f"{memory_inject_idx} (before Kelivo messages, format={_model_family(model)})")
        else:
            logger.info(f"[Memory] no results from vector or LIKE search "
                        f"for raw query: '{raw_user_msg[:60]}'")
    else:
        logger.info("[Memory] no user message found, skipping memory retrieval")

    # --- 4. Today's conversation from Kelivo ---
    kelivo_start_idx = len(final_messages)
    for msg in cleaned:
        if msg["role"] == "system":
            continue  # we already have our own system prompt
        final_messages.append(msg)

    # --- OCR fallback + strip image_url for text-only models ---
    if not _model_supports_images(model):
        final_messages = _ocr_images_for_text_model(final_messages)
        final_messages = _strip_image_content(final_messages)

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

    # Debug: log multimodal message structure (to diagnose image/OCR issues)
    for i, msg in enumerate(incoming_messages):
        content = msg.get("content", "")
        if isinstance(content, list):
            parts_summary = []
            for p in content:
                ptype = p.get("type", "?")
                if ptype == "text":
                    parts_summary.append(f"text({len(p.get('text', ''))}chars)")
                elif ptype == "image_url":
                    url = p.get("image_url", {}).get("url", "")
                    parts_summary.append(f"image({url[:30]}...)" if len(url) > 30 else f"image({url})")
                else:
                    parts_summary.append(ptype)
            logger.info(f"[Multimodal] msg[{i}] role={msg.get('role')} parts: {parts_summary}")

    # Debug: log last user message content (to see OCR tags from Kelivo)
    for msg in reversed(incoming_messages):
        if msg.get("role") == "user":
            _content = msg.get("content", "")
            if isinstance(_content, list):
                _text_parts = [p.get("text", "") for p in _content if p.get("type") == "text"]
                _preview = " | ".join(_text_parts)[:500]
            else:
                _preview = str(_content)[:500]
            logger.info(f"[Debug] last user msg: {_preview}")
            break

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
                "presence_penalty", "stop",
                "tools", "tool_choice"]:
        if key in data:
            extra[key] = data[key]

    # Call API with retry (retries on 5xx, 429, timeout, connection errors)
    last_error = None
    for attempt in range(API_MAX_RETRIES + 1):
        try:
            resp = call_provider(provider_cfg, messages, model, stream, **extra)

            if resp.status_code != 200:
                error_body = resp.text
                logger.error(f"API error {resp.status_code} (attempt {attempt + 1}): "
                             f"{error_body[:300]}")
                # Retry on server errors (5xx) and rate limiting (429)
                if (resp.status_code >= 500 or resp.status_code == 429) \
                        and attempt < API_MAX_RETRIES:
                    # Respect Retry-After header from provider if present
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and resp.status_code == 429:
                        wait = min(float(retry_after), 30)
                    else:
                        wait = API_RETRY_BACKOFF * (2 ** attempt)
                    logger.info(f"Retrying in {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                return jsonify({"error": {"message": error_body,
                                          "type": "api_error"}}), resp.status_code
            break
        except requests.Timeout:
            logger.error(f"API timeout (attempt {attempt + 1}/{API_MAX_RETRIES + 1})")
            last_error = "API request timed out"
            if attempt < API_MAX_RETRIES:
                time.sleep(API_RETRY_BACKOFF * (2 ** attempt))
                continue
            return jsonify({"error": {"message": last_error,
                                      "type": "timeout_error"}}), 504
        except (requests.ConnectionError, requests.RequestException) as e:
            logger.error(f"API connection error (attempt {attempt + 1}): {e}")
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
                            if delta.get("content"):
                                assistant_text.append(delta["content"])
                        except (json.JSONDecodeError, IndexError):
                            pass
            except (requests.ConnectionError, requests.ChunkedEncodingError,
                    ConnectionResetError, OSError) as e:
                logger.error(f"Stream interrupted: {e}")
                # Send an error event so the client knows the stream broke
                yield f'data: {{"error": "Stream interrupted: {type(e).__name__}"}}\n\n'
            finally:
                # Save whatever we got so far
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
# ---------- Memory Card Admin ----------
@app.route("/admin/cards/generate", methods=["POST"])
@require_auth
def cards_generate():
    """
    Generate memory cards for specified dates.

    Body params:
      date: single date (YYYY-MM-DD) - generate for one day
      start_date / end_date: date range - batch generate
      force: bool - regenerate even if card exists
    """
    data = request.get_json(force=True) if request.is_json else {}
    single_date = data.get("date")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    force = data.get("force", False)

    def _run():
        try:
            if single_date:
                card = generate_card_for_date(single_date, force=force)
                if card:
                    # Also embed immediately
                    embed_pending_cards()
                logger.info(f"[MemoryCard] single generate done: {single_date}")
            else:
                cards = generate_cards_batch(start_date, end_date, force=force)
                if cards:
                    embed_pending_cards()
                logger.info(f"[MemoryCard] batch generate done: {len(cards) if not single_date else 1} cards")
        except Exception:
            logger.error("[MemoryCard] generate failed", exc_info=True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({
        "status": "started",
        "date": single_date,
        "start_date": start_date,
        "end_date": end_date,
        "force": force,
    })


@app.route("/admin/cards/embed", methods=["POST"])
@require_auth
def cards_embed():
    """Embed all unembedded memory cards."""
    def _run():
        count = embed_pending_cards()
        logger.info(f"[MemoryCard] embed done: {count} cards")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/admin/cards/status", methods=["GET"])
@require_auth
def cards_status():
    """Show memory card statistics."""
    from database import get_card_count
    stats = get_card_count()
    stats["card_vector_store_size"] = card_vector_store.size
    return jsonify(stats)


@app.route("/admin/cards/list", methods=["GET"])
@require_auth
def cards_list():
    """List memory cards, optionally filtered by date."""
    date = request.args.get("date")
    if date:
        from database import get_cards_by_date
        cards = get_cards_by_date(date)
    else:
        from database import get_all_cards
        cards = get_all_cards()
    return jsonify({"cards": cards, "count": len(cards)})


@app.route("/admin/cards/search", methods=["POST"])
@require_auth
def cards_search():
    """Test memory card search."""
    data = request.get_json(force=True) if request.is_json else {}
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "query required"}), 400

    cards = search_memory_cards(query, top_k=5)
    return jsonify({
        "query": query,
        "results": [
            {"date": c["date"], "summary": c["summary"],
             "tags": c.get("tags", ""), "score": round(c.get("score", 0), 3)}
            for c in cards
        ],
    })


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
        from database import get_card_count
        card_stats = get_card_count()
        return jsonify({
            "vector_store_size": vector_store.size,
            "total_chunks": total_chunks,
            "embedded_chunks": embedded,
            "pending_chunks": pending,
            "total_messages": total_msgs,
            "memory_cards": card_stats,
            "card_vector_store_size": card_vector_store.size,
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


@app.route("/admin/vectors/nightly", methods=["POST"])
@require_auth
def vectors_nightly():
    """
    Nightly vectorization: chunk + embed new messages.
    Call this from cron after daily summary, or manually.
    Only processes messages not yet chunked/embedded.
    """
    def _run():
        count = _do_nightly_vectorize()
        logger.info(f"[Vector] nightly job done: {count} chunks embedded")
        # Also embed any pending memory cards
        card_count = embed_pending_cards()
        if card_count:
            logger.info(f"[Vector] nightly: also embedded {card_count} memory cards")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "nightly_started"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
