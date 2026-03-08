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
from database import backup_database, init_db, save_message, search_history, start_writer
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
def filter_kelivo_messages(messages: list[dict]) -> list[dict]:
    """
    Filter out Kelivo's hidden system messages:
    - Memory Tool injections
    - Empty content messages
    - Duplicate system prompts (keep only our own)
    """
    filtered = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle content that's a list (multimodal messages with images)
        if isinstance(content, list):
            # Keep image messages as-is
            filtered.append(msg)
            continue

        content_str = str(content).strip() if content else ""

        # Skip empty messages
        if not content_str:
            continue

        # Skip Kelivo Memory Tool hidden messages
        if role == "system" and any(kw in content_str for kw in [
            "Memory Tool", "memory_tool", "MEMORY:", "<memory>",
            "User preferences:", "Previous context:"
        ]):
            logger.info("Filtered out Kelivo Memory Tool message")
            continue

        filtered.append(msg)
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
    - Keep our system prompt at the top
    - Filter Kelivo hidden messages
    - Add memory_context as user message (not system)
    - Pass through Kelivo's history as-is (no gateway history appending)
    """
    system_prompt = load_system_prompt()
    notion_content = get_notion_content()

    # Filter out Kelivo junk
    cleaned = filter_kelivo_messages(incoming_messages)

    # Prompt structure per plan:
    #   1. system(persona) - first, purest, highest weight
    #   2. system(Notion core memory) - separate, truncated
    #   3. user messages from Kelivo
    #   4. history context injected as user message
    final_messages = []
    if system_prompt:
        final_messages.append({"role": "system", "content": system_prompt})
    if notion_content:
        # Truncate Notion content to budget
        truncated_notion = notion_content[:MAX_NOTION_CHARS]
        if len(notion_content) > MAX_NOTION_CHARS:
            truncated_notion += "\n...(truncated)"
        final_messages.append({
            "role": "system",
            "content": f"[Core Memory from Knowledge Base]\n{truncated_notion}",
        })

    # Add cleaned Kelivo messages (skip any system messages from Kelivo)
    for msg in cleaned:
        if msg["role"] == "system":
            continue  # we already have our own system prompt
        final_messages.append(msg)

    # Search history for context, inject as user message before the last user msg
    user_query = extract_latest_user_message(cleaned)
    if user_query:
        history_results = search_history(user_query, HISTORY_SEARCH_LIMIT, MAX_HISTORY_CHARS)
        if history_results:
            memory_lines = []
            for h in history_results:
                date = h.get("created_at", "")[:10]
                role = h.get("role", "")
                snippet = (h.get("content", "") or "")[:200]
                memory_lines.append(f"[{date}] {role}: {snippet}")
            memory_text = (
                "[Historical context from previous conversations - "
                "use as reference only, prioritize current conversation]\n"
                + "\n".join(memory_lines)
            )
            # Insert before the last user message
            insert_idx = len(final_messages) - 1
            for i in range(len(final_messages) - 1, -1, -1):
                if final_messages[i].get("role") == "user":
                    insert_idx = i
                    break
            final_messages.insert(insert_idx, {
                "role": "user",
                "content": memory_text,
            })

    return final_messages


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
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---------- Non-streaming response ----------
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
