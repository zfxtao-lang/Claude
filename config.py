"""
Gateway configuration - multi-provider routing, timeout, rate limiting

Provider routing is config-driven via providers.json.
Adding a new provider = adding a JSON block, zero code changes.
"""
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()  # Load .env file

logger = logging.getLogger(__name__)

# ---------- API Providers (loaded from providers.json) ----------
PROVIDERS_FILE = os.getenv("PROVIDERS_FILE", "providers.json")


def _load_providers() -> dict:
    """
    Load providers from JSON config file.
    Each entry's api_key supports ${ENV_VAR} syntax for env var substitution.
    Falls back to empty dict if file not found.
    """
    try:
        with open(PROVIDERS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        logger.warning(f"providers.json not found at {PROVIDERS_FILE}, no providers loaded")
        return {}

    providers = {}
    for name, cfg in raw.items():
        # Resolve ${ENV_VAR} or $ENV_VAR in string values
        resolved = {}
        for key, val in cfg.items():
            if isinstance(val, str) and val.startswith("$"):
                env_name = val.lstrip("$").strip("{}")
                resolved[key] = os.getenv(env_name, "")
            else:
                resolved[key] = val
        # Ensure required fields have defaults
        resolved.setdefault("prefixes", [])
        resolved.setdefault("timeout", 120)
        resolved.setdefault("base_url", "")
        resolved.setdefault("api_key", "")
        providers[name] = resolved

    configured = [n for n, c in providers.items() if c.get("api_key")]
    logger.info(f"Loaded {len(providers)} providers, {len(configured)} with API keys: "
                f"{configured}")
    return providers


PROVIDERS = _load_providers()

# ---------- Gateway Auth ----------
GATEWAY_AUTH_TOKEN = os.getenv("GATEWAY_AUTH_TOKEN", "")

# ---------- System Prompt ----------
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")

# ---------- Database ----------
DB_PATH = os.getenv("DB_PATH", "chats.db")
DB_BACKUP_DIR = os.getenv("DB_BACKUP_DIR", "backups")
DB_BACKUP_KEEP_DAYS = int(os.getenv("DB_BACKUP_KEEP_DAYS", "30"))

# ---------- Notion Cache ----------
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_PAGE_IDS = os.getenv("NOTION_PAGE_IDS", "").split(",")  # comma-separated
NOTION_CACHE_TTL = int(os.getenv("NOTION_CACHE_TTL", "3600"))  # seconds

# ---------- History Retrieval ----------
HISTORY_SEARCH_LIMIT = int(os.getenv("HISTORY_SEARCH_LIMIT", "3"))
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "2000"))

# ---------- Rate Limiting ----------
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "30"))       # requests per minute
RATE_LIMIT_RPD = int(os.getenv("RATE_LIMIT_RPD", "500"))      # requests per day

# ---------- CORS ----------
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")

# ---------- Token Budget ----------
MAX_NOTION_CHARS = int(os.getenv("MAX_NOTION_CHARS", "6000"))  # truncate Notion content

# ---------- Retry ----------
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
API_RETRY_BACKOFF = float(os.getenv("API_RETRY_BACKOFF", "1.5"))  # seconds

# ---------- Memory Worker ----------
MEMORY_WORKER_ENABLED = os.getenv("MEMORY_WORKER_ENABLED", "1").lower() in ("1", "true", "yes")
MEMORY_WORKER_PROVIDER = os.getenv("MEMORY_WORKER_PROVIDER", "deepseek")
MEMORY_WORKER_MODEL = os.getenv("MEMORY_WORKER_MODEL", "deepseek-chat")
MEMORY_WORKER_API_KEY = os.getenv("MEMORY_WORKER_API_KEY", "")
MEMORY_WORKER_BASE_URL = os.getenv("MEMORY_WORKER_BASE_URL", "https://api.deepseek.com/v1")
MEMORY_WORKER_RUN_MODE = os.getenv("MEMORY_WORKER_RUN_MODE", "manual")
# Periodic in-process pipeline (gateway daemon thread); does not replace manual/cron CLI.
MEMORY_WORKER_AUTO_ENABLED = os.getenv("MEMORY_WORKER_AUTO_ENABLED", "1").lower() in ("1", "true", "yes")
MEMORY_WORKER_AUTO_INTERVAL_SEC = int(os.getenv("MEMORY_WORKER_AUTO_INTERVAL_SEC", "900"))
MEMORY_DIARY_ENABLED = os.getenv("MEMORY_DIARY_ENABLED", "1").lower() in ("1", "true", "yes")
MEMORY_PERSONA_ENABLED = os.getenv("MEMORY_PERSONA_ENABLED", "1").lower() in ("1", "true", "yes")
MEMORY_SLICE_SIZE = int(os.getenv("MEMORY_SLICE_SIZE", "20"))
MEMORY_SLICE_COMPACT_COUNT = int(os.getenv("MEMORY_SLICE_COMPACT_COUNT", "4"))
MEMORY_CONTEXT_SLICE_LIMIT = int(os.getenv("MEMORY_CONTEXT_SLICE_LIMIT", "4"))
MEMORY_CONTEXT_RAW_LIMIT = int(os.getenv("MEMORY_CONTEXT_RAW_LIMIT", "20"))
LONG_TERM_MEMORY_TOP_K = int(os.getenv("LONG_TERM_MEMORY_TOP_K", "6"))


def get_provider_for_model(model: str) -> dict | None:
    """
    Find which provider handles a given model name (prefix matching).
    Longer prefix matches first (e.g. "deepseek-chat" beats "deep").
    """
    best_match = None
    best_prefix_len = 0
    for name, cfg in PROVIDERS.items():
        if not cfg.get("api_key"):
            continue  # skip unconfigured providers
        for prefix in cfg.get("prefixes", []):
            if model.startswith(prefix) and len(prefix) > best_prefix_len:
                best_match = {"provider": name, **cfg}
                best_prefix_len = len(prefix)
    return best_match


def reload_providers():
    """Hot-reload providers from config file without restarting."""
    global PROVIDERS
    PROVIDERS = _load_providers()
    return PROVIDERS


def load_system_prompt() -> str:
    """Load system prompt from file, return empty string if missing."""
    try:
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
