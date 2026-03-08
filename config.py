"""
Gateway configuration - multi-provider routing, timeout, rate limiting
"""
import os

from dotenv import load_dotenv

load_dotenv()  # Load .env file

# ---------- API Providers ----------
# Each provider: name -> {base_url, api_key, prefixes[], timeout}
# Model routing uses prefix matching: model "anthropic/claude-3" matches prefix "anthropic/"
PROVIDERS = {
    "openrouter": {
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "prefixes": ["anthropic/", "openai/", "google/"],
        "timeout": int(os.getenv("OPENROUTER_TIMEOUT", "120")),
    },
    "deepseek": {
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "prefixes": ["deepseek-"],
        "timeout": int(os.getenv("DEEPSEEK_TIMEOUT", "120")),
    },
    "zhipu": {
        "base_url": os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        "api_key": os.getenv("ZHIPU_API_KEY", ""),
        "prefixes": ["glm-"],
        "timeout": int(os.getenv("ZHIPU_TIMEOUT", "120")),
    },
    "alibaba": {
        "base_url": os.getenv("ALIBABA_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "api_key": os.getenv("ALIBABA_API_KEY", ""),
        "prefixes": ["qwen-"],
        "timeout": int(os.getenv("ALIBABA_TIMEOUT", "120")),
    },
}

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
HISTORY_SEARCH_LIMIT = int(os.getenv("HISTORY_SEARCH_LIMIT", "5"))
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "4000"))

# ---------- Rate Limiting ----------
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "30"))       # requests per minute
RATE_LIMIT_RPD = int(os.getenv("RATE_LIMIT_RPD", "500"))      # requests per day

# ---------- CORS ----------
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")

# ---------- Token Budget ----------
MAX_NOTION_CHARS = int(os.getenv("MAX_NOTION_CHARS", "6000"))  # truncate Notion content

# ---------- Retry ----------
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "2"))
API_RETRY_BACKOFF = float(os.getenv("API_RETRY_BACKOFF", "1.0"))  # seconds


def get_provider_for_model(model: str) -> dict | None:
    """Find which provider handles a given model name (prefix matching)."""
    for name, cfg in PROVIDERS.items():
        for prefix in cfg["prefixes"]:
            if model.startswith(prefix):
                return {"provider": name, **cfg}
    return None


def load_system_prompt() -> str:
    """Load system prompt from file, return empty string if missing."""
    try:
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
