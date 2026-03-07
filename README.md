# AI Chat Gateway

Tencent Cloud lightweight server gateway for routing AI chat requests from Kelivo to multiple providers (OpenAI, Anthropic, DeepSeek).

## Architecture

```
Kelivo App
    |
  HTTPS (443)
    |
  Nginx (reverse proxy + SSL)
    |
  Gunicorn (127.0.0.1:5000)
    |
  gateway.py (Flask)
    |
    +-- config.py          # Provider routing, auth, settings
    +-- database.py        # SQLite WAL + FTS5 + jieba
    +-- notion_cache.py    # Notion content cache with TTL
```

## Key Features

- **Multi-provider routing**: OpenAI / Anthropic / DeepSeek, auto-routes by model name
- **Kelivo compatibility**: Filters duplicate history, hidden Memory Tool messages
- **System prompt**: Fixed persona loaded from `system_prompt.txt`
- **History search**: jieba Chinese tokenization + FTS5 + LIKE fallback
- **Notion knowledge base**: Cached with configurable TTL
- **Security**: Bearer token auth, rate limiting (RPM/RPD), CORS control
- **Provider adapter**: Anthropic responses converted to OpenAI format
- **Health check**: `GET /health` for monitoring
- **Auto-backup**: Daily cron job for SQLite

## Deployment

```bash
# 1. Setup
cd /opt/gateway
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
vim .env  # fill in API keys and tokens
vim system_prompt.txt  # write your persona

# 3. Run
gunicorn -c gunicorn.conf.py gateway:app

# 4. Systemd (process guard)
sudo cp gateway.service /etc/systemd/system/
sudo systemctl enable --now gateway

# 5. Backup cron
chmod +x backup_cron.sh
crontab -e
# Add: 0 3 * * * /opt/gateway/backup_cron.sh >> /var/log/gateway-backup.log 2>&1
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat (OpenAI-compatible) |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |
| GET | `/admin/stats` | Usage statistics |
| POST | `/admin/backup` | Trigger database backup |
| POST | `/admin/notion/refresh` | Force refresh Notion cache |

## Issues Fixed

1. Kelivo sends 64 messages history + gateway appended 10 more -> now gateway passes through Kelivo's history as-is
2. Database dedup: same role+content within 5 seconds = skip
3. Kelivo Memory Tool system messages filtered out
4. Raw API response logged for debugging empty replies
5. Non-streaming mode supported alongside streaming
6. `memory_context` injected as user message (not system) before last user message
7. FTS5 with jieba tokenization + LIKE fallback for Chinese search
8. Provider adapter: Anthropic -> OpenAI format conversion (streaming + non-streaming)
9. Notion content cached with TTL (default 1 hour)
10. SQLite WAL mode enabled by default
