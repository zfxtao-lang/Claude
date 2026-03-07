FROM python:3.12-slim

WORKDIR /app

# Install SQLite with FTS5 support (already included in python:3.12-slim)
RUN apt-get update && apt-get install -y --no-install-recommends sqlite3 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-load jieba dictionary at build time (avoids slow first request)
RUN python -c "import jieba; jieba.initialize()"

EXPOSE 5000

# Volume for persistent data (SQLite DB, backups, .env, system_prompt)
VOLUME ["/app/data"]

ENV DB_PATH=/app/data/chats.db
ENV DB_BACKUP_DIR=/app/data/backups
ENV SYSTEM_PROMPT_FILE=/app/data/system_prompt.txt

CMD ["gunicorn", "-c", "gunicorn.conf.py", "gateway:app"]
