#!/bin/bash
# Daily database backup - add to crontab:
#   0 3 * * * /path/to/backup_cron.sh >> /var/log/gateway-backup.log 2>&1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BACKUP_DIR="${SCRIPT_DIR}/backups"
DB_PATH="${SCRIPT_DIR}/chats.db"
KEEP_DAYS=30

mkdir -p "$BACKUP_DIR"

# SQLite safe backup using .backup command
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEST="${BACKUP_DIR}/chats_${TIMESTAMP}.db"

sqlite3 "$DB_PATH" ".backup '${DEST}'"
echo "[$(date)] Backup created: ${DEST}"

# Clean old backups
find "$BACKUP_DIR" -name "chats_*.db" -mtime +${KEEP_DAYS} -delete
echo "[$(date)] Cleaned backups older than ${KEEP_DAYS} days"
