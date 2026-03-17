"""
Standalone worker for memory pipeline tasks.

This script is intentionally separated from the chat request path so the gateway
can keep using interactive providers while memory summarization uses the
dedicated DeepSeek worker credentials from config.py.
"""
import argparse
import json
import logging
from datetime import datetime

import config
from database import init_db, save_worker_run, update_worker_run
from memory_pipeline import process_memory_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _default_entry_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def main():
    parser = argparse.ArgumentParser(description="Run the memory worker pipeline.")
    parser.add_argument("--date", default=_default_entry_date(), help="Entry date in YYYY-MM-DD")
    parser.add_argument("--mode", default=config.MEMORY_WORKER_RUN_MODE, help="Run mode label")
    args = parser.parse_args()

    init_db()

    run_id = save_worker_run(
        worker_name="memory_worker",
        run_mode=args.mode,
        status="running",
        phase="boot",
        message=f"starting pipeline for {args.date}",
    )

    if not config.MEMORY_WORKER_ENABLED:
        update_worker_run(
            run_id,
            status="failed",
            phase="disabled",
            message="memory worker disabled by configuration",
            completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            error="MEMORY_WORKER_ENABLED is false",
        )
        raise RuntimeError("memory worker is disabled by configuration")

    usage_stats = {"token_input": 0, "token_output": 0, "token_total": 0}

    def progress_cb(phase, **extra):
        update_worker_run(
            run_id,
            phase=phase,
            message=json.dumps(extra, ensure_ascii=False) if extra else phase,
        )

    try:
        result = process_memory_pipeline(
            entry_date=args.date,
            progress_cb=progress_cb,
            worker_run_id=run_id,
            usage_stats=usage_stats,
        )
        usage = result.get("usage", {})
        update_worker_run(
            run_id,
            status="success",
            phase="done",
            message=f"pipeline finished for {args.date}",
            completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            token_input=int(usage.get("token_input", 0)),
            token_output=int(usage.get("token_output", 0)),
            token_total=int(usage.get("token_total", 0)),
            result_json=result,
        )
        logger.info("memory worker finished successfully")
    except Exception as exc:
        logger.exception("memory worker failed")
        update_worker_run(
            run_id,
            status="failed",
            phase="error",
            message=f"pipeline failed for {args.date}",
            completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            token_input=int(usage_stats.get("token_input", 0)),
            token_output=int(usage_stats.get("token_output", 0)),
            token_total=int(usage_stats.get("token_total", 0)),
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
