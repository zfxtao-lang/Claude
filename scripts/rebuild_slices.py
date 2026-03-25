"""
Rebuild memory_slices using the current SLICE_PROMPT.

This script is designed for a "history data reset & backfill" workflow:
1) (optional) clear `memory_slices` so `get_unsliced_messages()` cursor resets
2) repeatedly call `process_pending_slices()` until there is no full slice left

Usage examples:
  python3 scripts/rebuild_slices.py --reset
  python3 scripts/rebuild_slices.py --max-iterations 500
"""

from __future__ import annotations

import argparse
import logging
import time

from config import MEMORY_SLICE_SIZE
from database import get_db, init_db, get_unsliced_messages
from memory_pipeline import process_pending_slices


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def reset_memory_slices() -> int:
    conn = get_db()
    try:
        conn.execute("DELETE FROM memory_slices")
        conn.commit()
        remaining = conn.execute("SELECT COUNT(*) FROM memory_slices").fetchone()[0]
        return int(remaining)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all rows from memory_slices before rebuilding.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Sleep between iterations (helps reduce API bursts).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10_000,
        help="Hard stop to avoid infinite loops if the LLM keeps failing.",
    )
    parser.add_argument(
        "--fail-threshold",
        type=int,
        default=20,
        help="If process_pending_slices returns no generated slices this many times in a row, exit.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    log = logging.getLogger("rebuild_slices")

    init_db()  # Ensure schema exists / migrations are applied.

    if args.reset:
        log.warning("Reset requested: clearing memory_slices ...")
        remaining = reset_memory_slices()
        log.info("Reset done: memory_slices remaining=%s", remaining)

    pending_failures = 0
    total_generated = 0

    # Cursor is driven by memory_slices' MAX(msg_id_end), so looping is enough after reset.
    for i in range(args.max_iterations):
        pending = get_unsliced_messages(limit=MEMORY_SLICE_SIZE * 20)
        if len(pending) < MEMORY_SLICE_SIZE:
            log.info(
                "Done: pending_messages=%s < MEMORY_SLICE_SIZE=%s (iterations=%s, total_generated=%s).",
                len(pending),
                MEMORY_SLICE_SIZE,
                i,
                total_generated,
            )
            break

        generated = process_pending_slices(progress_cb=None, usage_stats=None)
        if not generated:
            pending_failures += 1
            log.warning(
                "No slices generated in iteration=%s (streak=%s/%s). pending_messages=%s",
                i,
                pending_failures,
                args.fail_threshold,
                len(pending),
            )
            if pending_failures >= args.fail_threshold:
                raise RuntimeError("Too many consecutive empty generations; aborting.")
        else:
            pending_failures = 0
            total_generated += len(generated)
            last_id = generated[-1].get("id")
            log.info(
                "Iteration=%s generated=%s total_generated=%s last_slice_id=%s pending_messages_left≈? (MEMORY_SLICE_SIZE=%s)",
                i,
                len(generated),
                total_generated,
                last_id,
                MEMORY_SLICE_SIZE,
            )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    else:
        raise RuntimeError(f"Reached max-iterations={args.max_iterations} without completing.")


if __name__ == "__main__":
    main()

