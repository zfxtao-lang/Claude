#!/usr/bin/env python3
"""
Debug tool: simulate the full memory pipeline and show exactly what the model sees.

Usage:
    python test_search.py '大伯家的狗叫什么'
    python test_search.py '你还记得那本书吗' --context '我之前看了一本很有名的书'
    python test_search.py '大伯家的狗叫什么' --limit 10 --exclude 20
"""
import argparse
import json
import logging
import sys
import textwrap

# Enable all [Memory] logs to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(message)s",
    stream=sys.stderr,
)

from database import init_db, search_history, jieba_tokenize

# Import gateway functions — this also initializes Flask app, but we won't run it
from gateway import extract_search_query, build_messages, vector_search_memories
from embedding import vector_store


BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def truncate(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ↵ ")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def main():
    parser = argparse.ArgumentParser(description="Debug memory search pipeline")
    parser.add_argument("question", help="The user question to test")
    parser.add_argument("--context", "-c", action="append", default=[],
                        help="Extra context messages (earlier in conversation). "
                             "Can be repeated: -c 'msg1' -c 'msg2'")
    parser.add_argument("--limit", "-l", type=int, default=5,
                        help="search_history result limit (default: 5)")
    parser.add_argument("--exclude", "-e", type=int, default=50,
                        help="Exclude latest N messages (default: 50)")
    parser.add_argument("--full", "-f", action="store_true",
                        help="Print full message content (no truncation)")
    parser.add_argument("--model", "-m", default="deepseek-chat",
                        help="Model name for build_messages (default: deepseek-chat)")
    args = parser.parse_args()

    init_db()

    # --- Step 1: Build simulated Kelivo messages ---
    incoming = []
    for ctx in args.context:
        incoming.append({"role": "user", "content": ctx})
        incoming.append({"role": "assistant", "content": "(simulated prior reply)"})
    incoming.append({"role": "user", "content": args.question})

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  Memory Pipeline Debug{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    # --- Step 2: Extract search query ---
    print(f"\n{YELLOW}▸ Step 1: extract_search_query{RESET}")
    print(f"  Input messages: {len(incoming)}")
    for i, m in enumerate(incoming):
        print(f"    [{i}] {m['role']}: {truncate(m['content'], 80)}")

    search_query = extract_search_query(incoming)
    print(f"  {BOLD}Search query:{RESET} '{GREEN}{search_query}{RESET}'")

    if not search_query:
        print(f"\n  {RED}✗ No search query extracted — memory retrieval will be skipped{RESET}")
        print(f"    This means the input was too short or contained no usable keywords.\n")
        sys.exit(0)

    # --- Step 3: Jieba tokenization detail ---
    print(f"\n{YELLOW}▸ Step 2: jieba tokenization{RESET}")
    tokens = jieba_tokenize(args.question)
    print(f"  Original: '{args.question}'")
    print(f"  Tokens:   [{', '.join(tokens.split())}]")

    # --- Step 4: Vector search ---
    print(f"\n{YELLOW}▸ Step 3a: vector search (semantic){RESET}")
    print(f"  Vector store size: {vector_store.size}")
    if vector_store.size > 0:
        vec_results = vector_search_memories(search_query, top_k=args.limit)
        print(f"  {BOLD}Results: {len(vec_results)}{RESET}")
        for i, vc in enumerate(vec_results):
            score = vc.get("score", 0)
            date = vc.get("created_at", "?")[:10]
            content = vc.get("content", "")
            color = GREEN if score >= 0.5 else YELLOW
            if args.full:
                print(f"\n  [{i}] {color}score={score:.3f}{RESET} | {date}")
                for line in content.split("\n"):
                    print(f"       {line}")
            else:
                print(f"  [{i}] {color}score={score:.3f}{RESET} | {date} | "
                      f"{truncate(content, 100)}")
    else:
        print(f"  {DIM}(no vectors yet — run POST /admin/vectors/rebuild first){RESET}")

    # --- Step 5: LIKE search (keyword fallback) ---
    print(f"\n{YELLOW}▸ Step 3b: LIKE search (keyword fallback){RESET}")
    print(f"  query='{search_query[:60]}', limit={args.limit}, exclude_recent={args.exclude}")
    results = search_history(search_query, limit=args.limit, exclude_recent=args.exclude)
    print(f"  {BOLD}Results: {len(results)}{RESET}")

    if results:
        for i, r in enumerate(results):
            role = r.get("role", "?")
            date = r.get("created_at", "?")[:16]
            content = r.get("content", "")
            conv = r.get("conversation_id", "?")[:8]
            color = CYAN if role == "user" else GREEN
            if args.full:
                print(f"\n  [{i}] {color}{role}{RESET} | {date} | conv={conv}")
                for line in content.split("\n"):
                    print(f"       {line}")
            else:
                print(f"  [{i}] {color}{role}{RESET} | {date} | conv={conv} | "
                      f"{truncate(content, 100)}")
    else:
        print(f"  {RED}✗ No results found!{RESET}")
        print(f"    Possible causes:")
        print(f"    - No matching content in database")
        print(f"    - All matches are within the latest {args.exclude} messages (excluded)")
        print(f"    - Database is empty or not at expected path")

    # --- Step 5: Build full messages ---
    print(f"\n{YELLOW}▸ Step 4: build_messages (what model actually sees){RESET}")
    final = build_messages(incoming, args.model)
    print(f"  {BOLD}Total messages: {len(final)}{RESET}")

    memory_idx = None
    kelivo_idx = None
    for i, m in enumerate(final):
        role = m["role"]
        content = m.get("content", "")

        # Detect memory and kelivo positions
        if "真实对话记忆" in str(content):
            memory_idx = i
        if kelivo_idx is None and role in ("user", "assistant") and "真实对话记忆" not in str(content):
            if not (role == "user" and content == "(simulated prior reply)"):
                kelivo_idx = i

        # Color by role
        if role == "system":
            color = BLUE
        elif role == "user":
            color = CYAN
        else:
            color = GREEN

        # Label
        label = ""
        if "真实对话记忆" in str(content):
            label = f" {BOLD}← MEMORY{RESET}"
        elif i == len(final) - 1 and role == "user":
            label = f" {BOLD}← LATEST USER MSG{RESET}"
        elif "Core Memory" in str(content):
            label = f" {DIM}← Notion{RESET}"

        if args.full:
            print(f"\n  [{i}] {color}{role}{RESET}{label}")
            for line in str(content).split("\n"):
                print(f"       {line}")
        else:
            print(f"  [{i}] {color}{role}{RESET}: {truncate(str(content), 90)}{label}")

    # --- Summary ---
    print(f"\n{BOLD}{'─' * 70}{RESET}")
    if memory_idx is not None:
        if kelivo_idx is not None and memory_idx < kelivo_idx:
            print(f"  {GREEN}✓ Memory at [{memory_idx}], Kelivo chat starts at [{kelivo_idx}] "
                  f"— memory is BEFORE conversation (correct){RESET}")
        elif kelivo_idx is not None:
            print(f"  {RED}✗ Memory at [{memory_idx}], Kelivo chat starts at [{kelivo_idx}] "
                  f"— memory is AFTER conversation (wrong!){RESET}")
        else:
            print(f"  {GREEN}✓ Memory injected at [{memory_idx}]{RESET}")
    else:
        print(f"  {RED}✗ No memory injected into final messages{RESET}")
    print()


if __name__ == "__main__":
    main()
