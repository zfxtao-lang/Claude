"""
Memory Cards - AI-generated daily summaries for dual-layer memory.

Layer 1: Memory cards (short, high-density, embedding-searchable)
Layer 2: Raw conversations (linked via msg_id range, for detail lookup)

Flow:
  1. Group messages by date
  2. Call DeepSeek/GLM to generate structured summary
  3. Store as memory_card in SQLite
  4. Embed card text → store in card_vector_store
"""
import json
import logging
import os
import time

import requests

from config import PROVIDERS
from database import (
    get_cards_by_date,
    get_dates_without_cards,
    get_messages_by_date,
    get_unembedded_cards,
    mark_cards_embedded,
    save_memory_card,
)
from embedding import card_vector_store, get_embedding, get_embeddings_batch

logger = logging.getLogger(__name__)

# Which model to use for summarization (cheap + good at Chinese)
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "deepseek-chat")
SUMMARY_PROVIDER = os.getenv("SUMMARY_PROVIDER", "deepseek")

# Prompt for generating memory cards
CARD_PROMPT = """你是一个记忆整理助手。请将以下对话整理成记忆卡片。

要求：
1. 用简洁生动的语言总结当天发生的事件、梗、约定、情感时刻
2. 每个独立事件/话题用一行概括，保留关键细节和有趣的梗
3. 总结长度控制在200-500字
4. 最后一行输出标签，格式：tags: 标签1,标签2,标签3（用逗号分隔，不加#）

示例输出：
淘淘让煲汤，小克把"给你煲汤"说成"把老公煲成汤"，笑了很久。
淘淘认真分析了锅的尺寸问题，发了小红书帖子，84浏览2评论1收藏。
晚上聊了关于搬家的计划，淘淘倾向于离公司近的地方。
tags: 煲汤口误,小红书,搬家计划

以下是{date}的对话内容：
---
{conversations}
---

请输出记忆卡片："""


def _call_summary_api(prompt: str) -> str | None:
    """Call the summarization LLM (DeepSeek by default)."""
    provider_cfg = PROVIDERS.get(SUMMARY_PROVIDER)
    if not provider_cfg or not provider_cfg.get("api_key"):
        logger.error(f"[MemoryCard] provider '{SUMMARY_PROVIDER}' not configured")
        return None

    try:
        resp = requests.post(
            f"{provider_cfg['base_url']}/chat/completions",
            headers={
                "Authorization": f"Bearer {provider_cfg['api_key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": SUMMARY_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个精准的记忆整理助手，擅长提取对话中的关键事件和有趣细节。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1000,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            logger.error(f"[MemoryCard] API error {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception:
        logger.error("[MemoryCard] API call failed", exc_info=True)
        return None


def _parse_tags(text: str) -> tuple[str, str]:
    """
    Extract tags line from summary text.
    Returns (summary_without_tags, tags_string).
    """
    lines = text.strip().split("\n")
    tags = ""
    summary_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("tags:") or stripped.startswith("标签:"):
            # Extract tags
            tags_part = stripped.split(":", 1)[1].strip()
            # Clean up: remove #, extra spaces
            tags = ",".join(
                t.strip().lstrip("#") for t in tags_part.replace("，", ",").split(",") if t.strip()
            )
        else:
            summary_lines.append(line)

    summary = "\n".join(summary_lines).strip()
    return summary, tags


def generate_card_for_date(date: str, force: bool = False) -> dict | None:
    """
    Generate a memory card for a specific date.

    Args:
        date: YYYY-MM-DD format
        force: If True, regenerate even if card already exists

    Returns:
        Card dict or None if failed/no messages.
    """
    # Check if card already exists
    if not force:
        existing = get_cards_by_date(date)
        if existing:
            logger.info(f"[MemoryCard] card already exists for {date}, skipping (use force=True to regenerate)")
            return existing[0]

    # Get messages for the date
    messages = get_messages_by_date(date)
    if not messages:
        logger.info(f"[MemoryCard] no messages for {date}")
        return None

    # Filter out very short messages and build conversation text
    conv_lines = []
    conv_ids = set()
    msg_id_start = messages[0]["id"]
    msg_id_end = messages[-1]["id"]

    for msg in messages:
        role = msg["role"]
        content = (msg.get("content") or "").strip()
        if not content or len(content) < 2:
            continue
        # Truncate very long messages
        if len(content) > 500:
            content = content[:500] + "..."
        label = "淘淘" if role == "user" else "小克"
        conv_lines.append(f"{label}: {content}")
        conv_ids.add(msg["conversation_id"])

    if len(conv_lines) < 2:
        logger.info(f"[MemoryCard] too few messages for {date} ({len(conv_lines)} lines)")
        return None

    # Truncate total conversation to ~6000 chars for the summary API
    conversation_text = "\n".join(conv_lines)
    if len(conversation_text) > 6000:
        conversation_text = conversation_text[:6000] + "\n...(对话过长，已截断)"

    # Build prompt and call API
    prompt = CARD_PROMPT.format(date=date, conversations=conversation_text)
    logger.info(f"[MemoryCard] generating card for {date} "
                f"({len(messages)} messages, {len(conversation_text)} chars)")

    raw_summary = _call_summary_api(prompt)
    if not raw_summary:
        return None

    # Parse tags from response
    summary, tags = _parse_tags(raw_summary)
    if not summary:
        logger.warning(f"[MemoryCard] empty summary for {date}")
        return None

    # Save to database
    card_id = save_memory_card(
        date=date,
        summary=summary,
        tags=tags,
        conversation_ids=",".join(sorted(conv_ids)),
        msg_id_start=msg_id_start,
        msg_id_end=msg_id_end,
    )

    logger.info(f"[MemoryCard] saved card #{card_id} for {date}: "
                f"{len(summary)} chars, tags={tags}")

    return {
        "id": card_id,
        "date": date,
        "summary": summary,
        "tags": tags,
        "message_count": len(messages),
    }


def generate_cards_batch(start_date: str = None, end_date: str = None,
                         force: bool = False) -> list[dict]:
    """
    Generate memory cards for all dates that don't have cards yet.
    Returns list of generated cards.
    """
    if force:
        # When force=True, we need to get all dates with messages in range
        from database import get_db
        conn = get_db()
        try:
            sql = "SELECT DISTINCT date(created_at) as d FROM messages WHERE 1=1"
            params = []
            if start_date:
                sql += " AND date(created_at) >= ?"
                params.append(start_date)
            if end_date:
                sql += " AND date(created_at) <= ?"
                params.append(end_date)
            sql += " ORDER BY d"
            rows = conn.execute(sql, params).fetchall()
            dates = [r[0] for r in rows if r[0]]
        finally:
            conn.close()
    else:
        dates = get_dates_without_cards(start_date, end_date)

    if not dates:
        logger.info("[MemoryCard] no dates need cards")
        return []

    logger.info(f"[MemoryCard] generating cards for {len(dates)} dates: "
                f"{dates[0]} to {dates[-1]}")

    results = []
    for date in dates:
        card = generate_card_for_date(date, force=force)
        if card:
            results.append(card)
        # Rate limit: don't hammer the API
        time.sleep(1.0)

    logger.info(f"[MemoryCard] batch complete: {len(results)}/{len(dates)} cards generated")
    return results


def embed_pending_cards() -> int:
    """Embed all memory cards that don't have embeddings yet."""
    pending = get_unembedded_cards(limit=100)
    if not pending:
        return 0

    embedded_ids = []
    texts = [c["summary"] for c in pending]

    # Batch embed (6 per API call)
    for i in range(0, len(texts), 6):
        batch = texts[i:i + 6]
        vectors = get_embeddings_batch(batch)
        for card, vec in zip(pending[i:i + 6], vectors):
            if vec is not None:
                card_vector_store.add(card["id"], vec)
                embedded_ids.append(card["id"])
        time.sleep(0.2)

    if embedded_ids:
        mark_cards_embedded(embedded_ids)
        card_vector_store.save()
        logger.info(f"[MemoryCard] embedded {len(embedded_ids)} cards, "
                    f"store size: {card_vector_store.size}")

    return len(embedded_ids)


def search_memory_cards(query: str, top_k: int = 5,
                        min_score: float = 0.25) -> list[dict]:
    """
    Search memory cards by vector similarity.
    Returns cards with scores, sorted by relevance.
    """
    if card_vector_store.size == 0:
        return []

    query_vec = get_embedding(query)
    if query_vec is None:
        return []

    results = card_vector_store.search(query_vec, top_k=top_k)
    good = [(cid, score) for cid, score in results if score >= min_score]

    if not good:
        return []

    card_ids = [cid for cid, _ in good]
    scores = {cid: score for cid, score in good}

    from database import get_cards_by_ids
    cards = get_cards_by_ids(card_ids)
    for card in cards:
        card["score"] = scores.get(card["id"], 0)

    return cards
