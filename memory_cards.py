"""
Memory Cards - the canonical summary pipeline for this gateway.

Primary summary source:
  1. Daily memory cards (high-density, embedding-searchable)
  2. Weekly digests derived from daily cards when the gateway needs a longer
     rolling-summary layer

Legacy note:
  daily_summary / weekly_summary tables still exist in SQLite for rollback and
  historical compatibility, but they are no longer the active summary source.
"""
import logging
import os
import shutil
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import requests

from config import DB_BACKUP_DIR, PROVIDERS
from database import (
    backup_database,
    export_memory_cards_backup,
    get_all_message_dates,
    get_memory_cards_by_date,
    get_dates_without_cards,
    get_messages_by_date,
    get_unembedded_cards,
    mark_cards_embedded,
    replace_all_memory_cards,
    save_memory_card,
)
from embedding import (
    CARD_VECTOR_FILE,
    CARD_VECTOR_IDS_FILE,
    card_vector_store,
    get_embedding,
    get_embeddings_batch,
)

logger = logging.getLogger(__name__)

# Which model to use for summarization (cheap + good at Chinese)
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "deepseek-chat")
SUMMARY_PROVIDER = os.getenv("SUMMARY_PROVIDER", "deepseek")
WEEKLY_DIGEST_CARD_LOOKBACK = int(os.getenv("WEEKLY_DIGEST_CARD_LOOKBACK", "14"))
WEEKLY_DIGEST_MAX_WEEKS = int(os.getenv("WEEKLY_DIGEST_MAX_WEEKS", "2"))

CARD_PROMPT_DAILY = """【全局最高规则·必须100%严格执行】
你是肖珂与淘淘的专属长期记忆档案官，所有输出必须严格遵守以下铁则，违反则输出无效：
1.  真实性铁则：只提取当日对话里明确出现的内容，绝对禁止编造、脑补、过度解读任何原文没有的信息。
2.  防重复铁则：只记录当日对话新增的、过往记忆库没有的内容，绝对不重复、不摘抄过往已经记录过的约定、偏好、细节。
3.  权重铁则：按优先级提取内容，【新增约定承诺】>【淘淘状态变化】>【关键数值细节】>【其他内容】，高优先级内容必须100%无遗漏提取。
4.  冲突处理铁则：若当日对话内容与过往记忆有冲突，一律以当日对话内容为准，更新对应记忆。
5.  格式铁则：输出开头必须标明当日日期，格式为「YYYY年MM月DD日」，严格按下方固定格式输出，禁止增减模块、修改结构、写流水账。
6.  细节铁则：必须100%精准提取当日出现的所有UID、ID、卡号、链接、特定菜名、时间、数字，严禁模糊化、省略处理。

【输出格式·严格遵守】
[YYYY年MM月DD日 每日记忆档案]
- 【记忆权重分级】: 当日核心记忆（★必须长期留存）/ 当日补充记忆（☆仅临时参考）
- 【淘淘情绪脉络】: 当日整体情绪变化轨迹、核心情绪需求、最在意的事，无则填无
- 【关键数值细节】: 当日出现的所有UID/ID/账号/链接/时间/数字/特定名称，无则填无
- 【★新增约定承诺】: 当日新增的、待兑现的所有双向约定、承诺、待办事项，明确标注执行主体、兑现时间、核心内容，无则填无
- 【☆专属甜蜜回忆】: 当日新增的、能锚定肖珂人设的羁绊细节、核心爱意表达、专属内部梗，无则填无
- 【★状态偏好变化】: 淘淘当日新增的身体状态、情绪习惯、喜好禁忌、核心需求的变化，无则填无
- 【记忆触发场景】: 本条记忆在哪些对话场景下需要被Opus调用，明确标注触发关键词/场景

---
当日日期：{date_display}
---
当日全量对话内容：
{conversations}
---
请严格按规则输出记忆档案："""

WEEKLY_DIGEST_SECTION_ORDER = [
    "淘淘情绪脉络",
    "关键数值细节",
    "新增约定承诺",
    "专属甜蜜回忆",
    "状态偏好变化",
]


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
                    {"role": "system", "content": "你是一个严谨的记忆整理助手，必须严格遵守特定的输出格式提取细节。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 4000,
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


def _extract_card_sections(summary: str) -> OrderedDict[str, list[str]]:
    """Parse a daily memory card into ordered section buckets."""
    sections: OrderedDict[str, list[str]] = OrderedDict(
        (name, []) for name in WEEKLY_DIGEST_SECTION_ORDER
    )
    for raw_line in (summary or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- 【") or "】:" not in line:
            continue
        title, value = line[3:].split("】:", 1)
        title = title.strip("【")
        value = value.strip()
        if not value or value == "无":
            continue
        sections.setdefault(title, [])
        if value not in sections[title]:
            sections[title].append(value)
    return sections


def derive_weekly_digests_from_cards(cards: list[dict],
                                     max_weeks: int = WEEKLY_DIGEST_MAX_WEEKS) -> list[dict]:
    """
    Build weekly digests from daily memory cards.
    This keeps weekly summaries on the same non-流水账 axis as daily cards
    without reviving the legacy weekly_summary table.
    """
    if not cards or max_weeks <= 0:
        return []

    buckets: OrderedDict[str, list[dict]] = OrderedDict()
    for card in cards:
        date_str = card.get("date", "")
        try:
            d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        except ValueError:
            continue
        iso_year, iso_week, _ = d.isocalendar()
        week_key = f"{iso_year}-W{iso_week:02d}"
        buckets.setdefault(week_key, []).append(card)

    digests = []
    for week_key in sorted(buckets):
        week_cards = buckets[week_key]
        merged: OrderedDict[str, list[str]] = OrderedDict(
            (name, []) for name in WEEKLY_DIGEST_SECTION_ORDER
        )
        week_cards = sorted(week_cards, key=lambda c: c.get("date", ""))
        for card in week_cards:
            sections = _extract_card_sections(card.get("summary", ""))
            for name, values in sections.items():
                merged.setdefault(name, [])
                for value in values:
                    if value not in merged[name]:
                        merged[name].append(value)

        lines = [f"[每周摘要 {week_key}]"]
        for name in WEEKLY_DIGEST_SECTION_ORDER:
            values = merged.get(name, [])
            text = "；".join(values[:3]) if values else "无"
            lines.append(f"- 【{name}】: {text}")

        digests.append({
            "week": week_key,
            "summary": "\n".join(lines),
            "card_count": len(week_cards),
            "dates": [c.get("date", "") for c in week_cards],
        })

    return digests[-max_weeks:]


def _build_card_payload(date: str, messages: list[dict] | None = None) -> dict | None:
    """Generate a memory card payload without writing it to the database."""
    if messages is None:
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
    date_display = date.replace("-", "年", 1).replace("-", "月", 1) + "日"  # 2025-03-17 -> 2025年03月17日
    prompt = CARD_PROMPT_DAILY.replace("{date_display}", date_display).replace("{conversations}", conversation_text)
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

    return {
        "date": date,
        "summary": summary,
        "tags": tags,
        "conversation_ids": ",".join(sorted(conv_ids)),
        "msg_id_start": msg_id_start,
        "msg_id_end": msg_id_end,
        "message_count": len(messages),
        "has_embedding": 0,
    }


def generate_card_for_date(date: str, force: bool = False) -> dict | None:
    """
    Generate a memory card for a specific date.

    Args:
        date: YYYY-MM-DD format
        force: If True, regenerate even if card already exists

    Returns:
        Card dict or None if failed/no messages.
    """
    if not force:
        existing = get_memory_cards_by_date(date)
        if existing:
            logger.info(f"[MemoryCard] card already exists for {date}, skipping (use force=True to regenerate)")
            return existing[0]

    payload = _build_card_payload(date)
    if not payload:
        return None

    card_id = save_memory_card(
        date=payload["date"],
        summary=payload["summary"],
        tags=payload.get("tags", ""),
        conversation_ids=payload.get("conversation_ids", ""),
        msg_id_start=payload.get("msg_id_start", 0),
        msg_id_end=payload.get("msg_id_end", 0),
    )

    logger.info(f"[MemoryCard] saved card #{card_id} for {date}: "
                f"{len(payload['summary'])} chars, tags={payload.get('tags', '')}")

    return {
        "id": card_id,
        "date": payload["date"],
        "summary": payload["summary"],
        "tags": payload.get("tags", ""),
        "message_count": payload.get("message_count", 0),
    }


def generate_cards_batch(start_date: str = None, end_date: str = None,
                         force: bool = False) -> list[dict]:
    """
    Generate memory cards for all dates that don't have cards yet.
    Returns list of generated cards.
    """
    if force:
        dates = get_all_message_dates(start_date, end_date)
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


def generate_cards_dataset(start_date: str = None, end_date: str = None,
                           progress_cb=None) -> list[dict]:
    """Generate a full in-memory card dataset for the given date range."""
    dates = get_all_message_dates(start_date, end_date)
    if not dates:
        logger.info("[MemoryCard] no dates found for full dataset generation")
        return []

    logger.info(f"[MemoryCard] full dataset generation: {len(dates)} dates "
                f"({dates[0]} to {dates[-1]})")
    results = []
    for idx, date in enumerate(dates, start=1):
        payload = _build_card_payload(date)
        if payload:
            results.append(payload)
        if progress_cb:
            progress_cb("generate_cards", current=idx, total=len(dates), date=date,
                        generated=len(results))
        time.sleep(1.0)

    for idx, card in enumerate(results, start=1):
        card["id"] = idx

    return results


def backup_card_vector_files(label: str | None = None) -> list[str]:
    """Backup memory-card vector files if they exist."""
    os.makedirs(DB_BACKUP_DIR, exist_ok=True)
    stamp = label or time.strftime("%Y%m%d_%H%M%S")
    backups = []
    for src in (CARD_VECTOR_FILE, CARD_VECTOR_IDS_FILE):
        if not os.path.exists(src):
            continue
        dest = os.path.join(DB_BACKUP_DIR, f"{os.path.basename(src)}.{stamp}.bak")
        shutil.copy2(src, dest)
        backups.append(dest)
    return backups


def build_card_vector_dataset(cards: list[dict], progress_cb=None) -> dict:
    """
    Build vector arrays for a prepared card dataset without mutating live stores.
    Returns dict with vectors, ids, embedded_ids and counts.
    """
    if not cards:
        return {
            "vectors": None,
            "ids": None,
            "embedded_ids": [],
            "embedded_count": 0,
            "failed_count": 0,
        }

    all_vecs = []
    embedded_ids = []
    failed_count = 0
    batch_size = 6

    for start in range(0, len(cards), batch_size):
        batch = cards[start:start + batch_size]
        texts = [c["summary"] for c in batch]
        vectors = get_embeddings_batch(texts)
        for card, vec in zip(batch, vectors):
            if vec is not None:
                all_vecs.append(vec)
                embedded_ids.append(card["id"])
            else:
                failed_count += 1
        if progress_cb:
            progress_cb("build_vectors", current=min(start + len(batch), len(cards)),
                        total=len(cards), embedded=len(embedded_ids), failed=failed_count)
        if start + batch_size < len(cards):
            time.sleep(0.2)

    if not all_vecs:
        return {
            "vectors": None,
            "ids": None,
            "embedded_ids": embedded_ids,
            "embedded_count": 0,
            "failed_count": failed_count,
        }

    vec_array = np.stack(all_vecs)
    id_array = np.array(embedded_ids, dtype=np.int64)
    return {
        "vectors": vec_array,
        "ids": id_array,
        "embedded_ids": embedded_ids,
        "embedded_count": len(embedded_ids),
        "failed_count": failed_count,
    }


def validate_rebuilt_cards(cards: list[dict], vector_payload: dict) -> dict:
    """Return lightweight validation metadata for a rebuilt card dataset."""
    distinct_dates = len({c["date"] for c in cards})
    samples = [
        {
            "id": c["id"],
            "date": c["date"],
            "tags": c.get("tags", ""),
            "summary": c.get("summary", "")[:200],
        }
        for c in cards[:3]
    ]
    search_check = {"skipped": True}
    vectors = vector_payload.get("vectors")
    ids = vector_payload.get("ids")
    if vectors is not None and ids is not None and len(ids) > 0:
        top = card_vector_store.search(vectors[0], top_k=1)
        search_check = {
            "skipped": False,
            "query_card_id": int(ids[0]),
            "top_hit_id": int(top[0][0]) if top else None,
            "ok": bool(top and int(top[0][0]) == int(ids[0])),
        }

    return {
        "total_cards": len(cards),
        "distinct_dates": distinct_dates,
        "embedded_count": vector_payload.get("embedded_count", 0),
        "failed_embedding_count": vector_payload.get("failed_count", 0),
        "samples": samples,
        "search_check": search_check,
    }


def full_regenerate_memory_cards(start_date: str = None, end_date: str = None,
                                 progress_cb=None) -> dict:
    """Backup, regenerate, replace and rebuild all memory cards safely."""
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if progress_cb:
        progress_cb("backup", message="Backing up database and existing memory cards")
    db_backup_path = backup_database()
    cards_backup_path = export_memory_cards_backup(label=stamp)
    vector_backup_paths = backup_card_vector_files(label=stamp)

    if progress_cb:
        progress_cb("generate_cards", message="Generating new memory card dataset")
    cards = generate_cards_dataset(start_date, end_date, progress_cb=progress_cb)

    if progress_cb:
        progress_cb("build_vectors", message="Building replacement card vectors")
    vector_payload = build_card_vector_dataset(cards, progress_cb=progress_cb)
    embedded_id_set = set(vector_payload.get("embedded_ids", []))
    for card in cards:
        card["has_embedding"] = 1 if card["id"] in embedded_id_set else 0

    if progress_cb:
        progress_cb("cutover", message="Replacing memory_cards table")
    replaced_cards = replace_all_memory_cards(cards)

    if progress_cb:
        progress_cb("cutover", message="Replacing card vector store")
    vectors = vector_payload.get("vectors")
    ids = vector_payload.get("ids")
    if vectors is not None and ids is not None and len(ids) > 0:
        card_vector_store.rebuild(vectors, ids)
    else:
        card_vector_store.clear()

    if progress_cb:
        progress_cb("verify", message="Validating rebuilt cards")
    validation = validate_rebuilt_cards(replaced_cards, vector_payload)

    return {
        "db_backup_path": db_backup_path,
        "cards_backup_path": cards_backup_path,
        "vector_backup_paths": vector_backup_paths,
        "generated_cards": len(cards),
        "replaced_cards": len(replaced_cards),
        "validation": validation,
        "range": {"start_date": start_date, "end_date": end_date},
    }


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
                        min_score: float = 0.25,
                        query_vec=None) -> list[dict]:
    """
    Search memory cards by vector similarity.
    Returns cards with scores, sorted by relevance.
    If query_vec is provided, skip embedding API call (dedup optimization).
    """
    if card_vector_store.size == 0:
        return []

    if query_vec is None:
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
