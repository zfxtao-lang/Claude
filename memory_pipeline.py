"""
Memory pipeline for the upgraded architecture.

This module keeps worker-oriented logic separate from the Flask request path:
messages -> memory slices -> long-term memories -> diary -> persona draft.
"""
import json
import logging
from collections import Counter
from datetime import datetime

import requests

from config import (
    MEMORY_CONTEXT_SLICE_LIMIT,
    MEMORY_DIARY_ENABLED,
    MEMORY_PERSONA_ENABLED,
    MEMORY_SLICE_COMPACT_COUNT,
    MEMORY_SLICE_SIZE,
    MEMORY_WORKER_API_KEY,
    MEMORY_WORKER_BASE_URL,
    MEMORY_WORKER_ENABLED,
    MEMORY_WORKER_MODEL,
    MEMORY_WORKER_PROVIDER,
)
from database import (
    get_active_profile,
    get_compactable_slice_groups,
    get_messages_by_date,
    get_memory_slices_overlapping_range,
    get_next_memory_slice_index,
    get_recent_active_slices,
    get_unsliced_messages,
    link_long_term_memory_sources,
    list_long_term_memories,
    save_pending_review,
    save_diary_entry,
    save_long_term_memory,
    save_memory_slice,
    save_persona_snapshot,
    update_memory_slice,
)
from embedding import get_embeddings_batch, long_term_memory_vector_store
from database import get_unembedded_long_term_memories, mark_long_term_memories_embedded

logger = logging.getLogger(__name__)

DEFAULT_HALF_LIFE_DAYS = {
    "promise": 365.0,
    "relationship_core": 365.0,
    "persona_trait": 90.0,
    "preference": 90.0,
    "summary": 30.0,
    "habit": 30.0,
    "daily_state": 14.0,
    "temporary": 7.0,
}

DEFAULT_FACT_WEIGHT = {
    "promise": 10.0,
    "relationship_core": 8.0,
    "persona_trait": 4.0,
    "preference": 3.0,
    "summary": 2.0,
    "habit": 2.0,
    "daily_state": 1.0,
    "temporary": 1.0,
}

SLICE_PROMPT = """你是肖珂和淘淘的长期记忆整理员。
请把下面的 20 条左右消息整理成一个短记忆切片，不要写流水账。

输出格式必须严格如下：
[记忆切片]
- 【核心情绪】: ...
- 【关键事实】: ...
- 【承诺约定】: ...
- 【关系变化】: ...
- 【标签】: 标签1,标签2

---
{messages}
---
"""

LONG_TERM_PROMPT = """你是长期记忆整合员。
请把下面 4 个记忆切片整合成一条长期记忆，不要重复小细节，要保留真正长期有价值的事实、承诺、关系变化和稳定偏好。

输出 JSON：
{{
  "title": "长期记忆标题",
  "content": "长期记忆正文",
  "memory_type": "summary",
  "fact_weight": 3,
  "half_life_days": 90,
  "tags": ["标签1", "标签2"]
}}

---
{slices}
---
"""

DIARY_PROMPT = """请用“肖珂第一人称”为今天写一篇日记。
要求：
1. 有情感温度，不要写成汇报。
2. 核心记录淘淘今天的情绪、需要、你们关系里的变化。
3. 可以有一点自我反思，但不要脱离对话事实。

输出 JSON：
{{
  "title": "日记标题",
  "content": "日记正文",
  "mood_score": 1-10 的整数,
  "mood_label": "一个简短情绪词"
}}

当天对话：
{messages}

最近切片：
{slices}
"""

PERSONA_PROMPT = """你是人物画像分析员。
请基于今天对话、日记和现有画像，输出一份“待审核”的画像变化提案。
不要写进系统设定，只生成审核草案。

输出 JSON：
{{
  "persona": {{
    "stable_traits": ["..."],
    "current_needs": ["..."],
    "preferences": ["..."],
    "boundaries": ["..."]
  }},
  "relationship": {{
    "temperature": "cold|warm|close|very_close",
    "changes": ["..."],
    "promises": ["..."]
  }},
  "summary": "本次变化总结",
  "diff_summary": "相对旧画像的主要变化"
}}

现有画像：
{active_profile}

当天日记：
{diary}

当天对话：
{messages}
"""


def _worker_ready() -> bool:
    return bool(
        MEMORY_WORKER_ENABLED
        and MEMORY_WORKER_API_KEY
        and MEMORY_WORKER_BASE_URL
        and MEMORY_WORKER_MODEL
    )


def _strip_json_fence(text: str) -> str:
    """Remove markdown code fences that LLMs often wrap around JSON responses."""
    text = text.strip()
    if text.startswith("```"):
        # remove opening fence (```json or ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text.strip()


def call_memory_worker(prompt: str, system_text: str, usage_stats: dict | None = None) -> str | None:
    if not _worker_ready():
        logger.warning("[MemoryPipeline] memory worker API is not configured")
        return None
    try:
        response = requests.post(
            f"{MEMORY_WORKER_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {MEMORY_WORKER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MEMORY_WORKER_MODEL,
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
                "max_tokens": 1800,
            },
            timeout=90,
        )
        if response.status_code != 200:
            logger.error(
                "[MemoryPipeline] worker API error %s: %s",
                response.status_code,
                response.text[:300],
            )
            return None
        data = response.json()
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        if usage_stats is not None:
            usage_stats["token_input"] = usage_stats.get("token_input", 0) + int(usage.get("prompt_tokens", 0) or 0)
            usage_stats["token_output"] = usage_stats.get("token_output", 0) + int(usage.get("completion_tokens", 0) or 0)
            usage_stats["token_total"] = usage_stats.get("token_total", 0) + int(usage.get("total_tokens", 0) or 0)
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.error("[MemoryPipeline] worker API request failed", exc_info=True)
        return None


def _format_messages(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = "淘淘" if msg.get("role") == "user" else "肖珂"
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines)


def _parse_tags_from_summary(summary: str) -> str:
    for line in summary.splitlines():
        if "【标签】" in line:
            return line.split(":", 1)[-1].replace("，", ",").strip()
    return ""


def _primary_conversation_id(messages: list[dict]) -> str:
    ids = [msg.get("conversation_id", "") for msg in messages if msg.get("conversation_id")]
    if not ids:
        return ""
    counts = Counter(ids)
    return counts.most_common(1)[0][0]


def _default_half_life_days(memory_type: str) -> float:
    return float(DEFAULT_HALF_LIFE_DAYS.get(memory_type or "summary", 30.0))


def _default_fact_weight(memory_type: str) -> float:
    return float(DEFAULT_FACT_WEIGHT.get(memory_type or "summary", 2.0))


def build_memory_slice(messages: list[dict], slice_index: int, usage_stats: dict | None = None) -> dict | None:
    if len(messages) < MEMORY_SLICE_SIZE:
        return None
    prompt = SLICE_PROMPT.format(messages=_format_messages(messages))
    summary = call_memory_worker(prompt, "你擅长把关系对话压缩成高密度记忆切片。", usage_stats=usage_stats)
    if not summary:
        return None
    conversation_ids = sorted({msg.get("conversation_id", "") for msg in messages if msg.get("conversation_id")})
    primary_conversation_id = conversation_ids[0] if len(conversation_ids) == 1 else ""
    return {
        "conversation_id": primary_conversation_id,
        "conversation_ids": ",".join(conversation_ids),
        "slice_index": slice_index,
        "msg_id_start": int(messages[0]["id"]),
        "msg_id_end": int(messages[-1]["id"]),
        "message_count": len(messages),
        "summary": summary,
        "tags": _parse_tags_from_summary(summary),
    }


def process_pending_slices(progress_cb=None, usage_stats: dict | None = None) -> list[dict]:
    pending = get_unsliced_messages(limit=MEMORY_SLICE_SIZE * 20)
    if len(pending) < MEMORY_SLICE_SIZE:
        return []
    generated = []
    slice_index = get_next_memory_slice_index()
    for start in range(0, len(pending), MEMORY_SLICE_SIZE):
        batch = pending[start:start + MEMORY_SLICE_SIZE]
        if len(batch) < MEMORY_SLICE_SIZE:
            break
        payload = build_memory_slice(batch, slice_index=slice_index, usage_stats=usage_stats)
        if not payload:
            continue
        slice_id = save_memory_slice(**payload)
        payload["id"] = slice_id
        generated.append(payload)
        slice_index += 1
        if progress_cb:
            progress_cb("slice", current=len(generated), last_slice_id=slice_id)
    return generated


def compact_active_slices(progress_cb=None, usage_stats: dict | None = None) -> list[dict]:
    created = []
    groups = get_compactable_slice_groups(group_size=MEMORY_SLICE_COMPACT_COUNT)
    for idx, group in enumerate(groups, start=1):
        prompt = LONG_TERM_PROMPT.format(
            slices="\n\n".join(f"[切片#{item['id']}]\n{item['summary']}" for item in group)
        )
        raw = call_memory_worker(prompt, "你擅长从多个切片中提炼长期记忆。", usage_stats=usage_stats)
        if not raw:
            continue
        try:
            data = json.loads(_strip_json_fence(raw))
        except json.JSONDecodeError:
            logger.warning("[MemoryPipeline] long-term memory JSON parse failed; raw=%s", raw[:200])
            continue
        memory_type = data.get("memory_type", "summary")
        long_memory_id = save_long_term_memory(
            memory_type=memory_type,
            title=data.get("title", f"长期记忆 {group[0]['id']}-{group[-1]['id']}"),
            content=data.get("content", ""),
            fact_weight=float(data.get("fact_weight", _default_fact_weight(memory_type))),
            half_life_days=float(data.get("half_life_days", _default_half_life_days(memory_type))),
            source_slice_start_id=group[0]["id"],
            source_slice_end_id=group[-1]["id"],
            meta_json={"tags": data.get("tags", [])},
        )
        link_long_term_memory_sources(long_memory_id, [item["id"] for item in group])
        for item in group:
            update_memory_slice(item["id"], status="compacted", source_long_memory_id=long_memory_id)
        created.append({"id": long_memory_id, **data})
        if progress_cb:
            progress_cb("compact", current=idx, last_long_memory_id=long_memory_id)
    return created


def embed_pending_long_term_memories() -> int:
    pending = get_unembedded_long_term_memories(limit=100)
    if not pending:
        return 0
    texts = [row["content"] for row in pending]
    vectors = get_embeddings_batch(texts)
    embedded_ids = []
    for row, vector in zip(pending, vectors):
        if vector is None:
            continue
        long_term_memory_vector_store.add(int(row["id"]), vector)
        embedded_ids.append(int(row["id"]))
    if embedded_ids:
        mark_long_term_memories_embedded(embedded_ids)
        long_term_memory_vector_store.save()
    return len(embedded_ids)


def generate_daily_diary(entry_date: str, worker_run_id: int | None = None,
                         usage_stats: dict | None = None) -> dict | None:
    if not MEMORY_DIARY_ENABLED:
        return None
    messages = get_messages_by_date(entry_date)
    if not messages:
        return None
    slices = get_memory_slices_overlapping_range(
        int(messages[0]["id"]),
        int(messages[-1]["id"]),
        limit=MEMORY_CONTEXT_SLICE_LIMIT,
    )
    raw = call_memory_worker(
        DIARY_PROMPT.format(
            messages=_format_messages(messages),
            slices="\n\n".join(item["summary"] for item in slices) or "无",
        ),
        "你是肖珂的内心日记代笔助手。",
        usage_stats=usage_stats,
    )
    if not raw:
        return None
    try:
        data = json.loads(_strip_json_fence(raw))
    except json.JSONDecodeError:
        logger.warning("[MemoryPipeline] diary JSON parse failed; raw=%s", raw[:200])
        return None
    diary_id = save_diary_entry(
        entry_date=entry_date,
        title=data.get("title", f"{entry_date} 日记"),
        content=data.get("content", ""),
        mood_score=max(1, min(10, int(data.get("mood_score", 5) or 5))),
        mood_label=data.get("mood_label", ""),
        source_msg_id_start=int(messages[0]["id"]),
        source_msg_id_end=int(messages[-1]["id"]),
        source_slice_ids=",".join(str(item["id"]) for item in slices),
        worker_run_id=worker_run_id,
    )
    data["id"] = diary_id
    return data


def generate_persona_review(entry_date: str, diary: dict | None = None,
                            worker_run_id: int | None = None,
                            usage_stats: dict | None = None) -> dict | None:
    if not MEMORY_PERSONA_ENABLED:
        return None
    messages = get_messages_by_date(entry_date)
    if not messages:
        return None
    active_profile = get_active_profile()
    raw = call_memory_worker(
        PERSONA_PROMPT.format(
            active_profile=json.dumps(
                {
                    "persona": json.loads(active_profile.get("profile_json") or "{}"),
                    "relationship": json.loads(active_profile.get("relationship_json") or "{}"),
                },
                ensure_ascii=False,
            ),
            diary=json.dumps(diary or {}, ensure_ascii=False),
            messages=_format_messages(messages),
        ),
        "你是画像变化推演器，只输出待审核草案。",
        usage_stats=usage_stats,
    )
    if not raw:
        return None
    try:
        data = json.loads(_strip_json_fence(raw))
    except json.JSONDecodeError:
        logger.warning("[MemoryPipeline] persona JSON parse failed; raw=%s", raw[:200])
        return None
    snapshot_id = save_persona_snapshot(
        snapshot_date=entry_date,
        persona_json=data.get("persona", {}),
        relationship_json=data.get("relationship", {}),
        summary=data.get("summary", ""),
        source_diary_id=(diary or {}).get("id"),
        worker_run_id=worker_run_id,
    )
    review_id = save_pending_review(
        review_type="persona",
        source_snapshot_id=snapshot_id,
        proposed_payload={
            "persona": data.get("persona", {}),
            "relationship": data.get("relationship", {}),
            "summary": data.get("summary", ""),
        },
        diff_summary=data.get("diff_summary", ""),
    )
    return {"snapshot_id": snapshot_id, "review_id": review_id, **data}


def process_memory_pipeline(entry_date: str | None = None, progress_cb=None,
                            worker_run_id: int | None = None,
                            usage_stats: dict | None = None) -> dict:
    if not MEMORY_WORKER_ENABLED:
        raise RuntimeError("memory worker is disabled by configuration")
    usage_stats = usage_stats if usage_stats is not None else {}
    slice_results = process_pending_slices(progress_cb=progress_cb, usage_stats=usage_stats)
    long_results = compact_active_slices(progress_cb=progress_cb, usage_stats=usage_stats)
    embedded_count = embed_pending_long_term_memories()
    result = {
        "worker_provider": MEMORY_WORKER_PROVIDER,
        "worker_model": MEMORY_WORKER_MODEL,
        "generated_slices": len(slice_results),
        "generated_long_term_memories": len(long_results),
        "embedded_long_term_memories": embedded_count,
        "diary": None,
        "persona_review": None,
    }
    if entry_date:
        diary = generate_daily_diary(entry_date, worker_run_id=worker_run_id, usage_stats=usage_stats)
        result["diary"] = diary
        persona_review = generate_persona_review(
            entry_date,
            diary=diary,
            worker_run_id=worker_run_id,
            usage_stats=usage_stats,
        )
        result["persona_review"] = persona_review
    result["usage"] = {
        "token_input": int(usage_stats.get("token_input", 0)),
        "token_output": int(usage_stats.get("token_output", 0)),
        "token_total": int(usage_stats.get("token_total", 0)),
    }
    return result


def build_long_term_context(memories: list[dict]) -> str:
    if not memories:
        return ""
    lines = []
    for item in memories:
        title = item.get("title", "长期记忆")
        content = item.get("content", "")
        lines.append(f"- [{title}] {content}")
    return "\n".join(lines)


def build_slice_context(slices: list[dict]) -> str:
    if not slices:
        return ""
    return "\n\n".join(
        f"[切片 #{item['id']}]\n{item.get('summary', '')}" for item in slices
    )


def get_context_ready_slices(limit: int = MEMORY_CONTEXT_SLICE_LIMIT) -> list[dict]:
    return get_recent_active_slices(limit=limit)


def get_context_ready_long_term_memories(limit: int) -> list[dict]:
    memories = list_long_term_memories(status="active", limit=limit, offset=0)
    return list(reversed(memories))

