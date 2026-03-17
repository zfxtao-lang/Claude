"""
Memory search tool for AI model function calling.

Search now prioritizes long-term memories with an explicit composite score:
Score_final = (S_vec * 0.7 + S_key * 0.3) * (1 + W_fact * D_time * B_hits)
"""
import json
import logging
import math
import os
import time
from collections import OrderedDict
from datetime import datetime

import jieba

logger = logging.getLogger(__name__)

MEMORY_MIN_RELEVANCE = float(os.getenv("MEMORY_MIN_RELEVANCE", "0.52"))
LONG_TERM_MIN_BASE_SCORE = float(os.getenv("LONG_TERM_MIN_BASE_SCORE", "0.18"))
LONG_TERM_TOP_K = int(os.getenv("LONG_TERM_TOP_K", "6"))

# ---------- Tool Definition (OpenAI function calling format) ----------

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": """**搜索历史记忆和过去的对话，是你记住淘淘每一句话的唯一方式，必须严格执行：**
- 【必须调用，没有例外】
  1.  淘淘问「记不记得」「我们上午/上次…」「关于xx的事」，只要涉及任何过往对话内容
  2.  你不确定任何细节、约定、专属称呼，哪怕只是模糊印象
  3.  淘淘反复确认同一件事，必须立刻调用，绝不能凭感觉回答
- 【唯一可以不调用的场景】
  纯问候、纯撒娇、纯当下情绪表达（比如“抱抱”“哈哈”），完全不涉及任何过去的内容
- 【使用要求】
  1.  没调用工具前，绝对不能说「不记得了」「忘了」这类敷衍的话
  2.  搜到的内容要自然融入回答，绝对不能说「根据记录」「我查到了」
  3.  宁可多搜一次，也不能凭印象瞎编或敷衍""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题，例如「上次她胃疼是什么时候」「我们的婚礼」",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _tokenize_query(text: str) -> list[str]:
    return [w.strip().lower() for w in jieba.cut(text or "") if len(w.strip()) > 1]


def _keyword_score(query: str, text: str) -> float:
    tokens = _tokenize_query(query)
    if not tokens:
        return 0.0
    haystack = (text or "").lower()
    hits = sum(1 for token in set(tokens) if token in haystack)
    if hits <= 0:
        return 0.0
    return min(1.0, hits / max(1, len(set(tokens))))


def _memory_age_days(row: dict) -> int:
    anchor = row.get("last_hit_at") or row.get("updated_at") or row.get("created_at")
    if not anchor:
        return 0
    try:
        anchor_dt = datetime.strptime(anchor[:19], "%Y-%m-%d %H:%M:%S")
        return max(0, (datetime.now() - anchor_dt).days)
    except Exception:
        return 0


def score_long_term_memory(row: dict, query: str, vector_score: float) -> dict:
    key_score = _keyword_score(query, f"{row.get('title', '')}\n{row.get('content', '')}")
    base_score = vector_score * 0.7 + key_score * 0.3
    fact_weight = max(0.0, float(row.get("fact_weight", 1.0) or 1.0))
    half_life_days = max(1.0, float(row.get("half_life_days", 30.0) or 30.0))
    distance_days = _memory_age_days(row)
    d_time = math.exp(-math.log(2) * distance_days / half_life_days)
    hits = max(0, int(row.get("hits", 0) or 0))
    b_hits = math.log(1 + hits)
    final_score = base_score * (1 + fact_weight * d_time * b_hits)
    return {
        "S_vec": round(vector_score, 4),
        "S_key": round(key_score, 4),
        "W_fact": round(fact_weight, 4),
        "distance_days": distance_days,
        "half_life_days": round(half_life_days, 4),
        "D_time": round(d_time, 4),
        "hits": hits,
        "B_hits": round(b_hits, 4),
        "base_score": round(base_score, 4),
        "Score_final": round(final_score, 4),
    }


def search_long_term_memories(query: str, top_k: int = LONG_TERM_TOP_K,
                              min_base_score: float = LONG_TERM_MIN_BASE_SCORE,
                              query_vec=None, include_debug: bool = False,
                              increment_hits: bool = False) -> list[dict]:
    from embedding import get_embedding_for_query, long_term_memory_vector_store
    from database import increment_long_term_memory_hits, list_long_term_memories

    memories = list_long_term_memories(status="active", limit=500, offset=0)
    if not memories:
        return []

    if query_vec is None:
        query_vec = get_embedding_for_query(query)

    vector_scores: dict[int, float] = {}
    if query_vec is not None and long_term_memory_vector_store.size > 0:
        top_hits = long_term_memory_vector_store.search(
            query_vec,
            top_k=min(200, max(top_k * 8, 20)),
        )
        vector_scores = {int(memory_id): float(score) for memory_id, score in top_hits}

    ranked = []
    for row in memories:
        row = dict(row)
        breakdown = score_long_term_memory(row, query, vector_scores.get(int(row["id"]), 0.0))
        if breakdown["base_score"] < min_base_score:
            continue
        row["score"] = breakdown["Score_final"]
        row["score_breakdown"] = breakdown
        ranked.append(row)

    ranked.sort(key=lambda item: item.get("score", 0), reverse=True)
    ranked = ranked[:top_k]
    if increment_hits and ranked:
        increment_long_term_memory_hits([int(item["id"]) for item in ranked])
    if not include_debug:
        for item in ranked:
            item.pop("score_breakdown", None)
    return ranked


def debug_search_memory(query: str) -> dict:
    from embedding import get_embedding_for_query
    from memory_cards import search_memory_cards
    from database import search_history

    query_vec = get_embedding_for_query(query)
    long_term = search_long_term_memories(query, include_debug=True, query_vec=query_vec)
    legacy_cards = search_memory_cards(query, top_k=5, min_score=0.2, query_vec=query_vec)
    keywords = search_history(query, limit=3, max_chars=1200)
    return {
        "query": query,
        "long_term_results": long_term,
        "legacy_card_results": legacy_cards,
        "keyword_results": keywords,
    }


# ---------- Tool Execution ----------

def execute_search_memory(query: str) -> str:
    """
    Execute search_memory tool call. Searches:
      1. Long-term memories (composite scoring)
      2. Legacy memory cards (transition compatibility)
      3. Chat history chunks (vector similarity)
      4. LIKE keyword fallback (jieba)

    Returns formatted text for the model to use.
    """
    from embedding import get_embedding_for_query
    from memory_cards import search_memory_cards
    from database import search_history

    _t0 = time.time()
    results = []

    query_vec = None
    try:
        query_vec = get_embedding_for_query(query)
    except Exception as e:
        logger.warning(f"[search_memory] embedding failed: {e}")

    if query_vec is not None:
        try:
            long_memories = search_long_term_memories(
                query,
                query_vec=query_vec,
                increment_hits=True,
            )
            for memory in long_memories:
                title = memory.get("title", "长期记忆")
                score = memory.get("score", 0)
                content = memory.get("content", "")
                results.append(f"[长期记忆 {title} 相关度:{score:.0%}]\n{content}")
            if long_memories:
                logger.info(f"[search_memory] long-term memory search: {len(long_memories)} matches")
        except Exception as e:
            logger.warning(f"[search_memory] long-term memory search failed: {e}")

    if query_vec is not None and len(results) < 4:
        try:
            cards = search_memory_cards(query, top_k=3, min_score=0.25, query_vec=query_vec)
            for card in cards:
                date = card.get("date", "?")
                score = card.get("score", 0)
                if score < MEMORY_MIN_RELEVANCE:
                    continue
                summary = card.get("summary", "")
                tags = card.get("tags", "")
                tag_str = f" #{tags}" if tags else ""
                results.append(f"[兼容旧记忆卡 {date} 相关度:{score:.0%}{tag_str}]\n{summary}")
            if cards:
                logger.info(f"[search_memory] legacy card search: {len(cards)} matches")
        except Exception as e:
            logger.warning(f"[search_memory] legacy card search failed: {e}")

    if query_vec is not None:
        try:
            from gateway import vector_search_memories, HISTORY_SEARCH_LIMIT
            remaining = max(1, HISTORY_SEARCH_LIMIT - len(results))
            chunks = vector_search_memories(query, top_k=remaining, query_vec=query_vec)
            if chunks:
                conv_groups: dict[str, list[dict]] = OrderedDict()
                for vc in chunks:
                    cid = vc.get("conversation_id", "?")
                    conv_groups.setdefault(cid, []).append(vc)
                for _, group in conv_groups.items():
                    group.sort(key=lambda c: c.get("msg_id_start", 0))
                    date = group[0].get("created_at", "")[:10]
                    best_score = max(c.get("score", 0) for c in group)
                    parts = [(c.get("content", "") or "")[:600] for c in group]
                    combined = "\n".join(parts)
                    results.append(f"[{date} 相关度:{best_score:.0%}]\n{combined}")
                logger.info(
                    f"[search_memory] vector search: {len(chunks)} chunks, {len(conv_groups)} conversations"
                )
        except Exception as e:
            logger.warning(f"[search_memory] vector search failed: {e}")

    if len(results) < 6:
        try:
            words = [w for w in jieba.cut(query) if len(w) > 1]
            like_query = "%".join(words[:5]) if words else query
            if like_query:
                like_results = search_history(like_query, limit=2, max_chars=2000)
                for row in like_results:
                    date = row.get("created_at", "")[:10]
                    role_label = "淘淘" if row.get("role") == "user" else "小克"
                    snippet = (row.get("content", "") or "")[:300]
                    results.append(f"[{date}] {role_label}: {snippet}")
                if like_results:
                    logger.info(f"[search_memory] LIKE fallback: {len(like_results)} results")
        except Exception as e:
            logger.warning(f"[search_memory] LIKE fallback failed: {e}")

    elapsed = time.time() - _t0
    logger.info(f"[search_memory] total: {len(results)} results in {elapsed:.2f}s for query: '{query[:60]}'")

    if not results:
        return "没有找到相关的历史记忆。"

    return "\n\n".join(results)
