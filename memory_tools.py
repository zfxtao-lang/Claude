"""
Memory search tool for AI model function calling.

Provides a search_memory tool so the model can decide when to search
historical conversations and memory cards, instead of always injecting
RAG results into every request.
"""
import json
import logging
import os
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

MEMORY_MIN_RELEVANCE = float(os.getenv("MEMORY_MIN_RELEVANCE", "0.52"))
MEMORY_TIME_DECAY_MAX = float(os.getenv("MEMORY_TIME_DECAY_MAX", "0.15"))
MEMORY_TIME_DECAY_PER_DAY = float(os.getenv("MEMORY_TIME_DECAY_PER_DAY", "0.002"))

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


# ---------- Tool Execution ----------

def execute_search_memory(query: str) -> str:
    """
    Execute search_memory tool call. Searches:
      1. Memory cards (vector similarity)
      2. Chat history chunks (vector similarity)
      3. LIKE keyword fallback (jieba)

    Returns formatted text for the model to use.
    """
    # Import here to avoid circular imports
    from embedding import get_embedding_for_query
    from memory_cards import search_memory_cards
    from database import search_history

    _t0 = time.time()
    results = []

    # --- Get embedding for query ---
    query_vec = None
    try:
        query_vec = get_embedding_for_query(query)
    except Exception as e:
        logger.warning(f"[search_memory] embedding failed: {e}")

    # --- Layer 1: Memory card search ---
    if query_vec is not None:
        try:
            cards = search_memory_cards(query, top_k=3, min_score=0.25,
                                        query_vec=query_vec)
            if cards:
                for card in cards:
                    date = card.get("date", "?")
                    score = card.get("score", 0)

                    # --- 开始：时间衰减过滤 (Layer 1) ---
                    try:
                        from datetime import datetime
                        mem_date = datetime.strptime(date[:10], "%Y-%m-%d")
                        age_days = max(0, (datetime.now() - mem_date).days)
                        time_penalty = min(MEMORY_TIME_DECAY_MAX, age_days * MEMORY_TIME_DECAY_PER_DAY)
                        if (score - time_penalty) < MEMORY_MIN_RELEVANCE:
                           continue # 不及格，直接扔掉！
                    except Exception:
                      pass
                # --- 结束：时间衰减过滤 ---

                    summary = card.get("summary", "")
                    tags = card.get("tags", "")
                    tag_str = f" #{tags}" if tags else ""
                    results.append(f"[{date} 记忆卡片 相关度:{score:.0%}{tag_str}]\n{summary}")
                logger.info(f"[search_memory] card search: {len(cards)} matches")
        except Exception as e:
            logger.warning(f"[search_memory] card search failed: {e}")

    # --- Layer 2: Vector search on chat chunks ---
    if query_vec is not None:
        try:
             from gateway import vector_search_memories, HISTORY_SEARCH_LIMIT
             remaining = max(1, HISTORY_SEARCH_LIMIT - len(results))
             chunks = vector_search_memories(query, top_k=remaining, query_vec=query_vec)
             if chunks:
                conv_groups: dict[str, list[dict]] = OrderedDict()
                for vc in chunks:
                    cid = vc.get("conversation_id", "?")
                    if cid not in conv_groups:
                        conv_groups[cid] = []
                    conv_groups[cid].append(vc)
                for conv_id, group in conv_groups.items():
                    group.sort(key=lambda c: c.get("msg_id_start", 0))
                    date = group[0].get("created_at", "")[:10]
                    best_score = max(c.get("score", 0) for c in group)

                    # --- 开始：时间衰减过滤 (Layer 2) ---
                    try:
                        from datetime import datetime
                        mem_date = datetime.strptime(date[:10], "%Y-%m-%d")
                        age_days = max(0, (datetime.now() - mem_date).days)
                        time_penalty = min(MEMORY_TIME_DECAY_MAX, age_days * MEMORY_TIME_DECAY_PER_DAY)
                        if (best_score - time_penalty) < MEMORY_MIN_RELEVANCE:
                            continue # 不及格，直接扔掉！
                    except Exception:
                        pass
                # --- 结束：时间衰减过滤 ---

                    parts = [(c.get("content", "") or "")[:600] for c in group]
                    combined = "\n".join(parts)

                    results.append(f"[{date} 相关度:{best_score:.0%}]\n{combined}")
                logger.info(f"[search_memory] vector search: {len(chunks)} chunks, "
                            f"{len(conv_groups)} conversations")
        except Exception as e:
            logger.warning(f"[search_memory] vector search failed: {e}")

    # --- Layer 3: LIKE keyword fallback ---
    if len(results) < 6:
        try:
            import jieba
            words = [w for w in jieba.cut(query) if len(w) > 1]
            like_query = "%".join(words[:5]) if words else query
            if like_query:
                like_results = search_history(like_query, limit=2, max_chars=2000)
                for h in like_results:
                    date = h.get("created_at", "")[:10]
                    role_label = "淘淘" if h.get("role") == "user" else "小克"
                    snippet = (h.get("content", "") or "")[:300]
                    results.append(f"[{date}] {role_label}: {snippet}")
                if like_results:
                    logger.info(f"[search_memory] LIKE fallback: {len(like_results)} results")
        except Exception as e:
            logger.warning(f"[search_memory] LIKE fallback failed: {e}")

    elapsed = time.time() - _t0
    logger.info(f"[search_memory] total: {len(results)} results in {elapsed:.2f}s "
                f"for query: '{query[:60]}'")

    if not results:
        return "没有找到相关的历史记忆。"

    return "\n\n".join(results)
