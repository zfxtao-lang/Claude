"""
Notion content cache - avoids downloading on every request
"""
import json
import os
import time
import logging

import requests

from config import NOTION_TOKEN, NOTION_PAGE_IDS, NOTION_CACHE_TTL

logger = logging.getLogger(__name__)

_cache: dict[str, dict] = {}  # page_id -> {"content": str, "fetched_at": float}


def _fetch_notion_page(page_id: str) -> str:
    """Fetch a Notion page's content via API."""
    if not NOTION_TOKEN:
        return ""
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
    }
    try:
        # Get block children (page content)
        url = f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=100"
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        blocks = resp.json().get("results", [])

        texts = []
        for block in blocks:
            btype = block.get("type", "")
            bdata = block.get(btype, {})
            rich_texts = bdata.get("rich_text", []) or bdata.get("text", [])
            for rt in rich_texts:
                plain = rt.get("plain_text", "")
                if plain:
                    texts.append(plain)
        return "\n".join(texts)
    except Exception as e:
        logger.warning(f"Failed to fetch Notion page {page_id}: {e}")
        return ""


def get_notion_content(force_refresh: bool = False) -> str:
    """Get all Notion pages content, using cache with TTL."""
    now = time.time()
    all_content = []

    for page_id in NOTION_PAGE_IDS:
        page_id = page_id.strip()
        if not page_id:
            continue

        cached = _cache.get(page_id)
        if cached and not force_refresh and (now - cached["fetched_at"]) < NOTION_CACHE_TTL:
            content = cached["content"]
        else:
            content = _fetch_notion_page(page_id)
            _cache[page_id] = {"content": content, "fetched_at": now}

        if content:
            all_content.append(content)

    return "\n---\n".join(all_content)


def invalidate_cache():
    """Clear all cached Notion content."""
    _cache.clear()
