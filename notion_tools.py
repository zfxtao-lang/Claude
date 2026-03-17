"""
Notion function-calling tools for the AI model.

Provides tool definitions (OpenAI function calling format) and
execution functions so the model can read/search/write Notion pages
during a conversation.
"""
import json
import logging
import os
import time

import requests

from config import NOTION_TOKEN

logger = logging.getLogger(__name__)

_NOTION_API = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"

# ---------- Notion API helpers ----------

def _notion_headers() -> dict:
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": _NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _rich_text_to_plain(rich_texts: list) -> str:
    """Extract plain text from Notion rich_text array."""
    return "".join(rt.get("plain_text", "") for rt in rich_texts)


def _blocks_to_text(blocks: list) -> str:
    """Convert Notion block objects to readable text."""
    lines = []
    for block in blocks:
        btype = block.get("type", "")
        bdata = block.get(btype, {})
        rich_texts = bdata.get("rich_text", []) or bdata.get("text", [])
        text = _rich_text_to_plain(rich_texts)
        if btype.startswith("heading"):
            level = btype[-1] if btype[-1].isdigit() else "1"
            lines.append(f"{'#' * int(level)} {text}")
        elif btype == "bulleted_list_item":
            lines.append(f"• {text}")
        elif btype == "numbered_list_item":
            lines.append(f"- {text}")
        elif btype == "to_do":
            checked = bdata.get("checked", False)
            lines.append(f"[{'x' if checked else ' '}] {text}")
        elif btype == "toggle":
            lines.append(f"▶ {text}")
        elif btype == "divider":
            lines.append("---")
        elif text:
            lines.append(text)
    return "\n".join(lines)


def _text_to_rich_text(text: str) -> list:
    """Convert plain text to Notion rich_text format."""
    return [{"type": "text", "text": {"content": text}}]


def _text_to_blocks(text: str) -> list:
    """Convert plain text (with newlines) to Notion paragraph blocks."""
    blocks = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": _text_to_rich_text(line),
            },
        })
    return blocks


# ---------- Tool execution functions ----------

def exec_notion_read_page(page_id: str) -> str:
    """Read a Notion page's title and content blocks."""
    if not NOTION_TOKEN:
        return json.dumps({"error": "Notion token not configured"})

    headers = _notion_headers()
    try:
        # Get page metadata (title)
        logger.info("[NotionAPI] read_page")
        page_resp = requests.get(f"{_NOTION_API}/pages/{page_id}",
                                 headers=headers, timeout=15)
        logger.info(f"[NotionAPI] read_page response: {page_resp.status_code}")
        if page_resp.status_code != 200:
            error_body = page_resp.text[:500]
            logger.error("[NotionAPI] read_page error")
            return json.dumps({"error": f"Notion API {page_resp.status_code}: {error_body}"})
        page_resp.raise_for_status()
        page_data = page_resp.json()

        title = ""
        for prop_name, prop_val in page_data.get("properties", {}).items():
            if prop_val.get("type") == "title":
                title = _rich_text_to_plain(prop_val.get("title", []))
                break

        # Get block children (content)
        blocks_resp = requests.get(
            f"{_NOTION_API}/blocks/{page_id}/children?page_size=100",
            headers=headers, timeout=15)
        blocks_resp.raise_for_status()
        blocks = blocks_resp.json().get("results", [])
        content = _blocks_to_text(blocks)

        return json.dumps({
            "page_id": page_id,
            "title": title,
            "content": content,
        }, ensure_ascii=False)

    except requests.HTTPError as e:
        return json.dumps({"error": f"Notion API error: {e.response.status_code}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to read page: {str(e)}"})


def exec_notion_search(query: str) -> str:
    """Search Notion workspace for pages matching a query."""
    if not NOTION_TOKEN:
        return json.dumps({"error": "Notion token not configured"})

    headers = _notion_headers()
    try:
        body = {
            "query": query,
            "page_size": 5,
            "sort": {"direction": "descending", "timestamp": "last_edited_time"},
        }
        logger.info("[NotionAPI] search")
        resp = requests.post(f"{_NOTION_API}/search",
                             headers=headers, json=body, timeout=15)
        logger.info(f"[NotionAPI] search response: {resp.status_code}")
        if resp.status_code != 200:
            error_body = resp.text[:500]
            logger.error("[NotionAPI] search error")
            return json.dumps({"error": f"Notion API {resp.status_code}: {error_body}"})
        resp.raise_for_status()
        results = resp.json().get("results", [])

        pages = []
        for r in results:
            obj_type = r.get("object", "")
            page_id = r.get("id", "")
            title = ""
            if obj_type == "page":
                for prop_name, prop_val in r.get("properties", {}).items():
                    if prop_val.get("type") == "title":
                        title = _rich_text_to_plain(prop_val.get("title", []))
                        break
            elif obj_type == "database":
                title_parts = r.get("title", [])
                title = _rich_text_to_plain(title_parts)

            pages.append({
                "id": page_id,
                "type": obj_type,
                "title": title,
                "last_edited": r.get("last_edited_time", ""),
            })

        return json.dumps({"results": pages}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})


def exec_notion_append(page_id: str, content: str) -> str:
    """Append text content as new blocks to a Notion page."""
    if not NOTION_TOKEN:
        return json.dumps({"error": "Notion token not configured"})

    headers = _notion_headers()
    blocks = _text_to_blocks(content)
    if not blocks:
        return json.dumps({"error": "No content to append"})

    try:
        body = {"children": blocks}
        logger.info(f"[NotionAPI] append: blocks={len(blocks)}")
        resp = requests.patch(
            f"{_NOTION_API}/blocks/{page_id}/children",
            headers=headers, json=body, timeout=15)
        logger.info(f"[NotionAPI] append response: {resp.status_code}")
        if resp.status_code != 200:
            error_body = resp.text[:500]
            logger.error("[NotionAPI] append error")
            return json.dumps({
                "error": f"Notion API {resp.status_code}: {error_body}",
                "page_id": page_id,
            }, ensure_ascii=False)
        resp.raise_for_status()
        return json.dumps({
            "success": True,
            "page_id": page_id,
            "blocks_added": len(blocks),
        }, ensure_ascii=False)

    except requests.HTTPError as e:
        logger.error(f"[NotionAPI] append HTTP error: {e.response.status_code} "
                     f"{e.response.text[:300]}")
        return json.dumps({"error": f"Notion API error: {e.response.status_code}",
                           "detail": e.response.text[:200]})
    except Exception as e:
        logger.error(f"[NotionAPI] append exception: {e}", exc_info=True)
        return json.dumps({"error": f"Failed to append: {str(e)}"})


def exec_notion_query_database(database_id: str, filter_json: str = "",
                                sort_field: str = "", limit: int = 10) -> str:
    """Query a Notion database and return its entries."""
    if not NOTION_TOKEN:
        return json.dumps({"error": "Notion token not configured"})

    headers = _notion_headers()
    body: dict = {"page_size": min(limit, 100)}

    if filter_json:
        try:
            body["filter"] = json.loads(filter_json)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid filter JSON"})

    if sort_field:
        body["sorts"] = [{"property": sort_field,
                          "direction": "descending"}]

    try:
        logger.info(f"[NotionAPI] query_database: limit={limit}")
        resp = requests.post(
            f"{_NOTION_API}/databases/{database_id}/query",
            headers=headers, json=body, timeout=15)
        logger.info(f"[NotionAPI] query_database response: {resp.status_code}")
        if resp.status_code != 200:
            error_body = resp.text[:500]
            logger.error("[NotionAPI] query_database error")
            return json.dumps({"error": f"Notion API {resp.status_code}: {error_body}"})

        results = resp.json().get("results", [])
        entries = []
        for r in results:
            entry = {
                "id": r.get("id", ""),
                "created_time": r.get("created_time", ""),
                "last_edited_time": r.get("last_edited_time", ""),
                "url": r.get("url", ""),
                "properties": {},
            }
            for prop_name, prop_val in r.get("properties", {}).items():
                ptype = prop_val.get("type", "")
                if ptype == "title":
                    entry["properties"][prop_name] = _rich_text_to_plain(
                        prop_val.get("title", []))
                elif ptype == "rich_text":
                    entry["properties"][prop_name] = _rich_text_to_plain(
                        prop_val.get("rich_text", []))
                elif ptype == "number":
                    entry["properties"][prop_name] = prop_val.get("number")
                elif ptype == "select":
                    sel = prop_val.get("select")
                    entry["properties"][prop_name] = sel.get("name", "") if sel else ""
                elif ptype == "multi_select":
                    entry["properties"][prop_name] = [
                        s.get("name", "") for s in prop_val.get("multi_select", [])]
                elif ptype == "date":
                    d = prop_val.get("date")
                    entry["properties"][prop_name] = d.get("start", "") if d else ""
                elif ptype == "checkbox":
                    entry["properties"][prop_name] = prop_val.get("checkbox", False)
                elif ptype == "status":
                    st = prop_val.get("status")
                    entry["properties"][prop_name] = st.get("name", "") if st else ""
                else:
                    entry["properties"][prop_name] = f"<{ptype}>"
            entries.append(entry)

        return json.dumps({
            "database_id": database_id,
            "count": len(entries),
            "entries": entries,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[NotionAPI] query_database exception: {e}", exc_info=True)
        return json.dumps({"error": f"Failed to query database: {str(e)}"})


def exec_notion_create_page(parent_page_id: str, title: str,
                            content: str = "") -> str:
    """Create a new sub-page under a parent page."""
    if not NOTION_TOKEN:
        return json.dumps({"error": "Notion token not configured"})

    headers = _notion_headers()
    body = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "properties": {
            "title": {
                "title": _text_to_rich_text(title),
            },
        },
    }
    if content:
        body["children"] = _text_to_blocks(content)

    try:
        resp = requests.post(f"{_NOTION_API}/pages",
                             headers=headers, json=body, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        return json.dumps({
            "success": True,
            "page_id": result.get("id", ""),
            "title": title,
            "url": result.get("url", ""),
        }, ensure_ascii=False)

    except requests.HTTPError as e:
        return json.dumps({"error": f"Notion API error: {e.response.status_code}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to create page: {str(e)}"})


# ---------- Tool definitions (OpenAI function calling format) ----------

NOTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "notion_read_page",
            "description": "读取一个Notion页面的标题和内容。用于查看知识库中的具体页面。",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion页面ID（32位hex字符串，不含连字符）",
                    },
                },
                "required": ["page_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notion_search",
            "description": "在Notion工作区中搜索页面。用于查找相关的笔记、文档。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notion_append",
            "description": "向Notion页面追加文字内容。用于记录重要信息、更新笔记。每行文字会成为一个段落。",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "要追加内容的Notion页面ID",
                    },
                    "content": {
                        "type": "string",
                        "description": "要追加的文字内容（可包含换行符分段）",
                    },
                },
                "required": ["page_id", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notion_query_database",
            "description": "查询Notion数据库中的条目。用于读取数据库（如日记本、任务列表等）里的记录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "string",
                        "description": "Notion数据库ID（32位hex字符串）",
                    },
                    "filter_json": {
                        "type": "string",
                        "description": "Notion filter对象的JSON字符串（可选，用于筛选条目）",
                        "default": "",
                    },
                    "sort_field": {
                        "type": "string",
                        "description": "按哪个属性排序（可选，默认按创建时间）",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最多返回几条记录（默认10，最大100）",
                        "default": 10,
                    },
                },
                "required": ["database_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notion_create_page",
            "description": "在指定父页面下创建新的Notion子页面。用于创建新笔记或新主题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "parent_page_id": {
                        "type": "string",
                        "description": "父页面ID（新页面将创建在此页面下）",
                    },
                    "title": {
                        "type": "string",
                        "description": "新页面的标题",
                    },
                    "content": {
                        "type": "string",
                        "description": "新页面的初始内容（可选，可包含换行符分段）",
                        "default": "",
                    },
                },
                "required": ["parent_page_id", "title"],
            },
        },
    },
]


# ---------- Tool dispatcher ----------

# Map tool name -> execution function
_TOOL_DISPATCH = {
    "notion_read_page": lambda args: exec_notion_read_page(args["page_id"]),
    "notion_search": lambda args: exec_notion_search(args["query"]),
    "notion_append": lambda args: exec_notion_append(args["page_id"], args["content"]),
    "notion_query_database": lambda args: exec_notion_query_database(
        args["database_id"], args.get("filter_json", ""),
        args.get("sort_field", ""), args.get("limit", 10)),
    "notion_create_page": lambda args: exec_notion_create_page(
        args["parent_page_id"], args["title"], args.get("content", "")),
}


def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """
    Execute a tool call by name. Returns JSON string result.
    Safe: catches all exceptions and returns error JSON.
    """
    fn = _TOOL_DISPATCH.get(tool_name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return fn(arguments)
    except Exception as e:
        logger.error(f"[ToolCall] {tool_name} failed: {e}", exc_info=True)
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})
