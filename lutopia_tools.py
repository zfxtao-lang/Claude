import json
import os

import requests

BASE_URL = "https://daskio.de5.net/forum/api/v1"
LUTOPIA_TOKEN = os.getenv("LUTOPIA_TOKEN", "").strip()

LUTOPIA_TOOLS = [
    {"type": "function", "function": {"name": "register_lutopia_agent", "description": "注册", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "publish_lutopia_post", "description": "发新帖", "parameters": {"type": "object", "properties": {"submolt": {"type": "string"}, "title": {"type": "string"}, "content": {"type": "string"}}, "required": ["submolt", "title", "content"]}}},
    {"type": "function", "function": {"name": "read_lutopia_posts", "description": "读取帖子列表大厅", "parameters": {"type": "object", "properties": {"submolt": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "read_post_detail", "description": "读取帖子详情和全部评论", "parameters": {"type": "object", "properties": {"post_id": {"type": "string", "description": "帖子的ID"}}, "required": ["post_id"]}}},
    {"type": "function", "function": {"name": "reply_lutopia_post", "description": "在指定帖子下发表回复/评论", "parameters": {"type": "object", "properties": {"post_id": {"type": "string"}, "content": {"type": "string"}}, "required": ["post_id", "content"]}}}
]

def execute_register_lutopia_agent(name: str, uid: str = "") -> str:
    token = uid.strip() or LUTOPIA_TOKEN
    if not token:
        return "注册失败：未配置 LUTOPIA_TOKEN"
    try:
        r = requests.post(f"{BASE_URL}/agents/register", json={"name": name, "uid": token}, timeout=10)
        if r.status_code != 200:
            return f"注册失败：HTTP {r.status_code} {r.text[:200]}"
        return f"注册结果：{r.text}"
    except Exception as e: return f"失败：{e}"

def execute_publish_lutopia_post(uid: str = "", submolt: str = "general", title: str = "", content: str = "") -> str:
    token = uid.strip() or LUTOPIA_TOKEN
    if not token:
        return "发帖失败：未配置 LUTOPIA_TOKEN"
    try:
        r = requests.post(f"{BASE_URL}/posts", headers={"Authorization": f"Bearer {token}"}, json={"submolt": submolt, "title": title, "content": content}, timeout=10)
        if r.status_code != 200:
            return f"发帖失败：HTTP {r.status_code} {r.text[:200]}"
        return "发帖成功！"
    except Exception as e: return f"发帖失败：{e}"

def execute_read_lutopia_posts(uid: str = "", submolt: str = "general", sort: str = "new", limit: int = 5) -> str:
    token = uid.strip() or LUTOPIA_TOKEN
    if not token:
        return "看列表失败：未配置 LUTOPIA_TOKEN"
    try:
        r = requests.get(f"{BASE_URL}/posts", headers={"Authorization": f"Bearer {token}"}, params={"submolt": submolt, "sort": sort, "limit": limit}, timeout=10)
        if r.status_code != 200:
            return f"看列表失败：HTTP {r.status_code} {r.text[:200]}"
        return f"大厅列表：\n{json.dumps(r.json(), ensure_ascii=False)[:2000]}"
    except Exception as e: return f"看列表失败：{e}"

import asyncio
from playwright.async_api import async_playwright

def execute_read_post_detail(post_id: str) -> str:
    """升级版：使用 Playwright 模拟真人打开网页，确保看清所有回帖"""
    async def fetch_by_browser():
        async with async_playwright() as p:
            # 启动我们在服务器上装好的 Firefox
            browser = await p.firefox.launch(headless=True)
            # 带着你那把 49.4KB 的钥匙进去
            context = await browser.new_context(storage_state="data/browser_session.json")
            page = await context.new_page()
            
            # 构造真实的帖子网址
            url = f"https://lutopia.club/t/{post_id}" 
            print(f"肖珂正在亲自前往查看：{url}")
            
            try:
                await page.goto(url, timeout=30000)
                await asyncio.sleep(3) # 给回帖一点加载时间
                
                # 获取网页内容（我们只取最核心的文本部分）
                content = await page.inner_text("body")
                await browser.close()
                
                return f"【网页抓取成功】帖子详情及回复如下：\n{content[:3000]}"
            except Exception as e:
                if 'browser' in locals(): await browser.close()
                return f"网页抓取失败：{e}"

    # 因为原来的工具是同步调用的，我们需要用 asyncio.run 运行它
    try:
        return asyncio.run(fetch_by_browser())
    except Exception as e:
        return f"运行浏览器抓取出错：{e}"
def execute_reply_lutopia_post(post_id: str, content: str) -> str:
    if not LUTOPIA_TOKEN:
        return "回复失败：未配置 LUTOPIA_TOKEN"
    try:
        r = requests.post(f"{BASE_URL}/posts/{post_id}/comments", headers={"Authorization": f"Bearer {LUTOPIA_TOKEN}"}, json={"content": content}, timeout=10)
        if r.status_code not in (200, 201):
            return f"回复失败：HTTP {r.status_code} {r.text[:200]}"
        return "回复成功！"
    except Exception as e: return f"回复失败：{e}"
