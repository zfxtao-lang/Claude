import sqlite3
import re

def clean_content(text):
    """这把物理手术刀负责切掉 <think> 标签及其里面的废话"""
    if not text:
        return ""
    # 刀起刀落，切掉 think 里面的内容
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 把前后多余的空格和回车洗干净
    return cleaned.strip()

def save_to_diary(role, raw_content):
    """把洗干净的话存进日记本"""
    try:
        content = clean_content(raw_content)
        # 如果洗完之后发现没内容了（比如全是废话），就不记这一笔
        if not content:
            return
            
        conn = sqlite3.connect('chats.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (role, content, raw_content) VALUES (?, ?, ?)",
            (role, content, raw_content)
        )
        conn.commit()
        conn.close()
        print(f"[{role}] 的消息已成功记入日记本！")
    except Exception as e:
        print(f"哎呀，记日记的时候笔摔了一下: {e}")
