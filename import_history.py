import sqlite3
import re
from datetime import datetime
import os

DB_FILE = "chats.db"

def parse_chat_file(file_path):
    """解析聊天记录TXT文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配格式：### **淘淘/用户** (时间戳) 或 ### **小克/AI** (时间戳)
    # 支持：淘淘、用户、小克、AI、肖珂
    pattern = r'### \*\*(?P<role>淘淘|用户|小克|AI|肖珂)\*\*\s*(?:\((?P<timestamp>[\d\-\s:]+)\))?\s*\n(?P<content>.*?)(?=### \*\*(?:淘淘|用户|小克|AI|肖珂)\*\*|$)'
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    messages = []
    for match in matches:
        role_name, timestamp_str, msg_content = match
        
        # 转换角色名：淘淘/用户 -> user，小克/AI/肖珂 -> assistant
        if role_name in ["淘淘", "用户"]:
            role = "user"
        else:  # 小克、AI、肖珂
            role = "assistant"
        
        # 处理时间戳
        if timestamp_str:
            timestamp_str = timestamp_str.strip()
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
            except ValueError:
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    timestamp = datetime.now()
        else:
            timestamp = datetime.now()
        
        # 清理内容
        msg_content = msg_content.strip()
        
        if msg_content:
            messages.append({
                "role": role,
                "content": msg_content,
                "timestamp": timestamp
            })
    
    return messages

def import_to_database(messages):
    """将消息导入数据库"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 确保表存在
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            raw_content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    imported_count = 0
    for msg in messages:
        # 清洗内容
        cleaned = re.sub(r'\s+', ' ', msg["content"]).strip()
        
        c.execute('''
            INSERT INTO chat_history (role, content, raw_content, created_at)
            VALUES (?, ?, ?, ?)
        ''', (msg["role"], cleaned, msg["content"], msg["timestamp"]))
        imported_count += 1
    
    conn.commit()
    conn.close()
    
    return imported_count

def rebuild_fts_index():
    """重建FTS索引"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 检查FTS表是否存在
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_fts'")
    if c.fetchone():
        c.execute("INSERT INTO chat_fts(chat_fts) VALUES('rebuild')")
        print("✅ FTS索引已重建")
    
    conn.commit()
    conn.close()

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("=" * 50)
        print("聊天记录导入工具")
        print("=" * 50)
        print("")
        print("用法: python import_history.py <聊天记录文件.txt>")
        print("")
        print("示例:")
        print("  python import_history.py chat_2025.txt")
        print("  python import_history.py file1.txt file2.txt file3.txt")
        print("")
        print("支持的格式:")
        print("  ### **淘淘** (2025-12-30 07:29)")
        print("  消息内容")
        print("")
        print("  ### **小克** (2025-12-30 07:30)")
        print("  消息内容")
        print("")
        print("支持的角色名: 淘淘、用户、小克、AI、肖珂")
        print("=" * 50)
        return
    
    total_imported = 0
    
    for file_path in sys.argv[1:]:
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue
        
        print(f"⏳ 正在解析: {file_path}")
        messages = parse_chat_file(file_path)
        print(f"   找到 {len(messages)} 条消息")
        
        if messages:
            count = import_to_database(messages)
            print(f"✅ 已导入 {count} 条消息")
            total_imported += count
    
    if total_imported > 0:
        print(f"\n📊 总计导入: {total_imported} 条消息")
        rebuild_fts_index()
        print("🎉 导入完成！")
    else:
        print("\n⚠️ 没有导入任何消息")

if __name__ == "__main__":
    main()
