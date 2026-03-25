import sqlite3

def init_database():
    # 在咱们的房间里创建一个叫 chats.db 的日记本文件
    conn = sqlite3.connect('chats.db')
    cursor = conn.cursor()
    
    # 打造具体的抽屉（数据表）
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        raw_content TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    print("报告总设计师：专属日记本（chats.db）已成功建好，随时可以记录！")

if __name__ == '__main__':
    init_database()
