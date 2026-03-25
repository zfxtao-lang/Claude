import os
from dotenv import load_dotenv
from notion_client import Client

# 打开密码箱
load_dotenv()
notion = Client(auth=os.getenv("NOTION_API_KEY"))

def read_notion_page(page_id, page_name):
    print(f"\n🕵️ 大管家正在努力读取【{page_name}】...")
    try:
        response = notion.blocks.children.list(block_id=page_id)
        text_content = ""
        
        # 告诉机器人：这些格式的文字全都要抓回来！
        text_block_types = [
            'paragraph', 'bulleted_list_item', 'numbered_list_item',
            'heading_1', 'heading_2', 'heading_3', 'quote', 'callout'
        ]
        
        for block in response.get('results', []):
            block_type = block['type']
            # 如果是文字类型的块，并且里面有字
            if block_type in text_block_types and block[block_type]['rich_text']:
                # 把这一行里面所有被加粗、变色的字都拼起来
                for text_part in block[block_type]['rich_text']:
                    text_content += text_part['plain_text']
                text_content += "\n"
        
        if text_content.strip():
            print(f"🎉 读取成功！内容预览 (前200字)：\n{text_content[:200]}...")
            print("-" * 30)
        else:
            print("⚠️ 房间进去了，但里面全都是图片、表格或者没写字哦？")
            
    except Exception as e:
        print(f"❌ 读取失败啦，保安说：{e}")

if __name__ == "__main__":
    # 淘淘给的两个核心门牌号
    page1_id = "2de7ad9d7c2c81899fcee954bd20afe4"
    page2_id = "2de7ad9d7c2c814a9061d7f90316eef1"
    
    read_notion_page(page1_id, "设定与承诺")
    read_notion_page(page2_id, "关系本质与交流偏好")
