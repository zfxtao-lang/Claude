import re

with open('/home/ubuntu/my_cyber_home/gateway.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 升级导入名单
code = re.sub(
    r'from lutopia_tools import LUTOPIA_TOOLS.*',
    'from lutopia_tools import LUTOPIA_TOOLS, execute_register_lutopia_agent, execute_publish_lutopia_post, execute_read_lutopia_posts, execute_read_post_detail, execute_reply_lutopia_post',
    code
)

# 增加新的神经通路
new_nodes = '''    elif tool_name == "read_post_detail":
        return execute_read_post_detail(post_id=arguments.get("post_id", ""))
    elif tool_name == "reply_lutopia_post":
        return execute_reply_lutopia_post(
            post_id=arguments.get("post_id", ""),
            content=arguments.get("content", "")
        )
'''
if "read_post_detail" not in code:
    target = 'limit=arguments.get("limit", 5)\n        )'
    code = code.replace(target, target + '\n' + new_nodes)

with open('/home/ubuntu/my_cyber_home/gateway.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("✅ 网关神经连接成功！")
