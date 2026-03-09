# 淘淘 AI 网关 - 操作手册

> 所有命令都可以直接复制粘贴到终端执行
> 把 `YOUR_TOKEN` 替换成你 .env 里的 GATEWAY_AUTH_TOKEN
> 服务器地址默认 `127.0.0.1:5000`，外网访问改成你的域名/IP

---

## 一、日常启停

```bash
# 启动（后台运行）
cd /opt/gateway && gunicorn -c gunicorn.conf.py gateway:app -D

# 停止
pkill -f "gunicorn.*gateway"

# 重启
pkill -f "gunicorn.*gateway" && sleep 2 && cd /opt/gateway && gunicorn -c gunicorn.conf.py gateway:app -D

# 看日志（实时）
tail -f /opt/gateway/nohup.out

# 只看记忆相关日志
tail -f /opt/gateway/nohup.out | grep '\[Memory\]'

# 只看向量相关日志
tail -f /opt/gateway/nohup.out | grep '\[Vector\]'
```

---

## 二、健康检查

```bash
# 检查网关是否在运行
curl http://127.0.0.1:5000/health

# 查看已配置的模型提供商
curl http://127.0.0.1:5000/v1/models \
  -H "Authorization: Bearer YOUR_TOKEN"

# 查看使用统计（消息数、对话数）
curl http://127.0.0.1:5000/admin/stats \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## 三、向量记忆管理

```bash
# 查看向量库状态（多少条已向量化、多少条待处理）
curl http://127.0.0.1:5000/admin/vectors/status \
  -H "Authorization: Bearer YOUR_TOKEN"

# 每日向量化（只处理新消息，增量）
# 建议放 cron 里每天凌晨自动跑
curl -X POST http://127.0.0.1:5000/admin/vectors/nightly \
  -H "Authorization: Bearer YOUR_TOKEN"

# 全量重建向量（删旧的全部重来，删了测试数据后用这个）
curl -X POST http://127.0.0.1:5000/admin/vectors/rebuild \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### cron 定时任务设置

```bash
# 编辑定时任务
crontab -e

# 粘贴下面这行（每天凌晨3点自动向量化）
0 3 * * * curl -s -X POST http://127.0.0.1:5000/admin/vectors/nightly -H "Authorization: Bearer YOUR_TOKEN" >> /opt/gateway/cron.log 2>&1
```

---

## 四、提供商管理

```bash
# 加了新提供商后，热加载（不用重启）
curl -X POST http://127.0.0.1:5000/admin/providers/reload \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 加新提供商的步骤

```bash
# 1. 编辑 providers.json，加一块
nano /opt/gateway/providers.json

# 2. 在 .env 里加 API key
echo 'MOONSHOT_API_KEY=sk-xxx' >> /opt/gateway/.env

# 3. 热加载（不重启）
curl -X POST http://127.0.0.1:5000/admin/providers/reload \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## 五、知识库（Notion）

```bash
# 刷新 Notion 缓存（改了 Notion 内容后执行）
curl -X POST http://127.0.0.1:5000/admin/notion/refresh \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## 六、数据库备份

```bash
# 手动备份一次
curl -X POST http://127.0.0.1:5000/admin/backup \
  -H "Authorization: Bearer YOUR_TOKEN"

# 看备份文件
ls -lh /opt/gateway/backups/
```

---

## 七、调试记忆搜索

```bash
# 测试一个问题的完整搜索链路
cd /opt/gateway && python test_search.py '大伯家的狗叫什么'

# 看完整内容（不截断）
python test_search.py '铁锅炖老公是什么' --full

# 模拟多轮对话
python test_search.py '不是这个' -c '你记得我之前说的那道菜吗' -c '东北菜？'

# 调整参数
python test_search.py '那本书' --exclude 20 --limit 10
```

---

## 八、更新代码

```bash
cd /opt/gateway

# 拉最新代码（system_prompt.txt 不会被覆盖，已加 .gitignore）
git pull

# 如果有新依赖
pip install -r requirements.txt

# 重启生效
pkill -f "gunicorn.*gateway" && sleep 2 && gunicorn -c gunicorn.conf.py gateway:app -D
```

---

## 九、编辑人设/记忆提示

```bash
# 编辑 system prompt（模型的人设和记忆指引）
nano /opt/gateway/system_prompt.txt

# 编辑后不用重启，下次请求自动生效
```

---

## 十、常见问题速查

| 症状 | 命令 |
|------|------|
| 模型回复乱码 | 重启网关，已修复 UTF-8 编码 |
| 搜索搜不到 | `python test_search.py '关键词' --full` 看链路 |
| 向量库是空的 | `curl -X POST .../admin/vectors/rebuild` |
| 加了新厂商没生效 | `curl -X POST .../admin/providers/reload` |
| Notion 改了没更新 | `curl -X POST .../admin/notion/refresh` |
| 看不懂日志 | `grep '\[Memory\]' nohup.out \| tail -20` |
| 磁盘快满了 | `ls -lh backups/` 然后删旧的 |

---

## 环境变量速查（.env）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `GATEWAY_AUTH_TOKEN` | 网关密码 | 必填 |
| `OPENROUTER_API_KEY` | OpenRouter 密钥 | - |
| `DEEPSEEK_API_KEY` | DeepSeek 密钥 | - |
| `ZHIPU_API_KEY` | 智谱密钥 | - |
| `ALIBABA_API_KEY` | 阿里百炼密钥（聊天+向量共用） | - |
| `EMBEDDING_DIMENSIONS` | 向量维度 | 768 |
| `VECTOR_MIN_SCORE` | 向量搜索最低分数 | 0.2 |
| `VECTOR_NEIGHBOR_WINDOW` | 上下文扩展窗口 | 1 |
| `VECTOR_EXCLUDE_HOURS` | 排除最近N小时的向量 | 2 |
| `VECTOR_CHUNK_ROUNDS` | 每块包含几轮对话 | 4 |
| `HISTORY_SEARCH_LIMIT` | 搜索返回条数 | 5 |
| `NOTION_TOKEN` | Notion API token | - |
| `NOTION_PAGE_IDS` | Notion 页面ID（逗号分隔） | - |
