# 更新说明

更新日期：2026-03-17

本文档用于记录这一轮网关项目的主要代码修改、已上线功能、后台使用方式、手机访问方式，以及后续需要记住的关键事项。

## 一、本轮完成的核心事项

这一轮主要完成了两大块工作：

1. 整理和增强网关后端的安全性、记忆系统和摘要体系。
2. 新增了一个可视化的 `/admin` 管理后台，让后续大部分常用操作可以直接在网页上完成。

另外，还把域名 `xiaoketao.cc` 的后台访问链路打通到了手机浏览器可访问的状态。

## 二、已修改的主要代码与功能

### 1. `gateway.py`

主要新增和调整了以下内容：

- 增加后台页面路由：
  - `/admin`
  - `/admin/ui`
- 增加管理后台页面渲染：
  - 返回 `templates/admin.html`
- 增加和完善后台接口能力：
  - `/admin/cards/list` 支持更适合后台使用的分页和完整字段读取
  - `/admin/vectors/status` 增加向量重建状态信息
  - `/admin/vectors/rebuild/status` 新增向量全量重建状态接口
- 增加向量重建任务状态管理：
  - `_vector_rebuild_state`
  - `_set_vector_rebuild_state()`
  - `_get_vector_rebuild_state()`
  - `_vector_rebuild_running()`
- 后台和运维相关的已有接口继续复用：
  - `/health`
  - `/admin/stats`
  - `/admin/cards/status`
  - `/admin/cards/search`
  - `/admin/cards/generate`
  - `/admin/cards/embed`
  - `/admin/cards/rebuild_full`
  - `/admin/cards/rebuild_full/status`
  - `/admin/vectors/rebuild`
  - `/admin/vectors/nightly`
  - `/admin/notion/test`
  - `/admin/notion/refresh`
  - `/admin/backup`
  - `/admin/providers/reload`

同时，这一轮之前已经完成过的后端改造也仍然生效，包括：

- 认证逻辑增强：
  - 远程请求在未配置 token 时会被阻止
  - 本地请求保留必要豁免
- `/health` 变为更简洁的健康检查输出
- 对话上下文结构调整为更清晰的三层结构：
  - system prompt
  - 滚动摘要
  - 最近原始对话
  - 长期记忆按需召回
- 清理和收敛了思维链重写相关输出，避免把内部思考结构直接污染回复
- 减少日志中的敏感信息泄露

### 2. `database.py`

已做的关键改动包括：

- 修正历史搜索多关键词匹配逻辑
- 修正 FTS 搜索与消息 ID 的关联方式
- 增加适合后台和全量重建使用的接口：
  - `get_all_message_dates()`
  - `get_all_cards_full()`
  - `export_memory_cards_backup()`
  - `replace_all_memory_cards()`
- 将旧的 `daily_summary` / `weekly_summary` 结构明确标为 legacy

### 3. `memory_cards.py`

已做的关键改动包括：

- 将 `memory_cards` 作为统一摘要主链路
- 增加从每日卡片派生每周摘要的逻辑
- 增加全量内存数据集生成和重建支持
- 增加向量文件备份与重建能力
- 增加全量记忆卡重建流程：
  - 备份
  - 重新生成
  - 重建卡片向量
  - 原子替换
  - 校验结果

### 4. `memory_tools.py`

已做的关键改动包括：

- 记忆搜索的时间衰减和最低相关度阈值改为环境变量可配
- 修复 `tags` 变量读取问题

### 5. `lutopia_tools.py`

已做的关键改动包括：

- 不再使用硬编码身份
- 改为读取环境变量 token
- 增加更清晰的失败返回和状态判断

### 6. `calendar_tools.py`、`notion_tools.py`

已做的关键改动包括：

- 日志进一步脱敏，减少敏感内容直接出现在日志中

### 7. `test_gateway_regressions.py`

新增了回归测试文件，用来覆盖这一轮比较关键的逻辑，包括：

- 远程认证校验
- 搜索逻辑修正
- 记忆卡构建
- 记忆检索格式
- 上下文组装
- inner monologue 清理
- 周摘要派生
- 摘要架构状态输出

## 三、新增的可视化后台

### 1. 新增文件

- `templates/admin.html`
- `static/admin.css`
- `static/admin.js`

### 2. 后台定位

这是一个直接内嵌在现有 Flask 网关里的后台页面，不需要单独前端工程，不需要再额外起一个管理站点。

### 3. 当前后台入口

- 本地入口：
  - `http://127.0.0.1:5000/admin`
  - `http://127.0.0.1:5000/admin/ui`
- 域名入口：
  - `https://xiaoketao.cc/admin`
  - `https://xiaoketao.cc/admin/ui`

### 4. 后台当前已有功能

#### 首页总览

- 查看网关健康状态
- 查看消息总数、对话总数、今日消息数
- 查看当前摘要架构状态
- 查看记忆卡数量、向量数量
- 查看模型提供商概览
- 查看长任务状态

#### 记忆卡管理

- 查看最近记忆卡
- 按日期筛选
- 关键词搜索记忆卡
- 触发单日生成
- 触发区间生成
- 补做记忆卡向量
- 全量重建记忆卡
- 查看记忆卡状态和分页列表

#### 向量与诊断

- 查看消息向量状态
- 执行 nightly 向量补量
- 全量重建向量
- 刷新 Notion 缓存
- 检查 Notion 连通性
- 手动数据库备份
- 查看诊断返回结果

#### 任务中心

- 查看记忆卡全量重建状态
- 查看向量重建状态
- 自动轮询进度

### 5. 后台交互风格

这版后台是按“非技术用户也能看懂”的方向写的，具体做法包括：

- 用人话说明，不要求理解代码
- 第一屏先填“管理员口令”
- 没填口令时只提示下一步，不抛一堆报错
- 危险操作使用二次确认弹窗
- 页面自动轮询长任务状态
- 结果和报错尽量可直接读懂

## 四、手机访问和线上访问链路

### 1. 当前线上访问方式

手机或电脑浏览器可以直接访问：

- `https://xiaoketao.cc/admin`

兼容入口：

- `https://xiaoketao.cc/admin/ui`

### 2. 当前链路结构

- 域名：`xiaoketao.cc`
- Cloudflare：已可作为 HTTPS 入口使用
- Nginx：负责 80/443 对外入口和反向代理
- Gunicorn：运行 Flask 网关
- Flask：提供 `/admin` 页面和 `/admin/*` JSON 接口

### 3. 当前内部端口

网关进程当前运行端口：

- `5000`

说明：

- Gunicorn 实际监听的是 `0.0.0.0:5000`
- Nginx 对外提供域名和 HTTPS
- 外部不需要直接访问 `5000`

## 五、这轮做过的线上运维调整

除了代码文件改动，这一轮还做了几项已经生效的线上操作：

1. 对运行中的 Gunicorn 做了平滑重载，让最新的 `/admin` 页面路由正式生效。
2. 清理了 Nginx 中重复启用的站点配置，消除了重复 `server_name` 的告警。
3. 确认了 `xiaoketao.cc` 的 Let’s Encrypt 证书有效。
4. 验证了以下链接返回正常：
   - `https://xiaoketao.cc/admin`
   - `https://xiaoketao.cc/admin/ui`

## 六、你需要记住的关键信息

### 1. 管理员口令

后台页面需要填写管理员口令才能调用后台接口。

注意：

- 这个口令是网关环境变量 `GATEWAY_AUTH_TOKEN` 的值
- 出于安全原因，本日志文件里不明文写出口令
- 如果以后忘记了，应去当前网关实际使用的环境配置中查 `GATEWAY_AUTH_TOKEN`
- 不建议把口令明文写进 Git、聊天记录或公开文档

### 2. 浏览器本地保存位置

后台页面会把口令保存在浏览器本地存储里，键名是：

- `gateway_admin_token`

这表示：

- 同一个浏览器下，填过一次通常就不用反复输入
- 如果点了“清空口令”，浏览器里保存的值会被删除

### 3. Cloudflare 建议设置

当前推荐保持：

- DNS 记录使用橙云 `Proxied`
- `SSL/TLS` 模式使用 `Full (strict)`
- 不要使用 `Flexible`

### 4. 你日常最常用的后台链接

- `https://xiaoketao.cc/admin`

## 七、建议你后续如何使用

以后如果只是日常管理，优先按下面顺序操作：

1. 打开 `https://xiaoketao.cc/admin`
2. 输入管理员口令
3. 在首页先看状态是否正常
4. 需要检查记忆时进入“记忆卡管理”
5. 需要修复检索时进入“向量与诊断”
6. 做全量重建前，先确认操作范围和时间

## 八、关于“还需要做什么保存吗”

结论：

- 这次我创建和修改的文件已经写到磁盘里了
- 如果你现在直接关闭窗口，文件内容本身不会消失
- 线上已经生效的 Nginx / Gunicorn 调整也不会因为你关掉 Cursor 窗口就恢复

也就是说：

- 你现在不需要额外点什么“保存”
- 直接关闭窗口也可以

但是有两点要知道：

1. 这些改动目前还是工作区里的未提交改动，不是 Git 提交
2. 如果以后你想长期留档、回滚或迁移到别的机器，最好再做一次 Git 提交或额外备份

## 九、当前最重要的结论

目前你已经可以不用 SSH 和命令行，直接通过网页进入后台处理很多常用管理操作。

最重要的入口就是：

- `https://xiaoketao.cc/admin`

最重要需要记住的不是明文密码本身，而是：

- 后台口令来自 `GATEWAY_AUTH_TOKEN`
- 后台口令会临时保存在浏览器本地的 `gateway_admin_token`
- Cloudflare 推荐保持 `Full (strict)`

