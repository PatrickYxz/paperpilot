# PaperPilot Web 工作台开发记录与 0 基础讲解

日期：2026-06-05

## 这份文档解决什么问题

今天我们把 PaperPilot 从“命令行研究 Agent 原型”往“本地 Web 研究工作台”推进了一大步。

这份文档分成两部分：

1. 今天实际做了什么。
2. 这些改动背后的后端、前端、数据库、API、异步任务和测试知识，按 0 基础方式解释。

读完后，你应该能回答：

- 为什么要先做 Web 工作台骨架，而不是直接接完整 deep-read？
- FastAPI 在这里负责什么？
- SQLite 表为什么这样设计？
- 前端页面怎么和后端 API 通信？
- workflow、event、artifact 分别是什么意思？
- simulated 和 real PaperPilot 执行有什么区别？
- 后面要接真实 deep-read，为什么现在这些基础是必要的？

---

# 第一部分：今天实际做了什么

## 1. 先审计当前工作区

正式做 Web 工作台之前，我们先检查了当前仓库状态。

原因是 PaperPilot 已经有不少未提交改动：

- 多轮会话相关代码；
- session 存储；
- 用户文档存储；
- bulk input 检测；
- context 自动压缩；
- ask_user 工具；
- agent loop 相关测试。

如果直接开做 Web，很容易重复实现，或者不小心覆盖已有工作。

审计结果：

- 当前已有改动大多是“CLI 多轮会话增强层”。
- 它们不是 Web 工作台本身。
- 但其中一些能力可以在后续 Web 版本里复用。

可复用能力包括：

- `ConversationSession`：长会话边界。
- `SessionStore`：命名会话持久化。
- `DocumentStore`：用户粘贴长文档存储。
- `BulkPaperInputDetector`：检测超长论文输入，避免直接塞进模型上下文。
- `ContextManager`：上下文窗口预检查。
- `compact_messages`：自动压缩旧消息。

审计相关计划文档：

- `docs/codex-only-plans/2026-06-05-paperpilot-audit-and-web-mvp-plan.md`

---

## 2. A1：做 Web 任务工作台骨架

A1 的目标是先让 PaperPilot 有一个最小 Web 产品壳。

这一版还不运行真实 deep-read。

它只解决最基础的问题：

```text
用户输入研究问题
-> 后端创建任务
-> 任务写入 SQLite
-> 页面显示任务列表
-> 刷新页面后任务仍然存在
```

新增后端模块：

- `paperpilot/web/__init__.py`
- `paperpilot/web/task_store.py`
- `paperpilot/web/app.py`

新增前端静态资源：

- `paperpilot/web/static/index.html`
- `paperpilot/web/static/styles.css`
- `paperpilot/web/static/app.js`

新增测试：

- `tests/web/test_task_store.py`
- `tests/web/test_web_app.py`

新增依赖：

- `fastapi`
- `uvicorn`

A1 提供的 API：

```text
POST /api/tasks
GET /api/tasks
GET /api/tasks/{task_id}
```

A1 的 SQLite 表：

```sql
CREATE TABLE research_tasks (
  id TEXT PRIMARY KEY,
  question TEXT NOT NULL,
  depth TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
```

任务字段解释：

- `id`：任务唯一编号。
- `question`：用户输入的研究问题。
- `depth`：研究深度，当前有 `quick`、`standard`、`deep`。
- `status`：任务状态，比如 `pending`、`running`、`completed`、`failed`。
- `created_at`：创建时间。
- `updated_at`：最后更新时间。

A1 计划文档：

- `docs/codex-only-plans/2026-06-05-paperpilot-web-task-mvp-a1-plan.md`

---

## 3. A2：加入模拟 workflow 和事件日志

A1 只能创建任务，但任务不会“动”。

A2 加入了模拟 workflow：

```text
pending -> running -> completed
```

同时记录事件：

```text
queued -> started -> progress -> progress -> completed
```

这一步的重点不是做真实研究，而是建立“任务进度可观察”的结构。

新增 SQLite 表：

```sql
CREATE TABLE task_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  type TEXT NOT NULL,
  message TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES research_tasks(id)
);
```

事件字段解释：

- `id`：事件编号。
- `task_id`：这个事件属于哪个任务。
- `type`：事件类型，比如 `queued`、`started`、`progress`、`completed`、`failed`。
- `message`：给人看的事件说明。
- `created_at`：事件发生时间。

新增 API：

```text
GET /api/tasks/{task_id}/events
```

前端详情页开始展示事件历史，并对运行中的任务轮询刷新。

A2 计划文档：

- `docs/codex-only-plans/2026-06-05-paperpilot-web-workflow-a2-plan.md`

---

## 4. A2.5：让事件日志支持结构化信息

A2 的事件只有 `type` 和 `message`，对真实 deep-read 来说不够。

真实任务里我们会想记录：

- 当前是搜索阶段，还是 deep-read 阶段？
- 是模拟执行，还是真实执行？
- 当前 step 是第几步？
- 后面是否和某个 paper_id、tool_call、trace 相关？

所以 A2.5 给事件表加了两个字段：

```sql
stage TEXT
payload_json TEXT
```

目标表结构变成：

```sql
CREATE TABLE task_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  type TEXT NOT NULL,
  stage TEXT,
  message TEXT NOT NULL,
  payload_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES research_tasks(id)
);
```

字段解释：

- `stage`：更具体的 workflow 阶段，比如 `queue`、`start`、`prepare`。
- `payload_json`：机器可读的小型 JSON 数据。

例如：

```json
{
  "type": "progress",
  "stage": "prepare",
  "message": "Preparing research workflow state.",
  "payload": {
    "simulated": true,
    "step": 2
  }
}
```

为什么要有 `payload_json`？

因为 `message` 是给人看的，不能依赖它做程序逻辑。

例如：

```text
Preparing research workflow state.
```

这句话适合显示在页面上。

但如果程序要知道“这是第几步”，就应该看结构化字段：

```json
{
  "step": 2
}
```

A2.5 还做了旧库兼容。

如果本地已经有 A2 的旧数据库，启动时会自动执行：

```sql
ALTER TABLE task_events ADD COLUMN stage TEXT;
ALTER TABLE task_events ADD COLUMN payload_json TEXT;
```

这样不会删除旧数据。

A2.5 计划文档：

- `docs/codex-only-plans/2026-06-05-paperpilot-web-event-structure-a25-plan.md`

---

## 5. A3.0：抽出 WorkflowRunner，并加入 artifact 结果存储

A2/A2.5 的 workflow 逻辑一开始写在 FastAPI 的 `app.py` 里。

这对原型可以，但不适合继续扩展。

原因是 `app.py` 应该主要负责 HTTP：

- 接收请求；
- 校验参数；
- 返回响应；
- 调度任务。

真正的任务执行逻辑应该放到单独模块里。

所以 A3.0 新增：

- `paperpilot/web/workflow.py`

里面有：

```python
class WorkflowRunner:
    def run_simulated(self, task_id: str) -> None:
        ...
```

这样后续真实 deep-read 接入时，修改点会集中在 runner，不会把 FastAPI 路由文件写乱。

A3.0 还新增了 artifact 存储。

为什么需要 artifact？

事件日志适合记录过程：

```text
任务开始了
进入 prepare 阶段
deep-read placeholder 跳过了
任务完成了
```

但最终结果不应该只存在事件里。

最终结果应该是一个 artifact，比如：

- final answer；
- Markdown report；
- evidence summary；
- trace summary；
- candidate papers。

A3.0 新增表：

```sql
CREATE TABLE task_artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  payload_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES research_tasks(id)
);
```

artifact 字段解释：

- `id`：artifact 编号。
- `task_id`：属于哪个任务。
- `kind`：artifact 类型，比如 `result`、`report`、`evidence`。
- `title`：标题。
- `content`：正文内容，可以是普通文本或 Markdown。
- `payload_json`：小型结构化 metadata。
- `created_at`：创建时间。

A3.0 新增 API：

```text
GET /api/tasks/{task_id}/artifacts
```

页面详情页开始展示 artifact。

A3.0 计划文档：

- `docs/codex-only-plans/2026-06-05-paperpilot-web-runner-artifacts-a30-plan.md`

---

## 6. A3.1：加入真实 PaperPilot 执行模式

A3.1 是今天最接近真实能力的一步。

它没有默认把所有任务都改成真实执行。

而是新增一个显式选择：

```text
Execution: Simulated
Execution: Real PaperPilot
```

默认仍是：

```text
Simulated
```

原因是 Real PaperPilot 会触发：

- 模型调用；
- MCP server 启动；
- arXiv / ColBERT / graph / VLM 等工具链；
- `.env` 和 API key；
- 更长运行时间；
- 可能的网络或环境问题。

所以必须让用户明确选择。

A3.1 修改了创建任务请求：

```json
{
  "question": "What are the main retrieval methods for long-context RAG?",
  "depth": "standard",
  "execution_mode": "real"
}
```

`execution_mode` 有两个值：

```text
simulated
real
```

后端逻辑：

```text
execution_mode = simulated
-> WorkflowRunner.run_simulated(task_id)

execution_mode = real
-> WorkflowRunner.run_real(task_id)
```

真实执行入口：

```python
def _default_real_runner(query: str) -> list[dict]:
    from paperpilot.conversation import run

    return run(query)
```

也就是说，A3.1 的 Real PaperPilot 最小版本调用的是现有的一次性 PaperPilot runner：

```python
paperpilot.conversation.run(query)
```

然后从返回 messages 里提取最终 assistant 文本，保存成 result artifact。

真实模式成功后会写入：

事件：

```text
queued / queue
started / real_start
completed / real_complete
```

artifact：

```json
{
  "kind": "result",
  "title": "PaperPilot result",
  "content": "<final assistant text>",
  "payload": {
    "execution_mode": "real",
    "source": "WorkflowRunner.run_real"
  }
}
```

真实模式失败时：

```text
status = failed
event type = failed
stage = failure
message = Real PaperPilot execution failed: ...
```

A3.1 计划文档：

- `docs/codex-only-plans/2026-06-05-paperpilot-web-real-runner-a31-plan.md`

---

## 7. 今天的验证情况

我们多次运行了测试。

最终 A3.1 后的验证结果：

Web focused tests：

```text
24 passed, 1 warning
```

Web + agent/session 组合测试：

```text
71 passed, 1 deselected, 1 warning
```

排除已知 SOCKS 代理环境问题后的项目测试：

```text
194 passed, 10 deselected, 1 warning
```

有一个已知环境问题：

```text
tests/mcp_servers/test_ss_client.py
```

如果运行默认全量测试，它可能因为当前系统里有 SOCKS 代理环境变量，但 venv 缺少 `socksio` 而失败。

这是既有环境问题，不是今天 Web 工作台改动导致。

---

## 8. 当前服务状态

本地服务地址：

```text
http://127.0.0.1:8000/
```

当前运行的是 A3.1 版本。

最近启动的服务进程：

```text
python PID 27348
```

启动命令等价于：

```powershell
.venv\Scripts\python.exe -m uvicorn paperpilot.web.app:app --host 127.0.0.1 --port 8000
```

---

# 第二部分：0 基础知识讲解

下面开始从 0 基础解释今天涉及到的后端和前端知识。

---

# 一、什么是 Web 应用

一个 Web 应用通常由两部分组成：

```text
前端
后端
```

前端就是你在浏览器里看到和点击的东西。

例如：

- 输入框；
- 按钮；
- 下拉菜单；
- 任务列表；
- 任务详情；
- 事件日志；
- 结果卡片。

后端就是运行在本机或服务器上的程序。

它负责：

- 接收前端请求；
- 保存数据；
- 查询数据；
- 调用模型或工具；
- 返回结果。

在今天的 PaperPilot Web 工作台里：

```text
浏览器页面 = 前端
FastAPI 程序 = 后端
SQLite 文件 = 数据库
WorkflowRunner = 后台任务执行逻辑
```

整体关系：

```text
用户
  -> 浏览器页面
  -> JavaScript 发 HTTP 请求
  -> FastAPI 接收请求
  -> TaskStore 读写 SQLite
  -> WorkflowRunner 推进任务
  -> FastAPI 返回 JSON
  -> JavaScript 更新页面
```

---

# 二、什么是 HTTP

浏览器和后端之间主要靠 HTTP 通信。

HTTP 可以理解成一种“请求-响应”的规则。

例如你打开：

```text
http://127.0.0.1:8000/
```

浏览器会向后端发请求：

```text
GET /
```

后端返回：

```text
index.html
```

页面加载后，JavaScript 又会请求：

```text
GET /api/tasks
```

后端返回：

```json
[
  {
    "id": "task_...",
    "question": "...",
    "depth": "standard",
    "status": "completed"
  }
]
```

## 常见 HTTP 方法

今天用到两个：

```text
GET
POST
```

`GET` 表示读取数据。

例如：

```text
GET /api/tasks
```

意思是：

```text
请给我任务列表。
```

`POST` 表示创建数据或提交操作。

例如：

```text
POST /api/tasks
```

意思是：

```text
我要创建一个新任务。
```

---

# 三、什么是 API

API 可以理解成“前端和后端约定好的接口”。

前端不能直接打开 Python 变量，也不能直接读 SQLite 文件。

前端只能通过 HTTP 请求问后端：

```text
我要创建任务。
我要任务列表。
我要某个任务详情。
我要这个任务的事件。
我要这个任务的结果。
```

后端提供这些入口。

今天 PaperPilot 的 API 有：

```text
POST /api/tasks
GET /api/tasks
GET /api/tasks/{task_id}
GET /api/tasks/{task_id}/events
GET /api/tasks/{task_id}/artifacts
```

`{task_id}` 是占位符。

例如真实请求可能是：

```text
GET /api/tasks/task_d7945841b1554fbab44f3a80bb786875
```

---

# 四、什么是 JSON

JSON 是前后端之间常用的数据格式。

它长这样：

```json
{
  "question": "What are the main retrieval methods for long-context RAG?",
  "depth": "standard",
  "execution_mode": "simulated"
}
```

它像 Python 的 dict，但它是文本格式。

前端发 JSON 给后端。

后端也返回 JSON 给前端。

例如创建任务时，前端发：

```json
{
  "question": "Compare ColBERT and BM25",
  "depth": "standard",
  "execution_mode": "simulated"
}
```

后端返回：

```json
{
  "id": "task_abc",
  "question": "Compare ColBERT and BM25",
  "depth": "standard",
  "status": "pending",
  "created_at": "2026-06-05T...",
  "updated_at": "2026-06-05T..."
}
```

---

# 五、什么是 FastAPI

FastAPI 是一个 Python Web 框架。

框架就是别人提前写好的基础设施。

如果不用框架，你要自己处理：

- HTTP 请求怎么解析；
- JSON 怎么读取；
- 路由怎么匹配；
- 参数怎么校验；
- 错误怎么返回；
- 静态文件怎么服务。

FastAPI 帮我们处理这些。

今天的核心文件：

```text
paperpilot/web/app.py
```

里面有：

```python
app = FastAPI(title="PaperPilot Web Workbench")
```

这表示创建一个 Web 应用。

路由例子：

```python
@app.get("/api/tasks")
def list_tasks(...):
    ...
```

意思是：

```text
当浏览器请求 GET /api/tasks 时，执行 list_tasks 这个 Python 函数。
```

另一个例子：

```python
@app.post("/api/tasks")
def create_task(...):
    ...
```

意思是：

```text
当前端请求 POST /api/tasks 时，执行 create_task。
```

---

# 六、什么是 Uvicorn

FastAPI 定义了应用逻辑，但还需要一个服务器把它跑起来。

Uvicorn 就是这个服务器。

启动命令：

```powershell
.venv\Scripts\python.exe -m uvicorn paperpilot.web.app:app --host 127.0.0.1 --port 8000
```

拆开看：

```text
.venv\Scripts\python.exe
```

表示使用项目虚拟环境里的 Python。

```text
-m uvicorn
```

表示运行 uvicorn。

```text
paperpilot.web.app:app
```

表示找到：

```text
paperpilot/web/app.py
```

里面的：

```python
app = create_app()
```

```text
--host 127.0.0.1
```

表示只在本机访问。

```text
--port 8000
```

表示端口是 8000。

所以浏览器打开：

```text
http://127.0.0.1:8000/
```

就能访问它。

---

# 七、什么是 SQLite

SQLite 是一个轻量数据库。

它不是一个单独运行的数据库服务器。

它就是一个文件。

今天默认数据库路径：

```text
data/web/tasks.sqlite3
```

为什么用 SQLite？

因为 PaperPilot 当前第一阶段目标是：

```text
本地单用户研究工作台
```

这时 SQLite 很合适：

- 简单；
- 不需要安装 PostgreSQL；
- 不需要启动数据库服务；
- 一个文件就能保存任务；
- 适合本地原型；
- 可以测试持久化。

为什么暂时不用 PostgreSQL？

PostgreSQL 更适合：

- 多用户；
- 生产部署；
- 高并发；
- 权限管理；
- 更复杂查询。

但这些不是当前第一阶段目标。

---

# 八、什么是表、行、字段

数据库里最基础的概念是表。

表像 Excel 表格。

例如 `research_tasks`：

```text
id | question | depth | status | created_at | updated_at
```

每一行是一条任务。

例如：

```text
task_123 | Compare ColBERT and BM25 | standard | completed | ... | ...
```

字段就是列名。

例如：

- `id`
- `question`
- `status`

今天我们有三张核心表：

```text
research_tasks
task_events
task_artifacts
```

它们的关系：

```text
一个 research_task
  -> 可以有多个 task_events
  -> 可以有多个 task_artifacts
```

例如：

```text
任务：Compare ColBERT and BM25

事件：
- queued
- started
- progress
- completed

结果：
- Simulated research result
```

---

# 九、为什么要分 task、event、artifact

这是今天最重要的设计之一。

## task 是任务本体

task 记录“用户想做什么”。

例如：

```text
问题：Compare ColBERT and BM25
深度：standard
状态：completed
```

它是任务的主记录。

## event 是过程日志

event 记录“任务执行过程中发生了什么”。

例如：

```text
queued
started
progress
completed
```

它适合展示进度。

它不适合保存最终报告。

## artifact 是产物

artifact 记录“任务最后生成了什么东西”。

例如：

```text
最终回答
Markdown 报告
证据摘要
trace summary
候选论文列表
```

event 和 artifact 的区别：

```text
event = 过程
artifact = 结果
```

如果只用 event，会出现问题：

```text
completed: 这篇论文主要提出了...
```

这样最终结果混在日志里，不利于后续查询和展示。

所以我们单独建 `task_artifacts`。

---

# 十、什么是 workflow

workflow 是工作流。

它表示任务要按哪些阶段执行。

比如一个真实 PaperPilot 研究任务未来可能是：

```text
创建任务
-> 生成搜索 query
-> 搜索候选论文
-> 用户确认论文
-> 下载论文
-> 建 ColBERT index
-> 并发 deep-read
-> 收集证据
-> 生成报告
-> 保存结果
```

这就是 workflow。

为什么不用“让 Agent 自己随便跑”？

因为随便跑很难控制：

- 当前跑到哪一步了？
- 哪一步失败了？
- 能不能重试？
- 用户能不能看到进度？
- 最终结果有没有保存？
- 是否能复盘 trace？

workflow 的价值是让 Agent 系统更工程化。

PaperPilot 后续的定位不是简单聊天机器人，而是：

```text
本地研究工作台
```

所以需要 workflow。

---

# 十一、什么是 WorkflowRunner

今天我们新增了：

```text
paperpilot/web/workflow.py
```

里面有：

```python
class WorkflowRunner:
    ...
```

它负责：

- 更新任务状态；
- 写 task_events；
- 写 task_artifacts；
- 调用 simulated 或 real 执行逻辑。

为什么不直接写在 `app.py`？

因为 `app.py` 是 HTTP 层。

它应该关注：

```text
请求来了
参数是否合法
调用哪个服务
返回什么响应
```

它不应该塞满：

```text
怎么执行 deep-read
怎么保存 artifact
怎么处理失败
怎么写事件
```

这些应该放在 runner 里。

这种拆分叫做“职责分离”。

简单理解：

```text
app.py = 前台接待
workflow.py = 后台办事员
task_store.py = 档案管理员
SQLite = 档案柜
```

---

# 十二、什么是后台任务

用户点击 Create Task 后，后端不应该一直卡住浏览器等待任务完成。

尤其真实 PaperPilot 可能很慢。

所以后端做的是：

```text
收到创建任务请求
-> 立刻创建任务
-> 立刻返回 task_id
-> 后台继续执行 workflow
```

在 FastAPI 里，我们用了：

```python
BackgroundTasks
```

它允许请求返回后继续做一些工作。

当前是本地原型，所以这样够用。

以后如果进入更复杂阶段，可能要考虑：

- Celery；
- Redis Queue；
- Dramatiq；
- APScheduler；
- 自己的本地任务线程池。

但现在不需要。

原因是当前目标是本地单用户工作台。

---

# 十三、什么是 simulated execution

Simulated execution 是模拟执行。

它不调用真实模型，也不启动真实 MCP。

它只是做几件事：

```text
等待一下
写 started 事件
写 progress 事件
写 result artifact
改状态 completed
```

为什么需要模拟执行？

因为 UI、API、数据库、workflow 状态都可以先测试。

如果一开始就接真实模型，会有很多干扰：

- API key 是否配置；
- 网络是否通；
- MCP server 是否启动；
- arXiv 是否下载成功；
- ColBERT 是否可用；
- 模型是否超时；
- 成本是否可控。

这些会让你分不清：

```text
到底是 Web 工作台有 bug，
还是模型/MCP/网络出了问题？
```

所以先用模拟执行把产品骨架跑通。

---

# 十四、什么是 real execution

Real execution 是真实执行。

A3.1 里，它调用：

```python
paperpilot.conversation.run(query)
```

这会走现有 PaperPilot 逻辑。

大致路径是：

```text
用户问题
-> PaperPilot conversation runner
-> agent loop
-> LLM
-> tools / MCP
-> final assistant answer
-> Web 保存成 artifact
```

A3.1 的真实模式还很小。

它只做：

```text
把最终 assistant 文本保存下来
```

它还没有做：

- 候选论文确认；
- 每篇论文状态；
- evidence 表；
- report 表；
- trace summary；
- 工具调用全过程可视化。

这些是下一步。

---

# 十五、什么是前端

前端就是浏览器里的页面。

今天前端文件有三个：

```text
paperpilot/web/static/index.html
paperpilot/web/static/styles.css
paperpilot/web/static/app.js
```

## HTML 是结构

HTML 决定页面上有什么。

例如：

```html
<textarea id="question"></textarea>
<select id="depth"></select>
<button>Create Task</button>
```

意思是页面上有：

- 一个输入框；
- 一个下拉菜单；
- 一个按钮。

## CSS 是样式

CSS 决定页面长什么样。

例如：

```css
.panel {
  background: #ffffff;
  border: 1px solid #d9dee7;
  border-radius: 8px;
}
```

意思是：

- 面板背景是白色；
- 有边框；
- 圆角是 8px。

## JavaScript 是行为

JavaScript 决定页面怎么动。

例如：

```javascript
taskForm.addEventListener("submit", async (event) => {
  ...
});
```

意思是：

```text
当用户提交表单时，执行这里的代码。
```

---

# 十六、前端怎么调用后端

前端用 `fetch()` 调后端 API。

例如创建任务：

```javascript
const task = await requestJson("/api/tasks", {
  method: "POST",
  body: JSON.stringify({
    question,
    depth: depthInput.value,
    execution_mode: executionModeInput.value,
  }),
});
```

这段代码做了几件事：

1. 从页面输入框拿到 `question`。
2. 从 depth 下拉框拿到 `depth`。
3. 从 execution mode 下拉框拿到 `execution_mode`。
4. 转成 JSON。
5. 发送给后端 `/api/tasks`。
6. 等后端返回新任务。

后端收到后创建任务。

然后前端再调用：

```text
GET /api/tasks/{task_id}
GET /api/tasks/{task_id}/events
GET /api/tasks/{task_id}/artifacts
```

用这些数据更新详情页。

---

# 十七、什么是轮询

任务运行不是瞬间完成的。

前端需要不断问后端：

```text
这个任务现在怎么样了？
有没有新事件？
有没有结果？
```

这种定时请求叫轮询。

当前页面逻辑：

```text
如果任务是 pending 或 running
-> 1 秒后再次请求任务详情
```

这样页面不用手动刷新，也能看到状态变化。

未来也可以用 WebSocket 或 Server-Sent Events。

但第一版用轮询最简单。

---

# 十八、什么是测试

测试就是用代码验证代码。

今天我们写了 Web 相关测试：

```text
tests/web/test_task_store.py
tests/web/test_web_app.py
tests/web/test_workflow.py
```

## store 测试

验证 SQLite 存储是否正确。

例如：

- 创建任务后能查到；
- 任务列表排序正确；
- 事件能保存和读取；
- artifact 能保存和读取；
- 旧版事件表能迁移。

## API 测试

验证 HTTP 接口是否正确。

例如：

- `POST /api/tasks` 返回 201；
- invalid payload 返回 422；
- missing task 返回 404；
- events endpoint 返回事件；
- artifacts endpoint 返回结果。

## workflow 测试

验证 runner 是否正确。

例如：

- simulated runner 会写事件和 result artifact；
- real runner 会调用注入的 fake runner；
- real runner 失败时任务变成 failed。

为什么测试里不用真实模型？

因为测试应该稳定、快、便宜。

真实模型会受到：

- API key；
- 网络；
- 价格；
- 速率限制；
- MCP server；
- 外部服务。

所以测试用 fake runner。

---

# 十九、什么是依赖

依赖就是项目需要安装的第三方库。

今天新增了：

```text
fastapi
uvicorn
```

写在：

```text
requirements.txt
```

别人拿到项目后，可以运行：

```powershell
.venv\Scripts\pip.exe install -r requirements.txt
```

安装项目需要的库。

---

# 二十、为什么不直接上 React

今天前端只用了：

```text
HTML + CSS + JavaScript
```

没有用 React/Vue。

原因：

当前目标不是做复杂商业前端，而是：

```text
本地单用户研究工作台 MVP
```

用原生前端的好处：

- 文件少；
- 依赖少；
- 构建简单；
- 直接由 FastAPI 服务静态文件；
- 适合学习 HTTP、DOM、API 基础。

什么时候再考虑 React？

当页面复杂到需要：

- 多级状态管理；
- 复杂组件复用；
- 大量交互；
- 多页面路由；
- 大型 UI 组件系统。

现在还不到。

---

# 二十一、为什么不直接上 Celery/Redis

真实生产系统常用 Celery + Redis 做后台任务。

但当前阶段没有用。

原因：

当前是：

```text
本地单用户原型
```

FastAPI `BackgroundTasks` 已经够用。

如果现在引入 Celery/Redis，会增加：

- Redis 安装；
- worker 启动；
- 队列配置；
- 任务序列化；
- 错误排查；
- 部署复杂度。

这些不服务于当前第一目标。

当前更重要的是先把：

```text
任务
状态
事件
结果
页面展示
```

跑通。

---

# 二十二、为什么不直接把 deep-read 接完整

完整 deep-read 不只是一个函数调用。

它涉及：

- 搜索候选论文；
- 下载论文；
- 建 ColBERT index；
- 多次检索；
- 子 agent 并发读论文；
- 证据保存；
- 失败处理；
- 用户确认候选论文；
- 报告生成；
- trace 复盘；
- 成本和延迟控制。

如果一开始全接进来，bug 会很难定位。

所以今天按阶段推进：

```text
A1 任务持久化
A2 事件和状态
A2.5 结构化事件
A3.0 artifact 结果存储
A3.1 真实 runner 最小接入
```

这是把复杂系统拆小。

每一步都能测试。

每一步都有明确价值。

---

# 二十三、现在 PaperPilot Web 的整体架构

当前架构可以这样理解：

```text
Browser
  |
  | HTTP
  v
FastAPI app.py
  |
  | creates task / schedules runner
  v
WorkflowRunner workflow.py
  |
  | status / events / artifacts
  v
TaskStore task_store.py
  |
  | SQL
  v
SQLite data/web/tasks.sqlite3
```

真实模式时，runner 还会调用：

```text
WorkflowRunner.run_real
  -> paperpilot.conversation.run
  -> agent loop
  -> LLM + tools + MCP
  -> final assistant answer
  -> task_artifacts
```

---

# 二十四、现在还缺什么

当前只是第一阶段 Web 化。

还缺：

## 1. 真实运行的事件映射

现在真实模式只保存最终回答。

后面应该把 `on_event` 映射到 Web event log：

- tool_call；
- tool_result；
- guardrail_stop；
- context_preflight；
- auto_compact；
- errors。

这样页面上能看到真实 PaperPilot 到底调用了什么工具。

## 2. 候选论文确认

完整研究 workflow 应该是：

```text
搜索候选论文
-> 用户确认范围
-> 再 deep-read
```

现在还没有这个 UI。

## 3. 证据表

最终报告需要可追溯证据。

后面可能需要：

```text
task_evidence
```

字段可能包括：

- task_id；
- paper_id；
- chunk_id；
- quote；
- score；
- source_tool；
- created_at。

## 4. 报告表或 report artifact

现在 result artifact 是普通文本。

后面可以扩展：

```text
kind = report
content = Markdown report
payload = report metadata
```

## 5. retry / resume

当前失败只能显示 failed。

以后可能需要：

- retry task；
- retry failed stage；
- resume from last successful stage。

这些都依赖今天做的任务状态和事件结构。

---

# 二十五、今天最重要的工程取舍

## 取舍 1：先做任务壳，再接 Agent

优点：

- 产品形态先立住；
- 数据模型清楚；
- 后续真实执行有落点；
- 容易测试。

缺点：

- 一开始看不到完整 deep-read。

这是值得的，因为它降低了复杂度。

## 取舍 2：用 SQLite，不用 PostgreSQL

优点：

- 本地简单；
- 不需要额外服务；
- 适合单用户原型。

缺点：

- 不适合未来多用户生产部署。

当前阶段 SQLite 更合适。

## 取舍 3：用原生 HTML/JS，不用 React

优点：

- 学习成本低；
- 文件少；
- 没有构建系统；
- 直接理解浏览器和 API 的关系。

缺点：

- 页面复杂后会难维护。

当前阶段原生前端更合适。

## 取舍 4：默认 simulated，real 需要手动选择

优点：

- 不误触发真实模型成本；
- UI 测试稳定；
- 开发时不依赖 API key。

缺点：

- 用户需要手动切换真实模式。

这符合本地研究工具的安全边界。

---

# 二十六、你现在可以怎么演示

打开：

```text
http://127.0.0.1:8000/
```

演示流程：

1. 输入一个研究问题。
2. 选择 depth。
3. 保持 Execution 为 Simulated。
4. 点击 Create Task。
5. 观察任务进入列表。
6. 点击任务详情。
7. 观察 status 从 pending/running 到 completed。
8. 观察 events。
9. 观察 artifact result。
10. 说明 Real PaperPilot 模式已经接入，但需要 API key 和真实工具链，因此不默认触发。

可以用这句话解释项目进展：

```text
我先把 PaperPilot 从 CLI 原型封装成本地 Web 研究工作台。现在已经有持久化任务、状态机事件日志、结构化事件 payload、结果 artifact 存储，以及一个显式选择的真实 PaperPilot runner。下一步会把真实 agent loop 的 tool_call/tool_result trace 映射到 Web 事件流，并逐步接候选论文确认和 evidence 保存。
```

---

# 二十七、下一步建议

下一步建议做：

```text
A3.2：真实运行的 trace/event 映射
```

目标：

```text
真实 PaperPilot 运行时，
页面能看到它调用了哪些工具，
每个工具返回了什么摘要，
是否触发 context compact，
最终结果保存在哪里。
```

具体可以做：

1. 给 `WorkflowRunner.run_real()` 传入 `on_event`。
2. 将 PaperPilot 的事件转写到 `task_events`。
3. 对 tool_result 做长度限制，避免页面被长文本撑爆。
4. 在 UI 中区分：
   - workflow event；
   - tool event；
   - artifact event。

这一步完成后，Web 工作台就不只是“能显示最终回答”，而是能展示真实 Agent 执行过程。

