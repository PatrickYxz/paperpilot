# PaperPilot FastAPI Auth And Concurrency Plan

## 背景与目标

当前分支为 `codex/goal_test`，现有工作区在计划开始时为干净状态。PaperPilot 已经有一个可运行的 Web 后端雏形，不需要从零新建 FastAPI 项目：

- `paperpilot/web/app.py` 已提供 FastAPI `create_app()`、任务 API、静态页面、eval API。
- `paperpilot/web/task_store.py` 已用 SQLite 保存 research tasks、events、artifacts。
- `paperpilot/web/workflow.py` 已封装 simulated / real workflow runner。
- `tests/web/*` 已覆盖任务 API、任务存储、workflow runner、eval summary 等。

本次目标是在当前基础上继续产品化后端：

1. 支持用户注册、登录、退出和当前用户查询。
2. 让任务、事件和 artifacts 可以按用户归属隔离。
3. 增加简单并发执行能力，让多个任务可以由受控后台执行器处理。
4. 保持实现轻量，优先沿用 FastAPI + SQLite + 标准库，避免不必要的新依赖。

## GitHub 参考结论

已按要求使用 GitHub 插件检索 FastAPI 后端模板和项目。搜索方向包括：

- `FastAPI production backend JWT auth SQLAlchemy background tasks`
- `tiangolo full stack fastapi template`
- `fastapi best practices backend template auth users`

参考仓库结果包括：

- `kkeenee/tiangolo-full-stack-fastapi-template`
- `waseemhnyc/arq-full-stack-fastapi-template`
- `kuldeepghorpade05/fastapi-async-beyond-crud`

这些项目/模板的共性是：

- 认证和用户模块通常独立于业务 API。
- API 层尽量薄，业务和数据访问有单独边界。
- 后台任务或并发执行不直接混在请求处理函数里，而是通过 worker、queue、executor 或 task service 表达。
- 生产级模板往往引入 Postgres、JWT、Redis/Celery/ARQ 等，但 PaperPilot 当前更适合先做 SQLite 本地产品化，不宜过早上重依赖。

## 约束条件

- 遵守现有代码风格，不做无关重构。
- 未经用户确认，不新增第三方认证依赖。
- 未经用户确认，不删除旧 API、旧数据或旧配置。
- 登录和任务隔离属于行为变化：如果任务 API 从匿名访问改为必须登录，需要用户确认。
- 当前仓库没有 `pyproject.toml`，依赖集中在 `requirements.txt`。
- 现有依赖包含 `fastapi`、`uvicorn`、`pytest`、`httpx`，不包含 `passlib`、`bcrypt`、`python-jose`、`PyJWT`。

## 推荐方案

推荐采用 A 方案：本地 SQLite 登录，不新增依赖。

### 认证设计

- 新增 `paperpilot/web/auth.py`。
- 使用 SQLite 表保存用户和 session：
  - `users(id, username, password_hash, password_salt, created_at)`
  - `sessions(token, user_id, created_at, expires_at)`
- 使用 Python 标准库 `hashlib.pbkdf2_hmac` + `secrets` 实现密码哈希。
- 登录成功后通过 HTTP-only cookie 保存 session token。
- 提供 API：
  - `POST /api/auth/register`
  - `POST /api/auth/login`
  - `POST /api/auth/logout`
  - `GET /api/auth/me`

### 数据隔离设计

- 扩展 `research_tasks` 增加 `user_id`。
- `TaskStore.create_task()` 接收 `user_id`。
- `TaskStore.list_tasks()`、`get_task()`、`list_events()`、`list_artifacts()` 支持按 `user_id` 过滤。
- 任务相关 API 通过 `current_user` dependency 获取用户，再只访问该用户数据。
- 迁移旧数据时可以允许 `user_id` 为空，避免破坏历史本地数据；新 API 创建的任务必须有用户。

### 并发设计

- 新增 `paperpilot/web/task_executor.py`。
- 使用 `concurrent.futures.ThreadPoolExecutor`，默认 `max_workers=2`。
- `create_app()` 支持注入 executor，测试可以注入同步或可控 executor。
- `/api/tasks` 创建任务后提交到 executor，而不是直接把执行细节散落在路由函数里。
- executor 只负责排队和调用 `WorkflowRunner`，任务状态仍由 `WorkflowRunner` / `TaskStore` 记录。

### 前端最小适配

- 在 `paperpilot/web/static/index.html` 增加登录/注册区域。
- 在 `paperpilot/web/static/app.js` 增加 auth API 调用：
  - 页面加载时调用 `/api/auth/me`。
  - 未登录时显示登录/注册。
  - 登录后显示现有 workbench 并加载任务。
  - 退出后隐藏工作台。
- 保持现有页面结构，不做大规模 UI 重写。

## 分步骤执行计划

### Step 1: 添加认证测试

目标：先定义用户注册、登录、退出、当前用户查询的预期行为。

涉及文件：

- 新增或修改：`tests/web/test_auth.py`
- 修改：`tests/web/test_web_app.py`

测试点：

- 注册新用户返回当前用户信息，不返回密码哈希。
- 重复用户名返回 409。
- 登录成功设置 session cookie。
- 登录错误返回 401。
- `/api/auth/me` 在登录后返回当前用户。
- logout 后 `/api/auth/me` 返回 401。

### Step 2: 实现认证存储与服务

目标：实现标准库版本的本地认证，不新增依赖。

涉及文件：

- 新增：`paperpilot/web/auth.py`
- 修改：`paperpilot/web/task_store.py`

实现点：

- 用户名规范化和基础校验。
- PBKDF2 密码哈希和常量时间比较。
- session token 生成、保存、查询和删除。
- SQLite schema 创建和轻量迁移。

### Step 3: 接入 FastAPI auth routes

目标：让 FastAPI 提供登录相关 API。

涉及文件：

- 修改：`paperpilot/web/app.py`

实现点：

- 注册 auth route。
- 定义 `CurrentUser` dependency。
- 设置 HTTP-only session cookie。
- 统一 401 / 409 错误响应。

### Step 4: 添加任务用户隔离测试

目标：确保用户只能访问自己的 task、events、artifacts。

涉及文件：

- 修改：`tests/web/test_web_app.py`
- 修改：`tests/web/test_task_store.py`

测试点：

- 未登录创建任务返回 401。
- 用户 A 创建的任务不会出现在用户 B 的任务列表里。
- 用户 B 访问用户 A 的 task detail/events/artifacts 返回 404。
- 用户 A 能正常访问自己的任务结果。

### Step 5: 实现 TaskStore 用户归属

目标：把任务数据归属到用户。

涉及文件：

- 修改：`paperpilot/web/task_store.py`
- 修改：`paperpilot/web/app.py`
- 视需要修改：`paperpilot/web/workflow.py`

实现点：

- `research_tasks` 增加 `user_id`。
- 新数据库直接创建带 `user_id` 的表。
- 旧数据库通过 `ALTER TABLE` 迁移。
- 新任务必须传入当前用户 ID。
- 任务查询按当前用户过滤。

### Step 6: 添加并发执行器测试

目标：验证任务执行不再被路由函数直接承担，且可以受控并发。

涉及文件：

- 新增：`tests/web/test_task_executor.py`
- 修改：`tests/web/test_web_app.py`

测试点：

- executor 会提交 simulated / real workflow。
- 可配置 `max_workers`。
- 测试用 executor 可以同步执行，保证现有 API 测试稳定。
- 两个任务可以被提交到同一个后台执行边界。

### Step 7: 实现轻量 TaskExecutor

目标：增加简单并发能力，不引入 Celery/Redis。

涉及文件：

- 新增：`paperpilot/web/task_executor.py`
- 修改：`paperpilot/web/app.py`

实现点：

- 封装 `ThreadPoolExecutor`。
- `submit(task_id, execution_mode)` 根据模式调用 runner。
- `create_app()` 支持注入 executor。
- FastAPI shutdown 时关闭 executor，避免悬挂线程。

### Step 8: 前端最小登录适配

目标：让用户能从现有页面完成登录/注册并使用 workbench。

涉及文件：

- 修改：`paperpilot/web/static/index.html`
- 修改：`paperpilot/web/static/app.js`
- 修改：`paperpilot/web/static/styles.css`

实现点：

- 增加登录/注册表单。
- 增加当前用户显示和 logout 按钮。
- 未登录隐藏任务创建和任务列表。
- API 请求自动携带 cookie。
- 401 时回到登录状态。

### Step 9: 验证与回归

目标：证明功能满足目标且没有破坏现有 web 流程。

命令：

```bash
pytest tests/web -q
```

必要时补充：

```bash
pytest tests/web/test_auth.py tests/web/test_task_executor.py tests/web/test_web_app.py -q
```

验证证据：

- auth API 测试通过。
- 用户隔离测试通过。
- 并发执行器测试通过。
- 现有 task workflow 测试通过。

## 风险与待确认项

### 必须确认

- 是否确认采用推荐 A 方案：本地 SQLite 登录，不新增依赖。
- 是否接受任务 API 从匿名访问改为必须登录。

### 风险

- 使用标准库 PBKDF2 可以满足本地演示和轻量产品化，但不是完整生产认证方案。
- SQLite + ThreadPoolExecutor 适合本地并发和小规模任务，不适合多进程、多机器部署。
- 旧数据库迁移需要谨慎处理 `user_id` 为空的历史任务，避免破坏已有本地数据。

### 后续可升级方向

- 如果未来要部署到公网，再考虑引入 `passlib[bcrypt]` 或 Argon2。
- 如果未来要多用户长期运行，再考虑 JWT refresh token、Postgres、任务队列、权限模型。
- 如果 real workflow 任务变长或资源占用变高，再考虑独立 worker 进程。
