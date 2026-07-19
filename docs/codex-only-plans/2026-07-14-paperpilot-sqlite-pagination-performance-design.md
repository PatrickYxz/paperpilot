# PaperPilot SQLite Pagination And Incremental Updates Design

## 文档属性

- 目录：`docs/codex-only-plans/`，仅供 Codex 记录设计与实施上下文。
- 状态：用户已于 2026-07-14 确认采用完整分页方案，待书面设计复核后进入实施计划。
- 范围：优化 FastAPI Web 工作台的数据读取边界、SQLite 并发读写能力和运行中任务的轮询开销。

## 背景与基准

前一阶段已经把耗时研究任务从 FastAPI 请求进程分离到可选的 Celery/Redis worker，并让每个 worker 子进程复用 MCP Runtime。API 进程当前的主要压力转移到了 SQLite 读写和前端轮询。

2026-07-14 在未修改生产代码的工作区上执行临时基准，结果如下：

- 预置 2,000 条同一用户任务后，`list_tasks()` 一次返回 2,000 条，序列化 JSON 约 410,890 bytes。
- 任务列表查询耗时约 2.9 ms，但执行计划为 `SCAN research_tasks`，并使用临时 B-Tree 排序；数据增长后会线性退化。
- 8 个线程并发创建 200 条任务时无错误，单次写入 P50 约 0.29 ms、P95 约 20.68 ms、最大约 94.41 ms。
- 数据库处于 `DELETE` journal mode，外键检查默认关闭，业务查询没有组合索引。
- 运行中任务每秒触发任务详情、完整事件、完整 artifact 和完整任务列表四个请求，历史数据会在每次轮询中重复传输。

这些数字只描述本机一次基准，不作为跨机器性能承诺。它们用于确认优化方向，并为同机前后对比提供基线。

## 目标

1. 让 SQLite 在少量 API 请求和 Celery worker 同时读写时，允许读请求与写请求尽可能并行。
2. 为用户任务列表、任务事件和任务 artifact 提供有界查询，避免响应体随历史数据无限增长。
3. 使用 keyset pagination，避免 offset pagination 在深分页时扫描和丢失稳定性。
4. 把运行中任务的常规轮询从每秒四个完整请求降为一个增量请求。
5. 保持认证、用户隔离、Celery 投递、任务状态机和 MCP Runtime 复用行为不变。
6. 建立可重复的性能基准，能展示查询计划、响应大小和并发写入延迟的优化前后差异。

## 非目标

- 本阶段不迁移到 PostgreSQL、SQLAlchemy 或 Alembic。
- 本阶段不引入 Redis cache、SSE、WebSocket 或消息推送。
- 本阶段不解决 SQLite 与 Redis 双写之间的 transactional outbox 缺口。
- 本阶段不改变任务创建、认证、单任务详情或 Celery task 的请求和返回结构。
- 本阶段不以提高 Celery 研究并发度为目标；研究并发仍受 worker 数量、模型内存和每进程一个 MCP Runtime 的约束。

## 方案选择

### 方案一：只优化 SQLite

启用 WAL 和索引，但保留无界列表和完整轮询。改动最小，却不能限制网络传输和 JSON 序列化成本，因此不采用。

### 方案二：SQLite + 完整分页 + 增量 updates

同时优化存储、查询、API 合约和前端轮询。列表接口改为明确的分页 envelope，运行中任务通过一个聚合 updates 接口只获取新增数据。该方案能覆盖目前观察到的主要瓶颈，作为本阶段实施方案。

### 方案三：兼容旧列表结构

只增加可选的 `limit`、`cursor` 和 `after_id` 参数，未传参数时仍返回全部数据。虽然兼容性最好，但无法保证后端查询有界，不符合本阶段的成熟 API 目标，因此不采用。

## 总体架构

```text
Browser
  -> GET /api/tasks?limit=50&cursor=...
       -> FastAPI validates auth, filter and page size
       -> TaskStore.list_tasks_page()
       -> SQLite WAL snapshot + covering business index
       -> TaskPageResponse

Browser selects/runs a task
  -> GET /api/tasks/{id}/updates?after_event_id=N&after_artifact_id=M
       -> FastAPI validates ownership once
       -> TaskStore.get_task_updates() uses one connection/read transaction
       -> current task + newly appended events/artifacts
       -> browser merges by numeric id and advances watermarks

Celery worker
  -> task status/event/artifact writes
       -> SQLite WAL writer
       -> API readers continue from consistent snapshots
```

## SQLite 设计

### Journal 与连接设置

`TaskStore` 初始化 schema 时执行一次幂等的：

```sql
PRAGMA journal_mode = WAL;
```

每个新连接设置：

```sql
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 30000;
PRAGMA synchronous = NORMAL;
```

- WAL 允许读请求与一个写事务并行，适合当前少量 Uvicorn worker 和 Celery worker 共用本机 SQLite 文件的模型。
- SQLite 仍然只有一个 writer；WAL 不代表数据库已经适合高写入并发。
- `busy_timeout` 与现有 30 秒连接 timeout 对齐，减少短暂写锁直接暴露为 `database is locked`。
- `synchronous=NORMAL` 在 WAL 模式下平衡本地项目的持久性和写入延迟；进程或系统异常仍可能丢失最近尚未 checkpoint 的事务。
- 外键检查在每个连接显式启用，避免只在 schema 连接开启却对业务连接无效。

### 索引

schema 初始化使用 `CREATE INDEX IF NOT EXISTS` 增加：

```sql
CREATE INDEX IF NOT EXISTS idx_tasks_user_created_id
ON research_tasks(user_id, created_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_tasks_user_status_created_id
ON research_tasks(user_id, status, created_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_events_task_id_id
ON task_events(task_id, id);

CREATE INDEX IF NOT EXISTS idx_artifacts_task_id_id
ON task_artifacts(task_id, id);
```

不为当前没有查询路径的字段预建索引。新增索引会增加少量写放大，但任务写入频率远低于运行中页面的读取频率，收益更高。

## 分页模型

### 任务 keyset cursor

任务固定按以下顺序排列：

```sql
ORDER BY created_at DESC, id DESC
```

cursor 包含版本号、当前 `user_id`、标准化后的 `status`、最后一条记录的 `created_at` 和 `id`，使用 URL-safe Base64 编码的紧凑 JSON。服务端要求 cursor 中的用户和 status 与当前请求一致，不一致时返回 `422`。cursor 是不透明定位信息，不承担授权功能，也不做密码学签名；所有查询仍必须显式包含当前 `user_id` 和可选 `status`。

下一页条件为：

```sql
created_at < :created_at
OR (created_at = :created_at AND id < :id)
```

查询读取 `limit + 1` 条，用额外一条判断 `has_more`。实际响应只返回 `limit` 条，并根据最后一条生成 `next_cursor`。该规则在静态数据集上、包括多条任务拥有相同秒级时间戳时，保持稳定、无重复、无遗漏。它不是跨多次请求的数据库 snapshot；分页期间新建或改变 status 的任务可能出现在下一次从第一页开始的遍历中，这是普通 keyset pagination 的明确语义。

### 事件与 artifact watermarks

事件和 artifact 使用 SQLite 自增整数 `id`，按 `id ASC` 读取：

```sql
WHERE task_id = :task_id AND id > :after_id
ORDER BY id ASC
LIMIT :limit_plus_one
```

响应返回 `next_after_id` 和 `has_more`。这不是 offset；新增记录不会改变已读取记录的位置。

### 页面大小

- 默认 `limit=50`。
- 最小值 1，最大值 100。
- FastAPI 使用参数约束处理越界值并返回 `422`。
- updates 中事件与 artifact 共用一个 `limit`。若任一集合仍有更多数据，前端立即继续拉取，排空后再恢复一秒轮询。

## API 合约

### 任务列表

```http
GET /api/tasks?status=running&limit=50&cursor=<opaque>
```

```json
{
  "items": [
    {
      "id": "task_...",
      "question": "...",
      "depth": "standard",
      "status": "running",
      "created_at": "2026-07-14T08:00:00+00:00",
      "updated_at": "2026-07-14T08:00:01+00:00"
    }
  ],
  "next_cursor": "...",
  "has_more": true
}
```

响应模型为 `TaskPageResponse`。这是已确认的行为变更，静态前端会在同一实施中迁移。

### 事件列表

```http
GET /api/tasks/{task_id}/events?after_id=0&limit=50
```

返回 `TaskEventPageResponse`：

```json
{
  "items": [],
  "next_after_id": 0,
  "has_more": false
}
```

### Artifact 列表

```http
GET /api/tasks/{task_id}/artifacts?after_id=0&limit=50
```

返回 `TaskArtifactPageResponse`，结构与事件分页一致。

### 增量 updates

```http
GET /api/tasks/{task_id}/updates
    ?after_event_id=0
    &after_artifact_id=0
    &limit=50
```

返回 `TaskUpdatesResponse`：

```json
{
  "task": {
    "id": "task_...",
    "question": "...",
    "depth": "standard",
    "status": "running",
    "created_at": "...",
    "updated_at": "..."
  },
  "events": {
    "items": [],
    "next_after_id": 0,
    "has_more": false
  },
  "artifacts": {
    "items": [],
    "next_after_id": 0,
    "has_more": false
  }
}
```

`TaskStore.get_task_updates()` 在同一个 SQLite 连接上显式执行 `BEGIN` 开启读事务，然后完成所有权检查、任务读取、事件读取和 artifact 读取，最后提交或回滚。不能只依赖多个连续 `SELECT`，因为 Python sqlite3 不保证它们自动共享同一个显式读事务。该边界避免四个 HTTP 请求看到互相矛盾的快照。

## 前端数据流

1. 登录后请求任务第一页并渲染最多 50 条。
2. 用户选择任务时，把本地事件和 artifact watermarks 重置为 0，调用一次 updates。
3. 浏览器按 numeric id 去重合并返回的事件和 artifact，并保存 `next_after_id`。
4. 若 `has_more=true`，立即再次请求 updates，直到历史数据排空。
5. 若任务仍为 `pending` 或 `running`，一秒后只调用一次 updates。
6. 任务状态变化时更新当前任务行；任务结束后刷新一次任务第一页。
7. 用户点击刷新或改变状态过滤器时重新加载任务第一页。

本阶段只实现第一页任务列表，不增加“加载更多”视觉控件；API cursor 能力完整保留，后续前端需要时可以直接接入。这样能够先保证后端边界和当前工作台性能，同时避免扩大 UI 范围。

## 错误处理与安全

- 无效、损坏、版本不支持、字段缺失，或与当前用户/status 不匹配的 cursor 返回 `422 invalid task cursor`，不退回第一页，避免客户端误以为分页成功。
- `after_id`、`after_event_id`、`after_artifact_id` 小于 0 时由 FastAPI 返回 `422`。
- 不存在或不属于当前用户的任务继续统一返回 `404`，不泄露其他用户任务是否存在。
- cursor 只影响排序位置，不能替代 `user_id` 条件，也不能绕过 status filter。
- 所有 SQL 参数继续使用 sqlite 参数绑定，不拼接用户输入；唯一动态分支只选择预定义查询模板。
- updates 每次最多返回 100 条事件和 100 条 artifact，避免单请求内存和响应体失控。
- SQLite 锁等待超过 timeout 时保留现有异常传播路径；本阶段不增加容易造成重复写入的应用层自动重试。

## 兼容与迁移

- `CREATE INDEX IF NOT EXISTS` 和 WAL 切换会在现有 SQLite 文件上原地执行，不需要重建任务数据。
- `journal_mode=WAL` 会生成 `-wal` 和 `-shm` 伴随文件，API 与 Celery worker 必须继续使用同一个本地文件系统路径。
- 任务列表、事件列表和 artifact 列表响应从裸数组改为分页 envelope。OpenAPI schema、pytest 和静态前端在同一次提交范围内更新。
- 单任务详情、认证、任务创建和 eval API 不变。
- 不提供双响应格式或 legacy 参数；当前项目没有已发布的外部 API 版本，维持两套协议会增加长期维护成本。

## 测试设计

### TaskStore 单元测试

- 新数据库初始化后 `journal_mode` 为 WAL；每个业务连接启用 foreign keys 和 busy timeout。
- schema 包含四个预期业务索引。
- 无 status 与有 status 的任务查询执行计划都使用对应索引，不出现任务全表扫描或临时排序。
- 任务 keyset 分页在相同 `created_at` 下无重复、无遗漏，且严格隔离用户。
- event/artifact watermarks 只返回 `id > after_id` 的记录，并正确计算 `has_more`。
- `get_task_updates()` 对不存在或其他用户的任务返回 `None`，对合法任务返回一致快照。

### API 测试

- 三个列表接口返回新分页 envelope，并执行默认/最大 page size 约束。
- cursor 能连续取完全部任务；损坏 cursor 返回明确的 `422`。
- status filter 与 cursor 组合正确。
- updates 只返回 watermark 之后的数据，并保持 `404` 用户隔离语义。
- 认证、任务创建、任务详情、Celery 投递失败等现有行为继续回归通过。

### 前端验证

- `tests/web/test_web_app.py` 的静态资源契约检查确认脚本使用分页 `items` 和 updates endpoint。
- 启动本地 FastAPI 后，用浏览器 network 记录执行一个 simulated task：运行中每个轮询周期只出现一个 updates 请求，不重复调用任务列表、事件和 artifact 三个接口。
- 浏览器 smoke 确认状态完成后停止轮询并仅刷新一次任务第一页。
- 不为本阶段新增 Node/JavaScript 测试依赖；浏览器 network 记录是轮询行为的验收证据。

### 性能基准

新增 `scripts/benchmark_web_task_store.py`，默认使用临时 SQLite 文件并输出 JSON：

- seed 数量与耗时；
- 第一页查询耗时、返回条数和序列化 bytes；
- `EXPLAIN QUERY PLAN`；
- 指定线程数下并发写入成功数、错误数、wall time、P50/P95/max；
- SQLite journal mode、busy timeout、foreign key 和索引状态。

pytest 不对毫秒数设置硬阈值，避免硬件和 CI 负载造成脆弱测试。验收要求是：

- 任务第一页始终受 page size 限制；
- 查询计划命中业务索引且不使用临时排序；
- 同机同参数基准不出现 `database is locked`；
- 同机优化后分页读取和响应大小优于本设计记录的无界基线。

## 实施边界

实施文件：

- `paperpilot/web/pagination.py`
- `paperpilot/web/task_store.py`
- `paperpilot/web/app.py`
- `paperpilot/web/static/app.js`
- `tests/web/test_pagination.py`
- `tests/web/test_task_store.py`
- `tests/web/test_web_app.py`
- `tests/web/test_workflow.py`
- `tests/web/test_celery_worker.py`
- `scripts/benchmark_web_task_store.py`
- `README.md`

不新增 Python 运行依赖，不删除现有数据或文件，不迁移数据库引擎。

## 风险与后续

- WAL 只改善读写并行，SQLite 仍是单 writer。用户量和写入量继续增长后，应迁移 PostgreSQL。
- `synchronous=NORMAL` 比 `FULL` 更偏向延迟，极端断电场景的最近事务持久性更弱；这是本地小规模部署的明确取舍。
- 分页响应是 breaking API change，必须确保前端和 API 在同一版本部署。
- cursor 基于秒级 `created_at` 与随机任务 id，排序稳定但同秒任务的顺序不再等价于旧 rowid 顺序；没有业务语义依赖旧顺序。
- 一秒增量轮询仍是 polling。用户规模增大后可考虑 SSE，并通过 Redis pub/sub 或独立事件总线避免每个连接查询 SQLite。
- SQLite/Redis 双写窗口仍存在，后续可靠性阶段应设计 transactional outbox 或 pending-task reconciler。
- 本阶段完成后，下一优先项建议是可观测性：health/readiness、request id、结构化日志、指标与限流。
