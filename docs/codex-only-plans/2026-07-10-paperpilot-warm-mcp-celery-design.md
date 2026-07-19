# PaperPilot Warm MCP Runtime And Celery Design

## 文档属性

- 目录：`docs/codex-only-plans/`，仅供 Codex 记录设计与实施上下文。
- 状态：已由用户在 2026-07-10 确认实施。
- 范围：解决真实 Web 研究任务重复启动 MCP 子进程、重复加载 ColBERT 模型，以及进程内任务队列不可恢复的问题。

## 背景

当前 Web 真实任务调用链为：

```text
POST /api/tasks
  -> TaskExecutor(ThreadPoolExecutor)
  -> WorkflowRunner.run_real()
  -> paperpilot.conversation.run()
  -> ConversationSession.start()
  -> MCPClient.start()
  -> 启动 arxiv / colbert / graph / vlm stdio 子进程
  -> ColBERT IndexManager 加载模型和索引
  -> 完成任务
  -> ConversationSession.close()
  -> MCPClient.close() 关闭全部 MCP 子进程
```

这会让每条真实任务重复承担进程启动、MCP initialize、ColBERT 模型加载和论文索引加载开销。当前 `ThreadPoolExecutor` 还存在队列只在 API 进程内、重启丢任务、多个 Uvicorn worker 互不协调的问题。

## 目标

1. 每个长期存活的 Celery worker 子进程只创建一个 MCP Runtime。
2. 第一个真实任务完成 Runtime 初始化，后续任务复用 MCP 子进程、ColBERT 模型和内存索引。
3. 每个研究任务仍创建独立的消息、TodoStore、事件回调和用户上下文，禁止跨任务状态泄漏。
4. FastAPI 只负责持久化任务并投递队列，使用 Celery 时不在 API 进程运行真实研究任务。
5. 保留当前线程执行器作为本地开发默认值，通过环境变量显式启用 Celery。
6. Redis 使用 AOF 持久化，为 Celery broker 提供可恢复队列。

## 非目标

- 本阶段不把 SQLite 迁移到 PostgreSQL。SQLite 仍会限制高写入并发，后续高并发数据库阶段单独实施。
- 本阶段不把 stdio MCP 改成 HTTP MCP 服务。
- 本阶段不允许多个线程同时调用同一个 ColBERT `IndexManager`。
- 本阶段不重写静态前端，也不改变现有任务、事件和 artifact API 返回结构。
- 本阶段不在 Celery result backend 保存研究结果；任务状态和结果仍以 `TaskStore` 为唯一来源。

## 方案比较

### 方案 A：API 进程内共享 MCP Runtime

优点是改动最少。缺点是每个 Uvicorn worker 都会复制模型，研究任务仍与 API 生命周期耦合，进程重启仍会丢任务。因此不采用。

### 方案 B：Celery 预热进程池，每进程一个 MCP Runtime

API、队列和研究 worker 分离。每个 Celery prefork 子进程拥有一个长期 Runtime，并串行处理研究任务；通过增加 worker 进程数量扩展研究并行度。该方案兼顾可靠性、资源上限和现有同步 PaperPilot 链路，作为本阶段方案。

### 方案 C：独立 HTTP MCP / ColBERT 检索集群

多个 worker 可以共享独立检索服务，适合更大规模和 GPU 集群。代价是服务发现、网络协议、鉴权、索引路由和部署复杂度明显增加，留作后续阶段。

## 架构

```text
FastAPI/Uvicorn
  -> TaskStore.create_task()
  -> CeleryTaskExecutor.submit()
  -> Redis broker (AOF)
  -> Celery prefork worker child
       -> 第一次真实任务：MCPRuntime.start()
       -> 后续真实任务：复用同一个 MCPRuntime
       -> 每条任务：创建独立 ConversationSession
       -> WorkflowRunner 写入 task events / artifacts
```

### 进程与并发模型

- API 可以运行多个 Uvicorn worker，但 API 进程不创建 MCP Runtime。
- Celery 使用 prefork。每个子进程一次只执行一个研究任务。
- 每个子进程有一个进程内 `MCPRuntime` 单例，由 worker task 模块懒创建。
- 研究并发度等于 Celery worker `--concurrency`，初始建议为 2。
- `MCPRuntime.lease_tools()` 使用锁保护整个任务租约，即使将来误用线程池，也不会并发调用同一个 ColBERT manager。
- worker 使用 SQLite 原子 claim；普通重复消息不能接管 `running`，只有 Celery 标记的 redelivery 可以恢复 worker-loss 后的任务。

## 组件设计

### `paperpilot/tools/mcp_runtime.py`

新增 `MCPRuntime`：

```python
class MCPRuntime:
    def __init__(self, client_factory: Callable[[], MCPClientLike]) -> None: ...
    def start(self) -> None: ...
    def close(self) -> None: ...
    @contextmanager
    def lease_tools(self) -> Iterator[list[Tool]]: ...
```

职责：

- 在当前进程内最多启动一个 `MCPClient`。
- 多次任务租约复用同一个 client。
- 用独占锁包住一条任务的完整工具使用周期。
- `close()` 幂等，供 worker 子进程退出时尽力清理。

### `paperpilot/conversation.py`

保留 CLI 的一次性行为，并为 worker 增加可选 Runtime：

```python
def run(
    query: str,
    *,
    max_iter: int = 8,
    on_event=None,
    mcp_runtime: MCPRuntime | None = None,
) -> list[dict]: ...
```

- `mcp_runtime is None`：沿用当前每次创建和关闭 `MCPClient` 的行为。
- 提供 Runtime：从 `lease_tools()` 借用 MCP tools，重新构建本任务的 builtin tools、TodoStore、messages 和事件回调。
- 会话关闭时只关闭任务级资源，不关闭共享 Runtime。

### `paperpilot/web/celery_app.py`

创建 Celery app，配置：

- broker：`PAPERPILOT_CELERY_BROKER_URL`，默认 `redis://127.0.0.1:6379/0`。
- JSON serializer。
- `worker_prefetch_multiplier=1`，避免长任务被单个 worker 预取过多。
- `task_acks_late=True`。
- `task_reject_on_worker_lost=True`。
- `worker_cancel_long_running_tasks_on_connection_loss=True`，broker 连接丢失时取消仍在执行的 late-ack 任务，再由 redelivery 恢复。
- 任务软/硬时限默认 10800/11100 秒，Redis visibility timeout 默认 14400 秒。
- 不配置 result backend，任务状态由 SQLite 保存。

### `paperpilot/web/worker_tasks.py`

提供任务 `paperpilot.web.execute_research_task`：

```python
def execute_research_task(task_id: str, execution_mode: ExecutionMode) -> None: ...
```

- simulated 模式直接运行模拟 workflow，不启动 MCP Runtime。
- real 模式懒创建进程级 Runtime，并把它注入 `conversation.run()`。
- 使用 `worker_process_shutdown` 尽力调用 Runtime `close()`。
- 不在 `worker_process_init` 加载 ColBERT：Celery 官方要求该 signal handler 不得阻塞超过 4 秒，而模型初始化可能超过该限制。

### `paperpilot/web/task_executor.py`

新增 `TaskExecutorLike` 协议和 `CeleryTaskExecutor`：

```python
class TaskExecutorLike(Protocol):
    def submit(self, task_id: str, execution_mode: ExecutionMode) -> object: ...
    def shutdown(self) -> None: ...

class CeleryTaskExecutor:
    def submit(self, task_id: str, execution_mode: ExecutionMode) -> object: ...
    def shutdown(self) -> None: ...
```

通过 `PAPERPILOT_TASK_EXECUTOR=thread|celery` 选择默认执行器。默认保持 `thread`，避免本地开发在没有 Redis 时失效。

### `paperpilot/web/task_store.py`

支持 `PAPERPILOT_TASK_DB_PATH`。API 与 Celery worker 必须指向同一个数据库文件。默认仍为 `data/web/tasks.sqlite3`。

## 请求与任务数据流

1. 用户创建任务。
2. API 写入 `research_tasks` 和 queued event。
3. API 调用执行器投递任务。
4. 投递成功后返回现有 `201 TaskResponse`。
5. 投递失败时将任务标记为 `failed`，记录 queue failure event，并返回 `503 task queue unavailable`。
6. worker 取得任务并调用 `WorkflowRunner`。
7. worker 通过当前进程的 Runtime 执行真实 PaperPilot 链路。
8. 结果和事件继续写入现有 TaskStore，前端无需修改协议。

## 生命周期与失败处理

- Runtime 首次启动失败：真实任务由 `WorkflowRunner` 标记为 failed；下一次任务重新尝试创建 Runtime。
- 已启动 Runtime 的 MCP timeout 或连接中断：当前租约标记失效，租约退出后关闭 client，下一个任务重新创建。
- `MCPClient` 将 MCP SDK 与 AnyIO stream 断连统一翻译为 `MCPTransportError`，避免 Runtime 依赖第三方异常细节。
- worker 被强制终止：Celery late ack + reject on worker lost 允许 broker 重新投递；任务执行必须容忍重复进入。
- shutdown signal 不保证执行，因此正确性不能依赖 `close()`；它只负责正常退出时释放子进程。
- API 投递失败：只原子执行 `pending -> failed`；若 worker 已将任务 claim 为 running/completed，不覆盖其状态并按已接受返回。

## 配置

```text
PAPERPILOT_TASK_EXECUTOR=thread|celery
PAPERPILOT_CELERY_BROKER_URL=redis://127.0.0.1:6379/0
PAPERPILOT_TASK_DB_PATH=data/web/tasks.sqlite3
PAPERPILOT_TASK_SOFT_TIME_LIMIT_SECONDS=10800
PAPERPILOT_TASK_TIME_LIMIT_SECONDS=11100
PAPERPILOT_REDIS_VISIBILITY_TIMEOUT_SECONDS=14400
```

`compose.yaml` 只启动带 AOF 的 Redis，API 和 worker 仍由本地 `.venv` 命令运行，避免本阶段额外引入完整镜像构建。

## 测试与验收

### 单元测试

- 多次 `MCPRuntime.lease_tools()` 只启动一次 client。
- `close()` 幂等。
- 同一个 Runtime 的任务租约不会并行。
- `conversation.run(..., mcp_runtime=runtime)` 为每次调用创建独立会话工具，但不关闭 Runtime。
- worker 连续执行两个真实任务时只创建一个 Runtime。
- simulated worker task 不初始化 MCP。
- Celery executor 发送固定任务名和 JSON 参数。
- 队列投递失败后 API 返回 503，任务状态为 failed。
- 环境变量可以选择 thread/celery backend 和共享数据库路径。

### 回归测试

```bash
.venv/bin/python -m pytest tests/test_mcp_runtime.py tests/test_conversation_session.py tests/web -q
.venv/bin/python -m pytest -q
git diff --check
```

### 运行验证

- 启动 Redis。
- 启动 concurrency=1 的 Celery worker。
- 连续投递两个 simulated 任务，确认队列和状态链路。
- Runtime 复用由注入 fake runtime 的集成测试证明；真实 ColBERT smoke 需要本地模型与 API 凭据，不纳入默认测试。

## 风险与后续

- SQLite 仍是单写者，本阶段只解决任务队列和 MCP 启动开销，不代表整个后端已经达到最终高并发目标。
- SQLite 提交和 Redis 发布不是同一事务；API 在两者之间崩溃仍可能留下 pending orphan，完整修复需要 transactional outbox 或 reconciler。
- 每个 Celery prefork 子进程仍会拥有一份 ColBERT 模型；提高 concurrency 会线性增加模型内存，需要压测后配置。
- `_states` 当前没有容量上限，长期 worker 可能缓存越来越多论文索引。LRU/TTL 作为后续优化，不在本阶段改变检索行为。
- 当单机 warm worker 不足时，再把 ColBERT 拆成独立 HTTP MCP/检索服务。
