# PaperPilot 后端运行保护层设计

## 文档属性

- 日期：2026-07-15
- 状态：书面设计已由用户确认，可进入实施
- 范围：`paperpilot/web/*` 后端运行时保护、可观测性与线程任务准入
- 不包含：PostgreSQL 迁移、Redis 强依赖、通用限流、全局请求超时、Prometheus/OpenTelemetry 接入
- 目录说明：本文件位于 `docs/codex-only-plans/`，仅用于 Codex 设计与执行上下文

## 1. 背景

PaperPilot 当前已经具备 FastAPI Web 后端、SQLite 持久化、Cookie 会话、线程任务执行器，以及可选的 Celery/Redis 执行路径。此前完成的任务分页、增量更新和 MCP 常驻化已经减少了重复查询与子进程启动开销。

下一阶段的重点不是继续增加吞吐组件，而是给现有运行路径增加一层明确、可测试的保护机制，使服务面对短时并发、慢任务和异常请求时具备以下性质：

1. 线程任务积压有硬上限，不会无界占用内存。
2. 过载请求在写入任务数据前被拒绝，不制造永远无法执行的任务。
3. 每个请求都可以通过请求 ID 追踪，并形成结构化访问日志。
4. 健康检查能区分“进程活着”和“当前可接收流量”。
5. 非预期异常不会向客户端泄露内部实现或堆栈信息。
6. 所有运行参数都集中校验，错误配置在启动时立即失败。

## 2. 当前缺口

### 2.1 线程执行器队列无界

当前线程执行器使用固定 worker 数量，但底层 `ThreadPoolExecutor` 的等待队列没有容量上限。任务生产速度高于消费速度时，请求仍会持续创建数据库任务并进入内存队列。

风险包括：

- 排队时间持续增长，客户端看到任务已创建但很久不运行。
- 队列对象和任务上下文持续占用内存。
- 服务虽然仍能接受 HTTP 请求，但已经没有真实处理能力。
- 进程重启时，大量仅存在于本地队列中的任务丢失。

### 2.2 缺少统一请求上下文

当前请求没有稳定的关联 ID，访问日志也缺少统一字段。出现单次 500、慢请求或用户反馈时，很难把客户端响应、HTTP 日志和任务创建过程关联起来。

### 2.3 健康检查语义不完整

进程存活、SQLite 可用、任务执行器可以接受新任务是不同状态。单一健康接口无法同时服务进程重启判断和流量接入判断。

### 2.4 配置入口分散

线程 worker 数、执行模式和日志行为如果由各模块独立读取环境变量，会形成默认值不一致、测试难注入、错误值延迟到运行期才暴露的问题。

### 2.5 异常响应缺少统一关联信息

非预期异常应返回稳定、无敏感信息的响应，同时保留请求 ID 供排查。还需要正确处理“响应已经开始发送后才发生异常”的 ASGI 边界，避免尝试发送第二个响应。

## 3. 目标与非目标

### 3.1 本阶段目标

- 为本地线程执行模式增加有界任务容量和原子准入。
- 在数据库写入前完成容量判断。
- 将任务记录和初始 `queued` 事件放在同一 SQLite 事务中创建。
- 增加轻量、低基数、默认安全的结构化请求日志。
- 增加 `/health/live` 和 `/health/ready`。
- 集中解析并校验运行配置。
- 为异常响应、过载响应和健康检查建立明确 API 语义。
- 使用确定性测试覆盖并发容量，不依赖脆弱的绝对性能阈值。

### 3.2 本阶段非目标

- 不迁移到 PostgreSQL。
- 不引入新的 Python 依赖。
- 不引入 Prometheus、OpenTelemetry 或外部日志服务。
- 不实现跨进程的通用 HTTP 限流。
- 不为所有 HTTP 请求设置强制超时。
- 不改变现有任务查询、用户隔离和认证规则。
- 不解决 SQLite 与 Redis 之间的数据库/消息代理双写窗口。
- 不把 Redis 变为 Web 读接口的硬健康依赖。

## 4. 方案比较

### 4.1 方案 A：仅调整 worker 数和反向代理限制

优点：改动最小。

缺点：无法限制 `ThreadPoolExecutor` 内部积压，应用仍可能在数据库中创建无法及时执行的任务；应用自身也无法给出一致的过载语义。

### 4.2 方案 B：应用内运行保护层

内容包括集中配置、纯 ASGI 请求中间件、健康检查，以及线程执行器的容量预留协议。

优点：

- 与当前 FastAPI、SQLite 和线程执行架构一致。
- 不增加部署依赖。
- 可以在任务数据写入前拒绝过载。
- 行为可通过单元测试和 API 测试确定性验证。

缺点：

- 线程容量仍是单进程范围；使用多个 Uvicorn worker 时，每个进程拥有独立容量。
- 只能保护本地线程执行路径，不能替代分布式队列的全局容量治理。

### 4.3 方案 C：立即使用 Redis/Celery 作为唯一执行路径

优点：队列容量和 worker 扩缩可以独立管理。

缺点：增加部署复杂度和外部故障面；当前简单部署不需要强制依赖 Redis；数据库与消息发布仍有双写问题。

### 4.4 选择

本阶段采用方案 B。线程模式继续作为简单、默认的运行路径；Celery 保持可选。保护层先把单进程的资源边界、日志和健康语义做正确，再根据真实负载决定是否进入数据库和分布式队列升级阶段。

## 5. 总体架构

```text
HTTP request
    |
    v
RequestObservabilityMiddleware
    |-- validate/generate request_id
    |-- initialize request.state
    |
    v
FastAPI routing + authentication
    |-- require_user writes request.state.user_id
    |
    v
Task creation endpoint
    |
    v
TaskExecutorLike.reserve()
    |                     \
    | accepted             \ rejected
    v                       v
SQLite transaction       503 + Retry-After
task + queued event      no database write
    |
    v
reservation.submit(task_id, mode)
    |                     \
    | accepted             \ publish/submit failure
    v                       v
background execution     pending -> failed + failed event
                          claimed -> preserve task + 201
    |
    v
future callback releases capacity
```

运行保护层由四个相互独立但协作的部分组成：

1. `WebRuntimeConfig`：集中配置和启动校验。
2. `RequestObservabilityMiddleware`：请求 ID、结构化访问日志和异常响应。
3. 健康接口：存活与就绪状态。
4. 任务准入：容量预留、原子任务创建和容量归还。

## 6. 集中配置设计

新增 `paperpilot/web/config.py`，提供不可变的 `WebRuntimeConfig`。全局应用启动时从环境变量构造配置，测试和 `create_app` 调用方可以直接注入配置对象。

### 6.1 配置项

| 环境变量 | 默认值 | 约束 | 用途 |
| --- | ---: | --- | --- |
| `PAPERPILOT_TASK_EXECUTOR` | `thread` | `thread` 或 `celery` | 后台任务执行方式 |
| `PAPERPILOT_THREAD_WORKERS` | `2` | 正整数 | 线程任务并行 worker 数 |
| `PAPERPILOT_THREAD_QUEUE_CAPACITY` | `4` | 大于等于 0 | worker 之外允许等待的任务数 |
| `PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS` | `1` | 正整数 | 过载响应的 `Retry-After` 秒数 |
| `PAPERPILOT_LOG_LEVEL` | `INFO` | 标准日志级别 | PaperPilot 日志级别 |
| `PAPERPILOT_LOG_FORMAT` | `json` | `json` 或 `text` | 日志输出格式 |
| `PAPERPILOT_SLOW_REQUEST_MS` | `1000` | 正整数 | 慢请求告警阈值 |
| `PAPERPILOT_ENV` | `development` | 非空字符串 | 运行环境标识 |

线程模式总准入容量为：

```text
PAPERPILOT_THREAD_WORKERS + PAPERPILOT_THREAD_QUEUE_CAPACITY
```

例如默认配置允许 2 个运行中任务和 4 个等待任务，总计 6 个未完成任务。

### 6.2 校验行为

- 环境变量不存在时使用默认值。
- 整数格式错误、越界值或未知枚举值在应用启动时抛出清晰的配置错误。
- 不静默回退到默认值，避免生产环境以非预期配置启动。
- `build_task_executor(runner, config=...)` 使用同一配置对象，不再独立解析 executor 类型或线程参数。
- 日志配置只管理 PaperPilot 自身 logger，不修改第三方库的 root handler。

## 7. 请求上下文与结构化日志

新增 `paperpilot/web/observability.py`，使用纯 ASGI middleware，而不是 `BaseHTTPMiddleware`。这样可以直接观察 `http.response.start`、注入响应头，并明确处理流式响应或响应已开始后的异常。

在当前 FastAPI 0.138.0 / Starlette 1.3.1 中，middleware stack 顺序为：

```text
ServerErrorMiddleware
    -> PaperPilot user middleware
        -> ExceptionMiddleware
            -> router
```

保护层必须注册为 FastAPI user middleware。这样已知的 `HTTPException` 先由内层 `ExceptionMiddleware` 转成正常错误响应；未处理异常先到达 PaperPilot 保护层，只有保护层无法安全构造响应时才继续到外层 `ServerErrorMiddleware`。

### 7.1 请求 ID

请求进入时读取 `X-Request-ID`：

- 缺失时生成规范格式的 UUID4 字符串。
- 上游值必须匹配 `^[A-Za-z0-9._:-]{1,128}$`。
- 过长、包含控制字符或不符合字符集时忽略原值并生成新 ID。
- 有效值保留，便于上游网关和客户端串联追踪。
- 所有正常、HTTP 错误和可安全接管的 500 响应都返回 `X-Request-ID`。
- 发送 `http.response.start` 时移除下游可能设置的同名响应头，再追加保护层确认的唯一值，避免重复或不一致的 `X-Request-ID`。

请求 ID 写入 `scope["state"]["request_id"]`，供 FastAPI 的 `request.state.request_id` 访问。

### 7.2 用户 ID 传递

FastAPI 的同步依赖可能在线程池中执行。在线程内修改 `ContextVar`，不能可靠地把新值传播回外层 ASGI middleware。

因此认证依赖调整为接收 `Request`，认证成功后执行：

```python
request.state.user_id = user.id
```

middleware 在响应结束后读取 `scope["state"]`。`ContextVar` 可以作为内部日志辅助，但不作为用户 ID 从同步依赖回传的唯一机制。

### 7.3 每请求日志

专用 logger：`paperpilot.web.access`。

每个请求最多产生一条访问日志，字段包括：

- `event`: 固定为 `http.request.completed` 或明确的异常事件名
- `request_id`
- `method`
- `route`
- `status_code`
- `duration_ms`
- `user_id`，未认证时为空
- `environment`
- `exception_type`，仅异常时存在

`route` 使用 FastAPI 匹配后的路由模板，例如 `/api/tasks/{task_id}`，不记录原始高基数路径。未匹配或在路由建立前失败时使用 `<unmatched>`。

### 7.4 日志级别

- 正常请求：`INFO`
- 耗时达到 `PAPERPILOT_SLOW_REQUEST_MS`：`WARNING`
- 非预期异常：`ERROR`，并带 `exc_info`
- 成功的 `/health/live`：`DEBUG`，减少探针日志噪声
- `/health/ready` 失败：`WARNING`

### 7.5 安全边界

访问日志不记录：

- URL query string
- Cookie 或 Authorization 信息
- 密码和会话令牌
- 任务问题文本
- 请求体或响应体
- artifact 内容
- 异常消息原文

异常类型可以记录，但内部异常文本和堆栈只进入服务端异常日志，不进入客户端响应。

### 7.6 日志配置幂等性

- 配置函数可以被测试或 app factory 多次调用，不重复添加 handler。
- JSON 格式使用标准库 formatter 输出单行 JSON。
- text 格式用于本地阅读，字段保持一致。
- README 的启动示例建议关闭 Uvicorn 自带 access log，避免每个请求重复记录两次。

## 8. 异常响应设计

### 8.1 路由和业务错误

FastAPI/Starlette 已知的 `HTTPException` 继续由现有异常处理链生成响应。middleware 在 `http.response.start` 时补充 `X-Request-ID`，不改变既有业务错误结构。

### 8.2 非预期异常

如果异常发生时响应尚未开始，middleware：

1. 记录一次结构化异常日志。
2. 发送状态码 500。
3. 返回 `application/json`。
4. 添加 `X-Request-ID`。
5. 使用稳定、无内部细节的响应：

```json
{
  "detail": "internal server error",
  "request_id": "<request-id>"
}
```

这样可以避免异常继续进入外层 `ServerErrorMiddleware` 后丢失请求 ID，或产生重复访问日志。

### 8.3 响应已开始后的异常

如果 `http.response.start` 已经发送，middleware 不能再生成第二个 500 响应。此时：

- 记录包含请求 ID 的异常日志。
- 不发送第二个 response start。
- 重新抛出异常，由 ASGI server 关闭或终止当前响应。

这类情况的客户端响应可能不完整，但服务端日志仍可关联。

## 9. 健康检查设计

两个接口均公开、无需登录，并从 OpenAPI schema 隐藏。

### 9.1 `GET /health/live`

语义：当前进程能否处理最基本的 HTTP 请求。

- 不访问 SQLite、Redis、MCP 或外部资源。
- 进程能够进入路由并生成响应时返回 200。
- 用于容器或进程管理器判断是否需要重启进程。

示例：

```json
{"status":"ok"}
```

### 9.2 `GET /health/ready`

语义：当前 Web 进程是否具备接收正常流量的基础条件。

检查：

1. SQLite 执行轻量 `SELECT 1` 成功。
2. 当前任务执行器未进入 shutdown 状态。

成功返回 200：

```json
{"status":"ready"}
```

失败返回 503，并返回稳定的组件状态，不暴露底层异常消息：

```json
{
  "status": "not_ready",
  "checks": {
    "database": "failed",
    "executor": "ok"
  }
}
```

### 9.3 明确不纳入 readiness 的状态

- 线程执行器暂时饱和不会让 readiness 失败。饱和是可恢复的瞬时背压，应由任务创建接口返回 503。
- Redis 不作为 Web readiness 硬依赖。即使 Celery broker 暂时不可用，已有任务、会话和 artifact 的读接口仍应提供服务。
- 不执行 `PRAGMA quick_check` 等昂贵数据库检查。
- 不探测 MCP 子进程或检索资源，避免健康检查本身触发昂贵初始化。

## 10. 任务准入与容量预留

### 10.1 核心原则

准入判断必须发生在创建数据库任务之前。仅检查一个瞬时计数然后再提交会产生竞态，因此使用“预留对象”把容量获取和任务提交绑定起来。

建议协议：

```python
class TaskExecutorLike(Protocol):
    @property
    def is_shutdown(self) -> bool: ...
    def reserve(self) -> TaskSubmissionReservation: ...

class TaskSubmissionReservation(Protocol):
    def submit(self, task_id: str, mode: str) -> object: ...
    def release(self) -> None: ...
```

`reserve()` 成功时返回一次性 reservation。拒绝时使用两个明确异常：

- `TaskExecutorAtCapacityError`：执行器暂时饱和。
- `TaskExecutorShuttingDownError`：执行器已经开始关闭。

两种情况都发生在数据库写入前，并记录 `task.admission_rejected`；日志中的 `reason` 分别为 `capacity` 或 `shutdown`。

### 10.2 线程执行器实现

线程执行器创建一个容量为 `workers + queue_capacity` 的 `threading.BoundedSemaphore`，并使用独立状态锁保护 shutdown 标志和 reservation 状态。`BoundedSemaphore` 可以在实现错误导致超额释放时立即暴露问题，而不是静默扩大容量。

`reserve()`：

- 非阻塞获取一个 permit。
- 获取失败立即拒绝，不等待 HTTP 请求占住连接。
- executor 已 shutdown 时也拒绝。
- 检查 shutdown 与获取 permit 在同一个 executor 状态锁内完成；shutdown 也在该锁内先设置状态，保证 shutdown 开始后不会再成功建立新 reservation。

reservation：

- 只允许调用一次 `submit()`。
- `submit()` 成功后，permit 的所有权转移给 future。
- future 完成、失败或取消时，由 done callback 恰好释放一次 permit。
- `submit()` 如果在 future 所有权建立前抛出异常，必须先由 reservation 内部归还 permit，再把异常抛给调用方。
- 在 `submit()` 前调用 `release()` 会归还 permit。
- `submit()` 返回或抛出后，调用方都不再额外调用 `release()`。
- 重复 `submit()`、提交后的 `release()` 或重复 `release()` 不得导致 permit 数量增加。

reservation 使用明确的内部状态机：

```text
RESERVED -> SUBMITTED -> RELEASED
    |                       ^
    +-----------------------+
          release/failure
```

- `submit()` 只接受 `RESERVED` 状态；其他状态调用时抛出 `RuntimeError`。
- `release()` 和 future callback 共用受锁保护的 `_release_once()`；只有第一次状态转换真正调用 semaphore release。
- 在 `SUBMITTED` 状态手工调用 `release()` 不能提前释放仍在运行任务的容量；由 future callback 完成释放。

### 10.3 同步测试执行器

同步执行器返回无容量限制的 reservation。`submit()` 立即执行任务，保持现有测试和明确同步运行模式的语义。

它仍暴露 shutdown 状态，以便 readiness 和关闭测试使用同一协议。

### 10.4 Celery 执行器

Celery 的队列由 Redis/Celery 管理，本地 Web 进程不使用线程 semaphore，因此 reservation 是轻量的无本地容量预留对象。

Celery 路径继续保留当前语义：

- 发布成功表示 broker 已接受任务。
- 发布失败时将已经创建的任务标记为 failed，并追加 failed 事件。
- 如果发布调用抛错，但 worker 已经把任务 claim 为 running 或已完成，则把它视为“发布结果不明确但任务已被接受”：不覆盖任务状态，不追加伪造的 failed 事件，并返回原有 201 响应。
- 本阶段不宣称解决数据库提交成功、消息发布前进程退出的双写窗口。

### 10.5 shutdown

executor 开始 shutdown 后：

- 不再接受新 reservation。
- readiness 返回 503。
- 已提交任务按照现有 shutdown 策略结束或等待。
- 所有已获取 permit 最终只能释放一次。

## 11. 原子任务创建

当前任务创建和首个事件如果分两次事务完成，中间失败可能留下没有 `queued` 事件的 pending 任务。

新增 `TaskStore.create_queued_task(...)` 或等价方法，在同一个 SQLite 事务中：

1. 插入 task 记录。
2. 插入初始 `queued` event。
3. 一起提交。

任何一步失败都回滚两项写入。

现有 `create_task` 可以保留给内部调用或测试，但 HTTP 任务创建路径必须使用原子方法。

## 12. 任务创建完整流程

### 12.1 成功流程

1. 完成认证和请求校验。
2. 调用 `executor.reserve()`。
3. 在单个 SQLite 事务中创建 task 和 `queued` event。
4. 调用 `reservation.submit(task_id, mode)`。
5. 返回现有成功响应。
6. 后台任务完成后归还容量。

### 12.2 容量饱和

1. `reserve()` 非阻塞失败。
2. 记录 `task.admission_rejected` 结构化日志，包括 request ID、user ID、执行器类型和容量配置。
3. 返回 HTTP 503。
4. 添加 `Retry-After: <configured-seconds>`。
5. 不创建 task、event 或 artifact。

响应沿用 FastAPI 错误格式：

```json
{"detail":"task executor is at capacity"}
```

executor 已开始 shutdown 时同样在数据库写入前返回 503，响应 detail 为 `task executor is shutting down`，但不添加 `Retry-After`；当前进程正在退出，重试时间无法由它可靠承诺。

### 12.3 数据库创建失败

1. 原子事务回滚。
2. 调用 `reservation.release()`。
3. 异常进入统一异常处理。
4. 不提交后台任务。

### 12.4 submit 或 broker publish 明确失败

此时任务和 `queued` event 已经提交：

1. `reservation.submit()` 按协议保证失败时已归还本地容量，调用方不再次释放。
2. 使用 `fail_pending_task()` 仅将仍为 pending 的任务更新为 failed。
3. 更新成功时追加 failed event。
4. 返回 503，表示本次任务没有成功交给执行器。

如果“标记 failed”本身再次失败，保留原始 submit 异常作为主错误，并记录清理失败；不向客户端暴露异常文本。

### 12.5 broker publish 结果不明确但任务已被 claim

Celery publish 可能出现“broker 已接受消息，但客户端在收到确认前断开”的不明确结果。`reservation.submit()` 抛错后：

1. 先尝试 `fail_pending_task()`。
2. 如果更新失败，重新读取任务。
3. 如果任务已经是 running 或 completed，说明 worker 已接手任务。
4. 保留当前状态，不追加 failed event，并返回原有 201 task 响应。
5. 其他状态仍按提交失败返回 503，并记录状态供排查。

这条分支保留当前已有的并发可靠性保护，避免用 Web 进程观察到的发布异常覆盖 worker 已经取得的执行事实。

## 13. API 与兼容性

### 13.1 新增接口

- `GET /health/live`
- `GET /health/ready`

### 13.2 新增响应头

- 所有可正常构造的 HTTP 响应增加 `X-Request-ID`。
- 任务过载响应增加 `Retry-After`；shutdown 拒绝不增加该响应头。

### 13.3 新增错误状态

任务创建接口在本地执行器饱和、executor 已关闭或任务明确提交失败时可以返回 503。Celery 发布结果不明确但任务已被 worker claim 时继续返回 201。

这是有意的行为变化：与接受一个无法及时执行的任务相比，明确拒绝允许客户端稍后重试，也保护服务内存和数据库状态。

### 13.4 保持不变

- 不改变任务成功响应的数据结构。
- 不改变认证和用户隔离规则。
- 不改变任务列表、详情、事件、artifact 和增量更新接口。
- 不把任务 API 改为新的登录策略。
- Celery 仍为可选执行模式。

## 14. 测试设计

### 14.1 配置测试

- 所有默认值正确。
- 合法环境变量覆盖生效。
- 非整数、负数、零值越界和未知日志格式启动失败。
- `create_app` 注入配置时不依赖全局环境。

### 14.2 请求中间件测试

- 缺少请求 ID 时生成并回显。
- 合法上游请求 ID 保留。
- 非法、过长或含控制字符的请求 ID 被替换。
- 404、401、业务错误和非预期 500 都带请求 ID。
- 非预期 500 响应不包含异常消息。
- 响应已开始后的异常不会尝试发送第二个响应。
- 用户 ID 能通过 `request.state` 出现在日志中。
- 日志使用路由模板，不包含 task ID 等原始路径值。
- 日志不包含 query、Cookie、密码、问题文本或 artifact 内容。
- 慢请求、live 探针和异常使用预期日志级别。

### 14.3 健康检查测试

- 正常状态下 live 和 ready 返回 200。
- SQLite `SELECT 1` 失败时 live 仍为 200，ready 为 503。
- executor shutdown 后 ready 为 503。
- executor 饱和时 ready 仍为 200。
- 健康接口不需要认证且不出现在 OpenAPI schema。

### 14.4 准入测试

使用可阻塞的确定性 runner 控制任务完成时机：

- 总容量为 `workers + queue_capacity`。
- 容量内的任务都能创建并提交。
- 下一次请求立即返回 503 和 `Retry-After`。
- 饱和拒绝前后 task/event 表行数不变。
- 释放一个阻塞任务后，可以再次成功创建任务。
- 数据库写入失败时 permit 被归还。
- 明确的 submit 失败使仍为 pending 的任务转为 failed，且 permit 被归还。
- Celery 发布抛错但任务已被 claim 时保留 running/completed 状态并返回 201。
- reservation 的重复提交和重复释放不会扩大容量。
- shutdown 后新 reservation 被拒绝，API 返回 503、不写数据库且不错误添加 `Retry-After`。

### 14.5 回归验证

- Web focused pytest。
- 全量非 slow pytest。
- Python `compileall`。
- `git diff --check`。
- 现有分页和增量更新测试继续通过。

## 15. 基准与运行验证

新增可复现的准入基准或压力脚本，验证行为而不是绑定机器性能：

1. 启动一个会阻塞的任务 runner。
2. 并发提交超过配置总容量的任务。
3. 断言成功数量不超过总容量。
4. 断言其余请求快速收到 503，而不是长时间等待。
5. 断言拒绝请求没有产生数据库行。
6. 释放任务后断言容量恢复。

基准输出可记录吞吐和延迟供人工比较，但不在测试中设置跨机器固定毫秒阈值。

## 16. 部署与文档调整

README 需要补充：

- 新增环境变量及默认值。
- `/health/live` 与 `/health/ready` 的用途。
- 线程总容量计算方式。
- 503 和 `Retry-After` 的客户端重试语义。
- 使用 PaperPilot 结构化访问日志时关闭 Uvicorn 重复 access log 的建议。
- 多 Uvicorn worker 下容量按进程独立计算。例如 2 个进程、每进程默认容量 6，理论本地总未完成容量为 12，但请求分配并不保证均匀。

## 17. 为什么本阶段不接 Prometheus

Prometheus Python client 的多进程模式要求：

- 在进程启动前设置专用目录。
- 每次启动前清理目录。
- 使用专用 multiprocess registry。
- 正确处理 worker 退出生命周期。
- 接受部分 metric 类型和 gauge 行为限制。

当前 README 推荐直接使用 Uvicorn 多 worker，缺少类似 Gunicorn `child_exit` 的现成生命周期钩子。现在直接加入指标依赖容易得到看似可用、实际跨进程不准确的数据。

因此本阶段先输出稳定的结构化日志和健康状态。Prometheus/OpenTelemetry 将作为单独的、与部署拓扑一起设计的阶段处理。

## 18. 已知边界与剩余风险

### 18.1 多进程容量不是全局容量

semaphore 只保护单个 Python 进程。多个 Uvicorn worker 会各自拥有容量。部署时应按“每进程容量乘以进程数”估算上限，并结合 SQLite 写入能力配置。

### 18.2 SQLite 仍是写入瓶颈

有界准入避免无界积压，但不会让 SQLite 获得线性写扩展能力。真实并发持续增长后，仍可能需要 PostgreSQL。

### 18.3 Celery 双写窗口仍存在

任务事务提交后、Celery publish 前进程退出，可能留下 queued/pending 任务但 broker 中没有消息。完整解决需要 outbox 或等价可靠发布设计，不属于本阶段。

### 18.4 503 重试需要客户端策略

`Retry-After` 只提供最小提示。客户端应避免无抖动的紧密重试；更完整的指数退避可以在前端或 API client 阶段补充。

### 18.5 不做强制请求取消

同步 Python 工作在线程池中执行时，HTTP timeout 无法可靠终止底层工作。贸然增加全局 timeout 会造成客户端已超时但服务端仍消耗资源的假象，因此本阶段不实现。

## 19. 验收标准

实现完成需同时满足：

1. 默认线程执行器最多存在 6 个未完成任务，且容量可配置。
2. 第 7 个并发未完成任务在数据库写入前收到 503 和 `Retry-After`。
3. 被准入拒绝的请求不产生 task、event 或 artifact。
4. 任务完成、失败、取消、数据库异常和 submit 异常都不会泄漏 permit。
5. task 与初始 `queued` event 原子创建。
6. 所有可构造响应都携带有效 `X-Request-ID`。
7. 非预期异常响应不泄露内部消息，并可通过 request ID 在服务端日志定位。
8. 访问日志不记录敏感内容，动态资源路径使用路由模板。
9. live 和 ready 接口符合本文定义的依赖边界。
10. 不安装新依赖，不改变既有认证和用户隔离语义。
11. focused 测试、全量非 slow 测试、compileall 和 diff check 全部通过。

## 20. 后续阶段

运行保护层稳定后，再依据真实测量结果决定下一步：

1. 若主要瓶颈是 SQLite 写锁和多实例部署，进入 PostgreSQL 迁移设计。
2. 若任务可靠投递成为主要风险，设计 transactional outbox。
3. 若需要跨进程 HTTP 限流，基于 Redis 或网关实现全局策略。
4. 若需要长期指标和告警，结合实际 Uvicorn/Gunicorn/容器拓扑设计 Prometheus 或 OpenTelemetry。
