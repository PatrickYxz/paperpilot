# PaperPilot SQLAlchemy/Alembic Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 SQLAlchemy 2.0 和 Alembic 接管 PaperPilot 现有 Web 业务数据库的数据访问与结构迁移，同时保持当前 `TaskStore` 公共接口、SQLite 文件、API 行为、状态值、分页语义和 Agent 持久化表不变。

**Architecture:** 本阶段只替换数据基础设施，不引入 Repository/Unit of Work 等额外自定义层。`TaskStore` 继续作为兼容入口；它内部改用同步 SQLAlchemy `Engine + Session`，ORM 行模型与现有对外 dataclass 分离。Alembic 用一个非破坏式 adoption revision 同时支持空库、当前库和历史残缺库，并且只管理 Web 的五张表，不触碰同一 SQLite 文件中的 Agent 表。

**Tech Stack:** 项目当前虚拟环境 Python 3.12、SQLAlchemy `>=2.0.51,<2.1`、Alembic `>=1.18.5,<2`、SQLite、pytest、FastAPI、Celery。

## Global Constraints

- 本计划是 [PaperPilot LangGraph 重构设计](./2026-08-06-paperpilot-langgraph-refactor-design.md) 的第一个可独立交付子项目。
- 本阶段不实现 LangGraph、LangChain agent、会话/消息模型、checkpoint、LangGraph Store 或用户记忆。
- 不改变现有 HTTP API 的路径、入参、返回体、鉴权行为和错误码。
- 不改变 `TaskStore` 的现有公开方法名、参数和返回 dataclass；只允许新增幂等的 `close()` 生命周期方法。
- 不改变任务 depth/status/event type 等现有字符串值及状态转换规则。
- 继续使用同步 SQLAlchemy；FastAPI 当前同步 handler、线程池和 Celery worker 不需要切换到异步数据库栈。
- 不使用 `Base.metadata.create_all()` 作为生产建库或升级机制；所有 schema 变化只通过 Alembic。
- 初始迁移只允许创建缺失对象、补充历史缺失列和索引；不得删除、重命名或重建现有表。
- 同一 SQLite 文件里的 `agent_runs`、`agent_steps`、`run_checkpoints`、`tool_executions` 以及任何未知表都必须原样保留。
- 当前真实数据库中的 `research_tasks.user_id` 是历史 `ALTER TABLE` 加入的，没有到 `users.id` 的外键。旧库升级时接受并保留这一差异；新库按完整外键模型创建。补旧库外键若以后确有必要，应单独设计数据清洗和重建迁移。
- `TaskStore(db_path)` 仍要兼容“给一个空路径即可使用”的现有行为。实现上由 Alembic 幂等升级代替手写 `_ensure_schema()`；生产部署仍必须在 Web/worker 启动前显式执行 `alembic upgrade head`，避免多进程首次启动同时迁移。
- 每个任务遵循测试先行；每个提交只包含该任务相关文件，不夹带已有未跟踪计划或用户改动。
- 安装新增依赖属于外部环境变更，执行本计划时应先获得用户授权。

---

## 1. Current-State Contract

### 1.1 当前数据库边界

`paperpilot/web/task_store.py` 目前用 `sqlite3` 同时承担：

1. SQLite 路径解析和连接参数；
2. 运行时建表、补列、建索引；
3. 用户和登录 session 持久化；
4. 研究任务、事件、产物的 CRUD 与分页；
5. 任务状态原子转换和聚合快照读取。

它管理五张表：

| 表 | 作用 | 本阶段所有者 |
|---|---|---|
| `users` | Web 用户和密码哈希 | SQLAlchemy/Alembic Web metadata |
| `sessions` | 登录 token 和过期时间 | SQLAlchemy/Alembic Web metadata |
| `research_tasks` | 研究任务主记录 | SQLAlchemy/Alembic Web metadata |
| `task_events` | 任务事件流 | SQLAlchemy/Alembic Web metadata |
| `task_artifacts` | 任务输出产物 | SQLAlchemy/Alembic Web metadata |

同一文件中还可能存在 `paperpilot/agent/store.py` 管理的四张表：

- `agent_runs`
- `agent_steps`
- `run_checkpoints`
- `tool_executions`

Alembic 的 target metadata 和 autogenerate filter 必须排除这些 Agent 表，初始迁移也不得枚举删除未知表。

### 1.2 必须保持的 `TaskStore` 接口

以下现有方法全部保持签名和返回值语义：

```python
create_user
get_user_by_username
get_user_by_id
create_session
get_user_for_session
delete_session
create_task
create_queued_task
check_health
list_tasks_page
get_task
update_status
claim_task
fail_pending_task
add_event
list_events_page
add_artifact
list_artifacts_page
get_task_updates
```

以下现有 dataclass 保留在 `paperpilot/web/task_store.py`，避免第一阶段引发跨模块导入迁移：

```python
ResearchTask
TaskPage
WebUser
TaskEvent
TaskArtifact
TaskEventBatch
TaskArtifactBatch
TaskUpdates
```

### 1.3 目标运行路径

```text
FastAPI / Celery / AuthService / WorkflowRunner
                    |
                    v
          TaskStore（兼容业务入口）
                    |
        +-----------+-----------+
        |                       |
        v                       v
SQLAlchemy Session       对外 dataclass 映射
        |
        v
Web ORM rows（仅五张表）
        |
        v
同一个 SQLite 文件

部署/本地首次启动：Alembic -> 非破坏式升级到 head
```

---

## 2. Target File Layout and Interfaces

### 2.1 新增文件

```text
alembic.ini
migrations/
  env.py
  script.py.mako
  versions/
    20260806_0001_adopt_web_schema.py
paperpilot/web/
  database.py
  db_models.py
  db_migrations.py
tests/web/
  test_database.py
  test_db_models.py
  test_db_migrations.py
```

### 2.2 修改文件

```text
requirements.txt
paperpilot/web/task_store.py
paperpilot/web/app.py
paperpilot/web/worker_tasks.py
tests/web/test_task_store.py
tests/web/test_auth.py
tests/web/test_web_app.py
tests/agent/test_store.py       # 原则上只验证；仅在错误类型断言确需调整时修改
README.md                       # 增加迁移和备份操作说明
```

### 2.3 数据库基础接口

`paperpilot/web/database.py` 只负责 SQLite/SQLAlchemy 的机械配置，公开
`DEFAULT_TASK_DB_PATH`、`resolve_task_db_path()`、`build_sqlite_url()`、
`create_task_engine()` 和 `create_session_factory()`。完整签名与实现见 Task 1。

`paperpilot/web/db_migrations.py` 只负责 Alembic 的调用边界，公开
`build_alembic_config()`、`get_database_heads()`、`get_script_heads()`、
`upgrade_database()` 和 `ensure_database_current()`。完整签名与实现见 Task 3。

这里不再包一层通用 Repository。业务查询仍直接写在 `TaskStore` 里，使调用链保持可读：handler/service -> `TaskStore` -> SQLAlchemy statement。

---

## 3. Pre-Implementation Baseline

- [ ] 确认工作树，记录并保留所有既有用户改动：

  ```bash
  git status --short
  ```

- [ ] 运行本阶段相关基线测试，保存测试数量和失败信息：

  ```bash
  ./.venv/bin/python -m pytest \
    tests/web/test_task_store.py \
    tests/web/test_auth.py \
    tests/web/test_web_app.py \
    tests/agent/test_store.py -q
  ```

- [ ] 只读记录当前真实库的表、列、索引和行数；禁止直接在真实库执行迁移：

  ```bash
  sqlite3 data/web/tasks.sqlite3 ".tables"
  sqlite3 data/web/tasks.sqlite3 ".schema"
  sqlite3 data/web/tasks.sqlite3 "PRAGMA integrity_check;"
  ```

- [ ] 若基线已有失败，在开始修改前把失败记录到实施日志；不要把既有失败误判成重构回归。

---

## Task 1: Add SQLAlchemy Engine and SQLite Transaction Policy

**Files:**

- Modify: `requirements.txt`
- Create: `paperpilot/web/database.py`
- Create: `tests/web/test_database.py`

### Rationale

先把数据库路径、连接 PRAGMA 和事务策略独立验证，再引入 ORM。SQLite 在 Python 驱动的 legacy transaction mode 下不会为所有 SELECT/DDL 自动发出 `BEGIN`，会破坏 `get_task_updates()` 当前依赖的一致性快照；因此 Engine 必须显式控制事务开始。

### Steps

- [ ] 1. 在 `requirements.txt` 增加稳定版本范围：

  ```text
  SQLAlchemy>=2.0.51,<2.1
  alembic>=1.18.5,<2
  ```

- [ ] 2. 经用户授权后，在项目虚拟环境安装锁定后的依赖：

  ```bash
  ./.venv/bin/python -m pip install -r requirements.txt
  ```

- [ ] 3. 新建 `tests/web/test_database.py`，先写以下失败测试：

  ```python
  from __future__ import annotations

  from pathlib import Path

  from sqlalchemy import text

  from paperpilot.web.database import (
      build_sqlite_url,
      create_session_factory,
      create_task_engine,
      resolve_task_db_path,
  )


  def test_resolve_task_db_path_prefers_argument_over_environment(
      tmp_path: Path, monkeypatch
  ) -> None:
      env_path = tmp_path / "from-env.sqlite3"
      explicit_path = tmp_path / "explicit.sqlite3"
      monkeypatch.setenv("PAPERPILOT_TASK_DB_PATH", str(env_path))

      assert resolve_task_db_path(explicit_path) == explicit_path.resolve()
      assert resolve_task_db_path() == env_path.resolve()


  def test_sqlite_url_preserves_spaces_in_absolute_path(tmp_path: Path) -> None:
      db_path = (tmp_path / "folder with spaces" / "tasks.sqlite3").resolve()

      url = build_sqlite_url(db_path)

      assert url.drivername == "sqlite+pysqlite"
      assert Path(url.database or "") == db_path


  def test_engine_applies_sqlite_pragmas(tmp_path: Path) -> None:
      engine = create_task_engine(tmp_path / "tasks.sqlite3")
      try:
          with engine.connect() as connection:
              assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
              assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 30_000
              assert connection.exec_driver_sql("PRAGMA synchronous").scalar_one() == 1
              assert connection.exec_driver_sql("PRAGMA journal_mode").scalar_one() == "wal"
      finally:
          engine.dispose()


  def test_session_transactions_provide_repeatable_read_snapshot(tmp_path: Path) -> None:
      engine = create_task_engine(tmp_path / "tasks.sqlite3")
      sessions = create_session_factory(engine)
      try:
          with engine.begin() as connection:
              connection.exec_driver_sql("CREATE TABLE probe (value INTEGER NOT NULL)")
              connection.exec_driver_sql("INSERT INTO probe(value) VALUES (1)")

          with sessions.begin() as first_session:
              first_value = first_session.execute(text("SELECT value FROM probe")).scalar_one()
              with sessions.begin() as second_session:
                  second_session.execute(text("UPDATE probe SET value = 2"))
              repeated_value = first_session.execute(text("SELECT value FROM probe")).scalar_one()

          assert first_value == 1
          assert repeated_value == 1
      finally:
          engine.dispose()
  ```

- [ ] 4. 运行测试，确认因模块不存在而失败：

  ```bash
  ./.venv/bin/python -m pytest tests/web/test_database.py -q
  ```

  Expected: `ModuleNotFoundError: No module named 'paperpilot.web.database'`。

- [ ] 5. 新建 `paperpilot/web/database.py`，最小实现如下：

  ```python
  from __future__ import annotations

  import os
  from pathlib import Path

  from sqlalchemy import URL, Engine, event
  from sqlalchemy import create_engine
  from sqlalchemy.orm import Session, sessionmaker


  DEFAULT_TASK_DB_PATH = Path("data/web/tasks.sqlite3")


  def resolve_task_db_path(db_path: Path | str | None = None) -> Path:
      configured = db_path or os.getenv("PAPERPILOT_TASK_DB_PATH") or DEFAULT_TASK_DB_PATH
      return Path(configured).expanduser().resolve()


  def build_sqlite_url(db_path: Path) -> URL:
      return URL.create("sqlite+pysqlite", database=str(db_path))


  def create_task_engine(db_path: Path) -> Engine:
      resolved_path = resolve_task_db_path(db_path)
      resolved_path.parent.mkdir(parents=True, exist_ok=True)
      engine = create_engine(
          build_sqlite_url(resolved_path),
          connect_args={"timeout": 30},
      )

      @event.listens_for(engine, "connect")
      def configure_sqlite(dbapi_connection, _connection_record) -> None:
          dbapi_connection.isolation_level = None
          cursor = dbapi_connection.cursor()
          try:
              cursor.execute("PRAGMA journal_mode=WAL")
              cursor.execute("PRAGMA foreign_keys=ON")
              cursor.execute("PRAGMA busy_timeout=30000")
              cursor.execute("PRAGMA synchronous=NORMAL")
          finally:
              cursor.close()

      @event.listens_for(engine, "begin")
      def begin_sqlite_transaction(connection) -> None:
          connection.exec_driver_sql("BEGIN")

      return engine


  def create_session_factory(engine: Engine) -> sessionmaker[Session]:
      return sessionmaker(bind=engine, expire_on_commit=False)
  ```

- [ ] 6. 运行单测并确认全部通过：

  ```bash
  ./.venv/bin/python -m pytest tests/web/test_database.py -q
  ```

  Expected: `4 passed`。

- [ ] 7. 运行格式/静态基本检查：

  ```bash
  ./.venv/bin/python -m compileall -q paperpilot/web/database.py tests/web/test_database.py
  git diff --check
  ```

- [ ] 8. 提交本任务：

  ```bash
  git add requirements.txt paperpilot/web/database.py tests/web/test_database.py
  git commit -m "feat(db): add SQLAlchemy SQLite foundation"
  ```

---

## Task 2: Map the Existing Five Web Tables as ORM Rows

**Files:**

- Create: `paperpilot/web/db_models.py`
- Create: `tests/web/test_db_models.py`

### Rationale

ORM 类型只表达数据库行，不替代 `ResearchTask` 等业务返回 dataclass。时间仍存为 ISO-8601 `TEXT`，避免第一阶段出现时区、序列化或排序变化。这里不添加 relationship，查询继续显式 join，减少隐藏加载行为。

### Steps

- [ ] 1. 新建 `tests/web/test_db_models.py`，先固定五张表、关键列、索引和新库外键契约：

  ```python
  from paperpilot.web.db_models import Base


  def test_web_metadata_owns_only_the_five_existing_tables() -> None:
      assert set(Base.metadata.tables) == {
          "users",
          "sessions",
          "research_tasks",
          "task_events",
          "task_artifacts",
      }


  def test_research_task_columns_match_existing_text_schema() -> None:
      table = Base.metadata.tables["research_tasks"]
      assert list(table.columns) == [
          table.c.id,
          table.c.question,
          table.c.depth,
          table.c.status,
          table.c.created_at,
          table.c.updated_at,
          table.c.user_id,
      ]
      assert str(table.c.created_at.type) == "TEXT"
      assert str(table.c.updated_at.type) == "TEXT"


  def test_user_and_artifact_columns_match_existing_schema() -> None:
      users = Base.metadata.tables["users"]
      artifacts = Base.metadata.tables["task_artifacts"]
      assert [column.name for column in users.columns] == [
          "id",
          "username",
          "password_hash",
          "password_salt",
          "created_at",
      ]
      assert [column.name for column in artifacts.columns] == [
          "id",
          "task_id",
          "kind",
          "title",
          "content",
          "payload_json",
          "created_at",
      ]


  def test_metadata_declares_expected_indexes() -> None:
      index_names = {
          index.name
          for table in Base.metadata.sorted_tables
          for index in table.indexes
      }
      assert index_names == {
          "idx_tasks_user_created_id",
          "idx_tasks_user_status_created_id",
          "idx_events_task_id_id",
          "idx_artifacts_task_id_id",
      }


  def test_new_schema_declares_expected_foreign_keys_and_autoincrement() -> None:
      metadata = Base.metadata
      foreign_keys = {
          (foreign_key.parent.table.name, foreign_key.parent.name, foreign_key.target_fullname)
          for table in metadata.sorted_tables
          for foreign_key in table.foreign_keys
      }
      assert foreign_keys == {
          ("sessions", "user_id", "users.id"),
          ("research_tasks", "user_id", "users.id"),
          ("task_events", "task_id", "research_tasks.id"),
          ("task_artifacts", "task_id", "research_tasks.id"),
      }
      assert metadata.tables["task_events"].dialect_options["sqlite"]["autoincrement"]
      assert metadata.tables["task_artifacts"].dialect_options["sqlite"]["autoincrement"]
  ```

- [ ] 2. 运行测试，确认因模型模块不存在而失败：

  ```bash
  ./.venv/bin/python -m pytest tests/web/test_db_models.py -q
  ```

- [ ] 3. 新建 `paperpilot/web/db_models.py`：

  ```python
  from __future__ import annotations

  from sqlalchemy import ForeignKey, Index, Integer, Text
  from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


  class Base(DeclarativeBase):
      pass


  class UserRow(Base):
      __tablename__ = "users"

      id: Mapped[str] = mapped_column(Text, primary_key=True)
      username: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
      password_hash: Mapped[str] = mapped_column(Text, nullable=False)
      password_salt: Mapped[str] = mapped_column(Text, nullable=False)
      created_at: Mapped[str] = mapped_column(Text, nullable=False)


  class LoginSessionRow(Base):
      __tablename__ = "sessions"

      token: Mapped[str] = mapped_column(Text, primary_key=True)
      user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
      created_at: Mapped[str] = mapped_column(Text, nullable=False)
      expires_at: Mapped[str] = mapped_column(Text, nullable=False)


  class ResearchTaskRow(Base):
      __tablename__ = "research_tasks"

      id: Mapped[str] = mapped_column(Text, primary_key=True)
      question: Mapped[str] = mapped_column(Text, nullable=False)
      depth: Mapped[str] = mapped_column(Text, nullable=False)
      status: Mapped[str] = mapped_column(Text, nullable=False)
      created_at: Mapped[str] = mapped_column(Text, nullable=False)
      updated_at: Mapped[str] = mapped_column(Text, nullable=False)
      user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)


  class TaskEventRow(Base):
      __tablename__ = "task_events"
      __table_args__ = {"sqlite_autoincrement": True}

      id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
      task_id: Mapped[str] = mapped_column(ForeignKey("research_tasks.id"), nullable=False)
      type: Mapped[str] = mapped_column(Text, nullable=False)
      stage: Mapped[str | None] = mapped_column(Text, nullable=True)
      message: Mapped[str] = mapped_column(Text, nullable=False)
      payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
      created_at: Mapped[str] = mapped_column(Text, nullable=False)


  class TaskArtifactRow(Base):
      __tablename__ = "task_artifacts"
      __table_args__ = {"sqlite_autoincrement": True}

      id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
      task_id: Mapped[str] = mapped_column(ForeignKey("research_tasks.id"), nullable=False)
      kind: Mapped[str] = mapped_column(Text, nullable=False)
      title: Mapped[str] = mapped_column(Text, nullable=False)
      content: Mapped[str] = mapped_column(Text, nullable=False)
      payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
      created_at: Mapped[str] = mapped_column(Text, nullable=False)


  Index(
      "idx_tasks_user_created_id",
      ResearchTaskRow.user_id,
      ResearchTaskRow.created_at.desc(),
      ResearchTaskRow.id.desc(),
  )
  Index(
      "idx_tasks_user_status_created_id",
      ResearchTaskRow.user_id,
      ResearchTaskRow.status,
      ResearchTaskRow.created_at.desc(),
      ResearchTaskRow.id.desc(),
  )
  Index("idx_events_task_id_id", TaskEventRow.task_id, TaskEventRow.id)
  Index("idx_artifacts_task_id_id", TaskArtifactRow.task_id, TaskArtifactRow.id)
  ```

- [ ] 4. 运行并修正模型测试：

  ```bash
  ./.venv/bin/python -m pytest tests/web/test_db_models.py -q
  ```

  Expected: `5 passed`。

- [ ] 5. 确认 metadata 未意外包含 Agent 表：

  ```bash
  ./.venv/bin/python -c "from paperpilot.web.db_models import Base; print(sorted(Base.metadata.tables))"
  ```

- [ ] 6. 提交本任务：

  ```bash
  git add paperpilot/web/db_models.py tests/web/test_db_models.py
  git commit -m "feat(db): map existing web tables"
  ```

---

## Task 3: Introduce a Non-Destructive Alembic Adoption Migration

**Files:**

- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/script.py.mako`
- Create: `migrations/versions/20260806_0001_adopt_web_schema.py`
- Create: `paperpilot/web/db_migrations.py`
- Create: `tests/web/test_db_migrations.py`

### Rationale

现有用户数据库没有 `alembic_version`，并且可能处于三种状态：完全空库、当前五表结构、早期缺少 `user_id/stage/payload_json` 的结构。不能简单 autogenerate 一个“创建所有表”的 revision，否则当前库首次升级会因表已存在失败。初始 revision 必须检查并接管已有对象。

### Steps

- [ ] 1. 先创建 `tests/web/test_db_migrations.py`，覆盖四条不可妥协的迁移路径：

  ```python
  from __future__ import annotations

  import sqlite3
  from pathlib import Path

  import pytest
  from alembic import command

  from paperpilot.web.db_migrations import (
      build_alembic_config,
      ensure_database_current,
      get_database_heads,
      get_script_heads,
      upgrade_database,
  )


  MANAGED_TABLES = {
      "users",
      "sessions",
      "research_tasks",
      "task_events",
      "task_artifacts",
  }


  def table_names(db_path: Path) -> set[str]:
      with sqlite3.connect(db_path) as connection:
          return {
              row[0]
              for row in connection.execute(
                  "SELECT name FROM sqlite_master WHERE type = 'table'"
              )
          }


  def column_names(db_path: Path, table_name: str) -> set[str]:
      with sqlite3.connect(db_path) as connection:
          return {
              row[1]
              for row in connection.execute(f"PRAGMA table_info({table_name})")
          }


  def create_legacy_database(db_path: Path) -> None:
      with sqlite3.connect(db_path) as connection:
          connection.executescript(
              """
              CREATE TABLE research_tasks (
                  id TEXT PRIMARY KEY,
                  question TEXT NOT NULL,
                  depth TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
              );
              CREATE TABLE task_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  task_id TEXT NOT NULL,
                  type TEXT NOT NULL,
                  message TEXT NOT NULL,
                  created_at TEXT NOT NULL
              );
              CREATE TABLE agent_runs (
                  id TEXT PRIMARY KEY,
                  task_id TEXT NOT NULL,
                  status TEXT NOT NULL
              );
              INSERT INTO research_tasks VALUES (
                  'task-1', 'question', 'quick', 'pending',
                  '2026-08-06T00:00:00+00:00', '2026-08-06T00:00:00+00:00'
              );
              INSERT INTO task_events(task_id, type, message, created_at)
              VALUES ('task-1', 'queued', 'queued', '2026-08-06T00:00:00+00:00');
              INSERT INTO agent_runs VALUES ('run-1', 'task-1', 'running');
              """
          )


  def test_blank_database_upgrades_to_managed_schema(tmp_path: Path) -> None:
      db_path = tmp_path / "blank.sqlite3"

      upgrade_database(db_path)

      assert MANAGED_TABLES <= table_names(db_path)
      assert "alembic_version" in table_names(db_path)
      assert get_database_heads(db_path) == get_script_heads()


  def test_legacy_database_is_adopted_without_losing_rows_or_unknown_tables(
      tmp_path: Path,
  ) -> None:
      db_path = tmp_path / "legacy.sqlite3"
      create_legacy_database(db_path)

      upgrade_database(db_path)

      assert MANAGED_TABLES <= table_names(db_path)
      assert "agent_runs" in table_names(db_path)
      assert "user_id" in column_names(db_path, "research_tasks")
      assert {"stage", "payload_json"} <= column_names(db_path, "task_events")
      with sqlite3.connect(db_path) as connection:
          assert connection.execute(
              "SELECT question FROM research_tasks WHERE id = 'task-1'"
          ).fetchone() == ("question",)
          assert connection.execute(
              "SELECT status FROM agent_runs WHERE id = 'run-1'"
          ).fetchone() == ("running",)


  def test_upgrade_is_idempotent(tmp_path: Path) -> None:
      db_path = tmp_path / "tasks.sqlite3"
      ensure_database_current(db_path)
      first_heads = get_database_heads(db_path)

      ensure_database_current(db_path)

      assert get_database_heads(db_path) == first_heads == get_script_heads()


  def test_initial_revision_refuses_destructive_downgrade(tmp_path: Path) -> None:
      db_path = tmp_path / "tasks.sqlite3"
      upgrade_database(db_path)

      with pytest.raises(RuntimeError, match="downgrade is intentionally unsupported"):
          command.downgrade(build_alembic_config(db_path), "base")


  def test_adoption_revision_refuses_offline_sql(tmp_path: Path) -> None:
      config = build_alembic_config(tmp_path / "offline.sqlite3")

      with pytest.raises(RuntimeError, match="Offline SQL is unsupported"):
          command.upgrade(config, "head", sql=True)
  ```

- [ ] 2. 运行测试，确认因 `db_migrations` 不存在而失败：

  ```bash
  ./.venv/bin/python -m pytest tests/web/test_db_migrations.py -q
  ```

- [ ] 3. 新建 `alembic.ini`，只保留本项目需要的配置；数据库 URL 由运行时 helper 或环境变量覆盖：

  ```ini
  [alembic]
  script_location = %(here)s/migrations
  prepend_sys_path = .
  sqlalchemy.url = sqlite+pysqlite:///data/web/tasks.sqlite3

  [loggers]
  keys = root,sqlalchemy,alembic

  [handlers]
  keys = console

  [formatters]
  keys = generic

  [logger_root]
  level = WARN
  handlers = console
  qualname =

  [logger_sqlalchemy]
  level = WARN
  handlers =
  qualname = sqlalchemy.engine

  [logger_alembic]
  level = INFO
  handlers =
  qualname = alembic

  [handler_console]
  class = StreamHandler
  args = (sys.stderr,)
  level = NOTSET
  formatter = generic

  [formatter_generic]
  format = %(levelname)-5.5s [%(name)s] %(message)s
  datefmt = %H:%M:%S
  ```

- [ ] 4. 新建 `paperpilot/web/db_migrations.py`：

  ```python
  from __future__ import annotations

  from pathlib import Path
  from threading import Lock

  from alembic import command
  from alembic.config import Config
  from alembic.migration import MigrationContext
  from alembic.script import ScriptDirectory

  from paperpilot.web.database import (
      build_sqlite_url,
      create_task_engine,
      resolve_task_db_path,
  )


  PROJECT_ROOT = Path(__file__).resolve().parents[2]
  ALEMBIC_INI_PATH = PROJECT_ROOT / "alembic.ini"
  MIGRATIONS_PATH = PROJECT_ROOT / "migrations"
  _MIGRATION_LOCK = Lock()


  def build_alembic_config(db_path: Path | str) -> Config:
      resolved_path = resolve_task_db_path(db_path)
      config = Config(str(ALEMBIC_INI_PATH))
      config.set_main_option("script_location", str(MIGRATIONS_PATH))
      rendered_url = build_sqlite_url(resolved_path).render_as_string(
          hide_password=False
      )
      config.set_main_option("sqlalchemy.url", rendered_url.replace("%", "%%"))
      return config


  def get_database_heads(db_path: Path | str) -> set[str]:
      engine = create_task_engine(resolve_task_db_path(db_path))
      try:
          with engine.connect() as connection:
              context = MigrationContext.configure(connection)
              return set(context.get_current_heads())
      finally:
          engine.dispose()


  def get_script_heads() -> set[str]:
      config = Config(str(ALEMBIC_INI_PATH))
      config.set_main_option("script_location", str(MIGRATIONS_PATH))
      return set(ScriptDirectory.from_config(config).get_heads())


  def upgrade_database(db_path: Path | str, revision: str = "head") -> None:
      command.upgrade(build_alembic_config(db_path), revision)


  def ensure_database_current(db_path: Path | str) -> None:
      resolved_path = resolve_task_db_path(db_path)
      with _MIGRATION_LOCK:
          if get_database_heads(resolved_path) == get_script_heads():
              return
          upgrade_database(resolved_path)
  ```

- [ ] 5. 新建 `migrations/env.py`。配置必须支持程序化 Config 和 CLI，并忽略 metadata 之外的已反射表。初始 adoption revision 依赖实时 schema introspection，因此离线 `--sql` 必须明确拒绝，不能生成一份假定空库的危险 DDL：

  ```python
  from __future__ import annotations

  from logging.config import fileConfig
  from pathlib import Path

  from alembic import context
  from sqlalchemy import Connection, make_url

  from paperpilot.web.database import create_task_engine
  from paperpilot.web.db_models import Base


  config = context.config
  if config.config_file_name is not None:
      fileConfig(config.config_file_name)

  target_metadata = Base.metadata


  def include_object(object_, name, type_, reflected, compare_to):
      if type_ == "table" and reflected and name not in target_metadata.tables:
          return False
      return True


  def configure_context(connection: Connection) -> None:
      context.configure(
          connection=connection,
          target_metadata=target_metadata,
          compare_type=True,
          compare_server_default=False,
          render_as_batch=True,
          include_object=include_object,
      )


  def run_migrations_offline() -> None:
      raise RuntimeError(
          "Offline SQL is unsupported for the adoption migration because "
          "it must inspect the target SQLite schema"
      )
      with context.begin_transaction():
          context.run_migrations()


  def run_migrations_online() -> None:
      supplied_connection = config.attributes.get("connection")
      if supplied_connection is not None:
          configure_context(supplied_connection)
          with context.begin_transaction():
              context.run_migrations()
          return

      url = make_url(config.get_main_option("sqlalchemy.url"))
      if not url.database:
          raise RuntimeError("SQLite database path is required")
      engine = create_task_engine(Path(url.database))
      try:
          with engine.connect() as connection:
              configure_context(connection)
              with context.begin_transaction():
                  context.run_migrations()
      finally:
          engine.dispose()


  if context.is_offline_mode():
      run_migrations_offline()
  else:
      run_migrations_online()
  ```

- [ ] 6. 创建 `migrations/script.py.mako`。该文件是 Alembic 生成未来 revision 时使用的标准模板，不包含项目业务逻辑：

  ```mako
  """${message}

  Revision ID: ${up_revision}
  Revises: ${down_revision | comma,n}
  Create Date: ${create_date}
  """
  from typing import Sequence, Union

  from alembic import op
  import sqlalchemy as sa
  ${imports if imports else ""}

  revision: str = ${repr(up_revision)}
  down_revision: Union[str, Sequence[str], None] = ${repr(down_revision)}
  branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
  depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


  def upgrade() -> None:
      ${upgrades if upgrades else "pass"}


  def downgrade() -> None:
      ${downgrades if downgrades else "pass"}
  ```

- [ ] 7. 新建 `migrations/versions/20260806_0001_adopt_web_schema.py`。实现顺序固定为：用户表 -> 登录 session -> 任务表 -> 事件表 -> 产物表 -> 索引。核心结构如下：

  ```python
  """Adopt the existing PaperPilot Web schema without destructive rewrites."""

  from typing import Sequence, Union

  from alembic import op
  import sqlalchemy as sa


  revision: str = "20260806_0001"
  down_revision: Union[str, Sequence[str], None] = None
  branch_labels: Union[str, Sequence[str], None] = None
  depends_on: Union[str, Sequence[str], None] = None


  def _table_names(bind) -> set[str]:
      return set(sa.inspect(bind).get_table_names())


  def _column_names(bind, table_name: str) -> set[str]:
      return {column["name"] for column in sa.inspect(bind).get_columns(table_name)}


  def _index_names(bind, table_name: str) -> set[str]:
      return {index["name"] for index in sa.inspect(bind).get_indexes(table_name)}


  def _create_index_if_missing(
      bind, table_name: str, index_name: str, ddl: str
  ) -> None:
      if index_name not in _index_names(bind, table_name):
          op.execute(sa.text(ddl))


  def upgrade() -> None:
      bind = op.get_bind()
      tables = _table_names(bind)

      if "users" not in tables:
          op.create_table(
              "users",
              sa.Column("id", sa.Text(), primary_key=True),
              sa.Column("username", sa.Text(), nullable=False, unique=True),
              sa.Column("password_hash", sa.Text(), nullable=False),
              sa.Column("password_salt", sa.Text(), nullable=False),
              sa.Column("created_at", sa.Text(), nullable=False),
          )

      tables = _table_names(bind)
      if "sessions" not in tables:
          op.create_table(
              "sessions",
              sa.Column("token", sa.Text(), primary_key=True),
              sa.Column("user_id", sa.Text(), sa.ForeignKey("users.id"), nullable=False),
              sa.Column("created_at", sa.Text(), nullable=False),
              sa.Column("expires_at", sa.Text(), nullable=False),
          )

      tables = _table_names(bind)
      if "research_tasks" not in tables:
          op.create_table(
              "research_tasks",
              sa.Column("id", sa.Text(), primary_key=True),
              sa.Column("question", sa.Text(), nullable=False),
              sa.Column("depth", sa.Text(), nullable=False),
              sa.Column("status", sa.Text(), nullable=False),
              sa.Column("created_at", sa.Text(), nullable=False),
              sa.Column("updated_at", sa.Text(), nullable=False),
              sa.Column("user_id", sa.Text(), sa.ForeignKey("users.id"), nullable=True),
          )
      elif "user_id" not in _column_names(bind, "research_tasks"):
          # SQLite cannot add this FK without rebuilding the historical table.
          # Preserve data and add only the nullable column during adoption.
          op.add_column(
              "research_tasks", sa.Column("user_id", sa.Text(), nullable=True)
          )

      tables = _table_names(bind)
      if "task_events" not in tables:
          op.create_table(
              "task_events",
              sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
              sa.Column(
                  "task_id",
                  sa.Text(),
                  sa.ForeignKey("research_tasks.id"),
                  nullable=False,
              ),
              sa.Column("type", sa.Text(), nullable=False),
              sa.Column("stage", sa.Text(), nullable=True),
              sa.Column("message", sa.Text(), nullable=False),
              sa.Column("payload_json", sa.Text(), nullable=True),
              sa.Column("created_at", sa.Text(), nullable=False),
              sqlite_autoincrement=True,
          )
      else:
          event_columns = _column_names(bind, "task_events")
          if "stage" not in event_columns:
              op.add_column("task_events", sa.Column("stage", sa.Text(), nullable=True))
          if "payload_json" not in event_columns:
              op.add_column(
                  "task_events",
                  sa.Column("payload_json", sa.Text(), nullable=True),
              )

      tables = _table_names(bind)
      if "task_artifacts" not in tables:
          op.create_table(
              "task_artifacts",
              sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
              sa.Column(
                  "task_id",
                  sa.Text(),
                  sa.ForeignKey("research_tasks.id"),
                  nullable=False,
              ),
              sa.Column("kind", sa.Text(), nullable=False),
              sa.Column("title", sa.Text(), nullable=False),
              sa.Column("content", sa.Text(), nullable=False),
              sa.Column("payload_json", sa.Text(), nullable=True),
              sa.Column("created_at", sa.Text(), nullable=False),
              sqlite_autoincrement=True,
          )

      _create_index_if_missing(
          bind,
          "research_tasks",
          "idx_tasks_user_created_id",
          "CREATE INDEX idx_tasks_user_created_id "
          "ON research_tasks(user_id, created_at DESC, id DESC)",
      )
      _create_index_if_missing(
          bind,
          "research_tasks",
          "idx_tasks_user_status_created_id",
          "CREATE INDEX idx_tasks_user_status_created_id "
          "ON research_tasks(user_id, status, created_at DESC, id DESC)",
      )
      _create_index_if_missing(
          bind,
          "task_events",
          "idx_events_task_id_id",
          "CREATE INDEX idx_events_task_id_id ON task_events(task_id, id)",
      )
      _create_index_if_missing(
          bind,
          "task_artifacts",
          "idx_artifacts_task_id_id",
          "CREATE INDEX idx_artifacts_task_id_id ON task_artifacts(task_id, id)",
      )


  def downgrade() -> None:
      raise RuntimeError(
          "PaperPilot schema downgrade is intentionally unsupported; "
          "restore a verified database backup instead"
      )
  ```

- [ ] 8. 运行迁移测试：

  ```bash
  ./.venv/bin/python -m pytest tests/web/test_db_migrations.py -q
  ```

  Expected: `5 passed`。

- [ ] 9. 对一个全新的临时库执行 CLI 升级和 schema drift 检查：

  ```bash
  PAPERPILOT_ALEMBIC_SMOKE_DIR="$(mktemp -d)"
  PAPERPILOT_TASK_DB_PATH="$PAPERPILOT_ALEMBIC_SMOKE_DIR/tasks.sqlite3" \
    ./.venv/bin/python -m alembic -c alembic.ini upgrade head
  PAPERPILOT_TASK_DB_PATH="$PAPERPILOT_ALEMBIC_SMOKE_DIR/tasks.sqlite3" \
    ./.venv/bin/python -m alembic -c alembic.ini current
  PAPERPILOT_TASK_DB_PATH="$PAPERPILOT_ALEMBIC_SMOKE_DIR/tasks.sqlite3" \
    ./.venv/bin/python -m alembic -c alembic.ini check
  ```

  Expected: `current` 显示 `20260806_0001 (head)`，`check` 显示没有新的 upgrade operations。

- [ ] 10. 提交本任务：

  ```bash
  git add \
    alembic.ini \
    migrations \
    paperpilot/web/db_migrations.py \
    tests/web/test_db_migrations.py
  git commit -m "feat(db): adopt existing schema with Alembic"
  ```

---

## Task 4: Move User and Login Session Persistence to SQLAlchemy

**Files:**

- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/web/test_task_store.py`
- Verify: `tests/web/test_auth.py`

### Rationale

先迁移边界最清晰的用户和登录 session，再迁移任务状态机数据。为了让每个提交可运行，本任务暂时保留 `_connect()` 给尚未迁移的方法使用，但删除手写 schema 创建/补列，把 schema 所有权交给 Alembic。

### Steps

- [ ] 1. 在 `tests/web/test_task_store.py` 增加/调整测试，先固定以下行为：

  ```python
  def test_constructor_upgrades_a_blank_database(tmp_path: Path) -> None:
      store = TaskStore(tmp_path / "tasks.sqlite3")
      try:
          assert store.check_health() is None
      finally:
          store.close()


  def test_user_and_session_persist_across_store_instances(tmp_path: Path) -> None:
      db_path = tmp_path / "tasks.sqlite3"
      first = TaskStore(db_path)
      user = first.create_user(
          username="reader",
          password_hash="hash",
          password_salt="salt",
      )
      token = first.create_session(user.id)
      first.close()

      second = TaskStore(db_path)
      try:
          assert second.get_user_by_username("reader") == user
          assert second.get_user_for_session(token) == user
      finally:
          second.close()
  ```

- [ ] 2. 先运行新增测试，确认 `TaskStore.close()` 尚不存在：

  ```bash
  ./.venv/bin/python -m pytest \
    tests/web/test_task_store.py -k "constructor_upgrades or user_and_session_persist" -q
  ```

- [ ] 3. 修改 `TaskStore.__init__`，保留现有 `db_path` 参数语义：

  ```python
  class TaskStore:
      def __init__(self, db_path: Path | str | None = None) -> None:
          self.db_path = resolve_task_db_path(db_path)
          ensure_database_current(self.db_path)
          self.engine = create_task_engine(self.db_path)
          self._session_factory = create_session_factory(self.engine)

      def close(self) -> None:
          self.engine.dispose()
  ```

  删除构造器对 `_ensure_schema()` 的调用，并删除 `_ensure_schema()`、`_migrate_*()` 等手写 DDL 逻辑。此时暂时保留只服务于未迁移方法的 `_connect()`。

- [ ] 4. 添加明确的 ORM row -> 现有 dataclass 映射函数：

  ```python
  @staticmethod
  def _user_from_model(row: UserRow) -> WebUser:
      return WebUser(
          id=row.id,
          username=row.username,
          password_hash=row.password_hash,
          password_salt=row.password_salt,
          created_at=row.created_at,
      )
  ```

- [ ] 5. 用 SQLAlchemy 2.0 statement 改写六个用户/session 方法。实现应遵循以下形式：

  ```python
  def create_user(
      self,
      *,
      username: str,
      password_hash: str,
      password_salt: str,
  ) -> WebUser:
      username = username.strip()
      if not username:
          raise ValueError("username is required")
      row = UserRow(
          id=f"user_{uuid.uuid4().hex}",
          username=username,
          password_hash=password_hash,
          password_salt=password_salt,
          created_at=_utc_now(),
      )
      try:
          with self._session_factory.begin() as session:
              session.add(row)
      except IntegrityError as exc:
          raise DuplicateUsernameError("username already exists") from exc
      return self._user_from_model(row)

  def get_user_by_username(self, username: str) -> WebUser | None:
      with self._session_factory() as session:
          row = session.scalar(
              select(UserRow).where(UserRow.username == username)
          )
      return self._user_from_model(row) if row is not None else None

  def get_user_by_id(self, user_id: str) -> WebUser | None:
      with self._session_factory() as session:
          row = session.get(UserRow, user_id)
      return self._user_from_model(row) if row is not None else None

  def create_session(self, user_id: str) -> str:
      if self.get_user_by_id(user_id) is None:
          raise ValueError(f"user not found: {user_id}")
      token = f"session_{secrets.token_urlsafe(32)}"
      with self._session_factory.begin() as session:
          session.add(
              LoginSessionRow(
                  token=token,
                  user_id=user_id,
                  created_at=_utc_now(),
                  expires_at=_utc_in(days=7),
              )
          )
      return token

  def get_user_for_session(self, token: str) -> WebUser | None:
      now = _utc_now()
      statement = (
          select(UserRow)
          .join(LoginSessionRow, LoginSessionRow.user_id == UserRow.id)
          .where(
              LoginSessionRow.token == token,
              LoginSessionRow.expires_at > now,
          )
      )
      with self._session_factory() as session:
          row = session.scalar(statement)
      return self._user_from_model(row) if row is not None else None

  def delete_session(self, token: str) -> None:
      with self._session_factory.begin() as session:
          session.execute(
              delete(LoginSessionRow).where(LoginSessionRow.token == token)
          )
  ```

- [ ] 6. 运行用户/session 与认证测试：

  ```bash
  ./.venv/bin/python -m pytest \
    tests/web/test_task_store.py -k "user or session or environment_path or persistence" \
    tests/web/test_auth.py -q
  ```

- [ ] 7. 运行整个 `TaskStore` 测试，确认混合迁移期间未破坏剩余 sqlite3 方法：

  ```bash
  ./.venv/bin/python -m pytest tests/web/test_task_store.py -q
  ```

- [ ] 8. 提交本任务：

  ```bash
  git add paperpilot/web/task_store.py tests/web/test_task_store.py
  git commit -m "refactor(db): move auth persistence to SQLAlchemy"
  ```

---

## Task 5: Move Research Task State and Pagination to SQLAlchemy

**Files:**

- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/web/test_task_store.py`

### Rationale

任务数据包含本阶段风险最高的行为：稳定 keyset pagination、queued task + initial event 原子写入、并发 claim 只能成功一次、pending -> failed 条件更新。必须复用现有测试并明确检查 SQLAlchemy 的事务结果，不能只做等价 CRUD。

### Steps

- [ ] 1. 保留现有分页、状态转换和双线程 claim 测试；把仅依赖 `sqlite3.IntegrityError` 的内部异常断言调整为 `sqlalchemy.exc.IntegrityError`。新增一个 SQLAlchemy session 回滚后任务也不存在的断言：

  ```python
  def test_create_queued_task_rolls_back_task_when_initial_event_fails(
      tmp_path: Path,
  ) -> None:
      store = TaskStore(tmp_path / "tasks.sqlite3")
      user = store.create_user("reader", "hash")
      with store.engine.begin() as connection:
          connection.exec_driver_sql(
              """
              CREATE TRIGGER reject_queued_event
              BEFORE INSERT ON task_events
              WHEN NEW.type = 'queued'
              BEGIN
                  SELECT RAISE(ABORT, 'reject queued event');
              END
              """
          )

      with pytest.raises(IntegrityError):
          store.create_queued_task(
              question="question",
              depth="quick",
              user_id=user.id,
              execution_mode="simulated",
          )

      with store._session_factory() as session:
          count = session.scalar(
              select(func.count()).select_from(ResearchTaskRow)
          )
      assert count == 0
  ```

- [ ] 2. 运行目标测试，确认它仍经过旧 sqlite3 路径或异常类型不匹配：

  ```bash
  ./.venv/bin/python -m pytest tests/web/test_task_store.py \
    -k "queued or page or claim or fail_pending or update_status" -q
  ```

- [ ] 3. 添加 ORM task -> `ResearchTask` 映射：

  ```python
  @staticmethod
  def _task_to_model(task: ResearchTask) -> ResearchTaskRow:
      return ResearchTaskRow(
          id=task.id,
          question=task.question,
          depth=task.depth,
          status=task.status,
          created_at=task.created_at,
          updated_at=task.updated_at,
          user_id=task.user_id,
      )

  @staticmethod
  def _task_from_model(row: ResearchTaskRow) -> ResearchTask:
      return ResearchTask(
          id=row.id,
          question=row.question,
          depth=row.depth,
          status=row.status,
          created_at=row.created_at,
          updated_at=row.updated_at,
          user_id=row.user_id,
      )
  ```

- [ ] 4. 改写 `create_task()` 和 `create_queued_task()`；后者必须在同一个 `Session.begin()` 中插入任务和 queued event：

  ```python
  def create_task(
      self,
      *,
      question: str,
      depth: str = "standard",
      user_id: str | None = None,
  ) -> ResearchTask:
      task = _new_task(question=question, depth=depth, user_id=user_id)
      with self._session_factory.begin() as session:
          session.add(self._task_to_model(task))
      return task


  def create_queued_task(
      self,
      *,
      question: str,
      depth: str,
      user_id: str,
      execution_mode: str,
  ) -> ResearchTask:
      if execution_mode not in {"simulated", "real"}:
          raise ValueError(f"invalid execution mode: {execution_mode!r}")
      task = _new_task(question=question, depth=depth, user_id=user_id)
      payload = {
          "depth": task.depth,
          "execution_mode": execution_mode,
          "simulated": execution_mode == "simulated",
      }
      task_row = self._task_to_model(task)
      with self._session_factory.begin() as session:
          session.add(task_row)
          session.flush()
          session.add(
              TaskEventRow(
                  task_id=task_row.id,
                  type="queued",
                  stage="queue",
                  message=f"Task queued for {execution_mode} workflow.",
                  payload_json=json.dumps(payload, ensure_ascii=False),
                  created_at=_utc_now(),
              )
          )
      return self._task_from_model(task_row)
  ```

- [ ] 5. 用 `select()` 改写 `get_task()` 和 `list_tasks_page()`。keyset 条件必须保持 `(created_at, id)` 倒序语义：

  ```python
  def get_task(
      self,
      task_id: str,
      *,
      user_id: str | None = None,
  ) -> ResearchTask | None:
      filters = [ResearchTaskRow.id == task_id]
      if user_id is not None:
          filters.append(ResearchTaskRow.user_id == user_id)
      with self._session_factory() as session:
          row = session.scalar(select(ResearchTaskRow).where(*filters))
      return self._task_from_model(row) if row is not None else None


  def list_tasks_page(
      self,
      *,
      user_id: str | None,
      limit: int,
      status: str | None = None,
      before_created_at: str | None = None,
      before_id: str | None = None,
  ) -> TaskPage:
      if status is not None and status not in VALID_STATUSES:
          raise ValueError(f"invalid status: {status!r}")
      if limit < 1 or limit > 100:
          raise ValueError("limit must be between 1 and 100")
      if (before_created_at is None) != (before_id is None):
          raise ValueError("task page position requires created_at and id")

      filters = [
          ResearchTaskRow.user_id.is_(None)
          if user_id is None
          else ResearchTaskRow.user_id == user_id
      ]
      if status is not None:
          filters.append(ResearchTaskRow.status == status)
      if before_created_at is not None:
          assert before_id is not None
          filters.append(
              or_(
                  ResearchTaskRow.created_at < before_created_at,
                  and_(
                      ResearchTaskRow.created_at == before_created_at,
                      ResearchTaskRow.id < before_id,
                  ),
              )
          )
      statement = (
          select(ResearchTaskRow)
          .where(*filters)
          .order_by(ResearchTaskRow.created_at.desc(), ResearchTaskRow.id.desc())
          .limit(limit + 1)
      )
      with self._session_factory() as session:
          rows = list(session.scalars(statement))
      tasks = [self._task_from_model(row) for row in rows]
      return TaskPage(items=tasks[:limit], has_more=len(tasks) > limit)
  ```

  `TaskPage` 仍只返回 `items + has_more`，不要新增 cursor 字段。

- [ ] 6. 用带旧状态条件的单条 `UPDATE` 改写状态方法。`claim_task()` 不得先 SELECT 再无条件 UPDATE：

  ```python
  def claim_task(
      self,
      task_id: str,
      *,
      allow_running: bool = False,
  ) -> ResearchTask | None:
      claimable_statuses = (
          ("pending", "running") if allow_running else ("pending",)
      )
      now = _utc_now()
      with self._session_factory.begin() as session:
          result = session.execute(
              update(ResearchTaskRow)
              .where(
                  ResearchTaskRow.id == task_id,
                  ResearchTaskRow.status.in_(claimable_statuses),
              )
              .values(status="running", updated_at=now)
          )
          if result.rowcount != 1:
              return None
          row = session.get(ResearchTaskRow, task_id)
          if row is None:
              raise RuntimeError("claimed task disappeared within transaction")
          return self._task_from_model(row)


  def update_status(self, task_id: str, status: str) -> ResearchTask | None:
      if status not in VALID_STATUSES:
          raise ValueError(f"invalid status: {status!r}")
      with self._session_factory.begin() as session:
          result = session.execute(
              update(ResearchTaskRow)
              .where(ResearchTaskRow.id == task_id)
              .values(status=status, updated_at=_utc_now())
          )
          if result.rowcount != 1:
              return None
      return self.get_task(task_id)


  def fail_pending_task(self, task_id: str) -> ResearchTask | None:
      with self._session_factory.begin() as session:
          result = session.execute(
              update(ResearchTaskRow)
              .where(
                  ResearchTaskRow.id == task_id,
                  ResearchTaskRow.status == "pending",
              )
              .values(status="failed", updated_at=_utc_now())
          )
          if result.rowcount != 1:
              return None
          row = session.get(ResearchTaskRow, task_id)
          if row is None:
              raise RuntimeError("failed task disappeared within transaction")
          return self._task_from_model(row)
  ```

  `update_status()` 与 `fail_pending_task()` 采用相同的条件更新方式，并保留旧实现中的状态白名单、`allow_running` redelivery 语义和异常类型。现有这三个状态方法都只按 `task_id` 工作，不要擅自增加 `user_id` 参数；ownership 仍由调用它们的 Web/worker 路径保证。

- [ ] 7. 运行任务层全部针对性测试：

  ```bash
  ./.venv/bin/python -m pytest tests/web/test_task_store.py \
    -k "task or page or status or claim or queued or pending" -q
  ```

- [ ] 8. 重复运行并发 claim 测试至少 10 次，排除偶现双成功：

  ```bash
  for PAPERPILOT_TEST_RUN in {1..10}; do
    ./.venv/bin/python -m pytest tests/web/test_task_store.py \
      -k "atomic_claim" -q || break
  done
  ```

- [ ] 9. 提交本任务：

  ```bash
  git add paperpilot/web/task_store.py tests/web/test_task_store.py
  git commit -m "refactor(db): move task state to SQLAlchemy"
  ```

---

## Task 6: Move Events, Artifacts, Health, and Snapshot Reads to SQLAlchemy

**Files:**

- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/web/test_task_store.py`

### Rationale

完成剩余数据访问后才能彻底删除 `sqlite3` 连接和 row mapping。`get_task_updates()` 必须像当前实现一样，在同一个显式读事务里依次读取 owned task、event batch 和 artifact batch；Task 1 的 Engine `BEGIN` hook 用来保证三次查询看到同一 SQLite 快照。

### Steps

- [ ] 1. 保留并加强现有 snapshot 测试：在第一次 event 查询和 artifact 查询之间由第二连接插入新数据，断言本次 `TaskUpdates` 不包含新数据、下一次调用才包含。

- [ ] 2. 将 PRAGMA 和 query-plan 测试从私有 `_connect()` 改为 SQLAlchemy Engine：

  ```python
  with store.engine.connect() as connection:
      journal_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()
      plan = connection.exec_driver_sql(
          "EXPLAIN QUERY PLAN "
          "SELECT id FROM research_tasks "
          "WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
          (user_id, 20),
      ).all()
  assert journal_mode == "wal"
  assert any("idx_tasks_user_created_id" in row[-1] for row in plan)
  ```

- [ ] 3. 添加 event/artifact 两个 ORM row 映射，JSON 的序列化/反序列化继续使用当前 `_decode_payload()` 和错误策略：

  ```python
  def _event_from_model(row: TaskEventRow) -> TaskEvent:
      return TaskEvent(
          id=row.id,
          task_id=row.task_id,
          type=row.type,
          stage=row.stage,
          message=row.message,
          payload=_decode_payload(row.payload_json),
          created_at=row.created_at,
      )


  def _artifact_from_model(row: TaskArtifactRow) -> TaskArtifact:
      return TaskArtifact(
          id=row.id,
          task_id=row.task_id,
          kind=row.kind,
          title=row.title,
          content=row.content,
          payload=_decode_payload(row.payload_json),
          created_at=row.created_at,
      )
  ```

- [ ] 4. 改写 `add_event/list_events_page` 和 `add_artifact/list_artifacts_page`。分页继续使用自增 `id`，查询 `limit + 1` 后返回现有 `next_after_id`，不改变升序约定：

  ```python
  def _select_owned_task_model(
      session: Session,
      task_id: str,
      user_id: str | None,
  ) -> ResearchTaskRow | None:
      filters = [ResearchTaskRow.id == task_id]
      if user_id is not None:
          filters.append(ResearchTaskRow.user_id == user_id)
      return session.scalar(select(ResearchTaskRow).where(*filters))


  def _read_event_batch(
      session: Session,
      task_id: str,
      *,
      after_id: int,
      limit: int,
  ) -> TaskEventBatch:
      rows = list(
          session.scalars(
              select(TaskEventRow)
              .where(
                  TaskEventRow.task_id == task_id,
                  TaskEventRow.id > after_id,
              )
              .order_by(TaskEventRow.id.asc())
              .limit(limit + 1)
          )
      )
      items = [_event_from_model(row) for row in rows[:limit]]
      return TaskEventBatch(
          items=items,
          next_after_id=items[-1].id if items else after_id,
          has_more=len(rows) > limit,
      )


  def _read_artifact_batch(
      session: Session,
      task_id: str,
      *,
      after_id: int,
      limit: int,
  ) -> TaskArtifactBatch:
      rows = list(
          session.scalars(
              select(TaskArtifactRow)
              .where(
                  TaskArtifactRow.task_id == task_id,
                  TaskArtifactRow.id > after_id,
              )
              .order_by(TaskArtifactRow.id.asc())
              .limit(limit + 1)
          )
      )
      items = [_artifact_from_model(row) for row in rows[:limit]]
      return TaskArtifactBatch(
          items=items,
          next_after_id=items[-1].id if items else after_id,
          has_more=len(rows) > limit,
      )


  def add_event(
      self,
      *,
      task_id: str,
      type: str,
      message: str,
      stage: str | None = None,
      payload: dict | None = None,
  ) -> TaskEvent:
      if not message.strip():
          raise ValueError("event message is required")
      row = TaskEventRow(
          task_id=task_id,
          type=type,
          stage=stage,
          message=message.strip(),
          payload_json=json.dumps(payload or {}, ensure_ascii=False),
          created_at=_utc_now(),
      )
      with self._session_factory.begin() as session:
          if session.get(ResearchTaskRow, task_id) is None:
              raise ValueError(f"task not found: {task_id}")
          session.add(row)
          session.flush()
      return _event_from_model(row)


  def list_events_page(
      self,
      task_id: str,
      *,
      user_id: str | None,
      after_id: int,
      limit: int,
  ) -> TaskEventBatch | None:
      _validate_incremental_page(after_id, limit)
      with self._session_factory() as session:
          if _select_owned_task_model(session, task_id, user_id) is None:
              return None
          return _read_event_batch(
              session,
              task_id,
              after_id=after_id,
              limit=limit,
          )
  ```

  同一步实现 artifact 方法，逐项保留当前校验与规范化：

  ```python
  def add_artifact(
      self,
      *,
      task_id: str,
      kind: str,
      title: str,
      content: str,
      payload: dict | None = None,
  ) -> TaskArtifact:
      if not kind.strip():
          raise ValueError("artifact kind is required")
      if not title.strip():
          raise ValueError("artifact title is required")
      if not content.strip():
          raise ValueError("artifact content is required")
      row = TaskArtifactRow(
          task_id=task_id,
          kind=kind.strip(),
          title=title.strip(),
          content=content.strip(),
          payload_json=json.dumps(payload or {}, ensure_ascii=False),
          created_at=_utc_now(),
      )
      with self._session_factory.begin() as session:
          if session.get(ResearchTaskRow, task_id) is None:
              raise ValueError(f"task not found: {task_id}")
          session.add(row)
          session.flush()
      return _artifact_from_model(row)


  def list_artifacts_page(
      self,
      task_id: str,
      *,
      user_id: str | None,
      after_id: int,
      limit: int,
  ) -> TaskArtifactBatch | None:
      _validate_incremental_page(after_id, limit)
      with self._session_factory() as session:
          if _select_owned_task_model(session, task_id, user_id) is None:
              return None
          return _read_artifact_batch(
              session,
              task_id,
              after_id=after_id,
              limit=limit,
          )
  ```

- [ ] 5. 用一个 `Session.begin()` 实现 `get_task_updates()`：

  ```python
  def get_task_updates(
      self,
      task_id: str,
      *,
      user_id: str | None,
      after_event_id: int,
      after_artifact_id: int,
      limit: int,
  ) -> TaskUpdates | None:
      _validate_incremental_page(after_event_id, limit)
      _validate_incremental_page(after_artifact_id, limit)
      with self._session_factory.begin() as session:
          task_filters = [ResearchTaskRow.id == task_id]
          if user_id is not None:
              task_filters.append(ResearchTaskRow.user_id == user_id)
          task_row = session.scalar(select(ResearchTaskRow).where(*task_filters))
          if task_row is None:
              return None
          return TaskUpdates(
              task=self._task_from_model(task_row),
              events=_read_event_batch(
                  session,
                  task_id,
                  after_id=after_event_id,
                  limit=limit,
              ),
              artifacts=_read_artifact_batch(
                  session,
                  task_id,
                  after_id=after_artifact_id,
                  limit=limit,
              ),
          )
  ```

  `_read_event_batch()` 和 `_read_artifact_batch()` 只把参数从 `sqlite3.Connection` 改成 SQLAlchemy `Session`，继续分别构造当前 `TaskEventBatch/TaskArtifactBatch`，不要新建通用分页框架。

- [ ] 6. 用最轻量 SQL 改写健康检查，并保留异常向上传播：

  ```python
  def check_health(self) -> None:
      with self.engine.connect() as connection:
          row = connection.execute(text("SELECT 1")).one_or_none()
      if row is None or int(row[0]) != 1:
          raise RuntimeError("SQLite health probe returned an invalid result")
  ```

  相应测试应断言成功返回 `None`，并把底层错误断言改为 SQLAlchemy 的 `DBAPIError/OperationalError`。SQL trace 只要求应用查询仍是 `SELECT 1`；Engine 自身的 `BEGIN/ROLLBACK` 不算额外健康检查语句。

- [ ] 7. 删除 `task_store.py` 中已经无调用者的内容：

  - `sqlite3` import；
  - `_connect()`；
  - 所有 sqlite `Row` 映射；
  - 手写 `CREATE TABLE/ALTER TABLE/CREATE INDEX`；
  - 为旧连接上下文服务的私有事务 helper。

  删除前用 `rg` 确认无调用者：

  ```bash
  rg "_connect|_ensure_schema|sqlite3\.Row|ALTER TABLE|CREATE TABLE" \
    paperpilot tests
  ```

- [ ] 8. 运行完整数据层测试：

  ```bash
  ./.venv/bin/python -m pytest \
    tests/web/test_database.py \
    tests/web/test_db_models.py \
    tests/web/test_db_migrations.py \
    tests/web/test_task_store.py -q
  ```

- [ ] 9. 再运行认证和 Agent 共享数据库测试：

  ```bash
  ./.venv/bin/python -m pytest \
    tests/web/test_auth.py \
    tests/agent/test_store.py -q
  ```

- [ ] 10. 提交本任务：

  ```bash
  git add paperpilot/web/task_store.py tests/web/test_task_store.py
  git commit -m "refactor(db): complete SQLAlchemy task store"
  ```

---

## Task 7: Make Engine Ownership Explicit in Web and Worker Lifecycles

**Files:**

- Modify: `paperpilot/web/app.py`
- Modify: `paperpilot/web/worker_tasks.py`
- Modify: `tests/web/test_web_app.py`
- Modify: `tests/web/test_celery_worker.py`

### Rationale

Web 进程拥有默认构造的 `TaskStore` Engine，Celery 每次任务也会构造一个 store。SQLAlchemy 连接会归还连接池，但 Engine 仍应在所有者生命周期结束时显式 `dispose()`。调用者注入给 `create_app(task_store=...)` 的 store 不应由 app 擅自关闭。

### Steps

- [ ] 1. 为 Web 生命周期写失败测试，使用记录 `close()` 调用次数的 store：

  ```python
  class TrackingTaskStore(TaskStore):
      def __init__(self, db_path: Path) -> None:
          super().__init__(db_path)
          self.close_calls = 0

      def close(self) -> None:
          self.close_calls += 1
          super().close()


  def test_app_shutdown_closes_only_the_store_it_owns(monkeypatch, tmp_path: Path) -> None:
      owned_store = TrackingTaskStore(tmp_path / "owned.sqlite3")
      monkeypatch.setattr("paperpilot.web.app.TaskStore", lambda: owned_store)
      app = create_app()

      with TestClient(app):
          pass

      assert owned_store.close_calls == 1


  def test_app_does_not_close_injected_store(tmp_path: Path) -> None:
      injected_store = TrackingTaskStore(tmp_path / "injected.sqlite3")
      app = create_app(task_store=injected_store)

      with TestClient(app):
          pass

      assert injected_store.close_calls == 0
      injected_store.close()
  ```

- [ ] 2. 在 `create_app()` 构造处记录所有权，并把 executor shutdown 和 store close 放在同一个有序 shutdown handler：

  ```python
  owns_task_store = task_store is None
  store = task_store or TaskStore()

  def shutdown_resources() -> None:
      executor.shutdown()
      if owns_task_store:
          store.close()

  if hasattr(app, "add_event_handler"):
      app.add_event_handler("shutdown", shutdown_resources)
  else:
      app.router.add_event_handler("shutdown", shutdown_resources)
  ```

- [ ] 3. 为 Celery 执行路径写失败测试：无论 runner 成功还是抛错，store 的 `close()` 都被调用一次：

  ```python
  def test_worker_closes_store_after_success(tmp_path, monkeypatch) -> None:
      store = TrackingTaskStore(tmp_path / "success.sqlite3")
      task = store.create_task(question="simulation")
      monkeypatch.setattr(worker_tasks, "_store_factory", lambda: store)

      worker_tasks._execute_research_task(task.id, "simulated")

      assert store.close_calls == 1


  def test_worker_closes_store_when_runner_fails(tmp_path, monkeypatch) -> None:
      store = TrackingTaskStore(tmp_path / "failure.sqlite3")
      task = store.create_task(question="simulation")

      class FailingRunner:
          def __init__(self, task_store) -> None:
              assert task_store is store

          def run_simulated(self, task_id: str) -> None:
              raise RuntimeError(f"runner failed for {task_id}")

      monkeypatch.setattr(worker_tasks, "_store_factory", lambda: store)
      monkeypatch.setattr(worker_tasks, "WorkflowRunner", FailingRunner)

      with pytest.raises(RuntimeError, match="runner failed"):
          worker_tasks._execute_research_task(task.id, "simulated")

      assert store.close_calls == 1
  ```

  `tests/web/test_celery_worker.py` 可定义自己的 `TrackingTaskStore`，或从同一测试 helper 导入；不要让生产代码依赖测试类型。

- [ ] 4. 在 worker task 的最外层 `finally` 中释放 store：

  ```python
  def _execute_research_task(
      task_id: str,
      execution_mode: ExecutionMode,
      *,
      redelivered: bool = False,
  ) -> None:
      if execution_mode not in {"simulated", "real"}:
          raise ValueError(f"invalid execution mode: {execution_mode!r}")
      store = _store_factory()
      try:
          claimed = store.claim_task(task_id, allow_running=redelivered)
          if claimed is None and store.get_task(task_id) is None:
              raise ValueError(f"task not found: {task_id}")
          if claimed is None:
              return
          if execution_mode == "simulated":
              WorkflowRunner(store).run_simulated(task_id)
              return

          runtime = _get_runtime()

          def real_runner(query: str, *, on_event=None) -> list[dict]:
              return run_conversation(
                  query,
                  on_event=on_event,
                  mcp_runtime=runtime,
              )

          WorkflowRunner(store, real_runner=real_runner).run_real(task_id)
      finally:
          store.close()
  ```

  如果测试中的假 store 尚无 `close()`，应补上明确的 no-op/recording method，不使用 `getattr` 隐藏接口错误。

- [ ] 5. 运行生命周期测试：

  ```bash
  ./.venv/bin/python -m pytest \
    tests/web/test_web_app.py \
    tests/web/test_celery_worker.py -q
  ```

- [ ] 6. 运行整个 Web 测试集：

  ```bash
  ./.venv/bin/python -m pytest tests/web -q
  ```

- [ ] 7. 提交本任务：

  ```bash
  git add \
    paperpilot/web/app.py \
    paperpilot/web/worker_tasks.py \
    tests/web/test_web_app.py \
    tests/web/test_celery_worker.py
  git commit -m "refactor(db): manage SQLAlchemy engine lifecycle"
  ```

  提交前用 `git diff --cached --name-only` 确认没有 stage 任何无关文件。

---

## Task 8: Document Migration Operations and Run the Compatibility Gate

**Files:**

- Modify: `README.md`
- Verify only: `data/web/tasks.sqlite3` 的临时副本
- Verify: all relevant tests

### Rationale

代码兼容不等于迁移可上线。最终门槛必须用真实数据库的副本验证 Alembic 接管、数据行数、未知表和完整性，并把“先备份、再升级、再启动 Web/worker”的顺序写进项目文档。真实数据库本身不能在验证中被修改。

### Steps

- [ ] 1. 在 `README.md` 增加“Database migrations”小节，包含以下内容：

  ```markdown
  ### Database migrations

  PaperPilot keeps Web and agent persistence in the configured SQLite file.
  Back up the database before every schema migration, then upgrade it before
  starting the Web process or Celery workers.

  ```bash
  cp data/web/tasks.sqlite3 /path/to/verified-backup/tasks.sqlite3
  PAPERPILOT_TASK_DB_PATH=data/web/tasks.sqlite3 \
    ./.venv/bin/python -m alembic -c alembic.ini upgrade head
  PAPERPILOT_TASK_DB_PATH=data/web/tasks.sqlite3 \
    ./.venv/bin/python -m alembic -c alembic.ini current
  ```

  `TaskStore` retains an automatic migration compatibility path for local and
  test use, but production startup must not rely on concurrent workers racing
  to initialize a database.
  ```

  实施时按 README 现有语言调整中英文，但命令和警告不能省略。

- [ ] 2. 创建独立临时目录并复制真实库；不要对 `data/web/tasks.sqlite3` 原文件运行升级：

  ```bash
  PAPERPILOT_MIGRATION_CHECK_DIR="$(mktemp -d)"
  cp data/web/tasks.sqlite3 \
    "$PAPERPILOT_MIGRATION_CHECK_DIR/tasks-before.sqlite3"
  cp data/web/tasks.sqlite3 \
    "$PAPERPILOT_MIGRATION_CHECK_DIR/tasks-after.sqlite3"
  ```

- [ ] 3. 在副本迁移前记录完整性、业务表行数和所有表名：

  ```bash
  sqlite3 "$PAPERPILOT_MIGRATION_CHECK_DIR/tasks-before.sqlite3" \
    "PRAGMA integrity_check; SELECT name FROM sqlite_master WHERE type='table' ORDER BY name; SELECT COUNT(*) FROM users; SELECT COUNT(*) FROM sessions; SELECT COUNT(*) FROM research_tasks; SELECT COUNT(*) FROM task_events; SELECT COUNT(*) FROM task_artifacts;"
  ```

- [ ] 4. 只升级 `tasks-after.sqlite3`：

  ```bash
  PAPERPILOT_TASK_DB_PATH="$PAPERPILOT_MIGRATION_CHECK_DIR/tasks-after.sqlite3" \
    ./.venv/bin/python -m alembic -c alembic.ini upgrade head
  ```

- [ ] 5. 对迁移后副本执行同样的完整性、表名和行数查询。验收条件：

  - `PRAGMA integrity_check` 返回 `ok`；
  - 五张 Web 表迁移前后行数一致；
  - 所有 Agent/未知表仍存在，行数不减少；
  - 新增且仅新增 `alembic_version`（若此前不存在）；
  - `alembic current` 为 `20260806_0001 (head)`；
  - 第二次 `upgrade head` 成功且不改变行数。

- [ ] 6. 运行本阶段核心回归：

  ```bash
  ./.venv/bin/python -m pytest \
    tests/web/test_database.py \
    tests/web/test_db_models.py \
    tests/web/test_db_migrations.py \
    tests/web/test_task_store.py \
    tests/web/test_auth.py \
    tests/web/test_web_app.py \
    tests/agent/test_store.py -q
  ```

- [ ] 7. 运行项目完整测试集：

  ```bash
  ./.venv/bin/python -m pytest -q
  ```

  记录通过/失败/跳过数量。任何新增失败都必须定位并修复；不能用“与数据库无关”跳过未诊断的失败。

- [ ] 8. 做最终静态与变更边界检查：

  ```bash
  ./.venv/bin/python -m compileall -q paperpilot tests
  git diff --check
  git status --short
  git diff --stat
  ```

- [ ] 9. 确认以下禁止项没有出现：

  ```bash
  rg "metadata\.create_all|DROP TABLE|DROP COLUMN|batch_alter_table" \
    paperpilot migrations
  rg "sqlite3" paperpilot/web/task_store.py
  ```

  Expected: 第一条不应在生产代码/迁移中命中禁止的 schema 操作；第二条无命中。

- [ ] 10. 提交运维文档：

  ```bash
  git add README.md
  git commit -m "docs(db): document Alembic migration workflow"
  ```

---

## 4. Acceptance Criteria

只有同时满足以下条件，第一阶段才算完成：

- [ ] `TaskStore` 所有现有公开方法仍可由原调用方使用，返回 dataclass 与字段语义不变。
- [ ] `paperpilot/web/task_store.py` 不再包含手写 schema DDL 或直接 `sqlite3` 数据访问。
- [ ] 新 SQLite 文件可由 `alembic upgrade head` 完整创建五张 Web 表和四个既有索引。
- [ ] 当前 SQLite 副本可在不丢数据、不重建历史表、不删除 Agent/未知表的前提下升级到 head。
- [ ] 历史缺列库可补 `research_tasks.user_id`、`task_events.stage`、`task_events.payload_json` 并保留旧行。
- [ ] queued task + queued event 写入仍是同一事务，任一失败全部回滚。
- [ ] 并发 `claim_task()` 仍只有一个调用者成功。
- [ ] `get_task_updates()` 仍提供一致性快照，不混入事务开始后的 event/artifact。
- [ ] WAL、foreign keys、30 秒 busy timeout、NORMAL synchronous 和当前查询索引继续生效。
- [ ] Web app 和 Celery worker 的 Engine 生命周期有测试覆盖。
- [ ] Web/Auth/Agent 共享数据库相关测试全部通过。
- [ ] 项目完整测试集无新增失败。
- [ ] README 明确生产部署顺序：备份 -> Alembic upgrade -> Web/worker 启动。

---

## 5. Risks and Explicit Follow-Ups

### 5.1 历史外键差异

真实库 `research_tasks.user_id` 没有外键。初始 migration 不做 SQLite table rebuild，因此旧库继续保持这一状态，新库则有外键。这是刻意的风险隔离，不是遗漏。后续若要统一，必须先检查孤儿 `user_id`、Agent 表对 `research_tasks` 的引用、trigger/index 保留，再独立设计迁移。

### 5.2 首次多进程迁移竞争

`TaskStore` 为兼容本地/测试保留 `ensure_database_current()`，线程内有锁，但它不应替代生产部署迁移。多个 Web/worker 进程同时面对未迁移库仍可能竞争 SQLite DDL 锁，因此 README 和部署流程必须把显式 `upgrade head` 作为启动前置条件。

### 5.3 SQLAlchemy 异常包装

SQLAlchemy 会把 DBAPI 错误包装为 `sqlalchemy.exc.IntegrityError/OperationalError`。用户可见 API 当前由上层统一处理，预计行为不变；内部测试若断言了 `sqlite3` 异常类型，应改为 SQLAlchemy 类型。`DuplicateUsernameError` 仍是业务层稳定异常，不应泄露底层类型。

### 5.4 不在本阶段解决的问题

- 不拆分 `TaskStore` 为多个 Repository；等会话/线程领域模型确定后再按真实聚合边界拆。
- 不把 Agent 四张表迁入同一 SQLAlchemy metadata；它们仍由旧 runtime 管理，避免扩大变更面。
- 不建立 `Conversation/Message/Branch` 表；这是下一阶段会话管理子项目。
- 不建立 LangGraph checkpoint 或 Store 表；它们由后续框架集成阶段管理。
- 不切 PostgreSQL；先把 schema 与迁移纪律建立起来，再评估数据库迁移。

---

## 6. Recommended Execution Order

严格按 Task 1 -> 8 顺序执行。每个任务完成后必须满足：

1. 该任务新增测试先失败、实现后通过；
2. 已完成任务的相关回归继续通过；
3. `git diff --check` 通过；
4. 只提交当前任务文件；
5. 遇到 schema、公开接口或行为需要超出本计划的变化时立即暂停，回到用户确认，而不是在实现中临时扩大设计。
