# Citation Anchoring Check Plan（引用真实性校验）

## 背景与目标

LLM 研究系统最典型的翻车模式是"编造引用"——答案格式完美、引用编号齐全，
但引用锚点不指向任何真实检索到的证据。管线 schema 层已有
`ResearchResult.validate_evidence_references`（引用必须指向证据池、ID 唯一），
但那是管线内部自证；冒烟层需要独立复核**最终序列化载荷**：

- 答案里每条引用的 `evidence_id` 必须能在该轮 `research_result.evidence_items`
  的 `id` 集合里找到。
- 2026-10-04 真实运行已验证结构：证据池条目键为 `id`，引用条目键为
  `evidence_id`（键名不同，校验时分别取用），当次两条引用全部锚定。

## 约束条件

- 校验数据全部来自现有 outcome 载荷（`assistant_message.metadata`），不需要
  新增观测字段或 API 调用。
- fail-closed：有引用但证据池缺失/为空时判不通过（"无法验证"不算通过），
  而不是静默放行。
- 无引用时空真通过（数量是否达标由 `citations_gte` 把关，职责分离）。
- 逐轮校验（多轮场景每轮各自的池），聚合进一条检查。

## 分步骤执行计划

### Step 1：提取函数
- `checks.py` 新增 `extract_evidence_ids(message)`：取
  `metadata.research_result.evidence_items[].id`；载荷形状变化的唯一改动点。

### Step 2：断言项
- `evaluate_checks` 新增 `citations_anchored`：
  - 全部轮的引用 ID ⊆ 该轮池 ID → 通过，detail 报锚定总数；
  - 未锚定 ID 逐个列出并标轮次；
  - 池缺失且有引用 → 失败，detail 标 `evidence pool missing (turn N)`。

### Step 3：测试
- conftest 的 `passing_outcome` 补齐真实形状（`research_result.evidence_items`
  与 citations 的 ID 对齐），健康基线含新检查。
- 新增用例：锚定通过 / 编造 ID 失败并点名 / 池缺失 fail-closed /
  无引用空真通过 / 多轮聚合 / 真实载荷形状回归（`id` vs `evidence_id` 键名）。
- 既有健康基线的检查名集合断言同步更新。

### Step 4：验证
- `pytest tests/smoke -q`、全量 `pytest -q`、`git diff --check`。
- 用 2026-10-04 真实 outcome 载荷离线重放校验（不改代码路径，直接调
  `evaluate_checks` 断言新检查通过）。

## 验证方式

- 全部单元测试通过；真实载荷离线重放 `citations_anchored` 通过。
- 下一次真实运行起该检查自动生效。

## 风险与待确认项

- 若未来 `evidence_items` 序列化改为截断子集（当前 schema 无上限），校验会
  误报——届时按 detail 指引收紧提取来源（如改读 context artifact）。
- fail-closed 语义意味着载荷形状变更（池字段改名）会让场景失败——这是有意
  的：形状变更本就该走适配层单点修复后再放行。
