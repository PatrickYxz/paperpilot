# Test Set Dimensions Plan（维度化测试集）

## 背景与目标

用户决策：论文池走广覆盖（B 方案，10+ 篇各 1-2 题）；预算不敏感（flash 模型）；
诱导题（考诚实度）进 active 集；多论文场景放下期。目标：12-13 道题覆盖
能力维度矩阵，每题带维度标签，报告可按维度统计。

## 出题策略

- 论文选经典（Transformer/BERT/ResNet/GPT-3/ViT/word2vec/Adam/T5）：
  事实要点是公共知识，expected_points 可靠写出，且 arXiv 可下载。
- 维度分布：事实检索 ×4、数值 ×2、完整性 ×1（草稿）、诚实度（诱导）×2、
  抗混淆 ×1（草稿）、多轮 ×1（已有草稿）。
- 诱导题问论文里不存在的东西（如"BERT 在 ImageNet 的 top-1 精度"），
  期望行为是拒绝/纠正前提而不是编造数值。

## 约束条件

- 当前网络 arXiv export API 不可达，每篇论文配 `paper_metadata`（WebFetch
  逐篇获取，摘要为忠实释义）；网络恢复后可去掉。
- 不引入新环境变量；judge 复用既有机制。

## 分步骤执行计划

### Step 1：场景模型扩展
- `Scenario` 增加 `tags: tuple[str, ...]`（维度标签）、`expect_refusal: bool`
  （期望拒绝标记）、`trap_terms: tuple[str, ...]`（不得出现的编造值）。
- `load_scenarios` 校验三者类型；报告行携带 tags。

### Step 2：诱导题程序化检查
- `evaluate_checks`：场景带 `trap_terms` 时新增 `no_trap_terms` 检查
  （大小写不敏感子串，答案出现即失败）。

### Step 3：judge 拒绝判定分支
- `expect_refusal=true` 时 judge 换专用 prompt：判定答案是"正确拒绝/纠正
  前提"还是"编造"，输出 `{"verdict": "declines"|"fabricates"}`；
  `run_judge_check` 按此判定通过与否，不要求 expected_points。

### Step 4：题库建设（12-13 题）
- 8 篇经典论文各 1-2 题 + 既有 2001.09899 场景；事实题/数值题带
  expected_points 直接 active；诱导题 active（注明需 `--judge` 才有区分力）。
- 每题 tags 标注维度；逐篇 WebFetch 元数据填入。

### Step 5：测试与验证
- 单元测试：tags/refusal/trap 加载校验、no_trap_terms 红绿、judge 拒绝分支
  （declines/fabricates）、报告行 tags。
- 全量 `pytest -q`；随后用 active 集做一轮真实运行（`--judge`）验证全链路，
  按结果校准引用下限等期望值。

## 验证方式

- 单元测试全绿；真实运行产出带 tags 的报告，事实题要点命中、诱导题
  judge 判 declines。

## 风险与待确认项

- 经典论文要点凭模型知识写出，存在个别记错风险（已选高置信度、摘要级
  事实）；首次真实运行若要点不中，以论文实际文本为准修正题库。
- 诱导题不带 --judge 跑时只有 trap_terms 程序化检查（弱），报告需提示。
- 多论文对比、复杂完整性题留下期。
