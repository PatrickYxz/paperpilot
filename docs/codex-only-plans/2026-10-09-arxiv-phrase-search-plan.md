# 网页论文搜索短语化（phrase quoting）修复计划

> Codex only。Claude Code 运行时不必参考。

## 背景与目标

- 现象：网页 "Select a paper" 搜索 "attention is all u need"（甚至拼对的
  "attention is all you need"）返回的都是别的论文，目标论文 1706.03762
  不在结果里。
- 根因：`default_web_paper_search`（`paperpilot/web/routes/papers.py`）把
  自由文本原样透传给 arXiv API。arXiv 对不带引号的多词查询按"词袋 AND +
  相关度"处理：`u` 字面匹配到铀（U）相关物理论文；拼对时也被一堆
  "X Is All You Need" 仿名论文挤掉。加短语引号 `all:"..."` 后目标论文
  可正常返回（已用真实 API 验证）。
- 目标：网页搜索对多词自由文本优先做短语查询；短语无结果时回退到原样
  查询；其余入口（MCP 工具、deep_reading）语义不变。

## 约束条件

- 只改 web 层（`paperpilot/web/routes/papers.py`），不动
  `paperpilot/papers.py` 的公共语义（方案 A，用户已确认）。
- 已是 arXiv 查询语法（`ti:`/`au:`/`abs:`/`cat:`/`all:`/`co:`/`jr:`/`rn:`/`sr:`
  字段前缀或 `AND`/`OR`/`ANDNOT` 布尔词）、单词、或已含双引号的查询不
  加引号，避免破坏用户显式语法。
- arXiv ID / URL 仍走精确解析路径，行为不变。

## 分步骤执行计划

1. `paperpilot/web/routes/papers.py`
   - 新增私有辅助 `_phrase_query(query) -> str | None`：多词、无字段
     前缀、无布尔词、不含 `"` 时返回 `all:"<query>"`，否则 None。
   - `default_web_paper_search` 增加可选参数 `client`（默认 None，便于
     单测注入 fake；对路由 `PaperSearch` 类型保持兼容）：
     - ID/URL 精确解析不变；
     - 自由文本：先短语查询，有结果即返回；为空则回退原样查询。
2. 新增 `tests/web/routes/test_papers.py`（镜像源码路径）：
   - 多词自由文本 → 只发短语查询；
   - 短语为空 → 回退原样查询并返回其结果；
   - 短语与原样都为空 → 返回空列表；
   - 单词 / 已含查询语法 / 已含双引号 → 不加引号；
   - arXiv ID → 走精确解析，不发关键词查询。

## 验证方式

- `pytest tests/web/routes/test_papers.py tests/papers/test_arxiv_catalog.py`
  全绿。
- 回归：`pytest tests/web/routes tests/web/test_web_app.py`（路由注入
  fake `paper_search`，应不受影响）。
- 真实 API 冒烟：`default_web_paper_search("attention is all you need", 10)`
  返回含 1706.03762；"attention is all u need"（拼写错误场景）短语为空 →
  回退原样（仍可能返回无关论文，属已知限制）。

## 风险与已知限制

- 拼写错误的查询短语无结果时回退词袋，仍可能出无关结果（arXiv 无模糊
  匹配），本次不解决。
- 短语查询会把整串当精确短语，用户想"多个关键词 AND 检索"时结果变少；
  需要词袋语义可显式写 `AND` 或字段前缀绕过。
- 短语为空时最多两次 API 调用，arXiv Client 自带限速，风险可忽略。
