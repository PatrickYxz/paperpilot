# Day 16: deep-read 召回精度 eval(QASPER 子集 + 2 baseline 三段对比)

**目标:** 给 PaperPilot 出第一份**第三方数据 + 多 baseline 对比**的定量 eval 报告。用 AI2 QASPER 的 NLP paper 子集做"事实召回"评测,对比 (1) DeepSeek 看 abstract、(2) DeepSeek 看 full text、(3) PaperPilot 全栈端到端,出三段递进通过率。**配套 trace 落盘机制**(零侵入主链路),为 Day 17 的 case study + README 备弹药。

**Why:** 简历向求职阶段需要"做得多好"的定量数字,而非只讲"做了什么"。小样本通过率(< 20 case)经不起追问,这版用 QASPER ≈ 50 paper × 3 题 = 150 题做主集,数据来自第三方,通过率硬。同时三段 baseline 递进直接回答"为什么 PaperPilot 这种结构有意义,而不是把全文塞 LLM context 当大水缸用"这个最容易被面试官质疑的点。

**Scope(Day 16 一天):** 单点能力 eval pipeline(prepare/run/summarize 三步)+ 一份 markdown summary + 全栈那组的 trace 落盘 + 失败归因 Top-N。**红线全守恒**:`paperpilot/main.py / core/loop.py / core/adapter.py / builtin_tools/*.py / mcp_servers.json` zero diff。

**不在本 spec 范围(留 Day 17):**
- README 改写、demo 视频、case study walkthrough(用本 spec 的 trace 产物展开)
- LLM-as-judge / 人工评分 / 复合任务 eval(综述生成 / 引用扩展)
- BM25 第三个 baseline(若 Day 16 跑出"全栈 ≈ full-text"即颜面危机时,Day 17 临时加,否则 YAGNI)

---

## §1 设计决策(brainstorm 沉淀)

| Q | 选择 | 理由 |
|---|---|---|
| Q1: Day 16 方向 | 3+5 求职向(eval + demo polish)| 内核功能已闭环,差"做得多好"的数字 + 简历可交付物;走 2 天: Day 16 = eval 层,Day 17 = polish 层 |
| Q2: 单点能力 | A deep-read 召回精度 | Oracle 简单(原文 contains),触发链路最长(load_skill→colbert.build→colbert.search×N→综合),baseline 对比叙事天然成立 |
| Q3: 数据集 | X QASPER 子集 | 第三方数据无作弊嫌疑;省下数据生成 2-3h 的 quota 给 trace + 报告;题型偏"细节召回"正是 colbert 甜区 |
| Q4: baseline 数量 | 2 个(三段对比)| abstract-only / full-text dump / PaperPilot 全栈;故事:"摘要 X% → 全文 Y% → colbert+deep_read Z%"三段递进;**Z > Y 才证明栈有意义** |
| Q5a: pipeline 切几步 | 3 步分离 prepare/run/summarize | 跑半挂能续(append jsonl skip 已完成);summarize 纯本地可反复迭代不烧 API |
| Q5b: trace 落盘 | 复用现有 `on_event` hook + 外部 JSONL tracer | 0 改主链路,新增 `paperpilot/eval/jsonl_tracer.py`(纯新增模块) |
| Q5c: 产物形态 | `summary.md` + 三份 `results_<baseline>.jsonl` | md 给人看,jsonl 给 Day 17 程序读挑亮点;不出 CSV(YAGNI) |
| Q5d: arxiv mapping | QASPER `paper_url` 字段正则 + dry-run download 验证 | 自动化,失败的换下一篇,不手工 mapping(YAGNI)|

**红线守恒(zero diff)清单:**
- `paperpilot/main.py`
- `paperpilot/core/loop.py` / `core/adapter.py` / `core/llm_client.py` / `core/guardrail.py`
- `paperpilot/builtin_tools/*.py`(skill_loader / research_todo / subagent / compact)
- `paperpilot/skills/*.md`(6 个 skill 一字不改)
- `paperpilot/mcp_servers.json` 与 5 个 MCP server 实现
- 现有 116 个 fast test 一个不退

**唯一 paperpilot 包内的代码改动:** 新增模块 `paperpilot/eval/`(单独子包),不被 main / loop / 任何已有模块 import。

---

## §2 文件结构

| 路径 | 动作 | 责任 |
|---|---|---|
| `paperpilot/eval/__init__.py` | 新建,空 | 子包 marker |
| `paperpilot/eval/jsonl_tracer.py` | 新建 | `make_jsonl_tracer(run_id, out_dir) -> Callable`,把 `on_event` 事件落盘 |
| `paperpilot/eval/qasper_loader.py` | 新建 | 从本地 QASPER dump 抽 50 paper × 3 extractive QA,解析 arxiv_id,产出 `EvalCase` 列表 |
| `paperpilot/eval/baselines.py` | 新建 | `run_abstract_only(case) -> Answer` / `run_full_text_dump(case) -> Answer` / `run_paperpilot(case) -> Answer`(后者内调 `main.run`)|
| `paperpilot/eval/scorer.py` | 新建 | `is_pass(predicted: str, oracle_spans: list[str]) -> bool`(lowercase contains 任一 oracle span 即过)+ `cluster_failures(records) -> dict[str, int]` |
| `tests/eval/test_qasper_loader.py` | 新建 | 单测:固定 mini fixture(2 paper × 3 QA),验证 arxiv_id 提取、QA 抽取 |
| `tests/eval/test_scorer.py` | 新建 | 单测:contains 大小写 / 多 oracle span / 失败归因分桶 |
| `tests/eval/test_jsonl_tracer.py` | 新建 | 单测:tracer 把假 event 落对位置 |
| `scripts/day16_prepare_eval.py` | 新建 | 一次性:加载 QASPER → 抽样 → arxiv mapping dry-run → 写 `data/eval/qasper_subset.jsonl` |
| `scripts/day16_run_eval.py` | 新建 | 主跑:对每个 case 跑 3 baseline,append 到 `data/eval/results_<baseline>.jsonl`,可中断续跑(skip 已存在 case_id) |
| `scripts/day16_summarize.py` | 新建 | 读 3 份 results → 生成 `data/eval/summary.md` + Top-N 失败归因 |
| `data/eval/.gitkeep` | 新建,空 | 让产物目录入仓 |
| `data/traces/.gitkeep` | 新建,空 | trace 落盘目录 |

**外部依赖**:用户需要先手工下载一次 QASPER 数据集(~30MB)放到 `data/eval/qasper-source/qasper-train-v0.3.json`(或 dev/test 任一)。下载方式 spec §6 说明,**不入仓**(走 .gitignore)。

---

## §3 核心数据模型

### EvalCase(`paperpilot/eval/qasper_loader.py`)

```python
@dataclass(frozen=True)
class EvalCase:
    case_id: str            # 形如 "qasper-1606.07947-q0"
    arxiv_id: str           # "1606.07947"
    paper_title: str
    abstract: str           # 用于 abstract-only baseline
    full_text: str          # 用于 full-text baseline(QASPER 含 paragraph 级别全文)
    question: str
    oracle_spans: list[str] # extractive answer 的 evidence text(一题可有多条 reference answer,任一 hit 即过)
```

### AnswerRecord(三组 baseline 共用 jsonl 行格式)

```python
{
    "case_id": "qasper-1606.07947-q0",
    "baseline": "abstract_only" | "full_text" | "paperpilot",
    "question": "...",
    "oracle_spans": ["...", "..."],
    "predicted": "...",      # baseline 的最终回答 text
    "passed": true,
    "elapsed_s": 42.7,
    "trace_path": "data/traces/qasper-1606.07947-q0__paperpilot.jsonl" | null,
    "tool_calls": ["load_skill", "mcp__arxiv__download_paper", ...] | null,
    "error": null            # 异常信息(eg. download 超时)
}
```

只有 `paperpilot` baseline 有 `trace_path` / `tool_calls`,其它两组为 null。

---

## §4 三组 baseline 跑法

**统一 LLM**:全部用 `LLMClient()`(DeepSeek via anthropic SDK),保证模型变量受控。

### B1. `abstract_only`

```
prompt = f"""你是学术论文助手。下面是论文 "{title}" 的 abstract:

{abstract}

请基于 abstract 简洁回答以下问题。如果 abstract 不含答案,直接说"abstract 中未提及":

Q: {question}
A:"""
```

单次 LLM call, 不挂 tool, 不开 agent loop. 预期 5-10s/题。

### B2. `full_text`

```
prompt = f"""你是学术论文助手。下面是论文 "{title}" 的全文:

{full_text[:截到模型 context 上限]}

请基于全文回答以下问题:

Q: {question}
A:"""
```

单次 LLM call, 截断策略: 若用 `tiktoken cl100k_base` 估算 token > 60K(留 4K 给 prompt+answer),**保留全文前 60K tokens**(从尾部裁剪超出的尾段)+ 在 prompt 顶部注明"⚠ 全文过长,已截至 N tokens, 后段省略"。abstract 字段不重复(已在全文开头)。预期 15-30s/题。

### B3. `paperpilot` 全栈

```python
prompt = (
    f"请精读 arxiv:{case.arxiv_id}(标题《{case.paper_title}》),回答下面的问题。"
    f"使用 deep-read-paper skill 的工作流(load_skill → download_paper → "
    f"build_index → 多次 colbert.search → 综合)。\n\nQ: {case.question}\nA:"
)
tracer = make_jsonl_tracer(run_id=case.case_id, out_dir=Path("data/traces"))
messages = main.run(prompt, max_iter=12, on_event=tracer)
predicted = _extract_final_text(messages[-1])
```

`main.run` 端到端,LLM 自决路径(可能不 load_skill 直接拍脑袋答 → 也是真实能力, 该测就该测)。预期 60-120s/题。

**为什么不把 colbert+download 写死调用?** 因为我们要测的是"agent 自主选 skill + 串联 tool 的端到端能力",不是"colbert 单工具的召回率"。后者是 colbert library 的事,跟 PaperPilot 没关系。

---

## §5 评分与失败归因

### `is_pass`(`paperpilot/eval/scorer.py`)

```python
def is_pass(predicted: str, oracle_spans: list[str]) -> bool:
    pred_lower = predicted.lower()
    return any(
        _normalize(span).lower() in _normalize(pred_lower)
        for span in oracle_spans
    )

def _normalize(s: str) -> str:
    # 折叠空白 / 去前后空格 / 去标点尾巴(. , ;)
    return re.sub(r"\s+", " ", s.strip(" .,;:"))
```

**为什么不更严格(BLEU / F1 / exact match)?** QASPER extractive answer span 已经是 evidence text,只要 LLM 输出包含该 span 就说明答对了关键事实。BLEU/F1 会把"基于 evidence 重述一遍"打低分,跟"是否答对事实"脱钩。

**LLM-as-judge 兜底?** 不做。判 `contains` 失败的 case 全部记为 fail,Day 17 case study 阶段人工挑出"实际答对但 contains 没匹配"的当 false negative 单独讨论,不混入主指标。

### 失败归因(`cluster_failures`)

输入是 `results_paperpilot.jsonl` 解析后的 record 列表(每行已含 `tool_calls` 字段, run_eval 阶段从 trace 抽好);**不重新打开 trace 文件**。对 `passed=False` 的每个 record,按 `tool_calls` 序列归到下表 6 桶之一(优先级从上到下,匹配第一桶即停):

| 归因桶 | 判定规则 |
|---|---|
| `no_load_skill` | trace 中 `load_skill` 调用为 0 次 |
| `no_download` | trace 中 `mcp__arxiv__download_paper` 为 0 次 |
| `no_colbert_search` | trace 中 `mcp__colbert__search` 为 0 次 |
| `colbert_searched_low` | colbert.search 次数 ∈ {1, 2}(deep-read 推荐 ≥ 3) |
| `iter_exhausted` | guardrail_stop 事件 = max_iter |
| `synthesis_miss` | 以上都通过但 predicted 仍不含 oracle(LLM 综合错)|

`summary.md` 输出 Top-3 failure reason + 计数。

---

## §6 QASPER 数据获取

QASPER 来自 AI2 Allen Institute, 公开下载:

```
https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
```

(或者从 HuggingFace Hub `allenai/qasper`, 二选一)

`scripts/day16_prepare_eval.py` 假设用户已下载并解压到 `data/eval/qasper-source/qasper-train-v0.3.json`(spec 在脚本顶部 docstring 写明,缺文件直接报错指引)。

**抽样规则**(顺序为目标筛选,固定 random seed=42 保证可重现):
1. 遍历 QASPER 中的 paper,过滤 `paper_url` 含 `arxiv.org`
2. 提 arxiv_id(正则 `arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})`)
3. 对每篇 paper,遍历其 `qas` 列表,**仅保留 `answers[0].extractive_spans` 非空的 QA**;**该 paper 的 extractive QA 数量 < 3 则整篇丢弃**(保证每篇都贡献 3 题)
4. 通过的 paper 取**前 3 个** extractive QA(QASPER 内原始顺序稳定)
5. 对该 paper 跑 `mcp__arxiv__download_paper` dry-run(走真 arxiv API,~5-15s),失败则丢弃整篇
6. 累计直到攒满 50 篇有效 paper 为止;若遍历完整 QASPER 仍 < 50 paper,降级到底线 40 paper(< 40 则脚本 FAIL)
7. 写出 `data/eval/qasper_subset.jsonl`,每行一个 EvalCase

**oracle_spans 取值:** 用 QASPER 该 QA 的 `answers[0].extractive_spans`(list of str)。若同一 QA 有多个 annotator(`answers` 数组长度 > 1),取并集。

---

## §7 Pipeline 流程

```
            [手工: 下载 QASPER → data/eval/qasper-source/]
                          |
                          v
               day16_prepare_eval.py
                  (一次性, ~10 min)
                          |
                          v
            data/eval/qasper_subset.jsonl
                  (50 paper × 3 题 = 150 case)
                          |
            +-------------+-------------+
            |             |             |
            v             v             v
       B1 abstract   B2 full_text   B3 paperpilot
        ~15 min       ~50 min       ~2.5h (含 colbert build)
            |             |             |
            v             v             v
   results_abstract  results_full_  results_paperpilot
       _only.jsonl    text.jsonl      .jsonl
                          |
                          v
                day16_summarize.py
                  (纯本地, ~30s)
                          |
                          v
              data/eval/summary.md
              (3 段对比表 + Top-N 失败归因)
```

**断点续跑机制**(`day16_run_eval.py`):
- 启动时读现有 `results_<baseline>.jsonl`,跳过 `case_id` 已存在的行
- 每个 case 完成立即 append 一行(不批量)
- 异常 case 也写一行 `error` 字段非 null,不阻塞剩余 case
- CLI 参数:`--baseline {abstract_only|full_text|paperpilot|all}` / `--limit N`(用于 dry-run)

---

## §8 测试策略

**单测(fast,无 LLM call):**
- `tests/eval/test_qasper_loader.py`(~5 case): 固定 fixture(`tests/eval/fixtures/qasper_mini.json`,2 paper × 3 QA),验证 arxiv_id 提取 / QA 过滤(extractive only)/ 多 annotator 并集 / 截断保护(paper_url 非 arxiv 时跳过)
- `tests/eval/test_scorer.py`(~6 case): contains 命中 / 大小写 / 标点尾巴 / 多 oracle 任一即过 / 失败归因 6 桶分类(用造的 mock trace dict)
- `tests/eval/test_jsonl_tracer.py`(~3 case): tracer fn 把 (kind, payload) append 到目标文件,目录不存在自动创建

预期 fast suite 总数:**116 + 14 ≈ 130 passed**。

**slow / 真 LLM:** 不写 pytest,所有真 LLM 验证都在 `scripts/day16_*` 里跑,产物落 `data/eval/`。

**回归红线断言**: Task 4(详见 plan)末跑全套 fast suite,确认 130 个全过 + 红线文件 git diff 为空。

---

## §9 产物示例(`summary.md` 草样)

```markdown
# Day 16 deep-read 召回精度 eval

**Date**: 2026-05-08
**Dataset**: AI2 QASPER NLP subset (50 papers × 3 extractive QA = 150 cases)
**LLM**: DeepSeek (via anthropic SDK), shared across all baselines

## 三段对比

| Baseline | Pass | Fail | Error | Pass Rate | Avg latency |
|---|---|---|---|---|---|
| abstract_only | 47 | 102 | 1 | **31.5%** | 6.8s |
| full_text     | 89 | 60  | 1 | **59.7%** | 22.4s |
| paperpilot    | 118 | 30 | 2 | **79.7%** | 87.2s |

(数字为占位,实跑后填)

## PaperPilot 失败归因(Top-3)

| 桶 | 计数 | 占比 |
|---|---|---|
| synthesis_miss | 14 | 47% |
| colbert_searched_low | 8 | 27% |
| no_load_skill | 5 | 17% |
| (其它) | 3 | 10% |

## 解读

- abstract → full_text 提升 +28pts:细节召回必须看正文,abstract 远不够
- full_text → paperpilot 提升 +20pts:**colbert 选段 + 多次召回** 比"全文一次性塞 LLM" 显著更好,验证了 RAG 路线在 ≤30K 长 paper 上仍有价值
- 主要失败模式 synthesis_miss(47%):colbert 召回到对的段,但 LLM 综合时遗漏关键 token → 后续可加 reranker / 提示工程
```

---

## §10 工时估算与风险

| 阶段 | 估时 | 备注 |
|---|---|---|
| 写 `qasper_loader` + 单测 | 1h | 含手工下载 QASPER 30MB |
| 写 `scorer` + 单测 | 0.5h | |
| 写 `jsonl_tracer` + 单测 | 0.5h | |
| 写 `baselines.py` 三组 | 1h | |
| 写 `prepare_eval.py` + dry-run 验证 50 paper | 0.5h | 含 arxiv 下载缓存 |
| 写 `run_eval.py`(支持续跑)| 1h | |
| 跑 B1 abstract(~15 min)| 0.3h | |
| 跑 B2 full_text(~50 min)| 0.8h | |
| 跑 B3 paperpilot(~2.5h)| 2.5h | 可同期写其它代码 |
| 写 `summarize.py` + 调格式 | 1h | |
| 自审 + 红线 diff 验证 + commit | 0.5h | |
| **总计** | **~9.5h** | Day 16 跑满,可能溢出 1-2h |

**风险与对策:**

1. **B3 跑 2.5h 真就 2.5h** — 不可压缩(LLM call + colbert build × 50 paper)。**对策**:B3 后台跑同时主线写 `summarize.py`,并行最大化
2. **arxiv download 失败率高** — 部分 paper 已撤回 / arxiv 限流。**对策**:prepare 阶段 dry-run 全 50 篇,失败 > 10% 则降到 40 paper 主集 + 报告备注
3. **DeepSeek 长 context 截断**(B2)— 全文 > 60K tokens。**对策**:头部截断保留前 60K + abstract 叠在 prompt 顶部,prompt 注明"截至 N tokens"
4. **某 baseline 异常率 > 5%** — 视为该 baseline 数字不可信。**对策**:summary.md 单独标记,Day 17 case study 时 retry 该批
5. **三段差距不显著(eg. paperpilot ≈ full_text)** — 故事破产。**对策**:Day 16 不抢救,Day 17 加 BM25 第三 baseline + 改用更难的题型(只取 QASPER 中 abstract 不含答案的 QA 子集),把"细节召回"难度抬高

---

## §11 DoD(Definition of Done)

- [ ] `paperpilot/eval/` 4 个新模块 + 3 份单测,fast suite **130 passed**
- [ ] `git diff main -- paperpilot/main.py paperpilot/core/ paperpilot/builtin_tools/ paperpilot/skills/ paperpilot/mcp_servers.json` → 空(红线零 diff)
- [ ] `data/eval/qasper_subset.jsonl` 含 ≥ 40 paper × 3 题 ≥ 120 case
- [ ] `data/eval/results_{abstract_only,full_text,paperpilot}.jsonl` 三份就位,paperpilot 那份每行含 trace_path
- [ ] `data/traces/` 含 ≥ 120 个 jsonl 文件(每 case 一份)
- [ ] `data/eval/summary.md` 含三段通过率表 + Top-3 失败归因 + 解读段
- [ ] `paperpilot 通过率 > full_text 通过率`(若不成立,记录在 summary 里 + Day 17 处理)
- [ ] git log 含 spec / plan / 至少 4 个 Task commit