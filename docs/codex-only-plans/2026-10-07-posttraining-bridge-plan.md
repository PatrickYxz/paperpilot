# 第七章衔接：评估资产 → 训练数据导出（rag 分支）

> Codex only。第七章对 paperpilot 的定位是"Harness 优先"的确认章
> （书 7.6：Harness 能解决就不训练，大多数应用落在这里），不引入
> SFT/RL。唯一落地是表 7-4 的前半段桥接。

## 落地：training_export（表 7-4 映射的具体化）

- `smoke/training_export.py` + CLI `--export-training`：
  - 成功场景回答 → `sft_messages.jsonl`（user/assistant messages）
  - 同场景成功/失败回答 → `dpo_pairs.jsonl`（chosen/rejected）
  - 失败 + 归因 → `negative_labels.jsonl`（过程监督负标签，
    首错误类别随样本）
- 前置改造：outcome 文件按 rep 落盘（`-r{n}` 后缀 + 内嵌
  question/passed/attribution），不再被 repeat 覆盖
- 只做桥接不做消费：训练管线是未来决策（YAGNI 约束）

## 验证

- 单测 4 个（SFT 来源、负标签带归因、DPO 配对、文件往返）
- live：2 场景 × repeat 2 全过 → sft=4 / dpo=0（无失败对，正确）
- 全量 1029 passed

## 明确不做

SFT/RL 训练、轨迹前缀回归（第六章已评估过大工程低收益）、
仿真环境（无训练需求）。架构已符合书末"RAG 管事实 + ICL 试验策略 +
程序固化约束"的组合建议。
