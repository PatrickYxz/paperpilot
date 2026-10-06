# 第五章落地：计算沙盒 + 重复调用指纹（rag 分支）

> Codex only。用户已确认 A+B。PaperPilot 是垂直 Agent，不引入通用
> Coding Agent 架构，只取书中两个直击痛点的方向。

## A：代码作为思考工具（书 5.2.1）

论文数值题（百分比/比率/参数量/表格对比）当前靠 LLM 心算——概率性
推理短板。新增 `run_computation(code)` 工具：agent 先从证据取数、再写
码精确计算。

- `paperpilot/compute/sandbox.py`：进程级沙盒
  `run_python_code(code, timeout_s=10, output_limit=4000)` ——
  subprocess + 临时 cwd + **环境变量完全剥离**（仅 PYTHONIOENCODING，
  API key 绝不进子进程；书中网络出口控制的本地等价）、超时结构化
  返回、输出截断、代码长度上限。
- `paperpilot/compute/agent_tool.py`：`build_computation_tool`
  （harness helper 注入模式，同 user_memory/agent_tool）；描述按第四章
  标准：何时用（精确数值先取数再计算）、边界（无网络/无文件写/10s/
  输出截断；数字必须来自检索证据，不得靠计算编造事实）。
- research_agent 注册 + prompt v8 → v9。
- smoke 数值场景：Transformer base/big 参数量比等（检索取数+计算），
  校准后激活。

## B：重复调用指纹检测（书 5.1.5）

同工具+同参数反复调用 = 无进展循环信号（修合约 bug 时实际观察到）。

- `_emit_tool_call` 内按 `name + canonical arguments` 计指纹
  （WeakKeyDictionary 按 context 隔离，不进 checkpoint）。
- 第 3 次同指纹：发 `tool_repetition_warning` 事件。
- repair message 附上重复指纹清单，引导改变策略（模型可见通道）。
- 不改变正常路径行为。

## 验证与风险

- 单测：沙盒（计算/超时/截断/env 剥离/语法错误）、工具描述断言、
  指纹计数与警告；全量 pytest。
- smoke：数值计算场景 + 既有场景回归。
- 风险：子进程开销小（~100ms）；deepseek 写的代码失败时结构化错误
  回流（工具返回 stderr），属预期自纠路径。
