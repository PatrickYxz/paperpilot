# ExamMate — 笔试错题整理助手 设计文档

## 背景与动机

用户备考央国企笔试，需要大量刷题并整理错题到 Obsidian。原本计划全程用 Codex + MCP 完成，但因国内网络环境下 Codex 代理不稳定，高频使用体验差。

**核心思路**：将工作流拆分为两段——

1. **高频交互**（边做题边用）→ 用国内稳定且便宜的 vision 模型，截图 OCR + 字段归类
2. **低频写入**（做完一批后）→ 用 Codex + MCP 把汇总文件写入 Obsidian

## 目标

一个本地运行的 Web 聊天界面，用户拖入错题截图 + 打一行简短备注，AI 自动填好 13 个字段，批处理完成后导出一个结构化 markdown 文件供 Codex 消费。

## 技术栈

| 组件 | 选型 | 原因 |
|------|------|------|
| Web UI | Gradio (ChatInterface + multimodal) | 零前端代码，Python 全栈，聊天+图片上传开箱即用 |
| 主模型 | 通义千问 Qwen-VL-Plus | 中文 OCR 最强，稳定，便宜 (~¥0.004/K tokens) |
| 备选模型 | 豆包 Doubao-1.5-vision-pro | 更便宜 (~¥0.003/K tokens)，量大时切换 |
| API 协议 | OpenAI-compatible | Qwen/豆包都兼容，用 `openai` Python SDK 统一调用 |
| 配置 | python-dotenv + .env | API Key 不入仓库 |
| 运行方式 | `python app.py` → 浏览器 localhost:7860 | 本地运行，无需部署 |

## 文件结构

```
exam-mate/
├── app.py              # Gradio 入口，聊天界面 + 导出按钮
├── model_client.py     # 封装模型 API 调用（Qwen / 豆包切换）
├── prompts.py          # 系统提示词 + 字段定义
├── formatter.py        # 聊天消息 → 汇总 markdown 导出
├── .env                # API Keys（不提交 git）
├── .env.example        # API Key 模板
├── requirements.txt    # gradio, openai, python-dotenv
└── README.md           # 使用说明
```

## 输出格式

每条错题按以下模板输出：

```markdown
- 编号：{自增}
- 一级模块：{行测/申论/公基/专业课}
- 二级题型：{数量关系/资料分析/判断推理/言语理解/常识判断}
- 三级题型：{具体题型名}
- 题目位置：{来源 + 题号}
- 我的答案：X
- 正确答案：X
- 首次耗时：{min}
- 错误标签：{计算错误/概念不清/审题失误/时间不足/知识盲区}
- 一句话错因：{具体描述}
- 正确方法：{分步骤解法}
- 下次重做日期：{日期}
- 连续做对次数：0
```

## 交互流程

1. 用户在 Gradio 聊天中上传错题截图
2. 用户附带简短文字备注（例如："行测/数量关系/工程问题 P23 耗时2min"）
3. AI 调用 Qwen-VL 完成 OCR + 字段填充
4. 返回填好的 13 个字段，用户在聊天中预览确认
5. 单题确认无误后，用户可继续上传下一题
6. 一个批次完成后，点击"导出汇总"→ 下载 `错题整理-YYYY-MM-DD.md`
7. 将该文件发给 Codex → Codex 按 Obsidian 模板逐条入库

## 系统提示词设计

```text
你是一个笔试备考错题整理助手。用户会提供一张错题截图和简短描述。

你的任务：
1. OCR 识别截图中的题目内容、用户的答案、正确答案
2. 根据用户的备注和截图内容，分析并填充以下字段：
   - 错误标签：从 计算错误/概念不清/审题失误/时间不足/知识盲区 中选择
   - 一句话错因：简洁描述为什么错
   - 正确方法：给出分步骤的解法
   - 下次重做日期：今天 + 7 天
   - 连续做对次数：默认为 0

严格按以下格式输出，不要添加额外内容：

- 编号：（占位，用户自己填或后续补）
- 一级模块：
- 二级题型：
- 三级题型：
- 题目位置：
- 我的答案：
- 正确答案：
- 首次耗时：
- 错误标签：
- 一句话错因：
- 正确方法：
- 下次重做日期：
- 连续做对次数：0
```

## 数据流

```
截图 (PIL Image) + 备注 (str)
    │
    ▼
app.py: 构建 messages = [
  {"role": "system", "content": SYSTEM_PROMPT},
  {"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    {"type": "text", "text": "行测/数量关系/工程问题 P23 耗时2min"}
  ]}
]
    │
    ▼
model_client.py: openai.ChatCompletion.create(
  model="qwen-vl-plus" | "doubao-1.5-vision-pro-250428",
  messages=messages,
  max_tokens=1000
)
    │
    ▼
app.py: 返回 markdown 文本 → 显示在 Gradio 聊天中
    │
    ▼
formatter.py: 收集所有回复 → 拼接为汇总 markdown → 导出文件
```

## 错误处理

| 场景 | 处理方式 |
|------|---------|
| API 调用超时/失败 | 聊天中显示"整理失败：{错误信息}，请重试"，不丢失已有结果 |
| API Key 未配置 | 启动时 `.env` 检查，缺失则退出并提示 |
| 截图质量差/OCR 失败 | AI 返回尽量识别的内容，标注不确定部分；用户可手动补充 |
| 模型返回格式不标准 | 显示原始返回内容，用户可编辑或重试 |
| 导出时无内容 | 按钮响应"暂无错题记录" |

## 配置

```bash
# .env.example
QWEN_API_KEY=sk-xxx           # 通义千问 API Key（阿里云百炼）
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DOUBAO_API_KEY=               # （可选）豆包 API Key（火山引擎）
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DEFAULT_MODEL=qwen-vl-plus    # 默认模型
```

## 非功能需求

- **不联网可跑**：除模型 API 调用外，全部本地运行（调用 API 需国内网络，无需 VPN）
- **不存储数据**：聊天记录仅内存保存，关闭即清；导出文件由用户明确触发
- **不收集个人信息**：截图和备注仅在 API 调用时传输给模型服务商
- **启动简单**：`pip install -r requirements.txt && python app.py`

## 不在本次范围内

- 不做题库管理/搜索/统计
- 不做 Obsidian 写入（那是 Codex 的活）
- 不部署到公网（纯本地工具）
- 不做用户认证/多用户
- 不接入知识库/向量检索