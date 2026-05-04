# Day 14: `vlm-mcp` Qwen-VL-Max 图表理解 设计

**目标:** 第 4 个 MCP server。LLM 给定 `arxiv_id + page_num + query`,vlm-mcp 内部下 PDF / fitz render 该页为 PNG / 喂 Qwen-VL-Max,返回 text。同时新增 `analyze-figures` skill 引导"先文本定位再调 VLM"工作流。

**Why:** 论文里大量信息在 figure / table / architecture diagram 里,纯 colbert 文本检索拿不到。Qwen-VL-Max 能看图说话,补齐多模态短板;同时是简历亮点关键词(VLM / 多模态 Agent)。

**Scope:** 一个新 MCP server(单 tool)+ 一个新 skill + 一行 manifest。不改主 agent / loop / adapter / subagent,红线守恒。

---

## §1 设计决策(brainstorm 沉淀)

| Q | 选择 | 理由 |
|---|---|---|
| Q1: tool 接口边界 | B `understand_paper_page(arxiv_id, page_num, query)` | LLM 整链路只用 arxiv_id 和页码,跟 colbert.search / download_paper 风格一致;整页 render 让 VLM 自己定位图表,绕开 PyMuPDF 抽 figure bbox 多列论文上易错位 |
| Q2: VLM 提供方 | A DashScope 原生 SDK + Qwen-VL-Max | 一行 `MultiModalConversation.call`,加一个 env 变量,不污染 LLMClient 的 anthropic-only 路径 |
| Q3: 缓存策略 | B 持久化 PNG + 进程内 PDF bytes dict | render 在大 PDF 上 1-2s,重复访问同页常见;PDF 大不入盘;不缓存 VLM 响应避免 LLM 决策被旧答案误导 |
| Q4: skill 暴露 | A 新建 `analyze-figures` skill | 引导"先文本定位再调 VLM",不在 deep-read-paper 里横插 if-else 风格 prose |

**红线守恒:**
- `paperpilot/main.py` / `paperpilot/core/loop.py` / `paperpilot/core/adapter.py` / `paperpilot/builtin_tools/subagent.py` 一行不改
- 新 server 通过 manifest 注册,MCPClient 自动发现

---

## §2 文件结构

| 路径 | 动作 | 责任 |
|---|---|---|
| `paperpilot/mcp_servers/vlm/__init__.py` | 新建,空 | package marker |
| `paperpilot/mcp_servers/vlm/server.py` | 新建,~50 行 | FastMCP 协议层 + schema 校验 + dispatch |
| `paperpilot/mcp_servers/vlm/page_renderer.py` | 新建,~80 行 | PDF 下载 + fitz render + PNG 缓存 + 进程内 PDF bytes 缓存 |
| `paperpilot/mcp_servers/vlm/qwen_client.py` | 新建,~40 行 | DashScope SDK 包装,`(image_bytes, query) -> text` |
| `paperpilot/mcp_servers.json` | 改 1 处 | 加 "vlm" entry |
| `paperpilot/skills/analyze-figures.md` | 新建 | 引导工作流的 skill prose |
| `tests/mcp_servers/vlm/__init__.py` | 新建,空 | |
| `tests/mcp_servers/vlm/test_page_renderer.py` | 新建 | fitz/urllib mock + tmp_path,~5 case |
| `tests/mcp_servers/vlm/test_qwen_client.py` | 新建 | dashscope mock,~3 case |
| `tests/mcp_servers/vlm/test_vlm_server.py` | 新建 | renderer + qwen 全 mock,~4 case |
| `tests/mcp_servers/vlm/test_vlm_via_client.py` | 新建,@slow | 真起 vlm-mcp 进程 + 真 DashScope 调,1 case |
| `scripts/day14_smoke.py` | 新建 | 真 LLM 端到端 |

---

## §3 Tool 接口

```python
@mcp.tool()
def understand_paper_page(arxiv_id: str, page_num: int, query: str) -> str:
    """View one page of an arXiv paper as an image and answer a query about
    its visual content (figures, tables, architecture diagrams, plots).

    Args:
        arxiv_id: arXiv id, e.g. "1706.03762".
        page_num: 1-based page index. The first page is 1.
        query: What to look for, e.g. "describe Figure 3" or
               "what does the architecture diagram show".

    Returns:
        plain text description from Qwen-VL-Max.
    """
```

设计取舍:
- **1-based 页号:** 论文里通常这么写,LLM 直觉对齐;内部 `doc.load_page(page_num - 1)` 转
- **单 tool 无副:** 不暴露 list_pages / list_figures(YAGNI)
- **query 必填:** 强制 LLM 表达意图,VLM 输出更精准也更省 token

---

## §4 PageRenderer (`page_renderer.py`)

### 4.1 类设计

```python
class PageRenderer:
    PDF_DPI = 150
    CACHE_ROOT = Path("data/vlm_cache")

    def __init__(self) -> None:
        self.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self._pdf_bytes_mem: dict[str, bytes] = {}

    def get_page_png(self, arxiv_id: str, page_num: int) -> bytes:
        """Return PNG bytes for the given 1-based page. Cache-first."""
        png_path = self._png_path(arxiv_id, page_num)
        if png_path.exists():
            return png_path.read_bytes()

        pdf = self._get_pdf_bytes(arxiv_id)
        png_bytes = self._render_page(pdf, page_num)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(png_bytes)
        return png_bytes

    def _get_pdf_bytes(self, arxiv_id: str) -> bytes:
        if arxiv_id not in self._pdf_bytes_mem:
            self._pdf_bytes_mem[arxiv_id] = _fetch_pdf(arxiv_id)
        return self._pdf_bytes_mem[arxiv_id]

    def _render_page(self, pdf_bytes: bytes, page_num: int) -> bytes:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            if page_num < 1 or page_num > doc.page_count:
                raise PageOutOfRange(
                    f"page {page_num} out of 1..{doc.page_count}"
                )
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(dpi=self.PDF_DPI)
            return pix.tobytes("png")
        finally:
            doc.close()

    def _png_path(self, arxiv_id: str, page_num: int) -> Path:
        return self.CACHE_ROOT / _paper_key(arxiv_id) / f"page_{page_num}.png"
```

### 4.2 异常

```python
class ArxivNotFoundError(RuntimeError):
    """arxiv 返回 404 / 无效 id。"""

class PDFParseError(RuntimeError):
    """fitz 解析 PDF 字节失败。"""

class PageOutOfRange(RuntimeError):
    """page_num 超出 PDF 页数。"""
```

### 4.3 复用与 DRY 取舍

- **`_fetch_pdf`** 复制 `paperpilot/mcp_servers/arxiv.py` 的实现(~10 行,SSL/certifi/urllib/HTTPError 处理),不抽公共。理由:Day 14 是第二处用 arxiv PDF 下载,跨 MCP server 抽公共要建 `paperpilot/core/arxiv_pdf.py`,引入两个 server 跨依赖,YAGNI。Day 15+ 真出第三处再抽
- **`_paper_key`** 复制 `paperpilot/mcp_servers/colbert/index_manager.py` 的 sha 转义逻辑(~5 行)。同样理由:Day 15+ 真出第三处再抽

### 4.4 缓存键 / 路径

- `data/vlm_cache/<paper_key>/page_<N>.png`,`paper_key = sha10 + safe-regex(arxiv_id)`,与 colbert 同模式
- PNG 文件名**不带 DPI 标识**;若以后调 PDF_DPI 常量,旧 PNG 会被错误命中。v1 固定 150 DPI,§9 列入局限

---

## §5 QwenClient (`qwen_client.py`)

```python
import base64
import os

import dashscope
from dashscope import MultiModalConversation


class QwenAPIError(RuntimeError):
    """DashScope 返回非 200 status_code。"""


class QwenClient:
    MODEL = "qwen-vl-max"

    def __init__(self, api_key: str | None = None) -> None:
        dashscope.api_key = api_key or os.environ["DASHSCOPE_API_KEY"]

    def describe_page(self, png_bytes: bytes, query: str) -> str:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        response = MultiModalConversation.call(
            model=self.MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"image": f"data:image/png;base64,{b64}"},
                    {"text": query},
                ],
            }],
        )
        if response.status_code != 200:
            raise QwenAPIError(
                f"DashScope {response.status_code}: {response.message}"
            )
        return response.output.choices[0].message.content[0]["text"]
```

设计:
- 类比 LLMClient,轻包装,失败抛自定义异常;handler 捕不捕由 server.py 决定
- 不暴露 model 参数(YAGNI);如要切 qwen-vl-plus 后期改常量
- 不做 retry / 超时 — DashScope SDK 默认值够用,失败让 LLM 决定换页码或放弃

---

## §6 Server (`server.py`)

```python
from mcp.server.fastmcp import FastMCP

from paperpilot.mcp_servers.vlm.page_renderer import PageRenderer
from paperpilot.mcp_servers.vlm.qwen_client import QwenClient

mcp = FastMCP("vlm")
_renderer: PageRenderer | None = None
_qwen: QwenClient | None = None


@mcp.tool()
def understand_paper_page(arxiv_id: str, page_num: int, query: str) -> str:
    """[docstring as §3]"""
    return _impl(arxiv_id, page_num, query)


def _impl(arxiv_id: str, page_num: int, query: str) -> str:
    if not isinstance(arxiv_id, str) or not arxiv_id.strip():
        raise ValueError("arxiv_id must be a non-empty string")
    if not isinstance(page_num, int) or page_num < 1:
        raise ValueError(f"page_num must be int >= 1, got {page_num!r}")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    assert _renderer is not None, "PageRenderer not initialized"
    assert _qwen is not None, "QwenClient not initialized"

    png_bytes = _renderer.get_page_png(arxiv_id, page_num)
    return _qwen.describe_page(png_bytes, query)


if __name__ == "__main__":
    _renderer = PageRenderer()
    _qwen = QwenClient()
    mcp.run(transport="stdio")
```

---

## §7 Manifest (`paperpilot/mcp_servers.json`)

加一条:
```json
"vlm": {
  "command": "python",
  "args": ["-m", "paperpilot.mcp_servers.vlm.server"]
}
```

`MCPClient` 启动时自动 list_tools,LLM 在 system prompt 看到 `mcp__vlm__understand_paper_page`,无需改 main.py。

---

## §8 Skill (`paperpilot/skills/analyze-figures.md`)

```markdown
---
name: analyze-figures
description: 看论文里的 figure / table / architecture diagram:先文本定位再调 VLM
when_to_use: 用户问某篇 paper 的 figure / table / 架构图含义,或要解释画的是什么
---

# Analyze Figures

## 适用场景
- 用户给定 arxiv_id 问 "Figure N 画的是什么 / Table M 数据怎么读 / 架构图怎么理解"
- 用户问"这篇 paper 的 X 模块结构"且文本不足以说清

## 步骤
1. 先用 colbert 文本定位:
   - `mcp__arxiv__download_paper(arxiv_id="...")`
   - `mcp__colbert__build_index(documents=[<download 返回值>])`
   - `mcp__colbert__search(query="Figure N", paper_id="...")` 找到 caption / 引用上下文,从中读出页码
2. 调 `mcp__vlm__understand_paper_page(arxiv_id="...", page_num=<step1 拿到的页>, query="describe Figure N in detail")`
3. 综合 VLM 描述 + colbert 上下文回答

## 注意
- 不要不经文本定位就乱调 VLM,VLM 比 colbert 贵
- 一次 understand_paper_page 只看一页;跨页或对比多页就分多次调
- 用户只问 paper 整体内容时用 deep-read-paper,不要用本 skill
```

---

## §9 测试矩阵

### 9.1 Fast unit

**`test_page_renderer.py`**(fitz / urllib mock + tmp_path):
1. `test_cold_render_writes_png_and_returns_bytes` — 第一次调 `get_page_png` → mock urllib 返 PDF bytes / mock fitz render 返 PNG bytes → 断 PNG 文件落盘 + 返回值正确
2. `test_disk_hit_skips_render` — 预先把 PNG 写入缓存 → mock fitz 不应被调
3. `test_pdf_bytes_memo_skips_second_download` — 同 PageRenderer 实例对同 arxiv_id 不同页连调两次 → urllib 只被调一次
4. `test_page_num_out_of_range_raises` — mock fitz 返 page_count=5,调 page_num=10 → 抛 PageOutOfRange
5. `test_paper_key_escapes_path_separators` — `arxiv_id="cs/0501001"` 不应在文件系统出现 `/`

**`test_qwen_client.py`**(dashscope.MultiModalConversation 整体 monkeypatch):
1. `test_describe_page_returns_text_on_200` — mock 返 status_code=200 + 嵌套 output → 取出 text
2. `test_non_200_raises_qwen_api_error` — mock 返 401 → QwenAPIError 含状态码
3. `test_image_b64_payload_structure` — 断 call 时 messages content[0] 是 `{"image":"data:image/png;base64,..."}`,content[1] 是 `{"text":query}`

**`test_vlm_server.py`**(`_renderer` / `_qwen` monkeypatch):
1. `test_impl_routes_to_renderer_and_qwen` — mock 两者,调 `_impl("X", 3, "q")` → 断 `renderer.get_page_png("X", 3)` + `qwen.describe_page(<那 bytes>, "q")` 都被调,返 qwen 返回值
2. `test_impl_rejects_empty_arxiv_id` — `ValueError`
3. `test_impl_rejects_zero_page_num` — `ValueError`
4. `test_impl_rejects_empty_query` — `ValueError`

### 9.2 Slow integration(`test_vlm_via_client.py`)

```python
@pytest.mark.slow
def test_understand_first_page_of_attention_paper(tmp_path):
    """真起 vlm-mcp 进程 + 真 DashScope 调,确认链路通。"""
    c = MCPClient(_make_manifest(tmp_path))   # 仅含 vlm 一个 server,page_renderer 内部自己下 PDF
    c.start()
    try:
        tool = next(t for t in c.list_tools() if t.name == "mcp__vlm__understand_paper_page")
        result = tool.handler({
            "arxiv_id": "1706.03762",
            "page_num": 3,           # Attention paper 架构图所在页
            "query": "describe the model architecture diagram",
        })
        assert isinstance(result, str) and len(result) > 50
        assert any(kw in result.lower() for kw in ("attention", "encoder", "decoder"))
    finally:
        c.close()
```

需要环境有 `DASHSCOPE_API_KEY`;CI 不跑,本地手动跑。

### 9.3 Smoke (`scripts/day14_smoke.py`)

真 LLM 端到端:
- prompt:`"我想了解 arxiv 2010.11929 (Vision Transformer) 的 Figure 1 画的是什么。请按 analyze-figures skill 操作。"`
- 断言:tracer 看到
  - `load_skill("analyze-figures")`
  - `mcp__colbert__search`(任意,验证文本定位)
  - `mcp__vlm__understand_paper_page` ≥ 1 次
  - 最终回答含 "patch" 或 "16x16" 或 "linear projection" 之一(ViT 经典关键词)

---

## §10 错误处理 / 已知局限

| 项 | 描述 | 后续 |
|---|---|---|
| arxiv PDF 拉取失败 | 复用 `ArxivNotFoundError`(同字符串复制一份);LLM 拿到 error 决定换 id 或放弃 | Day 15+ 第三处用时抽公共 |
| fitz render 异常 | `PDFParseError` 透传 | — |
| DashScope 限流 / 超时 | `QwenAPIError` 透传 | LLM 自决重试或换页;不在 client 层做 retry |
| `_pdf_bytes_mem` 进程内一直涨 | 长会话理论可累积几十 MB | 真撞撞再加 LRU |
| PNG 缓存不带 DPI 标识 | 调 PDF_DPI 常量后旧 PNG 命中错 | v1 固定 150 DPI;调时手删 `data/vlm_cache/` |
| subagent 不享受 vlm | `SUBAGENT_TOOL_NAMES` 不加 vlm,paper_deep_read 子 agent 仍只看文本 | 等真出"子 agent 也要看图"需求再加 |
| `_paper_key` / `_fetch_pdf` 复制 | DRY 暂时破,Day 14 是第二处 | 第三处再抽到 `paperpilot/core/` |

---

## §11 DoD

- [ ] vlm 子目录 4 个 .py 文件齐(`__init__` 空 / `server` / `page_renderer` / `qwen_client`)
- [ ] `paperpilot/mcp_servers.json` 加 vlm entry
- [ ] `paperpilot/skills/analyze-figures.md` 新建
- [ ] 3 个 fast test 文件全绿(预计 5+3+4 = 12 case)
- [ ] 1 个 slow 测试本地手动 PASS(`pytest -m slow tests/mcp_servers/vlm/test_vlm_via_client.py`)
- [ ] `python scripts/day14_smoke.py` 真 LLM 端到端 PASS
- [ ] `git diff main -- paperpilot/main.py paperpilot/core/loop.py paperpilot/core/adapter.py paperpilot/builtin_tools/subagent.py` 空
- [ ] `requirements.txt`(或 `pyproject.toml`)加 `dashscope`
