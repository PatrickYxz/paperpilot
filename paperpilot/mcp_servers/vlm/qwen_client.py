"""DashScope Qwen-VL-Max wrapper."""
from __future__ import annotations

import base64
import os

import dashscope
from dashscope import MultiModalConversation
from dotenv import load_dotenv


class QwenAPIError(RuntimeError):
    """DashScope returned a non-200 status."""


class QwenClient:
    MODEL = "qwen-vl-max"

    def __init__(self, api_key: str | None = None) -> None:
        load_dotenv()
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
