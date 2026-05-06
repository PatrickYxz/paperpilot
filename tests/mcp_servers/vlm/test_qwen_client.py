"""QwenClient unit tests. dashscope.MultiModalConversation.call is mocked."""
from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paperpilot.mcp_servers.vlm import qwen_client as qc_module


def _ok_response(text: str) -> SimpleNamespace:
    """Mimic DashScope's nested response object."""
    return SimpleNamespace(
        status_code=200,
        message="",
        output=SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=[{"text": text}])
            )]
        ),
    )


def test_describe_page_returns_text_on_200(monkeypatch):
    fake_call = MagicMock(return_value=_ok_response("a transformer diagram"))
    monkeypatch.setattr(qc_module.MultiModalConversation, "call", fake_call)

    client = qc_module.QwenClient(api_key="test-key")
    out = client.describe_page(b"FAKEPNG", "describe Figure 1")
    assert out == "a transformer diagram"
    fake_call.assert_called_once()


def test_non_200_raises_qwen_api_error(monkeypatch):
    fake_call = MagicMock(return_value=SimpleNamespace(
        status_code=401,
        message="invalid key",
        output=None,
    ))
    monkeypatch.setattr(qc_module.MultiModalConversation, "call", fake_call)

    client = qc_module.QwenClient(api_key="bad-key")
    with pytest.raises(qc_module.QwenAPIError, match="401"):
        client.describe_page(b"FAKE", "q")


def test_image_b64_payload_structure(monkeypatch):
    fake_call = MagicMock(return_value=_ok_response("ok"))
    monkeypatch.setattr(qc_module.MultiModalConversation, "call", fake_call)

    client = qc_module.QwenClient(api_key="test-key")
    client.describe_page(b"PNGBYTES", "describe")

    kwargs = fake_call.call_args.kwargs
    assert kwargs["model"] == "qwen-vl-max"
    content = kwargs["messages"][0]["content"]
    assert content[0]["image"].startswith("data:image/png;base64,")
    expected_b64 = base64.b64encode(b"PNGBYTES").decode("ascii")
    assert content[0]["image"].endswith(expected_b64)
    assert content[1] == {"text": "describe"}
