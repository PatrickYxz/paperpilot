"""ss_client unit tests."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from paperpilot.mcp_servers.graph.ss_client import (
    SSAPIError,
    SSClient,
    SSRateLimitError,
)


FIXTURE = Path(__file__).parent.parent / "fixtures" / "ss_attention_bert.json"


def _load_fixture() -> list[dict | None]:
    with FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


def _mock_response(
    status_code: int, json_data: list[dict | None] | None = None
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or []
    if 400 <= status_code < 600:
        req = httpx.Request(
            "POST", "https://api.semanticscholar.org/graph/v1/paper/batch"
        )
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=req, response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_fetch_papers_batch_happy(monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("PAPERPILOT_SS_FIXTURE_DIR", raising=False)
    fixture_data = _load_fixture()
    client = SSClient()

    with patch.object(
        client._http, "post", return_value=_mock_response(200, fixture_data)
    ) as post:
        result = client.fetch_papers_batch(["2401.00001", "2401.00002"])

    assert result == fixture_data
    assert post.call_count == 1
    call = post.call_args
    assert "fields" in call.kwargs["params"]
    assert call.kwargs["json"] == {"ids": ["ARXIV:2401.00001", "ARXIV:2401.00002"]}


def test_fetch_429_retry_then_success(monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    fixture_data = _load_fixture()
    client = SSClient()
    responses = [_mock_response(429), _mock_response(200, fixture_data)]

    with patch.object(client._http, "post", side_effect=responses) as post, patch(
        "paperpilot.mcp_servers.graph.ss_client.time.sleep"
    ) as sleep:
        result = client.fetch_papers_batch(["2401.00001", "2401.00002"])

    assert result == fixture_data
    assert post.call_count == 2
    assert sleep.call_args_list[0][0][0] == 2.0


def test_fetch_429_retry_exhaust(monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    client = SSClient()

    with patch.object(
        client._http,
        "post",
        side_effect=[_mock_response(429), _mock_response(429), _mock_response(429)],
    ) as post, patch("paperpilot.mcp_servers.graph.ss_client.time.sleep") as sleep:
        with pytest.raises(SSRateLimitError):
            client.fetch_papers_batch(["2401.00001"])

    assert post.call_count == 3
    assert [c[0][0] for c in sleep.call_args_list] == [2.0, 4.0]


def test_fetch_5xx_retry(monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    fixture_data = _load_fixture()
    client = SSClient()

    with patch.object(
        client._http,
        "post",
        side_effect=[_mock_response(503), _mock_response(200, fixture_data)],
    ), patch("paperpilot.mcp_servers.graph.ss_client.time.sleep"):
        result = client.fetch_papers_batch(["2401.00001"])

    assert result == fixture_data


def test_fetch_4xx_no_retry(monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    client = SSClient()

    with patch.object(client._http, "post", return_value=_mock_response(400)) as post, patch(
        "paperpilot.mcp_servers.graph.ss_client.time.sleep"
    ) as sleep:
        with pytest.raises(SSAPIError):
            client.fetch_papers_batch(["bad_id"])

    assert post.call_count == 1
    assert sleep.call_count == 0


def test_api_key_header(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")
    fixture_data = _load_fixture()
    client = SSClient()

    with patch.object(
        client._http, "post", return_value=_mock_response(200, fixture_data)
    ) as post:
        client.fetch_papers_batch(["2401.00001"])

    assert post.call_args.kwargs["headers"]["x-api-key"] == "test-key"


def test_fixture_mode(monkeypatch, tmp_path):
    fixture_dir = tmp_path
    (fixture_dir / "ss_attention_bert.json").write_text(
        json.dumps(_load_fixture()), encoding="utf-8"
    )
    monkeypatch.setenv("PAPERPILOT_SS_FIXTURE_DIR", str(fixture_dir))
    client = SSClient()

    with patch.object(client._http, "post") as post:
        result = client.fetch_papers_batch(["2401.00001", "missing-id"])

    assert post.call_count == 0
    assert result[0] is not None
    assert result[1] is None
