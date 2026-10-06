"""Reviewer model routing tests."""
from __future__ import annotations

from paperpilot.user_memory.reviewer_model import (
    DashScopeChatModel,
    DeepSeekReviewerModel,
    build_reviewer_model_from_env,
)


def test_unset_disables_reviewer(monkeypatch):
    monkeypatch.delenv("PAPERPILOT_MEMORY_REVIEWER_MODEL", raising=False)
    assert build_reviewer_model_from_env() is None


def test_off_disables_reviewer(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_MEMORY_REVIEWER_MODEL", "off")
    assert build_reviewer_model_from_env() is None


def test_qwen_routes_to_dashscope(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_MEMORY_REVIEWER_MODEL", "qwen-plus")
    model = build_reviewer_model_from_env()
    assert isinstance(model, DashScopeChatModel)
    assert model.model_name == "qwen-plus"


def test_deepseek_name_routes_to_deepseek_reviewer(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_MEMORY_REVIEWER_MODEL", "deepseek-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    model = build_reviewer_model_from_env()
    assert isinstance(model, DeepSeekReviewerModel)
