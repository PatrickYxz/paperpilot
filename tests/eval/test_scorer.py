"""Tests for paperpilot.eval.scorer."""
from paperpilot.eval.scorer import cluster_failures, is_pass


def test_is_pass_contains_simple() -> None:
    assert is_pass("the dataset is SQuAD 2.0", ["SQuAD 2.0"])


def test_is_pass_case_insensitive() -> None:
    assert is_pass("we use squad 2.0 for training", ["SQuAD 2.0"])


def test_is_pass_strips_punct() -> None:
    assert is_pass("Answer: SQuAD.", ["SQuAD"])


def test_is_pass_normalizes_whitespace() -> None:
    assert is_pass("uses\nSQuAD\t2.0", ["SQuAD 2.0"])


def test_is_pass_multi_oracle_any_match() -> None:
    assert is_pass("uses MNLI dataset", ["SNLI", "MNLI"])


def test_is_pass_no_match() -> None:
    assert not is_pass("uses CoLA", ["SQuAD", "MNLI"])


def test_is_pass_empty_oracle_filtered() -> None:
    assert not is_pass("anything", ["", "  "])


def test_cluster_failures_bucket_priority() -> None:
    records = [
        {"passed": False, "tool_calls": ["mcp__arxiv__search_papers"]},
        {"passed": False, "tool_calls": ["load_skill"]},
        {"passed": False, "tool_calls": [
            "load_skill", "mcp__arxiv__download_paper",
            "mcp__colbert__build_index",
        ]},
        {"passed": False, "tool_calls": [
            "load_skill", "mcp__arxiv__download_paper",
            "mcp__colbert__build_index",
            "mcp__colbert__search", "mcp__colbert__search",
        ]},
        {"passed": False, "tool_calls": [
            "load_skill", "mcp__arxiv__download_paper",
            "mcp__colbert__build_index",
            "mcp__colbert__search", "mcp__colbert__search", "mcp__colbert__search",
        ]},
        {"passed": False, "tool_calls": ["load_skill"], "error": "GuardrailStop: max_iter"},
        {"passed": True, "tool_calls": []},
    ]
    counts = cluster_failures(records)
    assert counts == {
        "no_load_skill": 1,
        "no_download": 1,
        "no_colbert_search": 1,
        "colbert_searched_low": 1,
        "synthesis_miss": 1,
        "iter_exhausted": 1,
    }
