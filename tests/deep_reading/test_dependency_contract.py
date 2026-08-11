from importlib.metadata import version
from pathlib import Path


def _major_minor(distribution: str) -> tuple[int, int]:
    major, minor, *_rest = version(distribution).split(".")
    return int(major), int(minor)


def test_langgraph_stack_uses_approved_minor_lines() -> None:
    assert _major_minor("langgraph") == (1, 2)
    assert _major_minor("langchain") == (1, 3)
    assert _major_minor("langgraph-checkpoint-sqlite") == (3, 1)
    assert _major_minor("langchain-deepseek") == (1, 1)


def test_legacy_anthropic_sdk_is_not_a_direct_requirement() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert not any(
        line.strip().startswith("anthropic")
        for line in requirements.splitlines()
    )


def test_lock_is_generated_for_python_312_across_supported_platforms() -> None:
    header = Path("requirements-lock.txt").read_text(encoding="utf-8").splitlines()[:3]
    command = " ".join(header)
    assert "--universal" in command
    assert "--python-version 3.12" in command
