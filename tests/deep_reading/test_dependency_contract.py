from importlib.metadata import version


def _major_minor(distribution: str) -> tuple[int, int]:
    major, minor, *_rest = version(distribution).split(".")
    return int(major), int(minor)


def test_langgraph_stack_uses_approved_minor_lines() -> None:
    assert _major_minor("langgraph") == (1, 2)
    assert _major_minor("langchain") == (1, 3)
    assert _major_minor("langgraph-checkpoint-sqlite") == (3, 1)
    assert _major_minor("langchain-deepseek") == (1, 1)
