from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


FORBIDDEN_MODULES = {
    "paperpilot.agent",
    "paperpilot.builtin_tools",
    "paperpilot.core",
    "paperpilot.main",
    "paperpilot.conversation",
    "paperpilot.bulk_input",
    "paperpilot.document_store",
    "paperpilot.message_codec",
    "paperpilot.session_store",
    "paperpilot.web.workflow",
    "paperpilot.web.event_mapper",
    "paperpilot.web.eval_summary",
    "paperpilot.web.pagination",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "paperpilot"


def _is_forbidden(module_name: str) -> bool:
    return any(
        module_name == forbidden or module_name.startswith(f"{forbidden}.")
        for forbidden in FORBIDDEN_MODULES
    )


def _resolve_import_from(path: Path, node: ast.ImportFrom) -> str | None:
    if not node.module and not node.level:
        return None
    if not node.level:
        return node.module

    module_parts = path.relative_to(PROJECT_ROOT).with_suffix("").parts[:-1]
    parent_count = node.level - 1
    if parent_count > len(module_parts):
        return node.module
    base_parts = module_parts[: len(module_parts) - parent_count]
    if node.module:
        base_parts = (*base_parts, *node.module.split("."))
    return ".".join(base_parts)


def _forbidden_references(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    references: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            references.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_name = _resolve_import_from(path, node)
            if module_name:
                references.add(module_name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            references.add(node.value.strip())
    return {reference for reference in references if _is_forbidden(reference)}


def test_forbidden_legacy_modules_are_absent() -> None:
    for name in FORBIDDEN_MODULES:
        assert importlib.util.find_spec(name) is None


def test_retained_package_does_not_reference_forbidden_legacy_modules() -> None:
    violations = {
        str(path.relative_to(PROJECT_ROOT)): sorted(references)
        for path in PACKAGE_ROOT.rglob("*.py")
        if (references := _forbidden_references(path))
    }

    assert violations == {}
