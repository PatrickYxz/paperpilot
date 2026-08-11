from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


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


def _static_string(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left)
        right = _static_string(node.right)
        if left is not None and right is not None:
            return left + right
        return None
    if not isinstance(node, ast.JoinedStr):
        return None

    parts: list[str] = []
    for part in node.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            parts.append(part.value)
            continue
        if not isinstance(part, ast.FormattedValue):
            return None
        rendered = _static_formatted_value(part)
        if rendered is None:
            return None
        parts.append(rendered)
    return "".join(parts)


def _static_formatted_value(node: ast.FormattedValue) -> str | None:
    value = _static_string(node.value)
    if value is None:
        return None

    if node.conversion == ord("r"):
        value = repr(value)
    elif node.conversion == ord("a"):
        value = ascii(value)
    elif node.conversion in {-1, ord("s")}:
        value = str(value)
    else:
        return None

    format_spec = ""
    if node.format_spec is not None:
        format_spec = _static_string(node.format_spec)
        if format_spec is None:
            return None
    try:
        return format(value, format_spec)
    except (TypeError, ValueError):
        return None


def _importlib_bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    function_bindings: set[str] = set()
    module_bindings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_bindings.update(
                alias.asname or "importlib"
                for alias in node.names
                if alias.name == "importlib"
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "importlib"
        ):
            function_bindings.update(
                alias.asname or "import_module"
                for alias in node.names
                if alias.name == "import_module"
            )
    return function_bindings, module_bindings


def _dynamic_import_argument(
    node: ast.Call,
    *,
    function_bindings: set[str],
    module_bindings: set[str],
) -> ast.expr | None:
    is_dynamic_import = (
        isinstance(node.func, ast.Name)
        and (
            node.func.id == "__import__"
            or node.func.id in function_bindings
        )
    ) or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in module_bindings
    )
    if not is_dynamic_import:
        return None
    if node.args:
        return node.args[0]
    return next(
        (keyword.value for keyword in node.keywords if keyword.arg == "name"),
        None,
    )


def _forbidden_references(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function_bindings, module_bindings = _importlib_bindings(tree)
    references: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            references.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_name = _resolve_import_from(path, node)
            if module_name:
                references.add(module_name)
                references.update(
                    f"{module_name}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
        elif isinstance(node, ast.Call):
            argument = _dynamic_import_argument(
                node,
                function_bindings=function_bindings,
                module_bindings=module_bindings,
            )
            if argument is not None and (module_name := _static_string(argument)):
                references.add(module_name.strip())
    return {reference for reference in references if _is_forbidden(reference)}


def _scan_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    relative_path: str,
    source: str,
) -> set[str]:
    monkeypatch.setitem(globals(), "PROJECT_ROOT", tmp_path)
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return _forbidden_references(path)


@pytest.mark.parametrize(
    ("relative_path", "source", "expected"),
    [
        (
            "paperpilot/module.py",
            "from paperpilot import core\n",
            {"paperpilot.core"},
        ),
        (
            "paperpilot/module.py",
            "from paperpilot import agent\n",
            {"paperpilot.agent"},
        ),
        (
            "paperpilot/module.py",
            "from . import core\n",
            {"paperpilot.core"},
        ),
        (
            "paperpilot/web/module.py",
            "from . import workflow\n",
            {"paperpilot.web.workflow"},
        ),
        (
            "paperpilot/web/module.py",
            "from .. import core\n",
            {"paperpilot.core"},
        ),
    ],
)
def test_scanner_qualifies_import_from_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    source: str,
    expected: set[str],
) -> None:
    assert _scan_source(
        tmp_path,
        monkeypatch,
        relative_path=relative_path,
        source=source,
    ) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            'from importlib import import_module\n'
            'import_module("paperpilot." + "agent")\n',
            {"paperpilot.agent"},
        ),
        (
            'from importlib import import_module\n'
            'import_module(f"paperpilot.core")\n',
            {"paperpilot.core"},
        ),
        (
            'from importlib import import_module\n'
            'import_module(f"paperpilot.{\'core\'}")\n',
            {"paperpilot.core"},
        ),
        (
            'from importlib import import_module\n'
            'import_module(f"{\'paperpilot.\' + \'agent\'}")\n',
            {"paperpilot.agent"},
        ),
        (
            '__import__("paperpilot." + "core")\n',
            {"paperpilot.core"},
        ),
    ],
)
def test_scanner_folds_static_dynamic_import_strings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    expected: set[str],
) -> None:
    assert _scan_source(
        tmp_path,
        monkeypatch,
        relative_path="paperpilot/module.py",
        source=source,
    ) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from importlib import import_module as load_module\n"
            'load_module("paperpilot.agent")\n',
            {"paperpilot.agent"},
        ),
        (
            "from importlib import import_module as load_module\n"
            'load_module(name="paperpilot.core")\n',
            {"paperpilot.core"},
        ),
        (
            "import importlib as il\n"
            'il.import_module(name="paperpilot.core")\n',
            {"paperpilot.core"},
        ),
        (
            "def import_module(name):\n"
            "    return name\n"
            'import_module("paperpilot.agent")\n',
            set(),
        ),
        (
            'registry.import_module("paperpilot.core")\n',
            set(),
        ),
        (
            "import utilities as il\n"
            'il.import_module("paperpilot.agent")\n',
            set(),
        ),
    ],
)
def test_scanner_requires_real_importlib_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    expected: set[str],
) -> None:
    assert _scan_source(
        tmp_path,
        monkeypatch,
        relative_path="paperpilot/module.py",
        source=source,
    ) == expected


def test_scanner_ignores_forbidden_module_names_used_only_as_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _scan_source(
        tmp_path,
        monkeypatch,
        relative_path="paperpilot/module.py",
        source='EXAMPLE_MODULES = {"paperpilot.agent", "paperpilot.core"}\n',
    ) == set()


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
