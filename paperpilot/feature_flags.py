"""Runtime feature flags for ablation experiments (book ch.6).

Every major feature is independently switchable so its real contribution
can be measured. Flags are read at use sites (never captured in module
constants), letting an in-process ablation matrix toggle them dynamically.
"""
from __future__ import annotations

import os

MEMORY_TOOL_FLAG = "PAPERPILOT_MEMORY_TOOL_ENABLED"
PROFILE_INJECTION_FLAG = "PAPERPILOT_PROFILE_INJECTION_ENABLED"
COMPUTATION_TOOL_FLAG = "PAPERPILOT_COMPUTATION_TOOL_ENABLED"
RETRIEVAL_MODE_FLAG = "PAPERPILOT_RETRIEVAL_MODE"

ALL_FLAGS = (
    MEMORY_TOOL_FLAG,
    PROFILE_INJECTION_FLAG,
    COMPUTATION_TOOL_FLAG,
    RETRIEVAL_MODE_FLAG,
)


def flag_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "off", "disabled"}


def memory_tool_enabled() -> bool:
    return flag_enabled(MEMORY_TOOL_FLAG)


def profile_injection_enabled() -> bool:
    return flag_enabled(PROFILE_INJECTION_FLAG)


def computation_tool_enabled() -> bool:
    return flag_enabled(COMPUTATION_TOOL_FLAG)


def retrieval_mode() -> str:
    """Default search mode for the colbert server: hybrid|dense."""
    raw = os.environ.get(RETRIEVAL_MODE_FLAG, "").strip().lower()
    return raw if raw in {"hybrid", "dense"} else "hybrid"
