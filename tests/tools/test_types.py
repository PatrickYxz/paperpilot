from paperpilot.tools.types import Tool


def test_tool_exposes_framework_neutral_mcp_contract() -> None:
    tool = Tool(
        name="mcp__test__echo",
        description="Echo input",
        input_schema={"type": "object"},
        handler=lambda args: {"echo": args["value"]},
    )

    assert tool.name == "mcp__test__echo"
    assert tool.handler({"value": "ok"}) == {"echo": "ok"}
