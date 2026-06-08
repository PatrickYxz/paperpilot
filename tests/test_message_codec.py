"""Message codec tests."""
from __future__ import annotations

from dataclasses import dataclass

from paperpilot.message_codec import decode_messages, encode_messages


@dataclass
class FakeTextBlock:
    type: str
    text: str


@dataclass
class FakeToolUseBlock:
    type: str
    id: str
    name: str
    input: dict


@dataclass
class FakeUnknownBlock:
    type: str
    value: dict


def test_string_message_roundtrip():
    messages = [{"role": "user", "content": "hello"}]

    assert decode_messages(encode_messages(messages)) == messages


def test_sdk_like_text_and_tool_use_blocks_encode_to_dicts():
    messages = [
        {
            "role": "assistant",
            "content": [
                FakeTextBlock(type="text", text="I will download the paper."),
                FakeToolUseBlock(
                    type="tool_use",
                    id="toolu_123",
                    name="mcp__arxiv__download_paper",
                    input={"arxiv_id": "1706.03762"},
                ),
            ],
        }
    ]

    decoded = decode_messages(encode_messages(messages))

    assert decoded == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will download the paper."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "mcp__arxiv__download_paper",
                    "input": {"arxiv_id": "1706.03762"},
                },
            ],
        }
    ]


def test_tool_result_pair_preserves_ids():
    messages = [
        {
            "role": "assistant",
            "content": [
                FakeToolUseBlock(
                    type="tool_use",
                    id="call-1",
                    name="ask_user",
                    input={"question": "Which paper?"},
                )
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": "1706.03762",
                    "is_error": False,
                }
            ],
        },
    ]

    decoded = decode_messages(encode_messages(messages))

    assert decoded[0]["content"][0]["id"] == "call-1"
    assert decoded[1]["content"][0]["tool_use_id"] == "call-1"


def test_unknown_block_type_keeps_payload_as_dict():
    messages = [
        {
            "role": "assistant",
            "content": [FakeUnknownBlock(type="thinking", value={"x": 1})],
        }
    ]

    decoded = decode_messages(encode_messages(messages))

    assert decoded == [
        {
            "role": "assistant",
            "content": [{"type": "thinking", "value": {"x": 1}}],
        }
    ]

