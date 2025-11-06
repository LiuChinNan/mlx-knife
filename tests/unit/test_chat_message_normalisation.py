import pytest

from mlx_knife.server import ChatMessage, format_chat_messages_for_runner


def test_tool_message_role_normalisation():
    messages = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="tool", content="result output", name="search_tool"),
    ]

    formatted = format_chat_messages_for_runner(messages)
    assert formatted[1]["role"] == "assistant"
    assert "search_tool" in formatted[1]["content"]
    assert "result output" in formatted[1]["content"]


def test_multi_part_message_content_list():
    content = [
        {"type": "text", "text": "Part A"},
        {"type": "text", "text": "Part B"},
        {"type": "image", "url": "http://example.com/example.png"},
    ]
    messages = [ChatMessage(role="user", content=content)]

    formatted = format_chat_messages_for_runner(messages)
    combined = formatted[0]["content"]
    assert "Part A" in combined and "Part B" in combined
    assert "example.png" in combined


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, ""),
        ({"type": "text", "text": "json"}, '{"type": "text", "text": "json"}'),
        (123, "123"),
    ],
)
def test_misc_content_conversion(raw, expected):
    message = ChatMessage(role="assistant", content=raw)
    formatted = format_chat_messages_for_runner([message])
    assert formatted[0]["content"] == expected
