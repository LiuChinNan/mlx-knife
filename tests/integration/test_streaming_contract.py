import json
from typing import List

import pytest
from fastapi.testclient import TestClient

from mlx_knife import server


class StubRunner:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens

    def generate_streaming(self, **_: object):
        for token in self.tokens:
            yield token

    def get_effective_max_tokens(self, requested: int | None, interactive: bool = False) -> int:
        return requested or 32

    def _format_conversation(self, messages, use_chat_template: bool = True) -> str:
        return "\n".join(entry["content"] for entry in messages)


@pytest.fixture
def api_client():
    return TestClient(server.app)


def _collect_events(response) -> List[str]:
    chunks: List[str] = []
    for chunk in response.iter_text():
        if chunk:
            chunks.extend(line for line in chunk.splitlines() if line.startswith("data:"))
    return chunks


def test_completion_streaming_contract(monkeypatch: pytest.MonkeyPatch, api_client: TestClient):
    runner = StubRunner(["Hello", " world!"])

    def fake_get_model(model: str):
        return runner, "stub-key"

    released: List[str] = []

    def fake_release(model_key: str) -> None:
        released.append(model_key)

    monkeypatch.setattr(server, "get_or_load_model", fake_get_model)
    monkeypatch.setattr(server, "release_runner", fake_release)

    payload = {
        "model": "stub-model",
        "prompt": "Say hello.",
        "stream": True,
        "max_tokens": 5,
    }

    with api_client.stream("POST", "/v1/completions", json=payload) as response:
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("text/event-stream")
        events = _collect_events(response)

    assert events[-1] == "data: [DONE]"
    data_events = [evt for evt in events if evt != "data: [DONE]"]
    assert len(data_events) >= 2  # initial + final (and optional token chunks)

    initial_payload = json.loads(data_events[0].split("data: ", 1)[1])
    assert initial_payload["choices"][0]["finish_reason"] is None

    final_payload = json.loads(data_events[-1].split("data: ", 1)[1])
    assert final_payload["choices"][0]["finish_reason"] == "stop"
    usage = final_payload.get("usage")
    assert usage is not None
    assert usage["completion_tokens"] == len(runner.tokens)
    assert usage["total_tokens"] >= usage["completion_tokens"]

    assert released == ["stub-key"]


def test_chat_streaming_contract(monkeypatch: pytest.MonkeyPatch, api_client: TestClient):
    runner = StubRunner(["Hello", "!"])

    def fake_get_model(model: str):
        return runner, "chat-key"

    released: List[str] = []

    def fake_release(model_key: str) -> None:
        released.append(model_key)

    monkeypatch.setattr(server, "get_or_load_model", fake_get_model)
    monkeypatch.setattr(server, "release_runner", fake_release)

    payload = {
        "model": "stub-chat",
        "stream": True,
        "messages": [
            {"role": "user", "content": "Hello?"}
        ],
        "max_tokens": 5,
    }

    with api_client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("text/event-stream")
        events = _collect_events(response)

    assert events[-1] == "data: [DONE]"
    data_events = [evt for evt in events if evt != "data: [DONE]"]
    assert len(data_events) >= 2

    initial_payload = json.loads(data_events[0].split("data: ", 1)[1])
    assert initial_payload["choices"][0]["delta"]["role"] == "assistant"

    final_payload = json.loads(data_events[-1].split("data: ", 1)[1])
    assert final_payload["choices"][0]["finish_reason"] == "stop"
    usage = final_payload.get("usage")
    assert usage is not None
    assert usage["completion_tokens"] == len(runner.tokens)
    assert usage["total_tokens"] >= usage["completion_tokens"]

    assert released == ["chat-key"]
