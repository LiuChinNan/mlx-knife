import logging
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlx_knife import cache_utils, server


class StubRunner:
    def __init__(self, model_path: str, verbose: bool = False):
        self.model_path = model_path
        self.verbose = verbose
        self.loaded = False
        self.cleaned = False

    def load_model(self) -> None:
        self.loaded = True

    def cleanup(self) -> None:
        self.cleaned = True

    def generate_streaming(self, **_: object):
        raise RuntimeError("stream failure")

    def get_effective_max_tokens(self, requested: int | None, interactive: bool = False) -> int:
        return requested or 32

    def _format_conversation(self, messages, use_chat_template: bool = True) -> str:
        return "\n".join(entry["content"] for entry in messages)


def _reset_server_state() -> None:
    server._model_cache.clear()
    server._runner_ref_counts.clear()
    server._current_model_path = None
    server._cache_lock = threading.RLock()


@pytest.fixture
def isolated_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cache_dir = cache_utils.hf_to_cache_dir("stub-model")
    snapshot = tmp_path / cache_dir / "snapshots" / "hash-1"
    snapshot.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cache_utils, "MODEL_CACHE", tmp_path)
    monkeypatch.setattr(server, "detect_framework", lambda *_: "MLX")
    monkeypatch.setattr(server, "MLXRunner", StubRunner)
    _reset_server_state()
    yield "stub-model"
    _reset_server_state()


def test_runner_load_and_release_logs(isolated_model, caplog: pytest.LogCaptureFixture):
    model_path, _, _ = server.get_model_path(isolated_model)
    assert model_path.exists()
    with caplog.at_level(logging.INFO, logger=server.logger.name):
        runner, key = server.get_or_load_model(isolated_model)
        assert runner.loaded
    load_record = next((record for record in caplog.records if getattr(record, "event", "") == "runner_loaded"), None)
    assert load_record is not None
    assert load_record.fields["model"] == isolated_model

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=server.logger.name):
        server.release_runner(key)
    release_record = next((record for record in caplog.records if getattr(record, "event", "") == "runner_released"), None)
    assert release_record is not None
    assert release_record.fields["model_key"] == key


def test_streaming_error_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    monkeypatch.setattr(server, "get_or_load_model", lambda model: (StubRunner("/tmp"), "stub-key"))
    monkeypatch.setattr(server, "release_runner", lambda model_key: None)

    client = TestClient(server.app)
    payload = {
        "model": "stub-model",
        "prompt": "Hello",
        "stream": True,
    }

    with caplog.at_level(logging.ERROR, logger=server.logger.name):
        with client.stream("POST", "/v1/completions", json=payload) as response:
            assert response.status_code == 200
            # consume generator to trigger stream failure
            list(response.iter_text())

    error_record = next((record for record in caplog.records if getattr(record, "event", "") == "stream_error"), None)
    assert error_record is not None
    assert error_record.fields["mode"] == "completion"
    assert error_record.fields["model"] == "stub-model"


def test_stream_start_and_complete_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    class SuccessfulRunner:
        def __init__(self, tokens):
            self.tokens = tokens
            self.loaded = True
            self.cleaned = False

        def load_model(self) -> None:
            self.loaded = True

        def cleanup(self) -> None:
            self.cleaned = True

        def generate_streaming(self, **_: object):
            yield from self.tokens

        def get_effective_max_tokens(self, requested: int | None, interactive: bool = False) -> int:
            return requested or len(self.tokens)

        def _format_conversation(self, messages, use_chat_template: bool = True) -> str:
            return "\n".join(entry["content"] for entry in messages)

        def generate_batch(self, **_: object) -> str:
            return "".join(self.tokens)

    monkeypatch.setattr(server, "get_or_load_model", lambda model: (SuccessfulRunner(["Hello", " world"]), "stub-key"))
    monkeypatch.setattr(server, "release_runner", lambda model_key: None)

    client = TestClient(server.app)
    payload = {"model": "stub-model", "prompt": "Hello", "stream": True}

    with caplog.at_level(logging.INFO, logger=server.logger.name):
        with client.stream("POST", "/v1/completions", json=payload) as response:
            assert response.status_code == 200
            list(response.iter_text())

    events = {record.event: record for record in caplog.records if hasattr(record, "event")}
    assert "stream_start" in events
    assert events["stream_start"].fields["mode"] == "completion"

    assert "stream_complete" in events
    complete_fields = events["stream_complete"].fields
    assert complete_fields["completion_tokens"] == 2


def test_generation_complete_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    class BatchRunner:
        def __init__(self, response: str):
            self.response = response

        def load_model(self) -> None:
            pass

        def cleanup(self) -> None:
            pass

        def get_effective_max_tokens(self, requested: int | None, interactive: bool = False) -> int:
            return requested or 32

        def generate_batch(self, **_: object) -> str:
            return self.response

    monkeypatch.setattr(server, "get_or_load_model", lambda model: (BatchRunner("Hi"), "stub-key"))
    monkeypatch.setattr(server, "release_runner", lambda model_key: None)

    client = TestClient(server.app)
    payload = {"model": "stub-model", "prompt": "Hello"}

    with caplog.at_level(logging.INFO, logger=server.logger.name):
        response = client.post("/v1/completions", json=payload)
        assert response.status_code == 200

    event_record = next((record for record in caplog.records if getattr(record, "event", "") == "generation_complete"), None)
    assert event_record is not None
    assert event_record.fields["mode"] == "completion"
    assert event_record.fields["completion_tokens"] >= 1
