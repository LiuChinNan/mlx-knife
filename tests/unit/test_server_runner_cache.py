import sys
import threading
import types
from pathlib import Path

import pytest

# Provide a lightweight stub for mlx_knife.mlx_runner before importing server
if "mlx_knife.mlx_runner" not in sys.modules:
    stub_runner_module = types.ModuleType("mlx_knife.mlx_runner")

    class _BaseStubRunner:  # pragma: no cover - replaced in tests
        def __init__(self, *args, **kwargs):
            pass

        def load_model(self) -> None:
            pass

        def cleanup(self) -> None:
            pass

    def _stub_get_model_context_length(_: str) -> int:
        return 2048

    stub_runner_module.MLXRunner = _BaseStubRunner
    stub_runner_module.get_model_context_length = _stub_get_model_context_length
    sys.modules["mlx_knife.mlx_runner"] = stub_runner_module

from mlx_knife import server


class DummyRunner:
    def __init__(self, model_path: str, verbose: bool = False):
        self.model_path = model_path
        self.verbose = verbose
        self.loaded = False
        self.cleaned = False

    def load_model(self) -> None:
        self.loaded = True

    def cleanup(self) -> None:
        self.cleaned = True


def _reset_server_state() -> None:
    server._model_cache.clear()
    server._runner_ref_counts.clear()
    server._current_model_path = None
    server._cache_lock = threading.RLock()


@pytest.fixture
def fake_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snapshot_a = tmp_path / "models--model-a" / "snapshots" / "hash-a"
    snapshot_b = tmp_path / "models--model-b" / "snapshots" / "hash-b"
    snapshot_a.mkdir(parents=True)
    snapshot_b.mkdir(parents=True)

    def fake_get_model_path(spec: str):
        if spec == "model-a":
            return snapshot_a, "model-a", None
        if spec == "model-b":
            return snapshot_b, "model-b", None
        raise ValueError(spec)

    monkeypatch.setattr("mlx_knife.cache_utils.get_model_path", fake_get_model_path)
    monkeypatch.setattr(server, "detect_framework", lambda *args, **kwargs: "MLX")
    monkeypatch.setattr(server, "MLXRunner", DummyRunner)

    _reset_server_state()

    yield {"a": str(snapshot_a), "b": str(snapshot_b)}

    _reset_server_state()


def test_switching_models_keeps_active_runner(fake_models):
    runner_a, key_a = server.get_or_load_model("model-a")
    assert runner_a.loaded

    runner_b, key_b = server.get_or_load_model("model-b")
    assert runner_b.loaded
    assert not runner_a.cleaned

    server.release_runner(key_a)
    assert runner_a.cleaned
    assert not runner_b.cleaned

    server.release_runner(key_b)
    assert runner_b.cleaned
    assert server._model_cache == {}
    assert server._runner_ref_counts == {}


def test_reference_count_requires_all_releases(fake_models):
    runner_1, key_1 = server.get_or_load_model("model-a")
    runner_2, key_2 = server.get_or_load_model("model-a")

    assert runner_1 is runner_2
    assert server._runner_ref_counts[key_1] == 2
    assert not runner_1.cleaned

    server.release_runner(key_1)
    assert server._runner_ref_counts[key_1] == 1
    assert not runner_1.cleaned

    server.release_runner(key_2)
    assert key_1 not in server._runner_ref_counts
    assert runner_1.cleaned
    assert server._model_cache == {}
