from pathlib import Path
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from mlx_knife import server
from mlx_knife import cache_utils


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(server.app)


def _create_model_dir(root: Path, model_name: str) -> Path:
    cache_name = model_name.replace("/", "--")
    snapshot_path = root / f"models--{cache_name}" / "snapshots" / "hash-1"
    snapshot_path.mkdir(parents=True, exist_ok=True)
    return snapshot_path


def test_list_models_includes_model_types(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_client: TestClient):
    snapshot_chat = _create_model_dir(tmp_path, "mlx-community/chat-model")
    snapshot_base = _create_model_dir(tmp_path, "mlx-community/base-model")

    monkeypatch.setattr(server, "detect_framework", lambda *_: "MLX")

    type_map: Dict[str, str] = {
        "mlx-community/chat-model": "chat",
        "mlx-community/base-model": "base",
    }

    def fake_model_type(_model_dir, hf_name):
        return type_map[hf_name]

    monkeypatch.setattr(server, "detect_model_type", fake_model_type)
    monkeypatch.setattr(server, "is_model_healthy", lambda model: True)

    def fake_get_model_path(model_spec: str):
        if model_spec == "mlx-community/chat-model":
            return snapshot_chat, model_spec, "hash-1"
        if model_spec == "mlx-community/base-model":
            return snapshot_base, model_spec, "hash-1"
        return None, model_spec, None

    monkeypatch.setattr(server, "get_model_path", fake_get_model_path)
    monkeypatch.setattr("mlx_knife.mlx_runner.get_model_context_length", lambda *_: 4096)
    monkeypatch.setattr(cache_utils, "MODEL_CACHE", tmp_path)

    response = api_client.get("/v1/models")
    assert response.status_code == 200

    body = response.json()
    assert body["object"] == "list"
    models: List[Dict[str, str]] = body["data"]
    ids = {item["id"] for item in models}
    assert "mlx-community/chat-model" in ids
    assert "mlx-community/base-model" in ids

    chat_model = next(item for item in models if item["id"] == "mlx-community/chat-model")
    base_model = next(item for item in models if item["id"] == "mlx-community/base-model")

    assert chat_model["model_type"] == "chat"
    assert base_model["model_type"] == "base"
    assert chat_model["context_length"] == 4096
    assert base_model["context_length"] == 4096
    assert chat_model["owned_by"] == "mlx-knife"
    assert base_model["owned_by"] == "mlx-knife"
