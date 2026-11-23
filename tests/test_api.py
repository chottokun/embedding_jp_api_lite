"""Unit and integration tests for embedding_jp_api_lite."""

import base64
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Keep HF cache inside the project directory during tests
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf_cache"))

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoints(client: TestClient):
    for path in ("/", "/health", "/healthz"):
        resp = client.get(path)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model"] == "Chottokun/ruri-v3-30m-lite"
        assert body["embedding_dim"] == 256


def test_models_endpoint(client: TestClient):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == "Chottokun/ruri-v3-30m-lite"


def test_embeddings_float_and_base64(client: TestClient):
    text = "日本の首都は東京です。"
    r_float = client.post(
        "/v1/embeddings",
        json={
            "model": "Chottokun/ruri-v3-30m-lite",
            "input": text,
            "input_type": "document",
            "encoding_format": "float",
        },
    )
    assert r_float.status_code == 200
    d_float = r_float.json()
    vec_float = d_float["data"][0]["embedding"]
    assert len(vec_float) == 256
    assert d_float["usage"]["prompt_tokens"] > 0

    r_b64 = client.post(
        "/v1/embeddings",
        json={
            "model": "Chottokun/ruri-v3-30m-lite",
            "input": text,
            "input_type": "document",
            "encoding_format": "base64",
        },
    )
    assert r_b64.status_code == 200
    b64_str = r_b64.json()["data"][0]["embedding"]
    raw = base64.b64decode(b64_str)
    vec_b64 = list(struct.unpack("<256f", raw))
    assert np.allclose(vec_float, vec_b64, atol=1e-6)


def test_embeddings_dimensions_matryoshka(client: TestClient):
    resp = client.post(
        "/v1/embeddings",
        json={
            "model": "Chottokun/ruri-v3-30m-lite",
            "input": ["テスト1", "テスト2"],
            "dimensions": 128,
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 2
    for item in data:
        vec = np.array(item["embedding"], dtype=np.float32)
        assert vec.shape == (128,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_api_key_auth(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_KEY", "secret-test-key")
    # Health remains public
    assert client.get("/health").status_code == 200
    # Protected endpoints reject missing/invalid key
    assert client.get("/v1/models").status_code == 401
    assert client.post("/v1/embeddings", json={"input": "hello"}).status_code == 401
    # Valid Bearer key succeeds
    headers = {"Authorization": "Bearer secret-test-key"}
    assert client.get("/v1/models", headers=headers).status_code == 200
    assert (
        client.post(
            "/v1/embeddings",
            headers=headers,
            json={"input": "hello"},
        ).status_code
        == 200
    )
