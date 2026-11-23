"""Unit tests for embedding_jp_api_lite (no model download required)."""

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import (
    MAX_INPUT_ITEMS,
    get_batch_size,
    get_lite_model_revision,
    get_max_length,
    get_port,
)
from app.loader import LiteEmbeddingEngine
from app.main import app


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class FakeEmbedder:
    """Deterministic lightweight stub matching LiteEmbedder / RuriV3Lite interface."""

    def __init__(self, config: dict[str, Any] | None = None, embedding_dim: int = 8) -> None:
        self.config = config or {
            "query_prefix": "検索クエリ: ",
            "passage_prefix": "文章: ",
        }
        self.embedding_dim = embedding_dim
        self.add_bos = True
        self.add_eos = True
        self.tokenizer = _FakeTokenizer()
        self.recorded_texts: list[list[str]] = []
        self._active_calls = 0
        self.max_concurrent_calls = 0
        self._guard = threading.Lock()

    def encode(
        self,
        texts: str | list[str],
        max_length: int = 512,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        del max_length, batch_size
        with self._guard:
            self._active_calls += 1
            self.max_concurrent_calls = max(self.max_concurrent_calls, self._active_calls)
        try:
            time.sleep(0.005)
            items = [texts] if isinstance(texts, str) else list(texts)
            self.recorded_texts.append(items)
            rows = []
            for item in items:
                digest = hashlib.sha256(item.encode("utf-8")).digest()
                seed = int.from_bytes(digest[:8], "little")
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(self.embedding_dim).astype(np.float32)
                if normalize_embeddings:
                    norm = float(np.linalg.norm(vec))
                    if norm > 0:
                        vec = vec / norm
                rows.append(vec)
            return np.vstack(rows)
        finally:
            with self._guard:
                self._active_calls -= 1


@pytest.fixture
def fake_client():
    embedder = FakeEmbedder()
    engine = LiteEmbeddingEngine.from_embedder(model_id="test/fake-model", embedder=embedder)
    app.state.engine = engine
    client = TestClient(app)
    try:
        yield client, embedder, engine
    finally:
        if hasattr(app.state, "engine"):
            delattr(app.state, "engine")


def test_prefix_resolution_and_deduplication(fake_client):
    client, embedder, _ = fake_client

    cases = [
        ("query", "日本", "検索クエリ: 日本"),
        ("QUERY", "東京", "検索クエリ: 東京"),
        ("document", "富士", "文章: 富士"),
        ("passage", "京都", "文章: 京都"),
        ("classification", "大阪", "トピック: 大阪"),
        ("clustering", "札幌", "トピック: 札幌"),
        ("sts", "沖縄", "沖縄"),
        ("query", "検索クエリ: 既存接頭辞", "検索クエリ: 既存接頭辞"),
    ]
    for input_type, raw_text, expected_text in cases:
        resp = client.post(
            "/v1/embeddings",
            json={"input": raw_text, "input_type": input_type},
        )
        assert resp.status_code == 200
        assert embedder.recorded_texts[-1] == [expected_text]

    # apply_ruri_prefix with single string -> query_prefix; with list -> passage_prefix
    resp_single = client.post(
        "/v1/embeddings",
        json={"input": "単一文", "apply_ruri_prefix": True},
    )
    assert resp_single.status_code == 200
    assert embedder.recorded_texts[-1] == ["検索クエリ: 単一文"]

    resp_list = client.post(
        "/v1/embeddings",
        json={"input": ["配列1", "文章: 配列2"], "apply_ruri_prefix": True},
    )
    assert resp_list.status_code == 200
    assert embedder.recorded_texts[-1] == ["文章: 配列1", "文章: 配列2"]

    # Default (no input_type and apply_ruri_prefix=False) -> raw text
    resp_raw = client.post("/v1/embeddings", json={"input": "そのまま"})
    assert resp_raw.status_code == 200
    assert embedder.recorded_texts[-1] == ["そのまま"]


def test_prefix_overrides_from_lite_config():
    embedder = FakeEmbedder(
        config={
            "query_prefix": "q: ",
            "passage_prefix": "p: ",
            "classification_prefix": "cls: ",
            "sts_prefix": "sts: ",
        }
    )
    engine = LiteEmbeddingEngine.from_embedder(model_id="test/custom", embedder=embedder)
    assert engine.resolve_prefix("classification", False, True) == "cls: "
    assert engine.resolve_prefix("clustering", False, True) == "cls: "
    assert engine.resolve_prefix("sts", False, True) == "sts: "


def test_validation_errors_422(fake_client):
    client, _, _ = fake_client

    invalid_payloads: list[dict[str, Any]] = [
        {"input": ""},
        {"input": []},
        {"input": ["x"] * (MAX_INPUT_ITEMS + 1)},
        {"input": "hello", "encoding_format": "int8"},
        {"input": "hello", "dimensions": 0},
        {"input": "hello", "input_type": "doc"},
    ]
    for payload in invalid_payloads:
        resp = client.post("/v1/embeddings", json=payload)
        assert resp.status_code == 422, f"Expected 422 for {payload}, got {resp.status_code}"


def test_dimensions_bounds(fake_client):
    client, _, engine = fake_client

    # Exceeding embedding_dim -> 400
    resp_too_large = client.post(
        "/v1/embeddings",
        json={"input": "hello", "dimensions": engine.embedding_dim + 1},
    )
    assert resp_too_large.status_code == 400
    assert "exceed" in resp_too_large.json()["detail"].lower()

    # Exact embedding_dim -> 200
    resp_exact = client.post(
        "/v1/embeddings",
        json={"input": "hello", "dimensions": engine.embedding_dim},
    )
    assert resp_exact.status_code == 200
    assert len(resp_exact.json()["data"][0]["embedding"]) == engine.embedding_dim

    # Smaller dimension (Matryoshka truncation + L2 normalization) -> 200
    resp_trunc = client.post(
        "/v1/embeddings",
        json={"input": "hello", "dimensions": 4},
    )
    assert resp_trunc.status_code == 200
    vec = np.array(resp_trunc.json()["data"][0]["embedding"], dtype=np.float32)
    assert vec.shape == (4,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_model_omitted_uses_loaded_model_id(fake_client):
    client, _, engine = fake_client
    resp = client.post("/v1/embeddings", json={"input": "hello"})
    assert resp.status_code == 200
    assert resp.json()["model"] == engine.model_id == "test/fake-model"


def test_returns_503_when_engine_not_loaded():
    app.state.engine = None
    client = TestClient(app)
    try:
        assert client.get("/health").status_code == 503
        assert client.get("/v1/models").status_code == 503
        assert client.post("/v1/embeddings", json={"input": "hello"}).status_code == 503
    finally:
        if hasattr(app.state, "engine"):
            delattr(app.state, "engine")


def test_concurrent_requests_serialized_and_identical(fake_client):
    client, embedder, _ = fake_client

    payloads = [{"input": f"concurrent item {i}", "input_type": "query"} for i in range(8)]
    expected = [
        client.post("/v1/embeddings", json=p).json()["data"][0]["embedding"] for p in payloads
    ]

    def _call(payload: dict[str, Any]) -> tuple[int, list[float]]:
        resp = client.post("/v1/embeddings", json=payload)
        return resp.status_code, resp.json()["data"][0]["embedding"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_call, payloads))

    for (status, vec), exp in zip(results, expected, strict=True):
        assert status == 200
        assert np.allclose(vec, exp, atol=1e-7)
    assert embedder.max_concurrent_calls == 1


def test_config_validation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PORT", "not-an-int")
    with pytest.raises(ValueError, match="PORT"):
        get_port()

    monkeypatch.setenv("PORT", "0")
    with pytest.raises(ValueError, match="PORT"):
        get_port()

    monkeypatch.setenv("PORT", "65536")
    with pytest.raises(ValueError, match="PORT"):
        get_port()

    monkeypatch.setenv("PORT", "8080")
    assert get_port() == 8080

    monkeypatch.setenv("LITE_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="LITE_BATCH_SIZE"):
        get_batch_size()

    monkeypatch.setenv("LITE_MAX_LENGTH", "-5")
    with pytest.raises(ValueError, match="LITE_MAX_LENGTH"):
        get_max_length()

    monkeypatch.delenv("LITE_MODEL_REVISION", raising=False)
    assert get_lite_model_revision() is None
    monkeypatch.setenv("LITE_MODEL_REVISION", "  ")
    assert get_lite_model_revision() is None
    monkeypatch.setenv("LITE_MODEL_REVISION", " abc123 ")
    assert get_lite_model_revision() == "abc123"
