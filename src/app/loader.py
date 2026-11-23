"""Lite model loader and inference wrapper (Zero-Torch).

Security note / trust boundary:
This module loads and executes the Python loader file (``lite_embed.py`` or
``ruri_v3_lite.py``) shipped inside the target Hugging Face repository or local
model directory via ``importlib.util.spec_from_file_location`` + ``exec_module``.
Only configure trusted model repositories, and pin ``LITE_MODEL_REVISION`` to
an immutable commit SHA in production environments.
"""

import base64
import importlib.util
import logging
import os
import threading
import time
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from app.config import (
    get_batch_size,
    get_lite_model,
    get_lite_model_revision,
    get_max_length,
)

logger = logging.getLogger("uvicorn.error")

LOADER_CANDIDATES = ("lite_embed.py", "ruri_v3_lite.py")


def _import_loader_module(loader_path: str) -> Any:
    """Import a lite loader module from file path without modifying sys.path."""
    spec = importlib.util.spec_from_file_location("_lite_loader_mod", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_loader_in_dir(model_dir: str) -> str:
    for filename in LOADER_CANDIDATES:
        candidate = os.path.join(model_dir, filename)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Neither lite_embed.py nor ruri_v3_lite.py found in {model_dir}")


def _resolve_loader_path(
    model_id: str,
    revision: str | None = None,
) -> tuple[str, str | None]:
    """Resolve (loader_file_path, local_model_dir) from a local directory or HF Hub."""
    if os.path.isdir(model_id):
        return _find_loader_in_dir(model_id), model_id

    if revision:
        snapshot_dir = snapshot_download(repo_id=model_id, revision=revision)
        return _find_loader_in_dir(snapshot_dir), snapshot_dir

    logger.warning(
        "LITE_MODEL_REVISION is not set; remote loader code from %r is unpinned.",
        model_id,
    )
    last_err: Exception | None = None
    for filename in LOADER_CANDIDATES:
        try:
            path = hf_hub_download(repo_id=model_id, filename=filename)
            return path, None
        except Exception as exc:
            last_err = exc
    raise FileNotFoundError(
        f"Failed to resolve lite loader ({LOADER_CANDIDATES}) from '{model_id}': {last_err}"
    )


class LiteEmbeddingEngine:
    """Thread-safe single-model inference wrapper around LiteEmbedder / RuriV3Lite."""

    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
    ) -> None:
        resolved_model_id = model_id or get_lite_model()
        resolved_revision = revision if revision is not None else get_lite_model_revision()

        loader_path, local_model_dir = _resolve_loader_path(
            resolved_model_id,
            revision=resolved_revision,
        )
        loader_mod = _import_loader_module(loader_path)

        embedder_cls = getattr(loader_mod, "LiteEmbedder", None) or getattr(
            loader_mod, "RuriV3Lite", None
        )
        if embedder_cls is None:
            raise RuntimeError(
                f"Loader at {loader_path} does not export LiteEmbedder or RuriV3Lite"
            )

        if local_model_dir is not None:
            embedder = embedder_cls(repo_id=resolved_model_id, model_dir=local_model_dir)
        else:
            embedder = embedder_cls(repo_id=resolved_model_id)

        self._init_common(
            model_id=resolved_model_id,
            embedder=embedder,
            max_length=get_max_length(),
            batch_size=get_batch_size(),
        )

    def _init_common(
        self,
        model_id: str,
        embedder: Any,
        max_length: int,
        batch_size: int,
    ) -> None:
        self.model_id = model_id
        self.embedder = embedder
        self.max_length = max_length
        self.batch_size = batch_size
        self.created_at = int(time.time())
        self._lock = threading.Lock()
        self.embedding_dim = int(self.embedder.embedding_dim)
        self.config: dict[str, Any] = dict(getattr(self.embedder, "config", {}))

    @classmethod
    def from_embedder(
        cls,
        model_id: str,
        embedder: Any,
        max_length: int = 512,
        batch_size: int = 64,
    ) -> "LiteEmbeddingEngine":
        """Construct an engine around a pre-built embedder (for unit tests)."""
        instance = cls.__new__(cls)
        instance._init_common(
            model_id=model_id,
            embedder=embedder,
            max_length=max_length,
            batch_size=batch_size,
        )
        return instance

    def resolve_prefix(
        self,
        input_type: str | None,
        apply_ruri_prefix: bool,
        is_single_str: bool,
    ) -> str:
        """Determine prefix string from lite_config.json and request parameters."""
        query_prefix = str(self.config.get("query_prefix", ""))
        passage_prefix = str(self.config.get("passage_prefix", ""))

        if input_type is not None:
            key = input_type.lower()
            if key == "query":
                return query_prefix
            if key in ("document", "passage"):
                return passage_prefix
            if key in ("classification", "clustering"):
                if (
                    "classification_prefix" in self.config
                    and self.config["classification_prefix"] is not None
                ):
                    return str(self.config["classification_prefix"])
                family = str(self.config.get("family", "")).lower()
                is_ruri = (
                    family.startswith("ruri")
                    if family
                    else ("ruri" in self.model_id.lower() or query_prefix == "検索クエリ: ")
                )
                return "トピック: " if is_ruri else ""
            if key == "sts":
                return str(self.config.get("sts_prefix", ""))
            return ""

        if apply_ruri_prefix:
            return query_prefix if is_single_str else passage_prefix

        return ""

    @staticmethod
    def apply_prefix(texts: list[str], prefix: str) -> list[str]:
        """Prepend prefix unless the text already starts with it."""
        if not prefix:
            return texts
        return [t if t.startswith(prefix) else f"{prefix}{t}" for t in texts]

    def count_tokens(self, texts: list[str]) -> int:
        """Count total tokens (including BOS/EOS reserve, clipped to max_length)."""
        add_bos = bool(getattr(self.embedder, "add_bos", False))
        add_eos = bool(getattr(self.embedder, "add_eos", False))
        reserve = (1 if add_bos else 0) + (1 if add_eos else 0)
        limit = max(0, self.max_length - reserve)
        tokenizer = self.embedder.tokenizer

        total = 0
        for text in texts:
            tokens = tokenizer.encode(text)
            total += min(len(tokens), limit) + reserve
        return total

    def encode_and_format(
        self,
        texts: list[str],
        dimensions: int | None = None,
        encoding_format: str = "float",
    ) -> tuple[list[list[float] | str], int]:
        """Run serialized inference via the lite loader and format embeddings."""
        # Keep count_tokens inside the lock: FastSBPTokenizer thread-safety is not
        # guaranteed. Tokenization happens twice (here and inside embedder.encode)
        # because the external lite loader owns encoding, and tokenization cost is
        # small compared to ONNX inference.
        with self._lock:
            token_count = self.count_tokens(texts)
            vectors: np.ndarray = self.embedder.encode(
                texts,
                max_length=self.max_length,
                batch_size=self.batch_size,
                normalize_embeddings=True,
            )

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        if dimensions is not None and 0 < dimensions < arr.shape[1]:
            arr = arr[:, :dimensions]
            arr_f64 = arr.astype(np.float64)
            norms = np.linalg.norm(arr_f64, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = (arr_f64 / norms).astype(np.float32)

        if encoding_format == "base64":
            arr_f32 = arr.astype("<f4", copy=False)
            formatted: list[list[float] | str] = [
                base64.b64encode(row.tobytes()).decode("utf-8") for row in arr_f32
            ]
        else:
            formatted = arr.tolist()

        return formatted, token_count
