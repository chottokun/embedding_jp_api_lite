"""Lite model loader and inference wrapper (Zero-Torch)."""

import base64
import importlib.util
import os
import threading
import time
from typing import Any, List, Optional, Union

import numpy as np
from huggingface_hub import hf_hub_download

from app.config import get_batch_size, get_lite_model, get_max_length

LOADER_CANDIDATES = ("lite_embed.py", "ruri_v3_lite.py")


def _import_loader_module(loader_path: str) -> Any:
    """Import a lite loader module from file path without modifying sys.path."""
    spec = importlib.util.spec_from_file_location("_lite_loader_mod", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_loader_path(model_id: str) -> tuple[str, bool]:
    """Resolve (loader_file_path, is_local_dir) from local directory or HF Hub."""
    is_local = os.path.isdir(model_id)
    if is_local:
        for filename in LOADER_CANDIDATES:
            candidate = os.path.join(model_id, filename)
            if os.path.isfile(candidate):
                return candidate, True
        raise FileNotFoundError(
            f"Neither lite_embed.py nor ruri_v3_lite.py found in {model_id}"
        )

    last_err: Optional[Exception] = None
    for filename in LOADER_CANDIDATES:
        try:
            path = hf_hub_download(repo_id=model_id, filename=filename)
            return path, False
        except Exception as exc:
            last_err = exc
    raise FileNotFoundError(
        f"Failed to resolve lite loader ({LOADER_CANDIDATES}) from '{model_id}': {last_err}"
    )


class LiteEmbeddingEngine:
    """Thread-safe single-model inference wrapper around LiteEmbedder / RuriV3Lite."""

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or get_lite_model()
        self.max_length = get_max_length()
        self.batch_size = get_batch_size()
        self.created_at = int(time.time())
        self._lock = threading.Lock()

        loader_path, is_local = _resolve_loader_path(self.model_id)
        loader_mod = _import_loader_module(loader_path)

        embedder_cls = getattr(loader_mod, "LiteEmbedder", None) or getattr(
            loader_mod, "RuriV3Lite", None
        )
        if embedder_cls is None:
            raise RuntimeError(
                f"Loader at {loader_path} does not export LiteEmbedder or RuriV3Lite"
            )

        if is_local:
            self.embedder = embedder_cls(model_dir=self.model_id)
        else:
            self.embedder = embedder_cls(repo_id=self.model_id)

        self.embedding_dim: int = int(self.embedder.embedding_dim)
        self.config: dict = dict(getattr(self.embedder, "config", {}))

    def resolve_prefix(
        self,
        input_type: Optional[str],
        apply_ruri_prefix: bool,
        is_single_str: bool,
    ) -> str:
        """Determine prefix string from lite_config.json and request parameters."""
        query_prefix = self.config.get("query_prefix", "")
        passage_prefix = self.config.get("passage_prefix", "")

        if input_type is not None:
            key = input_type.lower()
            if key == "query":
                return query_prefix
            if key in ("document", "passage"):
                return passage_prefix
            if key in ("classification", "clustering"):
                return "トピック: "
            if key == "sts":
                return ""
            return ""

        if apply_ruri_prefix:
            return query_prefix if is_single_str else passage_prefix

        return ""

    @staticmethod
    def apply_prefix(texts: List[str], prefix: str) -> List[str]:
        """Prepend prefix unless the text already starts with it."""
        if not prefix:
            return texts
        return [t if t.startswith(prefix) else f"{prefix}{t}" for t in texts]

    def count_tokens(self, texts: List[str]) -> int:
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
        texts: List[str],
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
    ) -> tuple[List[Union[list[float], str]], int]:
        """Run serialized inference via the lite loader and format embeddings."""
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
            formatted: List[Union[list[float], str]] = [
                base64.b64encode(row.tobytes()).decode("utf-8") for row in arr_f32
            ]
        else:
            formatted = arr.tolist()

        return formatted, token_count
