# embedding_jp_api_lite

**English** | [日本語](README.ja.md)

[![CI](https://github.com/chottokun/embedding_jp_api_lite/actions/workflows/ci.yml/badge.svg)](https://github.com/chottokun/embedding_jp_api_lite/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Docker linux/amd64](https://img.shields.io/badge/docker-linux%2Famd64-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![OpenAI compatible](https://img.shields.io/badge/OpenAI%20API-compatible-412991?logo=openai&logoColor=white)](#api-usage)
[![PyTorch free](https://img.shields.io/badge/PyTorch-free-EE4C2C?logo=pytorch&logoColor=white)](#features)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-ruri--v3--30m--lite-FFD21E)](https://huggingface.co/Chottokun/ruri-v3-30m-lite)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

OpenAI-compatible, lightweight Japanese text embedding API server — runs on `sentencepiece_lite` + ONNX Runtime (CPU) with zero PyTorch/Transformers dependencies. Default model: `Chottokun/ruri-v3-30m-lite` (256-dim).

## Features

- **Zero-Torch**: Depends only on `fastapi`, `uvicorn`, `onnxruntime`, `numpy`, `huggingface-hub`, and `sentencepiece_lite` (`cp311-abi3` wheel)
- **OpenAI-compatible API**:
  - `POST /v1/embeddings` (supports `float` / `base64`, `dimensions`, `input_type` / `apply_ruri_prefix`; defaults to the loaded model when `model` is omitted)
  - `GET /v1/models`
  - `GET /health` / `GET /healthz` / `GET /` (returns the loaded model name and embedding dimension)
- **Offline startup support**: Downloads and caches the model and lite loader (`lite_embed.py` / `ruri_v3_lite.py`) from Hugging Face Hub on first boot; subsequent boots run fully offline with `HF_HUB_OFFLINE=1`
- **Optional Bearer authentication**: Validates `Authorization: Bearer <API_KEY>` only when the `API_KEY` environment variable is set

## Supported Models

All 6 embedding models published by [`chottokun/embedding-sentencepiece-lite`](https://github.com/chottokun/embedding-sentencepiece-lite) run on the same Zero-Torch lightweight stack (1 process = 1 model, switchable via `LITE_MODEL`, no additional dependencies):

| Model ID | Dim | Languages | Query / Passage Prefix | License | Notes |
| --- | --- | --- | --- | --- | --- |
| [`Chottokun/ruri-v3-30m-lite`](https://huggingface.co/Chottokun/ruri-v3-30m-lite) | 256 | Japanese | `"検索クエリ: "` / `"文章: "` | Apache-2.0 | **Default / lightest** |
| [`Chottokun/ruri-v3-70m-lite`](https://huggingface.co/Chottokun/ruri-v3-70m-lite) | 384 | Japanese | `"検索クエリ: "` / `"文章: "` | Apache-2.0 | Balanced Japanese embedding |
| [`Chottokun/ruri-v3-130m-lite`](https://huggingface.co/Chottokun/ruri-v3-130m-lite) | 512 | Japanese | `"検索クエリ: "` / `"文章: "` | Apache-2.0 | Medium Japanese embedding |
| [`Chottokun/ruri-v3-310m-lite`](https://huggingface.co/Chottokun/ruri-v3-310m-lite) | 768 | Japanese | `"検索クエリ: "` / `"文章: "` | Apache-2.0 | High-accuracy Japanese embedding |
| [`Chottokun/multilingual-e5-base-lite`](https://huggingface.co/Chottokun/multilingual-e5-base-lite) | 768 | Multilingual | `"query: "` / `"passage: "` | MIT | 100+ languages |
| [`Chottokun/embeddinggemma-2-lite`](https://huggingface.co/Chottokun/embeddinggemma-2-lite) | 768 | Multilingual | `"task: search result \| query: "` / `"title: none \| text: "` | Apache-2.0 | Google EmbeddingGemma 2 text tower |

> Note: `Chottokun/ruri-v3-reranker-310m-lite` (Cross-Encoder reranker) is out of scope for this embeddings-only server.

## Platform Requirements

- Published wheels for `sentencepiece_lite` target `x86_64` (`manylinux_2_27_x86_64.manylinux_2_28_x86_64`) only, so local runs and the Docker image are **`linux/amd64`-only** (ARM64 hosts require `--platform=linux/amd64` emulation or a custom wheel).

## Quick Start

### 1. Run locally with uv

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### 2. Run with Docker

```bash
docker build -t embedding_jp_api_lite .
docker run --rm -p 8080:8080 -v hf_cache:/data/huggingface embedding_jp_api_lite
```

The container runs as a non-root user (`app`, UID 10001) and includes a `HEALTHCHECK` against `/health`.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `LITE_MODEL` | `Chottokun/ruri-v3-30m-lite` | Hugging Face repo ID or local model directory |
| `LITE_MODEL_REVISION` | `""` (unpinned) | Hugging Face commit SHA, branch, or tag (pinning a commit SHA is recommended in production) |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8080` | Listening port (integer between `1` and `65535`) |
| `API_KEY` | `""` (unset) | Enables Bearer token authentication when set |
| `LITE_MAX_LENGTH` | `512` | Maximum token length (`>= 1`) |
| `LITE_BATCH_SIZE` | `64` | Inference batch size (`>= 1`) |
| `MAX_INPUT_LENGTH` | `65536` | Maximum characters per input text (read at startup, `>= 1`) |
| `MAX_INPUT_ITEMS` | `256` | Maximum items per request array (read at startup, `>= 1`) |
| `HF_HUB_OFFLINE` | `0` | Set to `1` to use cached files only without network requests |

## Security Note (Remote Loader Trust Boundary)

This server imports and executes the Python loader (`lite_embed.py` or `ruri_v3_lite.py`) shipped inside the configured `LITE_MODEL` Hugging Face repository or local model directory via `importlib`.

- Configure **only trusted model repositories or local directories**.
- In production, **pin `LITE_MODEL_REVISION` to an immutable commit SHA** to lock both the loader code and model weights (a warning is logged when unpinned).

## Prefix Rules & Input Validation

### Prefix Rules

Asymmetric retrieval models such as `ruri-v3` expect different prefixes for search queries and passages. The server reads `lite_config.json` (or loader defaults if absent) and applies prefixes as follows:

- `input_type="query"`: prepends `query_prefix` (`"検索クエリ: "` for `ruri-v3`)
- `input_type="document"` or `"passage"`: prepends `passage_prefix` (`"文章: "` for `ruri-v3`)
- `input_type="classification"` / `"clustering"`: prepends `classification_prefix` from `lite_config.json` (defaults to `"トピック: "` for `ruri-v3` models, `""` for other model families)
- `input_type="sts"`: prepends `sts_prefix` from `lite_config.json` (defaults to `""`)
- `input_type` omitted and `apply_ruri_prefix=true`: prepends `query_prefix` for a single string input, or `passage_prefix` for a list input
- `input_type` omitted (default): embeds `input` as-is (does not inject prefixes automatically)

`input_type` is trimmed and lowercased during validation (e.g., `"QUERY"` → `"query"`). Prefixes are not duplicated if the input text already starts with the target prefix.

### Input Validation

- Unsupported `input_type` values (e.g., `"doc"`), empty strings `""`, empty lists `[]`, inputs exceeding `MAX_INPUT_ITEMS` / `MAX_INPUT_LENGTH`, `dimensions < 1`, or invalid `encoding_format` values return **`422 Unprocessable Entity`**.
- Requesting `dimensions` larger than the loaded model's embedding dimension (e.g., `256`) returns **`400 Bad Request`**.

## API Usage

### curl

```bash
# Health check
curl http://localhost:8080/health

# Embed a search query
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Chottokun/ruri-v3-30m-lite",
    "input": "日本の首都はどこですか？",
    "input_type": "query"
  }'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

resp = client.embeddings.create(
    model="Chottokun/ruri-v3-30m-lite",
    input=["日本の首都は東京です。", "大阪は関西の都市です。"],
    extra_body={"input_type": "document"},
)
print(len(resp.data[0].embedding))  # 256
```

## Offline Operation

After downloading and caching the model files on first startup (or pre-populating the cache), launch with `HF_HUB_OFFLINE=1`.

Because `pyproject.toml` references a direct-URL wheel (`sentencepiece-lite`), pass `--no-sync` (or `--offline`) to `uv run` in offline environments so `uv` does not attempt to fetch wheel metadata:

```bash
HF_HUB_OFFLINE=1 uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8080
```

With Docker, keep the populated cache volume mounted and pass `-e HF_HUB_OFFLINE=1`. Pinning `LITE_MODEL_REVISION` to a commit SHA also works offline directly from the `snapshot_download` cache.

## Swapping Models

Set `LITE_MODEL` to another lite model (a Hugging Face repo ID or local directory containing `model.onnx`, `.spm.fb`, `lite_config.json`, and `lite_embed.py`) and restart the server (one model per process):

```bash
# Larger Japanese ruri-v3 model
LITE_MODEL=Chottokun/ruri-v3-70m-lite uv run uvicorn app.main:app --host 0.0.0.0 --port 8080

# Multilingual E5 base
LITE_MODEL=Chottokun/multilingual-e5-base-lite uv run uvicorn app.main:app --host 0.0.0.0 --port 8080

# EmbeddingGemma 2
LITE_MODEL=Chottokun/embeddinggemma-2-lite uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Development & Testing

```bash
# Lint / Format / Type-check
uv run ruff format --check .
uv run ruff check .
uv run mypy src

# Unit tests (no model download required)
uv run pytest -m "not integration" -v

# Integration tests (uses real model; works with offline cache)
HF_HUB_OFFLINE=1 uv run --no-sync pytest -m integration -v

# Run all 5 verification gates
uv run python scripts/verify_gates.py

# Security checks (static analysis & dependency CVE scan)
uv run --no-project --with bandit bandit -r src --severity-level medium --confidence-level medium
uv export --frozen --no-dev --no-hashes --no-emit-project --no-annotate | grep -v '^sentencepiece-lite ' > /tmp/requirements-audit.txt
uv venv --python 3.12 .audit-venv
uv pip install --python .audit-venv/bin/python --no-deps -r /tmp/requirements-audit.txt
uv run --no-project --with pip-audit pip-audit --path .audit-venv/lib/python3.12/site-packages --strict
rm -rf .audit-venv /tmp/requirements-audit.txt
```

GitHub Actions automatically runs `pip-audit`, `bandit`, and `gitleaks` in parallel on every push/PR and on a weekly schedule (`cron: "0 0 * * 1"`).

## License

[MIT License](LICENSE)
