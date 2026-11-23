# embedding_jp_api_lite

[English](README.md) | **日本語**

[![CI](https://github.com/chottokun/embedding_jp_api_lite/actions/workflows/ci.yml/badge.svg)](https://github.com/chottokun/embedding_jp_api_lite/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Docker linux/amd64](https://img.shields.io/badge/docker-linux%2Famd64-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![OpenAI compatible](https://img.shields.io/badge/OpenAI%20API-compatible-412991?logo=openai&logoColor=white)](#api-利用例)
[![PyTorch free](https://img.shields.io/badge/PyTorch-free-EE4C2C?logo=pytorch&logoColor=white)](#特徴)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-ruri--v3--30m--lite-FFD21E)](https://huggingface.co/Chottokun/ruri-v3-30m-lite)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

OpenAI 互換の軽量テキスト埋め込みサーバーです。`torch` / `transformers` を一切使わず、`sentencepiece_lite` + ONNX Runtime（CPU）で動作します。既定モデルは `Chottokun/ruri-v3-30m-lite`（256 次元）です。

## 特徴

- **Zero-Torch**: 依存は `fastapi`, `uvicorn`, `onnxruntime`, `numpy`, `huggingface-hub`, `sentencepiece_lite`（`cp311-abi3` wheel）のみ
- **OpenAI 互換 API**:
  - `POST /v1/embeddings`（`float` / `base64`、`dimensions`、`input_type` / `apply_ruri_prefix` 対応、`model` 省略時はロード済みモデルを使用）
  - `GET /v1/models`
  - `GET /health` / `GET /healthz` / `GET /`（モデル名・次元数を返却）
- **オフライン起動対応**: 初回起動時に Hugging Face Hub からモデルと lite loader（`lite_embed.py` / `ruri_v3_lite.py`）を取得・キャッシュし、以降は `HF_HUB_OFFLINE=1` でオフライン起動可能
- **任意 Bearer 認証**: `API_KEY` 環境変数が設定されている場合のみ `Authorization: Bearer <API_KEY>` を検証

## 動作環境・プラットフォーム制約

- `sentencepiece_lite` の公開 wheel が `x86_64`（`manylinux_2_27_x86_64.manylinux_2_28_x86_64`）向けのみのため、ローカル実行および Docker イメージは **`linux/amd64` 専用** です（ARM64 環境では `--platform=linux/amd64` エミュレーションまたは対応 wheel が必要になります）。

## 起動方法

### 1. uv で起動（ローカル）

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### 2. Docker で起動

```bash
docker build -t embedding_jp_api_lite .
docker run --rm -p 8080:8080 -v hf_cache:/data/huggingface embedding_jp_api_lite
```

コンテナは非 root ユーザー（`app`, UID 10001）で動作し、`/health` に対する `HEALTHCHECK` を備えています。

## 環境変数

| 変数名 | 既定値 | 説明 |
| --- | --- | --- |
| `LITE_MODEL` | `Chottokun/ruri-v3-30m-lite` | Hugging Face repo ID またはローカルモデルディレクトリ |
| `LITE_MODEL_REVISION` | `""`（未固定） | Hugging Face リポジトリのコミット SHA / ブランチ / タグ（本番ではコミット SHA 固定を推奨） |
| `HOST` | `0.0.0.0` | バインドホスト |
| `PORT` | `8080` | 待ち受けポート（`1`〜`65535` の整数） |
| `API_KEY` | `""`（未設定） | 設定時のみ Bearer 認証を有効化 |
| `LITE_MAX_LENGTH` | `512` | 最大トークン長（`>= 1`） |
| `LITE_BATCH_SIZE` | `64` | 推論バッチサイズ（`>= 1`） |
| `MAX_INPUT_LENGTH` | `65536` | 1 テキストあたりの最大文字数（起動時読込、`>= 1`） |
| `MAX_INPUT_ITEMS` | `256` | 1 リクエストあたりの最大入力配列長（起動時読込、`>= 1`） |
| `HF_HUB_OFFLINE` | `0` | `1` にするとキャッシュのみを使用し外部通信を行わない |

## セキュリティ上の注意（リモートローダーの実行境界）

本サーバーは `LITE_MODEL` に指定された Hugging Face リポジトリ（またはローカルディレクトリ）同梱の Python ローダー（`lite_embed.py` / `ruri_v3_lite.py`）を `importlib` 経由で読み込み実行します。

- **信頼できるリポジトリ・モデルディレクトリのみ** を指定してください。
- 本番運用では **`LITE_MODEL_REVISION` に不変なコミット SHA を指定** してコードおよび重みをピン留めすることを強く推奨します（未設定時は警告ログを出力します）。

## 接頭辞（query / document）とバリデーション仕様

### 接頭辞ルール

`ruri-v3` などの非対称検索モデルでは、検索クエリと検索文書で異なる接頭辞を使います。本サーバーは `lite_config.json`（未同梱時は loader 既定値）を参照し、以下のルールで適用します。

- `input_type="query"`: `query_prefix`（ruri-v3 では `"検索クエリ: "`）を付与
- `input_type="document"` または `"passage"`: `passage_prefix`（ruri-v3 では `"文章: "`）を付与
- `input_type="classification"` / `"clustering"`: `lite_config.json` の `classification_prefix`（未設定時は `"トピック: "`）を付与
- `input_type="sts"`: `lite_config.json` の `sts_prefix`（未設定時は `""`）を付与
- `input_type` 未指定かつ `apply_ruri_prefix=true`: 単一文字列なら `query_prefix`、配列なら `passage_prefix` を付与
- `input_type` 未指定（既定）: `input` をそのまま埋め込む（既定で接頭辞を重複付与しない）

`input_type` の前後の空白と大文字・小文字は正規化されます（例: `"QUERY"` → `"query"`）。既に入力文字列が対象接頭辞で始まっている場合は重複付与されません。

### 入力バリデーション

- 許可されていない `input_type`（例: `"doc"`）、空文字列 `""`、空配列 `[]`、`MAX_INPUT_ITEMS` / `MAX_INPUT_LENGTH` 超過、`dimensions < 1`、不正な `encoding_format` は **`422 Unprocessable Entity`** を返します。
- `dimensions` がロード済みモデルの埋め込み次元数（例: `256`）を超えている場合は **`400 Bad Request`** を返します。

## API 利用例

### curl

```bash
# ヘルスチェック
curl http://localhost:8080/health

# 検索クエリの埋め込み
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

## オフライン運用

初回起動（または事前ダウンロード）で Hugging Face キャッシュにモデル一式を落とした後、`HF_HUB_OFFLINE=1` を指定して起動します。

`pyproject.toml` に直リンク wheel（`sentencepiece-lite`）が含まれるため、オフライン環境で `uv run` を使う場合は同期時のメタデータ取得を避けるため `--no-sync`（または `--offline`）を併用してください。

```bash
HF_HUB_OFFLINE=1 uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Docker の場合はボリュームにキャッシュを残した状態で `-e HF_HUB_OFFLINE=1` を渡します。`LITE_MODEL_REVISION` でコミット SHA を固定している場合も、`snapshot_download` のキャッシュからそのままオフライン起動できます。

## モデルの差し替え

`LITE_MODEL` に別の lite モデル（HF repo ID または `model.onnx` / `.spm.fb` / `lite_config.json` / `lite_embed.py` を含むローカルディレクトリ）を指定して再起動します（1プロセス1モデル）。

```bash
LITE_MODEL=Chottokun/ruri-v3-70m-lite uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## 開発・テスト

```bash
# Lint / Format / Type-check
uv run ruff format --check .
uv run ruff check .
uv run mypy src

# ユニットテスト（モデルダウンロード不要）
uv run pytest -m "not integration" -v

# 結合テスト（実モデル使用・オフラインキャッシュでも可）
HF_HUB_OFFLINE=1 uv run --no-sync pytest -m integration -v

# 全検証ゲート（5項目）の実行
uv run python scripts/verify_gates.py
```

## ライセンス

[MIT License](LICENSE)
