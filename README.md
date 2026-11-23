# embedding_jp_api_lite

OpenAI 互換の軽量テキスト埋め込みサーバーです。`torch` / `transformers` を一切使わず、`sentencepiece_lite` + ONNX Runtime（CPU）で動作します。既定モデルは `Chottokun/ruri-v3-30m-lite`（256 次元）です。

## 特徴

- **Zero-Torch**: 依存は `fastapi`, `uvicorn`, `onnxruntime`, `numpy`, `huggingface-hub`, `sentencepiece_lite`（`cp311-abi3` wheel）のみ
- **OpenAI 互換 API**:
  - `POST /v1/embeddings`（`float` / `base64`、`dimensions`、`input_type` / `apply_ruri_prefix` 対応）
  - `GET /v1/models`
  - `GET /health` / `GET /healthz` / `GET /`（モデル名・次元数を返却）
- **オフライン起動対応**: 初回起動時に Hugging Face Hub からモデルと lite loader（`lite_embed.py` / `ruri_v3_lite.py`）を取得・キャッシュし、以降は `HF_HUB_OFFLINE=1` でオフライン起動可能
- **任意 Bearer 認証**: `API_KEY` 環境変数が設定されている場合のみ `Authorization: Bearer <API_KEY>` を検証

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

## 環境変数

| 変数名 | 既定値 | 説明 |
| --- | --- | --- |
| `LITE_MODEL` | `Chottokun/ruri-v3-30m-lite` | Hugging Face repo ID またはローカルモデルディレクトリ |
| `HOST` | `0.0.0.0` | バインドホスト |
| `PORT` | `8080` | 待ち受けポート |
| `API_KEY` | `""`（未設定） | 設定時のみ Bearer 認証を有効化 |
| `LITE_MAX_LENGTH` | `512` | 最大トークン長 |
| `LITE_BATCH_SIZE` | `64` | 推論バッチサイズ |
| `HF_HUB_OFFLINE` | `0` | `1` にするとキャッシュのみを使用し外部通信を行わない |

## 接頭辞（query / document）の切り替え方法

`ruri-v3` などの非対称検索モデルでは、検索クエリと検索文書で異なる接頭辞を使います。本サーバーは `lite_config.json`（未同梱時は loader 既定値）の `query_prefix` / `passage_prefix` を参照し、以下のルールで適用します。

- `input_type="query"`: `query_prefix`（ruri-v3 では `"検索クエリ: "`）を付与
- `input_type="document"` または `"passage"`: `passage_prefix`（ruri-v3 では `"文章: "`）を付与
- `input_type="classification"` / `"clustering"`: `"トピック: "` を付与
- `input_type="sts"`: 接頭辞なし（`""`）
- `input_type` 未指定かつ `apply_ruri_prefix=true`: 単一文字列なら `query_prefix`、配列なら `passage_prefix` を付与
- `input_type` 未指定（既定）: `input` をそのまま埋め込む（既定で接頭辞を重複付与しない）

既に入力文字列が対象接頭辞で始まっている場合は重複付与されません。

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

```bash
HF_HUB_OFFLINE=1 uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Docker の場合はボリュームにキャッシュを残した状態で `-e HF_HUB_OFFLINE=1` を渡します。

## モデルの差し替え

`LITE_MODEL` に別の lite モデル（HF repo ID または `model.onnx` / `.spm.fb` / `lite_config.json` / `lite_embed.py` を含むローカルディレクトリ）を指定して再起動します（1プロセス1モデル）。

```bash
LITE_MODEL=Chottokun/ruri-v3-70m-lite uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## 検証

```bash
uv run python scripts/verify_gates.py
```
