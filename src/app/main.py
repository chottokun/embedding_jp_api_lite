"""FastAPI application for embedding_jp_api_lite."""

from contextlib import asynccontextmanager
import sys
from typing import Optional

import anyio
from fastapi import Depends, FastAPI, HTTPException
import uvicorn

from app.auth import verify_api_key
from app.config import get_host, get_lite_model, get_port
from app.loader import LiteEmbeddingEngine
from app.schemas import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    HealthResponse,
    ModelCard,
    ModelList,
    Usage,
)

_engine: Optional[LiteEmbeddingEngine] = None


def get_engine() -> LiteEmbeddingEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return _engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine
    model_id = get_lite_model()
    print(f"[embedding_jp_api_lite] Loading model: {model_id}")
    _engine = LiteEmbeddingEngine(model_id=model_id)
    torch_loaded = "torch" in sys.modules or "transformers" in sys.modules
    print(
        f"[embedding_jp_api_lite] Ready: model={_engine.model_id}, "
        f"dim={_engine.embedding_dim}, torch_or_transformers_loaded={torch_loaded}"
    )
    yield
    _engine = None


app = FastAPI(
    title="embedding_jp_api_lite",
    version="0.1.0",
    description="OpenAI-compatible lightweight embedding server (sentencepiece_lite + ONNX Runtime, Zero-Torch).",
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def health(engine: LiteEmbeddingEngine = Depends(get_engine)):
    """Health check returning loaded model ID and embedding dimension."""
    return HealthResponse(
        status="ok",
        model=engine.model_id,
        embedding_dim=engine.embedding_dim,
    )


@app.get(
    "/v1/models",
    response_model=ModelList,
    dependencies=[Depends(verify_api_key)],
    responses={401: {"model": ErrorResponse}},
    tags=["Models"],
)
async def list_models(engine: LiteEmbeddingEngine = Depends(get_engine)):
    """List the currently loaded model in OpenAI-compatible format."""
    return ModelList(
        data=[
            ModelCard(
                id=engine.model_id,
                created=engine.created_at,
                owned_by="local",
            )
        ]
    )


@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(verify_api_key)],
    responses={
        401: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Embeddings"],
)
async def create_embeddings(
    request: EmbeddingRequest,
    engine: LiteEmbeddingEngine = Depends(get_engine),
):
    """Create embeddings for input string or list of strings."""
    is_single_str = isinstance(request.input, str)
    raw_texts = [request.input] if is_single_str else list(request.input)

    prefix = engine.resolve_prefix(
        input_type=request.input_type,
        apply_ruri_prefix=request.apply_ruri_prefix,
        is_single_str=is_single_str,
    )
    processed_texts = engine.apply_prefix(raw_texts, prefix)

    formatted_embs, token_count = await anyio.to_thread.run_sync(
        engine.encode_and_format,
        processed_texts,
        request.dimensions,
        request.encoding_format,
    )

    return EmbeddingResponse(
        data=[EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(formatted_embs)],
        model=request.model or engine.model_id,
        usage=Usage(prompt_tokens=token_count, total_tokens=token_count),
    )


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=get_host(),
        port=get_port(),
        workers=1,
    )


if __name__ == "__main__":
    main()
