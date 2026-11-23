"""Pydantic schemas for OpenAI-compatible embedding API."""

from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field, StringConstraints

from app.config import MAX_INPUT_ITEMS, MAX_INPUT_LENGTH

LimitedString = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_INPUT_LENGTH)
]


class EmbeddingRequest(BaseModel):
    input: Union[
        LimitedString,
        Annotated[list[LimitedString], Field(min_length=1, max_length=MAX_INPUT_ITEMS)],
    ] = Field(description="Input text or list of texts to embed.")
    model: str = Field(
        default="Chottokun/ruri-v3-30m-lite",
        description="Model identifier.",
    )
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Format to return embeddings in ('float' or 'base64').",
    )
    dimensions: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional output embedding dimensions (Matryoshka truncation + L2 re-normalization).",
    )
    input_type: Optional[str] = Field(
        default=None,
        description="Input task type: 'query', 'document', 'passage', 'classification', 'clustering', 'sts'.",
    )
    apply_ruri_prefix: bool = Field(
        default=False,
        description="Automatically apply query/passage prefix based on input shape when input_type is omitted.",
    )
    user: Optional[str] = Field(
        default=None,
        description="Optional end-user identifier.",
    )


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: Union[list[float], str]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    embedding_dim: int


class ErrorResponse(BaseModel):
    detail: str
