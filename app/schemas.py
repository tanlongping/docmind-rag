"""Pydantic 入参/出参模型。

出参统一用 `model_config = ConfigDict(from_attributes=True)` 从 ORM 对象读取，
避免在路由里手写 dict 组装；入参一律做长度与格式约束，把脏数据挡在业务层之前。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(default="", max_length=64)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    display_name: str
    tenant_id: str
    role: str
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class KBCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)


class KBOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    created_at: datetime
    document_count: int = 0


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int
    status: str
    created_at: datetime


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(hybrid|vector|keyword)$")


class HitOut(BaseModel):
    chunk_id: int
    doc_id: int
    filename: str
    ordinal: int
    text: str
    score: float
    ranks: dict[str, int] = Field(default_factory=dict)


class SearchOut(BaseModel):
    query: str
    hits: list[HitOut]
    latency_ms: int


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: int | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(hybrid|vector|keyword)$")
    stream: bool = True


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime


class ChatAnswer(BaseModel):
    """非流式调用（stream=false）的返回体。"""

    session_id: int
    answer: str
    citations: list[dict]
    latency_ms: int


class StatsOut(BaseModel):
    users: int
    knowledge_bases: int
    documents: int
    chunks: int
    cached_indexes: int
    llm_mode: str
