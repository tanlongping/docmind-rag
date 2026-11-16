"""问答接口：SSE 流式输出 + 引用溯源 + 会话记忆。

为什么用 POST + SSE 而不是 WebSocket：
  - 一问一答的请求/响应语义，SSE 天然贴合，断线重连与代理穿透都比 WS 省心；
  - 需要双向交互（打断生成、多轮工具调用）时再升级到 WS，接口形状不用变。
注意 SSE 的响应头必须带 `X-Accel-Buffering: no`，否则 Nginx 会缓冲整个响应，
前端看起来就是「卡住十秒然后一次性吐完」。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.deps import (
    CurrentKB,
    CurrentUser,
    DbSession,
    get_embedder,
    get_metrics,
    get_pipeline,
    get_registry,
    get_settings,
)
from app.errors import NotFound
from app.metrics import Metrics
from app.models import ChatSession, Message
from app.rag.embeddings import Embedder
from app.rag.pipeline import AnswerPipeline, build_citations
from app.schemas import ChatAnswer, ChatIn, MessageOut
from app.services.retrieval import IndexRegistry

router = APIRouter(prefix="/api/v1/kbs", tags=["问答"])

HISTORY_LIMIT = 6


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _resolve_session(db: Session, kb_id: int, user_id: int, kb, session_id: int | None) -> ChatSession:
    if session_id is None:
        session = ChatSession(
            kb_id=kb_id, user_id=user_id, tenant_id=kb.tenant_id, title="新会话"
        )
        db.add(session)
        db.flush()
        return session

    session = db.get(ChatSession, session_id)
    if session is None or session.kb_id != kb_id or session.user_id != user_id:
        raise NotFound("会话不存在")
    return session


def _history(db: Session, session_id: int, limit: int = HISTORY_LIMIT) -> list[dict[str, str]]:
    messages = db.scalars(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.id.desc())
        .limit(limit)
    ).all()
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]


def _persist_assistant(session_factory, *, session_id: int, answer: str, citations: list[dict], latency_ms: int) -> None:
    """在独立 Session 里落库：流式响应期间请求级 Session 可能已经关闭。"""
    db = session_factory()
    try:
        db.add(
            Message(
                session_id=session_id,
                role="assistant",
                content=answer,
                citations=json.dumps(citations, ensure_ascii=False),
                latency_ms=latency_ms,
            )
        )
        db.commit()
    finally:
        db.close()


@router.post("/{kb_id}/chat", summary="知识库问答（默认 SSE 流式）")
async def chat(
    kb: CurrentKB,
    payload: ChatIn,
    request: Request,
    db: DbSession,
    user: CurrentUser,
    registry: Annotated[IndexRegistry, Depends(get_registry)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    pipeline: Annotated[AnswerPipeline, Depends(get_pipeline)],
    metrics: Annotated[Metrics, Depends(get_metrics)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    started = time.perf_counter()

    # 1) 检索
    retriever = registry.get(kb.id)
    query_vector = embedder.embed([payload.question])[0]
    hits = retriever.search(
        payload.question, query_vector, top_k=payload.top_k, mode=payload.mode
    )
    retrieve_ms = int((time.perf_counter() - started) * 1000)
    metrics.observe("docmind_retrieval_duration", retrieve_ms / 1000)
    metrics.inc("docmind_chat_requests_total")

    citations = [c.to_dict() for c in build_citations(hits)]

    # 2) 会话与历史
    session = _resolve_session(db, kb.id, user.id, kb, payload.session_id)
    history = _history(db, session.id)
    db.add(Message(session_id=session.id, role="user", content=payload.question))
    if session.title == "新会话":
        session.title = payload.question[:40]
    db.commit()
    session_id = session.id

    # 3) 非流式：一次性返回，方便脚本调用与自动化测试
    if not payload.stream:
        buffer: list[str] = []
        async for delta in pipeline.stream(question=payload.question, hits=hits, history=history):
            buffer.append(delta)
        answer = "".join(buffer)
        latency = int((time.perf_counter() - started) * 1000)
        _persist_assistant(
            request.app.state.session_factory,
            session_id=session_id,
            answer=answer,
            citations=citations,
            latency_ms=latency,
        )
        metrics.inc("docmind_chat_completions_total")
        return ChatAnswer(
            session_id=session_id, answer=answer, citations=citations, latency_ms=latency
        )

    # 4) 流式：先推 meta（引用），再逐个 token，最后 done
    session_factory = request.app.state.session_factory

    async def event_stream() -> AsyncIterator[str]:
        buffer: list[str] = []
        yield _sse(
            "meta",
            {
                "session_id": session_id,
                "mode": pipeline.mode,
                "citations": citations,
                "retrieve_ms": retrieve_ms,
                "model": settings.llm_model if pipeline.mode != "demo" else "demo",
            },
        )
        try:
            async for delta in pipeline.stream(
                question=payload.question, hits=hits, history=history
            ):
                if not delta:
                    continue
                buffer.append(delta)
                yield _sse("token", {"text": delta})
        except Exception as exc:  # noqa: BLE001 - 上游模型异常不应打断 SSE 连接
            metrics.inc("docmind_chat_errors_total")
            yield _sse("error", {"message": f"生成中断：{type(exc).__name__}"})
            return

        answer = "".join(buffer)
        latency = int((time.perf_counter() - started) * 1000)
        _persist_assistant(
            session_factory,
            session_id=session_id,
            answer=answer,
            citations=citations,
            latency_ms=latency,
        )
        metrics.inc("docmind_chat_completions_total")
        metrics.observe("docmind_chat_duration", latency / 1000)
        metrics.inc("docmind_chat_chars_total", value=len(answer))
        yield _sse("done", {"session_id": session_id, "chars": len(answer), "latency_ms": latency})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{kb_id}/sessions", summary="会话列表")
def list_sessions(kb: CurrentKB, db: DbSession, user: CurrentUser) -> list[dict]:
    sessions = db.scalars(
        select(ChatSession)
        .where(ChatSession.kb_id == kb.id, ChatSession.user_id == user.id)
        .order_by(ChatSession.id.desc())
    ).all()
    return [
        {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.isoformat(),
        }
        for s in sessions
    ]


@router.get("/{kb_id}/sessions/{session_id}/messages", response_model=list[MessageOut], summary="会话消息")
def list_messages(kb: CurrentKB, session_id: int, db: DbSession, user: CurrentUser) -> list[MessageOut]:
    session = db.get(ChatSession, session_id)
    if session is None or session.kb_id != kb.id or session.user_id != user.id:
        raise NotFound("会话不存在")
    messages = db.scalars(
        select(Message).where(Message.session_id == session_id).order_by(Message.id)
    ).all()
    return [MessageOut.model_validate(m) for m in messages]
