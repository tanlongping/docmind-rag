"""文档上传 / 列表 / 删除 / 检索。

上传走 `async def` 是为了异步读文件；真正的解析+向量化是 CPU/IO 混合操作，
用 `run_in_threadpool` 丢到线程池，避免阻塞事件循环（否则一个 5MB 文档就能把 SSE 挤住）。
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.deps import (
    CurrentKB,
    DbSession,
    get_embedder,
    get_metrics,
    get_pipeline,
    get_registry,
    get_settings,
)
from app.errors import BadRequest, NotFound
from app.metrics import Metrics
from app.models import Chunk, Document
from app.rag.embeddings import Embedder
from app.rag.pipeline import AnswerPipeline
from app.schemas import DocumentOut, HitOut, SearchIn, SearchOut
from app.services.ingest import DocumentIngestor
from app.services.retrieval import IndexRegistry

router = APIRouter(prefix="/api/v1/kbs", tags=["文档与检索"])


@router.post(
    "/{kb_id}/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="上传文档（txt / md / pdf）",
)
async def upload_document(
    kb: CurrentKB,
    db: DbSession,
    file: Annotated[UploadFile, File(description="待入库文件")],
    registry: Annotated[IndexRegistry, Depends(get_registry)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    metrics: Annotated[Metrics, Depends(get_metrics)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentOut:
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise BadRequest(f"文件超过大小限制（{settings.max_upload_bytes} 字节）")

    ingestor = DocumentIngestor(settings, embedder)
    result = await run_in_threadpool(
        ingestor.ingest,
        db,
        kb=kb,
        filename=file.filename or "untitled.txt",
        raw=raw,
        content_type=file.content_type or "text/plain",
    )
    registry.invalidate(kb.id)
    metrics.inc("docmind_documents_ingested_total")
    metrics.inc("docmind_chunks_ingested_total", value=result.chunk_count)

    document = db.get(Document, result.document_id)
    if document is None:  # pragma: no cover - 理论上不可能
        raise NotFound("文档创建失败")
    return DocumentOut.model_validate(document)


@router.get("/{kb_id}/documents", response_model=list[DocumentOut], summary="文档列表")
def list_documents(kb: CurrentKB, db: DbSession) -> list[DocumentOut]:
    documents = db.scalars(
        select(Document).where(Document.kb_id == kb.id).order_by(Document.id.desc())
    ).all()
    return [DocumentOut.model_validate(doc) for doc in documents]


@router.delete(
    "/{kb_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除文档及其切片",
)
def delete_document(
    kb: CurrentKB,
    document_id: int,
    db: DbSession,
    registry: Annotated[IndexRegistry, Depends(get_registry)],
) -> None:
    document = db.get(Document, document_id)
    if document is None or document.kb_id != kb.id:
        raise NotFound("文档不存在")
    db.delete(document)  # 关系上配置了 cascade，切片一并删除
    db.commit()
    registry.invalidate(kb.id)


@router.post("/{kb_id}/search", response_model=SearchOut, summary="混合检索（不调用大模型）")
def search(
    kb: CurrentKB,
    payload: SearchIn,
    registry: Annotated[IndexRegistry, Depends(get_registry)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    pipeline: Annotated[AnswerPipeline, Depends(get_pipeline)],
) -> SearchOut:
    started = time.perf_counter()
    retriever = registry.get(kb.id)
    query_vector = embedder.embed([payload.query])[0]
    hits = retriever.search(payload.query, query_vector, top_k=payload.top_k, mode=payload.mode)
    elapsed = int((time.perf_counter() - started) * 1000)

    return SearchOut(
        query=payload.query,
        hits=[
            HitOut(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                filename=hit.filename,
                ordinal=hit.ordinal,
                text=hit.text,
                score=hit.score,
                ranks=hit.ranks,
            )
            for hit in hits
        ],
        latency_ms=elapsed,
    )


@router.get("/{kb_id}/chunks/{chunk_id}", summary="查看切片原文（引用溯源）")
def get_chunk(kb: CurrentKB, chunk_id: int, db: DbSession) -> dict:
    chunk = db.get(Chunk, chunk_id)
    if chunk is None or chunk.kb_id != kb.id:
        raise NotFound("切片不存在")
    document = db.get(Document, chunk.doc_id)
    return {
        "chunk_id": chunk.id,
        "doc_id": chunk.doc_id,
        "filename": document.filename if document else "",
        "ordinal": chunk.ordinal,
        "text": chunk.text,
    }
