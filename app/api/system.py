"""健康检查、指标、运营统计。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, select

from app.deps import CurrentUser, DbSession, get_metrics, get_pipeline, get_registry
from app.metrics import Metrics
from app.models import Chunk, Document, KnowledgeBase, User
from app.rag.pipeline import AnswerPipeline
from app.schemas import StatsOut
from app.services.retrieval import IndexRegistry

router = APIRouter(tags=["系统"])


@router.get("/healthz", summary="存活探针", include_in_schema=False)
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz", summary="就绪探针", include_in_schema=False)
def readyz(
    db: DbSession,
    pipeline: Annotated[AnswerPipeline, Depends(get_pipeline)],
) -> dict:
    db.execute(select(1))
    return {"status": "ready", "llm_mode": pipeline.mode}


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def metrics_endpoint(
    db: DbSession,
    metrics: Annotated[Metrics, Depends(get_metrics)],
    registry: Annotated[IndexRegistry, Depends(get_registry)],
) -> str:
    metrics.set_gauge("docmind_documents_total", db.scalar(select(func.count(Document.id))) or 0)
    metrics.set_gauge("docmind_chunks_total", db.scalar(select(func.count(Chunk.id))) or 0)
    metrics.set_gauge("docmind_cached_indexes", registry.cached_kbs)
    return metrics.render()


@router.get("/api/v1/stats", response_model=StatsOut, summary="租户内统计")
def stats(
    db: DbSession,
    user: CurrentUser,
    registry: Annotated[IndexRegistry, Depends(get_registry)],
    pipeline: Annotated[AnswerPipeline, Depends(get_pipeline)],
) -> StatsOut:
    tenant = user.tenant_id
    return StatsOut(
        users=db.scalar(select(func.count(User.id)).where(User.tenant_id == tenant)) or 0,
        knowledge_bases=db.scalar(
            select(func.count(KnowledgeBase.id)).where(KnowledgeBase.tenant_id == tenant)
        )
        or 0,
        documents=db.scalar(select(func.count(Document.id)).where(Document.tenant_id == tenant)) or 0,
        chunks=db.scalar(select(func.count(Chunk.id)).where(Chunk.tenant_id == tenant)) or 0,
        cached_indexes=registry.cached_kbs,
        llm_mode=pipeline.mode,
    )
