"""知识库 CRUD。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import func, select

from app.deps import CurrentKB, CurrentUser, DbSession, get_registry
from app.errors import Conflict
from app.models import Document, KnowledgeBase
from app.schemas import KBCreate, KBOut
from app.services.retrieval import IndexRegistry

router = APIRouter(prefix="/api/v1/kbs", tags=["知识库"])


@router.post("", response_model=KBOut, status_code=status.HTTP_201_CREATED, summary="创建知识库")
def create_kb(payload: KBCreate, db: DbSession, user: CurrentUser) -> KBOut:
    exists = db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.tenant_id == user.tenant_id, KnowledgeBase.name == payload.name
        )
    )
    if exists is not None:
        raise Conflict("同名知识库已存在")

    kb = KnowledgeBase(
        tenant_id=user.tenant_id,
        name=payload.name,
        description=payload.description,
        created_by=user.id,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return KBOut.model_validate(kb)


@router.get("", response_model=list[KBOut], summary="我的知识库列表")
def list_kbs(db: DbSession, user: CurrentUser) -> list[KBOut]:
    rows = db.execute(
        select(KnowledgeBase, func.count(Document.id))
        .outerjoin(Document, Document.kb_id == KnowledgeBase.id)
        .where(KnowledgeBase.tenant_id == user.tenant_id)
        .group_by(KnowledgeBase.id)
        .order_by(KnowledgeBase.id.desc())
    ).all()

    result = []
    for kb, doc_count in rows:
        item = KBOut.model_validate(kb)
        item.document_count = int(doc_count)
        result.append(item)
    return result


@router.get("/{kb_id}", response_model=KBOut, summary="知识库详情")
def get_kb_detail(kb: CurrentKB, db: DbSession) -> KBOut:
    doc_count = db.scalar(select(func.count(Document.id)).where(Document.kb_id == kb.id)) or 0
    item = KBOut.model_validate(kb)
    item.document_count = int(doc_count)
    return item


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT, summary="删除知识库")
def delete_kb(
    kb: CurrentKB,
    db: DbSession,
    registry: Annotated[IndexRegistry, Depends(get_registry)],
) -> None:
    db.delete(kb)
    db.commit()
    registry.invalidate(kb.id)
