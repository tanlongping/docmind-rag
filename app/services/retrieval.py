"""索引注册表：按知识库懒加载内存索引，并用「版本号」判断是否需要重建。

版本号 = (切片数, 最大切片 id)。入库/删除都会改变其中之一，从而自动失效，
不需要在业务代码里手工调用 invalidate（虽然也保留了该方法做显式控制）。
"""

from __future__ import annotations

import threading

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Chunk, Document
from app.rag.index import ChunkRecord, HybridRetriever
from app.services.ingest import load_vector


class IndexRegistry:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        vector_weight: float = 0.35,
        max_kbs: int = 128,
    ) -> None:
        self._session_factory = session_factory
        self._vector_weight = vector_weight
        self._cache: dict[int, tuple[tuple[int, int], HybridRetriever]] = {}
        self._lock = threading.RLock()
        self._max_kbs = max_kbs

    def _version(self, session: Session, kb_id: int) -> tuple[int, int]:
        count, max_id = session.execute(
            select(func.count(Chunk.id), func.coalesce(func.max(Chunk.id), 0)).where(
                Chunk.kb_id == kb_id
            )
        ).one()
        return int(count), int(max_id)

    def get(self, kb_id: int) -> HybridRetriever:
        session = self._session_factory()
        try:
            version = self._version(session, kb_id)
            with self._lock:
                cached = self._cache.get(kb_id)
                if cached and cached[0] == version:
                    return cached[1]

            rows = session.execute(
                select(Chunk, Document.filename)
                .join(Document, Document.id == Chunk.doc_id)
                .where(Chunk.kb_id == kb_id)
                .order_by(Chunk.doc_id, Chunk.ordinal)
            ).all()

            records = [
                ChunkRecord(
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    filename=filename,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    vector=load_vector(chunk.embedding),
                )
                for chunk, filename in rows
            ]
            retriever = HybridRetriever(records, vector_weight=self._vector_weight)

            with self._lock:
                if len(self._cache) >= self._max_kbs:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[kb_id] = (version, retriever)
            return retriever
        finally:
            session.close()

    def invalidate(self, kb_id: int) -> None:
        with self._lock:
            self._cache.pop(kb_id, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def cached_kbs(self) -> int:
        with self._lock:
            return len(self._cache)
