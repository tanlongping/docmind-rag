"""文档入库：解析 → 切片 → 向量化 → 落库。

入库设计成幂等：同一知识库里 content 完全相同的文档会命中 checksum 冲突，
直接 409 而不是产生一堆重复切片（重复切片会挤占 top_k，是 RAG 里最隐蔽的脏数据问题之一）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.errors import BadRequest, Conflict
from app.models import Chunk, Document, KnowledgeBase
from app.rag.embeddings import Embedder
from app.rag.text import normalize, split_text

SUPPORTED_TEXT_TYPES = {
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/json",
    "text/csv",
}


@dataclass(slots=True)
class IngestResult:
    document_id: int
    filename: str
    chunk_count: int
    size_bytes: int
    duplicated: bool = False


class DocumentIngestor:
    def __init__(self, settings, embedder: Embedder) -> None:
        self.settings = settings
        self.embedder = embedder

    def ingest(
        self,
        session: Session,
        *,
        kb: KnowledgeBase,
        filename: str,
        raw: bytes,
        content_type: str = "text/plain",
    ) -> IngestResult:
        if not raw:
            raise BadRequest("文件内容为空")
        if len(raw) > self.settings.max_upload_bytes:
            raise BadRequest(
                f"文件超过大小限制（{len(raw)} > {self.settings.max_upload_bytes} 字节）"
            )

        text = self._decode(raw, filename, content_type)
        text = normalize(text)
        if not text:
            raise BadRequest("解析后没有可用文本内容")

        checksum = hashlib.sha256(raw).hexdigest()
        existed = session.scalar(
            select(Document).where(Document.kb_id == kb.id, Document.checksum == checksum)
        )
        if existed is not None:
            raise Conflict(f"该文档已存在于知识库中（document_id={existed.id}）")

        pieces = split_text(
            text, size=self.settings.chunk_size, overlap=self.settings.chunk_overlap
        )
        if not pieces:
            raise BadRequest("切片结果为空")

        vectors = self.embedder.embed(pieces)

        document = Document(
            kb_id=kb.id,
            tenant_id=kb.tenant_id,
            filename=filename[:255],
            content_type=content_type,
            size_bytes=len(raw),
            checksum=checksum,
            status="ready",
            chunk_count=len(pieces),
        )
        session.add(document)
        session.flush()  # 拿到 document.id，避免逐条 flush

        session.add_all(
            [
                Chunk(
                    kb_id=kb.id,
                    doc_id=document.id,
                    tenant_id=kb.tenant_id,
                    ordinal=ordinal,
                    text=piece,
                    embedding=_dump_vector(vector),
                )
                for ordinal, (piece, vector) in enumerate(zip(pieces, vectors, strict=True))
            ]
        )
        session.commit()

        return IngestResult(
            document_id=document.id,
            filename=document.filename,
            chunk_count=len(pieces),
            size_bytes=len(raw),
        )

    def _decode(self, raw: bytes, filename: str, content_type: str) -> str:
        lower = filename.lower()
        if content_type == "application/pdf" or lower.endswith(".pdf"):
            return self._extract_pdf(raw)
        if lower.endswith((".doc", ".docx")) or content_type in (
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ):
            raise BadRequest("暂不支持 Word 解析，请先导出为 PDF 或纯文本")

        # 中文文档常见 GBK 编码，按 UTF-8 → GB18030 顺序尝试
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _extract_pdf(raw: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - 依赖可选
            raise BadRequest("处理 PDF 需要安装 pypdf：pip install '.[pdf]'") from exc

        import io

        reader = PdfReader(io.BytesIO(raw))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(page for page in pages if page)
        if not text:
            raise BadRequest("PDF 中没有可提取的文本（可能是扫描件，需要 OCR）")
        return text


def _dump_vector(vector: list[float]) -> str:
    # 保留 6 位小数足够，能把存储体积压到原始 JSON 的 1/3 左右
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"


def load_vector(raw: str) -> list[float]:
    if not raw:
        return []
    try:
        import json

        return [float(v) for v in json.loads(raw)]
    except (ValueError, TypeError):
        return []
