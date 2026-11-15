"""检索：BM25 + 向量召回 + RRF 融合。

为什么要混合检索：
  - 哈希向量没有语义能力，但 Embedding 服务不可用时它是唯一可用的召回通道；
  - BM25 对「专有名词、错误码、版本号」这类精确匹配极强，纯向量反而容易漂；
  - 两路结果用 RRF（Reciprocal Rank Fusion）融合，只依赖排名不依赖分数尺度，
    天然回避了「BM25 分数 0~30、余弦 0~1 不可比」的问题。

RRF: score(d) = Σ_i 1 / (k + rank_i(d))，k 取 60（原论文经验值）。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from app.rag.embeddings import cosine
from app.rag.text import tokenize


@dataclass(slots=True)
class ChunkRecord:
    """内存索引里的一条切片，doc_id/filename 用于生成引用。"""

    chunk_id: int
    doc_id: int
    filename: str
    ordinal: int
    text: str
    vector: list[float] = field(default_factory=list)


@dataclass(slots=True)
class SearchHit:
    chunk_id: int
    doc_id: int
    filename: str
    ordinal: int
    text: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


class BM25Index:
    """标准 Okapi BM25，参数取 k1=1.5 / b=0.75。"""

    def __init__(self, documents: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_count = len(documents)
        self.doc_len = [len(doc) for doc in documents]
        self.avg_len = (sum(self.doc_len) / self.doc_count) if self.doc_count else 0.0
        self.term_freq: list[dict[str, int]] = []
        self.doc_freq: dict[str, int] = defaultdict(int)

        for doc in documents:
            tf: dict[str, int] = defaultdict(int)
            for token in doc:
                tf[token] += 1
            self.term_freq.append(dict(tf))
            for token in tf:
                self.doc_freq[token] += 1

    def idf(self, token: str) -> float:
        df = self.doc_freq.get(token, 0)
        # Lucene 风格 IDF，恒为正，避免高频词把分数压成负数
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.doc_count
        if not self.doc_count or not query_tokens:
            return scores

        for token in set(query_tokens):
            idf = self.idf(token)
            if idf <= 0:
                continue
            for i, tf_map in enumerate(self.term_freq):
                tf = tf_map.get(token)
                if not tf:
                    continue
                dl = self.doc_len[i] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores


class VectorIndex:
    """暴力余弦检索。几千条切片下延迟在毫秒级，比引入 FAISS/Chroma 更划算。"""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    def scores(self, query_vector: list[float]) -> list[float]:
        return [cosine(query_vector, vector) for vector in self.vectors]


class HybridRetriever:
    def __init__(
        self,
        records: list[ChunkRecord],
        *,
        rrf_k: int = 60,
        vector_weight: float = 0.35,
    ) -> None:
        self.records = records
        self.rrf_k = rrf_k
        # 向量通道权重 < 1 的原因：哈希向量没有语义能力，与 BM25 等权融合会把
        # 「字面命中」的强信号稀释掉（实测 recall@1 从 1.00 掉到 0.81）。
        # 换成真实 embedding 后可以把该值调到 1.0~1.5，用 scripts/eval_retrieval.py 复测。
        self.vector_weight = vector_weight
        self._bm25 = BM25Index([tokenize(record.text) for record in records])
        self._vectors = VectorIndex([record.vector for record in records])

    def __len__(self) -> int:
        return len(self.records)

    def search(
        self,
        query: str,
        query_vector: list[float],
        *,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[SearchHit]:
        if not self.records:
            return []
        top_k = max(1, top_k)
        ranks: dict[int, dict[str, int]] = defaultdict(dict)
        scores: dict[int, float] = defaultdict(float)

        def rank_by(values: list[float], label: str, *, weight: float = 1.0, floor: float = 0.0):
            order = sorted(
                (i for i, value in enumerate(values) if value > floor),
                key=lambda i: values[i],
                reverse=True,
            )
            for position, index in enumerate(order, start=1):
                ranks[index][label] = position
                scores[index] += weight / (self.rrf_k + position)

        if mode in ("hybrid", "keyword"):
            rank_by(self._bm25.scores(tokenize(query)), "bm25")
        if mode in ("hybrid", "vector") and any(query_vector):
            # 余弦可能为负，截断到 0 以上再排序
            rank_by(
                self._vectors.scores(query_vector),
                "vector",
                weight=self.vector_weight,
            )

        ordered = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
        hits: list[SearchHit] = []
        for index in ordered:
            record = self.records[index]
            hits.append(
                SearchHit(
                    chunk_id=record.chunk_id,
                    doc_id=record.doc_id,
                    filename=record.filename,
                    ordinal=record.ordinal,
                    text=record.text,
                    score=round(scores[index], 6),
                    ranks=dict(ranks[index]),
                )
            )
        return hits


def build_retriever(records: list[ChunkRecord], *, vector_weight: float = 0.35) -> HybridRetriever:
    return HybridRetriever(records, vector_weight=vector_weight)
