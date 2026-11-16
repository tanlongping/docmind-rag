"""向量化与检索的单元测试。"""

from __future__ import annotations

import unittest

from app.rag.embeddings import HashingEmbedder, cosine
from app.rag.index import ChunkRecord, HybridRetriever, build_retriever
from app.rag.text import tokenize

DOCS = {
    1: "退款政策：签收后 7 个自然日内，商品未拆封可以申请全额退款，原路退回。",
    2: "发票规则：订单完成后 30 天内可以申请电子普通发票，专用发票需联系客服。",
    3: "配送范围：全国大部分地区支持次日达，偏远地区 3 到 5 天送达。",
    4: "客服时间：每天 9:00 到 21:00 在线，节假日不休。",
}


def make_records() -> list[ChunkRecord]:
    embedder = HashingEmbedder(256)
    texts = list(DOCS.values())
    vectors = embedder.embed(texts)
    return [
        ChunkRecord(chunk_id=i, doc_id=i, filename=f"doc{i}.txt", ordinal=0, text=text, vector=vector)
        for i, (text, vector) in zip(DOCS, zip(texts, vectors, strict=True), strict=True)
    ]


class EmbeddingTests(unittest.TestCase):
    def test_deterministic_and_normalized(self) -> None:
        embedder = HashingEmbedder(128)
        first = embedder.embed(["退款政策说明"])[0]
        second = HashingEmbedder(128).embed(["退款政策说明"])[0]
        self.assertEqual(first, second, "同一文本在不同进程必须得到相同向量")
        norm = sum(v * v for v in first) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)
        self.assertEqual(len(first), 128)

    def test_self_similarity_is_one(self) -> None:
        embedder = HashingEmbedder(128)
        vector = embedder.embed(["退款政策说明"])[0]
        self.assertAlmostEqual(cosine(vector, vector), 1.0, places=6)

    def test_similar_text_scores_higher_than_unrelated(self) -> None:
        embedder = HashingEmbedder(256)
        base, near, far = embedder.embed(
            ["退款政策：七天无理由退款", "退款政策说明：支持七天无理由退款", "配送范围：偏远地区三天送达"]
        )
        self.assertGreater(cosine(base, near), cosine(base, far))

    def test_dim_validation(self) -> None:
        with self.assertRaises(ValueError):
            HashingEmbedder(8)


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = make_records()
        self.retriever = build_retriever(self.records)
        self.embedder = HashingEmbedder(256)

    def _search(self, query: str, mode: str = "hybrid", top_k: int = 2):
        return self.retriever.search(
            query, self.embedder.embed([query])[0], top_k=top_k, mode=mode
        )

    def test_keyword_mode_hits_expected_doc(self) -> None:
        hits = self._search("退款", mode="keyword")
        self.assertEqual(hits[0].doc_id, 1)
        self.assertIn("bm25", hits[0].ranks)

    def test_hybrid_fuses_two_channels(self) -> None:
        hits = self._search("发票怎么开", mode="hybrid")
        self.assertEqual(hits[0].doc_id, 2)
        self.assertTrue(hits[0].ranks)

    def test_vector_mode_returns_something(self) -> None:
        hits = self._search("配送要几天", mode="vector")
        self.assertTrue(hits)
        self.assertIn("vector", hits[0].ranks)

    def test_top_k_limit(self) -> None:
        hits = self._search("退款", top_k=1)
        self.assertEqual(len(hits), 1)

    def test_empty_index(self) -> None:
        empty = HybridRetriever([])
        self.assertEqual(empty.search("退款", [0.0] * 8, top_k=3), [])

    def test_ranked_scores_are_descending(self) -> None:
        hits = self._search("退款 发票 配送", top_k=3)
        scores = [hit.score for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_bm25_ignores_stopword_only_query(self) -> None:
        hits = self.retriever.search("的", [0.0] * 256, top_k=3, mode="keyword")
        self.assertEqual(hits, [])

    def test_tokenize_is_used_consistently(self) -> None:
        tokens = tokenize("七天无理由退款")
        self.assertIn("退款", tokens)
        self.assertIn("理由", tokens)


if __name__ == "__main__":
    unittest.main()
