"""离线检索评测：recall@k 与 MRR。

不启动 HTTP、不连数据库，直接把 data/corpus 切片建索引，跑 data/eval_set.jsonl。
这样评测结果只跟「切片 + 检索算法」有关，不会被接口层噪声污染，也方便回归对比。

用法：
    python scripts/eval_retrieval.py                       # 默认哈希向量、融合权重 0.6
    python scripts/eval_retrieval.py --vector-weight 1.0   # 复现「等权融合反而变差」
    python scripts/eval_retrieval.py --top-k 3 --dim 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag.embeddings import HashingEmbedder  # noqa: E402
from app.rag.index import ChunkRecord, HybridRetriever  # noqa: E402
from app.rag.text import split_text  # noqa: E402

CORPUS_DIR = ROOT / "data" / "corpus"
EVAL_SET = ROOT / "data" / "eval_set.jsonl"
MODES = ("hybrid", "keyword", "vector")


def build_index(dim: int, chunk_size: int, overlap: int, vector_weight: float):
    embedder = HashingEmbedder(dim)
    records: list[ChunkRecord] = []
    chunk_id = 0
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for ordinal, piece in enumerate(split_text(text, size=chunk_size, overlap=overlap)):
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id, doc_id=chunk_id, filename=path.name, ordinal=ordinal, text=piece
                )
            )
            chunk_id += 1

    vectors = embedder.embed([record.text for record in records])
    for record, vector in zip(records, vectors):
        record.vector = vector
    return HybridRetriever(records, vector_weight=vector_weight), embedder


def evaluate(retriever: HybridRetriever, embedder, cases: list[dict], mode: str, k: int) -> dict:
    hits_at = {1: 0, 3: 0, k: 0}
    reciprocal_rank = 0.0
    for case in cases:
        hits = retriever.search(
            case["query"], embedder.embed([case["query"]])[0], top_k=k, mode=mode
        )
        names = [hit.filename for hit in hits]
        expected = case["expected_doc"]
        if expected in names:
            rank = names.index(expected) + 1
            reciprocal_rank += 1.0 / rank
            hits_at[1] += rank <= 1
            hits_at[3] += rank <= 3
            hits_at[k] += 1
    total = max(1, len(cases))
    return {
        "recall@1": hits_at[1] / total,
        "recall@3": hits_at[3] / total,
        f"recall@{k}": hits_at[k] / total,
        "MRR": reciprocal_rank / total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="离线检索评测")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.35)
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in EVAL_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retriever, embedder = build_index(args.dim, args.chunk_size, args.overlap, args.vector_weight)

    print(f"语料：{len(list(CORPUS_DIR.glob('*.md')))} 篇 / {len(retriever)} 个切片")
    print(
        f"评测集：{len(cases)} 条查询 / 向量维度 {args.dim} / "
        f"top_k={args.top_k} / 向量融合权重={args.vector_weight}\n"
    )

    columns = ("recall@1", "recall@3", f"recall@{args.top_k}", "MRR")
    header = f"{'mode':<10}" + "".join(f"{c:>12}" for c in columns)
    print(header)
    print("-" * len(header))

    results = {}
    for mode in MODES:
        row = evaluate(retriever, embedder, cases, mode, args.top_k)
        results[mode] = row
        print(f"{mode:<10}" + "".join(f"{row[c]:>12.3f}" for c in columns))

    best = max(results, key=lambda m: (results[m]["MRR"], results[m]["recall@1"]))
    print(f"\n本语料 + 哈希向量下最优：{best}（MRR {results[best]['MRR']:.3f}）")
    print(
        "说明：哈希向量只有字面相似度、没有语义泛化能力，因此等权融合会稀释 BM25 的精确命中；\n"
        "     接入真实语义向量（EMBEDDING_PROVIDER=openai）后应重新调大 --vector-weight 复测。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
