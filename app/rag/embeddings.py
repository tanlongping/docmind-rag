"""向量化实现。

`hashing`：本地哈希向量（feature hashing），零依赖、确定性。
  做法：把 token 用 blake2b 散列到 [0, dim)，再按散列的另一位取 ±1 做符号散列，
  最后 L2 归一化。符号散列能把碰撞造成的偏差转成近似正交的噪声，比单向累加更稳。
  代价：没有语义泛化能力（「汽车」和「轿车」不会相近），所以召回主要靠 BM25 兜底，
  这也是本项目坚持做混合检索、而不是纯向量检索的现实原因。
`openai`：调用任意 OpenAI 兼容的 /v1/embeddings（OpenAI / 通义 / bge-m3 服务均可）。
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx

from app.rag.text import tokenize


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _feature_hash(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


class HashingEmbedder:
    """确定性哈希向量，可离线复现，进程重启后向量仍然一致。"""

    name = "hashing"

    def __init__(self, dim: int = 384) -> None:
        if dim < 32:
            raise ValueError("dim 过小")
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        counts: dict[str, int] = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1

        for token, tf in counts.items():
            h = _feature_hash(token)
            index = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            # 次线性词频：出现 10 次不该是出现 1 次的 10 倍权重
            vector[index] += sign * (1.0 + math.log(tf))

        return _l2_normalize(vector)


class OpenAIEmbedder:
    """走 OpenAI 兼容的 /v1/embeddings，批量请求以减少 RTT。"""

    name = "openai"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        timeout: float = 30.0,
        batch_size: int = 32,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.timeout = timeout
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json={"model": self.model, "input": batch},
                )
                response.raise_for_status()
                data = response.json()["data"]
                data.sort(key=lambda item: item["index"])
                out.extend(_l2_normalize(item["embedding"]) for item in data)
        if out and len(out[0]) != self.dim:
            self.dim = len(out[0])
        return out


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    """向量已归一化，点积即余弦相似度；长度不等时按较短的截断，避免脏数据炸掉检索。"""
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


def build_embedder(settings) -> Embedder:
    if settings.embedding_provider == "openai":
        if not settings.llm_api_key:
            raise RuntimeError("embedding_provider=openai 需要配置 DOCMIND_LLM_API_KEY")
        return OpenAIEmbedder(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            dim=settings.embedding_dim,
        )
    return HashingEmbedder(settings.embedding_dim)
