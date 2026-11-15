"""问答编排：检索 → 组装 Prompt → 流式生成。

LangChain 在链路里的位置：`ChatPromptTemplate | ChatOpenAI | StrOutputParser`（LCEL）。
选 LCEL 而不是老的 `load_qa_chain`，因为：
  - 流式是天然能力，不需要额外包装；
  - 组件可替换，换模型/换输出解析器只改一个管道节点；
  - 不依赖 langchain 主包那堆历史包袱，只装 langchain-core + langchain-openai 即可。
没装 LangChain 时自动走等价的原生实现，行为保持一致（同一份 Prompt、同一套消息序列）。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.rag import prompts
from app.rag.llm import ChatModel, demo_answer_stream
from app.rag.text import make_snippet

MAX_HISTORY_MESSAGES = 6


@dataclass(slots=True)
class Citation:
    index: int
    doc_id: int
    filename: str
    ordinal: int
    snippet: str
    score: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "chunk_ordinal": self.ordinal,
            "snippet": self.snippet,
            "score": self.score,
        }


def build_citations(hits) -> list[Citation]:
    return [
        Citation(
            index=index,
            doc_id=hit.doc_id,
            filename=hit.filename,
            ordinal=hit.ordinal,
            snippet=make_snippet(hit.text),
            score=hit.score,
        )
        for index, hit in enumerate(hits, start=1)
    ]


def build_messages(question: str, hits, history: list[dict[str, str]] | None = None) -> list[dict]:
    """原生链路用的消息序列；LangChain 链路复用同一份文本，保证两条路径输出可比。"""
    messages: list[dict] = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]
    for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
        if item.get("role") in ("user", "assistant") and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append(
        {
            "role": "user",
            "content": prompts.USER_TEMPLATE.format(
                context=prompts.format_context(hits), question=question
            ),
        }
    )
    return messages


class AnswerPipeline:
    def __init__(self, settings, chat_model: ChatModel | None) -> None:
        self.settings = settings
        self.chat_model = chat_model
        self._chain = self._build_lcel_chain()

    def _build_lcel_chain(self):
        """只有 LangChain 链路且装了 langchain-core 时才组装 LCEL 管道。"""
        if self.chat_model is None or self.chat_model.name != "langchain":
            return None
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        except ImportError:  # pragma: no cover - 依赖缺失时退回原生链路
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.SYSTEM_PROMPT),
                MessagesPlaceholder("history"),
                ("human", prompts.USER_TEMPLATE),
            ]
        )
        return prompt | self.chat_model.chat_model | StrOutputParser()

    @property
    def mode(self) -> str:
        if self.chat_model is None:
            return "demo"
        return self.chat_model.name

    async def stream(
        self,
        *,
        question: str,
        hits,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        started = time.perf_counter()

        if self.chat_model is None:
            async for piece in demo_answer_stream(question, hits):
                yield piece
            self.last_latency_ms = int((time.perf_counter() - started) * 1000)
            return

        if self._chain is not None:
            history_messages = []
            for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
                if item.get("role") in ("user", "assistant") and item.get("content"):
                    history_messages.append((item["role"], item["content"]))
            inputs = {
                "context": prompts.format_context(hits),
                "question": question,
                "history": history_messages,
            }
            async for delta in self._chain.astream(inputs):
                if delta:
                    yield delta
            self.last_latency_ms = int((time.perf_counter() - started) * 1000)
            return

        messages = build_messages(question, hits, history)
        async for delta in self.chat_model.astream(messages):
            yield delta
        self.last_latency_ms = int((time.perf_counter() - started) * 1000)
