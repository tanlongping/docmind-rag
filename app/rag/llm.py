"""大模型接入层。

三条链路，按可用性自动降级：
  1. LangChain（ChatOpenAI）—— 装了 `.[llm]` 且 provider=openai 时启用，
     好处是直接复用 LangChain 的生态（回调、追踪、换模型只需改一行）；
  2. 原生 httpx —— 没装 LangChain 时用，直接消费 OpenAI 兼容接口的 SSE；
  3. Demo —— 不联网，把检索到的片段抽成要点式回答，用于离线演示和 CI 测试。

注意：Demo 不是「假回答」，它只做抽取式摘要，不生成任何资料里没有的内容，
这一点和真实模型遵守同一条约束。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx

from app.rag.prompts import NO_CONTEXT_REPLY
from app.rag.text import keyword_overlap, split_sentences


class ChatModel(Protocol):
    name: str

    async def astream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...


class OpenAICompatibleChat:
    """直接用 httpx 打 OpenAI 兼容的 /chat/completions，兼容 DeepSeek / 通义 / vLLM / Ollama。"""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    async def astream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        timeout = httpx.Timeout(self.timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "ignore")[:300]
                    raise RuntimeError(f"模型接口返回 {response.status_code}: {detail}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content


class LangChainChat:
    """LangChain ChatOpenAI 包装：把 dict 消息转成 LangChain 消息对象再流式输出。"""

    name = "langchain"

    def __init__(self, chat_model: Any) -> None:
        self._model = chat_model

    @property
    def chat_model(self) -> Any:
        return self._model

    async def astream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        mapping = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        lc_messages = [
            mapping.get(message["role"], HumanMessage)(content=message["content"])
            for message in messages
        ]
        async for chunk in self._model.astream(lc_messages):
            text = getattr(chunk, "content", "")
            if isinstance(text, list):  # 多模态返回时拼接文本块
                text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
            if text:
                yield str(text)


def build_chat_model(settings) -> ChatModel | None:
    """按配置返回模型链路；返回 None 表示走 Demo。"""
    if settings.llm_provider == "demo":
        return None
    if not settings.llm_api_key:
        raise RuntimeError("llm_provider=openai 需要配置 DOCMIND_LLM_API_KEY")

    try:
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
        )
        return LangChainChat(chat)
    except ImportError:
        # 没装 LangChain 也能跑：退回原生 HTTP 实现，功能一致
        return OpenAICompatibleChat(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )


async def demo_answer_stream(question: str, hits) -> AsyncIterator[str]:
    """抽取式演示回答：只挑检索片段里与问题重合度最高的句子，不生成新内容。"""
    if not hits:
        for piece in _chunks_of_text(NO_CONTEXT_REPLY):
            yield piece
        return

    scored: list[tuple[float, str, int]] = []
    for index, hit in enumerate(hits, start=1):
        for sentence in split_sentences(hit.text):
            sentence = sentence.strip()
            # 跳过标题与引导句：它们通常只有「## 退款时间」这种短语，放进答案没有信息量
            if not sentence or sentence.startswith("#") or sentence.endswith(("：", ":")):
                continue
            if len(sentence) < 12:
                continue
            score = keyword_overlap(question, sentence)
            if score > 0:
                scored.append((score, sentence, index))

    scored.sort(key=lambda item: item[0], reverse=True)
    picked, seen = [], set()
    for _, sentence, index in scored:
        key = sentence[:24]
        if key in seen:
            continue
        seen.add(key)
        picked.append((sentence, index))
        if len(picked) == 3:
            break

    if not picked:
        picked = [(split_sentences(hits[0].text)[0], 1)]

    lines = ["根据检索到的资料，可以确认以下内容：", ""]
    for sentence, index in picked:
        lines.append(f"- {sentence} [{index}]")
    lines.append("")
    lines.append("以上要点均摘自知识库原文，点击引用可查看出处。")

    for piece in _chunks_of_text("\n".join(lines)):
        yield piece


def _chunks_of_text(text: str, size: int = 12):
    for start in range(0, len(text), size):
        yield text[start : start + size]
