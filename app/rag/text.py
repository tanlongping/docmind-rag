"""文本处理：归一化、分词、切片。

两个刻意的选择：
1. **分词自己写**：中文没有空格，BM25 若按空格切分等于随机命中。这里用「ASCII 词 + 中文二元组」
   的混合切分，查询与文档走同一套逻辑，召回率对短查询明显更友好。
2. **切片自己写**：LangChain 的 RecursiveCharacterTextSplitter 很通用，但本项目要保证
   中文标点（。！？；）优先断开、且不同环境下切分结果完全一致（便于回归测试与评测复现）。
"""

from __future__ import annotations

import re

_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_ASCII_WORD = re.compile(r"[a-z0-9_]+(?:[.\-][a-z0-9_]+)*")
_CJK_RUN = re.compile(f"[{_CJK}]+")
_WS = re.compile(r"[ \t\u3000]+")
_MULTI_NL = re.compile(r"\n{3,}")

# 断句优先级：段落换行 > 中文句末标点 > 英文句点 > 分号
_SENTENCE_ENDS = re.compile(r"(?<=[。！？；!?;])|(?<=[.])\s+|\n")

# 只把「全角字母/数字/空格」转成半角。刻意不用 NFKC：
# NFKC 会把中文全角标点（，！？：；（））也压成 ASCII，引用原文时看着很别扭，
# 而且会改变入库文本，等于悄悄篡改了用户文档。
_KEEP_FULLWIDTH = {ord(c) for c in "，！？：；（）［］｛｝"}
_WIDTH_TABLE = {
    code: chr(code - 0xFEE0)
    for code in range(0xFF01, 0xFF5F)
    if code not in _KEEP_FULLWIDTH
}
_WIDTH_TABLE[0x3000] = 0x20  # 表意空格


def normalize(text: str) -> str:
    """全角字母数字转半角、压缩空白，保留段落结构与中文标点。"""
    text = text.translate(_WIDTH_TABLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """ASCII 词 + 中文二元组（单字成句时退化为单字）。"""
    lowered = text.lower()
    tokens: list[str] = _ASCII_WORD.findall(lowered)
    for run in _CJK_RUN.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_ENDS.split(text) if p and p.strip()]
    return parts


def split_text(text: str, *, size: int = 420, overlap: int = 80) -> list[str]:
    """按句子聚合切片，块长不超过 size，块间保留不超过 overlap 的尾部重叠。

    做法：先把段落拆成句子，再贪心装箱；单句超长则按 size 硬切。
    重叠从上一块尾部按句子边界回退，避免把一句话劈成两半。
    """
    if size <= 0:
        raise ValueError("size 必须为正数")
    if overlap >= size:
        raise ValueError("overlap 必须小于 size")

    text = normalize(text)
    if not text:
        return []

    sentences: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            continue
        for sentence in split_sentences(paragraph):
            while len(sentence) > size:
                sentences.append(sentence[:size])
                sentence = sentence[size:]
            if sentence:
                sentences.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        extra = len(sentence) + (1 if current else 0)
        if current and current_len + extra > size:
            chunks.append("\n".join(current).strip())
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current = tail
            current_len = sum(len(s) for s in current) + max(0, len(current) - 1)
        current.append(sentence)
        current_len += extra

    if current:
        chunks.append("\n".join(current).strip())

    return [c for c in chunks if c]


def make_snippet(text: str, limit: int = 160) -> str:
    """用于引用展示的短摘要。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def keyword_overlap(query: str, text: str) -> float:
    """命中率：查询词在文本中出现的比例，用于演示模式下的置信度估计。"""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    return len(query_tokens & text_tokens) / len(query_tokens)
