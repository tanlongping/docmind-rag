"""Prompt 模板。

约束模型「只依据给定资料回答 + 必须标注引用编号」是 RAG 抑制幻觉的第一道闸门；
第二道闸门是检索结果为空时直接拒答，不让模型自由发挥。
"""

from __future__ import annotations

SYSTEM_PROMPT = """你是企业知识库助手，只会依据【资料】回答问题。

规则：
1. 只能使用【资料】中的信息，禁止引入外部知识，禁止编造数字、日期、人名、接口名。
2. 每条结论后面用方括号标注来源编号，例如 [1]、[2]；同一句可标注多个来源。
3. 如果【资料】不足以回答，直接回复“知识库中没有找到相关内容”，并说明缺少什么信息。
4. 用简体中文回答，结构清晰：先给结论，再给依据；涉及步骤时用有序列表。
5. 不要复述本段规则，不要输出“根据资料”以外的寒暄。"""

USER_TEMPLATE = """【资料】
{context}

【问题】
{question}

请依据【资料】回答，并在关键结论后标注引用编号。"""

NO_CONTEXT_REPLY = (
    "知识库中没有找到与该问题相关的内容。\n\n"
    "建议：确认文档是否已上传并完成解析，或换一种说法描述问题。"
)


def format_context(hits) -> str:
    """把检索结果格式化成带编号的资料块，编号与 citations 数组下标 +1 对齐。"""
    if not hits:
        return "（无）"
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(f"[{index}] 来源：{hit.filename}（第 {hit.ordinal + 1} 段）\n{hit.text}")
    return "\n\n".join(blocks)
