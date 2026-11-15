"""文本切片与分词的单元测试。"""

from __future__ import annotations

import unittest

from app.rag.text import keyword_overlap, normalize, split_text, tokenize


class NormalizeTests(unittest.TestCase):
    def test_fullwidth_and_whitespace(self) -> None:
        self.assertEqual(normalize("ＡＢＣ　１２３"), "ABC 123")
        self.assertEqual(normalize("a\r\n\r\n\r\nb"), "a\n\nb")
        self.assertEqual(normalize("   "), "")


class TokenizeTests(unittest.TestCase):
    def test_chinese_bigram(self) -> None:
        tokens = tokenize("退款政策")
        self.assertEqual(tokens, ["退款", "款政", "政策"])

    def test_single_char(self) -> None:
        self.assertEqual(tokenize("退"), ["退"])

    def test_mixed(self) -> None:
        tokens = tokenize("FastAPI 接口 v2.0")
        self.assertIn("fastapi", tokens)
        self.assertIn("v2.0", tokens)

    def test_keyword_overlap(self) -> None:
        self.assertGreater(keyword_overlap("退款", "支持七天无理由退款"), 0.5)
        self.assertEqual(keyword_overlap("发票", "支持七天无理由退款"), 0.0)


class SplitTests(unittest.TestCase):
    def test_respects_size(self) -> None:
        text = "退款政策说明。" * 80
        chunks = split_text(text, size=100, overlap=10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks), [len(c) for c in chunks])

    def test_covers_content_without_loss(self) -> None:
        text = "第一条：支持七天无理由退款。第二条：发票三十天内开具。第三条：偏远地区三天送达。"
        chunks = split_text(text, size=200, overlap=0)
        self.assertEqual(len(chunks), 1)
        self.assertIn("七天无理由退款", chunks[0])

    def test_empty(self) -> None:
        self.assertEqual(split_text("   \n\n  "), [])

    def test_invalid_params(self) -> None:
        with self.assertRaises(ValueError):
            split_text("abc", size=0)
        with self.assertRaises(ValueError):
            split_text("abc", size=10, overlap=10)

    def test_overlap_between_chunks(self) -> None:
        sentences = [f"第{i}条说明内容比较长一些用于测试重叠逻辑。" for i in range(1, 12)]
        chunks = split_text("".join(sentences), size=60, overlap=20)
        self.assertGreater(len(chunks), 2)
        # 相邻块之间应存在公共片段（重叠），且不重复整块
        self.assertTrue(any(chunks[i][-10:] in chunks[i + 1] for i in range(len(chunks) - 1)))


if __name__ == "__main__":
    unittest.main()
