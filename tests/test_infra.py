"""限流与指标的单元测试。"""

from __future__ import annotations

import time
import unittest

from app.metrics import Metrics
from app.rate_limit import SlidingWindowLimiter


class RateLimiterTests(unittest.TestCase):
    def test_allows_until_limit(self) -> None:
        limiter = SlidingWindowLimiter(3, 60)
        results = [limiter.allow("k")[0] for _ in range(4)]
        self.assertEqual(results, [True, True, True, False])

    def test_remaining_quota(self) -> None:
        limiter = SlidingWindowLimiter(5, 60)
        allowed, remaining = limiter.allow("k")
        self.assertTrue(allowed)
        self.assertEqual(remaining, 4)

    def test_keys_are_isolated(self) -> None:
        limiter = SlidingWindowLimiter(1, 60)
        self.assertTrue(limiter.allow("a")[0])
        self.assertTrue(limiter.allow("b")[0])
        self.assertFalse(limiter.allow("a")[0])

    def test_window_slides(self) -> None:
        limiter = SlidingWindowLimiter(2, 1)
        self.assertTrue(limiter.allow("k")[0])
        self.assertTrue(limiter.allow("k")[0])
        self.assertFalse(limiter.allow("k")[0])
        time.sleep(1.05)
        self.assertTrue(limiter.allow("k")[0], "窗口滑过后配额应恢复")

    def test_reset(self) -> None:
        limiter = SlidingWindowLimiter(1, 60)
        limiter.allow("k")
        self.assertFalse(limiter.allow("k")[0])
        limiter.reset("k")
        self.assertTrue(limiter.allow("k")[0])

    def test_invalid_limit(self) -> None:
        with self.assertRaises(ValueError):
            SlidingWindowLimiter(0)


class MetricsTests(unittest.TestCase):
    def test_counter_and_gauge_render(self) -> None:
        metrics = Metrics()
        metrics.describe("demo_requests_total", "counter", "请求数")
        metrics.inc("demo_requests_total", {"path": "/a"})
        metrics.inc("demo_requests_total", {"path": "/a"})
        metrics.set_gauge("demo_inflight", 3)

        text = metrics.render()
        self.assertIn("# TYPE demo_requests_total counter", text)
        self.assertIn('demo_requests_total{path="/a"} 2', text)
        self.assertIn("demo_inflight 3", text)

    def test_histogram_buckets(self) -> None:
        metrics = Metrics()
        metrics.describe("demo_latency", "histogram", "耗时")
        for value in (0.003, 0.02, 0.3):
            metrics.observe("demo_latency", value)

        text = metrics.render()
        self.assertIn("# TYPE demo_latency_seconds histogram", text)
        self.assertIn('demo_latency_seconds_bucket{le="0.005"} 1', text)
        self.assertIn('demo_latency_seconds_bucket{le="+Inf"} 3', text)
        self.assertIn("demo_latency_seconds_count 3", text)

    def test_counter_value(self) -> None:
        metrics = Metrics()
        metrics.inc("x_total")
        self.assertEqual(metrics.counter_value("x_total"), 1.0)
        self.assertEqual(metrics.histogram_stats("y")[0], 0)


if __name__ == "__main__":
    unittest.main()
