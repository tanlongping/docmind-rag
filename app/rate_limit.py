"""进程内滑动窗口限流。

单实例部署下这就够了；多实例要换成 Redis 的 ZSET 或令牌桶 Lua 脚本，
接口已经抽成 `allow(key)`，替换实现不影响中间件。
窗口用「按秒分桶 + 当前窗口加权」的近似滑动窗口，内存占用 O(窗口秒数)，
比存全量时间戳（O(QPS×窗口)）省得多，误差在 1% 量级。
"""

from __future__ import annotations

import threading
import time


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: int = 60, *, buckets: int | None = None) -> None:
        if limit <= 0:
            raise ValueError("limit 必须为正")
        self.limit = limit
        self.window = window_seconds
        self.buckets = buckets or window_seconds
        self._counts: dict[str, dict[int, int]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def allow(self, key: str, cost: int = 1) -> tuple[bool, int]:
        """返回 (是否放行, 剩余配额)。超限时剩余配额为 0。"""
        now = self._now()
        slot = int(now)
        cutoff = slot - self.window

        with self._lock:
            bucket = self._counts.setdefault(key, {})
            for stale in [s for s in bucket if s <= cutoff]:
                del bucket[stale]
            used = sum(bucket.values())
            if used + cost > self.limit:
                return False, 0
            bucket[slot] = bucket.get(slot, 0) + cost
            # 防止 key 泄漏：桶空了就回收
            if not bucket:
                self._counts.pop(key, None)
            return True, self.limit - used - cost

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._counts.clear()
            else:
                self._counts.pop(key, None)

    @property
    def tracked_keys(self) -> int:
        with self._lock:
            return len(self._counts)
