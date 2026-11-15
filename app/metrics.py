"""轻量指标采集：按 Prometheus 文本格式暴露。

没用 prometheus_client，原因很实际：多一个 C 扩展依赖，而对本项目只需要
counter / gauge / histogram 三种类型的一百来行实现。格式严格遵循 exposition format，
Prometheus / VictoriaMetrics 都能直接抓。
"""

from __future__ import annotations

import threading
from collections import defaultdict

DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _labels_key(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    items = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + items + "}"


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, str], float] = defaultdict(float)
        self._gauges: dict[tuple[str, str], float] = {}
        self._hist: dict[tuple[str, str], list[float]] = {}
        self._hist_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._hist_count: dict[tuple[str, str], int] = defaultdict(int)
        self._help: dict[str, str] = {}
        self._type: dict[str, str] = {}

    def describe(self, name: str, kind: str, help_text: str = "") -> None:
        self._type[name] = kind
        self._help[name] = help_text or name

    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        with self._lock:
            self._counters[(name, _labels_key(labels))] += value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._gauges[(name, _labels_key(labels))] = value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = (name, _labels_key(labels))
        with self._lock:
            self._hist.setdefault(key, []).append(value)
            self._hist_sum[key] += value
            self._hist_count[key] += 1

    def counter_value(self, name: str, labels: dict[str, str] | None = None) -> float:
        with self._lock:
            return self._counters.get((name, _labels_key(labels)), 0.0)

    def histogram_stats(self, name: str, labels: dict[str, str] | None = None) -> tuple[int, float]:
        key = (name, _labels_key(labels))
        with self._lock:
            return self._hist_count.get(key, 0), self._hist_sum.get(key, 0.0)

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            by_name: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for (name, labels), value in self._counters.items():
                by_name[name].append((labels, value))
            for name, samples in by_name.items():
                lines.append(f"# HELP {name} {self._help.get(name, name)}")
                lines.append(f"# TYPE {name} counter")
                for labels, value in sorted(samples):
                    lines.append(f"{name}{labels} {_fmt(value)}")

            for (name, labels), value in sorted(self._gauges.items()):
                if name not in by_name:
                    lines.append(f"# HELP {name} {self._help.get(name, name)}")
                    lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{labels} {_fmt(value)}")

            for (name, labels), buckets in sorted(self._hist.items()):
                lines.append(f"# HELP {name}_seconds {self._help.get(name, name)}")
                lines.append(f"# TYPE {name}_seconds histogram")
                total = self._hist_count[(name, labels)]
                sums = self._hist_sum[(name, labels)]
                cumulative = 0
                for bound in DEFAULT_BUCKETS:
                    cumulative = sum(1 for v in buckets if v <= bound)
                    lines.append(f'{name}_seconds_bucket{{le="{bound}"}} {cumulative}')
                lines.append(f'{name}_seconds_bucket{{le="+Inf"}} {total}')
                lines.append(f"{name}_seconds_sum {_fmt(sums)}")
                lines.append(f"{name}_seconds_count {total}")
        return "\n".join(lines) + "\n"


def _fmt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")
