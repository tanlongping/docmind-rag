"""并发压测：测量检索接口与问答接口的吞吐和尾延迟。

只用 httpx + asyncio，不引入 locust/wrk，方便在任何机器上复现。
    python scripts/bench_api.py --base-url http://127.0.0.1:8000 --concurrency 20 --requests 400
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

DEMO_EMAIL = "bench@docmind.dev"
DEMO_PASSWORD = "Bench12345!"


async def ensure_account(client: httpx.AsyncClient) -> tuple[str, int]:
    register = await client.post(
        "/api/v1/auth/register",
        json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD, "display_name": "Bench"},
    )
    if register.status_code == 409:
        login = await client.post(
            "/api/v1/auth/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD}
        )
        login.raise_for_status()
        token = login.json()["access_token"]
    else:
        register.raise_for_status()
        token = register.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    kbs = (await client.get("/api/v1/kbs", headers=headers)).json()
    if kbs:
        kb_id = kbs[0]["id"]
    else:
        created = await client.post(
            "/api/v1/kbs", json={"name": "压测库"}, headers=headers
        )
        created.raise_for_status()
        kb_id = created.json()["id"]
    return token, kb_id


async def run(concurrency: int, total: int, base_url: str, path: str, payload: dict) -> None:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=30, limits=limits) as client:
        token, kb_id = await ensure_account(client)
        headers = {"Authorization": f"Bearer {token}"}
        url = path.format(kb_id=kb_id)

        latencies: list[float] = []
        errors = 0
        semaphore = asyncio.Semaphore(concurrency)

        async def one() -> None:
            nonlocal errors
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post(url, json=payload, headers=headers)
                    if response.status_code >= 400:
                        errors += 1
                except httpx.HTTPError:
                    errors += 1
                finally:
                    latencies.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(total)))
        elapsed = time.perf_counter() - started

        latencies.sort()
        pct = lambda p: latencies[min(len(latencies) - 1, int(len(latencies) * p))]  # noqa: E731
        print(f"目标: {url}   并发 {concurrency}   请求数 {total}")
        print(f"总耗时 {elapsed:.2f}s   吞吐 {total / elapsed:.1f} req/s   错误 {errors}")
        print(
            f"P50 {pct(0.50):.1f}ms   P90 {pct(0.90):.1f}ms   "
            f"P95 {pct(0.95):.1f}ms   P99 {pct(0.99):.1f}ms   "
            f"均值 {statistics.mean(latencies):.1f}ms"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--path", default="/api/v1/kbs/{kb_id}/search")
    parser.add_argument("--question", default="退款多久到账")
    args = parser.parse_args()

    payload = (
        {"query": args.question}
        if args.path.endswith("/search")
        else {"question": args.question, "stream": False}
    )
    asyncio.run(run(args.concurrency, args.requests, args.base_url, args.path, payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
