"""测试辅助：构造隔离的 app 实例与常用请求封装。"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

TEST_SECRET = "unit-test-secret-key-please-do-not-use-in-prod"


def make_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "env": "test",
        "database_url": "sqlite://",
        "jwt_secret": TEST_SECRET,
        "llm_provider": "demo",
        "embedding_provider": "hashing",
        "embedding_dim": 128,
        "chunk_size": 120,
        "chunk_overlap": 20,
        "rate_limit_per_minute": 10_000,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def make_client(**overrides: Any) -> TestClient:
    return TestClient(create_app(make_settings(**overrides)))


def register(client: TestClient, email: str = "alice@example.com", password: str = "Passw0rd!") -> str:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "display_name": "Alice"},
    )
    assert response.status_code == 201, response.text
    return response.json()["access_token"]


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def create_kb(client: TestClient, token: str, name: str = "产品手册") -> int:
    response = client.post(
        "/api/v1/kbs", json={"name": name, "description": "测试用知识库"}, headers=auth(token)
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def upload(
    client: TestClient,
    token: str,
    kb_id: int,
    text: str,
    filename: str = "faq.txt",
) -> dict:
    response = client.post(
        f"/api/v1/kbs/{kb_id}/documents",
        headers=auth(token),
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event:
            events.append((event, data or {}))
    return events


SAMPLE_FAQ = """退款政策说明

本店支持 7 天无理由退款。用户签收商品后 7 个自然日内，商品未拆封且不影响二次销售的，可以申请全额退款。

退款到账时间：原路退回，银行卡 1-3 个工作日，支付宝与微信支付实时到账。

发票开具规则：订单完成后 30 天内可在「我的订单」页面申请电子普通发票，专用发票需联系客服提供开票资料。

配送范围：全国大部分地区支持次日达，新疆、西藏等偏远地区需要 3-5 天。

客服工作时间：每天 9:00-21:00，节假日不休。
"""
