"""写入演示数据：一个知识库 + 4 篇文档 + 一轮问答。

用法：
    python scripts/seed_demo.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"

DEMO_EMAIL = "demo@docmind.dev"
DEMO_PASSWORD = "Demo12345!"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        register = client.post(
            "/api/v1/auth/register",
            json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD, "display_name": "Demo"},
        )
        if register.status_code == 409:
            login = client.post(
                "/api/v1/auth/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD}
            )
            login.raise_for_status()
            token = login.json()["access_token"]
        else:
            register.raise_for_status()
            token = register.json()["access_token"]

        headers = {"Authorization": f"Bearer {token}"}
        kb = client.post(
            "/api/v1/kbs",
            json={"name": "客服知识库", "description": "演示用：售后 / 发票 / 物流 / 开放平台"},
            headers=headers,
        )
        if kb.status_code == 409:
            kb_id = client.get("/api/v1/kbs", headers=headers).json()[0]["id"]
        else:
            kb.raise_for_status()
            kb_id = kb.json()["id"]

        for path in sorted(CORPUS.glob("*.md")):
            with path.open("rb") as fh:
                response = client.post(
                    f"/api/v1/kbs/{kb_id}/documents",
                    headers=headers,
                    files={"file": (path.name, fh, "text/markdown")},
                )
            print(f"上传 {path.name}: {response.status_code}")

        answer = client.post(
            f"/api/v1/kbs/{kb_id}/chat",
            json={"question": "退款多久能到账？", "stream": False},
            headers=headers,
        )
        answer.raise_for_status()
        payload = answer.json()
        print("\n问答结果：")
        print(payload["answer"])
        print("\n引用：")
        print(json.dumps(payload["citations"], ensure_ascii=False, indent=2))
        print(f"\n知识库 ID = {kb_id}，登录账号 {DEMO_EMAIL} / {DEMO_PASSWORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
