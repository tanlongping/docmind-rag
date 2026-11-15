"""API 端到端测试：认证 → 建库 → 入库 → 检索 → 问答 → 租户隔离。"""

from __future__ import annotations

import unittest

from tests.helpers import SAMPLE_FAQ, auth, create_kb, make_client, parse_sse, register, upload


class AuthApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = make_client()
        self.addCleanup(self.client.close)  # 触发 lifespan 关闭，释放 SQLite 连接

    def test_register_login_me(self) -> None:
        token = register(self.client)
        me = self.client.get("/api/v1/auth/me", headers=auth(token))
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["email"], "alice@example.com")

        login = self.client.post(
            "/api/v1/auth/login", json={"email": "alice@example.com", "password": "Passw0rd!"}
        )
        self.assertEqual(login.status_code, 200)
        self.assertTrue(login.json()["access_token"])

    def test_duplicate_email_conflict(self) -> None:
        register(self.client)
        again = self.client.post(
            "/api/v1/auth/register",
            json={"email": "alice@example.com", "password": "Passw0rd!"},
        )
        self.assertEqual(again.status_code, 409)

    def test_wrong_password(self) -> None:
        register(self.client)
        bad = self.client.post(
            "/api/v1/auth/login", json={"email": "alice@example.com", "password": "wrong-pass"}
        )
        self.assertEqual(bad.status_code, 401)

    def test_missing_token(self) -> None:
        self.assertEqual(self.client.get("/api/v1/auth/me").status_code, 401)

    def test_invalid_password_length(self) -> None:
        response = self.client.post(
            "/api/v1/auth/register", json={"email": "b@example.com", "password": "123"}
        )
        self.assertEqual(response.status_code, 422)


class KnowledgeBaseApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = make_client()
        self.addCleanup(self.client.close)  # 触发 lifespan 关闭，释放 SQLite 连接
        self.token = register(self.client)

    def test_create_and_list(self) -> None:
        kb_id = create_kb(self.client, self.token)
        listed = self.client.get("/api/v1/kbs", headers=auth(self.token)).json()
        self.assertEqual([item["id"] for item in listed], [kb_id])

        detail = self.client.get(f"/api/v1/kbs/{kb_id}", headers=auth(self.token))
        self.assertEqual(detail.json()["document_count"], 0)

    def test_duplicate_name_conflict(self) -> None:
        create_kb(self.client, self.token)
        again = self.client.post(
            "/api/v1/kbs", json={"name": "产品手册"}, headers=auth(self.token)
        )
        self.assertEqual(again.status_code, 409)

    def test_tenant_isolation(self) -> None:
        kb_id = create_kb(self.client, self.token)
        other_token = register(self.client, email="bob@example.com")

        # 越权访问返回 404 而不是 403：不暴露资源是否存在
        self.assertEqual(
            self.client.get(f"/api/v1/kbs/{kb_id}", headers=auth(other_token)).status_code, 404
        )
        self.assertEqual(
            self.client.post(
                f"/api/v1/kbs/{kb_id}/search",
                json={"query": "退款"},
                headers=auth(other_token),
            ).status_code,
            404,
        )


class DocumentAndSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = make_client()
        self.addCleanup(self.client.close)  # 触发 lifespan 关闭，释放 SQLite 连接
        self.token = register(self.client)
        self.kb_id = create_kb(self.client, self.token)

    def test_upload_creates_chunks(self) -> None:
        document = upload(self.client, self.token, self.kb_id, SAMPLE_FAQ)
        self.assertGreater(document["chunk_count"], 1)
        self.assertEqual(document["size_bytes"], len(SAMPLE_FAQ.encode("utf-8")))

        listed = self.client.get(
            f"/api/v1/kbs/{self.kb_id}/documents", headers=auth(self.token)
        ).json()
        self.assertEqual(len(listed), 1)

    def test_duplicate_document_conflict(self) -> None:
        upload(self.client, self.token, self.kb_id, SAMPLE_FAQ)
        again = self.client.post(
            f"/api/v1/kbs/{self.kb_id}/documents",
            headers=auth(self.token),
            files={"file": ("faq.txt", SAMPLE_FAQ.encode("utf-8"), "text/plain")},
        )
        self.assertEqual(again.status_code, 409)

    def test_empty_file_rejected(self) -> None:
        response = self.client.post(
            f"/api/v1/kbs/{self.kb_id}/documents",
            headers=auth(self.token),
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_search_returns_citation_fields(self) -> None:
        upload(self.client, self.token, self.kb_id, SAMPLE_FAQ)
        response = self.client.post(
            f"/api/v1/kbs/{self.kb_id}/search",
            json={"query": "退款几天到账", "top_k": 3},
            headers=auth(self.token),
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["hits"])
        top = payload["hits"][0]
        self.assertIn("退款", top["text"] + payload["query"])
        self.assertEqual(top["filename"], "faq.txt")
        self.assertIn("score", top)
        self.assertLessEqual(payload["latency_ms"], 5000)

    def test_delete_document_clears_index(self) -> None:
        document = upload(self.client, self.token, self.kb_id, SAMPLE_FAQ)
        deleted = self.client.delete(
            f"/api/v1/kbs/{self.kb_id}/documents/{document['id']}", headers=auth(self.token)
        )
        self.assertEqual(deleted.status_code, 204)

        search = self.client.post(
            f"/api/v1/kbs/{self.kb_id}/search",
            json={"query": "退款"},
            headers=auth(self.token),
        )
        self.assertEqual(search.json()["hits"], [])

    def test_chunk_lookup(self) -> None:
        upload(self.client, self.token, self.kb_id, SAMPLE_FAQ)
        search = self.client.post(
            f"/api/v1/kbs/{self.kb_id}/search",
            json={"query": "发票"},
            headers=auth(self.token),
        ).json()
        chunk_id = search["hits"][0]["chunk_id"]
        detail = self.client.get(
            f"/api/v1/kbs/{self.kb_id}/chunks/{chunk_id}", headers=auth(self.token)
        )
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["chunk_id"], chunk_id)


class ChatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = make_client()
        self.addCleanup(self.client.close)  # 触发 lifespan 关闭，释放 SQLite 连接
        self.token = register(self.client)
        self.kb_id = create_kb(self.client, self.token)
        upload(self.client, self.token, self.kb_id, SAMPLE_FAQ)

    def test_non_stream_answer_has_citations(self) -> None:
        response = self.client.post(
            f"/api/v1/kbs/{self.kb_id}/chat",
            json={"question": "退款多久到账", "stream": False},
            headers=auth(self.token),
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["answer"])
        self.assertTrue(payload["citations"])
        self.assertIn("session_id", payload)

        messages = self.client.get(
            f"/api/v1/kbs/{self.kb_id}/sessions/{payload['session_id']}/messages",
            headers=auth(self.token),
        ).json()
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertTrue(messages[1]["content"])

    def test_stream_emits_meta_tokens_done(self) -> None:
        response = self.client.post(
            f"/api/v1/kbs/{self.kb_id}/chat",
            json={"question": "发票怎么开"},
            headers=auth(self.token),
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))

        events = parse_sse(response.text)
        names = [name for name, _ in events]
        self.assertEqual(names[0], "meta")
        self.assertEqual(names[-1], "done")
        self.assertIn("token", names)

        meta = events[0][1]
        self.assertTrue(meta["citations"])
        self.assertEqual(meta["mode"], "demo")

        answer = "".join(data["text"] for name, data in events if name == "token")
        self.assertTrue(answer.strip())

    def test_session_continuity(self) -> None:
        first = self.client.post(
            f"/api/v1/kbs/{self.kb_id}/chat",
            json={"question": "退款政策是什么", "stream": False},
            headers=auth(self.token),
        ).json()
        second = self.client.post(
            f"/api/v1/kbs/{self.kb_id}/chat",
            json={
                "question": "发票呢",
                "session_id": first["session_id"],
                "stream": False,
            },
            headers=auth(self.token),
        )
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["session_id"], first["session_id"])

        sessions = self.client.get(
            f"/api/v1/kbs/{self.kb_id}/sessions", headers=auth(self.token)
        ).json()
        self.assertEqual(len(sessions), 1)

    def test_question_validation(self) -> None:
        response = self.client.post(
            f"/api/v1/kbs/{self.kb_id}/chat",
            json={"question": "", "stream": False},
            headers=auth(self.token),
        )
        self.assertEqual(response.status_code, 422)


class SystemTests(unittest.TestCase):
    def test_health_and_metrics(self) -> None:
        client = make_client()
        self.assertEqual(client.get("/healthz").json()["status"], "ok")
        self.assertEqual(client.get("/readyz").json()["llm_mode"], "demo")

        metrics = client.get("/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("docmind_http_requests_total", metrics.text)

    def test_stats_requires_auth(self) -> None:
        client = make_client()
        self.assertEqual(client.get("/api/v1/stats").status_code, 401)
        token = register(client)
        stats = client.get("/api/v1/stats", headers=auth(token))
        self.assertEqual(stats.status_code, 200)
        self.assertEqual(stats.json()["llm_mode"], "demo")

    def test_request_id_header(self) -> None:
        client = make_client()
        response = client.get("/healthz")
        self.assertTrue(response.headers.get("X-Request-ID"))


if __name__ == "__main__":
    unittest.main()
