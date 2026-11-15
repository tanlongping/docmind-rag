"""口令散列与 JWT 的单元测试。"""

from __future__ import annotations

import unittest

from app.security import (
    AuthError,
    create_access_token,
    decode_token,
    hash_password,
    new_tenant_id,
    verify_password,
)

SECRET = "unit-test-secret-key-please-do-not-use-in-prod"


class PasswordTests(unittest.TestCase):
    def test_hash_is_salted_and_verifies(self) -> None:
        first = hash_password("Passw0rd!")
        second = hash_password("Passw0rd!")
        self.assertNotEqual(first, second, "相同口令必须产生不同散列（随机 salt）")
        self.assertTrue(verify_password("Passw0rd!", first))
        self.assertFalse(verify_password("Passw0rd?", first))

    def test_short_password_rejected(self) -> None:
        with self.assertRaises(ValueError):
            hash_password("1234567")

    def test_malformed_hash_returns_false(self) -> None:
        self.assertFalse(verify_password("whatever", "not-a-hash"))
        self.assertFalse(verify_password("whatever", "md5$1$aa$bb"))


class TokenTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        token = create_access_token("42", secret=SECRET, tenant_id="t_abc", role="owner")
        payload = decode_token(token, secret=SECRET)
        self.assertEqual(payload["sub"], "42")
        self.assertEqual(payload["tid"], "t_abc")
        self.assertEqual(payload["role"], "owner")

    def test_expired_token_rejected(self) -> None:
        token = create_access_token("42", secret=SECRET, tenant_id="t_abc", ttl_seconds=10, now=1000)
        with self.assertRaises(AuthError):
            decode_token(token, secret=SECRET, now=2000)

    def test_wrong_secret_rejected(self) -> None:
        token = create_access_token("42", secret=SECRET, tenant_id="t_abc")
        with self.assertRaises(AuthError):
            decode_token(token, secret="another-secret")

    def test_tampered_payload_rejected(self) -> None:
        token = create_access_token("42", secret=SECRET, tenant_id="t_abc")
        header, payload, signature = token.split(".")
        forged = f"{header}.{payload[:-2]}xx.{signature}"
        with self.assertRaises(AuthError):
            decode_token(forged, secret=SECRET)

    def test_garbage_rejected(self) -> None:
        with self.assertRaises(AuthError):
            decode_token("only-one-segment", secret=SECRET)

    def test_tenant_id_format(self) -> None:
        self.assertTrue(new_tenant_id().startswith("t_"))
        self.assertNotEqual(new_tenant_id(), new_tenant_id())


if __name__ == "__main__":
    unittest.main()
