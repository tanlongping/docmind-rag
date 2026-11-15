"""鉴权：PBKDF2 口令散列 + 自实现 HS256 JWT。

为什么不直接上 passlib / python-jose：
  1. 这两个库的维护状态和 CVE 历史都不算让人放心；
  2. 需求只有「签发/校验一个 HS256 token」，标准库 60 行就够，还少两个依赖。
算法实现遵循 RFC 7519：header.payload.signature，base64url 无填充，signature 覆盖前两段。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

PBKDF2_ITERATIONS = 210_000  # OWASP 2023 建议值下限
_HASH_NAME = "sha256"


class AuthError(Exception):
    """token 缺失 / 过期 / 签名不匹配。"""


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    """输出 pbkdf2_sha256$迭代次数$salt$digest，salt 每次随机 16 字节。"""
    if len(password) < 8:
        raise ValueError("密码至少 8 位")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(_HASH_NAME, password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_{_HASH_NAME}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != f"pbkdf2_{_HASH_NAME}":
            return False
        digest = hashlib.pbkdf2_hmac(
            _HASH_NAME, password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        # 定长比较，避免时序侧信道
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def _sign(message: bytes, secret: str) -> str:
    return _b64e(hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest())


def create_access_token(
    subject: str,
    *,
    secret: str,
    tenant_id: str,
    role: str = "member",
    ttl_seconds: int = 7200,
    now: int | None = None,
) -> str:
    issued_at = int(now if now is not None else time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "tid": tenant_id,
        "role": role,
        "iat": issued_at,
        "exp": issued_at + ttl_seconds,
    }
    segments = [
        _b64e(json.dumps(header, separators=(",", ":")).encode("utf-8")),
        _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
    ]
    signing_input = ".".join(segments).encode("ascii")
    segments.append(_sign(signing_input, secret))
    return ".".join(segments)


def decode_token(token: str, *, secret: str, leeway: int = 30, now: int | None = None) -> dict[str, Any]:
    try:
        header_b64, payload_b64, signature = token.split(".")
    except ValueError as exc:
        raise AuthError("token 格式错误") from exc

    expected = _sign(f"{header_b64}.{payload_b64}".encode("ascii"), secret)
    if not hmac.compare_digest(expected, signature):
        raise AuthError("token 签名校验失败")

    try:
        header = json.loads(_b64d(header_b64))
        payload = json.loads(_b64d(payload_b64))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthError("token 内容无法解析") from exc

    if header.get("alg") != "HS256":
        raise AuthError("不支持的签名算法")
    if int(payload.get("exp", 0)) + leeway < int(now if now is not None else time.time()):
        raise AuthError("token 已过期")
    if not payload.get("sub"):
        raise AuthError("token 缺少 sub")
    return payload


def new_tenant_id() -> str:
    return f"t_{secrets.token_hex(8)}"
