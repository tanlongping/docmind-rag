"""FastAPI 依赖注入。

所有运行时对象（引擎、索引、模型链路）都挂在 `app.state` 上，而不是模块级全局变量，
这样测试里可以并发创建多个互不影响的 app 实例（见 tests/test_api.py）。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import Settings
from app.errors import Forbidden, NotFound, Unauthorized
from app.metrics import Metrics
from app.models import KnowledgeBase, User
from app.rag.embeddings import Embedder
from app.rag.pipeline import AnswerPipeline
from app.security import AuthError, decode_token
from app.services.retrieval import IndexRegistry

bearer = HTTPBearer(auto_error=False, description="登录接口返回的 access_token")


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Iterator[Session]:
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_metrics(request: Request) -> Metrics:
    return request.app.state.metrics


def get_registry(request: Request) -> IndexRegistry:
    return request.app.state.registry


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_pipeline(request: Request) -> AnswerPipeline:
    return request.app.state.pipeline


def get_current_user(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)] = None,
) -> User:
    settings: Settings = request.app.state.settings
    if credentials is None or not credentials.credentials:
        raise Unauthorized("缺少 Bearer Token")

    try:
        payload = decode_token(credentials.credentials, secret=settings.jwt_secret)
    except AuthError as exc:
        raise Unauthorized(str(exc)) from exc

    try:
        user_id = int(payload["sub"])
    except (KeyError, ValueError) as exc:
        raise Unauthorized("token 中的用户标识非法") from exc

    user = db.get(User, user_id)
    if user is None:
        raise Unauthorized("用户不存在或已注销")
    if user.tenant_id != payload.get("tid"):
        raise Unauthorized("token 与租户不匹配")
    request.state.user_id = user.id
    request.state.tenant_id = user.tenant_id
    return user


def get_kb(
    kb_id: int,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
) -> KnowledgeBase:
    """按租户取知识库。

越权时返回 404 而不是 403：不泄露「这个 id 存在但不属于你」这一信息。
"""
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise NotFound("知识库不存在")
    if kb.tenant_id != user.tenant_id:
        raise NotFound("知识库不存在")
    return kb


def require_owner(user: Annotated[User, Depends(get_current_user)]) -> User:
    if user.role != "owner":
        raise Forbidden("需要管理员权限")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
DbSession = Annotated[Session, Depends(get_db)]
CurrentKB = Annotated[KnowledgeBase, Depends(get_kb)]
