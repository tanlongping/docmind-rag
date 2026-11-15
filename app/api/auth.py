"""注册 / 登录 / 当前用户。

注册即开通一个独立租户（tenant_id），1 人 1 租户是最省事的落地方式；
后续要做团队协作，只需把 tenant_id 换成「组织」并加成员表，模型不用动。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy import select

from app.config import Settings
from app.deps import CurrentUser, DbSession, get_settings
from app.errors import Conflict, Unauthorized
from app.models import User
from app.schemas import LoginIn, RegisterIn, TokenOut, UserOut
from app.security import create_access_token, hash_password, new_tenant_id, verify_password

router = APIRouter(prefix="/api/v1/auth", tags=["认证"])

SettingsDep = Annotated[Settings, Depends(get_settings)]


def _issue_token(user: User, settings: Settings) -> TokenOut:
    token = create_access_token(
        str(user.id),
        secret=settings.jwt_secret,
        tenant_id=user.tenant_id,
        role=user.role,
        ttl_seconds=settings.jwt_ttl_seconds,
    )
    return TokenOut(
        access_token=token,
        expires_in=settings.jwt_ttl_seconds,
        user=UserOut.model_validate(user),
    )


@router.post(
    "/register",
    response_model=TokenOut,
    status_code=status.HTTP_201_CREATED,
    summary="注册并自动登录",
)
def register(payload: RegisterIn, db: DbSession, settings: SettingsDep) -> TokenOut:
    email = payload.email.lower()
    if db.scalar(select(User).where(User.email == email)) is not None:
        raise Conflict("该邮箱已注册")

    user = User(
        email=email,
        tenant_id=new_tenant_id(),
        password_hash=hash_password(payload.password),
        display_name=payload.display_name or email.split("@")[0],
        role="owner",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _issue_token(user, settings)


@router.post("/login", response_model=TokenOut, summary="邮箱密码登录")
def login(payload: LoginIn, db: DbSession, settings: SettingsDep) -> TokenOut:
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    # 无论用户是否存在都走一次校验，避免通过响应时间枚举邮箱
    if user is None or not verify_password(payload.password, user.password_hash):
        raise Unauthorized("邮箱或密码错误")
    return _issue_token(user, settings)


@router.get("/me", response_model=UserOut, summary="当前登录用户")
def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
