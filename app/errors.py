"""统一错误模型与异常处理。

对外只暴露 code + message，绝不把堆栈/SQL 原文透给客户端；
同时给每个请求分配 request_id，日志里可用它反查完整上下文。
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class NotFound(AppError):
    def __init__(self, message: str = "资源不存在") -> None:
        super().__init__(status.HTTP_404_NOT_FOUND, "not_found", message)


class Forbidden(AppError):
    def __init__(self, message: str = "无权访问该资源") -> None:
        super().__init__(status.HTTP_403_FORBIDDEN, "forbidden", message)


class Conflict(AppError):
    def __init__(self, message: str = "资源冲突") -> None:
        super().__init__(status.HTTP_409_CONFLICT, "conflict", message)


class BadRequest(AppError):
    def __init__(self, message: str = "请求参数不合法") -> None:
        super().__init__(status.HTTP_400_BAD_REQUEST, "bad_request", message)


class Unauthorized(AppError):
    def __init__(self, message: str = "未认证或凭证已失效") -> None:
        super().__init__(status.HTTP_401_UNAUTHORIZED, "unauthorized", message)


class RateLimited(AppError):
    def __init__(self, message: str = "请求过于频繁") -> None:
        super().__init__(status.HTTP_429_TOO_MANY_REQUESTS, "rate_limited", message)


def _body(code: str, message: str, request_id: str, details=None) -> dict:
    payload = {"error": {"code": code, "message": message}, "request_id": request_id}
    if details is not None:
        payload["error"]["details"] = details
    return payload


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        rid = getattr(request.state, "request_id", "-")
        headers = {"X-Request-ID": rid}
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse(
            _body(exc.code, exc.message, rid), status_code=exc.status_code, headers=headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        rid = getattr(request.state, "request_id", "-")
        details = [
            {"loc": list(err.get("loc", ())), "msg": err.get("msg", "")} for err in exc.errors()
        ]
        return JSONResponse(
            _body("validation_error", "请求参数校验失败", rid, details), status_code=422
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        rid = getattr(request.state, "request_id", "-")
        return JSONResponse(
            _body("http_error", str(exc.detail), rid), status_code=exc.status_code
        )
