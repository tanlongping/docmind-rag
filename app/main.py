"""应用装配。

所有资源在 `create_app()` 里构建并挂到 `app.state`，lifespan 只负责释放；
这样不带 lifespan 的测试客户端（`TestClient(app)`）也能完整工作，
测试与生产跑的是同一条装配路径，减少「测试过了线上炸」的概率。
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.api import auth, chat, documents, kb, system
from app.config import Settings
from app.db import Base, build_engine, build_session_factory
from app.errors import register_exception_handlers
from app.metrics import Metrics
from app.rag.embeddings import build_embedder
from app.rag.llm import build_chat_model
from app.rag.pipeline import AnswerPipeline
from app.rate_limit import SlidingWindowLimiter
from app.security import AuthError, decode_token
from app.services.retrieval import IndexRegistry

logger = logging.getLogger("docmind")
STATIC_DIR = Path(__file__).parent / "static"

OPENAPI_TAGS = [
    {"name": "认证", "description": "注册、登录、令牌校验"},
    {"name": "知识库", "description": "租户内的知识库管理"},
    {"name": "文档与检索", "description": "文档入库、切片、混合检索"},
    {"name": "问答", "description": "带引用的流式问答"},
    {"name": "系统", "description": "健康检查与指标"},
]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """请求 id + 访问日志 + HTTP 指标。"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        metrics: Metrics = request.app.state.metrics

        try:
            response = await call_next(request)
        except Exception:
            metrics.inc(
                "docmind_http_requests_total",
                {"method": request.method, "status": "500", "path": "unhandled"},
            )
            raise

        elapsed = time.perf_counter() - started
        route = getattr(request.scope.get("route"), "path", None) or request.url.path
        metrics.inc(
            "docmind_http_requests_total",
            {"method": request.method, "status": str(response.status_code), "path": route},
        )
        metrics.observe("docmind_http_request_duration", elapsed)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s %s %.1fms rid=%s",
            request.method,
            route,
            response.status_code,
            elapsed * 1000,
            request_id,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """按「用户 > IP」维度限流，只保护 /api 前缀。"""

    EXEMPT_PATHS = {"/healthz", "/readyz", "/metrics"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS" or path in self.EXEMPT_PATHS or not path.startswith("/api"):
            return await call_next(request)

        limiter: SlidingWindowLimiter = request.app.state.limiter
        key = self._identify(request)
        allowed, remaining = limiter.allow(key)
        if not allowed:
            request.app.state.metrics.inc("docmind_http_rate_limited_total")
            return JSONResponse(
                {
                    "error": {"code": "rate_limited", "message": "请求过于频繁，请稍后重试"},
                    "request_id": getattr(request.state, "request_id", "-"),
                },
                status_code=429,
                headers={"Retry-After": "30", "X-RateLimit-Remaining": "0"},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    @staticmethod
    def _identify(request: Request) -> str:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:].strip()
            try:
                payload = decode_token(token, secret=request.app.state.settings.jwt_secret)
                return f"user:{payload['sub']}"
            except AuthError:
                pass
        client = request.client.host if request.client else "unknown"
        return f"ip:{client}"


def _configure_logging(settings: Settings) -> None:
    if settings.env == "test":  # 测试期间只保留告警，避免日志淹没断言失败信息
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s", force=True)
        for noisy in ("httpx", "httpcore", "asyncio", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        return
    logging.basicConfig(
        level=logging.DEBUG if settings.env == "development" else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    _configure_logging(settings)

    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    session_factory = build_session_factory(engine)

    embedder = build_embedder(settings)
    try:
        chat_model = build_chat_model(settings)
    except RuntimeError as exc:  # 配置缺失时降级，不让整个服务起不来
        logger.warning("大模型链路初始化失败，降级为 demo 模式：%s", exc)
        chat_model = None

    pipeline = AnswerPipeline(settings, chat_model)
    registry = IndexRegistry(session_factory, vector_weight=settings.retrieval_vector_weight)
    metrics = Metrics()
    for name, help_text in (
        ("docmind_http_requests_total", "HTTP 请求总数"),
        ("docmind_http_rate_limited_total", "被限流的请求数"),
        ("docmind_documents_ingested_total", "入库文档数"),
        ("docmind_chunks_ingested_total", "入库切片数"),
        ("docmind_chat_requests_total", "问答请求数"),
        ("docmind_chat_completions_total", "完成的问答数"),
        ("docmind_chat_errors_total", "生成失败的问答数"),
        ("docmind_chat_chars_total", "生成的字符总数"),
    ):
        metrics.describe(name, "counter", help_text)
    metrics.describe("docmind_http_request_duration", "histogram", "HTTP 请求耗时")
    metrics.describe("docmind_retrieval_duration", "histogram", "检索耗时")
    metrics.describe("docmind_chat_duration", "histogram", "端到端问答耗时")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info(
            "DocMind 启动 env=%s db=%s llm=%s embedding=%s",
            settings.env,
            settings.database_url.split("///")[-1],
            pipeline.mode,
            settings.embedding_provider,
        )
        try:
            yield
        finally:
            engine.dispose()
            logger.info("DocMind 已停止，连接池已释放")

    app = FastAPI(
        title="DocMind API",
        description="多租户知识库问答服务：混合检索（BM25 + 向量 + RRF）、引用溯源、SSE 流式生成",
        version="1.0.0",
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.embedder = embedder
    app.state.registry = registry
    app.state.pipeline = pipeline
    app.state.metrics = metrics
    app.state.limiter = SlidingWindowLimiter(settings.rate_limit_per_minute, 60)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining"],
    )
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    app.include_router(auth.router)
    app.include_router(kb.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(system.router)

    if STATIC_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def root() -> RedirectResponse:
            return RedirectResponse(url="/ui/")

    return app
