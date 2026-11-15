"""数据库层：SQLAlchemy 2.0 声明式模型 + 会话工厂。

选同步 Session 而不是 async：FastAPI 的 `def` 端点会自动丢到线程池执行，
在「中等并发 + SQLite/MySQL」这个量级下比 async 驱动更稳、生态更全；
真正的异步只留给 LLM 流式输出那条链路（见 api/chat.py）。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import DateTime, Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class UTCDateTime(DateTime):
    """统一按 UTC 存储、按 naive datetime 取出，避免各数据库时区行为不一致。"""

    def process_bind_param(self, value, dialect):  # noqa: D102
        if value is None:
            return None
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):  # noqa: D102
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc)


def build_engine(database_url: str, *, echo: bool = False) -> Engine:
    kwargs: dict[str, object] = {"echo": echo, "future": True, "pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        # 内存库必须用 StaticPool，否则每个线程拿到的是不同的库（表会「消失」）
        is_memory = database_url in {"sqlite://", "sqlite:///", "sqlite:///:memory:"} or (
            ":memory:" in database_url
        )
        if is_memory:
            kwargs["connect_args"] = {"check_same_thread": False}
            kwargs["poolclass"] = StaticPool
        else:
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 15}
    engine = create_engine(database_url, **kwargs)

    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragma(dbapi_conn, _record):  # pragma: no cover - 依赖驱动
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
