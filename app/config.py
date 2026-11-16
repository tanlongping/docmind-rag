"""配置加载。

刻意不用 pydantic-settings / python-dotenv：本项目的部署目标是「一条 pip install 就能跑」，
少一个依赖就少一类环境问题。这里用 40 行代码实现 env + .env 的优先级合并，
优先级：显式传入 > 进程环境变量 > .env 文件 > 默认值。
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, field_validator

ENV_PREFIX = "DOCMIND_"


def _read_dotenv(path: Path) -> dict[str, str]:
    """极简 .env 解析：KEY=VALUE、# 注释、去掉成对引号。"""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _env_value(key: str) -> str | None:
    raw = os.getenv(ENV_PREFIX + key)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


class Settings(BaseModel):
    """运行期配置。字段名转大写后加 DOCMIND_ 前缀即为环境变量名。"""

    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    database_url: str = "sqlite:///./docmind.db"
    data_dir: Path = Path("./data")

    jwt_secret: str = "dev-only-change-me-please-32bytes-min"
    jwt_ttl_seconds: int = 7200

    llm_provider: str = "demo"  # demo | openai
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 60.0

    embedding_provider: str = "hashing"  # hashing | openai
    embedding_dim: int = 384

    chunk_size: int = 420
    chunk_overlap: int = 80
    top_k: int = 5
    retrieval_vector_weight: float = 0.35
    max_upload_bytes: int = 8 * 1024 * 1024

    rate_limit_per_minute: int = 120
    retrieval_mode: str = "hybrid"  # hybrid | vector | keyword

    request_timeout_seconds: float = 30.0

    @field_validator("chunk_overlap")
    @classmethod
    def _check_overlap(cls, value: int, info) -> int:
        size = info.data.get("chunk_size", 420)
        if value >= size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return value

    @field_validator("embedding_dim")
    @classmethod
    def _check_dim(cls, value: int) -> int:
        if value < 32 or value > 4096:
            raise ValueError("embedding_dim 需在 32~4096 之间")
        return value

    @classmethod
    def load(cls, dotenv: str | Path | None = ".env", **overrides) -> Settings:
        """按 默认值 < .env < 环境变量 < 显式参数 的顺序构建配置。"""
        payload: dict[str, object] = {}
        if dotenv:
            for key, value in _read_dotenv(Path(dotenv)).items():
                if key.startswith(ENV_PREFIX):
                    payload[key[len(ENV_PREFIX):].lower()] = value
        for field_name in cls.model_fields:
            raw = _env_value(field_name.upper())
            if raw is not None:
                payload[field_name] = raw
        payload.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**payload)

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"prod", "production"}

    @property
    def sqlite_file(self) -> Path | None:
        if not self.database_url.startswith("sqlite"):
            return None
        _, _, path = self.database_url.partition("///")
        if not path or path == ":memory:":
            return None
        return Path(path)


# 依赖注入时用它取「当前 app 的配置」，避免全局可变状态。
DEFAULT_SETTINGS = Settings()
