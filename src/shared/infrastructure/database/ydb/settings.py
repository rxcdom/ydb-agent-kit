"""Connection settings of the YDB data-access layer."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

DEFAULT_ENDPOINT = "grpc://ydb:2136"
DEFAULT_DATABASE = "/local"
DEFAULT_CONNECT_ATTEMPTS = 12
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_CONNECT_RETRY_DELAY_SECONDS = 5.0
DEFAULT_POOL_SIZE = 100

_ENDPOINT_SCHEMES = ("grpc://", "grpcs://")


@dataclass(frozen=True)
class YDBSettings:
    endpoint: str = DEFAULT_ENDPOINT
    database: str = DEFAULT_DATABASE
    connect_attempts: int = DEFAULT_CONNECT_ATTEMPTS
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    connect_retry_delay_seconds: float = DEFAULT_CONNECT_RETRY_DELAY_SECONDS
    pool_size: int = DEFAULT_POOL_SIZE

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(_ENDPOINT_SCHEMES):
            raise ValueError(
                f"YDB_ENDPOINT must start with grpc:// or grpcs://, got {self.endpoint!r} "
                f"(example: {DEFAULT_ENDPOINT})"
            )
        if not self.database.startswith("/"):
            raise ValueError(
                f"YDB_DATABASE must be an absolute database path, got {self.database!r} "
                f"(example: {DEFAULT_DATABASE})"
            )
        if self.connect_attempts < 1:
            raise ValueError("YDB_CONNECT_ATTEMPTS must be at least 1")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("YDB_CONNECT_TIMEOUT_SECONDS must be positive")
        if self.connect_retry_delay_seconds < 0:
            raise ValueError("the connect retry delay cannot be negative")
        if self.pool_size < 1:
            raise ValueError("the session pool size must be at least 1")

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "YDBSettings":
        env = os.environ if environ is None else environ
        return cls(
            endpoint=env.get("YDB_ENDPOINT", "").strip() or DEFAULT_ENDPOINT,
            database=env.get("YDB_DATABASE", "").strip() or DEFAULT_DATABASE,
            connect_attempts=_read_int(env, "YDB_CONNECT_ATTEMPTS", DEFAULT_CONNECT_ATTEMPTS),
            connect_timeout_seconds=_read_float(
                env, "YDB_CONNECT_TIMEOUT_SECONDS", DEFAULT_CONNECT_TIMEOUT_SECONDS
            ),
        )


def _read_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


def _read_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw!r}") from error
