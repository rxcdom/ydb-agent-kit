"""Application settings, read once at startup and validated as a whole."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

from src.shared.infrastructure.database.ydb.settings import YDBSettings

DEFAULT_LLM_MODEL_NAME = "gpt-oss-120b"
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_AGENT_MAX_ITERATIONS = 5
DEFAULT_AGENT_HISTORY_LIMIT = 20
DEFAULT_AGENT_TIMEZONE = "UTC"
DEFAULT_MIGRATION_APPLIED_BY = "local"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_APP_PORT = 8000


@dataclass(frozen=True)
class Settings:
    ydb: YDBSettings
    yc_folder_id: str
    yc_api_key: str
    yc_iam_token: str
    llm_model_name: str
    llm_temperature: float
    agent_max_iterations: int
    agent_history_limit: int
    agent_timezone: str
    migration_applied_by: str
    log_level: str
    app_port: int

    def __post_init__(self) -> None:
        if not self.llm_model_name:
            raise ValueError("LLM_MODEL_NAME cannot be empty")
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if self.agent_max_iterations < 1:
            raise ValueError("AGENT_MAX_ITERATIONS must be at least 1")
        if self.agent_history_limit < 1:
            raise ValueError("AGENT_HISTORY_LIMIT must be at least 1")
        try:
            ZoneInfo(self.agent_timezone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(
                f"AGENT_TIMEZONE must be an IANA zone name, got {self.agent_timezone!r}"
            ) from error
        if not isinstance(logging.getLevelName(self.log_level), int):
            raise ValueError(f"LOG_LEVEL is not a logging level name: {self.log_level!r}")
        if not 1 <= self.app_port <= 65535:
            raise ValueError("APP_PORT must be between 1 and 65535")

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        """Build settings from ``.env`` plus the process environment.

        Passing ``environ`` explicitly skips both sources, which keeps tests
        independent of the developer's machine.
        """
        if environ is None:
            load_dotenv()
            environ = os.environ

        def text(name: str, default: str = "") -> str:
            return environ.get(name, "").strip() or default

        def number(name: str, default, parse):
            raw = text(name)
            if not raw:
                return default
            try:
                return parse(raw)
            except ValueError as error:
                raise ValueError(f"{name} has an invalid value: {raw!r}") from error

        return cls(
            ydb=YDBSettings.from_env(environ),
            yc_folder_id=text("YC_FOLDER_ID"),
            yc_api_key=text("YC_API_KEY"),
            yc_iam_token=text("YC_IAM_TOKEN"),
            llm_model_name=text("LLM_MODEL_NAME", DEFAULT_LLM_MODEL_NAME),
            llm_temperature=number("LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE, float),
            agent_max_iterations=number("AGENT_MAX_ITERATIONS", DEFAULT_AGENT_MAX_ITERATIONS, int),
            agent_history_limit=number("AGENT_HISTORY_LIMIT", DEFAULT_AGENT_HISTORY_LIMIT, int),
            agent_timezone=text("AGENT_TIMEZONE", DEFAULT_AGENT_TIMEZONE),
            migration_applied_by=text("MIGRATION_APPLIED_BY", DEFAULT_MIGRATION_APPLIED_BY),
            log_level=text("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
            app_port=number("APP_PORT", DEFAULT_APP_PORT, int),
        )
