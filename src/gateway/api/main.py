"""FastAPI application: lifespan, routers, exception mapping."""
from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI

from src.accounts.adapters.api import auth as accounts_auth
from src.accounts.adapters.api import router as accounts_router
from src.agent.adapters.api import router as agent_router
from src.agent.adapters.llm.model_profiles import get_model_profile
from src.gateway.api.errors import register_exception_handlers
from src.gateway.di.container import AppContainer
from src.gateway.settings import Settings
from src.tasks.adapters.api import router as tasks_router

API_PREFIX = "/api/v1"
WIRED_MODULES = [accounts_auth, accounts_router, tasks_router, agent_router]


class UnsupportedAgentModelError(RuntimeError):
    """The configured model cannot call tools, so the agent cannot work with it."""


def ensure_model_can_call_tools(model_name: str) -> None:
    """Refuse to start with a model the agent cannot drive.

    Unknown model names resolve to a conservative profile without tool calling,
    so a typo in ``LLM_MODEL_NAME`` is caught here and not on the first chat.
    """
    profile = get_model_profile(model_name)
    if not profile.supports_tool_calling:
        raise UnsupportedAgentModelError(
            f"LLM_MODEL_NAME={model_name!r} does not support tool calling "
            f"(registered: {profile.is_registered}). Pick a tool-calling model from "
            f"src/agent/adapters/llm/model_profiles.py."
        )


async def _settle(result: Any) -> None:
    """Await a container lifecycle call.

    The container returns an awaitable only when at least one of its resources
    is asynchronous, which is not the case once tests override them.
    """
    if inspect.isawaitable(result):
        await result


def build_container(settings: Settings) -> AppContainer:
    return AppContainer(settings=settings)


def create_app(container: Optional[AppContainer] = None) -> FastAPI:
    """Build the application.

    Without an argument the container is built from the environment when the
    application starts. Tests pass a container whose providers are overridden.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active = container
        if active is None:
            settings = Settings.from_env()
            logging.basicConfig(
                level=settings.log_level,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
            active = build_container(settings)

        ensure_model_can_call_tools(active.settings().llm_model_name)

        app.state.container = active
        await _settle(active.init_resources())
        active.wire(modules=WIRED_MODULES)
        try:
            yield
        finally:
            active.unwire()
            await _settle(active.shutdown_resources())

    app = FastAPI(title="ydb-agent-kit", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        return {"status": "ok"}

    app.include_router(accounts_router.router, prefix=API_PREFIX)
    app.include_router(tasks_router.router, prefix=API_PREFIX)
    app.include_router(agent_router.router, prefix=API_PREFIX)
    return app


app = create_app()
