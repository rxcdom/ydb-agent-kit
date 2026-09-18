"""FastAPI application: lifespan, routers, exception mapping."""
from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI

from src.accounts.adapters.api import auth as accounts_auth
from src.accounts.adapters.api import router as accounts_router
from src.gateway.api.errors import register_exception_handlers
from src.gateway.di.container import AppContainer
from src.gateway.settings import Settings

API_PREFIX = "/api/v1"
WIRED_MODULES = [accounts_auth, accounts_router]


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
    return app


app = create_app()
