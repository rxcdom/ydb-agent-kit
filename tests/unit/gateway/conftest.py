"""Gateway tests run the real FastAPI app with every outbound adapter replaced by a fake."""
from __future__ import annotations

from typing import AsyncIterator, Dict, Optional

import httpx
import pytest
from dependency_injector import providers

from src.accounts.domain.entities.user import User
from src.accounts.ports.user_repository import UserRepository
from src.gateway.api.main import build_container, create_app
from src.gateway.settings import Settings
from src.shared.domain.value_objects.user_id import UserId
from tests.unit.gateway.fakes import InMemoryAgentRepositoryManager, ScriptedLLMClient
from tests.unit.tasks.in_memory import InMemoryTasksRepositoryManager


class InMemoryUserRepository(UserRepository):
    def __init__(self) -> None:
        self.rows: Dict[UserId, User] = {}

    async def save(self, user: User) -> None:
        self.rows[user.user_id] = user

    async def find_by_id(self, user_id: UserId) -> Optional[User]:
        return self.rows.get(user_id)


class _ClosedConnection:
    """Stands in for the YDB connection; nothing in these tests may touch it."""

    database = "/local"

    @property
    def pool(self):
        raise AssertionError("a gateway unit test reached for the real session pool")

    driver = pool


@pytest.fixture
def llm() -> ScriptedLLMClient:
    return ScriptedLLMClient()


@pytest.fixture
def container(llm: ScriptedLLMClient):
    container = build_container(Settings.from_env({}))
    container.core.ydb_connection.override(providers.Object(_ClosedConnection()))
    container.accounts.user_repository.override(providers.Object(InMemoryUserRepository()))
    container.tasks.repository_manager.override(providers.Object(InMemoryTasksRepositoryManager()))
    container.agent.repository_manager.override(providers.Object(InMemoryAgentRepositoryManager()))
    container.agent.llm_client.override(providers.Object(llm))
    yield container
    container.unwire()


@pytest.fixture
async def client(container) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(container)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
            yield http


@pytest.fixture
async def token(client: httpx.AsyncClient) -> str:
    response = await client.post("/api/v1/users", json={"display_name": "Demo"})
    assert response.status_code == 201
    return response.json()["user_id"]


def bearer(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
