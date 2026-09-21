import httpx
import pytest
from dependency_injector import providers

from src.gateway.api.main import (
    UnsupportedAgentModelError,
    build_container,
    create_app,
    ensure_model_can_call_tools,
)
from src.gateway.settings import Settings
from src.shared.domain.exceptions import PersistenceError


def test_tool_calling_models_pass_the_startup_guard():
    ensure_model_can_call_tools("gpt-oss-120b")
    ensure_model_can_call_tools("deepseek-v4-flash")


@pytest.mark.parametrize("model_name", ["yandexgpt-5-lite", "a-model-nobody-registered"])
def test_models_without_tool_calling_are_refused(model_name):
    with pytest.raises(UnsupportedAgentModelError, match="does not support tool calling"):
        ensure_model_can_call_tools(model_name)


async def test_application_refuses_to_start_with_a_model_that_cannot_call_tools():
    container = build_container(Settings.from_env({"LLM_MODEL_NAME": "yandexgpt-5-lite"}))
    app = create_app(container)

    with pytest.raises(UnsupportedAgentModelError):
        async with app.router.lifespan_context(app):
            pass


class _BrokenUserRepository:
    def __init__(self, error: Exception):
        self._error = error

    async def save(self, user):
        raise self._error

    async def find_by_id(self, user_id):
        raise self._error


@pytest.mark.parametrize(
    ("failure", "status_code", "body"),
    [
        (PersistenceError("node is down"), 503, {"error": "storage_unavailable"}),
        (RuntimeError("a bug"), 500, {"error": "internal_error"}),
    ],
)
async def test_storage_and_unexpected_failures_do_not_leak_details(
    container, failure, status_code, body
):
    container.accounts.user_repository.override(providers.Object(_BrokenUserRepository(failure)))
    app = create_app(container)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
            response = await http.post("/api/v1/users", json={})

    assert response.status_code == status_code
    assert response.json() == body
