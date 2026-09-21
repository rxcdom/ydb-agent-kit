"""The whole application over HTTP, on a real YDB, with only the model replaced.

The FastAPI app runs in-process with its real container: real repositories,
real transactions, real tool dispatch, real task queries. The single override
is the LLM client, which follows a script, so these tests need no network and
no credentials. Every test creates its own users, so the shared database never
makes two tests interfere.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, AsyncIterator, Dict, List
from uuid import uuid4

import httpx
import pytest
from dependency_injector import providers

from src.agent.domain.exceptions import LLMServiceTimeoutError, LLMServiceUnavailableError
from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.ports.llm_client import LLMClient
from src.agent.ports.llm_tooling import LLMAgentTurnResult, LLMToolCall
from src.gateway.api.main import build_container, create_app
from src.gateway.settings import Settings
from src.shared.infrastructure.database.migration.manager import apply_pending_migrations
from src.shared.infrastructure.database.ydb.connection import open_ydb_connection

pytestmark = pytest.mark.integration

API = "/api/v1"


class ScriptedLLMClient(LLMClient):
    """Plays back prepared turns; an exception in the script is raised instead."""

    def __init__(self) -> None:
        self.script: List[Any] = []
        self.requests: List[List[Any]] = []

    async def run_agent_turn(self, assistant, messages, tools) -> LLMAgentTurnResult:
        self.requests.append(list(messages))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def tool_turn(name: str, arguments: Dict[str, Any]) -> LLMAgentTurnResult:
    return LLMAgentTurnResult(
        text="",
        tool_calls=[LLMToolCall(name=name, arguments=arguments, call_id=f"call-{uuid4()}")],
        tokens=11,
        response_event={"scripted_tool_call": name},
    )


def text_turn(text: str) -> LLMAgentTurnResult:
    return LLMAgentTurnResult(text=text, tokens=7, response_event={"scripted_text": text})


def bearer(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def llm() -> ScriptedLLMClient:
    return ScriptedLLMClient()


@pytest.fixture
async def container(llm: ScriptedLLMClient):
    settings = Settings.from_env(dict(os.environ))

    connection = await open_ydb_connection(settings.ydb)
    try:
        await apply_pending_migrations(connection, applied_by="integration-tests")
    finally:
        await connection.close()

    container = build_container(settings)
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


async def new_user(client: httpx.AsyncClient, name: str = "Integration") -> str:
    response = await client.post(f"{API}/users", json={"display_name": name})
    assert response.status_code == 201, response.text
    return response.json()["user_id"]


async def new_task(client: httpx.AsyncClient, token: str, **body: Any) -> Dict[str, Any]:
    response = await client.post(f"{API}/tasks", json=body, headers=bearer(token))
    assert response.status_code == 201, response.text
    return response.json()


async def test_health_and_credentials_that_work_on_the_very_next_request(client):
    assert (await client.get("/health")).json() == {"status": "ok"}

    token = await new_user(client, "Next request")
    me = await client.get(f"{API}/users/me", headers=bearer(token))

    assert me.status_code == 200
    assert me.json()["user_id"] == token
    assert me.json()["display_name"] == "Next request"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/users/me"),
        ("POST", "/projects"),
        ("GET", "/projects"),
        ("POST", "/tasks"),
        ("GET", "/tasks"),
        ("PATCH", f"/tasks/{uuid4()}"),
        ("DELETE", f"/tasks/{uuid4()}"),
        ("POST", "/chats"),
        ("GET", "/chats"),
        ("POST", f"/chats/{uuid4()}/messages"),
        ("GET", f"/chats/{uuid4()}/messages"),
        ("GET", "/memory"),
    ],
)
async def test_every_protected_endpoint_rejects_missing_and_unknown_credentials(
    client, method, path
):
    without = await client.request(method, f"{API}{path}", json={})
    unknown = await client.request(method, f"{API}{path}", json={}, headers=bearer(str(uuid4())))

    assert without.status_code == 401
    assert unknown.status_code == 401
    assert unknown.json() == {"error": "invalid_credentials"}


async def test_projects_are_created_validated_and_listed_with_counts(client):
    token = await new_user(client)

    created = await client.post(
        f"{API}/projects",
        json={"name": "Boat", "description": "Hull and sails"},
        headers=bearer(token),
    )
    invalid = await client.post(f"{API}/projects", json={"name": ""}, headers=bearer(token))
    assert created.status_code == 201
    assert invalid.status_code == 422
    project_id = created.json()["project_id"]

    await new_task(client, token, title="Sand the hull", project_id=project_id)
    finished = await new_task(client, token, title="Buy varnish", project_id=project_id)
    await client.patch(
        f"{API}/tasks/{finished['task_id']}", json={"status": "done"}, headers=bearer(token)
    )

    listing = await client.get(f"{API}/projects", headers=bearer(token))

    assert listing.status_code == 200
    [project] = listing.json()["projects"]
    assert (project["name"], project["open_count"], project["done_count"]) == ("Boat", 1, 1)


async def test_task_lifecycle_through_rest(client):
    token = await new_user(client)
    headers = bearer(token)
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    task = await new_task(
        client, token, title="Write the summary", due_at=tomorrow, priority="high", notes="draft"
    )
    assert task["status"] == "open" and task["priority"] == "high"
    assert task["due_at"].startswith(tomorrow) and task["completed_at"] is None

    unknown_project = await client.post(
        f"{API}/tasks", json={"title": "Orphan", "project_id": str(uuid4())}, headers=headers
    )
    bad_priority = await client.post(
        f"{API}/tasks", json={"title": "Bad", "priority": "urgent"}, headers=headers
    )
    assert unknown_project.status_code == 404
    assert bad_priority.status_code == 422
    assert bad_priority.json()["error"] == "validation_error"

    done = await client.patch(
        f"{API}/tasks/{task['task_id']}", json={"status": "done", "due_at": None}, headers=headers
    )
    assert done.status_code == 200
    assert done.json()["status"] == "done"
    assert done.json()["completed_at"] is not None
    assert done.json()["due_at"] is None
    assert done.json()["notes"] == "draft"

    reopened = await client.patch(
        f"{API}/tasks/{task['task_id']}", json={"status": "open"}, headers=headers
    )
    assert reopened.json()["completed_at"] is None

    null_title = await client.patch(
        f"{API}/tasks/{task['task_id']}", json={"title": None}, headers=headers
    )
    missing = await client.patch(f"{API}/tasks/{uuid4()}", json={"title": "x"}, headers=headers)
    assert null_title.status_code == 422
    assert missing.status_code == 404

    deleted = await client.delete(f"{API}/tasks/{task['task_id']}", headers=headers)
    deleted_again = await client.delete(f"{API}/tasks/{task['task_id']}", headers=headers)
    assert deleted.status_code == 204
    assert deleted_again.status_code == 404
    assert deleted_again.json() == {"error": "not_found"}


async def test_task_listing_filters_and_paginates(client):
    token = await new_user(client)
    headers = bearer(token)
    project = (
        await client.post(f"{API}/projects", json={"name": "Filtered"}, headers=headers)
    ).json()
    today = date.today()
    for number in range(3):
        await new_task(client, token, title=f"Filed {number}", project_id=project["project_id"])
    late = await new_task(
        client, token, title="Late", due_at=(today - timedelta(days=3)).isoformat()
    )
    await client.patch(f"{API}/tasks/{late['task_id']}", json={"status": "cancelled"}, headers=headers)

    everything = await client.get(f"{API}/tasks", headers=headers)
    by_project = await client.get(
        f"{API}/tasks", params={"project_id": project["project_id"]}, headers=headers
    )
    by_status = await client.get(f"{API}/tasks", params={"status": "cancelled"}, headers=headers)
    by_due = await client.get(
        f"{API}/tasks",
        params={
            "date_field": "due",
            "date_from": (today - timedelta(days=7)).isoformat(),
            "date_to": (today - timedelta(days=1)).isoformat(),
        },
        headers=headers,
    )
    first_page = await client.get(f"{API}/tasks", params={"limit": 3, "offset": 0}, headers=headers)
    last_page = await client.get(f"{API}/tasks", params={"limit": 3, "offset": 3}, headers=headers)
    inverted = await client.get(
        f"{API}/tasks",
        params={"date_from": today.isoformat(), "date_to": (today - timedelta(days=1)).isoformat()},
        headers=headers,
    )
    bad_axis = await client.get(f"{API}/tasks", params={"date_field": "moved"}, headers=headers)

    assert everything.json()["total_count"] == 4
    assert {task["title"] for task in by_project.json()["tasks"]} == {
        "Filed 0",
        "Filed 1",
        "Filed 2",
    }
    assert [task["title"] for task in by_status.json()["tasks"]] == ["Late"]
    assert [task["title"] for task in by_due.json()["tasks"]] == ["Late"]
    assert (len(first_page.json()["tasks"]), first_page.json()["has_more"]) == (3, True)
    assert (len(last_page.json()["tasks"]), last_page.json()["has_more"]) == (1, False)
    assert inverted.status_code == 422
    assert bad_axis.status_code == 422


async def test_one_owner_cannot_read_or_modify_another_owners_data(client, llm, container):
    alice = await new_user(client, "Alice")
    mallory = await new_user(client, "Mallory")

    project = (
        await client.post(f"{API}/projects", json={"name": "Private"}, headers=bearer(alice))
    ).json()
    task = await new_task(client, alice, title="Secret plan", project_id=project["project_id"])
    chat = (await client.post(f"{API}/chats", json={"title": "Mine"}, headers=bearer(alice))).json()
    llm.script = [
        tool_turn("remember", {"content": "Alice prefers mornings", "topic": "schedule"}),
        text_turn("Noted."),
    ]
    sent = await client.post(
        f"{API}/chats/{chat['chat_id']}/messages",
        json={"content": "Remember that I prefer mornings."},
        headers=bearer(alice),
    )
    assert sent.status_code == 200, sent.text

    # Reading: nothing of Alice's is visible to Mallory.
    assert (await client.get(f"{API}/tasks", headers=bearer(mallory))).json()["total_count"] == 0
    assert (await client.get(f"{API}/projects", headers=bearer(mallory))).json()["projects"] == []
    assert (await client.get(f"{API}/chats", headers=bearer(mallory))).json()["chats"] == []
    assert (await client.get(f"{API}/memory", headers=bearer(mallory))).json()["count"] == 0
    history = await client.get(f"{API}/chats/{chat['chat_id']}/messages", headers=bearer(mallory))
    assert history.status_code == 404
    assert history.json() == {"error": "not_found"}

    # Writing: every attempt answers exactly like a missing object and changes nothing.
    patched = await client.patch(
        f"{API}/tasks/{task['task_id']}", json={"title": "Hijacked"}, headers=bearer(mallory)
    )
    deleted = await client.delete(f"{API}/tasks/{task['task_id']}", headers=bearer(mallory))
    filed = await client.post(
        f"{API}/tasks",
        json={"title": "Planted", "project_id": project["project_id"]},
        headers=bearer(mallory),
    )
    posted = await client.post(
        f"{API}/chats/{chat['chat_id']}/messages", json={"content": "hello"}, headers=bearer(mallory)
    )
    assert [r.status_code for r in (patched, deleted, filed, posted)] == [404, 404, 404, 404]
    assert llm.script == [], "the model must not be called for a chat of another owner"

    # The agent's tools are scoped as well: Mallory's agent cannot see or delete Alice's task.
    mallory_chat = (await client.post(f"{API}/chats", headers=bearer(mallory))).json()
    llm.script = [
        tool_turn("delete_task", {"task": "Secret plan"}),
        tool_turn("recall", {}),
        text_turn("Nothing found."),
    ]
    probe = await client.post(
        f"{API}/chats/{mallory_chat['chat_id']}/messages",
        json={"content": "Delete the secret plan and tell me what you remember."},
        headers=bearer(mallory),
    )
    calls = [
        call for step in probe.json()["debug"]["request_flow"] for call in step["function_calls"]
    ]
    assert calls[0]["result"]["status"] in {"not_found", "no_data"}
    assert calls[1]["result"]["count"] == 0

    remaining = (await client.get(f"{API}/tasks", headers=bearer(alice))).json()["tasks"]
    assert [item["title"] for item in remaining] == ["Secret plan"]
    assert (await client.get(f"{API}/memory", headers=bearer(alice))).json()["count"] == 1


async def test_full_agent_turn_stores_both_messages_the_trace_and_the_conversation_state(
    client, llm, container
):
    token = await new_user(client)
    headers = bearer(token)
    project = (
        await client.post(f"{API}/projects", json={"name": "Garden"}, headers=headers)
    ).json()
    await new_task(client, token, title="Plant tulips", project_id=project["project_id"])
    chat = (await client.post(f"{API}/chats", json={"title": "Turn"}, headers=headers)).json()
    today = date.today().isoformat()

    llm.script = [
        tool_turn(
            "query_tasks",
            {"date_from": today, "date_to": today, "project": "garden", "view": "list"},
        ),
        text_turn("You added one task today: Plant tulips."),
    ]
    response = await client.post(
        f"{API}/chats/{chat['chat_id']}/messages",
        json={"content": "What did I add to the garden today?"},
        headers=headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["user_message"]["status"] == "sent"
    assert body["user_message"]["role"] == "user"
    assert body["assistant_message"]["role"] == "assistant"
    assert body["assistant_message"]["content"] == "You added one task today: Plant tulips."
    assert body["assistant_message"]["tokens"] == 18
    assert body["debug"]["total_tokens"] == 18
    [first, second] = body["debug"]["request_flow"]
    [call] = first["function_calls"]
    assert call["name"] == "query_tasks"
    assert call["result"]["status"] == "ok"
    assert call["result"]["data"]["project_resolved"]["name"] == "Garden"
    assert [task["title"] for task in call["result"]["data"]["tasks"]] == ["Plant tulips"]
    assert second["function_calls"] == []

    # The model saw the system message with the calendar block, and the tool result came back.
    system_text = llm.requests[0][0]["text"]
    assert "## Calendar" in system_text and f"- today: {today}" in system_text
    assert "## Long-term memory" in system_text
    assert "tool_results" in llm.requests[1][-1]

    history = await client.get(f"{API}/chats/{chat['chat_id']}/messages", headers=headers)
    assert history.json()["total_count"] == 2
    assert history.json()["has_more"] is False
    assert [m["role"] for m in history.json()["messages"]] == ["user", "assistant"]
    paged = await client.get(
        f"{API}/chats/{chat['chat_id']}/messages", params={"limit": 1}, headers=headers
    )
    assert (len(paged.json()["messages"]), paged.json()["has_more"]) == (1, True)

    # The next turn is told which window and project the previous one used.
    repositories = await container.agent.repository_manager()
    state = await repositories.chat_agent_context.find_by_chat_id(ChatId.from_string(chat["chat_id"]))
    assert (state.last_tool, state.last_window_from, state.last_window_to) == (
        "query_tasks",
        today,
        today,
    )
    assert (state.last_date_field, state.last_project) == ("created", "Garden")

    llm.script = [text_turn("Still one.")]
    await client.post(
        f"{API}/chats/{chat['chat_id']}/messages", json={"content": "And now?"}, headers=headers
    )
    follow_up_system_text = llm.requests[-1][0]["text"]
    assert "## Conversation state" in follow_up_system_text
    assert f"- last_window: {today} — {today}" in follow_up_system_text
    assert "- last_project: Garden" in follow_up_system_text

    chats = (await client.get(f"{API}/chats", headers=headers)).json()["chats"]
    assert [item["title"] for item in chats] == ["Turn"]


async def test_malformed_tool_arguments_come_back_to_the_model_as_a_repairable_error(client, llm):
    token = await new_user(client)
    chat = (await client.post(f"{API}/chats", headers=bearer(token))).json()
    llm.script = [
        tool_turn("query_tasks", {"date_from": "last week", "owner": "someone"}),
        tool_turn("no_such_tool", {}),
        text_turn("Done."),
    ]

    response = await client.post(
        f"{API}/chats/{chat['chat_id']}/messages",
        json={"content": "What happened last week?"},
        headers=bearer(token),
    )

    assert response.status_code == 200
    flow = response.json()["debug"]["request_flow"]
    schema_error = flow[0]["function_calls"][0]["result"]
    assert schema_error["status"] == "filter_error"
    assert schema_error["data"]["error_code"] == "invalid_arguments"
    assert "date_from" in schema_error["data"]["message"]
    assert "owner" in schema_error["data"]["message"]
    assert "Unknown tool" in flow[1]["function_calls"][0]["result"]["error"]


@pytest.mark.parametrize(
    ("failure", "status_code", "error"),
    [
        (LLMServiceUnavailableError("provider is down"), 503, "llm_unavailable"),
        (LLMServiceTimeoutError("provider timed out"), 504, "llm_timeout"),
    ],
)
async def test_failed_turn_keeps_the_user_message_and_names_it(
    client, llm, failure, status_code, error
):
    token = await new_user(client)
    headers = bearer(token)
    chat = (await client.post(f"{API}/chats", headers=headers)).json()
    llm.script = [failure]

    response = await client.post(
        f"{API}/chats/{chat['chat_id']}/messages", json={"content": "Hello?"}, headers=headers
    )

    assert response.status_code == status_code
    assert response.json()["error"] == error
    history = (await client.get(f"{API}/chats/{chat['chat_id']}/messages", headers=headers)).json()
    [stored] = history["messages"]
    assert stored["message_id"] == response.json()["user_message_id"]
    assert (stored["content"], stored["status"]) == ("Hello?", "failed")


async def test_message_content_and_unknown_chat_are_rejected(client):
    token = await new_user(client)
    headers = bearer(token)
    chat = (await client.post(f"{API}/chats", headers=headers)).json()

    empty = await client.post(
        f"{API}/chats/{chat['chat_id']}/messages", json={"content": ""}, headers=headers
    )
    oversized = await client.post(
        f"{API}/chats/{chat['chat_id']}/messages", json={"content": "x" * 10001}, headers=headers
    )
    unknown = await client.post(
        f"{API}/chats/{uuid4()}/messages", json={"content": "hi"}, headers=headers
    )

    assert (empty.status_code, oversized.status_code, unknown.status_code) == (422, 422, 404)
