import pytest

from src.agent.domain.exceptions import (
    LLMServiceAuthenticationError,
    LLMServiceInvalidResponseError,
    LLMServiceTimeoutError,
    LLMServiceUnavailableError,
)
from tests.unit.gateway.conftest import bearer
from tests.unit.gateway.fakes import text_turn, tool_turn

API = "/api/v1"


async def _new_chat(client, token) -> str:
    response = await client.post(f"{API}/chats", json={"title": "Demo"}, headers=bearer(token))
    assert response.status_code == 201
    return response.json()["chat_id"]


async def test_chat_turn_returns_both_messages_and_the_trace(client, token, llm):
    chat_id = await _new_chat(client, token)
    llm.script = [tool_turn("list_projects", {}), text_turn("You have no projects yet.")]

    response = await client.post(
        f"{API}/chats/{chat_id}/messages",
        json={"content": "Which projects do I have?"},
        headers=bearer(token),
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"user_message", "assistant_message", "debug"}
    assert set(body["user_message"]) == {
        "message_id",
        "chat_id",
        "role",
        "content",
        "tokens",
        "status",
        "created_at",
    }
    assert body["user_message"]["status"] == "sent"
    assert body["assistant_message"]["content"] == "You have no projects yet."
    assert body["debug"]["total_tokens"] == 8
    [call] = body["debug"]["request_flow"][0]["function_calls"]
    assert (call["name"], call["arguments"], call["result"]["status"]) == (
        "list_projects",
        {},
        "no_data",
    )

    history = await client.get(f"{API}/chats/{chat_id}/messages", headers=bearer(token))
    assert history.json()["total_count"] == 2
    chats = await client.get(f"{API}/chats", headers=bearer(token))
    assert [chat["chat_id"] for chat in chats.json()["chats"]] == [chat_id]


async def test_memory_written_by_the_agent_is_visible_through_the_memory_endpoint(
    client, token, llm
):
    chat_id = await _new_chat(client, token)
    llm.script = [
        tool_turn("remember", {"content": "Prefers short answers", "topic": "style"}),
        text_turn("Noted."),
    ]

    await client.post(
        f"{API}/chats/{chat_id}/messages",
        json={"content": "Remember that I prefer short answers."},
        headers=bearer(token),
    )
    memory = await client.get(f"{API}/memory", headers=bearer(token))

    assert memory.status_code == 200
    assert memory.json()["count"] == 1
    assert memory.json()["memories"][0]["content"] == "Prefers short answers"
    assert memory.json()["memories"][0]["topic"] == "style"


async def test_another_owners_chat_answers_like_a_missing_one(client, token, llm):
    chat_id = await _new_chat(client, token)
    other = (await client.post(f"{API}/users", json={})).json()["user_id"]

    posted = await client.post(
        f"{API}/chats/{chat_id}/messages", json={"content": "hello"}, headers=bearer(other)
    )
    read = await client.get(f"{API}/chats/{chat_id}/messages", headers=bearer(other))

    assert posted.status_code == read.status_code == 404
    assert posted.json() == read.json() == {"error": "not_found"}
    assert llm.calls == 0


@pytest.mark.parametrize(
    ("failure", "status_code", "error"),
    [
        (LLMServiceUnavailableError("down"), 503, "llm_unavailable"),
        (LLMServiceInvalidResponseError("empty completion"), 503, "llm_unavailable"),
        (LLMServiceAuthenticationError("bad key"), 503, "llm_misconfigured"),
        (LLMServiceTimeoutError("slow"), 504, "llm_timeout"),
    ],
)
async def test_model_failures_map_to_http_and_keep_the_user_message(
    client, token, llm, failure, status_code, error
):
    chat_id = await _new_chat(client, token)
    llm.script = [failure]

    response = await client.post(
        f"{API}/chats/{chat_id}/messages", json={"content": "Hello?"}, headers=bearer(token)
    )

    assert response.status_code == status_code
    assert response.json()["error"] == error
    assert response.json()["status"] == "failed"
    history = await client.get(f"{API}/chats/{chat_id}/messages", headers=bearer(token))
    [stored] = history.json()["messages"]
    assert stored["message_id"] == response.json()["user_message_id"]
    assert stored["status"] == "failed"


async def test_message_content_is_validated(client, token):
    chat_id = await _new_chat(client, token)

    empty = await client.post(
        f"{API}/chats/{chat_id}/messages", json={"content": ""}, headers=bearer(token)
    )
    blank = await client.post(
        f"{API}/chats/{chat_id}/messages", json={"content": "   "}, headers=bearer(token)
    )

    assert empty.status_code == blank.status_code == 422
    assert blank.json()["error"] == "validation_error"
