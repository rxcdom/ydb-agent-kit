"""Tests for the memory vault writer: privacy, update-or-insert, cap, recall, forget."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from src.agent.application.memory.user_memory_writer import (
    FORGET_MATCH_LIMIT,
    RECALL_RESULT_LIMIT,
    USER_MEMORY_MAX_ENTRIES,
    UserMemoryWriter,
)
from src.agent.domain.entities.user_memory import (
    USER_MEMORY_CONTENT_MAX_LENGTH,
    USER_MEMORY_TOPIC_MAX_LENGTH,
    UserMemory,
)
from src.shared.domain.value_objects.user_id import UserId

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def _memory(user_id: UserId, content: str, topic=None, age_days: int = 0) -> UserMemory:
    timestamp = NOW - timedelta(days=age_days)
    return UserMemory(
        memory_id=UserMemory.generate_id(),
        user_id=user_id,
        content=content,
        topic=topic,
        created_at=timestamp,
        updated_at=timestamp,
    )


@pytest.fixture
def user_id() -> UserId:
    return UserId.generate()


@pytest.fixture
def writer(mock_repository_manager) -> UserMemoryWriter:
    return UserMemoryWriter(mock_repository_manager, clock=lambda: NOW)


async def test_remember_creates_a_new_entry(writer, mock_repository_manager, user_id):
    outcome = await writer.remember(
        user_id, content="  Saving for a car, the target is 20000  ", topic=" car "
    )

    assert outcome.status == "created"
    assert outcome.memory_id is not None
    assert outcome.evicted is False
    saved = mock_repository_manager.user_memory.save.await_args.args[0]
    assert saved.memory_id == outcome.memory_id
    assert saved.content == "Saving for a car, the target is 20000"
    assert saved.topic == "car"
    assert saved.user_id == user_id
    assert saved.created_at == saved.updated_at == NOW


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("my bank password is tulip-garden-42", "looks_like_secret"),
        ("the PIN code of the card is 4321, do not forget", "looks_like_secret"),
        ("card number 1234 5678 9012 3456", "looks_like_card_number"),
        ("ok", "content_too_short"),
    ],
)
async def test_remember_rejects_secrets_and_noise(
    writer, mock_repository_manager, content, reason
):
    outcome = await writer.remember(UserId.generate(), content=content)

    assert outcome.status == "rejected"
    assert outcome.reason == reason
    assert outcome.memory_id is None
    # A rejected fact never reaches the datastore, not even as a lookup.
    mock_repository_manager.user_memory.save.assert_not_awaited()
    mock_repository_manager.user_memory.search_by_user_id.assert_not_awaited()


async def test_remember_updates_the_entry_with_the_same_topic(
    writer, mock_repository_manager, user_id
):
    existing = _memory(user_id, "Saving for a car", topic="Car", age_days=10)
    mock_repository_manager.user_memory.search_by_user_id.return_value = [existing]

    outcome = await writer.remember(
        user_id, content="Saving for a car, 5000 put aside so far", topic="car"
    )

    assert outcome.status == "updated"
    assert outcome.memory_id == existing.memory_id
    saved = mock_repository_manager.user_memory.save.await_args.args[0]
    assert saved is existing
    assert saved.content == "Saving for a car, 5000 put aside so far"
    assert saved.topic == "car"
    assert saved.updated_at == NOW
    assert saved.created_at == NOW - timedelta(days=10)
    # The fact was replaced in place: no second row, no eviction check.
    assert mock_repository_manager.user_memory.save.await_count == 1
    mock_repository_manager.user_memory.count_by_user_id.assert_not_awaited()


async def test_remember_updates_on_word_overlap_without_a_topic(
    writer, mock_repository_manager, user_id
):
    existing = _memory(user_id, "The user is saving money for a holiday.", topic="plans")
    mock_repository_manager.user_memory.search_by_user_id.return_value = [existing]

    outcome = await writer.remember(user_id, content="saving money, holiday in March")

    assert outcome.status == "updated"
    assert outcome.memory_id == existing.memory_id
    saved = mock_repository_manager.user_memory.save.await_args.args[0]
    # Without a new topic the stored topic is kept.
    assert saved.topic == "plans"


async def test_remember_keeps_an_unrelated_entry_untouched(
    writer, mock_repository_manager, user_id
):
    unrelated = _memory(user_id, "Works from home on Mondays and Thursdays")
    mock_repository_manager.user_memory.search_by_user_id.return_value = [unrelated]

    outcome = await writer.remember(
        user_id, content="Prefers deadlines before lunch, never on Mondays"
    )

    assert outcome.status == "created"
    saved = mock_repository_manager.user_memory.save.await_args.args[0]
    assert saved is not unrelated
    assert unrelated.content == "Works from home on Mondays and Thursdays"


async def test_remember_at_the_cap_evicts_the_oldest_entry(
    writer, mock_repository_manager, user_id
):
    oldest = _memory(user_id, "the very first fact", age_days=365)
    mock_repository_manager.user_memory.count_by_user_id.return_value = USER_MEMORY_MAX_ENTRIES
    mock_repository_manager.user_memory.find_oldest_by_user_id.return_value = oldest

    outcome = await writer.remember(user_id, content="a fresh fact about the summer holiday")

    assert outcome.status == "created"
    assert outcome.evicted is True
    mock_repository_manager.user_memory.delete.assert_awaited_once_with(
        oldest.memory_id, user_id
    )


async def test_remember_below_the_cap_evicts_nothing(writer, mock_repository_manager, user_id):
    mock_repository_manager.user_memory.count_by_user_id.return_value = (
        USER_MEMORY_MAX_ENTRIES - 1
    )

    outcome = await writer.remember(user_id, content="a fresh fact about the summer holiday")

    assert outcome.evicted is False
    mock_repository_manager.user_memory.find_oldest_by_user_id.assert_not_awaited()
    mock_repository_manager.user_memory.delete.assert_not_awaited()


async def test_remember_truncates_to_the_entity_caps(writer, mock_repository_manager, user_id):
    outcome = await writer.remember(
        user_id,
        content="fact " * USER_MEMORY_CONTENT_MAX_LENGTH,
        topic="t" * (USER_MEMORY_TOPIC_MAX_LENGTH + 50),
    )

    assert outcome.status == "created"
    saved = mock_repository_manager.user_memory.save.await_args.args[0]
    assert len(saved.content) == USER_MEMORY_CONTENT_MAX_LENGTH
    assert len(saved.topic) == USER_MEMORY_TOPIC_MAX_LENGTH


async def test_recall_returns_matches(writer, mock_repository_manager, user_id):
    match = _memory(user_id, "Saving for a car", topic="car")
    mock_repository_manager.user_memory.search_by_user_id.return_value = [match]

    results = await writer.recall(user_id, query="savings")

    assert results == [match]
    mock_repository_manager.user_memory.search_by_user_id.assert_awaited_once_with(
        user_id, query="savings", limit=RECALL_RESULT_LIMIT
    )
    mock_repository_manager.user_memory.find_by_user_id.assert_not_awaited()


async def test_recall_searches_each_significant_word_and_ranks_by_freshness(
    writer, mock_repository_manager, user_id
):
    older = _memory(user_id, "Holiday planned for March", age_days=30)
    newer = _memory(user_id, "Saving for a car", age_days=1)

    async def search(owner, *, query, limit):
        return {"holiday": [older], "savings": [newer, older]}.get(query, [])

    mock_repository_manager.user_memory.search_by_user_id.side_effect = search

    results = await writer.recall(user_id, query="My HOLIDAY and the savings, please")

    # Short words are skipped, punctuation and case are ignored, duplicates merge.
    searched = [
        call.kwargs["query"]
        for call in mock_repository_manager.user_memory.search_by_user_id.await_args_list
    ]
    assert searched == ["holiday", "savings", "please"]
    assert results == [newer, older]


async def test_recall_falls_back_to_the_freshest_entries_on_no_match(
    writer, mock_repository_manager, user_id
):
    fresh = _memory(user_id, "Holiday planned for March")
    mock_repository_manager.user_memory.search_by_user_id.return_value = []
    mock_repository_manager.user_memory.find_by_user_id.return_value = [fresh]

    results = await writer.recall(user_id, query="what do you know about me")

    assert results == [fresh]
    mock_repository_manager.user_memory.find_by_user_id.assert_awaited_once_with(
        user_id, limit=RECALL_RESULT_LIMIT
    )


@pytest.mark.parametrize("query", [None, "", "   "])
async def test_recall_without_a_query_returns_the_freshest_entries(
    writer, mock_repository_manager, user_id, query
):
    fresh = _memory(user_id, "Holiday planned for March")
    mock_repository_manager.user_memory.find_by_user_id.return_value = [fresh]

    results = await writer.recall(user_id, query=query)

    assert results == [fresh]
    mock_repository_manager.user_memory.search_by_user_id.assert_not_awaited()


async def test_a_query_of_short_words_is_searched_as_it_is(
    writer, mock_repository_manager, user_id
):
    match = _memory(user_id, "Saving for a car", topic="car")
    mock_repository_manager.user_memory.search_by_user_id.return_value = [match]

    results = await writer.recall(user_id, query="my car")

    assert results == [match]
    mock_repository_manager.user_memory.search_by_user_id.assert_awaited_once_with(
        user_id, query="my car", limit=RECALL_RESULT_LIMIT
    )


async def test_forget_deletes_every_match_scoped_to_the_owner(
    writer, mock_repository_manager, user_id
):
    first = _memory(user_id, "Saving for a car", topic="car")
    second = _memory(user_id, "Wants an electric car")
    mock_repository_manager.user_memory.search_by_user_id.return_value = [first, second]

    outcome = await writer.forget(user_id, query="electric")

    assert outcome.deleted_count == 2
    deleted = {
        call.args for call in mock_repository_manager.user_memory.delete.await_args_list
    }
    assert deleted == {(first.memory_id, user_id), (second.memory_id, user_id)}
    mock_repository_manager.user_memory.search_by_user_id.assert_awaited_once_with(
        user_id, query="electric", limit=FORGET_MATCH_LIMIT
    )


async def test_forget_without_a_match_deletes_nothing(writer, mock_repository_manager, user_id):
    mock_repository_manager.user_memory.search_by_user_id.return_value = []
    mock_repository_manager.user_memory.find_by_user_id.return_value = [
        _memory(user_id, "Holiday planned for March")
    ]

    outcome = await writer.forget(user_id, query="nothing like this is stored")

    # Unlike recall, forget has no fallback to the freshest entries.
    assert outcome.deleted_count == 0
    mock_repository_manager.user_memory.delete.assert_not_awaited()
    mock_repository_manager.user_memory.find_by_user_id.assert_not_awaited()


async def test_log_lines_never_carry_the_content_of_a_fact(
    writer, mock_repository_manager, user_id, caplog
):
    with caplog.at_level(logging.DEBUG, logger="src.agent.application.memory"):
        await writer.remember(user_id, content="Allergic to walnuts", topic="health")
        await writer.recall(user_id, query="walnuts")
        await writer.forget(user_id, query="walnuts")

    assert caplog.records
    assert "walnuts" not in caplog.text.lower()
    assert "health" not in caplog.text.lower()
