"""Application service behind the ``remember``, ``recall`` and ``forget`` tools.

It owns the policy of the long-term memory vault:

* privacy: ``MemoryPrivacyFilter`` rejects content that looks like a secret
  before anything reaches the datastore;
* update instead of duplicate: a new fact that restates a stored one (same topic,
  or mostly the same significant words) replaces it in place;
* size cap: a full vault evicts its oldest entry to make room, so the vault is a
  rolling set of the freshest facts and a write is never refused for lack of room;
* owner scope: every read and every delete is bound to the ``user_id`` of the
  authenticated principal.

Log lines of this service record lengths and counts, never the content of a fact.
"""
from __future__ import annotations

import logging
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional

from src.agent.domain.entities.user_memory import (
    USER_MEMORY_CONTENT_MAX_LENGTH,
    USER_MEMORY_TOPIC_MAX_LENGTH,
    UserMemory,
)
from src.agent.domain.services.memory_privacy_filter import MemoryPrivacyFilter
from src.agent.ports.repository_manager import RepositoryManager
from src.shared.domain.value_objects.user_id import UserId

logger = logging.getLogger(__name__)

# Entries per user. At the cap the oldest entry (by ``created_at``) is evicted.
USER_MEMORY_MAX_ENTRIES = 100

# Result caps of the read paths, so a tool result stays cheap in tokens.
RECALL_RESULT_LIMIT = 10
FORGET_MATCH_LIMIT = 20

# Shorter words ("a", "to", "the", "for") say nothing about what a fact is about.
_MIN_SIGNIFICANT_TOKEN_LENGTH = 4
# Bounds the number of searches one call can issue.
_MAX_SEARCH_TOKENS = 5
_SIMILAR_CANDIDATE_LIMIT = 5

REMEMBER_CREATED = "created"
REMEMBER_UPDATED = "updated"
REMEMBER_REJECTED = "rejected"


@dataclass(frozen=True)
class RememberOutcome:
    """Result of ``remember``.

    ``status`` is ``created``, ``updated`` or ``rejected``. ``reason`` is set
    only on rejection; ``evicted`` tells that the oldest entry made room.
    """

    status: str
    reason: Optional[str] = None
    memory_id: Optional[str] = None
    evicted: bool = False


@dataclass(frozen=True)
class ForgetOutcome:
    """Result of ``forget``."""

    deleted_count: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _significant_tokens(text: str) -> List[str]:
    """Lower-cased words of ``text`` that are long enough to carry meaning.

    Surrounding punctuation is stripped, so "car," and "car" are the same word.
    Order is kept and repeats are dropped.
    """
    tokens: List[str] = []
    for word in text.split():
        token = word.strip(string.punctuation).lower()
        if len(token) >= _MIN_SIGNIFICANT_TOKEN_LENGTH and token not in tokens:
            tokens.append(token)
    return tokens


class UserMemoryWriter:
    """Carry out the agent's memory tool calls on the vault of one user."""

    def __init__(
        self,
        repository_manager: RepositoryManager,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._repositories = repository_manager
        self._clock = clock

    async def remember(
        self, user_id: UserId, *, content: str, topic: Optional[str] = None
    ) -> RememberOutcome:
        """Store one fact: privacy check, then update or insert, then the cap."""
        cleaned = (content or "").strip()[:USER_MEMORY_CONTENT_MAX_LENGTH]
        cleaned_topic = (topic or "").strip()[:USER_MEMORY_TOPIC_MAX_LENGTH] or None

        verdict = MemoryPrivacyFilter.check(cleaned)
        if not verdict.allowed:
            logger.info(
                "Memory write rejected: reason=%s content_length=%d",
                verdict.reason,
                len(cleaned),
            )
            return RememberOutcome(status=REMEMBER_REJECTED, reason=verdict.reason)

        now = self._clock()
        existing = await self._find_similar(user_id, content=cleaned, topic=cleaned_topic)
        if existing is not None:
            existing.content = cleaned
            if cleaned_topic is not None:
                existing.topic = cleaned_topic
            existing.updated_at = now
            await self._repositories.user_memory.save(existing)
            logger.info(
                "Memory entry updated: memory_id=%s content_length=%d",
                existing.memory_id,
                len(cleaned),
            )
            return RememberOutcome(status=REMEMBER_UPDATED, memory_id=existing.memory_id)

        evicted = await self._make_room(user_id)
        memory = UserMemory(
            memory_id=UserMemory.generate_id(),
            user_id=user_id,
            content=cleaned,
            topic=cleaned_topic,
            created_at=now,
            updated_at=now,
        )
        await self._repositories.user_memory.save(memory)
        logger.info(
            "Memory entry created: memory_id=%s content_length=%d topic_present=%s evicted=%s",
            memory.memory_id,
            len(cleaned),
            cleaned_topic is not None,
            evicted,
        )
        return RememberOutcome(
            status=REMEMBER_CREATED, memory_id=memory.memory_id, evicted=evicted
        )

    async def recall(self, user_id: UserId, *, query: Optional[str] = None) -> List[UserMemory]:
        """Return the entries matching ``query``, or the freshest ones.

        The fallback to the freshest entries covers both a call without a query
        and a broad question such as "what do you know about me", whose literal
        words match nothing.
        """
        cleaned_query = (query or "").strip()
        results = await self._search_by_tokens(user_id, cleaned_query, RECALL_RESULT_LIMIT)
        if not results:
            results = await self._repositories.user_memory.find_by_user_id(
                user_id, limit=RECALL_RESULT_LIMIT
            )
        logger.info(
            "Memory recall: query_length=%d result_count=%d", len(cleaned_query), len(results)
        )
        return results

    async def forget(self, user_id: UserId, *, query: str) -> ForgetOutcome:
        """Delete every entry of the owner that matches ``query``.

        Unlike ``recall`` there is no fallback: a query that matches nothing
        deletes nothing.
        """
        cleaned_query = (query or "").strip()
        matches = await self._search_by_tokens(user_id, cleaned_query, FORGET_MATCH_LIMIT)
        for memory in matches:
            await self._repositories.user_memory.delete(memory.memory_id, user_id)
        logger.info(
            "Memory forget: query_length=%d deleted_count=%d", len(cleaned_query), len(matches)
        )
        return ForgetOutcome(deleted_count=len(matches))

    async def _make_room(self, user_id: UserId) -> bool:
        """Evict the oldest entry when the vault is full. Return True if it did."""
        count = await self._repositories.user_memory.count_by_user_id(user_id)
        if count < USER_MEMORY_MAX_ENTRIES:
            return False
        oldest = await self._repositories.user_memory.find_oldest_by_user_id(user_id)
        if oldest is None:
            return False
        await self._repositories.user_memory.delete(oldest.memory_id, user_id)
        return True

    async def _search_by_tokens(
        self, user_id: UserId, query: str, limit: int
    ) -> List[UserMemory]:
        """Search once per significant word of ``query`` and merge the results.

        One substring search over the whole phrase misses a change of word order
        or wording ("saving for a car" does not contain "car savings"), so every
        significant word is searched on its own. The union is ranked by
        ``updated_at``, newest first, and capped. A query without a significant
        word is searched as it is.
        """
        if not query:
            return []

        tokens = _significant_tokens(query) or [query]
        merged: dict[str, UserMemory] = {}
        for token in tokens[:_MAX_SEARCH_TOKENS]:
            found = await self._repositories.user_memory.search_by_user_id(
                user_id, query=token, limit=limit
            )
            for memory in found:
                merged[memory.memory_id] = memory

        ranked = sorted(merged.values(), key=lambda memory: memory.updated_at, reverse=True)
        return ranked[:limit]

    async def _find_similar(
        self, user_id: UserId, *, content: str, topic: Optional[str]
    ) -> Optional[UserMemory]:
        """Find the stored entry that a new fact restates, if there is one.

        An entry is the same fact when its topic equals the new topic (ignoring
        case), or when it shares at least half of the new fact's significant
        words. The test is conservative on purpose: a miss costs one extra row,
        while a false match would overwrite an unrelated memory.
        """
        if topic:
            by_topic = await self._repositories.user_memory.search_by_user_id(
                user_id, query=topic, limit=_SIMILAR_CANDIDATE_LIMIT
            )
            for memory in by_topic:
                if memory.topic and memory.topic.strip().lower() == topic.lower():
                    return memory

        tokens = set(_significant_tokens(content))
        if not tokens:
            return None

        candidates = await self._search_by_tokens(user_id, content, _SIMILAR_CANDIDATE_LIMIT)
        for memory in candidates:
            shared = tokens & set(_significant_tokens(memory.content))
            if len(shared) * 2 >= len(tokens):
                return memory
        return None
