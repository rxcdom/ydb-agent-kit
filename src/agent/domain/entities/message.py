from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.agent.domain.value_objects.chat_id import ChatId
from src.agent.domain.value_objects.message_id import MessageId
from src.agent.domain.value_objects.message_status import MessageStatus
from src.shared.domain.value_objects.user_id import UserId


class MessageRole(str, Enum):
    """Who authored a message."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """One turn of a chat.

    ``user_id`` is the owner of the chat, repeated on the row so ownership can be
    checked without reading the chat. ``tokens`` is the usage the provider
    reported for an assistant turn and zero for a user turn. ``trace`` is the
    debug trace of the agent loop that produced an assistant message (iterations
    and tool calls); user messages carry none.
    """

    message_id: MessageId
    chat_id: ChatId
    user_id: UserId
    role: MessageRole
    content: str
    tokens: int
    status: MessageStatus
    created_at: datetime
    trace: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("Message content cannot be empty")
        if self.tokens < 0:
            raise ValueError("Tokens cannot be negative")
