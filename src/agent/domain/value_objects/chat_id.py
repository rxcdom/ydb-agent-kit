from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatId:
    """Identifier of a chat."""

    value: uuid.UUID

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def generate(cls) -> ChatId:
        return cls(uuid.uuid4())

    @classmethod
    def from_string(cls, value: str) -> ChatId:
        return cls(uuid.UUID(value))
