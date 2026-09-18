from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class MessageId:
    """Identifier of a message."""

    value: uuid.UUID

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def generate(cls) -> MessageId:
        return cls(uuid.uuid4())

    @classmethod
    def from_string(cls, value: str) -> MessageId:
        return cls(uuid.UUID(value))
