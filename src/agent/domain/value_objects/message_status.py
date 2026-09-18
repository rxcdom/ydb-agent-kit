from __future__ import annotations

from enum import Enum


class MessageStatus(str, Enum):
    """Delivery state of a message.

    A user message is stored as ``processing`` before the model is called. It
    becomes ``sent`` once the assistant reply is stored next to it, or ``failed``
    when the turn could not be completed, so the client can offer a retry.
    Assistant messages are always stored as ``sent``.
    """

    PROCESSING = "processing"
    SENT = "sent"
    FAILED = "failed"
