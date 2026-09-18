"""Errors the agent module raises on purpose."""

from __future__ import annotations

from typing import Optional

from src.shared.domain.exceptions import (
    AccessDeniedError,
    DomainError,
    NotFoundError,
    ValidationError,
)


class LLMServiceError(DomainError):
    """The language-model provider could not produce a usable turn.

    ``cause`` is the provider or transport error this one was mapped from, so
    logs and error handlers can still see the original failure. It is stored as
    the standard ``__cause__``; passing ``cause=`` and ``raise ... from error``
    are therefore equivalent.
    """

    def __init__(self, message: str = "", *, cause: Optional[BaseException] = None):
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause

    @property
    def cause(self) -> Optional[BaseException]:
        return self.__cause__


class LLMServiceUnavailableError(LLMServiceError):
    """The provider is unreachable or answered with a server-side failure."""


class LLMServiceTimeoutError(LLMServiceError):
    """The provider did not answer in time."""


class LLMServiceAuthenticationError(LLMServiceError):
    """The provider rejected the configured credentials."""


class LLMServiceInvalidResponseError(LLMServiceError):
    """The provider answered, but the answer cannot be used.

    Examples: an empty completion without tool calls, or tool-call arguments that
    are not a JSON object.
    """


class LLMResponseGenerationFailedError(LLMServiceError):
    """The turn failed after the user message was stored.

    The user message stays persisted with status ``failed``. ``user_message_id``
    names it, so the client can show the failure on that message and retry.
    """

    def __init__(
        self,
        message: str = "",
        *,
        user_message_id: str,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message, cause=cause)
        self.user_message_id = user_message_id


class ChatNotFoundError(NotFoundError):
    """The addressed chat does not exist."""


class ChatAccessDeniedError(AccessDeniedError):
    """The chat exists but belongs to another user."""


class InvalidMessageContentError(ValidationError):
    """The message text is empty or otherwise unacceptable."""
