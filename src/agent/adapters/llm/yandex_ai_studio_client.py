"""LLM client for the chat-completions API of Yandex AI Studio.

One ``run_agent_turn`` call is one model turn of the agent loop. Tools are bound
natively, the request-scoped history goes to the SDK in the shape the loop built
it, and the answer comes back as either final text or tool calls. Transient
provider failures are retried; every failure leaves the client as an
``LLMServiceError`` subclass.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

try:
    from yandex_ai_studio_sdk import AsyncAIStudio
    from yandex_ai_studio_sdk import exceptions as sdk_exceptions
except ModuleNotFoundError:  # pragma: no cover - exercised by patching the names below
    # The client stays importable without the SDK; the first turn reports it.
    AsyncAIStudio = None  # type: ignore[assignment,misc]
    sdk_exceptions = None  # type: ignore[assignment]

from src.agent.adapters.llm.model_profiles import get_model_profile
from src.agent.domain.entities.assistant import Assistant
from src.agent.domain.exceptions import (
    LLMServiceAuthenticationError,
    LLMServiceError,
    LLMServiceInvalidResponseError,
    LLMServiceTimeoutError,
    LLMServiceUnavailableError,
)
from src.agent.ports.llm_client import LLMClient
from src.agent.ports.llm_tooling import LLMAgentTurnResult, LLMToolCall, LLMToolDefinition

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]

DEFAULT_RETRY_ATTEMPTS = 2
DEFAULT_RETRY_BACKOFF_SECONDS: Tuple[float, ...] = (0.5, 1.0)

# A repeat can succeed after these; after the other error types it fails the same way.
_TRANSIENT_ERRORS = (LLMServiceUnavailableError, LLMServiceTimeoutError)

# gRPC status names the SDK reports for its control-plane calls (token exchange).
_RPC_AUTHENTICATION_STATUSES = frozenset({"UNAUTHENTICATED", "PERMISSION_DENIED"})
_RPC_TIMEOUT_STATUSES = frozenset({"DEADLINE_EXCEEDED"})

# Transport errors reach the client unwrapped, as whatever the HTTP or RPC library
# raised, so they are recognised by their type name and message.
_TIMEOUT_TYPE_MARKERS = ("timeout", "timedout")
_TIMEOUT_MESSAGE_MARKERS = ("timeout", "timed out", "deadline exceeded")
_AUTHENTICATION_TYPE_MARKERS = ("auth",)
_AUTHENTICATION_MESSAGE_MARKERS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "authentication",
    "invalid api key",
    "unauthenticated",
)
_INVALID_RESPONSE_MESSAGE_MARKERS = ("invalid", "malformed", "parse", "decode")


def _contains_any(text: str, markers: Tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _read_result_attribute(result: Any, name: str) -> Any:
    """Read an attribute of a provider result; ``None`` when it cannot be read.

    The SDK result delegates ``text`` and ``tool_calls`` to its first
    alternative, so a response without alternatives fails on access instead of
    returning nothing.
    """
    try:
        return getattr(result, name, None)
    except (IndexError, KeyError, TypeError):
        return None


class YandexAIStudioLLMClient(LLMClient):
    """``LLMClient`` on top of the ``yandex-ai-studio-sdk`` chat-completions domain.

    History contract. ``messages`` is handed to the SDK unchanged, so every item
    has to be one of the three shapes the SDK converts itself:

    - ``{"role": "system" | "user" | "assistant", "text": str}``;
    - the ``response_event`` of an earlier turn of the same request, as returned;
    - ``{"tool_results": [{"name": str, "content": str}, ...]}``, placed directly
      after the response event whose tool calls it answers.

    The SDK pairs a tool result with a tool call by tool name, looking only at the
    response event right before the results. A tool-results event without that
    event is rejected, and when one response event calls the same tool more than
    once, every result of that tool is addressed to the last of those calls.

    Retry policy. An unavailable provider and a timeout are transient and are
    retried ``retry_attempts`` times; rejected credentials and an unusable answer
    are not, because a repeat would fail the same way.
    """

    def __init__(
        self,
        folder_id: str,
        api_key: str = "",
        iam_token: str = "",
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_backoff_seconds: Tuple[float, ...] = DEFAULT_RETRY_BACKOFF_SECONDS,
        sleep: Sleep = asyncio.sleep,
    ):
        backoff = tuple(retry_backoff_seconds)
        if retry_attempts < 0:
            raise ValueError("retry_attempts cannot be negative")
        if retry_attempts > 0 and not backoff:
            raise ValueError("retry_backoff_seconds cannot be empty when retries are enabled")
        if any(seconds < 0 for seconds in backoff):
            raise ValueError("retry_backoff_seconds cannot hold a negative delay")

        self._folder_id = (folder_id or "").strip()
        self._api_key = (api_key or "").strip()
        self._iam_token = (iam_token or "").strip()
        self._retry_attempts = retry_attempts
        self._retry_backoff_seconds = backoff
        self._sleep = sleep
        self._sdk: Optional[Any] = None

    async def run_agent_turn(
        self,
        assistant: Assistant,
        messages: List[Any],
        tools: List[LLMToolDefinition],
    ) -> LLMAgentTurnResult:
        """Run one model turn and return its text or its tool calls.

        The request parameters that depend on the model come from its profile.
        No output-token cap is set: a tight budget can leave a long answer that
        follows a large tool result empty.

        ``response_event`` of the result is the SDK result object itself. The loop
        appends it to the history unchanged; the SDK turns it back into the
        assistant message (its text, or its tool calls with their ids).
        """
        started_at = time.monotonic()
        try:
            self._validate_configuration()

            configure_kwargs: Dict[str, Any] = {"temperature": assistant.temperature}
            native_tools = self._build_native_tools(tools)
            if native_tools:
                configure_kwargs["tools"] = native_tools
            reasoning_params = get_model_profile(assistant.model_name).agent_reasoning_params
            if reasoning_params is not None:
                # The SDK accepts a plain dict only.
                configure_kwargs["extra_query"] = dict(reasoning_params)

            configured_model = self._get_chat_model(assistant.model_name).configure(
                **configure_kwargs
            )
            result = await self._run_model_with_retries(configured_model, messages, assistant)

            tool_calls = self._extract_tool_calls(result)
            text = "" if tool_calls else self._extract_content(result)
            tokens = self._extract_total_tokens(result)
        except Exception as error:
            mapped_error = self._map_exception(error)
            logger.error(
                "Agent turn failed: model=%s duration_ms=%d error=%s.%s mapped_to=%s: %s",
                assistant.model_name,
                _elapsed_ms(started_at),
                type(error).__module__,
                type(error).__name__,
                type(mapped_error).__name__,
                error,
                exc_info=True,
            )
            if mapped_error is error:
                raise
            raise mapped_error from error

        logger.info(
            "Agent turn succeeded: model=%s duration_ms=%d tokens=%d tool_calls=%s text_length=%d",
            assistant.model_name,
            _elapsed_ms(started_at),
            tokens,
            [call.name for call in tool_calls],
            len(text),
        )
        return LLMAgentTurnResult(
            text=text,
            tool_calls=tool_calls,
            tokens=tokens,
            cost=0.0,
            response_event=result,
        )

    async def _run_model_with_retries(
        self, configured_model: Any, messages: List[Any], assistant: Assistant
    ) -> Any:
        """Call the model, repeating the call after a transient failure.

        The backoff before retry ``n`` is ``retry_backoff_seconds[n]``; the last
        value repeats when there are more retries than values. The original error
        is re-raised, so the caller maps and logs the final failure once.
        """
        attempt = 0
        while True:
            try:
                return await configured_model.run(messages)
            except Exception as error:
                mapped_error = self._map_exception(error)
                if (
                    not isinstance(mapped_error, _TRANSIENT_ERRORS)
                    or attempt >= self._retry_attempts
                ):
                    raise
                backoff = self._retry_backoff_seconds[
                    min(attempt, len(self._retry_backoff_seconds) - 1)
                ]
                logger.warning(
                    "Agent turn retry %d/%d: model=%s error=%s mapped_to=%s backoff_seconds=%s",
                    attempt + 1,
                    self._retry_attempts,
                    assistant.model_name,
                    type(error).__name__,
                    type(mapped_error).__name__,
                    backoff,
                )
                await self._sleep(backoff)
                attempt += 1

    def _validate_configuration(self) -> None:
        if not self._folder_id:
            raise LLMServiceAuthenticationError(
                "The Yandex Cloud folder id is not configured (YC_FOLDER_ID)"
            )
        if not self._api_key and not self._iam_token:
            raise LLMServiceAuthenticationError(
                "Neither YC_API_KEY nor YC_IAM_TOKEN is configured for Yandex AI Studio"
            )

    def _get_sdk(self) -> Any:
        """Build the SDK on first use; the API key wins over the IAM token."""
        if AsyncAIStudio is None:
            raise LLMServiceUnavailableError(
                "yandex-ai-studio-sdk is not installed in the current environment"
            )
        if self._sdk is None:
            self._sdk = AsyncAIStudio(
                folder_id=self._folder_id,
                auth=self._api_key or self._iam_token,
            )
        return self._sdk

    def _get_chat_model(self, model_name: str) -> Any:
        chat_domain = getattr(self._get_sdk(), "chat", None)
        if chat_domain is None or not hasattr(chat_domain, "completions"):
            raise LLMServiceUnavailableError(
                "The installed yandex-ai-studio-sdk has no chat.completions API"
            )
        return chat_domain.completions(model_name)

    def _build_native_tools(self, tools: List[LLMToolDefinition]) -> List[Any]:
        if not tools:
            return []
        sdk = self._get_sdk()
        return [
            sdk.tools.function(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                strict=tool.strict,
            )
            for tool in tools
        ]

    def _extract_tool_calls(self, result: Any) -> List[LLMToolCall]:
        extracted: List[LLMToolCall] = []
        for item in _read_result_attribute(result, "tool_calls") or []:
            function_call = getattr(item, "function", None)
            if function_call is None:
                logger.warning("Skipping a tool call that carries no function: %r", item)
                continue

            arguments = getattr(function_call, "arguments", None)
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise LLMServiceInvalidResponseError(
                    f"The model returned tool call arguments that are not a JSON object "
                    f"({type(arguments).__name__})"
                )
            extracted.append(
                LLMToolCall(
                    name=getattr(function_call, "name", ""),
                    arguments=arguments,
                    call_id=getattr(item, "id", None),
                )
            )
        return extracted

    def _extract_content(self, result: Any) -> str:
        direct_text = _read_result_attribute(result, "text")
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text

        try:
            first_alternative = result[0]
        except (TypeError, IndexError, KeyError):
            first_alternative = None
        alternative_text = getattr(first_alternative, "text", None)
        if isinstance(alternative_text, str) and alternative_text.strip():
            return alternative_text

        raise LLMServiceInvalidResponseError(
            "The model returned neither text nor tool calls"
        )

    def _extract_total_tokens(self, result: Any) -> int:
        usage = getattr(result, "usage", None)
        if usage is None:
            return 0
        if isinstance(usage, dict):
            total_tokens = usage.get("total_tokens", 0)
        else:
            total_tokens = getattr(usage, "total_tokens", 0)
        try:
            return int(total_tokens or 0)
        except (TypeError, ValueError):
            return 0

    def _map_exception(self, error: Exception) -> LLMServiceError:
        """Turn any failure of a turn into the ``LLMServiceError`` family.

        An error that is already mapped passes through. Exception classes the
        SDK is known to raise are matched first; everything else is classified
        by its type name and message.
        """
        if isinstance(error, LLMServiceError):
            return error
        return _map_known_exception_class(error) or _map_by_name_and_message(error)


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


def _map_known_exception_class(error: Exception) -> Optional[LLMServiceError]:
    """Map the exception classes of the SDK; ``None`` when the class decides nothing."""
    if isinstance(error, json.JSONDecodeError):
        # The SDK parses the response body and the tool call arguments as JSON
        # and lets the parser error through.
        return LLMServiceInvalidResponseError(
            "The model provider returned a body or tool call arguments that are not valid JSON",
            cause=error,
        )
    if sdk_exceptions is None:
        return None

    if isinstance(error, sdk_exceptions.AioRpcError):
        status = getattr(error.code(), "name", "")
        if status in _RPC_AUTHENTICATION_STATUSES:
            return LLMServiceAuthenticationError(
                "The model provider rejected the credentials", cause=error
            )
        if status in _RPC_TIMEOUT_STATUSES:
            return LLMServiceTimeoutError("The model provider request timed out", cause=error)
        return LLMServiceUnavailableError(
            f"The model provider is unavailable (RPC status {status or 'unknown'})", cause=error
        )
    if isinstance(error, sdk_exceptions.AIStudioError):
        return LLMServiceUnavailableError(
            f"The model provider SDK failed: {error}", cause=error
        )
    return None


def _map_by_name_and_message(error: Exception) -> LLMServiceError:
    """Classify an error the SDK did not wrap by its type name and its message."""
    error_type = type(error).__name__.lower()
    message = str(error).lower()

    if _contains_any(error_type, _TIMEOUT_TYPE_MARKERS) or _contains_any(
        message, _TIMEOUT_MESSAGE_MARKERS
    ):
        return LLMServiceTimeoutError("The model provider request timed out", cause=error)

    if _contains_any(error_type, _AUTHENTICATION_TYPE_MARKERS) or _contains_any(
        message, _AUTHENTICATION_MESSAGE_MARKERS
    ):
        return LLMServiceAuthenticationError(
            "The model provider rejected the credentials", cause=error
        )

    if _contains_any(message, _INVALID_RESPONSE_MESSAGE_MARKERS):
        return LLMServiceInvalidResponseError(
            "The model provider returned a response that cannot be used", cause=error
        )

    return LLMServiceUnavailableError(f"The model provider is unavailable: {error}", cause=error)
