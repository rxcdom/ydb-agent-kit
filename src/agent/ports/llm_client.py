from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from src.agent.domain.entities.assistant import Assistant
from src.agent.ports.llm_tooling import LLMAgentTurnResult, LLMToolDefinition


class LLMClient(ABC):
    """Provider-agnostic interface of a tool-calling language model."""

    @abstractmethod
    async def run_agent_turn(
        self,
        assistant: Assistant,
        messages: List[Any],
        tools: List[LLMToolDefinition],
    ) -> LLMAgentTurnResult:
        """Run one model turn of the agent loop.

        ``assistant`` supplies the model name and generation settings.
        ``messages`` is the request-scoped history the loop built: role/text
        dictionaries, earlier provider response events, and tool-result events.
        ``tools`` are the definitions to bind for this turn.

        The result holds either final text or the tool calls the loop has to
        execute. Provider and transport failures are raised as
        ``LLMServiceError`` subclasses, never as provider-specific exceptions.
        """
