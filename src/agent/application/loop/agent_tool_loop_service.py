"""The tool-calling loop of the agent.

One call of ``generate_response`` is one user turn. The loop asks the model for a
turn, runs the tool calls the model requested, feeds the results back and repeats
until the model answers in text or the iteration cap is reached. It keeps no
state between calls: everything it needs arrives as arguments, and everything
worth keeping leaves as the returned trace.

The conversation sent to the model is a list of three kinds of items:

* ``{"role": "system" | "user" | "assistant", "text": ...}`` for the system
  message and for stored turns;
* the provider's own response event of an earlier iteration, appended unchanged,
  so the provider gets back everything it produced (tool calls, reasoning);
* ``{"tool_results": [{"name": ..., "content": <JSON text>}]}`` right after the
  response event whose tool calls it answers.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from src.agent.application.loop.system_prompt_composer import SystemPromptComposer
from src.agent.application.tools.dispatcher import ToolDispatcher
from src.agent.application.tools.registry import ToolRegistry
from src.agent.domain.entities.assistant import Assistant
from src.agent.domain.entities.message import Message, MessageRole
from src.agent.ports.llm_client import LLMClient
from src.agent.ports.llm_tooling import (
    AgentFunctionCallTrace,
    AgentGenerateResponseResult,
    AgentIterationTrace,
    AgentRequestDebugTrace,
    LLMToolResult,
)

logger = logging.getLogger(__name__)

# Returned when the model keeps calling tools until the iteration cap.
AGENT_LOOP_FALLBACK_TEXT = (
    "I could not finish this request within the allowed number of steps. Please try again."
)

_SYSTEM_ROLE = "system"


def _role_item(role: str, text: str) -> Dict[str, str]:
    return {"role": role, "text": text}


def _tool_results_item(results: List[LLMToolResult]) -> Dict[str, Any]:
    return {
        "tool_results": [
            {"name": result.name, "content": json.dumps(result.content, ensure_ascii=False)}
            for result in results
        ]
    }


class AgentToolLoopService:
    """Run one user turn: model calls and tool calls until there is an answer."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_dispatcher: ToolDispatcher,
        tool_registry: ToolRegistry,
        system_prompt_composer: SystemPromptComposer,
        max_iterations: int,
        history_limit: int,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if history_limit < 1:
            raise ValueError("history_limit must be at least 1")

        self._llm_client = llm_client
        self._tool_dispatcher = tool_dispatcher
        self._tool_registry = tool_registry
        self._system_prompt_composer = system_prompt_composer
        self._max_iterations = max_iterations
        self._history_limit = history_limit

    async def generate_response(
        self,
        assistant: Assistant,
        history: List[Message],
        new_message: str,
        *,
        owner_user_id: str,
        calendar_block: str,
        conversation_state_block: str,
        memory_pointer: str,
    ) -> AgentGenerateResponseResult:
        """Produce the assistant's reply to ``new_message`` and the trace of how.

        ``history`` holds the stored turns of the chat, oldest first; it may
        already end with ``new_message``. ``owner_user_id`` is the authenticated
        principal and is handed to every tool. The three blocks are rendered by
        the caller and placed into the system message. An ``LLMServiceError``
        raised by the client is not handled here; it ends the turn.
        """
        system_text = self._system_prompt_composer.compose(
            assistant, calendar_block, conversation_state_block, memory_pointer
        )
        conversation = self._build_conversation(system_text, history, new_message)
        # The new user message is the last item; everything after it belongs to
        # this turn.
        current_turn_start = len(conversation) - 1
        tools = self._tool_registry.definitions()

        logger.info(
            "Agent loop started: model=%s stored_turns=%d tools=%d max_iterations=%d",
            assistant.model_name,
            len(history),
            len(tools),
            self._max_iterations,
        )

        total_tokens = 0
        total_cost = 0.0
        request_flow: List[AgentIterationTrace] = []

        for iteration in range(1, self._max_iterations + 1):
            request = self._slice_for_model(conversation, current_turn_start)
            turn = await self._llm_client.run_agent_turn(
                assistant=assistant, messages=request, tools=tools
            )
            total_tokens += turn.tokens
            total_cost += turn.cost

            if turn.tool_calls:
                if turn.response_event is None:
                    # Without the event the next request could not show the
                    # provider which calls the results answer. Nothing is
                    # executed, so no write happens that could not be reported.
                    raise ValueError(
                        "The LLM client returned tool calls without a response event"
                    )
                conversation.append(turn.response_event)

                logger.info(
                    "Agent loop iteration %d: %d tool call(s): %s",
                    iteration,
                    len(turn.tool_calls),
                    [tool_call.name for tool_call in turn.tool_calls],
                )
                results = await self._tool_dispatcher.dispatch(
                    turn.tool_calls, owner_user_id=owner_user_id
                )
                if len(results) != len(turn.tool_calls):
                    raise ValueError(
                        f"The tool dispatcher returned {len(results)} result(s) "
                        f"for {len(turn.tool_calls)} tool call(s)"
                    )

                request_flow.append(
                    AgentIterationTrace(
                        iteration=iteration,
                        tokens=turn.tokens,
                        assistant_text=turn.text,
                        function_calls=[
                            AgentFunctionCallTrace(
                                name=tool_call.name,
                                arguments=tool_call.arguments,
                                result=result.content,
                                call_id=tool_call.call_id,
                            )
                            for tool_call, result in zip(turn.tool_calls, results)
                        ],
                    )
                )
                conversation.append(_tool_results_item(results))
                continue

            if turn.response_event is not None:
                conversation.append(turn.response_event)
            request_flow.append(
                AgentIterationTrace(
                    iteration=iteration,
                    tokens=turn.tokens,
                    assistant_text=turn.text,
                    function_calls=[],
                )
            )

            final_text = (turn.text or "").strip()
            if final_text:
                logger.info(
                    "Agent loop finished: iterations=%d tokens=%d text_length=%d",
                    iteration,
                    total_tokens,
                    len(final_text),
                )
                return self._result(assistant, final_text, total_tokens, total_cost, request_flow)

        logger.warning(
            "Agent loop reached the iteration cap without an answer: "
            "max_iterations=%d tokens=%d",
            self._max_iterations,
            total_tokens,
        )
        return self._result(
            assistant, AGENT_LOOP_FALLBACK_TEXT, total_tokens, total_cost, request_flow
        )

    @staticmethod
    def _build_conversation(
        system_text: str, history: List[Message], new_message: str
    ) -> List[Any]:
        """Return the system message, the stored turns and the new user turn.

        The caller stores the user message before it loads the history, so the
        history normally ends with the very text being answered. It is then not
        added a second time.
        """
        conversation: List[Any] = [_role_item(_SYSTEM_ROLE, system_text)]
        for message in history:
            conversation.append(_role_item(message.role.value, message.content))

        new_text = new_message.strip()
        last = history[-1] if history else None
        already_stored = (
            last is not None
            and last.role == MessageRole.USER
            and last.content.strip() == new_text
        )
        if not already_stored:
            conversation.append(_role_item(MessageRole.USER.value, new_text))
        return conversation

    def _slice_for_model(self, conversation: List[Any], current_turn_start: int) -> List[Any]:
        """Return the part of the conversation that is sent to the model.

        The system message and the whole current turn (the new user message and
        every response event and tool-results item after it) are always sent:
        without the question the request is meaningless, and a tool-results item
        is only valid right after the response event that requested it. Older
        turns fill what is left of ``history_limit``, most recent first, and are
        the only thing the limit ever drops. A new list is returned every time.
        """
        system_item = conversation[0]
        older_turns = conversation[1:current_turn_start]
        current_turn = conversation[current_turn_start:]

        room = max(self._history_limit - len(current_turn), 0)
        kept_older_turns = older_turns[len(older_turns) - min(room, len(older_turns)):]
        return [system_item, *kept_older_turns, *current_turn]

    @staticmethod
    def _result(
        assistant: Assistant,
        text: str,
        total_tokens: int,
        total_cost: float,
        request_flow: List[AgentIterationTrace],
    ) -> AgentGenerateResponseResult:
        return AgentGenerateResponseResult(
            text=text,
            tokens=total_tokens,
            cost=total_cost,
            debug=AgentRequestDebugTrace(
                model_name=assistant.model_name,
                total_tokens=total_tokens,
                request_flow=request_flow,
            ),
        )
