"""Provider-neutral data shapes shared by the LLM client, the loop and the tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LLMToolDefinition:
    """A tool as it is offered to the model: name, description, JSON schema."""

    name: str
    description: str
    parameters: Dict[str, Any]
    strict: bool = True


@dataclass(frozen=True)
class LLMToolCall:
    """A tool invocation requested by the model."""

    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass(frozen=True)
class LLMToolResult:
    """The outcome of one executed tool call, sent back to the model as JSON."""

    name: str
    content: Dict[str, Any]


@dataclass(frozen=True)
class LLMAgentTurnResult:
    """One model turn: either final text or tool calls to execute locally.

    ``response_event`` is the provider's native result object. The loop appends
    it to the history unchanged, so the next request round-trips everything the
    provider needs (including any reasoning content).
    """

    text: str
    tool_calls: List[LLMToolCall] = field(default_factory=list)
    tokens: int = 0
    cost: float = 0.0
    response_event: Any = None


@dataclass(frozen=True)
class AgentFunctionCallTrace:
    """Trace of one executed tool call: what was asked and what came back."""

    name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass(frozen=True)
class AgentIterationTrace:
    """Trace of one loop iteration."""

    iteration: int
    tokens: int
    assistant_text: str
    function_calls: List[AgentFunctionCallTrace] = field(default_factory=list)


@dataclass(frozen=True)
class AgentRequestDebugTrace:
    """Trace of one whole response generation: every iteration in order."""

    model_name: str
    total_tokens: int
    request_flow: List[AgentIterationTrace] = field(default_factory=list)


@dataclass(frozen=True)
class AgentGenerateResponseResult:
    """What the agent loop returns: the reply, its usage and its trace."""

    text: str
    tokens: int
    cost: float
    debug: AgentRequestDebugTrace
