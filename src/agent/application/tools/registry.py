"""Registry of the tools the agent may call.

A tool is three things that belong together: the definition the model sees, the
arguments model that validates what the model sends, and the executor that does
the work. The loop reads the definitions, the dispatcher looks tools up by name,
and neither of them knows which tools exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from src.agent.ports.llm_tooling import LLMToolDefinition, LLMToolResult

# Calling convention of every executor:
#
#     async def execute(args: <ArgsModel>, *, owner_user_id: str) -> LLMToolResult
#
# ``args`` is the validated arguments model of that tool. ``owner_user_id`` is
# keyword-only and always comes from the authenticated principal.
ToolExecutor = Callable[..., Awaitable[LLMToolResult]]


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool."""

    definition: LLMToolDefinition
    args_model: Type[BaseModel]
    executor: ToolExecutor

    @property
    def name(self) -> str:
        return self.definition.name


class ToolRegistry:
    """Tools by name, in registration order."""

    def __init__(self) -> None:
        self._specs: Dict[str, ToolSpec] = {}

    def register(
        self,
        definition: LLMToolDefinition,
        args_model: Type[BaseModel],
        executor: ToolExecutor,
    ) -> None:
        """Add a tool. A name can be registered only once."""
        name = definition.name
        if not name or not name.strip():
            raise ValueError("A tool definition must have a name")
        if name in self._specs:
            raise ValueError(f"Tool {name!r} is already registered")
        self._specs[name] = ToolSpec(
            definition=definition, args_model=args_model, executor=executor
        )

    def get(self, name: str) -> Optional[ToolSpec]:
        """Return the tool registered under ``name``, or ``None``."""
        return self._specs.get(name)

    def definitions(self) -> List[LLMToolDefinition]:
        """Return the definitions to offer to the model, in registration order."""
        return [spec.definition for spec in self._specs.values()]

    def specs(self) -> List[ToolSpec]:
        """Return every registered tool, in registration order."""
        return list(self._specs.values())
