"""The agent's tools: schemas, executors, registry and dispatcher."""
from __future__ import annotations

from typing import Optional, Tuple

from src.agent.application.memory.user_memory_writer import UserMemoryWriter
from src.agent.application.tools.memory_tools import build_memory_tool_executors
from src.agent.application.tools.registry import ToolRegistry
from src.agent.application.tools.schemas import (
    ALL_ARGS_MODELS,
    CREATE_TASK_DEFINITION,
    DELETE_TASK_DEFINITION,
    FORGET_DEFINITION,
    LIST_PROJECTS_DEFINITION,
    QUERY_TASKS_DEFINITION,
    RECALL_DEFINITION,
    REMEMBER_DEFINITION,
    UPDATE_TASK_DEFINITION,
)
from src.agent.application.tools.task_tools import build_task_tool_executors
from src.agent.ports.llm_tooling import LLMToolDefinition
from src.agent.ports.task_data_provider import TaskDataProvider

# The order in which the tools are offered to the model: read, write, memory.
TOOL_DEFINITIONS: Tuple[LLMToolDefinition, ...] = (
    LIST_PROJECTS_DEFINITION,
    QUERY_TASKS_DEFINITION,
    CREATE_TASK_DEFINITION,
    UPDATE_TASK_DEFINITION,
    DELETE_TASK_DEFINITION,
    REMEMBER_DEFINITION,
    RECALL_DEFINITION,
    FORGET_DEFINITION,
)


def build_tool_registry(
    task_data_provider: Optional[TaskDataProvider],
    user_memory_writer: Optional[UserMemoryWriter],
) -> ToolRegistry:
    """Register every tool of the agent.

    A missing collaborator does not remove its tools: they stay registered and
    answer with an error result, so the model can tell the user what is out of
    service.
    """
    executors = {
        **build_task_tool_executors(task_data_provider),
        **build_memory_tool_executors(user_memory_writer),
    }
    registry = ToolRegistry()
    for definition in TOOL_DEFINITIONS:
        registry.register(
            definition, ALL_ARGS_MODELS[definition.name], executors[definition.name]
        )
    return registry
