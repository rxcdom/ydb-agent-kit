"""Executors of the task tools.

Each executor forwards the validated arguments to the ``TaskDataProvider`` port
and returns the provider's envelope ``{"status": ..., "data": {...}}`` unchanged
as the tool result. Which outcome applies (an empty window, an ambiguous
reference, a refused write) is decided behind the port, next to the data; this
module adds nothing to it and hides nothing from the model.

``owner_user_id`` is the authenticated principal and is passed to the port as a
keyword. No argument of any tool can name a user or a stored row.
"""
from __future__ import annotations

from typing import Dict, Optional

from src.agent.application.tools.registry import ToolExecutor
from src.agent.application.tools.schemas import (
    CREATE_TASK,
    DELETE_TASK,
    LIST_PROJECTS,
    QUERY_TASKS,
    UPDATE_TASK,
    CreateTaskArgs,
    DeleteTaskArgs,
    ListProjectsArgs,
    QueryTasksArgs,
    UpdateTaskArgs,
)
from src.agent.ports.llm_tooling import LLMToolResult
from src.agent.ports.task_data_provider import TaskDataProvider

TASK_TOOL_NAMES = (LIST_PROJECTS, QUERY_TASKS, CREATE_TASK, UPDATE_TASK, DELETE_TASK)

TASK_DATA_UNAVAILABLE_ERROR = "task_data_unavailable"


def _list_projects_executor(provider: TaskDataProvider) -> ToolExecutor:
    async def execute(args: ListProjectsArgs, *, owner_user_id: str) -> LLMToolResult:
        envelope = await provider.list_projects(owner_user_id=owner_user_id)
        return LLMToolResult(name=LIST_PROJECTS, content=envelope)

    return execute


def _query_tasks_executor(provider: TaskDataProvider) -> ToolExecutor:
    async def execute(args: QueryTasksArgs, *, owner_user_id: str) -> LLMToolResult:
        envelope = await provider.query_tasks(
            owner_user_id=owner_user_id,
            date_from=args.date_from,
            date_to=args.date_to,
            date_field=args.date_field,
            project=args.project,
            statuses=args.statuses,
            priority=args.priority,
            text=args.text,
            group_by=args.group_by,
            view=args.view,
        )
        return LLMToolResult(name=QUERY_TASKS, content=envelope)

    return execute


def _create_task_executor(provider: TaskDataProvider) -> ToolExecutor:
    async def execute(args: CreateTaskArgs, *, owner_user_id: str) -> LLMToolResult:
        envelope = await provider.create_task(
            owner_user_id=owner_user_id,
            title=args.title,
            project=args.project,
            due_at=args.due_at,
            priority=args.priority,
            notes=args.notes,
        )
        return LLMToolResult(name=CREATE_TASK, content=envelope)

    return execute


def _update_task_executor(provider: TaskDataProvider) -> ToolExecutor:
    async def execute(args: UpdateTaskArgs, *, owner_user_id: str) -> LLMToolResult:
        envelope = await provider.update_task(
            owner_user_id=owner_user_id,
            task=args.task,
            project_scope=args.project_scope,
            set_status=args.set_status,
            set_title=args.set_title,
            set_due_at=args.set_due_at,
            set_priority=args.set_priority,
            set_project=args.set_project,
            set_notes=args.set_notes,
        )
        return LLMToolResult(name=UPDATE_TASK, content=envelope)

    return execute


def _delete_task_executor(provider: TaskDataProvider) -> ToolExecutor:
    async def execute(args: DeleteTaskArgs, *, owner_user_id: str) -> LLMToolResult:
        envelope = await provider.delete_task(
            owner_user_id=owner_user_id, task=args.task, project_scope=args.project_scope
        )
        return LLMToolResult(name=DELETE_TASK, content=envelope)

    return execute


def _unavailable_executor(tool_name: str) -> ToolExecutor:
    async def execute(args: object, *, owner_user_id: str) -> LLMToolResult:
        return LLMToolResult(name=tool_name, content={"error": TASK_DATA_UNAVAILABLE_ERROR})

    return execute


def build_task_tool_executors(
    provider: Optional[TaskDataProvider],
) -> Dict[str, ToolExecutor]:
    """Return the executor of every task tool, keyed by tool name.

    Without a provider the tools stay registered and answer with
    ``{"error": "task_data_unavailable"}``, so the model learns that the data is
    out of reach instead of meeting an unknown tool.
    """
    if provider is None:
        return {name: _unavailable_executor(name) for name in TASK_TOOL_NAMES}
    return {
        LIST_PROJECTS: _list_projects_executor(provider),
        QUERY_TASKS: _query_tasks_executor(provider),
        CREATE_TASK: _create_task_executor(provider),
        UPDATE_TASK: _update_task_executor(provider),
        DELETE_TASK: _delete_task_executor(provider),
    }
