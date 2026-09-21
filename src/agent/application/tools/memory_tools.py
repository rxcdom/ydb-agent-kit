"""Executors of the long-term memory tools: ``remember``, ``recall``, ``forget``.

Each executor hands the outcome back to the model as the tool result, and the
model words the confirmation for the user. Every operation is bound to
``owner_user_id``, the authenticated principal: nothing the model sends can name
another user's vault, and results carry no row identifiers.
"""
from __future__ import annotations

from typing import Dict, Optional

from src.agent.application.memory.user_memory_writer import (
    REMEMBER_REJECTED,
    UserMemoryWriter,
)
from src.agent.application.tools.registry import ToolExecutor
from src.agent.application.tools.schemas import (
    FORGET,
    RECALL,
    REMEMBER,
    ForgetArgs,
    RecallArgs,
    RememberArgs,
)
from src.agent.ports.llm_tooling import LLMToolResult
from src.shared.domain.value_objects.user_id import UserId

MEMORY_TOOL_NAMES = (REMEMBER, RECALL, FORGET)

MEMORY_UNAVAILABLE_ERROR = "memory_unavailable"

_REJECTION_NOTE = (
    "Nothing was stored. Secrets (passwords, codes, keys, card numbers) and "
    "entries that are too short are never kept in memory."
)


def _remember_executor(writer: UserMemoryWriter) -> ToolExecutor:
    async def execute(args: RememberArgs, *, owner_user_id: str) -> LLMToolResult:
        outcome = await writer.remember(
            UserId.from_string(owner_user_id), content=args.content, topic=args.topic
        )
        if outcome.status == REMEMBER_REJECTED:
            return LLMToolResult(
                name=REMEMBER,
                content={
                    "status": "rejected",
                    "reason": outcome.reason,
                    "note": _REJECTION_NOTE,
                },
            )
        return LLMToolResult(
            name=REMEMBER,
            content={
                "status": "ok",
                "action": outcome.status,
                "evicted_oldest": outcome.evicted,
            },
        )

    return execute


def _recall_executor(writer: UserMemoryWriter) -> ToolExecutor:
    async def execute(args: RecallArgs, *, owner_user_id: str) -> LLMToolResult:
        memories = await writer.recall(UserId.from_string(owner_user_id), query=args.query)
        return LLMToolResult(
            name=RECALL,
            content={
                "status": "ok",
                "count": len(memories),
                "memories": [
                    {
                        "content": memory.content,
                        "topic": memory.topic,
                        "updated_at": memory.updated_at.isoformat(),
                    }
                    for memory in memories
                ],
            },
        )

    return execute


def _forget_executor(writer: UserMemoryWriter) -> ToolExecutor:
    async def execute(args: ForgetArgs, *, owner_user_id: str) -> LLMToolResult:
        outcome = await writer.forget(UserId.from_string(owner_user_id), query=args.query)
        return LLMToolResult(
            name=FORGET, content={"status": "ok", "deleted_count": outcome.deleted_count}
        )

    return execute


def _unavailable_executor(tool_name: str) -> ToolExecutor:
    async def execute(args: object, *, owner_user_id: str) -> LLMToolResult:
        return LLMToolResult(name=tool_name, content={"error": MEMORY_UNAVAILABLE_ERROR})

    return execute


def build_memory_tool_executors(
    writer: Optional[UserMemoryWriter],
) -> Dict[str, ToolExecutor]:
    """Return the executor of every memory tool, keyed by tool name.

    Without a writer the tools stay registered and answer with
    ``{"error": "memory_unavailable"}``, so the model learns that memory is out
    of service instead of meeting an unknown tool.
    """
    if writer is None:
        return {name: _unavailable_executor(name) for name in MEMORY_TOOL_NAMES}
    return {
        REMEMBER: _remember_executor(writer),
        RECALL: _recall_executor(writer),
        FORGET: _forget_executor(writer),
    }
