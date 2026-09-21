"""Argument schemas and model-facing definitions of the agent's tools.

Rules every schema follows:

* ``extra="forbid"``: an argument the tool does not know is an error the model
  can read and repair, not something to ignore.
* Closed vocabularies are ``Literal`` types, so the JSON schema carries an enum.
* Optional arguments default to ``None``; the dispatcher folds an explicit
  ``null`` onto that default before validation.
* No schema has a field that identifies a user, a chat or a stored row. The
  owner comes from the authenticated principal; rows are addressed by the
  words the user used for them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field

from src.agent.ports.llm_tooling import LLMToolDefinition

TaskStatusName = Literal["open", "done", "cancelled"]
TaskPriorityName = Literal["low", "normal", "high"]
DateFieldName = Literal["created", "due", "completed"]
GroupByName = Literal["none", "project", "status", "priority", "week", "month"]
ViewName = Literal["summary", "list"]

ISO_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
ISO_DATE_OR_NONE_PATTERN = r"^(\d{4}-\d{2}-\d{2}|none)$"

LIST_PROJECTS = "list_projects"
QUERY_TASKS = "query_tasks"
CREATE_TASK = "create_task"
UPDATE_TASK = "update_task"
DELETE_TASK = "delete_task"
REMEMBER = "remember"
RECALL = "recall"
FORGET = "forget"


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListProjectsArgs(ToolArgs):
    pass


class QueryTasksArgs(ToolArgs):
    date_from: Optional[str] = Field(
        default=None,
        pattern=ISO_DATE_PATTERN,
        description="First day of the period, inclusive, YYYY-MM-DD. Omit for an open start.",
    )
    date_to: Optional[str] = Field(
        default=None,
        pattern=ISO_DATE_PATTERN,
        description="Last day of the period, inclusive, YYYY-MM-DD. Omit for an open end.",
    )
    date_field: DateFieldName = Field(
        default="created",
        description=(
            "Which date of a task the period applies to: created (when it was added), "
            "due (its deadline), completed (when it was finished)."
        ),
    )
    project: Optional[str] = Field(
        default=None,
        description="The project as the user named it (name or part of it). Omit for all projects.",
    )
    statuses: Optional[List[TaskStatusName]] = Field(
        default=None, description="Keep only tasks in these statuses. Omit for all statuses."
    )
    priority: Optional[List[TaskPriorityName]] = Field(
        default=None, description="Keep only tasks with these priorities. Omit for all priorities."
    )
    text: Optional[str] = Field(
        default=None,
        description="Keep only tasks whose title or notes contain this text (case-insensitive).",
    )
    group_by: GroupByName = Field(
        default="none", description="Break the counts down by this dimension."
    )
    view: ViewName = Field(
        default="summary",
        description="summary returns counts; list returns the tasks themselves.",
    )


class CreateTaskArgs(ToolArgs):
    title: str = Field(min_length=1, max_length=200, description="Title of the new task.")
    project: Optional[str] = Field(
        default=None,
        description="An existing project as the user named it. Omit to leave the task unfiled.",
    )
    due_at: Optional[str] = Field(
        default=None, pattern=ISO_DATE_PATTERN, description="Deadline day, YYYY-MM-DD."
    )
    priority: Optional[TaskPriorityName] = Field(
        default=None, description="Omit for normal priority."
    )
    notes: Optional[str] = Field(default=None, max_length=2000, description="Free-form notes.")


class UpdateTaskArgs(ToolArgs):
    task: str = Field(
        min_length=1,
        description="The task as the user named it: its title or a distinctive part of it.",
    )
    project_scope: Optional[str] = Field(
        default=None,
        description="Search for the task only inside this project (as the user named it).",
    )
    set_status: Optional[TaskStatusName] = Field(
        default=None, description="done finishes the task, open reopens it, cancelled cancels it."
    )
    set_title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    set_due_at: Optional[str] = Field(
        default=None,
        pattern=ISO_DATE_OR_NONE_PATTERN,
        description='New deadline day, YYYY-MM-DD, or the word "none" to remove the deadline.',
    )
    set_priority: Optional[TaskPriorityName] = None
    set_project: Optional[str] = Field(
        default=None,
        description='Move the task to this existing project, or "none" to unfile it.',
    )
    set_notes: Optional[str] = Field(default=None, max_length=2000)


class DeleteTaskArgs(ToolArgs):
    task: str = Field(
        min_length=1,
        description="The task as the user named it: its title or a distinctive part of it.",
    )
    project_scope: Optional[str] = Field(
        default=None,
        description="Search for the task only inside this project (as the user named it).",
    )


class RememberArgs(ToolArgs):
    content: str = Field(
        min_length=1,
        max_length=1000,
        description="The fact or preference, as one self-contained sentence.",
    )
    topic: Optional[str] = Field(
        default=None,
        max_length=100,
        description="A short label of two or three words, for example: working hours.",
    )


class RecallArgs(ToolArgs):
    query: Optional[str] = Field(
        default=None,
        description="Words to look for. Omit to get the most recent notes.",
    )


class ForgetArgs(ToolArgs):
    query: str = Field(
        min_length=1, description="Words that identify the note or notes to delete."
    )


def schema_from_model(model: Type[BaseModel]) -> Dict[str, Any]:
    """JSON schema of an arguments model, as it is sent to the provider."""
    return model.model_json_schema()


def _definition(name: str, description: str, model: Type[BaseModel]) -> LLMToolDefinition:
    return LLMToolDefinition(
        name=name, description=description, parameters=schema_from_model(model)
    )


LIST_PROJECTS_DEFINITION = _definition(
    LIST_PROJECTS,
    "List the user's projects. For each project: name, description, number of open and of "
    "finished tasks, and the first and last day with activity. Tasks that belong to no project "
    "are reported as one extra entry called unfiled. Use it for questions about which projects "
    "exist or how work is spread across them. Takes no arguments. "
    "Statuses: ok, or no_data when the user has neither projects nor tasks.",
    ListProjectsArgs,
)

QUERY_TASKS_DEFINITION = _definition(
    QUERY_TASKS,
    "Read the user's tasks. This is the only source of facts about tasks.\n"
    "A task has three independent dates and one call looks at exactly one of them (date_field):\n"
    "- created: when the task was added. Default; use it for 'what did I add' and for general "
    "questions about what exists.\n"
    "- completed: when the task was finished. Use it for 'what did I finish'. Only finished "
    "tasks have this date, so the result contains finished tasks only; combining it with "
    "statuses that leave out done is a contradiction and returns filter_error.\n"
    "- due: the deadline. Use it for 'what is due' and 'what is overdue'. Tasks without a "
    "deadline are left out, and excluded_without_date says how many.\n"
    "date_from and date_to are inclusive local days. Copy them from the Calendar block of the "
    "system message; never calculate them. Omit both for all time. Overdue is: date_field due, "
    "date_to = yesterday, no date_from, statuses ['open'].\n"
    "project is the user's wording for a project; it is matched against the existing projects. "
    "view summary returns counts (add group_by for a breakdown); view list returns the tasks.\n"
    "Statuses: ok (with window_used, coverage and the data); no_data (the user has no tasks); "
    "coverage_gap (the period lies entirely outside coverage, the dates the data spans on this "
    "date_field); no_records (the period is inside coverage and empty); empty_filter (the period "
    "has tasks but project, statuses, priority or text matched none; see count_without_filters, "
    "available_projects, available_statuses); ambiguous_source (project matches several "
    "projects; show matched to the user and ask, never choose); filter_error (fix the arguments).",
    QueryTasksArgs,
)

CREATE_TASK_DEFINITION = _definition(
    CREATE_TASK,
    "Add one new task. Use it only when the user explicitly asks to add or create a task. "
    "project must refer to an existing project; projects cannot be created here. "
    "Statuses: ok (created holds the new task); ambiguous_source (project matches several "
    "projects and nothing was created; show matched and ask); not_found (no such project and "
    "nothing was created; available_projects lists the existing ones); filter_error.",
    CreateTaskArgs,
)

UPDATE_TASK_DEFINITION = _definition(
    UPDATE_TASK,
    "Change one existing task: status, title, deadline, priority, project or notes. Use it only "
    "when the user explicitly asks for a change. Finishing a task is set_status done. Identify "
    "the task by the user's words in task; add project_scope when the user named the project. "
    "Pass only the set_ fields that should change. "
    "Statuses: ok (updated holds the task and changed lists every modified field with its old "
    "and new value; an empty changed list means nothing was modified); ambiguous_source "
    "(several tasks match and nothing was changed; show matched and ask, never choose); "
    "not_found (no task matches and nothing was changed); filter_error.",
    UpdateTaskArgs,
)

DELETE_TASK_DEFINITION = _definition(
    DELETE_TASK,
    "Permanently delete one task. Use it only when the user explicitly asks to delete or remove "
    "a task. Identify the task by the user's words in task; add project_scope when the user "
    "named the project. The task is deleted only when exactly one task matches. "
    "Statuses: ok (deleted holds the removed task); ambiguous_source (several tasks match and "
    "nothing was deleted; show matched and ask, never choose); not_found (no task matches and "
    "nothing was deleted).",
    DeleteTaskArgs,
)

REMEMBER_DEFINITION = _definition(
    REMEMBER,
    "Store one lasting fact or preference about the user for future conversations. Use it when "
    "the user asks you to remember or keep something in mind, or states a durable preference, "
    "habit or constraint, even in passing inside a message that is mainly about something else "
    "(then call this tool and the other tools the message needs in the same turn). Never store "
    "secrets (passwords, codes, keys, card numbers); such content is rejected. A note on the "
    "same topic is updated instead of duplicated. "
    "Result: status ok with action created or updated, or status rejected with a reason.",
    RememberArgs,
)

RECALL_DEFINITION = _definition(
    RECALL,
    "Read the user's stored notes. Use it when the user asks what you remember or know about "
    "them, or when a stored preference could matter for the answer. "
    "Result: count and memories. A count of 0 means nothing is stored.",
    RecallArgs,
)

FORGET_DEFINITION = _definition(
    FORGET,
    "Delete the stored notes that match the given words. Use it only when the user asks you to "
    "forget something. Result: deleted_count; 0 means no note matched and nothing was deleted.",
    ForgetArgs,
)

ALL_ARGS_MODELS: Dict[str, Type[BaseModel]] = {
    LIST_PROJECTS: ListProjectsArgs,
    QUERY_TASKS: QueryTasksArgs,
    CREATE_TASK: CreateTaskArgs,
    UPDATE_TASK: UpdateTaskArgs,
    DELETE_TASK: DeleteTaskArgs,
    REMEMBER: RememberArgs,
    RECALL: RecallArgs,
    FORGET: ForgetArgs,
}
