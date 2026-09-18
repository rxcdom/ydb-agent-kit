"""Port through which the agent's tools reach the task data.

The agent module depends on this abstract port only; an adapter wired by the
composition root implements it on top of the tasks module's use cases. No type of
that module crosses the boundary: every method returns a plain envelope ::

    {"status": <outcome>, "data": {...}}

``status`` is one word of the shared outcome vocabulary: ``ok``, ``no_data``,
``coverage_gap``, ``empty_filter``, ``no_records``, ``ambiguous_source``,
``not_found`` or ``filter_error``. ``data`` holds the fields of that outcome.

``owner_user_id`` always comes from the authenticated principal. Projects and
tasks are addressed by human text (a name, a title); the implementation resolves
those references inside the owner's own rows, and an ambiguous reference is
reported as ``ambiguous_source`` instead of being resolved silently. Dates are
ISO ``YYYY-MM-DD`` strings with both window bounds inclusive.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TaskDataProvider(ABC):
    """Read and write access to one owner's projects and tasks."""

    @abstractmethod
    async def list_projects(self, *, owner_user_id: str) -> dict:
        """List the owner's projects with their task counts and activity bounds."""

    @abstractmethod
    async def query_tasks(
        self,
        *,
        owner_user_id: str,
        date_from: str | None,
        date_to: str | None,
        date_field: str,
        project: str | None,
        statuses: list[str] | None,
        priority: list[str] | None,
        text: str | None,
        group_by: str,
        view: str,
    ) -> dict:
        """Query tasks in a window on the chosen date axis.

        ``date_field`` selects the axis the window and the coverage apply to.
        ``project`` is a project reference; ``statuses``, ``priority`` and ``text``
        narrow the result; ``group_by`` and ``view`` shape it.
        """

    @abstractmethod
    async def create_task(
        self,
        *,
        owner_user_id: str,
        title: str,
        project: str | None,
        due_at: str | None,
        priority: str | None,
        notes: str | None,
    ) -> dict:
        """Create a task, optionally filed under the referenced project."""

    @abstractmethod
    async def update_task(
        self,
        *,
        owner_user_id: str,
        task: str,
        project_scope: str | None,
        set_status: str | None,
        set_title: str | None,
        set_due_at: str | None,
        set_priority: str | None,
        set_project: str | None,
        set_notes: str | None,
    ) -> dict:
        """Change the single task matching ``task`` and report what changed.

        ``project_scope`` narrows the title lookup to one project. Each ``set_*``
        argument left as ``None`` keeps the current value. Nothing is changed
        unless the reference resolves to exactly one task.
        """

    @abstractmethod
    async def delete_task(
        self, *, owner_user_id: str, task: str, project_scope: str | None
    ) -> dict:
        """Delete the single task matching ``task``.

        Nothing is deleted unless the reference resolves to exactly one task.
        """
