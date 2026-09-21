from __future__ import annotations

from typing import Iterable, Optional

from src.tasks.domain.entities.project import Project
from src.tasks.domain.entities.task import Task
from src.tasks.domain.services.reference_resolution import Resolution, resolve_reference


class TaskReferenceResolver:
    """Finds the task a piece of text refers to among one owner's tasks.

    Every write the agent performs goes through this resolver, so a write can
    only ever reach a task the reference identifies without doubt.
    """

    @staticmethod
    def resolve(
        tasks: Iterable[Task],
        reference: Optional[str],
        project_scope: Optional[Project] = None,
    ) -> Resolution[Task]:
        """An exact title wins; otherwise the text may appear in a title or in notes.

        With a ``project_scope`` only that project's tasks take part, in both
        tiers: an exact title elsewhere does not outrank a partial match inside
        the scope.
        """
        in_scope = (
            tasks
            if project_scope is None
            else [task for task in tasks if task.project_id == project_scope.project_id]
        )
        return resolve_reference(
            in_scope,
            reference,
            primary_text=lambda task: task.title,
            searchable_texts=lambda task: (task.title, task.notes),
        )
