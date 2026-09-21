from enum import Enum
from typing import FrozenSet

from src.tasks.domain.value_objects.task_status import TaskStatus


class DateAxis(str, Enum):
    """The task timestamp a time question is about.

    A task carries three independent instants. "When did I add it", "when is it
    due" and "when did I finish it" are different questions over the same rows,
    so every windowed query names its axis explicitly.
    """

    CREATED = "created"
    DUE = "due"
    COMPLETED = "completed"

    @property
    def statuses_with_value(self) -> FrozenSet[TaskStatus]:
        """Statuses whose tasks can carry an instant on this axis.

        Only a finished task has a completion instant, so the ``COMPLETED`` axis
        is about finished tasks and nothing else. Any task can have a creation
        instant or a due instant.
        """
        if self is DateAxis.COMPLETED:
            return frozenset({TaskStatus.DONE})
        return frozenset(TaskStatus)
