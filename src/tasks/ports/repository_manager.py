from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, List, Sequence

from src.tasks.ports.project_repository import ProjectRepository
from src.tasks.ports.task_repository import TaskRepository

# Receives the transaction handle and passes it on to the repository calls it makes.
TransactionalOperation = Callable[[Any], Awaitable[Any]]


class TasksRepositoryManager(ABC):
    """The repositories of the tasks module plus the way to write through them atomically."""

    @property
    @abstractmethod
    def projects(self) -> ProjectRepository:
        """Storage of the owner's projects."""

    @property
    @abstractmethod
    def tasks(self) -> TaskRepository:
        """Storage of the owner's tasks."""

    @abstractmethod
    async def execute_in_transaction(
        self, operations: Sequence[TransactionalOperation]
    ) -> List[Any]:
        """Run the operations in order in one transaction: all of them commit or none does.

        An operation may run more than once when the transaction is retried, so
        it has to be safe to repeat.
        """
