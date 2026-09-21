from abc import ABC, abstractmethod
from typing import Any, List, Optional
from uuid import UUID

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.domain.entities.project import Project


class ProjectRepository(ABC):
    """Owner-scoped storage of projects. No method can reach another owner's row."""

    @abstractmethod
    async def save(self, project: Project, tx: Any = None) -> None:
        """Insert or replace the project, inside ``tx`` when one is given."""

    @abstractmethod
    async def find_by_id(self, user_id: UserId, project_id: UUID) -> Optional[Project]:
        """The owner's project with this id; ``None`` when the owner has no such project."""

    @abstractmethod
    async def list_by_user(self, user_id: UserId) -> List[Project]:
        """Every project of the owner, ordered by name."""
