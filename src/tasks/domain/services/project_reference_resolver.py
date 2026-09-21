from __future__ import annotations

from typing import Iterable, Optional

from src.tasks.domain.entities.project import Project
from src.tasks.domain.services.reference_resolution import Resolution, resolve_reference


class ProjectReferenceResolver:
    """Finds the project a piece of text refers to among one owner's projects."""

    @staticmethod
    def resolve(projects: Iterable[Project], reference: Optional[str]) -> Resolution[Project]:
        """An exact name wins; otherwise the text may appear in a name or a description."""
        return resolve_reference(
            projects,
            reference,
            primary_text=lambda project: project.name,
            searchable_texts=lambda project: (project.name, project.description),
        )
