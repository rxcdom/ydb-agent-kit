"""``CreateProjectUseCase``: the only way a project comes to exist besides the demo seed."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.shared.domain.exceptions import ValidationError
from src.shared.domain.value_objects.user_id import UserId
from src.tasks.application.create_project import CreateProjectUseCase
from src.tasks.domain.entities.project import (
    MAX_PROJECT_DESCRIPTION_LENGTH,
    MAX_PROJECT_NAME_LENGTH,
)
from src.tasks.domain.exceptions import InvalidProjectError
from tests.unit.tasks.builders import NOW
from tests.unit.tasks.in_memory import InMemoryTasksRepositoryManager

OWNER = UserId.generate()
STRANGER = UserId.generate()


def _clock() -> datetime:
    return NOW


@pytest.fixture
def repositories() -> InMemoryTasksRepositoryManager:
    return InMemoryTasksRepositoryManager()


async def test_creates_and_stores_a_project_of_the_owner(repositories):
    project = await CreateProjectUseCase(repositories, _clock).execute(
        OWNER, "  Home renovation ", "  Kitchen first  "
    )

    assert project.name == "Home renovation"
    assert project.description == "Kitchen first"
    assert project.user_id == OWNER
    assert project.created_at == project.updated_at == NOW
    assert await repositories.projects.find_by_id(OWNER, project.project_id) == project
    assert repositories.write_log == [("save_project", "Home renovation")]


@pytest.mark.parametrize("description", [None, "", "   "])
async def test_a_missing_or_blank_description_is_stored_as_none(repositories, description):
    project = await CreateProjectUseCase(repositories, _clock).execute(
        OWNER, "Garden", description
    )

    assert project.description is None


async def test_the_description_is_optional(repositories):
    project = await CreateProjectUseCase(repositories, _clock).execute(OWNER, "Garden")

    assert project.description is None


async def test_names_at_the_limits_are_accepted(repositories):
    use_case = CreateProjectUseCase(repositories, _clock)

    shortest = await use_case.execute(OWNER, "G")
    longest = await use_case.execute(
        OWNER, "n" * MAX_PROJECT_NAME_LENGTH, "d" * MAX_PROJECT_DESCRIPTION_LENGTH
    )

    assert len(shortest.name) == 1
    assert len(longest.name) == MAX_PROJECT_NAME_LENGTH
    assert len(longest.description) == MAX_PROJECT_DESCRIPTION_LENGTH


@pytest.mark.parametrize(
    "name, description",
    [
        ("", None),
        ("   ", None),
        ("n" * (MAX_PROJECT_NAME_LENGTH + 1), None),
        ("Garden", "d" * (MAX_PROJECT_DESCRIPTION_LENGTH + 1)),
    ],
)
async def test_invalid_attributes_are_a_validation_error_and_nothing_is_stored(
    repositories, name, description
):
    with pytest.raises(InvalidProjectError) as raised:
        await CreateProjectUseCase(repositories, _clock).execute(OWNER, name, description)

    assert isinstance(raised.value, ValidationError)
    assert repositories.write_log == []
    assert await repositories.projects.list_by_user(OWNER) == []


async def test_names_may_collide_because_identity_is_the_id(repositories):
    use_case = CreateProjectUseCase(repositories, _clock)

    first = await use_case.execute(OWNER, "Garden")
    second = await use_case.execute(OWNER, "Garden")

    assert first.project_id != second.project_id
    assert len(await repositories.projects.list_by_user(OWNER)) == 2


async def test_a_project_belongs_to_its_creator_only(repositories):
    project = await CreateProjectUseCase(repositories, _clock).execute(OWNER, "Garden")

    assert await repositories.projects.find_by_id(STRANGER, project.project_id) is None
    assert await repositories.projects.list_by_user(STRANGER) == []


async def test_the_default_clock_is_the_current_utc_instant(repositories):
    before = datetime.now(timezone.utc)

    project = await CreateProjectUseCase(repositories).execute(OWNER, "Garden")

    assert project.created_at.tzinfo == timezone.utc
    assert before <= project.created_at <= before + timedelta(minutes=1)
