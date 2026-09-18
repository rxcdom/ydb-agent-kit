from datetime import datetime, timezone

import pytest

from src.accounts.application.create_user import CreateUserUseCase
from src.accounts.application.resolve_principal import ResolvePrincipalUseCase
from src.accounts.domain.entities.user import User
from src.accounts.domain.exceptions import InvalidCredentialsError, InvalidUserError
from src.accounts.ports.user_repository import UserRepository
from src.shared.domain.value_objects.user_id import UserId


class InMemoryUserRepository(UserRepository):
    def __init__(self):
        self.rows = {}

    async def save(self, user: User) -> None:
        self.rows[user.user_id] = user

    async def find_by_id(self, user_id: UserId):
        return self.rows.get(user_id)


@pytest.fixture
def repository():
    return InMemoryUserRepository()


async def test_created_user_resolves_from_its_own_id_on_the_next_call(repository):
    fixed_now = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    user = await CreateUserUseCase(repository, clock=lambda: fixed_now).execute("  Demo  ")

    principal = await ResolvePrincipalUseCase(repository).execute(str(user.user_id))

    assert principal.user_id == user.user_id
    assert user.display_name == "Demo"
    assert user.created_at == fixed_now


async def test_each_user_gets_a_distinct_credential(repository):
    use_case = CreateUserUseCase(repository)

    first = await use_case.execute()
    second = await use_case.execute()

    assert first.user_id != second.user_id


@pytest.mark.parametrize("credential", [None, "", "   ", "not-a-uuid", "1234"])
async def test_absent_or_malformed_credential_is_rejected(repository, credential):
    with pytest.raises(InvalidCredentialsError):
        await ResolvePrincipalUseCase(repository).execute(credential)


async def test_well_formed_but_unknown_credential_is_rejected(repository):
    with pytest.raises(InvalidCredentialsError, match="unknown"):
        await ResolvePrincipalUseCase(repository).execute(str(UserId.generate()))


def test_user_rejects_naive_timestamps_and_oversized_names():
    with pytest.raises(InvalidUserError, match="timezone-aware"):
        User(UserId.generate(), None, datetime(2026, 9, 1))
    with pytest.raises(InvalidUserError, match="cannot exceed"):
        User(UserId.generate(), "x" * 81, datetime(2026, 9, 1, tzinfo=timezone.utc))
