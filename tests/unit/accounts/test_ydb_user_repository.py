from datetime import datetime, timezone

from src.accounts.adapters.persistence.ydb_user_repository import UserMapper, YDBUserRepository
from src.accounts.domain.entities.user import User
from src.shared.domain.value_objects.user_id import UserId


class _ResultSet:
    def __init__(self, rows):
        self.rows = rows
        self.index = 0


class _FakePool:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.calls = []

    async def execute_with_retries(self, query, parameters=None):
        self.calls.append((query, parameters))
        return [_ResultSet(self.rows)]


def test_mapper_round_trips_a_user():
    user = User(UserId.generate(), "Demo", datetime(2026, 9, 1, 8, 30, tzinfo=timezone.utc))
    mapper = UserMapper()
    params = mapper.to_ydb_params(user)

    row = {
        "user_id": params["$user_id"],
        "display_name": params["$display_name"],
        # The SDK returns naive UTC datetimes for Timestamp columns.
        "created_at": params["$created_at"].replace(tzinfo=None),
    }

    assert mapper.to_domain(row) == user


async def test_lookup_is_a_primary_key_read_without_a_secondary_index():
    pool = _FakePool()
    user_id = UserId.generate()

    found = await YDBUserRepository(pool).find_by_id(user_id)

    query, parameters = pool.calls[0]
    assert found is None
    assert "WHERE user_id = $user_id" in query
    assert "VIEW" not in query
    assert parameters["$user_id"].value == str(user_id)


async def test_save_upserts_every_column():
    pool = _FakePool()
    user = User(UserId.generate(), None, datetime(2026, 9, 1, tzinfo=timezone.utc))

    await YDBUserRepository(pool).save(user)

    query, parameters = pool.calls[0]
    assert "UPSERT INTO users (created_at, display_name, user_id)" in query
    assert parameters["$display_name"].value is None
