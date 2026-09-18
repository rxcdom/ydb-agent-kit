import pytest
import ydb

from src.shared.infrastructure.database.ydb import connection as connection_module
from src.shared.infrastructure.database.ydb.settings import YDBSettings


class _FakeDriver:
    def __init__(self, failures: int):
        self._failures = failures
        self.wait_calls = 0
        self.stopped = False

    async def wait(self, timeout, fail_fast):
        self.wait_calls += 1
        if self.wait_calls <= self._failures:
            raise TimeoutError("not ready")

    def discovery_debug_details(self):
        return "no endpoints"

    async def stop(self):
        self.stopped = True


class _FakePool:
    def __init__(self, driver, size):
        self.driver = driver
        self.size = size
        self.stopped = False

    async def stop(self):
        self.stopped = True


@pytest.fixture
def patched_sdk(monkeypatch):
    created = {}

    def install(failures: int):
        def make_driver(config):
            created["config"] = config
            created["driver"] = _FakeDriver(failures)
            return created["driver"]

        monkeypatch.setattr(ydb.aio, "Driver", make_driver)
        monkeypatch.setattr(ydb.aio, "QuerySessionPool", _FakePool)
        return created

    return install


async def _no_sleep(_seconds):
    return None


async def test_connection_retries_until_the_driver_is_ready(patched_sdk):
    created = patched_sdk(failures=2)
    settings = YDBSettings(connect_attempts=3, pool_size=7)

    connection = await connection_module.open_ydb_connection(settings, sleep=_no_sleep)

    assert created["driver"].wait_calls == 3
    assert connection.pool.size == 7
    assert connection.database == "/local"


async def test_driver_is_stopped_when_it_never_becomes_ready(patched_sdk):
    created = patched_sdk(failures=5)
    settings = YDBSettings(connect_attempts=2)

    with pytest.raises(ConnectionError, match="after 2 attempts"):
        await connection_module.open_ydb_connection(settings, sleep=_no_sleep)

    assert created["driver"].stopped is True


async def test_close_stops_the_pool_before_the_driver(patched_sdk):
    patched_sdk(failures=0)
    connection = await connection_module.open_ydb_connection(YDBSettings(), sleep=_no_sleep)

    await connection.close()

    assert connection.pool.stopped is True
    assert connection.driver.stopped is True
