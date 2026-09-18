import pytest

from src.shared.infrastructure.database.ydb.settings import YDBSettings


def test_defaults_target_the_compose_network():
    settings = YDBSettings.from_env({})

    assert settings.endpoint == "grpc://ydb:2136"
    assert settings.database == "/local"
    assert settings.connect_attempts == 12
    assert settings.connect_timeout_seconds == 5.0


def test_values_are_read_from_the_environment():
    settings = YDBSettings.from_env(
        {
            "YDB_ENDPOINT": "grpc://localhost:2136",
            "YDB_DATABASE": "/local",
            "YDB_CONNECT_ATTEMPTS": "3",
            "YDB_CONNECT_TIMEOUT_SECONDS": "1.5",
        }
    )

    assert settings.endpoint == "grpc://localhost:2136"
    assert settings.connect_attempts == 3
    assert settings.connect_timeout_seconds == 1.5


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({"YDB_ENDPOINT": "localhost:2136"}, "must start with grpc"),
        ({"YDB_DATABASE": "local"}, "absolute database path"),
        ({"YDB_CONNECT_ATTEMPTS": "0"}, "at least 1"),
        ({"YDB_CONNECT_ATTEMPTS": "many"}, "must be an integer"),
        ({"YDB_CONNECT_TIMEOUT_SECONDS": "-1"}, "must be positive"),
    ],
)
def test_invalid_values_are_rejected_at_startup(environ, message):
    with pytest.raises(ValueError, match=message):
        YDBSettings.from_env(environ)
