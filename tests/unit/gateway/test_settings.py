import pytest

from src.gateway.settings import Settings


def test_defaults_follow_the_documented_table():
    settings = Settings.from_env({})

    assert settings.llm_model_name == "gpt-oss-120b"
    assert settings.llm_temperature == 0.2
    assert settings.agent_max_iterations == 5
    assert settings.agent_history_limit == 20
    assert settings.agent_timezone == "UTC"
    assert settings.migration_applied_by == "local"
    assert settings.log_level == "INFO"
    assert settings.app_port == 8000
    assert settings.ydb.endpoint == "grpc://ydb:2136"
    assert settings.yc_api_key == ""


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({"LLM_TEMPERATURE": "3"}, "between 0 and 2"),
        ({"LLM_TEMPERATURE": "warm"}, "LLM_TEMPERATURE has an invalid value"),
        ({"AGENT_MAX_ITERATIONS": "0"}, "at least 1"),
        ({"AGENT_HISTORY_LIMIT": "0"}, "at least 1"),
        ({"AGENT_TIMEZONE": "Mars/Olympus"}, "IANA zone name"),
        ({"LOG_LEVEL": "chatty"}, "not a logging level"),
        ({"APP_PORT": "70000"}, "between 1 and 65535"),
    ],
)
def test_invalid_values_stop_the_application_at_startup(environ, message):
    with pytest.raises(ValueError, match=message):
        Settings.from_env(environ)


def test_settings_are_immutable():
    settings = Settings.from_env({})

    with pytest.raises(AttributeError):
        settings.app_port = 1
