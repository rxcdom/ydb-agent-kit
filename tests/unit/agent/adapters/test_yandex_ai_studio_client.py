"""The AI Studio LLM client against a fake SDK.

The real SDK class is never constructed here: every test that reaches the SDK
installs a fake factory, and a test that forgets to fails loudly. Retries never
sleep, because the client receives a recording ``sleep``.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from src.agent.adapters.llm import yandex_ai_studio_client as client_module
from src.agent.adapters.llm.yandex_ai_studio_client import YandexAIStudioLLMClient
from src.agent.domain.entities.assistant import Assistant
from src.agent.domain.exceptions import (
    LLMServiceAuthenticationError,
    LLMServiceError,
    LLMServiceInvalidResponseError,
    LLMServiceTimeoutError,
    LLMServiceUnavailableError,
)
from src.agent.ports.llm_tooling import LLMToolDefinition

UNREGISTERED_MODEL = "unit-test-model"
USER_TURN = [{"role": "user", "text": "What is due this week?"}]


class FakeConfiguredModel:
    """Walks a script: an exception step is raised, any other step is returned."""

    def __init__(self, *steps):
        self._steps = list(steps)
        self.run_calls = []

    async def run(self, messages):
        self.run_calls.append(messages)
        if not self._steps:
            raise AssertionError("the model was called more often than the test scripted")
        step = self._steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


class FakeBaseModel:
    def __init__(self, configured_model):
        self._configured_model = configured_model
        self.configure_calls = []

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)
        return self._configured_model


class FakeChat:
    def __init__(self, base_model):
        self._base_model = base_model
        self.completions_calls = []

    def completions(self, model_name):
        self.completions_calls.append(model_name)
        return self._base_model


class FakeTools:
    def __init__(self):
        self.function_calls = []

    def function(self, **kwargs):
        self.function_calls.append(kwargs)
        return {"native_tool": kwargs["name"]}


class FakeSDK:
    def __init__(self, auth, folder_id, base_model):
        self.auth = auth
        self.folder_id = folder_id
        self.chat = FakeChat(base_model)
        self.tools = FakeTools()


class FakeTextResult(list):
    """A final-text result: indexable like the SDK result, with ``text`` and ``usage``."""

    def __init__(self, text, total_tokens=0):
        super().__init__([SimpleNamespace(text=text)])
        self.text = text
        self.usage = SimpleNamespace(total_tokens=total_tokens)


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call-1"):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeToolCallResult:
    def __init__(self, tool_calls, total_tokens=17):
        self.tool_calls = tool_calls
        self.usage = SimpleNamespace(total_tokens=total_tokens)


class Harness:
    """A fake SDK installed into the client module, plus what it recorded."""

    def __init__(self, monkeypatch, *steps):
        self.configured_model = FakeConfiguredModel(*steps)
        self.base_model = FakeBaseModel(self.configured_model)
        self.sdks = []
        self.sleeps = []
        monkeypatch.setattr(client_module, "AsyncAIStudio", self._build_sdk)

    def _build_sdk(self, *, auth, folder_id):
        sdk = FakeSDK(auth, folder_id, self.base_model)
        self.sdks.append(sdk)
        return sdk

    async def _sleep(self, seconds):
        self.sleeps.append(seconds)

    def client(self, **overrides):
        kwargs = {"folder_id": "folder-id", "api_key": "api-key", "sleep": self._sleep}
        kwargs.update(overrides)
        return YandexAIStudioLLMClient(**kwargs)

    @property
    def configure_kwargs(self):
        assert len(self.base_model.configure_calls) == 1
        return self.base_model.configure_calls[0]


def _assistant(model_name=UNREGISTERED_MODEL):
    return Assistant(
        name="Test assistant",
        system_prompt="You are a helpful task assistant.",
        model_name=model_name,
        temperature=0.2,
    )


def _tool(name="query_tasks"):
    return LLMToolDefinition(
        name=name,
        description="Query the tasks of the user.",
        parameters={
            "type": "object",
            "properties": {"date_field": {"type": "string"}},
            "required": [],
        },
        strict=True,
    )


@pytest.fixture(autouse=True)
def forbid_the_real_sdk(monkeypatch):
    def refuse(**_kwargs):
        raise AssertionError("a unit test must not construct the real SDK")

    monkeypatch.setattr(client_module, "AsyncAIStudio", refuse)


class TestConfiguration:
    async def test_missing_credentials_fail_at_first_use_as_authentication_error(self):
        client = YandexAIStudioLLMClient(folder_id="folder-id")

        with pytest.raises(LLMServiceAuthenticationError, match="YC_API_KEY nor YC_IAM_TOKEN"):
            await client.run_agent_turn(_assistant(), USER_TURN, [])

    async def test_missing_folder_fails_at_first_use_as_authentication_error(self):
        client = YandexAIStudioLLMClient(folder_id="  ", api_key="api-key")

        with pytest.raises(LLMServiceAuthenticationError, match="YC_FOLDER_ID"):
            await client.run_agent_turn(_assistant(), USER_TURN, [])

    async def test_misconfigured_client_never_builds_the_sdk(self, monkeypatch):
        harness = Harness(monkeypatch)

        with pytest.raises(LLMServiceAuthenticationError):
            await harness.client(api_key="", iam_token="").run_agent_turn(
                _assistant(), USER_TURN, []
            )

        assert harness.sdks == []

    def test_constructing_the_client_does_not_build_the_sdk(self, monkeypatch):
        harness = Harness(monkeypatch)

        harness.client()

        assert harness.sdks == []

    async def test_sdk_is_built_once_and_reused(self, monkeypatch):
        harness = Harness(monkeypatch, FakeTextResult("first"), FakeTextResult("second"))
        client = harness.client()

        await client.run_agent_turn(_assistant(), USER_TURN, [])
        await client.run_agent_turn(_assistant(), USER_TURN, [])

        assert len(harness.sdks) == 1
        assert harness.sdks[0].folder_id == "folder-id"

    async def test_api_key_wins_over_the_iam_token(self, monkeypatch):
        harness = Harness(monkeypatch, FakeTextResult("ok"))

        await harness.client(api_key=" api-key ", iam_token="iam-token").run_agent_turn(
            _assistant(), USER_TURN, []
        )

        assert harness.sdks[0].auth == "api-key"

    async def test_iam_token_is_used_when_there_is_no_api_key(self, monkeypatch):
        harness = Harness(monkeypatch, FakeTextResult("ok"))

        await harness.client(api_key="", iam_token="iam-token").run_agent_turn(
            _assistant(), USER_TURN, []
        )

        assert harness.sdks[0].auth == "iam-token"

    async def test_missing_sdk_is_reported_as_unavailable_without_retries(self, monkeypatch):
        harness = Harness(monkeypatch)
        monkeypatch.setattr(client_module, "AsyncAIStudio", None)

        with pytest.raises(LLMServiceUnavailableError, match="not installed"):
            await harness.client().run_agent_turn(_assistant(), USER_TURN, [_tool()])

        assert harness.sleeps == []

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"retry_attempts": -1}, "retry_attempts"),
            ({"retry_attempts": 1, "retry_backoff_seconds": ()}, "cannot be empty"),
            ({"retry_backoff_seconds": (0.5, -1.0)}, "negative delay"),
        ],
    )
    def test_invalid_retry_settings_are_rejected(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            YandexAIStudioLLMClient(folder_id="folder-id", api_key="api-key", **overrides)

    def test_retries_can_be_switched_off_without_a_backoff(self):
        YandexAIStudioLLMClient(
            folder_id="folder-id", api_key="api-key", retry_attempts=0, retry_backoff_seconds=()
        )


class TestToolBindingAndToolCalls:
    async def test_tools_are_bound_natively_and_tool_calls_are_mapped(self, monkeypatch):
        result = FakeToolCallResult(
            [FakeToolCall("query_tasks", {"date_field": "due"}, call_id="call-7")],
            total_tokens=44,
        )
        harness = Harness(monkeypatch, result)
        assistant = _assistant()

        turn = await harness.client().run_agent_turn(assistant, USER_TURN, [_tool()])

        assert turn.text == ""
        assert turn.tokens == 44
        assert turn.cost == 0.0
        assert [(call.name, call.arguments, call.call_id) for call in turn.tool_calls] == [
            ("query_tasks", {"date_field": "due"}, "call-7")
        ]
        assert turn.response_event is result

        sdk = harness.sdks[0]
        assert sdk.chat.completions_calls == [assistant.model_name]
        assert sdk.tools.function_calls == [
            {
                "name": "query_tasks",
                "description": "Query the tasks of the user.",
                "parameters": _tool().parameters,
                "strict": True,
            }
        ]
        assert harness.configure_kwargs["tools"] == [{"native_tool": "query_tasks"}]
        assert harness.configure_kwargs["temperature"] == assistant.temperature

    async def test_history_reaches_the_sdk_unchanged(self, monkeypatch):
        harness = Harness(monkeypatch, FakeTextResult("done"))
        earlier_event = FakeToolCallResult([FakeToolCall("list_projects", {})])
        history = [
            {"role": "system", "text": "You are a helpful task assistant."},
            {"role": "user", "text": "Which projects do I have?"},
            earlier_event,
            {"tool_results": [{"name": "list_projects", "content": '{"status": "no_data"}'}]},
        ]

        await harness.client().run_agent_turn(_assistant(), history, [_tool("list_projects")])

        (sent,) = harness.configured_model.run_calls
        assert sent is history
        assert sent[2] is earlier_event

    async def test_agent_turn_sets_no_output_token_cap(self, monkeypatch):
        harness = Harness(monkeypatch, FakeTextResult("ok"))

        await harness.client().run_agent_turn(_assistant(), USER_TURN, [_tool()])

        assert "max_tokens" not in harness.configure_kwargs

    async def test_turn_without_tools_binds_none(self, monkeypatch):
        harness = Harness(monkeypatch, FakeTextResult("ok"))

        await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert "tools" not in harness.configure_kwargs
        assert harness.sdks[0].tools.function_calls == []

    async def test_tool_call_without_arguments_gets_an_empty_object(self, monkeypatch):
        harness = Harness(monkeypatch, FakeToolCallResult([FakeToolCall("list_projects", None)]))

        turn = await harness.client().run_agent_turn(_assistant(), USER_TURN, [_tool()])

        assert turn.tool_calls[0].arguments == {}

    @pytest.mark.parametrize("arguments", [["due"], "due", 7, []])
    async def test_tool_call_arguments_that_are_not_an_object_are_rejected(
        self, monkeypatch, arguments
    ):
        harness = Harness(monkeypatch, FakeToolCallResult([FakeToolCall("query_tasks", arguments)]))

        with pytest.raises(LLMServiceInvalidResponseError, match="not a JSON object"):
            await harness.client().run_agent_turn(_assistant(), USER_TURN, [_tool()])

        assert len(harness.configured_model.run_calls) == 1

    async def test_tool_call_without_a_function_is_skipped_with_a_warning(
        self, monkeypatch, caplog
    ):
        broken = SimpleNamespace(id="call-0", function=None)
        result = FakeToolCallResult([broken, FakeToolCall("query_tasks", {}, call_id="call-1")])
        harness = Harness(monkeypatch, result)

        with caplog.at_level(logging.WARNING, logger=client_module.__name__):
            turn = await harness.client().run_agent_turn(_assistant(), USER_TURN, [_tool()])

        assert [call.call_id for call in turn.tool_calls] == ["call-1"]
        assert any("no function" in record.getMessage() for record in caplog.records)


class TestContentAndUsage:
    async def test_final_text_and_token_usage_are_extracted(self, monkeypatch):
        result = FakeTextResult("Two tasks are due.", total_tokens=123)
        harness = Harness(monkeypatch, result)

        turn = await harness.client().run_agent_turn(_assistant(), USER_TURN, [_tool()])

        assert turn.text == "Two tasks are due."
        assert turn.tool_calls == []
        assert turn.tokens == 123
        assert turn.response_event is result

    async def test_text_is_read_from_the_first_alternative_when_the_shortcut_is_empty(
        self, monkeypatch
    ):
        result = FakeTextResult("From the alternative.")
        result.text = ""
        harness = Harness(monkeypatch, result)

        turn = await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert turn.text == "From the alternative."

    @pytest.mark.parametrize(
        "usage, expected",
        [
            (SimpleNamespace(total_tokens=9), 9),
            ({"total_tokens": "12"}, 12),
            ({"total_tokens": None}, 0),
            (SimpleNamespace(total_tokens="many"), 0),
            (None, 0),
        ],
    )
    async def test_token_usage_tolerates_every_reported_shape(self, monkeypatch, usage, expected):
        result = FakeTextResult("ok")
        result.usage = usage
        harness = Harness(monkeypatch, result)

        turn = await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert turn.tokens == expected

    @pytest.mark.parametrize("text", ["", "   ", None])
    async def test_empty_completion_without_tool_calls_is_an_invalid_response(
        self, monkeypatch, text
    ):
        harness = Harness(monkeypatch, FakeTextResult(text))

        with pytest.raises(LLMServiceInvalidResponseError, match="neither text nor tool calls"):
            await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert len(harness.configured_model.run_calls) == 1


class TestReasoningParametersComeFromTheProfile:
    @pytest.mark.parametrize(
        "model_name, expected_extra_query",
        [
            ("deepseek-v4-flash", {"reasoning_effort": "low"}),
            ("gpt-oss-120b", {"reasoning_effort": "low"}),
            ("gpt://example-folder/gpt-oss-120b/latest", {"reasoning_effort": "low"}),
            ("yandexgpt-5-lite", None),
            ("model-from-the-future", None),
        ],
    )
    async def test_extra_query_follows_the_model_profile(
        self, monkeypatch, model_name, expected_extra_query
    ):
        harness = Harness(monkeypatch, FakeTextResult("ok"))

        await harness.client().run_agent_turn(_assistant(model_name), USER_TURN, [])

        if expected_extra_query is None:
            assert "extra_query" not in harness.configure_kwargs
        else:
            assert harness.configure_kwargs["extra_query"] == expected_extra_query
            # The SDK accepts a plain dict only, not the read-only registry mapping.
            assert type(harness.configure_kwargs["extra_query"]) is dict

    async def test_model_name_reaches_the_sdk_as_configured(self, monkeypatch):
        harness = Harness(monkeypatch, FakeTextResult("ok"))

        await harness.client().run_agent_turn(_assistant("gpt-oss-120b"), USER_TURN, [])

        assert harness.sdks[0].chat.completions_calls == ["gpt-oss-120b"]


class TestRetries:
    async def test_transient_failures_are_retried_until_the_turn_succeeds(
        self, monkeypatch, caplog
    ):
        harness = Harness(
            monkeypatch,
            RuntimeError("503 service unavailable"),
            TimeoutError("request timed out"),
            FakeTextResult("Recovered.", total_tokens=5),
        )

        with caplog.at_level(logging.WARNING, logger=client_module.__name__):
            turn = await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert turn.text == "Recovered."
        assert len(harness.configured_model.run_calls) == 3
        assert harness.sleeps == [0.5, 1.0]
        retries = [r.getMessage() for r in caplog.records if "retry" in r.getMessage()]
        assert len(retries) == 2
        assert "retry 1/2" in retries[0] and "LLMServiceUnavailableError" in retries[0]
        assert "retry 2/2" in retries[1] and "LLMServiceTimeoutError" in retries[1]

    async def test_exhausted_retries_raise_the_mapped_error_with_its_cause(self, monkeypatch):
        failures = [TimeoutError("request timed out") for _ in range(3)]
        harness = Harness(monkeypatch, *failures)

        with pytest.raises(LLMServiceTimeoutError) as raised:
            await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert len(harness.configured_model.run_calls) == 1 + client_module.DEFAULT_RETRY_ATTEMPTS
        assert harness.sleeps == [0.5, 1.0]
        assert raised.value.cause is failures[-1]

    async def test_last_backoff_value_repeats_when_there_are_more_retries_than_values(
        self, monkeypatch
    ):
        harness = Harness(monkeypatch, *[RuntimeError("502 bad gateway") for _ in range(4)])
        client = harness.client(retry_attempts=3, retry_backoff_seconds=(0.1, 0.2))

        with pytest.raises(LLMServiceUnavailableError):
            await client.run_agent_turn(_assistant(), USER_TURN, [])

        assert harness.sleeps == [0.1, 0.2, 0.2]

    async def test_no_retries_when_they_are_switched_off(self, monkeypatch):
        harness = Harness(monkeypatch, RuntimeError("502 bad gateway"))
        client = harness.client(retry_attempts=0)

        with pytest.raises(LLMServiceUnavailableError):
            await client.run_agent_turn(_assistant(), USER_TURN, [])

        assert len(harness.configured_model.run_calls) == 1
        assert harness.sleeps == []

    @pytest.mark.parametrize(
        "error, expected",
        [
            (RuntimeError("401 unauthorized"), LLMServiceAuthenticationError),
            (LLMServiceAuthenticationError("rejected"), LLMServiceAuthenticationError),
            (ValueError("malformed payload"), LLMServiceInvalidResponseError),
            (LLMServiceInvalidResponseError("unusable"), LLMServiceInvalidResponseError),
        ],
    )
    async def test_permanent_failures_are_not_retried(self, monkeypatch, error, expected):
        harness = Harness(monkeypatch, error)

        with pytest.raises(expected):
            await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert len(harness.configured_model.run_calls) == 1
        assert harness.sleeps == []

    async def test_already_mapped_transient_errors_are_retried_too(self, monkeypatch):
        harness = Harness(
            monkeypatch, LLMServiceUnavailableError("provider is down"), FakeTextResult("ok")
        )

        turn = await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert turn.text == "ok"
        assert harness.sleeps == [0.5]


class TestErrorMapping:
    @pytest.mark.parametrize(
        "error, expected",
        [
            (TimeoutError("request timed out"), LLMServiceTimeoutError),
            (RuntimeError("401 unauthorized"), LLMServiceAuthenticationError),
            (RuntimeError("service temporarily down"), LLMServiceUnavailableError),
        ],
    )
    async def test_provider_failures_leave_the_client_as_domain_errors(
        self, monkeypatch, error, expected
    ):
        harness = Harness(monkeypatch, error, error, error)

        with pytest.raises(expected) as raised:
            await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert isinstance(raised.value, LLMServiceError)
        assert raised.value.cause is error

    async def test_unusable_answer_is_an_invalid_response(self, monkeypatch):
        harness = Harness(monkeypatch, SimpleNamespace(usage=SimpleNamespace(total_tokens=11)))

        with pytest.raises(LLMServiceInvalidResponseError):
            await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

    async def test_already_mapped_error_passes_through_untouched(self, monkeypatch):
        original = LLMServiceInvalidResponseError("unusable")
        harness = Harness(monkeypatch, original)

        with pytest.raises(LLMServiceInvalidResponseError) as raised:
            await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert raised.value is original
        # Re-raising must not make the error its own cause.
        assert raised.value.cause is None

    async def test_every_failure_is_logged_once_with_both_error_types(self, monkeypatch, caplog):
        harness = Harness(monkeypatch, RuntimeError("403 forbidden"))

        with caplog.at_level(logging.ERROR, logger=client_module.__name__):
            with pytest.raises(LLMServiceAuthenticationError):
                await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        (record,) = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert "RuntimeError" in record.getMessage()
        assert "LLMServiceAuthenticationError" in record.getMessage()
        assert "api-key" not in record.getMessage()

    @pytest.mark.parametrize(
        "error, expected",
        [
            (type("ReadTimeout", (Exception,), {})("stalled"), LLMServiceTimeoutError),
            (RuntimeError("Deadline Exceeded"), LLMServiceTimeoutError),
            (RuntimeError("504 gateway timeout"), LLMServiceTimeoutError),
            (type("AuthError", (Exception,), {})("rejected"), LLMServiceAuthenticationError),
            (RuntimeError("403 Forbidden"), LLMServiceAuthenticationError),
            (RuntimeError("Invalid API key"), LLMServiceAuthenticationError),
            (RuntimeError("UNAUTHENTICATED: token expired"), LLMServiceAuthenticationError),
            (RuntimeError("failed to parse the body"), LLMServiceInvalidResponseError),
            (RuntimeError("cannot decode the stream"), LLMServiceInvalidResponseError),
            (RuntimeError("500 internal server error"), LLMServiceUnavailableError),
            (ConnectionError("all connection attempts failed"), LLMServiceUnavailableError),
        ],
    )
    def test_unwrapped_errors_are_classified_by_type_name_and_message(self, error, expected):
        client = YandexAIStudioLLMClient(folder_id="folder-id", api_key="api-key")

        mapped = client._map_exception(error)

        assert type(mapped) is expected
        assert mapped.cause is error

    def test_json_parser_error_is_an_invalid_response(self):
        client = YandexAIStudioLLMClient(folder_id="folder-id", api_key="api-key")
        with pytest.raises(json.JSONDecodeError) as parser_error:
            json.loads("{not json")

        mapped = client._map_exception(parser_error.value)

        assert type(mapped) is LLMServiceInvalidResponseError
        assert mapped.cause is parser_error.value


class TestSdkExceptionClassesAreMatchedFirst:
    """Class matching wins over the message: the messages below would mislead it."""

    @pytest.fixture
    def fake_sdk_exceptions(self, monkeypatch):
        class FakeAIStudioError(Exception):
            pass

        class FakeAioRpcError(Exception):
            def __init__(self, status_name, message):
                super().__init__(message)
                self._status = SimpleNamespace(name=status_name)

            def code(self):
                return self._status

        namespace = SimpleNamespace(AIStudioError=FakeAIStudioError, AioRpcError=FakeAioRpcError)
        monkeypatch.setattr(client_module, "sdk_exceptions", namespace)
        return namespace

    @pytest.mark.parametrize(
        "status_name, message, expected",
        [
            ("UNAUTHENTICATED", "request timed out", LLMServiceAuthenticationError),
            ("PERMISSION_DENIED", "service is down", LLMServiceAuthenticationError),
            ("DEADLINE_EXCEEDED", "401 unauthorized", LLMServiceTimeoutError),
            ("UNAVAILABLE", "invalid api key", LLMServiceUnavailableError),
            ("RESOURCE_EXHAUSTED", "malformed quota", LLMServiceUnavailableError),
        ],
    )
    def test_rpc_errors_are_mapped_by_their_status(
        self, fake_sdk_exceptions, status_name, message, expected
    ):
        client = YandexAIStudioLLMClient(folder_id="folder-id", api_key="api-key")
        error = fake_sdk_exceptions.AioRpcError(status_name, message)

        mapped = client._map_exception(error)

        assert type(mapped) is expected
        assert mapped.cause is error

    def test_other_sdk_errors_mean_the_provider_is_unavailable(self, fake_sdk_exceptions):
        client = YandexAIStudioLLMClient(folder_id="folder-id", api_key="api-key")
        error = fake_sdk_exceptions.AIStudioError("401 unauthorized endpoint lookup")

        assert type(client._map_exception(error)) is LLMServiceUnavailableError

    def test_without_the_sdk_the_heuristics_still_classify(self, monkeypatch):
        monkeypatch.setattr(client_module, "sdk_exceptions", None)
        client = YandexAIStudioLLMClient(folder_id="folder-id", api_key="api-key")

        assert type(client._map_exception(RuntimeError("403"))) is LLMServiceAuthenticationError

    async def test_rpc_authentication_failure_is_not_retried(
        self, monkeypatch, fake_sdk_exceptions
    ):
        error = fake_sdk_exceptions.AioRpcError("UNAUTHENTICATED", "service is down")
        harness = Harness(monkeypatch, error)

        with pytest.raises(LLMServiceAuthenticationError):
            await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert len(harness.configured_model.run_calls) == 1

    def test_installed_sdk_exposes_the_classes_the_mapper_relies_on(self):
        from yandex_ai_studio_sdk import exceptions as installed

        assert client_module.sdk_exceptions is installed
        assert issubclass(installed.AioRpcError, Exception)
        assert callable(getattr(installed.AioRpcError, "code"))
        assert issubclass(installed.HttpSseError, installed.AIStudioError)
        assert issubclass(installed.UnknownEndpointError, installed.AIStudioError)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestHistoryContractWithTheInstalledSdk:
    """The history shapes the loop builds, checked against the SDK's own converter.

    Only pure functions of the installed SDK run here: a provider-shaped JSON
    body becomes a real result object, the client reads it, and the converter
    turns the next request's history into wire messages. Nothing is sent.
    """

    @pytest.fixture
    def wire(self):
        try:
            from yandex_ai_studio_sdk._chat.completions.message import messages_to_json
            from yandex_ai_studio_sdk._chat.completions.result import ChatModelResult
            from yandex_ai_studio_sdk._tools.tool_call import AsyncToolCall
        except ImportError as error:
            pytest.fail(f"the SDK internals this contract reads have moved: {error}")

        sdk_stub = SimpleNamespace(
            tools=SimpleNamespace(function=SimpleNamespace(_call_impl=AsyncToolCall))
        )

        def result_from(message, finish_reason, usage=None):
            body = {
                "id": "completion-1",
                "created": 1_789_000_000,
                "model": "gpt://example-folder/unit-test-model/latest",
                "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
            }
            if usage is not None:
                body["usage"] = usage
            return ChatModelResult._from_json(data=body, sdk=sdk_stub)

        def empty_result():
            body = {"id": "completion-2", "created": 1_789_000_000, "model": "m", "choices": []}
            return ChatModelResult._from_json(data=body, sdk=sdk_stub)

        return SimpleNamespace(
            to_json=messages_to_json, result_from=result_from, empty_result=empty_result
        )

    @staticmethod
    def _tool_call(call_id, name, arguments):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }

    async def test_tool_turn_round_trips_into_the_next_request(self, monkeypatch, wire):
        provider_result = wire.result_from(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    self._tool_call("call-a", "query_tasks", {"date_field": "due"}),
                    self._tool_call("call-b", "list_projects", {}),
                ],
            },
            "tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        harness = Harness(monkeypatch, provider_result)

        turn = await harness.client().run_agent_turn(_assistant(), USER_TURN, [_tool()])

        assert turn.text == ""
        assert turn.tokens == 15
        assert [(call.call_id, call.name, call.arguments) for call in turn.tool_calls] == [
            ("call-a", "query_tasks", {"date_field": "due"}),
            ("call-b", "list_projects", {}),
        ]

        next_history = [
            {"role": "system", "text": "You are a helpful task assistant."},
            *USER_TURN,
            turn.response_event,
            {
                "tool_results": [
                    {"name": "query_tasks", "content": json.dumps({"status": "ok"})},
                    {"name": "list_projects", "content": json.dumps({"status": "no_data"})},
                ]
            },
        ]
        sent = wire.to_json(next_history)

        assert sent[0] == {"role": "system", "content": "You are a helpful task assistant."}
        assert sent[1] == {"role": "user", "content": "What is due this week?"}
        assert sent[2]["role"] == "assistant"
        assert [call["id"] for call in sent[2]["tool_calls"]] == ["call-a", "call-b"]
        assert sent[3:] == [
            {"role": "tool", "tool_call_id": "call-a", "content": '{"status": "ok"}'},
            {"role": "tool", "tool_call_id": "call-b", "content": '{"status": "no_data"}'},
        ]

    async def test_final_text_turn_round_trips_as_an_assistant_message(self, monkeypatch, wire):
        provider_result = wire.result_from(
            {"role": "assistant", "content": "Two tasks are due."},
            "stop",
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        )
        harness = Harness(monkeypatch, provider_result)

        turn = await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert (turn.text, turn.tool_calls, turn.tokens) == ("Two tasks are due.", [], 3)
        assert wire.to_json([turn.response_event]) == [
            {"role": "assistant", "content": "Two tasks are due."}
        ]

    async def test_response_without_alternatives_is_an_invalid_response(self, monkeypatch, wire):
        harness = Harness(monkeypatch, wire.empty_result())

        with pytest.raises(LLMServiceInvalidResponseError):
            await harness.client().run_agent_turn(_assistant(), USER_TURN, [])

        assert len(harness.configured_model.run_calls) == 1

    def test_tool_results_need_the_response_event_right_before_them(self, wire):
        orphaned = [*USER_TURN, {"tool_results": [{"name": "query_tasks", "content": "{}"}]}]

        with pytest.raises(ValueError, match="failed to find tool call"):
            wire.to_json(orphaned)

    def test_repeated_tool_results_are_addressed_to_the_last_call_of_that_tool(self, wire):
        # Pins the converter behaviour the client docstring documents. A failure
        # here means the SDK changed and the docstring has to follow.
        twice = wire.result_from(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    self._tool_call("call-1", "query_tasks", {}),
                    self._tool_call("call-2", "query_tasks", {}),
                ],
            },
            "tool_calls",
        )
        results = {
            "tool_results": [
                {"name": "query_tasks", "content": "first"},
                {"name": "query_tasks", "content": "second"},
            ]
        }

        sent = wire.to_json([twice, results])

        assert [message["tool_call_id"] for message in sent[1:]] == ["call-2", "call-2"]
