"""The model capability registry: rows, safe fallback, name normalisation."""
import logging
import re
from pathlib import Path

import pytest

from src.agent.adapters.llm import model_profiles
from src.agent.adapters.llm.model_profiles import (
    ModelProfile,
    get_model_profile,
    is_known_model,
    list_model_profiles,
)

ADAPTERS_ROOT = Path(model_profiles.__file__).resolve().parents[1]
REGISTRY_FILE = Path(model_profiles.__file__).resolve()


class TestRegisteredProfiles:
    def test_registry_holds_exactly_the_planned_rows(self):
        assert [profile.model_name for profile in list_model_profiles()] == [
            "deepseek-v4-flash",
            "gpt-oss-120b",
            "yandexgpt-5-lite",
        ]

    @pytest.mark.parametrize("model_name", ["deepseek-v4-flash", "gpt-oss-120b"])
    def test_tool_calling_models_run_agent_turns_at_low_reasoning_effort(self, model_name):
        profile = get_model_profile(model_name)

        assert profile.is_registered is True
        assert profile.supports_tool_calling is True
        assert dict(profile.agent_reasoning_params) == {"reasoning_effort": "low"}

    def test_lite_model_cannot_call_tools_and_gets_no_reasoning_parameter(self):
        profile = get_model_profile("yandexgpt-5-lite")

        assert profile.is_registered is True
        assert profile.supports_tool_calling is False
        assert profile.agent_reasoning_params is None

    def test_every_registered_profile_is_marked_registered_and_named(self):
        profiles = list_model_profiles()

        assert profiles
        assert all(profile.is_registered for profile in profiles)
        assert all(profile.display_name.strip() for profile in profiles)

    def test_profiles_and_their_parameters_are_read_only(self):
        profile = get_model_profile("gpt-oss-120b")

        with pytest.raises(AttributeError):
            profile.supports_tool_calling = False
        with pytest.raises(TypeError):
            profile.agent_reasoning_params["reasoning_effort"] = "high"

    def test_listing_is_a_copy_of_the_registry(self):
        listed = list_model_profiles()
        listed.clear()

        assert list_model_profiles()


class TestUnknownModelFallback:
    def test_unknown_model_gets_the_safe_default_and_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=model_profiles.__name__):
            profile = get_model_profile("model-from-the-future")

        assert profile == ModelProfile(
            model_name="model-from-the-future",
            display_name="model-from-the-future",
            supports_tool_calling=False,
            agent_reasoning_params=None,
            is_registered=False,
        )
        assert any(
            record.levelno == logging.WARNING and "model-from-the-future" in record.getMessage()
            for record in caplog.records
        )

    def test_known_model_is_resolved_without_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=model_profiles.__name__):
            get_model_profile("gpt-oss-120b")

        assert caplog.records == []

    def test_empty_name_is_unknown(self):
        assert is_known_model("") is False
        assert get_model_profile("").is_registered is False


class TestNormalisation:
    @pytest.mark.parametrize(
        "raw",
        [
            "deepseek-v4-flash",
            " Deepseek-V4-Flash ",
            "deepseek-v4-flash/latest",
            "deepseek-v4-flash/rc",
            "gpt://example-folder/deepseek-v4-flash",
            "gpt://example-folder/deepseek-v4-flash/latest",
            "GPT://example-folder/DeepSeek-V4-Flash/rc",
        ],
    )
    def test_uri_case_and_version_forms_resolve_to_one_profile(self, raw):
        assert is_known_model(raw) is True
        assert get_model_profile(raw).model_name == "deepseek-v4-flash"

    @pytest.mark.parametrize("raw", ["gpt://deepseek-v4-flash", "gpt://", "gpt:///"])
    def test_uri_without_a_model_part_is_unknown(self, raw):
        assert is_known_model(raw) is False


class TestModelNamesStayInTheRegistry:
    """Model-specific behaviour is looked up in the registry, never compared inline."""

    @staticmethod
    def _other_adapter_sources():
        sources = [path for path in sorted(ADAPTERS_ROOT.rglob("*.py")) if path != REGISTRY_FILE]
        assert sources, "the adapter package was not found"
        return sources

    def test_no_adapter_compares_a_model_name(self):
        comparison = re.compile(r"model_name\s*(==|!=|\bnot\s+in\b|\bin\b)|\.startswith\(")
        offenders = []
        for path in self._other_adapter_sources():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "model_name" in line and comparison.search(line):
                    offenders.append(f"{path.relative_to(ADAPTERS_ROOT)}:{number}: {line.strip()}")

        assert offenders == []

    def test_no_adapter_spells_a_registered_model_name(self):
        names = [profile.model_name for profile in list_model_profiles()]
        offenders = []
        for path in self._other_adapter_sources():
            text = path.read_text(encoding="utf-8").lower()
            offenders += [
                f"{path.relative_to(ADAPTERS_ROOT)}: {name}" for name in names if name in text
            ]

        assert offenders == []
