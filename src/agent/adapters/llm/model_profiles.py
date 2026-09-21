"""Capabilities of the completion models the agent can run on.

Every model-specific assumption that shapes a provider request lives in this
registry. The LLM client states an intent ("this is an agent turn") and reads the
wire parameters of the configured model from its profile, so switching
``LLM_MODEL_NAME`` never sends one model a parameter that only another accepts.

Model names appear as literals in this module only. Code elsewhere asks the
registry and never compares model names itself.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import List, Mapping, Optional

logger = logging.getLogger(__name__)

_URI_SCHEME_SEPARATOR = "://"
_VERSION_SUFFIXES = ("/latest", "/rc")


@dataclass(frozen=True)
class ModelProfile:
    """What the client has to know about one model to build a request for it."""

    model_name: str
    display_name: str
    # The agent loop is built on function calling; a model without it cannot run it.
    supports_tool_calling: bool
    # Extra request parameters for an agent turn. ``None`` sends nothing, which
    # leaves the model at its own default.
    agent_reasoning_params: Optional[Mapping[str, str]] = None
    # ``False`` only on the fallback profile of a model missing from the registry.
    is_registered: bool = True


_PROFILES: Mapping[str, ModelProfile] = MappingProxyType(
    {
        profile.model_name: profile
        for profile in (
            ModelProfile(
                model_name="deepseek-v4-flash",
                display_name="DeepSeek V4 Flash",
                supports_tool_calling=True,
                agent_reasoning_params=MappingProxyType({"reasoning_effort": "low"}),
            ),
            ModelProfile(
                model_name="gpt-oss-120b",
                display_name="GPT-OSS 120B",
                supports_tool_calling=True,
                # "low" is the lowest reasoning effort this model accepts.
                agent_reasoning_params=MappingProxyType({"reasoning_effort": "low"}),
            ),
            ModelProfile(
                model_name="yandexgpt-5-lite",
                display_name="YandexGPT 5 Lite",
                supports_tool_calling=False,
            ),
        )
    }
)


def _normalize_model_name(model_name: str) -> str:
    """Reduce a model reference to its registry key.

    Accepts the bare name as well as the URI forms of the provider,
    ``gpt://<folder>/<model>`` with an optional ``/latest`` or ``/rc`` suffix.
    """
    name = (model_name or "").strip().lower()
    if _URI_SCHEME_SEPARATOR in name:
        path = name.split(_URI_SCHEME_SEPARATOR, 1)[1]
        parts = [part for part in path.split("/") if part]
        # The first part is the folder; a URI without a model part names nothing.
        name = parts[1] if len(parts) >= 2 else ""
    for suffix in _VERSION_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def get_model_profile(model_name: str) -> ModelProfile:
    """Return the profile of a model.

    A model missing from the registry gets a safe default profile: no
    provider-specific parameters, so the request cannot carry one the model
    rejects, and tool calling marked as unsupported. The miss is logged.
    """
    profile = _PROFILES.get(_normalize_model_name(model_name))
    if profile is not None:
        return profile

    logger.warning(
        "No profile is registered for model %r; using the safe default profile "
        "(no provider-specific parameters, tool calling unsupported)",
        model_name,
    )
    return ModelProfile(
        model_name=model_name,
        display_name=model_name,
        supports_tool_calling=False,
        agent_reasoning_params=None,
        is_registered=False,
    )


def is_known_model(model_name: str) -> bool:
    """Tell whether the registry holds a profile for the model."""
    return _normalize_model_name(model_name) in _PROFILES


def list_model_profiles() -> List[ModelProfile]:
    """Return every registered profile in registry order."""
    return list(_PROFILES.values())
