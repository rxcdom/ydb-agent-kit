"""Tests for the tool registry."""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from src.agent.application.tools.registry import ToolRegistry, ToolSpec
from src.agent.ports.llm_tooling import LLMToolDefinition, LLMToolResult


class _EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


def _definition(name: str) -> LLMToolDefinition:
    return LLMToolDefinition(
        name=name, description=f"Test tool {name}.", parameters=_EchoArgs.model_json_schema()
    )


async def _echo(args: _EchoArgs, *, owner_user_id: str) -> LLMToolResult:
    return LLMToolResult(name="echo", content={"text": args.text, "owner": owner_user_id})


def test_registered_tool_is_found_by_name():
    registry = ToolRegistry()
    definition = _definition("echo")

    registry.register(definition, _EchoArgs, _echo)

    spec = registry.get("echo")
    assert spec == ToolSpec(definition=definition, args_model=_EchoArgs, executor=_echo)
    assert spec.name == "echo"


def test_unknown_name_yields_none():
    assert ToolRegistry().get("missing") is None


def test_definitions_and_specs_keep_registration_order():
    registry = ToolRegistry()
    for name in ("second", "first", "third"):
        registry.register(_definition(name), _EchoArgs, _echo)

    assert [definition.name for definition in registry.definitions()] == [
        "second",
        "first",
        "third",
    ]
    assert [spec.name for spec in registry.specs()] == ["second", "first", "third"]


def test_duplicate_name_is_rejected():
    registry = ToolRegistry()
    registry.register(_definition("echo"), _EchoArgs, _echo)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_definition("echo"), _EchoArgs, _echo)

    assert len(registry.definitions()) == 1


@pytest.mark.parametrize("name", ["", "   "])
def test_blank_name_is_rejected(name: str):
    with pytest.raises(ValueError, match="must have a name"):
        ToolRegistry().register(_definition(name), _EchoArgs, _echo)


def test_returned_lists_are_copies():
    registry = ToolRegistry()
    registry.register(_definition("echo"), _EchoArgs, _echo)

    registry.definitions().clear()
    registry.specs().clear()

    assert len(registry.definitions()) == 1
    assert len(registry.specs()) == 1
