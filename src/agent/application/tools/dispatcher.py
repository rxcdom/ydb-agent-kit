"""Run the tool calls of one model turn.

The dispatcher stands between model output and the tool executors, so it treats
every call as untrusted input and never lets one call break the turn:

* an unknown tool name, arguments that do not fit the schema and an executor
  that raises all become a result the model can read and react to;
* every call gets exactly one result, in the order the calls arrived. The loop
  relies on that to pair calls with results.

The owner id is not part of any tool's arguments. It comes from the
authenticated principal and is handed to each executor as a keyword.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from pydantic import ValidationError

from src.agent.application.tools.registry import ToolRegistry, ToolSpec
from src.agent.ports.llm_tooling import LLMToolCall, LLMToolResult

logger = logging.getLogger(__name__)

# Longest rendering of one payload or result in the debug log.
LOG_VALUE_CAP = 1200


def _render_for_log(value: Any) -> str:
    """Render a payload or a result readably, capped in length."""
    try:
        rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        rendered = repr(value)
    if len(rendered) > LOG_VALUE_CAP:
        omitted = len(rendered) - LOG_VALUE_CAP
        return f"{rendered[:LOG_VALUE_CAP]}\n... [{omitted} more characters]"
    return rendered


def _summarise_result(content: Dict[str, Any]) -> str:
    """One-line summary of a result that carries no user content."""
    if "status" in content:
        return f"status={content['status']}"
    if "error" in content:
        return f"error={content['error']}"
    return f"keys={sorted(content)}"


def _drop_null_arguments(arguments: Any) -> Any:
    """Treat an explicit ``null`` as "argument not given".

    Some models send ``null`` for every optional argument they do not use. That
    is a habit of the protocol, not a mistake, but ``null`` is not a member of a
    closed vocabulary, so validation would reject it. Dropping the key maps
    ``null`` onto the default of the field. A ``null`` for a required field then
    surfaces as "is required" instead of as a type mismatch.
    """
    if not isinstance(arguments, dict):
        return arguments
    return {key: value for key, value in arguments.items() if value is not None}


def _describe_problem(problem: Dict[str, Any], spec: ToolSpec) -> str:
    """Turn one validation problem into a sentence the model can act on."""
    field = ".".join(str(part) for part in problem.get("loc", ())) or "arguments"
    kind = problem.get("type", "")
    context = problem.get("ctx") or {}

    if kind == "literal_error" and "expected" in context:
        return f"{field} must be one of: {context['expected']}"
    if kind == "missing":
        return f"{field} is required"
    if kind == "extra_forbidden":
        allowed = ", ".join(spec.args_model.model_fields)
        hint = f"allowed arguments: {allowed}" if allowed else "it takes no arguments"
        return f"{field} is not an argument of this tool ({hint})"
    if kind == "model_type":
        return "arguments must be a JSON object"
    return f"{field}: {problem.get('msg', 'invalid value')}"


def _schema_error_content(spec: ToolSpec, error: ValidationError) -> Dict[str, Any]:
    """Build the result for arguments that do not fit the schema.

    It is the same ``filter_error`` envelope the task tools return for arguments
    that are well-formed but contradictory, so the model has one way to repair a
    call. The message names every offending field, and for a closed vocabulary
    it lists the allowed values.
    """
    problems = [
        _describe_problem(problem, spec)
        for problem in error.errors(include_url=False, include_context=True)
    ]
    return {
        "status": "filter_error",
        "data": {
            "error_code": "invalid_arguments",
            "message": "; ".join(problems) or "invalid arguments",
        },
    }


def _require_json_object_result(tool_name: str, result: Any) -> None:
    """Raise unless ``result`` is a tool result whose content is a JSON object.

    The content is sent to the model as JSON text and stored with the trace, so
    a value JSON cannot carry (a datetime, a set) is a fault of the tool. It is
    caught here, where it costs one tool call, not later, where it would cost
    the whole turn.
    """
    if not isinstance(result, LLMToolResult) or not isinstance(result.content, dict):
        raise TypeError(
            f"Executor of {tool_name!r} must return an LLMToolResult whose content is a dict"
        )
    json.dumps(result.content)


class ToolDispatcher:
    """Validate and execute tool calls against the registry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def dispatch(
        self, tool_calls: List[LLMToolCall], *, owner_user_id: str
    ) -> List[LLMToolResult]:
        """Run ``tool_calls`` in order and return one result per call."""
        logger.info(
            "Tool turn: %d call(s): %s",
            len(tool_calls),
            [tool_call.name for tool_call in tool_calls],
        )
        results: List[LLMToolResult] = []
        for tool_call in tool_calls:
            result = await self._run_one(tool_call, owner_user_id=owner_user_id)
            self._log_result(tool_call, result)
            results.append(result)
        return results

    async def _run_one(self, tool_call: LLMToolCall, *, owner_user_id: str) -> LLMToolResult:
        self._log_call(tool_call)

        spec = self._registry.get(tool_call.name)
        if spec is None:
            return LLMToolResult(
                name=tool_call.name, content={"error": f"Unknown tool: {tool_call.name}"}
            )

        try:
            args = spec.args_model.model_validate(_drop_null_arguments(tool_call.arguments))
        except ValidationError as error:
            content = _schema_error_content(spec, error)
            logger.warning(
                "Tool arguments rejected: %s (id=%s): %s",
                tool_call.name,
                tool_call.call_id,
                content["data"]["message"],
            )
            return LLMToolResult(name=tool_call.name, content=content)

        try:
            result = await spec.executor(args, owner_user_id=owner_user_id)
            _require_json_object_result(tool_call.name, result)
        except Exception as error:
            # A failing tool must not end the turn: the model is told that the
            # tool failed and the loop goes on. The traceback goes to the log;
            # the model and the API trace get the error class only, so internal
            # details never reach the conversation.
            logger.exception(
                "Tool execution failed: %s (id=%s)", tool_call.name, tool_call.call_id
            )
            return LLMToolResult(
                name=tool_call.name,
                content={"error": f"tool_execution_failed: {type(error).__name__}"},
            )
        return result

    @staticmethod
    def _log_call(tool_call: LLMToolCall) -> None:
        # Argument values and results are user content (task titles, remembered
        # facts), so only the debug level shows them; the info level names the
        # arguments and the outcome.
        arguments = tool_call.arguments
        argument_names = sorted(arguments) if isinstance(arguments, dict) else []
        logger.info(
            "Tool call %s (id=%s): arguments=%s",
            tool_call.name,
            tool_call.call_id,
            argument_names,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Tool call %s (id=%s) payload:\n%s",
                tool_call.name,
                tool_call.call_id,
                _render_for_log(arguments),
            )

    @staticmethod
    def _log_result(tool_call: LLMToolCall, result: LLMToolResult) -> None:
        logger.info(
            "Tool result %s (id=%s): %s",
            tool_call.name,
            tool_call.call_id,
            _summarise_result(result.content),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Tool result %s (id=%s) content:\n%s",
                tool_call.name,
                tool_call.call_id,
                _render_for_log(result.content),
            )
