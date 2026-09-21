"""The checks every agent-facing envelope has to pass, whatever its outcome."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Set

from src.tasks.application.outcome import Outcome

# The agent addresses rows by human text only, so no envelope may hand it an id.
ID_LIKE_KEYS = frozenset({"id", "task_id", "project_id", "user_id", "owner_id"})


def data_of(outcome: Outcome, status: str, *, ids: Iterable[object] = ()) -> Dict[str, Any]:
    """The ``data`` of an outcome with the expected status, after the envelope checks.

    ``ids`` are identifiers known to the test (owners, rows); none of them may
    appear anywhere in the serialised envelope, under any key.
    """
    assert isinstance(outcome, Outcome)
    envelope = outcome.to_envelope()

    assert set(envelope) == {"status", "data"}
    assert envelope["status"] == status, envelope
    assert isinstance(envelope["data"], dict)

    serialised = json.dumps(envelope)
    assert json.loads(serialised) == envelope, "the envelope holds a value JSON cannot round-trip"

    leaked_keys = _keys_of(envelope) & ID_LIKE_KEYS
    assert not leaked_keys, f"id-like keys in the envelope: {sorted(leaked_keys)}"
    for identifier in ids:
        assert str(identifier) not in serialised, f"the id {identifier} leaked into the envelope"

    return envelope["data"]


def _keys_of(value: Any) -> Set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for nested in value.values():
            keys |= _keys_of(nested)
        return keys
    if isinstance(value, list):
        keys = set()
        for nested in value:
            keys |= _keys_of(nested)
        return keys
    return set()
