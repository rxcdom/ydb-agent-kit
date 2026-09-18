from __future__ import annotations

import abc
from typing import Dict, List

import ydb


class Migration(abc.ABC):
    """Base class of every migration.

    A migration defines:
    - ``version``: a 14-digit timestamp string such as ``"20260901000000"``;
      it must equal the prefix of the file name and be unique across modules.
    - ``description``: a short human-readable summary.
    - ``up()``: the coroutine that applies the change.

    A migration may also override ``get_artifacts()`` to declare what it
    creates. Declared artifacts are checked against the live schema right after
    ``up()`` returns, and a mismatch marks the migration as failed.
    """

    version: str
    description: str

    @abc.abstractmethod
    async def up(self, pool: ydb.aio.QuerySessionPool) -> None:  # pragma: no cover
        """Apply the migration.

        Must be idempotent: an interrupted or failed migration is run again
        from the start on the next ``up`` command.
        """
        raise NotImplementedError

    async def down(self, pool: ydb.aio.QuerySessionPool) -> None:  # pragma: no cover
        """Revert the migration. Optional; the framework never calls it."""
        return None

    def get_artifacts(self) -> Dict[str, List]:
        """Describe what this migration creates.

        Optional keys of the returned dictionary:
        - ``"tables"``: table names, e.g. ``["users"]``
        - ``"indexes"``: ``(table, index)`` pairs,
          e.g. ``[("chats", "idx_chats_user_id")]``
        - ``"columns"``: ``(table, column)`` or ``(table, column, type)``
          tuples, e.g. ``[("users", "display_name", "Utf8?")]``

        A declared column type is compared with the type the Table Service
        reports. YDB columns are nullable unless created ``NOT NULL``, so a
        column created as ``name Utf8`` is reported as ``Utf8?``; declare
        ``"Utf8"`` only for a ``NOT NULL`` column.

        The default declares nothing, which skips verification.
        """
        return {}
