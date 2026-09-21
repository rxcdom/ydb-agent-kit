from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, List, Sequence

import ydb

from src.shared.domain.exceptions import PersistenceError

logger = logging.getLogger(__name__)

TransactionalOperation = Callable[[ydb.aio.QueryTxContext], Awaitable[Any]]


class YDBTransactionManager:
    """Runs several repository operations as one atomic transaction.

    Built on ``QuerySessionPool.retry_tx_async``, which acquires a session,
    opens the transaction, commits it when the callback returns, and replays
    the whole callback on retriable errors. Operations therefore have to be
    safe to run more than once.
    """

    def __init__(self, pool: ydb.aio.QuerySessionPool):
        self.pool = pool

    async def execute_in_transaction(
        self, operations: Sequence[TransactionalOperation]
    ) -> List[Any]:
        """Execute ``operations`` in order inside one transaction.

        Each operation receives the transaction context and must pass it to the
        repository methods it calls. If any operation raises, nothing is
        committed. Returns the results in the order of the operations.
        """

        async def run_all(tx: ydb.aio.QueryTxContext) -> List[Any]:
            try:
                return [await operation(tx) for operation in operations]
            except BaseException:
                # Roll back before the error propagates. On a non-retriable
                # failure the server-side transaction would otherwise stay open
                # until it is garbage-collected, leaking the session under load.
                try:
                    await tx.rollback()
                except ydb.Error as rollback_error:
                    logger.warning(
                        "Rollback after a failed transaction did not succeed: %s", rollback_error
                    )
                raise

        try:
            return await self.pool.retry_tx_async(run_all)
        except ydb.Error as error:
            raise PersistenceError(f"YDB transaction failed: {error}") from error
