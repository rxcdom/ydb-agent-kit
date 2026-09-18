import pytest
import ydb

from src.shared.domain.exceptions import PersistenceError
from src.shared.infrastructure.database.ydb.transaction_manager import YDBTransactionManager


class _FakeTx:
    def __init__(self, rollback_error=None):
        self.rolled_back = False
        self._rollback_error = rollback_error

    async def rollback(self):
        self.rolled_back = True
        if self._rollback_error:
            raise self._rollback_error


class _FakePool:
    def __init__(self, tx, error=None):
        self.tx = tx
        self._error = error

    async def retry_tx_async(self, callee):
        if self._error:
            raise self._error
        return await callee(self.tx)


async def test_results_come_back_in_operation_order_with_the_shared_context():
    tx = _FakeTx()
    seen = []

    async def first(context):
        seen.append(context)
        return "first"

    async def second(context):
        seen.append(context)
        return "second"

    results = await YDBTransactionManager(_FakePool(tx)).execute_in_transaction([first, second])

    assert results == ["first", "second"]
    assert seen == [tx, tx]
    assert tx.rolled_back is False


async def test_failing_operation_rolls_back_and_stops_the_sequence():
    tx = _FakeTx()
    executed = []

    async def failing(_context):
        raise RuntimeError("second write failed")

    async def never_runs(_context):
        executed.append("ran")

    with pytest.raises(RuntimeError, match="second write failed"):
        await YDBTransactionManager(_FakePool(tx)).execute_in_transaction([failing, never_runs])

    assert tx.rolled_back is True
    assert executed == []


async def test_rollback_failure_does_not_mask_the_original_error():
    tx = _FakeTx(rollback_error=ydb.issues.BadSession("session is gone"))

    async def failing(_context):
        raise RuntimeError("original")

    with pytest.raises(RuntimeError, match="original"):
        await YDBTransactionManager(_FakePool(tx)).execute_in_transaction([failing])


async def test_sdk_failure_surfaces_as_persistence_error():
    pool = _FakePool(_FakeTx(), error=ydb.issues.Aborted("lock invalidated"))

    with pytest.raises(PersistenceError, match="transaction failed"):
        await YDBTransactionManager(pool).execute_in_transaction([])
