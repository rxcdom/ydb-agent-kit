from __future__ import annotations

from typing import Any, List, Sequence

import ydb

from src.shared.infrastructure.database.ydb.transaction_manager import YDBTransactionManager
from src.tasks.adapters.persistence.ydb_project_repository import YDBProjectRepository
from src.tasks.adapters.persistence.ydb_task_repository import YDBTaskRepository
from src.tasks.ports.repository_manager import TasksRepositoryManager, TransactionalOperation


class YDBTasksRepositoryManager(TasksRepositoryManager):
    """The tasks module's repositories on one session pool, with one transaction manager."""

    def __init__(self, pool: ydb.aio.QuerySessionPool):
        self._transaction_manager = YDBTransactionManager(pool)
        self._projects = YDBProjectRepository(pool)
        self._tasks = YDBTaskRepository(pool)

    @property
    def projects(self) -> YDBProjectRepository:
        return self._projects

    @property
    def tasks(self) -> YDBTaskRepository:
        return self._tasks

    async def execute_in_transaction(
        self, operations: Sequence[TransactionalOperation]
    ) -> List[Any]:
        return await self._transaction_manager.execute_in_transaction(operations)
