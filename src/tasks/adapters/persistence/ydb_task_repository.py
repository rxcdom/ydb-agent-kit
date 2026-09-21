from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import UUID

import ydb

from src.shared.domain.value_objects.user_id import UserId
from src.shared.infrastructure.database.ydb.repository import DataMapper, YDBRepository
from src.shared.infrastructure.database.ydb.utils import (
    normalize_datetime_to_utc,
    optional_decode,
    read_row_value,
    safe_decode,
)
from src.tasks.domain.entities.task import Task
from src.tasks.domain.value_objects.date_axis import DateAxis
from src.tasks.domain.value_objects.task_priority import TaskPriority
from src.tasks.domain.value_objects.task_status import TaskStatus
from src.tasks.ports.task_repository import ProjectActivity, TaskRepository, TaskSearchCriteria

TABLE_NAME = "tasks"
INDEX_USER_CREATED = "idx_tasks_user_created"
INDEX_USER_DUE = "idx_tasks_user_due"
INDEX_PROJECT_ID = "idx_tasks_project_id"

_COLUMNS = (
    "task_id, user_id, project_id, title, notes, status, priority, "
    "due_at, completed_at, created_at, updated_at"
)

_AXIS_COLUMNS = {
    DateAxis.CREATED: "created_at",
    DateAxis.DUE: "due_at",
    DateAxis.COMPLETED: "completed_at",
}

# A secondary index is consulted only when the query names it, so every axis
# names one. completed_at has no index of its own: the owner prefix of
# idx_tasks_user_created keeps the read inside the owner's rows, and the
# completion window is filtered from there.
_AXIS_INDEXES = {
    DateAxis.CREATED: INDEX_USER_CREATED,
    DateAxis.DUE: INDEX_USER_DUE,
    DateAxis.COMPLETED: INDEX_USER_CREATED,
}

# "%", "_" and "\" are not accepted as the LIKE escape character.
_LIKE_ESCAPE = "!"


def _numbered(prefix: str, count: int) -> List[str]:
    return [f"${prefix}_{position}" for position in range(count)]


_STATUS_PARAMS = _numbered("status", len(TaskStatus))
_PRIORITY_PARAMS = _numbered("priority", len(TaskPriority))


def _optional_uuid(value: Any, field_name: str) -> Optional[UUID]:
    text = optional_decode(value, field_name)
    return None if text is None else UUID(text)


class TaskMapper(DataMapper[Task]):
    def to_domain(self, row: Any) -> Task:
        return Task(
            task_id=UUID(safe_decode(read_row_value(row, "task_id"), "task_id")),
            user_id=UserId.from_string(safe_decode(read_row_value(row, "user_id"), "user_id")),
            project_id=_optional_uuid(read_row_value(row, "project_id"), "project_id"),
            title=safe_decode(read_row_value(row, "title"), "title"),
            notes=optional_decode(read_row_value(row, "notes"), "notes"),
            status=TaskStatus(safe_decode(read_row_value(row, "status"), "status")),
            priority=TaskPriority(safe_decode(read_row_value(row, "priority"), "priority")),
            due_at=normalize_datetime_to_utc(read_row_value(row, "due_at")),
            completed_at=normalize_datetime_to_utc(read_row_value(row, "completed_at")),
            created_at=normalize_datetime_to_utc(read_row_value(row, "created_at")),
            updated_at=normalize_datetime_to_utc(read_row_value(row, "updated_at")),
        )

    def to_ydb_params(self, entity: Task) -> Dict[str, Any]:
        # Fields are plain attributes, so the pair (status, completed_at) is
        # checked once more here: a contradictory pair is never written.
        entity.check_invariants()
        return {
            "$task_id": str(entity.task_id),
            "$user_id": str(entity.user_id),
            "$project_id": None if entity.project_id is None else str(entity.project_id),
            "$title": entity.title,
            "$notes": entity.notes,
            "$status": entity.status.value,
            "$priority": entity.priority.value,
            "$due_at": entity.due_at,
            "$completed_at": entity.completed_at,
            "$created_at": entity.created_at,
            "$updated_at": entity.updated_at,
        }

    def get_column_params(self) -> List[str]:
        return [
            "$task_id",
            "$user_id",
            "$project_id",
            "$title",
            "$notes",
            "$status",
            "$priority",
            "$due_at",
            "$completed_at",
            "$created_at",
            "$updated_at",
        ]

    def get_ydb_type_map(self) -> Dict[str, Any]:
        type_map: Dict[str, Any] = {
            "$task_id": ydb.PrimitiveType.Utf8,
            "$user_id": ydb.PrimitiveType.Utf8,
            "$project_id": ydb.OptionalType(ydb.PrimitiveType.Utf8),
            "$title": ydb.PrimitiveType.Utf8,
            "$notes": ydb.OptionalType(ydb.PrimitiveType.Utf8),
            "$status": ydb.PrimitiveType.Utf8,
            "$priority": ydb.PrimitiveType.Utf8,
            "$due_at": ydb.OptionalType(ydb.PrimitiveType.Timestamp),
            "$completed_at": ydb.OptionalType(ydb.PrimitiveType.Timestamp),
            "$created_at": ydb.PrimitiveType.Timestamp,
            "$updated_at": ydb.PrimitiveType.Timestamp,
            # Query-only parameters.
            "$window_start": ydb.PrimitiveType.Timestamp,
            "$window_end": ydb.PrimitiveType.Timestamp,
            "$text_pattern": ydb.PrimitiveType.Utf8,
            "$limit": ydb.PrimitiveType.Uint64,
            "$offset": ydb.PrimitiveType.Uint64,
        }
        # One slot per member of each closed vocabulary, for IN lists.
        for name in _STATUS_PARAMS + _PRIORITY_PARAMS:
            type_map[name] = ydb.PrimitiveType.Utf8
        return type_map


class YDBTaskRepository(YDBRepository[Task], TaskRepository):
    """Tasks in YDB. Every statement is scoped to the owner."""

    def __init__(self, pool: ydb.aio.QuerySessionPool, table_name: str = TABLE_NAME):
        super().__init__(pool, TaskMapper(), table_name)

    async def save(self, task: Task, tx: Any = None) -> None:
        await self._execute_rows(self._build_save_query(task), self._mapper.to_ydb_params(task), tx)

    async def find_by_id(self, user_id: UserId, task_id: UUID) -> Optional[Task]:
        # Primary-key read; the owner condition keeps another owner's row unreachable.
        query = f"""
        {self._build_declare_block(["$task_id", "$user_id"])}

        SELECT {_COLUMNS}
        FROM {self._table_path}
        WHERE task_id = $task_id AND user_id = $user_id;
        """
        tasks = await self._execute_query(
            query, {"$task_id": str(task_id), "$user_id": str(user_id)}
        )
        return tasks[0] if tasks else None

    async def delete(self, user_id: UserId, task_id: UUID) -> None:
        # Primary-key delete guarded by the owner: a foreign id removes nothing.
        query = f"""
        {self._build_declare_block(["$task_id", "$user_id"])}

        DELETE FROM {self._table_path}
        WHERE task_id = $task_id AND user_id = $user_id;
        """
        await self._execute_rows(query, {"$task_id": str(task_id), "$user_id": str(user_id)})

    async def list_by_user(self, user_id: UserId) -> List[Task]:
        # Uses idx_tasks_user_created: the owner prefix selects the rows and
        # the index order is the requested order.
        query = f"""
        {self._build_declare_block(["$user_id"])}

        SELECT {_COLUMNS}
        FROM {self._table_path} VIEW {INDEX_USER_CREATED}
        WHERE user_id = $user_id
        ORDER BY created_at, task_id;
        """
        return await self._execute_query(query, {"$user_id": str(user_id)})

    async def list_by_project(self, user_id: UserId, project_id: UUID) -> List[Task]:
        # Uses idx_tasks_project_id; the owner condition stays as a guard.
        query = f"""
        {self._build_declare_block(["$project_id", "$user_id"])}

        SELECT {_COLUMNS}
        FROM {self._table_path} VIEW {INDEX_PROJECT_ID}
        WHERE project_id = $project_id AND user_id = $user_id
        ORDER BY created_at, task_id;
        """
        return await self._execute_query(
            query, {"$project_id": str(project_id), "$user_id": str(user_id)}
        )

    async def search(
        self,
        user_id: UserId,
        criteria: TaskSearchCriteria,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
        descending: bool = False,
    ) -> List[Task]:
        # Uses the index of the criteria's axis (see _AXIS_INDEXES): the owner
        # and the window form a range on it.
        if limit is None and offset:
            raise ValueError("an offset needs a limit")

        conditions, params = self._conditions(user_id, criteria, with_axis_date=True)
        column = _AXIS_COLUMNS[criteria.axis]
        direction = "DESC" if descending else "ASC"
        paging = ""
        if limit is not None:
            paging = "LIMIT $limit OFFSET $offset"
            params["$limit"] = limit
            params["$offset"] = offset

        query = f"""
        {self._build_declare_block(list(params))}

        SELECT {_COLUMNS}
        FROM {self._table_path} VIEW {_AXIS_INDEXES[criteria.axis]}
        WHERE {conditions}
        ORDER BY {column} {direction}, task_id {direction}
        {paging};
        """
        return await self._execute_query(query, params)

    async def count(self, user_id: UserId, criteria: TaskSearchCriteria) -> int:
        # Uses the index of the criteria's axis, like search.
        conditions, params = self._conditions(user_id, criteria, with_axis_date=True)
        return await self._count(criteria.axis, conditions, params)

    async def count_without_axis_date(self, user_id: UserId, criteria: TaskSearchCriteria) -> int:
        # Uses the index of the criteria's axis: rows without a value on it sit
        # together at the start of the owner's range.
        conditions, params = self._conditions(user_id, criteria, with_axis_date=False)
        return await self._count(criteria.axis, conditions, params)

    async def date_bounds(
        self, user_id: UserId, axis: DateAxis
    ) -> Optional[Tuple[datetime, datetime]]:
        # Uses the index of the axis. MIN and MAX skip rows without a value.
        column = _AXIS_COLUMNS[axis]
        query = f"""
        {self._build_declare_block(["$user_id"])}

        SELECT MIN({column}) AS first_value, MAX({column}) AS last_value
        FROM {self._table_path} VIEW {_AXIS_INDEXES[axis]}
        WHERE user_id = $user_id;
        """
        rows = await self._execute_rows(query, {"$user_id": str(user_id)})
        if not rows:
            return None
        first_value = normalize_datetime_to_utc(read_row_value(rows[0], "first_value"))
        last_value = normalize_datetime_to_utc(read_row_value(rows[0], "last_value"))
        if first_value is None or last_value is None:
            return None
        return first_value, last_value

    async def summarise_by_project(self, user_id: UserId) -> List[ProjectActivity]:
        # Uses idx_tasks_user_created for the owner prefix.
        query = f"""
        {self._build_declare_block(["$user_id"])}

        SELECT
            project_id,
            status,
            COUNT(*) AS task_count,
            MIN(created_at) AS first_created_at,
            MAX(updated_at) AS last_updated_at
        FROM {self._table_path} VIEW {INDEX_USER_CREATED}
        WHERE user_id = $user_id
        GROUP BY project_id, status;
        """
        rows = await self._execute_rows(query, {"$user_id": str(user_id)})
        return [
            ProjectActivity(
                project_id=_optional_uuid(read_row_value(row, "project_id"), "project_id"),
                status=TaskStatus(safe_decode(read_row_value(row, "status"), "status")),
                task_count=int(read_row_value(row, "task_count")),
                first_created_at=normalize_datetime_to_utc(
                    read_row_value(row, "first_created_at")
                ),
                last_updated_at=normalize_datetime_to_utc(read_row_value(row, "last_updated_at")),
            )
            for row in rows
        ]

    async def _count(self, axis: DateAxis, conditions: str, params: Dict[str, Any]) -> int:
        query = f"""
        {self._build_declare_block(list(params))}

        SELECT COUNT(*) AS total
        FROM {self._table_path} VIEW {_AXIS_INDEXES[axis]}
        WHERE {conditions};
        """
        rows = await self._execute_rows(query, params)
        return int(read_row_value(rows[0], "total")) if rows else 0

    def _conditions(
        self, user_id: UserId, criteria: TaskSearchCriteria, *, with_axis_date: bool
    ) -> Tuple[str, Dict[str, Any]]:
        """The WHERE clause of a criteria and its parameters; the owner always comes first.

        With ``with_axis_date`` the rows must have a value on the axis, inside the
        window when there is one. Without it they must lack that value, and the
        window, which cannot apply to them, is left out.
        """
        column = _AXIS_COLUMNS[criteria.axis]
        conditions = ["user_id = $user_id"]
        params: Dict[str, Any] = {"$user_id": str(user_id)}

        if with_axis_date:
            conditions.append(f"{column} IS NOT NULL")
            if criteria.window_start is not None:
                conditions.append(f"{column} >= $window_start")
                params["$window_start"] = criteria.window_start
            if criteria.window_end is not None:
                conditions.append(f"{column} < $window_end")
                params["$window_end"] = criteria.window_end
        else:
            conditions.append(f"{column} IS NULL")

        if criteria.project_id is not None:
            conditions.append("project_id = $project_id")
            params["$project_id"] = str(criteria.project_id)
        if criteria.statuses is not None:
            conditions.append(_membership("status", _STATUS_PARAMS, criteria.statuses, params))
        if criteria.priorities is not None:
            conditions.append(
                _membership("priority", _PRIORITY_PARAMS, criteria.priorities, params)
            )
        if criteria.text is not None:
            # Both sides are lower-cased; wildcards in the text are escaped so
            # it is matched as a literal substring of the title or the notes.
            params["$text_pattern"] = f"%{_escape_like(criteria.text.lower())}%"
            conditions.append(
                f'(Unicode::ToLower(title) LIKE $text_pattern ESCAPE "{_LIKE_ESCAPE}"'
                f' OR Unicode::ToLower(notes) LIKE $text_pattern ESCAPE "{_LIKE_ESCAPE}")'
            )

        return "\n          AND ".join(conditions), params


def _membership(
    column: str, slots: List[str], members: Iterable[Any], params: Dict[str, Any]
) -> str:
    """``column IN ($slot_0, ...)`` over the values of enum members, in a stable order."""
    values = sorted(member.value for member in members)
    used = slots[: len(values)]
    params.update(zip(used, values))
    return f"{column} IN ({', '.join(used)})"


def _escape_like(text: str) -> str:
    escaped = text.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
    return escaped.replace("%", f"{_LIKE_ESCAPE}%").replace("_", f"{_LIKE_ESCAPE}_")
