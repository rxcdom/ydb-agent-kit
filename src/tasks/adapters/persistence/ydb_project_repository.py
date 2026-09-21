from __future__ import annotations

from typing import Any, Dict, List, Optional
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
from src.tasks.domain.entities.project import Project
from src.tasks.ports.project_repository import ProjectRepository

TABLE_NAME = "projects"
INDEX_USER_ID = "idx_projects_user_id"

_COLUMNS = "project_id, user_id, name, description, created_at, updated_at"


class ProjectMapper(DataMapper[Project]):
    def to_domain(self, row: Any) -> Project:
        return Project(
            project_id=UUID(safe_decode(read_row_value(row, "project_id"), "project_id")),
            user_id=UserId.from_string(safe_decode(read_row_value(row, "user_id"), "user_id")),
            name=safe_decode(read_row_value(row, "name"), "name"),
            description=optional_decode(read_row_value(row, "description"), "description"),
            created_at=normalize_datetime_to_utc(read_row_value(row, "created_at")),
            updated_at=normalize_datetime_to_utc(read_row_value(row, "updated_at")),
        )

    def to_ydb_params(self, entity: Project) -> Dict[str, Any]:
        return {
            "$project_id": str(entity.project_id),
            "$user_id": str(entity.user_id),
            "$name": entity.name,
            "$description": entity.description,
            "$created_at": entity.created_at,
            "$updated_at": entity.updated_at,
        }

    def get_ydb_type_map(self) -> Dict[str, Any]:
        return {
            "$project_id": ydb.PrimitiveType.Utf8,
            "$user_id": ydb.PrimitiveType.Utf8,
            "$name": ydb.PrimitiveType.Utf8,
            "$description": ydb.OptionalType(ydb.PrimitiveType.Utf8),
            "$created_at": ydb.PrimitiveType.Timestamp,
            "$updated_at": ydb.PrimitiveType.Timestamp,
        }


class YDBProjectRepository(YDBRepository[Project], ProjectRepository):
    """Projects in YDB. Every statement is scoped to the owner."""

    def __init__(self, pool: ydb.aio.QuerySessionPool, table_name: str = TABLE_NAME):
        super().__init__(pool, ProjectMapper(), table_name)

    async def save(self, project: Project, tx: Any = None) -> None:
        await self._execute_rows(
            self._build_save_query(project), self._mapper.to_ydb_params(project), tx
        )

    async def find_by_id(self, user_id: UserId, project_id: UUID) -> Optional[Project]:
        # Primary-key read; the owner condition keeps another owner's row unreachable.
        query = f"""
        {self._build_declare_block(["$project_id", "$user_id"])}

        SELECT {_COLUMNS}
        FROM {self._table_path}
        WHERE project_id = $project_id AND user_id = $user_id;
        """
        projects = await self._execute_query(
            query, {"$project_id": str(project_id), "$user_id": str(user_id)}
        )
        return projects[0] if projects else None

    async def list_by_user(self, user_id: UserId) -> List[Project]:
        # Uses idx_projects_user_id: without VIEW the owner filter scans the table.
        query = f"""
        {self._build_declare_block(["$user_id"])}

        SELECT {_COLUMNS}
        FROM {self._table_path} VIEW {INDEX_USER_ID}
        WHERE user_id = $user_id
        ORDER BY name, project_id;
        """
        return await self._execute_query(query, {"$user_id": str(user_id)})
