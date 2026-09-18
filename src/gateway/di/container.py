"""Composition root: the only place where modules are wired to each other.

Each module gets its own container that knows nothing about the others. The
application container instantiates them and closes the one seam between
modules: the agent's ``TaskDataProvider`` port is a declared-but-unbound
dependency inside the agent container and is bound here to an adapter built on
the tasks use cases.
"""
from __future__ import annotations

from typing import AsyncIterator
from zoneinfo import ZoneInfo

from dependency_injector import containers, providers

from src.accounts.adapters.persistence.ydb_user_repository import YDBUserRepository
from src.accounts.application.create_user import CreateUserUseCase
from src.accounts.application.resolve_principal import ResolvePrincipalUseCase
from src.gateway.settings import Settings
from src.shared.infrastructure.database.ydb.connection import YDBConnection, open_ydb_connection
from src.shared.infrastructure.database.ydb.settings import YDBSettings


async def _ydb_connection_resource(settings: YDBSettings) -> AsyncIterator[YDBConnection]:
    connection = await open_ydb_connection(settings)
    try:
        yield connection
    finally:
        await connection.close()


class CoreContainer(containers.DeclarativeContainer):
    settings = providers.Dependency(instance_of=Settings)

    ydb_connection = providers.Resource(_ydb_connection_resource, settings.provided.ydb)
    ydb_pool = providers.Callable(lambda connection: connection.pool, ydb_connection)
    timezone = providers.Singleton(ZoneInfo, settings.provided.agent_timezone)


class AccountsContainer(containers.DeclarativeContainer):
    core = providers.DependenciesContainer()

    user_repository = providers.Singleton(YDBUserRepository, pool=core.ydb_pool)

    create_user_use_case = providers.Factory(CreateUserUseCase, user_repository=user_repository)
    resolve_principal_use_case = providers.Factory(
        ResolvePrincipalUseCase, user_repository=user_repository
    )


class AppContainer(containers.DeclarativeContainer):
    settings = providers.Dependency(instance_of=Settings)

    core = providers.Container(CoreContainer, settings=settings)
    accounts = providers.Container(AccountsContainer, core=core)
