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
from src.agent.adapters.llm.yandex_ai_studio_client import YandexAIStudioLLMClient
from src.agent.adapters.persistence.ydb_agent_repository_manager import YDBAgentRepositoryManager
from src.agent.adapters.providers.task_data_provider_adapter import TaskDataProviderAdapter
from src.agent.application.context.conversation_state_writer import ConversationStateWriter
from src.agent.application.loop.agent_tool_loop_service import AgentToolLoopService
from src.agent.application.loop.prompts.system_prompt import SYSTEM_PROMPT
from src.agent.application.loop.system_prompt_composer import SystemPromptComposer
from src.agent.application.memory.user_memory_writer import UserMemoryWriter
from src.agent.application.tools import build_tool_registry
from src.agent.application.tools.dispatcher import ToolDispatcher
from src.agent.application.use_cases.create_chat import CreateChatUseCase
from src.agent.application.use_cases.get_chat_history import GetChatHistoryUseCase
from src.agent.application.use_cases.list_chats import ListChatsUseCase
from src.agent.application.use_cases.list_memory import ListMemoryUseCase
from src.agent.application.use_cases.send_message import SendMessageUseCase
from src.agent.domain.entities.assistant import Assistant
from src.gateway.settings import Settings
from src.shared.infrastructure.database.ydb.connection import YDBConnection, open_ydb_connection
from src.shared.infrastructure.database.ydb.settings import YDBSettings
from src.tasks.adapters.persistence.ydb_tasks_repository_manager import YDBTasksRepositoryManager
from src.tasks.application.create_project import CreateProjectUseCase
from src.tasks.application.create_task import CreateTaskUseCase
from src.tasks.application.delete_task import DeleteTaskUseCase
from src.tasks.application.list_projects import ListProjectsUseCase
from src.tasks.application.query_tasks import QueryTasksUseCase
from src.tasks.application.seed_demo_workspace import SeedDemoWorkspaceUseCase
from src.tasks.application.update_task import UpdateTaskUseCase

ASSISTANT_NAME = "Task assistant"


async def _ydb_connection_resource(settings: YDBSettings) -> AsyncIterator[YDBConnection]:
    connection = await open_ydb_connection(settings)
    try:
        yield connection
    finally:
        await connection.close()


def _build_assistant(settings: Settings) -> Assistant:
    return Assistant(
        name=ASSISTANT_NAME,
        system_prompt=SYSTEM_PROMPT,
        model_name=settings.llm_model_name,
        temperature=settings.llm_temperature,
    )


class CoreContainer(containers.DeclarativeContainer):
    settings = providers.Dependency(instance_of=Settings)

    ydb_connection = providers.Resource(_ydb_connection_resource, settings.provided.ydb)
    ydb_pool = providers.Callable(lambda connection: connection.pool, ydb_connection)
    # Day boundaries of the task queries and the calendar block of the agent
    # must agree, so both read the same zone.
    timezone = providers.Singleton(ZoneInfo, settings.provided.agent_timezone)


class AccountsContainer(containers.DeclarativeContainer):
    core = providers.DependenciesContainer()

    user_repository = providers.Singleton(YDBUserRepository, pool=core.ydb_pool)

    create_user_use_case = providers.Factory(CreateUserUseCase, user_repository=user_repository)
    resolve_principal_use_case = providers.Factory(
        ResolvePrincipalUseCase, user_repository=user_repository
    )


class TasksContainer(containers.DeclarativeContainer):
    core = providers.DependenciesContainer()

    repository_manager = providers.Singleton(YDBTasksRepositoryManager, pool=core.ydb_pool)

    create_project_use_case = providers.Factory(
        CreateProjectUseCase, repositories=repository_manager
    )
    list_projects_use_case = providers.Factory(
        ListProjectsUseCase, repositories=repository_manager, timezone=core.timezone
    )
    create_task_use_case = providers.Factory(
        CreateTaskUseCase, repositories=repository_manager, timezone=core.timezone
    )
    update_task_use_case = providers.Factory(
        UpdateTaskUseCase, repositories=repository_manager, timezone=core.timezone
    )
    delete_task_use_case = providers.Factory(
        DeleteTaskUseCase, repositories=repository_manager, timezone=core.timezone
    )
    query_tasks_use_case = providers.Factory(
        QueryTasksUseCase, repositories=repository_manager, timezone=core.timezone
    )
    seed_demo_workspace = providers.Factory(
        SeedDemoWorkspaceUseCase, repositories=repository_manager, timezone=core.timezone
    )


class AgentContainer(containers.DeclarativeContainer):
    core = providers.DependenciesContainer()

    # The agent's view of task data. Bound by the application container; the
    # agent module itself never learns where the data comes from.
    task_data_provider = providers.Dependency()

    assistant = providers.Singleton(_build_assistant, core.settings)
    llm_client = providers.Singleton(
        YandexAIStudioLLMClient,
        folder_id=core.settings.provided.yc_folder_id,
        api_key=core.settings.provided.yc_api_key,
        iam_token=core.settings.provided.yc_iam_token,
    )
    repository_manager = providers.Singleton(YDBAgentRepositoryManager, pool=core.ydb_pool)

    user_memory_writer = providers.Factory(UserMemoryWriter, repository_manager=repository_manager)
    tool_registry = providers.Singleton(
        build_tool_registry,
        task_data_provider=task_data_provider,
        user_memory_writer=user_memory_writer,
    )
    tool_dispatcher = providers.Factory(ToolDispatcher, registry=tool_registry)
    system_prompt_composer = providers.Factory(SystemPromptComposer)
    agent_tool_loop_service = providers.Factory(
        AgentToolLoopService,
        llm_client=llm_client,
        tool_dispatcher=tool_dispatcher,
        tool_registry=tool_registry,
        system_prompt_composer=system_prompt_composer,
        max_iterations=core.settings.provided.agent_max_iterations,
        history_limit=core.settings.provided.agent_history_limit,
    )
    conversation_state_writer = providers.Factory(
        ConversationStateWriter, repository_manager=repository_manager
    )

    create_chat_use_case = providers.Factory(
        CreateChatUseCase, repository_manager=repository_manager
    )
    list_chats_use_case = providers.Factory(ListChatsUseCase, repository_manager=repository_manager)
    get_chat_history_use_case = providers.Factory(
        GetChatHistoryUseCase, repository_manager=repository_manager
    )
    list_memory_use_case = providers.Factory(
        ListMemoryUseCase, repository_manager=repository_manager
    )
    send_message_use_case = providers.Factory(
        SendMessageUseCase,
        repository_manager=repository_manager,
        agent_tool_loop_service=agent_tool_loop_service,
        conversation_state_writer=conversation_state_writer,
        assistant=assistant,
        timezone=core.timezone,
        history_limit=core.settings.provided.agent_history_limit,
    )


class AppContainer(containers.DeclarativeContainer):
    settings = providers.Dependency(instance_of=Settings)

    core = providers.Container(CoreContainer, settings=settings)
    accounts = providers.Container(AccountsContainer, core=core)
    tasks = providers.Container(TasksContainer, core=core)
    agent = providers.Container(
        AgentContainer,
        core=core,
        task_data_provider=providers.Factory(
            TaskDataProviderAdapter,
            list_projects=tasks.list_projects_use_case,
            query_tasks=tasks.query_tasks_use_case,
            create_task=tasks.create_task_use_case,
            update_task=tasks.update_task_use_case,
            delete_task=tasks.delete_task_use_case,
        ),
    )
