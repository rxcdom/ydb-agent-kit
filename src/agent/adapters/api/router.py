"""HTTP face of the agent module: chats, messages and the memory vault.

Handlers take the owner from the authenticated principal and never from the
path or the body. They contain no ``try/except``: domain errors propagate to
the gateway, which maps them to HTTP responses in one place.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field

from src.accounts.adapters.api.auth import get_authorized_user
from src.agent.application.use_cases.create_chat import CreateChatUseCase
from src.agent.application.use_cases.get_chat_history import (
    DEFAULT_HISTORY_PAGE_SIZE,
    MAX_HISTORY_PAGE_SIZE,
    GetChatHistoryUseCase,
)
from src.agent.application.use_cases.list_chats import ListChatsUseCase
from src.agent.application.use_cases.list_memory import ListMemoryUseCase
from src.agent.application.use_cases.send_message import (
    MESSAGE_CONTENT_MAX_LENGTH,
    SendMessageUseCase,
    trace_to_dict,
)
from src.agent.domain.entities.chat import CHAT_TITLE_MAX_LENGTH, Chat
from src.agent.domain.entities.message import Message
from src.agent.domain.value_objects.chat_id import ChatId
from src.shared.domain.dtos.authorized_user import AuthorizedUser

chats_router = APIRouter(prefix="/chats", tags=["chats"])
memory_router = APIRouter(prefix="/memory", tags=["memory"])


class CreateChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(default=None, max_length=CHAT_TITLE_MAX_LENGTH)


class CreatedChatResponse(BaseModel):
    chat_id: str
    created_at: datetime


class ChatResponse(BaseModel):
    chat_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, chat: Chat) -> "ChatResponse":
        return cls(
            chat_id=str(chat.chat_id),
            title=chat.title,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
        )


class ChatListResponse(BaseModel):
    chats: List[ChatResponse]


class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=MESSAGE_CONTENT_MAX_LENGTH)


class MessageResponse(BaseModel):
    message_id: str
    chat_id: str
    role: str
    content: str
    tokens: int
    status: str
    created_at: datetime

    @classmethod
    def of(cls, message: Message) -> "MessageResponse":
        return cls(
            message_id=str(message.message_id),
            chat_id=str(message.chat_id),
            role=message.role.value,
            content=message.content,
            tokens=message.tokens,
            status=message.status.value,
            created_at=message.created_at,
        )


class SendMessageResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
    # The tool-call trace of the turn: model_name, total_tokens and request_flow.
    debug: Dict[str, Any]


class MessageListResponse(BaseModel):
    messages: List[MessageResponse]
    total_count: int
    has_more: bool


class MemoryResponse(BaseModel):
    memory_id: str
    content: str
    topic: Optional[str]
    created_at: datetime
    updated_at: datetime


class MemoryListResponse(BaseModel):
    count: int
    memories: List[MemoryResponse]


@chats_router.post("", status_code=status.HTTP_201_CREATED, response_model=CreatedChatResponse)
@inject
async def create_chat(
    request: Optional[CreateChatRequest] = None,
    principal: AuthorizedUser = Depends(get_authorized_user),
    use_case: CreateChatUseCase = Depends(Provide["agent.create_chat_use_case"]),
) -> CreatedChatResponse:
    title = request.title if request is not None else None
    chat = await use_case.execute(principal.user_id, title)
    return CreatedChatResponse(chat_id=str(chat.chat_id), created_at=chat.created_at)


@chats_router.get("", response_model=ChatListResponse)
@inject
async def list_chats(
    principal: AuthorizedUser = Depends(get_authorized_user),
    use_case: ListChatsUseCase = Depends(Provide["agent.list_chats_use_case"]),
) -> ChatListResponse:
    chats = await use_case.execute(principal.user_id)
    return ChatListResponse(chats=[ChatResponse.of(chat) for chat in chats])


@chats_router.post("/{chat_id}/messages", response_model=SendMessageResponse)
@inject
async def send_message(
    chat_id: UUID,
    request: SendMessageRequest,
    principal: AuthorizedUser = Depends(get_authorized_user),
    use_case: SendMessageUseCase = Depends(Provide["agent.send_message_use_case"]),
) -> SendMessageResponse:
    """Run one agent turn. The response carries both messages and the turn's tool-call trace."""
    result = await use_case.execute(principal.user_id, ChatId(chat_id), request.content)
    return SendMessageResponse(
        user_message=MessageResponse.of(result.user_message),
        assistant_message=MessageResponse.of(result.assistant_message),
        debug=trace_to_dict(result.debug),
    )


@chats_router.get("/{chat_id}/messages", response_model=MessageListResponse)
@inject
async def get_chat_history(
    chat_id: UUID,
    limit: int = Query(default=DEFAULT_HISTORY_PAGE_SIZE, ge=1, le=MAX_HISTORY_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    principal: AuthorizedUser = Depends(get_authorized_user),
    use_case: GetChatHistoryUseCase = Depends(Provide["agent.get_chat_history_use_case"]),
) -> MessageListResponse:
    page = await use_case.execute(principal.user_id, ChatId(chat_id), limit, offset)
    return MessageListResponse(
        messages=[MessageResponse.of(message) for message in page.messages],
        total_count=page.total_count,
        has_more=page.has_more,
    )


@memory_router.get("", response_model=MemoryListResponse)
@inject
async def list_memory(
    principal: AuthorizedUser = Depends(get_authorized_user),
    use_case: ListMemoryUseCase = Depends(Provide["agent.list_memory_use_case"]),
) -> MemoryListResponse:
    memories = await use_case.execute(principal.user_id)
    return MemoryListResponse(
        count=len(memories),
        memories=[
            MemoryResponse(
                memory_id=memory.memory_id,
                content=memory.content,
                topic=memory.topic,
                created_at=memory.created_at,
                updated_at=memory.updated_at,
            )
            for memory in memories
        ],
    )


router = APIRouter()
router.include_router(chats_router)
router.include_router(memory_router)
