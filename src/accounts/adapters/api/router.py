from __future__ import annotations

from datetime import datetime
from typing import Optional

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field

from src.accounts.adapters.api.auth import get_current_user
from src.accounts.application.create_user import CreateUserUseCase
from src.accounts.domain.entities.user import MAX_DISPLAY_NAME_LENGTH, User

router = APIRouter(prefix="/users", tags=["users"])


class CreateUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: Optional[str] = Field(default=None, max_length=MAX_DISPLAY_NAME_LENGTH)


class CreatedUserResponse(BaseModel):
    user_id: str
    created_at: datetime


class UserResponse(BaseModel):
    user_id: str
    display_name: Optional[str]
    created_at: datetime


@router.post("", status_code=status.HTTP_201_CREATED, response_model=CreatedUserResponse)
@inject
async def create_user(
    request: Optional[CreateUserRequest] = None,
    create_user_use_case: CreateUserUseCase = Depends(Provide["accounts.create_user_use_case"]),
) -> CreatedUserResponse:
    """Create a user. The returned ``user_id`` is the bearer credential of every other call."""
    display_name = request.display_name if request is not None else None
    user = await create_user_use_case.execute(display_name)
    return CreatedUserResponse(user_id=str(user.user_id), created_at=user.created_at)


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        user_id=str(user.user_id), display_name=user.display_name, created_at=user.created_at
    )
