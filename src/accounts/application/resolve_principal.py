from __future__ import annotations

from typing import Optional

from src.accounts.domain.entities.user import User
from src.accounts.domain.exceptions import InvalidCredentialsError
from src.accounts.ports.user_repository import UserRepository
from src.shared.domain.dtos.authorized_user import AuthorizedUser
from src.shared.domain.value_objects.user_id import UserId


class ResolvePrincipalUseCase:
    """Turns a bearer credential into the authenticated principal.

    The credential is the user id itself, so resolving it is one primary-key
    read. Absent, malformed and unknown credentials are indistinguishable to
    the caller.
    """

    def __init__(self, user_repository: UserRepository):
        self._user_repository = user_repository

    async def execute(self, credential: Optional[str]) -> AuthorizedUser:
        user = await self.load_user(credential)
        return AuthorizedUser(user_id=user.user_id)

    async def load_user(self, credential: Optional[str]) -> User:
        if not credential:
            raise InvalidCredentialsError("missing bearer credential")
        try:
            user_id = UserId.from_string(credential.strip())
        except ValueError as error:
            raise InvalidCredentialsError("malformed bearer credential") from error

        user = await self._user_repository.find_by_id(user_id)
        if user is None:
            raise InvalidCredentialsError("unknown bearer credential")
        return user
