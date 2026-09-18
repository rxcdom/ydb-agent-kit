"""Bearer-credential dependency shared by every authenticated router.

This is a deliberate simplification: the credential is the user id. The
dependency still goes through a standard ``Authorization: Bearer`` header and
the principal is still resolved server-side, so replacing this module with real
authentication does not change any handler.
"""
from __future__ import annotations

from typing import Optional

from dependency_injector.wiring import Provide, inject
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.accounts.application.resolve_principal import ResolvePrincipalUseCase
from src.accounts.domain.entities.user import User
from src.shared.domain.dtos.authorized_user import AuthorizedUser

# auto_error is off so that a missing header takes the same path as a bad one.
_bearer_scheme = HTTPBearer(auto_error=False, description="The user id returned by POST /users")


def _credential_of(header: Optional[HTTPAuthorizationCredentials]) -> Optional[str]:
    return header.credentials if header is not None else None


@inject
async def get_authorized_user(
    header: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    resolve_principal: ResolvePrincipalUseCase = Depends(
        Provide["accounts.resolve_principal_use_case"]
    ),
) -> AuthorizedUser:
    return await resolve_principal.execute(_credential_of(header))


@inject
async def get_current_user(
    header: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    resolve_principal: ResolvePrincipalUseCase = Depends(
        Provide["accounts.resolve_principal_use_case"]
    ),
) -> User:
    return await resolve_principal.load_user(_credential_of(header))
