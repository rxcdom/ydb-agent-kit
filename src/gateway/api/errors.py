"""The single place where domain errors become HTTP responses.

Routers contain no ``try/except``: they let domain errors propagate and this
module decides the status code and the body. Handlers are registered on base
classes where the base class alone determines the answer, so an error added to
a module later is mapped correctly without touching this file.
"""
from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.accounts.domain.exceptions import InvalidCredentialsError
from src.agent.domain.exceptions import (
    LLMResponseGenerationFailedError,
    LLMServiceAuthenticationError,
    LLMServiceError,
    LLMServiceTimeoutError,
)
from src.shared.domain.exceptions import (
    AccessDeniedError,
    ConflictError,
    NotFoundError,
    PersistenceError,
    ValidationError,
)

logger = logging.getLogger(__name__)


def _body(error: str, **extra: Any) -> Dict[str, Any]:
    return {"error": error, **extra}


def _llm_failure_response(error: LLMServiceError) -> JSONResponse:
    """Map an LLM failure by its root cause.

    A failed turn arrives wrapped in ``LLMResponseGenerationFailedError``, which
    names the persisted user message; the status code still follows the cause.
    """
    extra: Dict[str, Any] = {}
    cause: BaseException = error
    if isinstance(error, LLMResponseGenerationFailedError):
        extra = {"user_message_id": error.user_message_id, "status": "failed"}
        cause = error.__cause__ or error

    if isinstance(cause, LLMServiceAuthenticationError):
        return JSONResponse(
            _body("llm_misconfigured", **extra), status.HTTP_503_SERVICE_UNAVAILABLE
        )
    if isinstance(cause, LLMServiceTimeoutError):
        return JSONResponse(_body("llm_timeout", **extra), status.HTTP_504_GATEWAY_TIMEOUT)
    return JSONResponse(_body("llm_unavailable", **extra), status.HTTP_503_SERVICE_UNAVAILABLE)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(InvalidCredentialsError)
    async def _invalid_credentials(_request: Request, _error: InvalidCredentialsError):
        return JSONResponse(
            _body("invalid_credentials"),
            status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(NotFoundError)
    async def _not_found(_request: Request, _error: NotFoundError):
        return JSONResponse(_body("not_found"), status.HTTP_404_NOT_FOUND)

    @app.exception_handler(AccessDeniedError)
    async def _access_denied(_request: Request, error: AccessDeniedError):
        # Answer exactly like a missing object: existence is not revealed to a non-owner.
        logger.warning("Access denied, reported as not found: %s", error)
        return JSONResponse(_body("not_found"), status.HTTP_404_NOT_FOUND)

    @app.exception_handler(ValidationError)
    async def _validation(_request: Request, error: ValidationError):
        return JSONResponse(
            _body("validation_error", message=str(error)), HTTPStatus.UNPROCESSABLE_ENTITY
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation(_request: Request, error: RequestValidationError):
        details = [
            {"field": ".".join(str(part) for part in item["loc"]), "message": item["msg"]}
            for item in error.errors()
        ]
        message = "; ".join(f"{item['field']}: {item['message']}" for item in details)
        return JSONResponse(
            _body("validation_error", message=message, details=details),
            HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    @app.exception_handler(ConflictError)
    async def _conflict(_request: Request, error: ConflictError):
        return JSONResponse(_body("conflict", message=str(error)), status.HTTP_409_CONFLICT)

    @app.exception_handler(LLMServiceError)
    async def _llm_failure(_request: Request, error: LLMServiceError):
        logger.error("LLM failure: %s: %s", type(error).__name__, error)
        return _llm_failure_response(error)

    @app.exception_handler(PersistenceError)
    async def _persistence(_request: Request, error: PersistenceError):
        logger.error("Storage failure: %s", error)
        return JSONResponse(_body("storage_unavailable"), status.HTTP_503_SERVICE_UNAVAILABLE)

    @app.exception_handler(Exception)
    async def _unexpected(_request: Request, error: Exception):
        logger.error("Unhandled error", exc_info=error)
        return JSONResponse(_body("internal_error"), status.HTTP_500_INTERNAL_SERVER_ERROR)
