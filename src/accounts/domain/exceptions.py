from src.shared.domain.exceptions import DomainError, ValidationError


class InvalidCredentialsError(DomainError):
    """The request carries no credential, a malformed one, or one that matches no user."""


class InvalidUserError(ValidationError):
    """User attributes violate a domain rule."""
