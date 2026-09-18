"""Base classes of the domain error hierarchy.

Each module subclasses these in its own ``domain/exceptions.py``. The gateway
maps the base classes to HTTP responses in one place, so a new module error is
handled correctly the moment it picks the right parent.
"""


class DomainError(Exception):
    """Root of every error the application raises on purpose."""


class NotFoundError(DomainError):
    """The addressed object does not exist."""


class AccessDeniedError(DomainError):
    """The object exists but belongs to another owner."""


class ValidationError(DomainError):
    """The input violates a domain rule."""


class ConflictError(DomainError):
    """The operation contradicts the current state."""


class PersistenceError(DomainError):
    """The datastore failed or is unreachable."""
