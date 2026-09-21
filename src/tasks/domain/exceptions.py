from src.shared.domain.exceptions import AccessDeniedError, NotFoundError, ValidationError


class TaskNotFoundError(NotFoundError):
    """No task with this id exists among the owner's tasks."""


class ProjectNotFoundError(NotFoundError):
    """No project with this id exists among the owner's projects."""


class TaskAccessDeniedError(AccessDeniedError):
    """The task belongs to another owner."""


class ProjectAccessDeniedError(AccessDeniedError):
    """The project belongs to another owner."""


class InvalidTaskError(ValidationError):
    """Task attributes violate a domain rule."""


class InvalidProjectError(ValidationError):
    """Project attributes violate a domain rule."""


class InvalidDateWindowError(ValidationError):
    """A date window is inverted or reaches outside the supported calendar range."""
