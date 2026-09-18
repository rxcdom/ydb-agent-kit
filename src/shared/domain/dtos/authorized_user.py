from dataclasses import dataclass

from src.shared.domain.value_objects.user_id import UserId


@dataclass(frozen=True)
class AuthorizedUser:
    """The authenticated principal of a request.

    Every owner-scoped operation receives its owner id from this object and
    never from a path, a body or a tool argument.
    """

    user_id: UserId
