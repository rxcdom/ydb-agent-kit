from dataclasses import dataclass
from uuid import UUID, uuid4


@dataclass(frozen=True)
class UserId:
    """Identifier of a user.

    Wraps a UUID for type safety. It lives in the shared kernel so that modules
    can carry an owner id without depending on each other's domain.
    """

    value: UUID

    @classmethod
    def generate(cls) -> "UserId":
        return cls(value=uuid4())

    @classmethod
    def from_string(cls, value: str) -> "UserId":
        try:
            return cls(value=UUID(value))
        except (ValueError, AttributeError, TypeError) as error:
            raise ValueError(f"Invalid UUID format: {value!r}") from error

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"UserId('{self.value}')"
