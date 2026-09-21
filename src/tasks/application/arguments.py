"""Reading the loosely typed arguments of the text-addressed face.

Every reader returns either the typed value or the ``filter_error`` outcome that
tells the caller how to repair the argument, so a bad value from the agent is
answered in the same vocabulary as any other semantic problem.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import FrozenSet, Iterable, Optional, Type, TypeVar, Union

from src.tasks.application.outcome import UNKNOWN_VALUE, Outcome, filter_error
from src.tasks.domain.exceptions import InvalidTaskError
from src.tasks.domain.services.date_window import DATE_OUT_OF_RANGE, MALFORMED_DATE
from src.tasks.domain.value_objects.local_calendar import (
    EARLIEST_SUPPORTED_DAY,
    LATEST_SUPPORTED_DAY,
    LocalCalendar,
    is_supported_day,
    parse_iso_day,
)

E = TypeVar("E", bound=Enum)

# The word that clears an optional field in a text-addressed update.
CLEAR_WORD = "none"


def is_clear_word(text: str) -> bool:
    return text.strip().casefold() == CLEAR_WORD


def read_choice(enum_type: Type[E], value: object, argument: str) -> Union[E, Outcome]:
    """One member of a closed vocabulary, given as the member or as its text value."""
    try:
        return enum_type(value.strip().lower() if isinstance(value, str) else value)
    except ValueError:
        allowed = ", ".join(member.value for member in enum_type)
        return filter_error(
            UNKNOWN_VALUE, f"{argument} must be one of: {allowed}; got {value!r}."
        )


def read_choices(
    enum_type: Type[E], values: Optional[Iterable[object]], argument: str
) -> Union[Optional[FrozenSet[E]], Outcome]:
    """A set of members; ``None`` or an empty list means "no filter on this"."""
    if values is None:
        return None
    members = []
    for value in values:
        member = read_choice(enum_type, value, argument)
        if isinstance(member, Outcome):
            return member
        members.append(member)
    return frozenset(members) or None


def read_day(text: str, argument: str) -> Union[date, Outcome]:
    """A supported calendar day written as ``YYYY-MM-DD``."""
    day = parse_iso_day(text)
    if day is None:
        return filter_error(
            MALFORMED_DATE,
            f"{argument} must be a calendar day written as YYYY-MM-DD, for example "
            f"2026-03-01; got {text!r}.",
        )
    if not is_supported_day(day):
        return filter_error(
            DATE_OUT_OF_RANGE,
            f"{argument} ({day.isoformat()}) is outside the supported range "
            f"{EARLIEST_SUPPORTED_DAY.isoformat()} to {LATEST_SUPPORTED_DAY.isoformat()}.",
        )
    return day


def due_instant(calendar: LocalCalendar, value: Union[date, datetime]) -> datetime:
    """The stored due instant for a day or a moment given through the id-addressed face.

    A bare day is due at the local midnight that starts it. The day is checked
    before the conversion because a moment at the far edge of the calendar
    cannot be shifted between zones at all.
    """
    day = value.date() if isinstance(value, datetime) else value
    if not is_supported_day(day):
        raise InvalidTaskError(
            f"due_at must fall between {EARLIEST_SUPPORTED_DAY.isoformat()} and "
            f"{LATEST_SUPPORTED_DAY.isoformat()}"
        )
    return calendar.instant_of(value)
