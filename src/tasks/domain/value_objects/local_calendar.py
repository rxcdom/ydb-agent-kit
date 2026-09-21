"""Conversion between local calendar days and stored instants.

People and the agent talk about days ("due on the 20th", "last week"); the
store keeps UTC instants. Every conversion between the two goes through
``LocalCalendar`` so that one zone decides where a day starts and ends.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Optional, Tuple, Union

# Days a caller may name. The range is far wider than any real task list and
# keeps every derived instant representable by the datastore's timestamp type.
EARLIEST_SUPPORTED_DAY = date(1971, 1, 1)
LATEST_SUPPORTED_DAY = date(2100, 12, 31)

# One day wider than the day range on both sides: the local midnight of a
# supported day is a supported instant in every zone.
EARLIEST_SUPPORTED_INSTANT = datetime(1970, 12, 31, tzinfo=timezone.utc)
LATEST_SUPPORTED_INSTANT = datetime(2101, 1, 2, tzinfo=timezone.utc)

_ISO_DAY_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse_iso_day(text: str) -> Optional[date]:
    """Return the day written strictly as ``YYYY-MM-DD``; ``None`` for anything else."""
    candidate = text.strip()
    if not _ISO_DAY_PATTERN.fullmatch(candidate):
        return None
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        # Right shape, impossible date: "2026-02-30".
        return None


def is_supported_day(day: date) -> bool:
    return EARLIEST_SUPPORTED_DAY <= day <= LATEST_SUPPORTED_DAY


def to_supported_utc(instant: datetime) -> Optional[datetime]:
    """Convert an aware instant to UTC; ``None`` when it lies outside the supported range."""
    # Shifting a year at the very edge of the calendar to UTC can overflow,
    # so the year is checked before the conversion and the instant after it.
    if not EARLIEST_SUPPORTED_INSTANT.year <= instant.year <= LATEST_SUPPORTED_INSTANT.year:
        return None
    converted = instant.astimezone(timezone.utc)
    if not EARLIEST_SUPPORTED_INSTANT <= converted <= LATEST_SUPPORTED_INSTANT:
        return None
    return converted


@dataclass(frozen=True)
class LocalCalendar:
    """Calendar days as seen from one time zone."""

    zone: tzinfo

    def day_of(self, instant: datetime) -> date:
        """The local day an aware instant falls on."""
        if instant.tzinfo is None:
            raise ValueError("a naive datetime has no local day; pass an aware instant")
        return instant.astimezone(self.zone).date()

    def start_of(self, day: date) -> datetime:
        """UTC instant of the local midnight that starts ``day``."""
        return datetime.combine(day, time.min, tzinfo=self.zone).astimezone(timezone.utc)

    def end_of(self, day: date) -> datetime:
        """UTC instant of the local midnight that ends ``day``; an exclusive bound."""
        return self.start_of(day + timedelta(days=1))

    def span_of(self, first_day: date, last_day: date) -> Tuple[datetime, datetime]:
        """``[start, end)`` instants covering the inclusive day range."""
        return self.start_of(first_day), self.end_of(last_day)

    def instant_of(self, value: Union[date, datetime]) -> datetime:
        """Normalise a caller-supplied moment to a UTC instant.

        A bare day means the local midnight that starts it. A naive datetime is
        read as local wall-clock time. An aware datetime is only converted.
        """
        if isinstance(value, datetime):
            aware = value if value.tzinfo is not None else value.replace(tzinfo=self.zone)
            return aware.astimezone(timezone.utc)
        return self.start_of(value)
