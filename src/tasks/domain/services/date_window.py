"""Date windows and the coverage they are checked against.

A window is a pair of inclusive local days on one date axis. Before it is used
it is compared with the coverage of that axis, the first and last day on which
the owner has any value there. A window that misses the coverage entirely can
hold nothing, and saying so is more useful than returning an empty result that
looks like an answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Union

from src.tasks.domain.exceptions import InvalidDateWindowError
from src.tasks.domain.value_objects.date_axis import DateAxis
from src.tasks.domain.value_objects.local_calendar import (
    EARLIEST_SUPPORTED_DAY,
    LATEST_SUPPORTED_DAY,
    LocalCalendar,
    is_supported_day,
    parse_iso_day,
)

MALFORMED_DATE = "malformed_date"
DATE_OUT_OF_RANGE = "date_out_of_range"
INVERTED_WINDOW = "inverted_window"


@dataclass(frozen=True)
class CoverageWindow:
    """First and last local day on which the owner has a value on one axis."""

    axis: DateAxis
    first: date
    last: date

    @classmethod
    def for_axis(
        cls,
        axis: DateAxis,
        first_instant: datetime,
        last_instant: datetime,
        calendar: LocalCalendar,
    ) -> "CoverageWindow":
        """Coverage of ``axis`` from its earliest and latest stored instants."""
        return cls(
            axis=axis,
            first=calendar.day_of(first_instant),
            last=calendar.day_of(last_instant),
        )


@dataclass(frozen=True)
class Window:
    """A resolved window: both bounds known, both inclusive."""

    date_from: date
    date_to: date


@dataclass(frozen=True)
class InvalidWindow:
    """The bounds cannot form a window. ``message`` says how to repair them."""

    error_code: str
    message: str


@dataclass(frozen=True)
class RequestedWindow:
    """Bounds as the caller gave them; a missing bound leaves that side open."""

    date_from: Optional[date] = None
    date_to: Optional[date] = None

    def __post_init__(self) -> None:
        problem = _problem_with(self.date_from, self.date_to)
        if problem is not None:
            raise InvalidDateWindowError(problem.message)

    def within(self, coverage: Optional[CoverageWindow]) -> Union[Window, "CoverageGap"]:
        """Close the open bounds with the coverage, unless the two cannot meet.

        Without coverage the axis is empty and every window is a gap. Bounds
        the caller named are kept as they are; only open ones are filled in.
        """
        if coverage is None:
            return CoverageGap(requested=self, coverage=None)
        starts_after_coverage = self.date_from is not None and self.date_from > coverage.last
        ends_before_coverage = self.date_to is not None and self.date_to < coverage.first
        if starts_after_coverage or ends_before_coverage:
            return CoverageGap(requested=self, coverage=coverage)
        return Window(
            date_from=self.date_from if self.date_from is not None else coverage.first,
            date_to=self.date_to if self.date_to is not None else coverage.last,
        )


@dataclass(frozen=True)
class CoverageGap:
    """The requested window lies entirely outside the coverage of its axis."""

    requested: RequestedWindow
    coverage: Optional[CoverageWindow]


class DateWindow:
    """Turns the textual bounds of a request into a usable window."""

    @staticmethod
    def parse(
        date_from: Optional[str], date_to: Optional[str]
    ) -> Union[RequestedWindow, InvalidWindow]:
        """Read ``YYYY-MM-DD`` bounds; needs no data, so it can run before any read."""
        days = {}
        for name, text in (("date_from", date_from), ("date_to", date_to)):
            if text is None:
                days[name] = None
                continue
            day = parse_iso_day(text)
            if day is None:
                return InvalidWindow(
                    MALFORMED_DATE,
                    f"{name} must be a calendar day written as YYYY-MM-DD, for example "
                    f"2026-03-01; got {text!r}. Leave {name} out to keep that side open.",
                )
            days[name] = day

        problem = _problem_with(days["date_from"], days["date_to"])
        if problem is not None:
            return problem
        return RequestedWindow(date_from=days["date_from"], date_to=days["date_to"])

    @staticmethod
    def resolve(
        date_from: Optional[str],
        date_to: Optional[str],
        coverage: Optional[CoverageWindow],
    ) -> Union[Window, "CoverageGap", InvalidWindow]:
        """Parse the bounds and fit them to the coverage in one step."""
        requested = DateWindow.parse(date_from, date_to)
        if isinstance(requested, InvalidWindow):
            return requested
        return requested.within(coverage)


def _problem_with(date_from: Optional[date], date_to: Optional[date]) -> Optional[InvalidWindow]:
    """The one place that says which pairs of days cannot be a window."""
    for name, day in (("date_from", date_from), ("date_to", date_to)):
        if day is not None and not is_supported_day(day):
            return InvalidWindow(
                DATE_OUT_OF_RANGE,
                f"{name} ({day.isoformat()}) is outside the supported range "
                f"{EARLIEST_SUPPORTED_DAY.isoformat()} to {LATEST_SUPPORTED_DAY.isoformat()}. "
                f"Leave {name} out to keep that side of the window open.",
            )
    if date_from is not None and date_to is not None and date_from > date_to:
        return InvalidWindow(
            INVERTED_WINDOW,
            f"date_from ({date_from.isoformat()}) is after date_to ({date_to.isoformat()}). "
            "Swap the two bounds, or leave one out to keep that side open.",
        )
    return None
