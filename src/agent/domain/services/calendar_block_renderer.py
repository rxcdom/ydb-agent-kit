"""Render the ``## Calendar`` block of the system prompt.

A model has no clock, and language models are unreliable at relative calendar
arithmetic: "this week" drifts to the previous one, month lengths get guessed.
The block therefore carries ready-made intervals for every period convention the
prompt names, so the model copies dates instead of deriving them.

Conventions:

* "last 7 days" and "last 30 days" include today;
* "this week" runs from Monday to today, so on a Monday it is a single day;
* "last week" is the previous calendar week, Monday to Sunday;
* "this month" and "this year" run from their first day to today;
* "last month" is the previous calendar month;
* the default window, used when the user names no period, is the last 30 days;
* both bounds are inclusive, and days are days of the configured time zone.

The block does not carry the bounds of the user's own data. Those are per-user
and arrive with every tool result as its coverage.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


def _iso(day: date) -> str:
    return day.strftime("%Y-%m-%d")


def _span(date_from: date, date_to: date) -> str:
    return f"{_iso(date_from)} — {_iso(date_to)}"


class CalendarBlockRenderer:
    """Pure renderer: a UTC instant and a time zone in, the calendar block out."""

    @staticmethod
    def render(now_utc: datetime, tz: ZoneInfo) -> str:
        """Return the calendar block for the given instant.

        ``now_utc`` must be timezone-aware. It is converted to ``tz`` here, so
        callers keep every timestamp in UTC. The conversion follows the zone's
        daylight-saving rules for that instant.
        """
        if now_utc.tzinfo is None:
            raise ValueError("now_utc must be timezone-aware")

        now_local = now_utc.astimezone(tz)
        today = now_local.date()

        yesterday = today - timedelta(days=1)
        this_week_from = today - timedelta(days=today.weekday())
        last_week_to = this_week_from - timedelta(days=1)
        last_week_from = last_week_to - timedelta(days=6)
        last_7_from = today - timedelta(days=6)
        this_month_from = today.replace(day=1)
        last_month_to = this_month_from - timedelta(days=1)
        last_month_from = last_month_to.replace(day=1)
        last_30_from = today - timedelta(days=29)
        this_year_from = today.replace(month=1, day=1)

        # The default window is its own line, so changing the rule touches one
        # place and the prompt never has to restate it.
        default_from, default_to = last_30_from, today

        return (
            "## Calendar\n"
            f"Now: {_iso(today)} ({today.strftime('%A')}), "
            f"{now_local.strftime('%H:%M')} in {tz.key}.\n"
            "Ready intervals (both bounds inclusive, local days) — "
            "copy dates from here, do not compute them:\n"
            f"- today: {_iso(today)}\n"
            f"- yesterday: {_iso(yesterday)}\n"
            f"- this week (Mon–today): {_span(this_week_from, today)}\n"
            f"- last week (Mon–Sun): {_span(last_week_from, last_week_to)}\n"
            f"- last 7 days: {_span(last_7_from, today)}\n"
            f"- this month: {_span(this_month_from, today)}\n"
            f"- last month: {_span(last_month_from, last_month_to)}\n"
            f"- last 30 days: {_span(last_30_from, today)}\n"
            f"- this year: {_span(this_year_from, today)}\n"
            f"- default window when no period is named: {_span(default_from, default_to)}\n"
            '- "all time": call without dates'
        )
