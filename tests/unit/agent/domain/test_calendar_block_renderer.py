"""Tests for the calendar block of the system prompt."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.agent.domain.services.calendar_block_renderer import CalendarBlockRenderer

UTC = ZoneInfo("UTC")
BERLIN = ZoneInfo("Europe/Berlin")
TOKYO = ZoneInfo("Asia/Tokyo")


def test_renders_local_date_weekday_and_time_of_the_zone():
    # 2026-08-06 22:30 UTC is 2026-08-07 07:30 in Tokyo (a Friday).
    instant = datetime(2026, 8, 6, 22, 30, tzinfo=timezone.utc)
    block = CalendarBlockRenderer.render(instant, TOKYO)

    assert block.startswith("## Calendar\n")
    assert "Now: 2026-08-07 (Friday), 07:30 in Asia/Tokyo." in block
    assert "- today: 2026-08-07" in block  # the local date, not the UTC one
    assert "2026-08-06 (Thursday)" not in block


def test_same_instant_renders_differently_per_zone():
    instant = datetime(2026, 8, 6, 22, 30, tzinfo=timezone.utc)

    assert "- today: 2026-08-06" in CalendarBlockRenderer.render(instant, UTC)
    assert "- today: 2026-08-07" in CalendarBlockRenderer.render(instant, BERLIN)


def test_aware_input_in_another_offset_is_converted():
    # The same instant expressed with a +05:00 offset gives the same block.
    as_utc = datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)
    shifted = as_utc.astimezone(timezone(timedelta(hours=5)))

    assert CalendarBlockRenderer.render(shifted, BERLIN) == CalendarBlockRenderer.render(
        as_utc, BERLIN
    )


def test_renders_ready_intervals():
    # 2026-08-18 is a Tuesday.
    instant = datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)
    block = CalendarBlockRenderer.render(instant, UTC)

    assert "Now: 2026-08-18 (Tuesday), 09:00 in UTC." in block
    assert "- today: 2026-08-18" in block
    assert "- yesterday: 2026-08-17" in block
    assert "- this week (Mon–today): 2026-08-17 — 2026-08-18" in block
    assert "- last week (Mon–Sun): 2026-08-10 — 2026-08-16" in block
    assert "- last 7 days: 2026-08-12 — 2026-08-18" in block
    assert "- this month: 2026-08-01 — 2026-08-18" in block
    assert "- last month: 2026-07-01 — 2026-07-31" in block
    assert "- last 30 days: 2026-07-20 — 2026-08-18" in block
    assert "- this year: 2026-01-01 — 2026-08-18" in block
    assert "- default window when no period is named: 2026-07-20 — 2026-08-18" in block
    assert '- "all time": call without dates' in block


def test_monday_this_week_is_a_single_day():
    # On Monday 2026-08-17 "this week" collapses to that one day.
    instant = datetime(2026, 8, 17, 9, 0, tzinfo=timezone.utc)
    block = CalendarBlockRenderer.render(instant, UTC)

    assert "Now: 2026-08-17 (Monday)" in block
    assert "- this week (Mon–today): 2026-08-17 — 2026-08-17" in block
    assert "- last week (Mon–Sun): 2026-08-10 — 2026-08-16" in block


def test_sunday_this_week_is_the_full_week():
    instant = datetime(2026, 8, 23, 9, 0, tzinfo=timezone.utc)
    block = CalendarBlockRenderer.render(instant, UTC)

    assert "Now: 2026-08-23 (Sunday)" in block
    assert "- this week (Mon–today): 2026-08-17 — 2026-08-23" in block
    assert "- last week (Mon–Sun): 2026-08-10 — 2026-08-16" in block


def test_intervals_cross_the_year_boundary():
    # 2026-01-01 is a Thursday.
    instant = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    block = CalendarBlockRenderer.render(instant, UTC)

    assert "- yesterday: 2025-12-31" in block
    assert "- this week (Mon–today): 2025-12-29 — 2026-01-01" in block
    assert "- last week (Mon–Sun): 2025-12-22 — 2025-12-28" in block
    assert "- this month: 2026-01-01 — 2026-01-01" in block
    assert "- last month: 2025-12-01 — 2025-12-31" in block
    assert "- last 30 days: 2025-12-03 — 2026-01-01" in block
    assert "- this year: 2026-01-01 — 2026-01-01" in block


def test_last_month_handles_a_leap_february():
    instant = datetime(2028, 3, 10, 9, 0, tzinfo=timezone.utc)
    block = CalendarBlockRenderer.render(instant, UTC)

    assert "- last month: 2028-02-01 — 2028-02-29" in block


def test_daylight_saving_boundary_uses_the_offset_of_the_instant():
    # Berlin leaves summer time on 2026-10-25 at 01:00 UTC: +02:00 before it,
    # +01:00 after it. Each instant below lands on a different local day if the
    # wrong offset is applied.
    before_switch = datetime(2026, 10, 24, 22, 30, tzinfo=timezone.utc)
    block = CalendarBlockRenderer.render(before_switch, BERLIN)
    assert "Now: 2026-10-25 (Sunday), 00:30 in Europe/Berlin." in block
    assert "- yesterday: 2026-10-24" in block

    after_switch = datetime(2026, 10, 25, 22, 30, tzinfo=timezone.utc)
    block = CalendarBlockRenderer.render(after_switch, BERLIN)
    assert "Now: 2026-10-25 (Sunday), 23:30 in Europe/Berlin." in block
    assert "- today: 2026-10-25" in block
    assert "- this week (Mon–today): 2026-10-19 — 2026-10-25" in block


def test_rejects_naive_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        CalendarBlockRenderer.render(datetime(2026, 8, 6, 12, 0), UTC)


def test_whole_block_matches_the_documented_format():
    # 2026-09-17 12:05 UTC is 14:05 in Berlin (summer time, +02:00).
    instant = datetime(2026, 9, 17, 12, 5, tzinfo=timezone.utc)

    expected = "\n".join(
        [
            "## Calendar",
            "Now: 2026-09-17 (Thursday), 14:05 in Europe/Berlin.",
            "Ready intervals (both bounds inclusive, local days) — copy dates from here, "
            "do not compute them:",
            "- today: 2026-09-17",
            "- yesterday: 2026-09-16",
            "- this week (Mon–today): 2026-09-14 — 2026-09-17",
            "- last week (Mon–Sun): 2026-09-07 — 2026-09-13",
            "- last 7 days: 2026-09-11 — 2026-09-17",
            "- this month: 2026-09-01 — 2026-09-17",
            "- last month: 2026-08-01 — 2026-08-31",
            "- last 30 days: 2026-08-19 — 2026-09-17",
            "- this year: 2026-01-01 — 2026-09-17",
            "- default window when no period is named: 2026-08-19 — 2026-09-17",
            '- "all time": call without dates',
            "Upcoming days, for deadlines:",
            "- tomorrow: 2026-09-18",
            "- next week (Mon–Sun): 2026-09-21 — 2026-09-27",
            "- next month: 2026-10-01 — 2026-10-31",
            "- the next 14 days: Fri 2026-09-18, Sat 2026-09-19, Sun 2026-09-20, Mon 2026-09-21, "
            "Tue 2026-09-22, Wed 2026-09-23, Thu 2026-09-24, Fri 2026-09-25, Sat 2026-09-26, "
            "Sun 2026-09-27, Mon 2026-09-28, Tue 2026-09-29, Wed 2026-09-30, Thu 2026-10-01",
        ]
    )

    assert CalendarBlockRenderer.render(instant, BERLIN) == expected


def test_upcoming_periods_cross_the_year_boundary():
    block = CalendarBlockRenderer.render(datetime(2026, 12, 30, 9, 0, tzinfo=timezone.utc), UTC)

    assert "- tomorrow: 2026-12-31" in block
    assert "- next week (Mon–Sun): 2027-01-04 — 2027-01-10" in block
    assert "- next month: 2027-01-01 — 2027-01-31" in block
    assert "Thu 2026-12-31, Fri 2027-01-01" in block


def test_next_month_has_the_right_length_after_a_long_month():
    block = CalendarBlockRenderer.render(datetime(2027, 1, 31, 9, 0, tzinfo=timezone.utc), UTC)

    assert "- next month: 2027-02-01 — 2027-02-28" in block
