"""Date windows against the coverage of an axis, and the local-day arithmetic behind them."""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.tasks.domain.exceptions import InvalidDateWindowError
from src.tasks.domain.services.date_window import (
    DATE_OUT_OF_RANGE,
    INVERTED_WINDOW,
    MALFORMED_DATE,
    CoverageGap,
    CoverageWindow,
    DateWindow,
    InvalidWindow,
    RequestedWindow,
    Window,
)
from src.tasks.domain.value_objects.date_axis import DateAxis
from src.tasks.domain.value_objects.local_calendar import (
    EARLIEST_SUPPORTED_DAY,
    LATEST_SUPPORTED_DAY,
    LocalCalendar,
    parse_iso_day,
)

UTC = timezone.utc
BERLIN = ZoneInfo("Europe/Berlin")
LOS_ANGELES = ZoneInfo("America/Los_Angeles")

# The owner has data from the first of June to the end of August.
COVERAGE = CoverageWindow(DateAxis.CREATED, first=date(2026, 6, 1), last=date(2026, 8, 31))


class TestOpenBounds:
    def test_both_bounds_open_means_the_whole_coverage(self):
        assert DateWindow.resolve(None, None, COVERAGE) == Window(
            date(2026, 6, 1), date(2026, 8, 31)
        )

    def test_an_open_upper_bound_defaults_to_the_last_covered_day(self):
        assert DateWindow.resolve("2026-07-10", None, COVERAGE) == Window(
            date(2026, 7, 10), date(2026, 8, 31)
        )

    def test_an_open_lower_bound_defaults_to_the_first_covered_day(self):
        assert DateWindow.resolve(None, "2026-07-10", COVERAGE) == Window(
            date(2026, 6, 1), date(2026, 7, 10)
        )

    def test_an_open_bound_next_to_a_named_bound_outside_the_coverage(self):
        assert DateWindow.resolve("2026-01-01", None, COVERAGE) == Window(
            date(2026, 1, 1), date(2026, 8, 31)
        )
        assert DateWindow.resolve(None, "2026-12-31", COVERAGE) == Window(
            date(2026, 6, 1), date(2026, 12, 31)
        )


class TestOverlap:
    def test_a_window_inside_the_coverage_is_kept_as_named(self):
        assert DateWindow.resolve("2026-07-01", "2026-07-31", COVERAGE) == Window(
            date(2026, 7, 1), date(2026, 7, 31)
        )

    def test_named_bounds_that_reach_past_the_coverage_are_not_clipped(self):
        assert DateWindow.resolve("2026-05-01", "2026-06-15", COVERAGE) == Window(
            date(2026, 5, 1), date(2026, 6, 15)
        )
        assert DateWindow.resolve("2026-08-15", "2026-10-01", COVERAGE) == Window(
            date(2026, 8, 15), date(2026, 10, 1)
        )
        assert DateWindow.resolve("2026-01-01", "2026-12-31", COVERAGE) == Window(
            date(2026, 1, 1), date(2026, 12, 31)
        )

    def test_touching_the_coverage_by_a_single_day_is_an_overlap(self):
        assert DateWindow.resolve("2026-05-01", "2026-06-01", COVERAGE) == Window(
            date(2026, 5, 1), date(2026, 6, 1)
        )
        assert DateWindow.resolve("2026-08-31", "2026-09-30", COVERAGE) == Window(
            date(2026, 8, 31), date(2026, 9, 30)
        )

    def test_a_single_day_window_is_valid(self):
        assert DateWindow.resolve("2026-07-04", "2026-07-04", COVERAGE) == Window(
            date(2026, 7, 4), date(2026, 7, 4)
        )

    def test_a_single_day_of_coverage_works_like_any_other(self):
        one_day = CoverageWindow(DateAxis.DUE, first=date(2026, 7, 4), last=date(2026, 7, 4))

        assert DateWindow.resolve(None, None, one_day) == Window(date(2026, 7, 4), date(2026, 7, 4))
        assert isinstance(DateWindow.resolve("2026-07-05", None, one_day), CoverageGap)


class TestCoverageGap:
    def test_a_window_entirely_before_the_coverage_is_a_gap(self):
        gap = DateWindow.resolve("2026-04-01", "2026-05-31", COVERAGE)

        assert gap == CoverageGap(
            requested=RequestedWindow(date(2026, 4, 1), date(2026, 5, 31)), coverage=COVERAGE
        )

    def test_a_window_entirely_after_the_coverage_is_a_gap(self):
        gap = DateWindow.resolve("2026-09-01", "2026-09-30", COVERAGE)

        assert gap == CoverageGap(
            requested=RequestedWindow(date(2026, 9, 1), date(2026, 9, 30)), coverage=COVERAGE
        )

    def test_one_named_bound_is_enough_to_miss_the_coverage(self):
        before = DateWindow.resolve(None, "2026-05-31", COVERAGE)
        after = DateWindow.resolve("2026-09-01", None, COVERAGE)

        assert before == CoverageGap(RequestedWindow(None, date(2026, 5, 31)), COVERAGE)
        assert after == CoverageGap(RequestedWindow(date(2026, 9, 1), None), COVERAGE)

    def test_the_gap_keeps_the_bounds_as_requested_with_open_sides_open(self):
        gap = DateWindow.resolve(None, "2026-05-31", COVERAGE)

        assert gap.requested.date_from is None
        assert gap.requested.date_to == date(2026, 5, 31)

    @pytest.mark.parametrize(
        "date_from, date_to", [(None, None), ("2026-07-01", None), ("2026-07-01", "2026-07-31")]
    )
    def test_an_axis_without_any_value_makes_every_window_a_gap(self, date_from, date_to):
        gap = DateWindow.resolve(date_from, date_to, None)

        assert isinstance(gap, CoverageGap)
        assert gap.coverage is None


class TestInvalidBounds:
    def test_inverted_bounds_are_refused_with_a_repair_hint(self):
        problem = DateWindow.resolve("2026-07-31", "2026-07-01", COVERAGE)

        assert isinstance(problem, InvalidWindow)
        assert problem.error_code == INVERTED_WINDOW
        assert "2026-07-31" in problem.message and "2026-07-01" in problem.message

    @pytest.mark.parametrize(
        "text",
        [
            "18.09.2026",
            "2026/09/18",
            "2026-9-18",
            "26-09-18",
            "2026-13-01",
            "2026-02-30",
            "2026-09-18T00:00:00",
            "yesterday",
            "",
            "   ",
        ],
    )
    def test_malformed_dates_are_refused_and_the_message_names_the_argument(self, text):
        as_lower_bound = DateWindow.resolve(text, None, COVERAGE)
        as_upper_bound = DateWindow.resolve(None, text, COVERAGE)

        assert isinstance(as_lower_bound, InvalidWindow)
        assert as_lower_bound.error_code == MALFORMED_DATE
        assert "date_from" in as_lower_bound.message and "YYYY-MM-DD" in as_lower_bound.message
        assert isinstance(as_upper_bound, InvalidWindow)
        assert "date_to" in as_upper_bound.message

    def test_a_leap_day_is_a_date_only_in_a_leap_year(self):
        assert parse_iso_day("2028-02-29") == date(2028, 2, 29)
        assert parse_iso_day("2026-02-29") is None

    def test_whitespace_around_a_valid_day_is_tolerated(self):
        assert DateWindow.resolve(" 2026-07-01 ", "2026-07-31\n", COVERAGE) == Window(
            date(2026, 7, 1), date(2026, 7, 31)
        )

    @pytest.mark.parametrize("text", ["1970-12-31", "2101-01-01", "0001-01-01", "9999-12-31"])
    def test_days_outside_the_supported_range_are_refused(self, text):
        problem = DateWindow.resolve(text, None, COVERAGE)

        assert isinstance(problem, InvalidWindow)
        assert problem.error_code == DATE_OUT_OF_RANGE
        assert EARLIEST_SUPPORTED_DAY.isoformat() in problem.message
        assert LATEST_SUPPORTED_DAY.isoformat() in problem.message

    def test_the_edges_of_the_supported_range_are_accepted(self):
        requested = DateWindow.parse(
            EARLIEST_SUPPORTED_DAY.isoformat(), LATEST_SUPPORTED_DAY.isoformat()
        )

        assert requested == RequestedWindow(EARLIEST_SUPPORTED_DAY, LATEST_SUPPORTED_DAY)

    def test_bad_bounds_are_reported_even_when_the_axis_has_no_data(self):
        assert isinstance(DateWindow.resolve("2026-07-31", "2026-07-01", None), InvalidWindow)
        assert isinstance(DateWindow.resolve("soon", None, None), InvalidWindow)

    def test_parsing_needs_no_coverage(self):
        assert DateWindow.parse("2026-07-01", None) == RequestedWindow(date(2026, 7, 1), None)
        assert DateWindow.parse(None, None) == RequestedWindow(None, None)

    def test_a_typed_window_refuses_the_same_pairs_with_a_domain_error(self):
        with pytest.raises(InvalidDateWindowError, match="is after date_to"):
            RequestedWindow(date(2026, 7, 31), date(2026, 7, 1))
        with pytest.raises(InvalidDateWindowError, match="outside the supported range"):
            RequestedWindow(date(1900, 1, 1), None)


class TestCoverageWindow:
    def test_coverage_is_the_first_and_last_local_day_of_the_axis(self):
        coverage = CoverageWindow.for_axis(
            DateAxis.DUE,
            datetime(2026, 6, 1, 9, 30, tzinfo=UTC),
            datetime(2026, 8, 31, 17, 0, tzinfo=UTC),
            LocalCalendar(UTC),
        )

        assert coverage == CoverageWindow(DateAxis.DUE, date(2026, 6, 1), date(2026, 8, 31))

    def test_an_instant_late_in_the_utc_evening_belongs_to_the_next_day_east_of_utc(self):
        first = datetime(2026, 5, 31, 22, 30, tzinfo=UTC)
        last = datetime(2026, 8, 31, 21, 59, tzinfo=UTC)

        coverage = CoverageWindow.for_axis(DateAxis.CREATED, first, last, LocalCalendar(BERLIN))

        assert (coverage.first, coverage.last) == (date(2026, 6, 1), date(2026, 8, 31))

    def test_an_instant_early_in_the_utc_morning_belongs_to_the_previous_day_west_of_utc(self):
        first = datetime(2026, 6, 1, 6, 59, tzinfo=UTC)
        last = datetime(2026, 9, 1, 3, 0, tzinfo=UTC)

        coverage = CoverageWindow.for_axis(
            DateAxis.COMPLETED, first, last, LocalCalendar(LOS_ANGELES)
        )

        assert (coverage.first, coverage.last) == (date(2026, 5, 31), date(2026, 8, 31))

    def test_the_zone_decides_whether_a_window_is_a_gap(self):
        only_instant = datetime(2026, 8, 31, 22, 30, tzinfo=UTC)
        in_utc = CoverageWindow.for_axis(
            DateAxis.CREATED, only_instant, only_instant, LocalCalendar(UTC)
        )
        in_berlin = CoverageWindow.for_axis(
            DateAxis.CREATED, only_instant, only_instant, LocalCalendar(BERLIN)
        )

        assert isinstance(DateWindow.resolve("2026-09-01", None, in_utc), CoverageGap)
        assert DateWindow.resolve("2026-09-01", None, in_berlin) == Window(
            date(2026, 9, 1), date(2026, 9, 1)
        )

    def test_a_naive_instant_has_no_local_day(self):
        with pytest.raises(ValueError, match="naive"):
            CoverageWindow.for_axis(
                DateAxis.CREATED, datetime(2026, 6, 1), datetime(2026, 6, 2), LocalCalendar(UTC)
            )


class TestLocalDayBoundaries:
    def test_a_day_starts_and_ends_at_local_midnight(self):
        calendar = LocalCalendar(BERLIN)

        assert calendar.start_of(date(2026, 9, 18)) == datetime(2026, 9, 17, 22, 0, tzinfo=UTC)
        assert calendar.end_of(date(2026, 9, 18)) == datetime(2026, 9, 18, 22, 0, tzinfo=UTC)

    def test_west_of_utc_the_local_day_starts_later(self):
        calendar = LocalCalendar(LOS_ANGELES)

        assert calendar.start_of(date(2026, 9, 18)) == datetime(2026, 9, 18, 7, 0, tzinfo=UTC)
        assert calendar.end_of(date(2026, 9, 18)) == datetime(2026, 9, 19, 7, 0, tzinfo=UTC)

    def test_a_span_covers_exactly_the_instants_of_its_inclusive_days(self):
        calendar = LocalCalendar(BERLIN)
        first_day, last_day = date(2026, 9, 14), date(2026, 9, 20)

        start, end = calendar.span_of(first_day, last_day)

        assert calendar.day_of(start) == first_day
        assert calendar.day_of(start - timedelta(microseconds=1)) == first_day - timedelta(days=1)
        assert calendar.day_of(end - timedelta(microseconds=1)) == last_day
        assert calendar.day_of(end) == last_day + timedelta(days=1), "the end bound is exclusive"

    def test_days_around_a_clock_change_are_shorter_and_longer(self):
        calendar = LocalCalendar(BERLIN)
        spring_forward, fall_back = date(2026, 3, 29), date(2026, 10, 25)

        spring_start, spring_end = calendar.span_of(spring_forward, spring_forward)
        autumn_start, autumn_end = calendar.span_of(fall_back, fall_back)

        assert spring_end - spring_start == timedelta(hours=23)
        assert autumn_end - autumn_start == timedelta(hours=25)
        assert calendar.day_of(spring_start) == calendar.day_of(
            spring_end - timedelta(seconds=1)
        ) == spring_forward

    def test_a_fixed_offset_zone_works_like_a_named_one(self):
        calendar = LocalCalendar(timezone(timedelta(hours=5, minutes=30)))

        assert calendar.start_of(date(2026, 9, 18)) == datetime(2026, 9, 17, 18, 30, tzinfo=UTC)
        assert calendar.day_of(datetime(2026, 9, 17, 18, 29, tzinfo=UTC)) == date(2026, 9, 17)
        assert calendar.day_of(datetime(2026, 9, 17, 18, 30, tzinfo=UTC)) == date(2026, 9, 18)

    def test_caller_supplied_moments_are_normalised_to_utc_instants(self):
        calendar = LocalCalendar(BERLIN)
        plus_five = timezone(timedelta(hours=5))

        assert calendar.instant_of(date(2026, 9, 18)) == datetime(2026, 9, 17, 22, 0, tzinfo=UTC)
        assert calendar.instant_of(datetime(2026, 9, 18, 9, 0)) == datetime(
            2026, 9, 18, 7, 0, tzinfo=UTC
        ), "a naive moment is local wall-clock time"
        assert calendar.instant_of(datetime(2026, 9, 18, 9, 0, tzinfo=plus_five)) == datetime(
            2026, 9, 18, 4, 0, tzinfo=UTC
        ), "an aware moment is only converted"
        assert calendar.instant_of(date(2026, 9, 18)).tzinfo == UTC
