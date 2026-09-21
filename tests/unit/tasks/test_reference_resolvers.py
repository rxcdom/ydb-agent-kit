"""Resolution of human text references: one row, several candidates, or nothing."""
import pytest

from src.shared.domain.value_objects.user_id import UserId
from src.tasks.domain.services.project_reference_resolver import ProjectReferenceResolver
from src.tasks.domain.services.reference_resolution import (
    Ambiguous,
    Resolved,
    Unmatched,
    normalise_reference,
)
from src.tasks.domain.services.task_reference_resolver import TaskReferenceResolver
from src.tasks.domain.value_objects.task_status import TaskStatus
from tests.unit.tasks.builders import a_project, a_task

OWNER = UserId.generate()


class TestProjectReferenceResolver:
    def test_an_exact_name_beats_a_partial_match(self):
        home = a_project(OWNER, "Home")
        renovation = a_project(OWNER, "Home renovation")

        resolution = ProjectReferenceResolver.resolve([renovation, home], "Home")

        assert resolution == Resolved(home)

    def test_a_word_shared_by_two_names_is_ambiguous(self):
        renovation = a_project(OWNER, "Home renovation")
        office = a_project(OWNER, "Home office")
        garden = a_project(OWNER, "Garden")

        resolution = ProjectReferenceResolver.resolve([renovation, office, garden], "Home")

        assert resolution == Ambiguous((renovation, office))

    def test_two_exact_names_are_ambiguous_and_neither_is_picked(self):
        first = a_project(OWNER, "Garden", description="Front garden")
        second = a_project(OWNER, "garden", description="Allotment plot")
        partial = a_project(OWNER, "Garden tools")

        resolution = ProjectReferenceResolver.resolve([first, second, partial], "GARDEN")

        assert resolution == Ambiguous((first, second)), "the partial tier is never consulted"

    def test_a_single_partial_match_resolves(self):
        renovation = a_project(OWNER, "Home renovation")
        garden = a_project(OWNER, "Garden")

        assert ProjectReferenceResolver.resolve([renovation, garden], "renov") == Resolved(
            renovation
        )

    def test_the_description_takes_part_in_the_partial_tier(self):
        clients = a_project(OWNER, "Client work", description="Deliverables and invoices")
        garden = a_project(OWNER, "Garden", description="Seasonal planting")

        assert ProjectReferenceResolver.resolve([clients, garden], "invoices") == Resolved(clients)

    def test_a_name_match_and_a_description_match_tie_in_the_partial_tier(self):
        by_name = a_project(OWNER, "Bathroom remodel")
        by_description = a_project(OWNER, "Home renovation", description="Kitchen remodel first")

        resolution = ProjectReferenceResolver.resolve([by_name, by_description], "remodel")

        assert resolution == Ambiguous((by_name, by_description))

    def test_an_exact_name_beats_a_description_match(self):
        garden = a_project(OWNER, "Garden")
        chores = a_project(OWNER, "Chores", description="House and garden chores")

        assert ProjectReferenceResolver.resolve([chores, garden], "garden") == Resolved(garden)

    def test_a_description_is_never_an_exact_match(self):
        described = a_project(OWNER, "Chores", description="Garden")
        named = a_project(OWNER, "Garden shed")

        resolution = ProjectReferenceResolver.resolve([described, named], "garden")

        assert resolution == Ambiguous((described, named))

    def test_a_missing_description_is_skipped(self):
        bare = a_project(OWNER, "Garden", description=None)

        assert ProjectReferenceResolver.resolve([bare], "planting") == Unmatched()

    @pytest.mark.parametrize(
        "reference", ["HOME RENOVATION", "home renovation", " Home Renovation  "]
    )
    def test_matching_ignores_case_and_surrounding_whitespace(self, reference):
        renovation = a_project(OWNER, "Home renovation")
        office = a_project(OWNER, "Home office")

        assert ProjectReferenceResolver.resolve([renovation, office], reference) == Resolved(
            renovation
        )

    def test_nothing_matches(self):
        projects = [a_project(OWNER, "Home renovation"), a_project(OWNER, "Garden")]

        assert ProjectReferenceResolver.resolve(projects, "Garage") == Unmatched()
        assert ProjectReferenceResolver.resolve([], "Garden") == Unmatched()

    @pytest.mark.parametrize("reference", ["", "   ", "\t\n", None])
    def test_an_empty_reference_matches_nothing_rather_than_everything(self, reference):
        projects = [a_project(OWNER, "Home renovation"), a_project(OWNER, "Garden")]

        assert ProjectReferenceResolver.resolve(projects, reference) == Unmatched()


class TestTaskReferenceResolver:
    def test_an_exact_title_beats_a_partial_match(self):
        report = a_task(OWNER, "Report")
        quarterly = a_task(OWNER, "Quarterly report")

        assert TaskReferenceResolver.resolve([quarterly, report], "report") == Resolved(report)

    def test_a_word_shared_by_three_titles_is_ambiguous(self):
        quarterly = a_task(OWNER, "Quarterly report")
        expense = a_task(OWNER, "Expense report")
        weekly = a_task(OWNER, "Weekly status report")
        other = a_task(OWNER, "Order kitchen tiles")

        resolution = TaskReferenceResolver.resolve([quarterly, expense, other, weekly], "report")

        assert resolution == Ambiguous((quarterly, expense, weekly)), "candidates keep their order"

    def test_two_exact_titles_are_ambiguous_even_when_they_look_interchangeable(self):
        first = a_task(OWNER, "Water the plants")
        second = a_task(OWNER, "water the plants")
        partial = a_task(OWNER, "Water the plants on the balcony")

        resolution = TaskReferenceResolver.resolve([first, second, partial], "Water the plants")

        assert resolution == Ambiguous((first, second))

    def test_the_notes_take_part_in_the_partial_tier(self):
        lamp = a_task(OWNER, "Replace desk lamp", notes="Check the warranty first")
        tiles = a_task(OWNER, "Order kitchen tiles", notes="Matte white")

        assert TaskReferenceResolver.resolve([lamp, tiles], "warranty") == Resolved(lamp)

    def test_a_title_match_and_a_notes_match_tie_in_the_partial_tier(self):
        by_title = a_task(OWNER, "Renew warranty")
        by_notes = a_task(OWNER, "Replace desk lamp", notes="Check the warranty first")

        resolution = TaskReferenceResolver.resolve([by_title, by_notes], "warranty")

        assert resolution == Ambiguous((by_title, by_notes))

    def test_an_exact_title_beats_a_notes_match(self):
        exact = a_task(OWNER, "Warranty")
        by_notes = a_task(OWNER, "Replace desk lamp", notes="Check the warranty first")

        assert TaskReferenceResolver.resolve([by_notes, exact], "warranty") == Resolved(exact)

    def test_missing_notes_are_skipped(self):
        assert TaskReferenceResolver.resolve([a_task(OWNER, "Mow the lawn")], "warranty") == (
            Unmatched()
        )

    def test_a_project_scope_narrows_a_title_collision(self):
        renovation = a_project(OWNER, "Home renovation")
        office = a_project(OWNER, "Home office")
        in_renovation = a_task(OWNER, "Buy paint", project=renovation)
        in_office = a_task(OWNER, "Buy paint", project=office)
        tasks = [in_renovation, in_office]

        assert TaskReferenceResolver.resolve(tasks, "Buy paint") == Ambiguous(
            (in_renovation, in_office)
        )
        assert TaskReferenceResolver.resolve(tasks, "Buy paint", office) == Resolved(in_office)
        assert TaskReferenceResolver.resolve(tasks, "paint", renovation) == Resolved(in_renovation)

    def test_the_scope_applies_to_both_tiers(self):
        renovation = a_project(OWNER, "Home renovation")
        office = a_project(OWNER, "Home office")
        exact_elsewhere = a_task(OWNER, "Paint", project=office)
        partial_in_scope = a_task(OWNER, "Buy paint", project=renovation)

        resolution = TaskReferenceResolver.resolve(
            [exact_elsewhere, partial_in_scope], "paint", renovation
        )

        assert resolution == Resolved(partial_in_scope)

    def test_a_collision_inside_the_scope_stays_ambiguous(self):
        office = a_project(OWNER, "Home office")
        first = a_task(OWNER, "Buy paint", project=office)
        second = a_task(OWNER, "Buy paint brushes", project=office)

        assert TaskReferenceResolver.resolve([first, second], "paint", office) == Ambiguous(
            (first, second)
        )

    def test_a_scope_leaves_out_unfiled_tasks_and_other_projects(self):
        office = a_project(OWNER, "Home office")
        garden = a_project(OWNER, "Garden")
        unfiled = a_task(OWNER, "Buy paint")
        in_garden = a_task(OWNER, "Buy paint", project=garden)

        assert TaskReferenceResolver.resolve([unfiled, in_garden], "Buy paint", office) == (
            Unmatched()
        )

    def test_tasks_of_every_status_take_part(self):
        done = a_task(OWNER, "Paint hallway", status=TaskStatus.DONE)
        cancelled = a_task(OWNER, "Paint fence", status=TaskStatus.CANCELLED)

        assert TaskReferenceResolver.resolve([done, cancelled], "hallway") == Resolved(done)
        assert TaskReferenceResolver.resolve([done, cancelled], "paint") == Ambiguous(
            (done, cancelled)
        )

    @pytest.mark.parametrize("reference", ["ORDER KITCHEN TILES", " order Kitchen tiles ", "TILES"])
    def test_matching_ignores_case_and_surrounding_whitespace(self, reference):
        tiles = a_task(OWNER, "Order kitchen tiles")
        lamp = a_task(OWNER, "Replace desk lamp")

        assert TaskReferenceResolver.resolve([tiles, lamp], reference) == Resolved(tiles)

    def test_nothing_matches(self):
        assert TaskReferenceResolver.resolve([a_task(OWNER, "Mow the lawn")], "plumber") == (
            Unmatched()
        )
        assert TaskReferenceResolver.resolve([], "Mow the lawn") == Unmatched()

    @pytest.mark.parametrize("reference", ["", "   ", None])
    def test_an_empty_reference_matches_nothing_rather_than_everything(self, reference):
        tasks = [a_task(OWNER, "Mow the lawn"), a_task(OWNER, "Order compost")]

        assert TaskReferenceResolver.resolve(tasks, reference) == Unmatched()

    def test_any_iterable_of_tasks_is_accepted(self):
        lawn = a_task(OWNER, "Mow the lawn")

        assert TaskReferenceResolver.resolve((task for task in [lawn]), "lawn") == Resolved(lawn)


def test_references_are_compared_trimmed_and_case_folded():
    assert normalise_reference("  Home RENOVATION ") == "home renovation"
    assert normalise_reference(None) == ""
    assert normalise_reference("   ") == ""
