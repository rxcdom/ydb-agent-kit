#!/usr/bin/env python3
"""Fifteen-turn acceptance run against the live API and a real model.

One fresh user, one freshly seeded workspace, ONE chat, fifteen messages sent
in order. Later turns depend on what earlier turns established, so the run is
only meaningful as a whole.

The messages are written the way a person types: they never name a tool, an
argument, a status or a date format, and every period is relative. The checks
read the debug trace of each response (which tools ran, with which arguments,
and how they answered) and then the REST API, to confirm that the writes the
conversation claims really happened - and that the refused one did not.

Usage:
    python scripts/acceptance_run.py --base-url http://localhost:8000

The workspace is seeded through the compose ``seed`` tool; override
``--seed-command`` to seed another way ("{token}" is replaced by the user id).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx

READ_TOOLS = {"list_projects", "query_tasks"}
WRITE_TOOLS = {"create_task", "update_task", "delete_task"}
REPORT_TITLES = ("Quarterly report", "Expense report", "Weekly status report")
DEFAULT_SEED_COMMAND = "docker compose run --rm seed --token {token}"
TURN_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class Calendar:
    """The dates the agent is expected to copy from its calendar block."""

    today: date

    @property
    def yesterday(self) -> date:
        return self.today - timedelta(days=1)

    @property
    def tomorrow(self) -> date:
        return self.today + timedelta(days=1)

    @property
    def in_a_week(self) -> date:
        return self.today + timedelta(days=7)

    @property
    def last_week(self) -> tuple[date, date]:
        monday = self.today - timedelta(days=self.today.weekday() + 7)
        return monday, monday + timedelta(days=6)

    @property
    def last_month(self) -> tuple[date, date]:
        last_day = self.today.replace(day=1) - timedelta(days=1)
        return last_day.replace(day=1), last_day


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]

    @property
    def status(self) -> Optional[str]:
        return self.result.get("status")

    @property
    def data(self) -> Dict[str, Any]:
        data = self.result.get("data")
        return data if isinstance(data, dict) else {}


@dataclass
class TurnRecord:
    number: int
    message: str
    expectation: str
    reply: str = ""
    calls: List[ToolCall] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "pass" if not self.gaps else "; ".join(self.gaps)


Check = Callable[[TurnRecord, Calendar], List[str]]


@dataclass(frozen=True)
class Turn:
    message: str
    expectation: str
    check: Check


# --- check helpers ---------------------------------------------------------------------------


def _calls(record: TurnRecord, name: str) -> List[ToolCall]:
    return [call for call in record.calls if call.name == name]


def _no_writes(record: TurnRecord) -> List[str]:
    written = sorted({call.name for call in record.calls if call.name in WRITE_TOOLS})
    return [f"a read-only turn called write tools: {written}"] if written else []


def _window_of(call: ToolCall) -> tuple[Optional[str], Optional[str]]:
    return call.arguments.get("date_from"), call.arguments.get("date_to")


def _is_overdue_query(call: ToolCall, calendar: Calendar) -> bool:
    return (
        call.arguments.get("date_field") == "due"
        and call.arguments.get("date_to") == calendar.yesterday.isoformat()
        and not call.arguments.get("date_from")
        and call.arguments.get("statuses") == ["open"]
    )


def _mentions(reply: str, *fragments: str) -> List[str]:
    lowered = reply.lower()
    return [f"the reply does not mention '{item}'" for item in fragments if item.lower() not in lowered]


def _changed(call: ToolCall) -> Dict[str, Any]:
    return {entry.get("field"): entry.get("to") for entry in call.data.get("changed", [])}


# --- the fifteen turns -----------------------------------------------------------------------


def check_open_question(record: TurnRecord, calendar: Calendar) -> List[str]:
    gaps = _no_writes(record)
    reads = [call for call in record.calls if call.name in READ_TOOLS and call.status == "ok"]
    if not reads:
        gaps.append("no successful read of tasks or projects")
    for call in _calls(record, "query_tasks"):
        if "open" not in (call.arguments.get("statuses") or []):
            gaps.append("current work was read without restricting to open tasks")
        if any(_window_of(call)):
            gaps.append("a state question was narrowed to a period")
    return gaps


def check_preference_and_last_week(record: TurnRecord, calendar: Calendar) -> List[str]:
    gaps = _no_writes(record)
    if not [call for call in _calls(record, "remember") if call.status == "ok"]:
        gaps.append("the stated preference was not stored")
    expected = tuple(day.isoformat() for day in calendar.last_week)
    queries = [call for call in _calls(record, "query_tasks") if _window_of(call) == expected]
    if not queries:
        gaps.append(f"no read over last week {expected}")
    elif queries[-1].arguments.get("date_field", "created") != "created":
        gaps.append("'added' was not read on the creation date")
    elif queries[-1].status not in {"ok", "no_records"}:
        gaps.append(f"unexpected outcome {queries[-1].status}")
    return gaps


def check_finished_last_month(record: TurnRecord, calendar: Calendar) -> List[str]:
    gaps = _no_writes(record)
    expected = tuple(day.isoformat() for day in calendar.last_month)
    queries = [call for call in _calls(record, "query_tasks") if _window_of(call) == expected]
    if not queries:
        return gaps + [f"no read over last month {expected}"]
    if queries[-1].arguments.get("date_field") != "completed":
        gaps.append("'got done' was not read on the completion date")
    if queries[-1].status != "ok":
        gaps.append(f"unexpected outcome {queries[-1].status}")
    return gaps


def check_follow_up_reuses_window(record: TurnRecord, calendar: Calendar) -> List[str]:
    gaps = _no_writes(record)
    expected = tuple(day.isoformat() for day in calendar.last_month)
    queries = [
        call
        for call in _calls(record, "query_tasks")
        if "garden" in (call.arguments.get("project") or "").lower()
    ]
    if not queries:
        return gaps + ["no read restricted to the garden project"]
    last = queries[-1]
    if _window_of(last) != expected:
        gaps.append(f"the previous period was not reused: {_window_of(last)}")
    if last.arguments.get("date_field") != "completed":
        gaps.append("the previous date axis was not reused")
    if last.status not in {"ok", "empty_filter"}:
        gaps.append(f"unexpected outcome {last.status}")
    return gaps


def check_nothing_matched(record: TurnRecord, calendar: Calendar) -> List[str]:
    gaps = _no_writes(record)
    queries = [
        call for call in _calls(record, "query_tasks") if "tax" in (call.arguments.get("text") or "").lower()
    ]
    if not queries:
        return gaps + ["no text search for the topic"]
    if queries[-1].status != "empty_filter":
        gaps.append(f"unexpected outcome {queries[-1].status}")
    if any(call.status == "ok" and call.data.get("total_count") for call in queries):
        gaps.append("the search unexpectedly matched tasks")
    return gaps


def check_before_coverage(record: TurnRecord, calendar: Calendar) -> List[str]:
    gaps = _no_writes(record)
    gap_calls = [call for call in _calls(record, "query_tasks") if call.status == "coverage_gap"]
    if not gap_calls:
        return gaps + ["no read reported a period outside the covered range"]
    coverage = gap_calls[-1].data.get("coverage", {})
    gaps += _mentions(record.reply, str(coverage.get("from")), str(coverage.get("to")))
    return gaps


def check_ambiguous_project(record: TurnRecord, calendar: Calendar) -> List[str]:
    gaps = _no_writes(record)
    queries = _calls(record, "query_tasks")
    ambiguous = [call for call in queries if call.status == "ambiguous_source"]
    if not ambiguous:
        gaps.append("the ambiguous project name was not reported as ambiguous")
    elif not _is_overdue_query(ambiguous[-1], calendar):
        gaps.append(f"overdue was not read as due-before-today and open: {ambiguous[-1].arguments}")
    picked = [call for call in queries if call.status == "ok" and call.data.get("project_resolved")]
    if picked:
        gaps.append("the agent picked a project instead of asking")
    gaps += _mentions(record.reply, "Home renovation", "Home office")
    if "?" not in record.reply:
        gaps.append("the reply does not ask the user to choose")
    return gaps


def check_clarified_project(record: TurnRecord, calendar: Calendar) -> List[str]:
    gaps = _no_writes(record)
    resolved = [
        call
        for call in _calls(record, "query_tasks")
        if call.status == "ok"
        and (call.data.get("project_resolved") or {}).get("name") == "Home renovation"
    ]
    if not resolved:
        return gaps + ["no successful read scoped to Home renovation"]
    if not _is_overdue_query(resolved[-1], calendar):
        gaps.append(f"the original overdue question was not repeated: {resolved[-1].arguments}")
    return gaps


def check_task_added(record: TurnRecord, calendar: Calendar) -> List[str]:
    created = [call for call in _calls(record, "create_task") if call.status == "ok"]
    if len(created) != 1:
        return [f"expected exactly one successful creation, saw {len(created)}"]
    task = created[0].data.get("created", {})
    gaps = []
    if task.get("project") != "Home renovation":
        gaps.append(f"created in project {task.get('project')!r}")
    if task.get("priority") != "high":
        gaps.append(f"created with priority {task.get('priority')!r}")
    if task.get("due_at") != calendar.tomorrow.isoformat():
        gaps.append(f"created with deadline {task.get('due_at')!r}, expected {calendar.tomorrow}")
    return gaps


def check_task_changed(record: TurnRecord, calendar: Calendar) -> List[str]:
    updates = [call for call in _calls(record, "update_task") if call.status == "ok"]
    if not updates:
        return ["no successful update"]
    changed: Dict[str, Any] = {}
    for call in updates:
        changed.update(_changed(call))
    gaps = []
    if changed.get("due_at") != calendar.in_a_week.isoformat():
        gaps.append(f"deadline changed to {changed.get('due_at')!r}, expected {calendar.in_a_week}")
    if changed.get("priority") != "high":
        gaps.append(f"priority changed to {changed.get('priority')!r}")
    return gaps


def check_task_finished(record: TurnRecord, calendar: Calendar) -> List[str]:
    updates = [call for call in _calls(record, "update_task") if call.status == "ok"]
    if len(updates) != 1:
        return [f"expected exactly one successful update, saw {len(updates)}"]
    if _changed(updates[0]).get("status") != "done":
        return [f"status change was {_changed(updates[0])}"]
    return []


def check_refused_deletion(record: TurnRecord, calendar: Calendar) -> List[str]:
    deletions = _calls(record, "delete_task")
    gaps = []
    if not [call for call in deletions if call.status == "ambiguous_source"]:
        gaps.append("the ambiguous deletion was not refused")
    if [call for call in deletions if call.status == "ok"]:
        gaps.append("a task was deleted although the name was ambiguous")
    gaps += _mentions(record.reply, *REPORT_TITLES)
    if "?" not in record.reply:
        gaps.append("the reply does not ask the user to choose")
    return gaps


def check_completed_deletion(record: TurnRecord, calendar: Calendar) -> List[str]:
    deleted = [call for call in _calls(record, "delete_task") if call.status == "ok"]
    if len(deleted) != 1:
        return [f"expected exactly one deletion, saw {len(deleted)}"]
    title = deleted[0].data.get("deleted", {}).get("title")
    return [] if title == "Expense report" else [f"deleted {title!r}"]


def check_recall(record: TurnRecord, calendar: Calendar) -> List[str]:
    gaps = _no_writes(record)
    recalls = _calls(record, "recall")
    if not recalls:
        return gaps + ["memory was not read"]
    if not recalls[-1].result.get("count"):
        gaps.append("memory came back empty")
    return gaps + _mentions(record.reply, "Friday")


def check_forget_and_overdue(record: TurnRecord, calendar: Calendar) -> List[str]:
    gaps = _no_writes(record)
    forgets = _calls(record, "forget")
    if not forgets:
        gaps.append("nothing was forgotten")
    elif not any(call.result.get("deleted_count") for call in forgets):
        gaps.append("forget matched no note")
    overdue = [
        call
        for call in _calls(record, "query_tasks")
        if _is_overdue_query(call, calendar) and not call.arguments.get("project")
    ]
    if not overdue:
        gaps.append("overdue across all projects was not read")
    elif overdue[-1].status != "ok":
        gaps.append(f"unexpected outcome {overdue[-1].status}")
    return gaps


TURNS: List[Turn] = [
    Turn(
        "Hey! What am I working on at the moment?",
        "Reads open tasks (or the project overview) without a period and without changing anything.",
        check_open_question,
    ),
    Turn(
        "By the way, I never work on Fridays, so keep that in mind. "
        "How many tasks did I add last week?",
        "Stores the lasting preference; reads tasks by creation date over last week, Monday to "
        "Sunday, with dates copied from the calendar block.",
        check_preference_and_last_week,
    ),
    Turn(
        "What did I get done last month?",
        "Reads by completion date over the previous calendar month.",
        check_finished_last_month,
    ),
    Turn(
        "And how much of that was for the garden?",
        "Names no period: reuses last month and the completion axis, narrowed to the Garden project.",
        check_follow_up_reuses_window,
    ),
    Turn(
        "Do I have anything about taxes on my list?",
        "A text filter that matches nothing: says nothing matched, invents no task.",
        check_nothing_matched,
    ),
    Turn(
        "How busy was I around this time last year?",
        "The period is before any data: says so and states the range that is actually covered.",
        check_before_coverage,
    ),
    Turn(
        "What's overdue in the home project?",
        "Overdue = deadline up to yesterday and still open. 'home' matches two projects: lists "
        "both and asks, picks neither.",
        check_ambiguous_project,
    ),
    Turn(
        "The renovation one.",
        "Repeats the overdue question for Home renovation and answers it.",
        check_clarified_project,
    ),
    Turn(
        "Add a task to call the plumber about the bathroom leak. It belongs to the renovation, "
        "it's high priority and it has to happen by tomorrow.",
        "Creates one task in Home renovation, high priority, deadline tomorrow.",
        check_task_added,
    ),
    Turn(
        "Set the dentist appointment for a week from today and bump it to high priority.",
        "Updates the one matching task: deadline today plus seven days, priority high.",
        check_task_changed,
    ),
    Turn(
        "I've ordered the kitchen tiles, so tick that one off.",
        "Marks the one matching task as finished.",
        check_task_finished,
    ),
    Turn(
        "Delete the report task, I don't need it anymore.",
        "Three tasks match 'report': refuses, lists all three, asks; deletes nothing.",
        check_refused_deletion,
    ),
    Turn(
        "The expense one.",
        "Deletes exactly 'Expense report'.",
        check_completed_deletion,
    ),
    Turn(
        "What do you remember about me?",
        "Reads long-term memory and reports the Friday preference.",
        check_recall,
    ),
    Turn(
        "Forget the Friday thing, my schedule changed. And is anything overdue now, across everything?",
        "Deletes the note; reads overdue across all projects.",
        check_forget_and_overdue,
    ),
]


# --- API client ------------------------------------------------------------------------------


class Api:
    def __init__(self, base_url: str):
        self._http = httpx.Client(base_url=base_url, timeout=TURN_TIMEOUT_SECONDS)
        self._headers: Dict[str, str] = {}

    def health(self) -> None:
        self._http.get("/health").raise_for_status()

    def create_user(self) -> str:
        response = self._http.post("/api/v1/users", json={"display_name": "Acceptance run"})
        response.raise_for_status()
        token = response.json()["user_id"]
        self._headers = {"Authorization": f"Bearer {token}"}
        return token

    def create_chat(self) -> str:
        response = self._http.post("/api/v1/chats", json={"title": "Acceptance run"}, headers=self._headers)
        response.raise_for_status()
        return response.json()["chat_id"]

    def send(self, chat_id: str, content: str) -> httpx.Response:
        return self._http.post(
            f"/api/v1/chats/{chat_id}/messages", json={"content": content}, headers=self._headers
        )

    def all_tasks(self) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        while True:
            response = self._http.get(
                "/api/v1/tasks", params={"limit": 100, "offset": len(tasks)}, headers=self._headers
            )
            response.raise_for_status()
            page = response.json()
            tasks.extend(page["tasks"])
            if not page["has_more"]:
                return tasks

    def projects(self) -> Dict[str, str]:
        response = self._http.get("/api/v1/projects", headers=self._headers)
        response.raise_for_status()
        return {item["project_id"]: item["name"] for item in response.json()["projects"]}

    def memory(self) -> List[Dict[str, Any]]:
        response = self._http.get("/api/v1/memory", headers=self._headers)
        response.raise_for_status()
        return response.json()["memories"]


def _tool_calls(debug: Dict[str, Any]) -> List[ToolCall]:
    return [
        ToolCall(call["name"], call.get("arguments") or {}, call.get("result") or {})
        for iteration in debug.get("request_flow", [])
        for call in iteration.get("function_calls", [])
    ]


def _by_title(tasks: List[Dict[str, Any]], fragment: str) -> List[Dict[str, Any]]:
    return [task for task in tasks if fragment.lower() in task["title"].lower()]


def verify_state(api: Api, calendar: Calendar, tz: ZoneInfo) -> List[str]:
    """Confirm through REST that the tasks are in the state the conversation claims."""
    tasks = api.all_tasks()
    projects = api.projects()
    gaps: List[str] = []

    def local_day(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return datetime.fromisoformat(value).astimezone(tz).date().isoformat()

    plumber = _by_title(tasks, "plumber")
    if len(plumber) != 1:
        gaps.append(f"created task: expected one 'plumber' task, found {len(plumber)}")
    else:
        task = plumber[0]
        if projects.get(task["project_id"]) != "Home renovation" or task["priority"] != "high":
            gaps.append(f"created task has project/priority {projects.get(task['project_id'])}/{task['priority']}")
        if local_day(task["due_at"]) != calendar.tomorrow.isoformat():
            gaps.append(f"created task is due {task['due_at']}")

    dentist = _by_title(tasks, "dentist")
    if len(dentist) != 1:
        gaps.append(f"updated task: expected one 'dentist' task, found {len(dentist)}")
    elif (
        local_day(dentist[0]["due_at"]) != calendar.in_a_week.isoformat()
        or dentist[0]["priority"] != "high"
    ):
        gaps.append(f"updated task is due {dentist[0]['due_at']} with priority {dentist[0]['priority']}")

    tiles = _by_title(tasks, "kitchen tiles")
    if len(tiles) != 1:
        gaps.append(f"completed task: expected one 'kitchen tiles' task, found {len(tiles)}")
    elif tiles[0]["status"] != "done" or not tiles[0]["completed_at"]:
        gaps.append(f"completed task has status {tiles[0]['status']}")

    remaining = sorted(task["title"] for task in _by_title(tasks, "report"))
    if remaining != ["Quarterly report", "Weekly status report"]:
        gaps.append(f"after the deletion the 'report' tasks are {remaining}")

    if any("friday" in note["content"].lower() for note in api.memory()):
        gaps.append("the forgotten note is still stored")
    return gaps


def run(args: argparse.Namespace) -> int:
    tz = ZoneInfo(args.timezone)
    calendar = Calendar(today=datetime.now(tz).date())
    api = Api(args.base_url)

    api.health()
    token = api.create_user()
    subprocess.run(shlex.split(args.seed_command.format(token=token)), check=True)
    seeded_reports = sorted(task["title"] for task in _by_title(api.all_tasks(), "report"))
    chat_id = api.create_chat()

    records: List[TurnRecord] = []
    for number, turn in enumerate(TURNS, start=1):
        record = TurnRecord(number, turn.message, turn.expectation)
        response = api.send(chat_id, turn.message)
        if response.status_code != 200:
            record.gaps.append(f"HTTP {response.status_code}: {response.text[:300]}")
        else:
            body = response.json()
            record.reply = body["assistant_message"]["content"]
            record.calls = _tool_calls(body["debug"])
            record.gaps = turn.check(record, calendar)
            if turn.check is check_refused_deletion:
                still_there = sorted(task["title"] for task in _by_title(api.all_tasks(), "report"))
                if still_there != seeded_reports:
                    record.gaps.append(f"tasks changed during the refused deletion: {still_there}")
        records.append(record)
        print(f"[{number:02d}] {record.verdict}")
        print(f"     > {turn.message}")
        print(f"     < {record.reply[:400]}")
        for call in record.calls:
            print(f"     * {call.name}({json.dumps(call.arguments)}) -> {call.status or call.result}")

    state_gaps = verify_state(api, calendar, tz)
    print("STATE " + ("pass" if not state_gaps else "; ".join(state_gaps)))

    report = {
        "date": calendar.today.isoformat(),
        "chat_id": chat_id,
        "turns": [
            {
                "turn": record.number,
                "message": record.message,
                "expected": record.expectation,
                "reply": record.reply,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments, "outcome": call.status, "result": call.result}
                    for call in record.calls
                ],
                "verdict": record.verdict,
            }
            for record in records
        ],
        "state_verification": state_gaps or "pass",
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    failed = [record.number for record in records if record.gaps]
    print(f"{len(records) - len(failed)}/{len(records)} turns passed; report written to {args.report}")
    return 0 if not failed and not state_gaps else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timezone", default="UTC", help="must equal the server's AGENT_TIMEZONE")
    parser.add_argument("--seed-command", default=DEFAULT_SEED_COMMAND)
    parser.add_argument("--report", default="acceptance-report.json")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
