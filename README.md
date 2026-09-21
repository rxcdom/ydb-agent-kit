# ydb-agent-kit

A small, complete reference project: a hexagonal Python service on **YDB** with a **tool-calling
LLM agent** on top, runnable locally with one `docker compose up`.

The demo domain is a task manager, and it is deliberately a placeholder. Nothing outside
`src/tasks/` and the agent's tool layer knows about tasks; swap the domain and the machinery
around it stays. What the project is really about is that machinery.

## What it demonstrates

| # | Capability | Where |
|---|---|---|
| 1 | Hexagonal architecture (domain / ports / application / adapters) with enforced module boundaries and dependency injection | every module under `src/`, `.importlinter`, `src/gateway/di/container.py` |
| 2 | YDB as the primary datastore | `src/shared/infrastructure/database/ydb/connection.py`, the `ydb` compose service |
| 3 | A YDB data-access layer: data mappers, a generic repository, a transaction manager, connection handling | `src/shared/infrastructure/database/ydb/` |
| 4 | A migration framework: discovery, planning, execution, CLI, and post-apply verification of the declared artifacts against the live schema | `src/shared/infrastructure/database/migration/` |
| 5 | An LLM agent: tool-calling loop, tool registry and dispatcher, pydantic tool schemas | `src/agent/application/loop/`, `src/agent/application/tools/` |
| 6 | Error handling throughout: typed domain errors, recovery from malformed model output, one HTTP mapping | `src/*/domain/exceptions.py`, the dispatcher, `src/gateway/api/errors.py` |
| 7 | Long-term memory: remember / recall / forget, a privacy filter, pointer-style exposure in the prompt | `src/agent/application/memory/`, `src/agent/domain/services/` |
| 8 | An LLM client with retry classification, graceful degradation, provider-error mapping and a model capability registry | `src/agent/adapters/llm/` |
| 9 | Context engineering: a rendered calendar block of precomputed dates, and conversation state derived from the agent's own tool-call trace | `calendar_block_renderer.py`, `conversation_state_writer.py` |

Three ideas carry most of the agent's reliability:

- **The model never computes a date.** Every turn gets a calendar block with ready-made periods
  ("last week", "last month") and the coming days. "What is overdue" is answerable only by
  comparing deadlines with a *today* the model was handed.
- **Tools report the limits of their data.** A read does not just return rows or nothing: it says
  whether the period lies outside the data (`coverage_gap`), inside it but empty (`no_records`), or
  whether a filter matched nothing (`empty_filter`), and it always carries the covered range.
- **A write never acts on an ambiguous reference.** The agent addresses tasks by the user's words,
  never by id. When "the report task" matches three rows, the tool refuses, returns the candidates
  and changes nothing; the prompt obliges the agent to ask.

## Non-goals

Left out on purpose, and not planned: cloud infrastructure and CI/CD, real authentication and
account lifecycle, webhooks and external integrations, push notifications, an admin panel or any
frontend, feature flags, observability middleware, prompt templates stored in the database,
assistant personalities, attachments, a second LLM provider. The `LLMClient` port plus the model
profile registry are the intended extension point for another provider.

## The credential is not authentication

`POST /api/v1/users` returns a user id, and **that id is the bearer credential** of every other
request. It still travels in a standard `Authorization: Bearer` header and the owner is still
resolved server-side, which is what the demo needs in order to show owner scoping. But it is not
authentication: a user id has no secret component, it shows up in logs and shell history, it cannot
be rotated or revoked, and anyone who learns another user's id can act as that user. This is
acceptable only because the project runs locally against a disposable database. If you adapt this
project, replace `src/accounts/` wholesale.

## Run book

Requirements: Docker with Compose, and a Yandex AI Studio folder id and API key for the agent
(everything except the chat endpoints works without them).

```bash
cp .env.example .env            # fill YC_FOLDER_ID and YC_API_KEY
docker compose up -d --build    # starts YDB, waits for it, migrates, serves on :8000
curl -s localhost:8000/health

# create a user; the returned user_id is the token
curl -s -X POST localhost:8000/api/v1/users -H 'Content-Type: application/json' \
     -d '{"display_name":"Demo"}'
export TOKEN=<user_id from the response>

# seed a deterministic synthetic workspace: 4 projects, about 60 tasks over 9 months
docker compose run --rm seed --token "$TOKEN"

# chat
CHAT=$(curl -s -X POST localhost:8000/api/v1/chats -H "Authorization: Bearer $TOKEN" | jq -r .chat_id)
curl -s -X POST localhost:8000/api/v1/chats/$CHAT/messages \
     -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"content":"What is overdue in the home project?"}' | jq .
```

The seeded workspace has two projects whose names overlap, `Home renovation` and `Home office`, so
the reply lists both and asks which one is meant instead of picking one. `jq .debug.request_flow`
shows every tool call of the turn with its arguments and its result.

If port 8000, 2136 or 8765 is taken on your machine, change `APP_PORT`, `YDB_GRPC_HOST_PORT` or
`YDB_MON_HOST_PORT` in `.env`. Other tools:

```bash
docker compose run --rm migrate status     # migration status table
docker compose run --rm migrate up         # apply pending migrations (the app does this on start)
docker compose down                        # stop; the data stays on the ydb_data volume
docker compose down -v                     # stop and drop the database
```

The database lives on the `ydb_data` volume and survives a restart of the YDB container. The YDB
image is built for `linux/amd64`; on Apple Silicon it runs under emulation and needs about a minute
to become healthy on first start.

Interactive API documentation is served at `http://localhost:8000/docs`.

## Architecture

```
                 gateway  (composition root: FastAPI app, DI container, CLI)
                 |           |            |
                 v           v            v
             accounts      tasks        agent --(port TaskDataProvider)--> tasks use cases
                 |           |            |
                 +-----------+------------+--> shared.domain          (UserId, base errors)
                 +-----------+------------+--> shared.infrastructure  (YDB access, migrations; adapters only)
```

Every module has the same four layers, and the dependency direction is one way:

```
adapters  ->  application  ->  ports  ->  domain
(YDB, HTTP,   (use cases,      (abstract   (entities, value objects,
 LLM SDK)      agent loop)      interfaces)  pure services)
```

- `domain` imports nothing but the standard library and the shared kernel.
- `application` orchestrates domain objects through `ports`; it never sees YDB, FastAPI or the SDK.
- `adapters` implement the ports. Only they import third-party I/O libraries.
- `gateway` is the only place that instantiates one module's classes for another. The agent needs
  task data, so it declares a `TaskDataProvider` port; the gateway binds it to an adapter that calls
  the tasks use cases. The agent never imports a tasks type.

These rules are contracts in `.importlinter`, and `tests/architecture` runs them, so a plain
`pytest` fails when a boundary is crossed. Two cross-module imports are sanctioned there and
nowhere else: the provider adapter just described, and the bearer dependency of `accounts` that the
other routers authenticate through.

**Owner scoping.** Every chat, message, memory note, project and task carries `user_id`. The owner
always comes from the authenticated principal and flows as a keyword-only argument through
use case -> loop -> dispatcher -> tool executor -> provider port. No tool schema has a field for a
user, a chat or a row id, and a test asserts that.

**One turn of the agent** (`SendMessageUseCase`): persist the user message as `processing` ->
load recent history -> render the calendar block, the conversation-state block and the memory
pointer -> run the tool loop -> in one YDB transaction mark the user message `sent` and store the
assistant message with its tool-call trace -> record the conversation state from that trace. If
the model fails, the user message stays stored as `failed` and the 503 body names it.

**YDB notes.** Text columns are `Utf8`. A secondary index is used only when the query names it
(`FROM tasks VIEW idx_tasks_user_due`), so every repository method that filters on an indexed
column names its index, and a unit test asserts the rendered YQL. Migrations declare the tables,
indexes and typed columns they create; after `up()` the framework checks each one against the live
schema and marks the migration `failed` if anything is missing.

## Configuration

All settings come from `.env` / the environment and are validated at startup; see `.env.example`
for the full list. The ones you are likely to touch:

| Variable | Purpose | Default |
|---|---|---|
| `YC_FOLDER_ID`, `YC_API_KEY` | Yandex AI Studio credentials (`YC_IAM_TOKEN` as an alternative to the key) | empty |
| `LLM_MODEL_NAME` | a model from `src/agent/adapters/llm/model_profiles.py`; it must support tool calling, otherwise the app refuses to start | `gpt-oss-120b` |
| `AGENT_TIMEZONE` | IANA zone of the calendar block and of all day boundaries | `UTC` |
| `AGENT_MAX_ITERATIONS` / `AGENT_HISTORY_LIMIT` | tool-loop cap / messages sent to the model | `5` / `20` |

## Tests

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# unit, architecture and sanitisation suites: no Docker, no network
.venv/bin/python -m pytest tests/unit tests/architecture tests/sanitisation

# integration suite against the compose YDB, inside the compose network
docker compose up -d ydb
docker compose run --rm tests

# ...or from the host, through the published port
YDB_ENDPOINT=grpc://localhost:2136 YDB_DISABLE_DISCOVERY=true \
    .venv/bin/python -m pytest tests/integration -m integration
```

`YDB_DISABLE_DISCOVERY=true` is needed from the host because YDB's endpoint discovery answers with
the container's own host name, which only resolves inside the compose network.

The unit suites never construct the provider SDK and never sleep. Plugin autoloading is disabled
for the test session, so a package that merely happens to be installed cannot hook into it.

**Acceptance run.** `scripts/acceptance_run.py` drives the running stack with a real model: one
fresh user, one seeded workspace, one chat, fifteen natural-language messages that depend on each
other. It checks the tool-call trace of every turn and afterwards confirms through the REST API
that the claimed writes happened and the refused deletion did not.

```bash
.venv/bin/python scripts/acceptance_run.py --base-url http://localhost:8000
```

**Sanitisation gate.** `scripts/check_sanitised.py` (also run by `tests/sanitisation`) rejects
non-English text, key-shaped strings, local paths and oversized fixtures. Private tokens that must
never be published are read from an untracked `.sanitisation-denylist` file rather than listed in
the script; without that file the check reports itself as skipped.

## Known limitations

Observed with the default model (`gpt-oss-120b`, low reasoning effort) over some twenty fresh
acceptance runs. With the final prompt four of the last six runs passed all fifteen turns; the
other two failed as described in the first two items. None of this is worked around in code: there
is no special case for a phrase, and replies are never rewritten. The prompt states the rule and
the model follows it most of the time.

- **A second request in one message can be dropped.** "By the way, I never work on Fridays. How
  many tasks did I add last week?" asks for two things. Before the prompt told the agent to make
  memory calls first, the note was skipped in two runs out of five, once while the reply claimed
  "I'll remember that". Since then it was skipped in one run out of fourteen, and the false claim
  did not reappear. Later turns then fail honestly ("I have nothing stored about you"). This is
  why `GET /api/v1/memory` exists: what was stored is observable without asking the agent.
- **Reply text can mis-copy a value.** In about one run out of four the reply to that same message
  named the right days in the wrong year (2024) although the tool call and its result carried the
  right one. The trace in `debug` and the stored data are the source of truth, not the prose.
- **An already stored note is sometimes stored again** a few turns later, because earlier tool
  calls are not part of the history the model sees. Storing the same fact twice writes nothing
  (`action: unchanged`), so the vault is not affected; the call is merely redundant.
- **The route to an ambiguity varies.** Asked about "the home project" or told to delete "the
  report task", the agent usually sends the reference to the tool and gets the refusal with the
  candidates; sometimes it looks the name up first and asks on its own. Both end with every
  candidate listed and a question, and in neither case is anything picked or deleted.
- **A loosely worded period can trigger a question instead of an answer.** "Around this time last
  year" was once answered with "which dates do you mean?". The prompt now tells the agent to pick
  the closest reasonable period and say which dates it used.
- **Replies are typeset.** Dates come with non-breaking hyphens (`2026‑09‑21`), names with narrow
  no-break spaces. Replies are not post-processed, so a client that parses them must normalise.
- **Only text survives between turns.** Earlier tool results are not replayed to the model; what a
  follow-up needs (the last window, date axis and project) travels in the conversation-state block.
  A follow-up that depends on other details of an earlier result makes the agent read again.
- **A turn that dies for a non-model reason** (a bug, or the datastore failing mid-turn) returns
  500 or 503 and leaves the user message in status `processing`; only model failures mark it
  `failed`.
- **No rate limiting, no pagination of chats, no streaming.** Out of scope for the demo.

## License

MIT
