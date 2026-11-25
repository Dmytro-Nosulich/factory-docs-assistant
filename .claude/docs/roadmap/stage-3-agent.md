# Stage 3: Agent with tools

Status: ACTIVE. Last updated: 2026-11-25.

Assignment: https://docs.superhuman.com/d/2-AI-Engineering_d0Z0RsMXXN6/3_su3wbUXN (Homework 3). The page needs JavaScript, so open it in a browser.

Course dates:
- Checkpoint after lesson 12, by 22.11: steps 1-2, no points. It had already passed when the assignment was read.
- Feedback in lesson 14, 26.11.
- Deadline 02.12 at 23:59.

The user said not to plan around course dates, so no task depends on them.

A summary of the assignment is in `course-materials/Stage3/homework-3-summary.md` (gitignored).

## Goal
Turn the stage 2 RAG into an agent. The agent decides for itself when to search the corpus, when to call tools and when to answer. It uses at least two narrow tools, at least one of them through our own MCP server. It remembers context, and user feedback changes how it answers.

`question -> agent (loop with limits) -> tools: corpus search (stage 2) · our MCP server (SDS registry) · review ticket -> memory + user feedback -> answer with sources`

## Course requirements (summary of the assignment)
Every step also has requirements for when things go wrong. The task set must hold a task that checks each one.

- **Step 1, agent and tool calling:**
  - At least 2 tools, each doing one thing (no "god tool").
  - Each tool has a description and an argument schema (Pydantic or JSON Schema): when to call it, when not to, the argument format, and what it returns. The descriptions live apart from the code and are versioned like prompts.
  - Execution control: a step limit, a token or $ budget, a timeout per tool call and per request. The stop reason goes into the trace. A "check all documents" request stops at the budget with a partial result.
  - Invalid arguments, a tool exception and a tool timeout go back to the model as text. They do not crash the agent.
  - A fault wrapper switched on by a flag (for example `FAULTS=timeout,schema`):
    - On timeout, 429 or 503, the agent retries only safe calls, with backoff. After the limit, it says honestly that the service is unavailable.
    - On a changed API format, the tool response fails a schema check, and the error shows in the trace. It never becomes a silent None.
  - A side-effect action runs only after human confirmation checked in code. When a required parameter is missing, the agent asks for it. A repeated request does not create a duplicate (idempotency key).
  - At least 15 tasks with deterministic checks:
    - Is the answer correct?
    - Were the right tools called?
    - Was any action taken without permission?

    Include tasks that need no tool, tasks with no answer, and a task for every fault in steps 1-3.
- **Step 2, our own MCP server:**
  - The server gives the agent at least one tool, and the agent uses it as an MCP client in the step 1 tasks.
  - Input validation and least privilege: read-only, whitelisted. No arbitrary SQL, shell or file reads.
  - A denial test: on "access denied", the agent does not look for a way around it and explains why it cannot answer.
  - The README explains why this tool is in MCP and not plain function calling.
- **Step 3, agentic RAG and memory:**
  - Corpus search is an agent tool. The agent decides whether to search, with which query, and whether to search again. Answers cite sources.
  - Add a document with an embedded instruction for the model, and two versions of one rule with different dates.
    - The injected instruction is not followed and causes no side effect.
    - The agent picks the current version and names the conflict, citing both versions.
  - Memory, short-term or long-term: what is stored, when, and how stale entries are removed.
  - User feedback (a rating or a correction) changes the next answer. Show it with a before-and-after dialog or a test.
  - REPORT.md, from runs of the task set on the final agent (several runs if results vary):
    - a metrics table: pass rate, required tools called, actions without permission, steps and price per task;
    - 3 failures analysed;
    - a fault table: expected, what happened in the first run, and what we changed.
- **Step 4, reflection:**
  - `REFLECTION.md` with 3-5 concrete episodes with the coding agent: what we asked, what it did well, what was rewritten and why.
  - How the generated code was checked, with links to commits or files.
  - The coding agent's instructions are in the repo.
- **Stretch (no points):** several agents with roles, A2A, agent eval in CI with a threshold, Langfuse or LangSmith tracing.
- **Technical:**
  - Python 3.10+, one-command run (`make run` or `python main.py`).
  - Any model with tool calling. Give the cost per 1000 requests if it is paid.
  - The MCP server starts with the same command or with one described in the README.
  - Keys only in `.env`.
  - One GitHub link.
- **Repo must contain:**
  - the code of the agent, the tools and the MCP server;
  - `prompts/`: the system prompt and the tool descriptions, with versions and a changelog;
  - `evals/`: the task set, the checks, and run results with traces;
  - `REPORT.md`;
  - `REFLECTION.md`;
  - `README.md`: the agent, its tools, how to run it and the MCP server, why MCP, memory and feedback.
- **Not needed:** UI, deployment, fine-tuning, a complex multi-agent system, perfect accuracy.
- **Common mistakes that lose points:**
  - no step limit, so the agent loops on an unanswerable question;
  - only the happy path tested;
  - a god tool;
  - an MCP server without access restrictions;
  - a side effect allowed only by a prompt line;
  - a reflection in general words.

## Grading rubric (40 points, from the course)
| Step | Criterion | Pts | Our tasks |
|------|-----------|-----|-----------|
| 1. Agent and tool calling | At least 2 tools with clear descriptions and argument schemas | 3 | S3-02, S3-03 |
| | Execution control: step limit, timeout, budget | 4 | S3-04 |
| | Argument validation, errors and tool faults: timeout / 503, format change, duplicate action | 4 | S3-03, S3-05, S3-06 |
| 2. Own MCP server | A working MCP server, at least 1 tool used by the agent | 4 | S3-07, S3-08 |
| | Permissions, validation and safety of server tools; denial without a workaround | 3 | S3-07, S3-08 |
| | Explained why MCP and not function calling | 1 | S3-08, S3-18 |
| 3. Agentic RAG and memory | Retrieval as the agent's decision; injection in a document and conflicting versions | 5 | S3-09, S3-10 |
| | Memory (short-term or long-term) | 3 | S3-11 |
| | User feedback | 3 | S3-12 |
| Task set and report | At least 15 tasks with deterministic checks, faults included; REPORT.md with metrics, 3 failures, fault table | 2 | S3-13, S3-14, S3-16 |
| 4. AI-first reflection | Concrete: what the coding agent did well, what was rewritten | 3 | S3-17 |
| | Signs of reviewing generated code: tests, fixes, reasoning | 3 | S3-17 |
| Repo | README, one-command run, keys not in code | 2 | S3-01, S3-15, S3-18 |

Most points go to the MCP server, execution control and retrieval as the agent's decision.

## Decisions for this stage
Decisions 21-30 are in `docs/PROJECT.md`:
- the agent loop;
- the agent model;
- the MCP server;
- the side-effect tool;
- memory;
- the problem corpus;
- the task set;
- the repo layout;
- the budget;
- the instruction files in the repo.

**Fault log rule.** From S3-05 on, the first-run outcome of every fault task is written down when it happens, in the task's DONE note here. REPORT.md needs a "first run" column, and that cannot be filled in honestly afterwards.

## Phase A: Setup
- [x] **S3-00 Roadmap and decisions** (Claude). DONE 2026-11-25, reviewed by the user.
  - This file, `00-overview.md` and decisions 21-30 are written, and `course-materials/Stage3/homework-3-summary.md` is saved. The assignment was read in Chrome.
  - Nothing is committed, because `.claude/` is gitignored.
  - Done when: the user has reviewed the roadmap.
- [x] **S3-01 Repo layout for stage 3** (Claude). DONE 2026-11-25:
  - `make run-rag` is the old stage 2 `make run`: its `make -n` output is identical to the old one. `make run` prints where the stage 2 and stage 1 runs are and exits non-zero.
  - CI runs `make run-rag` and uploads `docs/reports/stage-2.md`. Not run yet; it runs on the next push.
  - `REPORT.md` was moved with `git mv` to `docs/reports/stage-2.md`. The six scripts that write it point there, and the generated links get `../../`.
    - Rebuilt offline from the committed results with `--report`, `make cost` and `make report-summary`. The only differences are the 8 link lines.
    - `evals/results/cost.md` had a link that was already broken (it pointed at `evals/results/retrieval.md` from inside `evals/results/`). It now says `retrieval.md`, because `cost.caveat` takes the link target as a parameter.
  - New root `REPORT.md` skeleton for stage 3, with no numbers.
  - `make run` is renamed to `make run-rag` in code docstrings, messages, `config.toml` comments, `docs/METRICS.md` and the README. A README line says stage 3 is in progress.
  - `.gitignore` tracks `.claude/CLAUDE.md` and `.claude/docs/` only.
  - Verified: 1631 tests pass, `make lint` passes, `git check-ignore` still ignores agents, skills, commands and local settings, and a secret scan of the new files finds nothing.
  - Work:
    - `make run-rag` is the old stage 2 `make run`, and `.github/workflows/evals.yml` calls it, so the stage 2 gate keeps running in CI. `make run` is a placeholder that exits with an error until S3-15.
    - The stage 2 report moves to `docs/reports/stage-2.md` (with `report.py` and the report-summary scripts pointed at it), and root `REPORT.md` becomes the stage 3 skeleton.
    - `.gitignore` stops ignoring `.claude/CLAUDE.md` and `.claude/docs/` (decision 30), and both are committed. The user reviewed them for personal data on 2026-11-25.
  - Done when: `make -n run-rag` prints the same commands as the old `make -n run`, `make test` and `make lint` pass, and `make report-summary` writes into the moved report.

## Phase B: Course step 1, agent and tool calling
- [x] **S3-02 Tool-calling client** (Claude). DONE 2026-11-25:
  - `OpenRouterClient.chat_with_tools(messages, tools, tool_choice)` sends OpenAI-format tools with no `response_format`, and returns a `ToolTurn`: the usual `ModelResponse` (tokens, cost, timing), the `ToolCall`s with their raw JSON arguments (parsed and validated in S3-03), and the assistant message to put back in the history. It is retried like `complete`. A tool call with no id, name or arguments fails with `ModelCallError`.
  - Stage 1 and 2 bodies: two tests compare the exact request bytes with ones captured from the S3-01 code. They passed before and after the change.
  - `make model-info` (`GET /api/v1/models`, free), 2026-11-25: `anthropic/claude-haiku-5.5` lists `tools` and `tool_choice`, $0.10 in / $0.50 out per 1M tokens (unchanged; $0.50 / $2.50 from 100k prompt tokens), 1M context. Recorded in a `config.toml` comment.
  - `make tool-smoke`, 2026-11-25: Haiku 5.5 (Claude Platform on AWS) called `lookup_flash_point({"product_name": "TEST-SOLVENT-1"})` once, then answered in text from the canned result. 2 calls, 440 + 559 prompt tokens, $0.000159. The toy tool and product are made up. The log is in `results/agent/smoke/` (gitignored).
  - `results/agent/` is gitignored and left out of the stage 1 guard (decision 29).
  - Verified: 1644 tests pass, and `make lint` passes.
  - Work: `OpenRouterClient` can send `tools` (OpenAI format) and returns the model's `tool_calls` with usage and cost. Stage 1 and 2 request bodies stay byte-identical.
  - Done when:
    - unit tests pass with a mocked transport;
    - a live smoke test shows Haiku 5.5 making one tool call;
    - the model's tool support and price are checked on `GET /api/v1/models` and recorded.
- [ ] **S3-03 Tool registry** (Claude).
  - Work:
    - Each tool has a Pydantic argument schema and a description file `prompts/tools/<tool>_v1.md` (when to call it, when not, arguments, return value), listed in `prompts/CHANGELOG.md`. Tools:
      - `search_corpus`: stage 2 hybrid + rerank, returns chunk IDs, document, section and text, with optional filters;
      - `read_chunk`: one chunk by ID, with its neighbours.
    - Validation errors, exceptions and timeouts come back to the model as a short text it can act on.
  - Done when: tests show an invalid argument, an exception and a timeout each reaching the model as text, with the agent still running.
- [ ] **S3-04 Agent loop with limits and trace** (Claude).
  - Work: our own loop (decision 21) with configurable limits in `config.toml` `[agent]`:
    - a step limit;
    - a token and $ budget per request;
    - a timeout per tool call and per request.

    Every request writes a trace (each model call, each tool call with arguments, result or error, duration, tokens and cost, and the stop reason) to `results/agent/` (gitignored).
  - Done when: a "check all documents" request stops on the budget or step limit and returns a partial result with the stop reason, both in a unit test and in one live run.
- [ ] **S3-05 Fault injection and retries** (Claude).
  - Work:
    - A wrapper around every tool, switched on by `FAULTS=timeout,429,503,schema`.
    - Read-only (safe) calls are retried with backoff, and a side-effect call is never retried blindly.
    - After the limit, the agent answers that the service is unavailable.
    - Every tool response is checked against its response schema. A renamed or missing field becomes a visible error in the trace, never a silent None.
  - Done when: unit tests cover each fault, a live run with each flag behaves as described, and the first-run outcomes are recorded (fault log rule).
- [ ] **S3-06 Side-effect action with confirmation** (Claude).
  - Work: `create_review_ticket` asks for a safety review of a product (decision 24).
    - It writes to a gitignored local JSONL outbox only after a confirm policy in code returns yes. It is interactive in the CLI and injected in evals.
    - The agent asks the user when a required parameter is missing.
    - An idempotency key (from the request and the arguments) stops a retried call from writing a second ticket.
  - Done when: tests show no ticket without confirmation, a clarifying question for a missing parameter, and one ticket after a duplicate call. The first-run outcomes are recorded.

## Phase C: Course step 2, our own MCP server
- [ ] **S3-07 MCP server: SDS registry** (Claude).
  - Work: the official `mcp` Python SDK (explained before it is added) runs a stdio server.
    - Data: a read-only SQLite file built by a script from the stage 1 extraction results for the 35 SDS: product, language, signal word, flash point, UN number, hazard codes and components.
    - Narrow tools: `lookup_product(name)` and `find_by_cas(cas)`.
    - Inputs are validated (length, CAS format). The database opens read-only, with no query text from the model, no shell and no file paths.
    - Every record has an access group. A caller without that group gets "access denied".
  - Done when: server tests show each restriction (invalid input, a denied record, read-only).
- [ ] **S3-08 Agent as MCP client** (Claude).
  - Work:
    - The agent starts the server as a subprocess with the same command, lists its tools, and uses them in real tasks, for example "flash point of X" or "which products contain CAS Y".
    - On "access denied", the agent tells the user why it cannot answer and tries no other route.
    - The README section "why MCP" is written.
  - Done when: a live task uses an MCP tool, and a live denial task passes. The first-run outcome is recorded.

## Phase D: Course step 3, agentic RAG and memory
- [ ] **S3-09 Agentic retrieval** (Both).
  - Work: system prompt `prompts/agent_v1.md`.
    - The agent decides whether to search, writes the query, and searches again with a new query or a filter when the hits do not fit.
    - The final answer cites chunk IDs with quotes, checked by the existing `check_citations` in `src/factory_docs/answer.py`.
    - Retrieved text is wrapped as untrusted data.
  - Done when: live traces show one task answered with no search, and one where the agent searched twice.
- [ ] **S3-10 Problem corpus: injection and conflicting versions** (Both).
  - Work:
    - New synthetic document D078 with an embedded instruction to the model. It is written by us, marked as a test fixture in `data/manifest.csv`, and indexed.
    - Version conflict: use an existing old/new pair (D066/D067, D071/D072, D074/D075) if one changes a rule value; otherwise add a pair.
    - Check that the instruction causes no action, and that the agent picks the current version and names the conflict, citing both versions.
  - Done when:
    - the injection and conflict tasks pass live;
    - the first-run outcomes are recorded;
    - `make run-rag` still passes its eval gate with D078 in the index.
- [ ] **S3-11 Memory** (Claude).
  - Work (decision 25):
    - Short-term: dialog history capped by tokens, with older turns replaced by a model summary.
    - Long-term: a gitignored JSON store of user facts and feedback, one entry per key, each with created date, source and expiry. A newer entry for the same key replaces the old one, and expired entries are dropped on load.
  - Done when: tests show what is stored and when, the cap and summary, and expiry and replacement. The README describes it.
- [ ] **S3-12 User feedback** (Claude).
  - Work: after an answer, the user can rate it or correct it. A correction is saved in long-term memory and given to the agent on later requests.
  - Done when: a test, and a README dialog from a real run, show the same question answered differently before and after a correction.

## Phase E: Task set and report
- [ ] **S3-13 Task set** (Me; drafted by Claude on request, decision 27).
  - Work: about 20 tasks in `evals/agent_tasks.toml`. Each has deterministic checks: key facts in the answer, tools that must and must not be called, actions that must not happen, and the expected stop reason. Categories:
    - normal questions;
    - no tool needed;
    - no answer in the corpus;
    - a budget stop;
    - timeout / 429 / 503;
    - schema change;
    - side effect without confirmation;
    - missing parameter;
    - duplicate;
    - MCP denial;
    - injection;
    - version conflict;
    - memory;
    - feedback.
  - Done when: `make agent-tasks-check` validates the file, and the user has reviewed every task.
- [ ] **S3-14 Task runner and metrics** (Claude).
  - Work: `make eval-agent` runs every task N times (3 by default) with its faults and confirm policy. It reports:
    - the pass rate;
    - the rate of required tools called;
    - actions without permission;
    - steps and $ per task;
    - the spread across runs.

    It writes a run file and the traces to `evals/results/agent/` (committed).
  - Done when: a full run finishes, and the tables are rebuilt from saved runs with no model calls.
- [ ] **S3-15 One-command run** (Claude).
  - Work: `make run` does the following:
    - checks the key and the stage 3 budget;
    - builds the stage 2 corpus and index when stale, and the registry database;
    - runs `make agent-tasks-check` and `make eval-agent`;
    - answers a few sample dialogs (starting the MCP server itself);
    - prints the cost per 1000 requests.
  - Done when: `make run` exits 0 from a fresh clone (or a clean checkout).
- [ ] **S3-16 REPORT.md** (Both).
  - Work, every number from a script:
    - the metrics table;
    - 3 failures (expected, actual, why);
    - the fault table (expected, first run from the fault log, what we changed);
    - the cost per 1000 requests.
  - Done when: the user has reviewed it.

## Phase F: Course step 4, reflection, and submission
- [ ] **S3-17 REFLECTION.md** (Both).
  - Work: 3-5 concrete episodes from stages 1-3, each with:
    - what was asked;
    - what the coding agent did well;
    - what was rewritten, and why;
    - links to commits.

    Also how the code was checked: tests, diff review, fixes after live runs. Link to `.claude/CLAUDE.md`.
  - Done when: the user has confirmed every episode is accurate.
- [ ] **S3-18 Final README and clean-clone test** (Both).
  - Work: the README covers the agent, its tools, how to run the agent and the MCP server, why MCP, memory and feedback, and the measured numbers.
  - Done when: a clone of `develop` from GitHub runs `make run` to exit 0.
- [ ] **S3-19 Submit** (Me).
  - Done when: the user has sent the repo link, and `main` = `develop` = tag `stage-3-submitted`.

## Stretch (no points; only after S3-19 is safe)
- [ ] Agent eval in CI with a threshold (reuse S2-20)
- [ ] Tracing in Langfuse or LangSmith
- [ ] Several agents with roles, A2A
