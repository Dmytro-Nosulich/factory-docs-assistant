# CLAUDE.md

Read at the start of every session. Keep it short; details live in `docs/`.

## What this project is
A portfolio-grade AI system built over an 8-week AI engineering course. It extracts structured data from open industrial documentation, then grows into RAG, an agent, and a production-style service.
Full context: @docs/PROJECT.md

## Where to look
- Goal, scope, decisions, open questions: @docs/PROJECT.md
- Stage index and key dates: @docs/roadmap/00-overview.md
- Current work: the stage file marked ACTIVE in the overview (today: stage 3). Read only the active stage file. Do not read or start later stages.

## How to work with me
- I am a senior iOS developer with no backend/Python-service experience (one toy FastAPI app only). Use iOS analogies where they help (pydantic ~ Codable, pyproject ~ Package.swift, pytest ~ XCTest, uv ~ SPM).
- Before each task: say what you will do and why (2-4 sentences). For anything non-trivial, propose a short plan and wait for my OK.
- One roadmap task at a time. Stop after each task so I can review the diff. Never commit before my explicit approval: leave the changes uncommitted, show me what changed, and commit only after I say OK. One commit per task, in Conventional Commits format with the task ID as scope: `<type>(<task ID>): <summary>`, e.g. `feat(S1-04): add extraction schema`, `chore(S1-00): repo skeleton`. Types: `feat`, `fix`, `docs`, `test`, `refactor`, `build`, `ci`, `chore`.
- Git branches: commit stage 3 work on `develop`. Do not commit to or merge into `main` until stage 3 is ready to submit and I approve it.
- When a task is done AND verified by running it, tick its checkbox in the stage file and add the date. Never tick a task you did not verify.
- This is a study project. After each finished task, before stopping for my review, print a structured explanation of the task in chat, following the format in `.claude/docs/study-notes/stage-2.md`. Do not save explanations to a file. Focus on the AI engineering: why the step is needed, how the idea works, a real example from our documents, the trade-offs, what to watch out for, what we decided and why, and how to read the results. Keep the technical details to a few lines, because AI writes the code.
- Explain every new dependency or tool before adding it. Prefer the standard library and boring choices.
- If a task is ambiguous or touches an open decision (see PROJECT.md section 7), ask me instead of guessing.

## Hard rules
- Never commit secrets. Keys live in `.env`; only `.env.example` is committed. Never ask me to paste a key into chat.
- The golden set is labeled by me, by hand. Do not label data for me unless I explicitly ask; if I do, record that in the report.
- Never invent metrics or results. Every number in REPORT.md must come from a script in this repo that I can re-run.
- Respect data licenses. Record source URL and license/terms of every document in `data/manifest.csv`. The corpus PDFs in `data/raw/` are committed on purpose (decision 9 in PROJECT.md), so the reviewer can run the app on the same documents.
- Stay in scope. Stage 1 excludes UI, deployment, fine-tuning, a large dataset, and chasing perfect accuracy.

## Conventions (proposed defaults, not course requirements; tell me if you disagree)
- Python 3.11+ managed with uv. Lint/format: ruff. Tests: pytest.
- Cloud models are called through OpenRouter (OpenAI-compatible API); the model slug lives in config. Set a spending limit on the OpenRouter key (course budget: $5; since S1-22 the guard is $10 and the user tops up credit as needed).
- Entry point: `make run`. It must work from a fresh clone after `cp .env.example .env` and filling in keys.
- Course materials (saved homework HTML, rubric screenshot `image.png`) live in `course-materials/Stage<N>/`, one folder per stage, which is gitignored and never committed.
