.PHONY: run run-extraction repair texts parse golden-check extract-one metrics comparison errors test lint format

# Stage 2 entry point; S2-15 replaces this placeholder with the RAG pipeline.
run:
	@echo "The stage 2 pipeline is not built yet (S2-15). For stage 1 use: make run-extraction"
	@exit 1

MODEL ?=
SAMPLING ?=
DOCS ?=
PROMPT ?=
SAMPLING_ARG = $(if $(SAMPLING),--sampling $(SAMPLING))
PROMPT_ARG = $(if $(PROMPT),--prompt-version $(PROMPT))
DOCS_ARG = $(if $(DOCS),--docs $(DOCS))
# Plain `make run-extraction` (S1-17): every configuration in [comparison] in config.toml, then the
# tables in docs/reports/stage-1.md; DOCS limits the documents. MODEL, SAMPLING or PROMPT: one run.
run-extraction:
ifeq ($(MODEL)$(SAMPLING)$(PROMPT),)
	uv run python -m factory_docs --all $(DOCS_ARG)
else
	uv run python -m factory_docs --model $(or $(MODEL),haiku-5.5) $(SAMPLING_ARG) $(PROMPT_ARG) \
		$(DOCS_ARG)
endif

# One repair attempt for each invalid reply of a saved run (S1-15): make repair RUN=<run_id>
RUN ?=
repair:
	uv run python -m factory_docs --repair $(RUN)

texts:
	uv run python scripts/extract_texts.py

# Parse every in-scope document into data/parsed/<id>.json and write parse_report.csv (S2-03)
parse:
	uv run python scripts/parse_corpus.py

golden-check:
	uv run python scripts/check_golden.py

DOC ?= D001
extract-one:
	uv run python scripts/extract_one.py --doc $(DOC) --model $(or $(MODEL),haiku-5.5) $(SAMPLING_ARG) $(PROMPT_ARG)

RUNS ?=
metrics:
	uv run python scripts/metrics.py $(RUNS)

# Rewrite the comparison tables in docs/reports/stage-1.md from saved runs, no model calls (S1-17)
comparison:
	uv run python scripts/metrics.py --report

# Every golden-set miss of the comparison runs, for the stage 1 error analysis (S1-18)
errors:
	uv run python scripts/errors.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .
