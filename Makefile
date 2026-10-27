.PHONY: run repair texts golden-check extract-one metrics comparison test lint format

MODEL ?=
SAMPLING ?=
DOCS ?=
PROMPT ?=
SAMPLING_ARG = $(if $(SAMPLING),--sampling $(SAMPLING))
PROMPT_ARG = $(if $(PROMPT),--prompt-version $(PROMPT))
DOCS_ARG = $(if $(DOCS),--docs $(DOCS))
# Plain `make run` (S1-17): every configuration in [comparison] in config.toml, then the
# comparison tables in REPORT.md; DOCS limits the documents. MODEL, SAMPLING or PROMPT: one run.
run:
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

golden-check:
	uv run python scripts/check_golden.py

DOC ?= D001
extract-one:
	uv run python scripts/extract_one.py --doc $(DOC) --model $(or $(MODEL),haiku-5.5) $(SAMPLING_ARG) $(PROMPT_ARG)

RUNS ?=
metrics:
	uv run python scripts/metrics.py $(RUNS)

# Rewrite the comparison tables in REPORT.md from saved runs, no model calls (S1-17)
comparison:
	uv run python scripts/metrics.py --report

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .
