.PHONY: run texts golden-check extract-one test lint format

MODEL ?= haiku-5.5
SAMPLING ?=
DOCS ?=
SAMPLING_ARG = $(if $(SAMPLING),--sampling $(SAMPLING))
run:
	uv run python -m factory_docs --model $(MODEL) $(SAMPLING_ARG) $(if $(DOCS),--docs $(DOCS))

texts:
	uv run python scripts/extract_texts.py

golden-check:
	uv run python scripts/check_golden.py

DOC ?= D001
extract-one:
	uv run python scripts/extract_one.py --doc $(DOC) --model $(MODEL) $(SAMPLING_ARG)

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .
