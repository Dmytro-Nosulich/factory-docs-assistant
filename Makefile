.PHONY: run texts golden-check extract-one test lint format

run:
	uv run python -m factory_docs

texts:
	uv run python scripts/extract_texts.py

golden-check:
	uv run python scripts/check_golden.py

DOC ?= D001
MODEL ?= haiku-5.5
extract-one:
	uv run python scripts/extract_one.py --doc $(DOC) --model $(MODEL)

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .
