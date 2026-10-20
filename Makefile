.PHONY: run texts test lint format

run:
	uv run python -m factory_docs

texts:
	uv run python scripts/extract_texts.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .
