.PHONY: run texts golden-check extract-one metrics comparison test lint format

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

RUNS ?=
metrics:
	uv run python scripts/metrics.py $(RUNS)

# S1-12 model x sampling matrix, prompt v1: Haiku 5.5, qwen3.5-9b default, qwen3.5-9b greedy
COMPARISON_RUNS = 20261021T213726Z_haiku-5.5_v1 20261021T213849Z_qwen3.5-9b_default_v1 \
	20261021T215749Z_qwen3.5-9b_greedy_v1
comparison:
	uv run python scripts/metrics.py --out results/comparison.md $(COMPARISON_RUNS)

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .
