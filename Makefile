.PHONY: run repair texts golden-check extract-one metrics comparison test lint format

MODEL ?= haiku-5.5
SAMPLING ?=
DOCS ?=
PROMPT ?=
SAMPLING_ARG = $(if $(SAMPLING),--sampling $(SAMPLING))
PROMPT_ARG = $(if $(PROMPT),--prompt-version $(PROMPT))
run:
	uv run python -m factory_docs --model $(MODEL) $(SAMPLING_ARG) $(PROMPT_ARG) \
		$(if $(DOCS),--docs $(DOCS))

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
	uv run python scripts/extract_one.py --doc $(DOC) --model $(MODEL) $(SAMPLING_ARG) $(PROMPT_ARG)

RUNS ?=
metrics:
	uv run python scripts/metrics.py $(RUNS)

# Model x sampling x prompt version (S1-12, S1-14): Haiku 5.5, qwen3.5-9b default and greedy,
# each with prompt v1, v2 and v3
COMPARISON_RUNS = 20261021T213726Z_haiku-5.5_v1 20261027T082437Z_haiku-5.5_v2 \
	20261027T092300Z_haiku-5.5_v3 \
	20261021T213849Z_qwen3.5-9b_default_v1 20261027T082633Z_qwen3.5-9b_default_v2 \
	20261027T092438Z_qwen3.5-9b_default_v3 \
	20261021T215749Z_qwen3.5-9b_greedy_v1 20261027T084610Z_qwen3.5-9b_greedy_v2 \
	20261027T094416Z_qwen3.5-9b_greedy_v3
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
