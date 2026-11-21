.PHONY: run run-extraction repair texts parse clean-corpus ingest compare-embeddings compare-rerankers index search retrieval-gold-check eval-retrieval section-labels golden-check extract-one metrics comparison errors test lint format

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

# Clean data/parsed/ into data/cleaned/ and report what was removed per document (S2-04)
clean-corpus:
	uv run python scripts/clean_corpus.py

# Chunk data/cleaned/ two ways (structure-aware and fixed-size) into data/chunks/ (S2-05)
ingest:
	uv run python scripts/ingest.py

# Compare the multilingual FastEmbed models in config.toml on size, speed and truncation (S2-06)
compare-embeddings:
	uv run python scripts/compare_embeddings.py

# Compare the multilingual FastEmbed cross-encoders in config.toml on size and speed (S2-08)
compare-rerankers:
	uv run python scripts/compare_rerankers.py

# Embed the chunk files and rebuild the Qdrant index in data/index/qdrant/ (S2-06)
index:
	uv run python scripts/build_index.py

# Search: make search Q="..." [COLLECTION=structure|fixed] [K=5] [MODE=vector|bm25|hybrid]
#   [RERANK=1|0] [TYPE=sds] [DOC_LANG=uk] [SINCE=2024-01-01] [UNTIL=2025-12-31] [LATEST=1]
# Without MODE or RERANK, [search] in config.toml applies (hybrid, reranked since S2-10). TYPE and
# DOC_LANG take a comma-separated list. DOC_LANG, because LANG is the shell's locale variable.
Q ?=
COLLECTION ?= structure
K ?= 5
MODE ?=
RERANK ?=
TYPE ?=
DOC_LANG ?=
SINCE ?=
UNTIL ?=
LATEST ?=
search:
	uv run python scripts/search.py "$(Q)" --collection $(COLLECTION) --k $(K) $(if $(MODE),--mode $(MODE)) \
		$(if $(filter 1,$(RERANK)),--rerank) $(if $(filter 0,$(RERANK)),--no-rerank) \
		$(if $(TYPE),--doc-type $(TYPE)) $(if $(DOC_LANG),--language $(DOC_LANG)) \
		$(if $(SINCE),--since $(SINCE)) $(if $(UNTIL),--until $(UNTIL)) \
		$(if $(filter 1,$(LATEST)),--latest-only)

# Check evals/retrieval_gold.toml: every quote is in its document, on its page, and in a chunk
# of every collection (S2-09)
retrieval-gold-check:
	uv run python scripts/check_retrieval_gold.py

# Recall@k and MRR of every setup in [retrieval_eval] on both chunkings, for the retrieval gold
# set; writes evals/results/ and the retrieval section of REPORT.md (S2-10)
eval-retrieval:
	uv run python scripts/eval_retrieval.py

# Do chunk section labels name the section of their text? Gold set and whole corpus, no models;
# writes evals/results/section_labels.md and the chunking table in README.md (S2-12)
section-labels:
	uv run python scripts/section_labels.py

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
