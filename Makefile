.PHONY: run run-rag model-info tool-smoke samples cost eval judge eval-gate answer-errors report-summary calibration-answers calibrate-judge run-extraction repair texts parse clean-corpus ingest compare-embeddings compare-rerankers index search ask retrieval-gold-check dev-questions-check test-set-check eval-retrieval eval-refusal section-labels golden-check extract-one metrics comparison errors test lint format

# Stage 3 one-command run: the agent pipeline, added in S3-15. Until then it only points to the
# stage 2 and stage 1 runs (S3-01).
run:
	@echo "make run is the stage 3 agent pipeline (added in S3-15). Stage 2: make run-rag. Stage 1: make run-extraction." >&2
	@exit 1

# Stage 2 one-command run (S2-15; was `make run` until S3-01): checks the key and budget, rebuilds
# the corpus files (no diff: every step is deterministic), the index only when stale, runs the
# checks and evals, then answers the sample questions and prints the cost per 1000 questions; it
# fails at the end when an eval is below its [eval_gate] threshold (S2-20; CI runs this target).
# Each step is a target of its own below. Stage 1: make run-extraction
run-rag:
	uv run python scripts/preflight.py
	$(MAKE) parse clean-corpus ingest
	uv run python scripts/build_index.py --if-stale
	$(MAKE) retrieval-gold-check dev-questions-check test-set-check section-labels
	$(MAKE) eval-retrieval eval-refusal eval
	$(MAKE) samples
	$(MAKE) cost
	$(MAKE) report-summary
	$(MAKE) eval-gate

# Tool support, context length and price of an OpenRouter model on GET /api/v1/models; free (S3-02)
model-info:
	uv run python scripts/model_info.py --model $(or $(MODEL),haiku-5.5)

# Live tool-calling smoke test: one tool call, then a text answer; 2 paid calls (S3-02)
tool-smoke:
	uv run python scripts/tool_smoke.py --model $(or $(MODEL),haiku-5.5)

# Answer the [run] sample_questions of config.toml like `make ask` and print them (S2-15)
samples:
	uv run python scripts/sample_answers.py

# Cost per 1000 questions from evals/results/refusal.json, no model calls; writes
# evals/results/cost.md and the cost section of docs/reports/stage-2.md (S2-15)
cost:
	uv run python scripts/cost_report.py
	$(MAKE) report-summary

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

# Answer a question with checked citations: make ask Q="..." [COLLECTION=] [TYPE=] [DOC_LANG=]
#   [SINCE=] [UNTIL=] [LATEST=1]. Search, budget and model come from [search] and [answer] in
# config.toml; the record goes to results/answers/ (gitignored) (S2-13)
ask:
	uv run python scripts/ask.py "$(Q)" $(if $(filter-out structure,$(COLLECTION)),--collection $(COLLECTION)) \
		$(if $(TYPE),--doc-type $(TYPE)) $(if $(DOC_LANG),--language $(DOC_LANG)) \
		$(if $(SINCE),--since $(SINCE)) $(if $(UNTIL),--until $(UNTIL)) \
		$(if $(filter 1,$(LATEST)),--latest-only)

# Check evals/retrieval_gold.toml: every quote is in its document, on its page, and in a chunk
# of every collection (S2-09)
retrieval-gold-check:
	uv run python scripts/check_retrieval_gold.py

# Check evals/dev_questions.toml: kinds and evidence match, no gold-set question repeated, every
# quote in its document, on its page, and in a chunk of every collection (S2-14)
dev-questions-check:
	uv run python scripts/check_dev_questions.py

# Check evals/test_set.toml: fields match each category, no dev or gold question repeated, at
# least 20 questions in all four categories, every label grounded in the corpus (S2-16)
test-set-check:
	uv run python scripts/check_test_set.py

# Recall@k and MRR of every setup in [retrieval_eval] on both chunkings, for the retrieval gold
# set; writes evals/results/ and the retrieval section of docs/reports/stage-2.md (S2-10)
eval-retrieval:
	uv run python scripts/eval_retrieval.py
	$(MAKE) report-summary

# "I don't know" on evals/dev_questions.toml: every [refusal_eval] prompt version with the score
# gate off, the gate threshold by the rule in docs/METRICS.md; writes evals/results/refusal.*
# and the refusal section of docs/reports/stage-2.md. Calls the [answer] model, under its budget guard (S2-14)
eval-refusal:
	uv run python scripts/eval_refusal.py

# Answer evals/test_set.toml like `make ask` (score gate on), score the answers with the S2-17
# scorers and judge each one for quality and faithfulness with [judge]; writes a new
# evals/results/runs/<UTC time>.json, evals/results/eval.md and the answer section of docs/reports/stage-2.md.
# Calls the [answer] and [judge] models, under the stage 2 budget guard (S2-18)
eval:
	PYTHONPATH=. uv run python scripts/eval_answers.py
	$(MAKE) report-summary

# Only the judges, on the answers of a saved run: make judge RUN=evals/results/runs/<time>.json
# (a new run file names the run its answers come from; S2-18)
judge:
	PYTHONPATH=. uv run python scripts/eval_answers.py --judge $(RUN)
	$(MAKE) report-summary

# Fail when the retrieval results or the latest test-set run are below a threshold in
# config.toml [eval_gate]; no model calls (S2-20). The latest run must be from the checked-out
# commit; RUN=<run file> checks that run instead: make eval-gate RUN=evals/results/runs/<time>.json
eval-gate:
	PYTHONPATH=. uv run python scripts/eval_gate.py $(if $(RUN),--run $(RUN))

# "Results at a glance" at the top of docs/reports/stage-2.md, copied from the saved results; no model calls
# (S2-21). Runs after every target that rewrites a section it copies from.
report-summary:
	PYTHONPATH=. uv run python scripts/report_summary.py

# Every test-set error of a run, for the error analysis in docs/reports/stage-2.md; no model calls, no files
# written (S2-21). RUN=<run file> (default: the latest), GATE_OFF=<answers> (default: the S2-19
# calibration answers, the same questions with the score gate off)
GATE_OFF ?=
answer-errors:
	PYTHONPATH=. uv run python scripts/answer_errors.py $(if $(RUN),--run $(RUN)) \
		$(if $(GATE_OFF),--gate-off $(GATE_OFF))

# Judge calibration, step 1 (S2-19): answer the test set like `make ask` with the score gate off
# and save the answers with no judge call; writes evals/results/calibration/answers.json, the
# blind grading sheet evals/results/calibration/sheet.md and evals/judge_grades.toml to fill in.
# Calls the [answer] model, under the stage 2 budget guard. Refuses to overwrite saved answers.
calibration-answers:
	PYTHONPATH=. uv run python scripts/calibrate_judge.py --answers

# Judge calibration, step 2 (S2-19): once every answer in evals/judge_grades.toml has a hand
# grade, run the quality judge 3 times per saved answer and report the agreement; writes
# evals/results/calibration.json, calibration.md and the calibration section of docs/reports/stage-2.md.
# `PYTHONPATH=. uv run python scripts/calibrate_judge.py --report` rebuilds them without models.
calibrate-judge:
	PYTHONPATH=. uv run python scripts/calibrate_judge.py
	$(MAKE) report-summary

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
