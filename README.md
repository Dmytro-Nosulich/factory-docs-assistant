# factory-docs-assistant

An AI assistant for industrial documentation: equipment manuals, safety data sheets, certificates and technical datasheets. It is built in stages: structured extraction to validated JSON (with a cloud vs local model comparison), then cited question answering (RAG), an agent with tools, and a production-style service.

Real factory documentation is closed, so this project uses only **public, openly available documents** as stand-ins. Each document's source URL and terms are listed in `data/manifest.csv`.

> Work in progress: stage 1 (structured extraction).

## Requirements
- [uv](https://docs.astral.sh/uv/) (installs the right Python version, 3.11, automatically)
- `make`
- [Ollama](https://ollama.com) running, with the local model pulled: `ollama pull qwen3.5:9b` (6.6 GB)
- An [OpenRouter](https://openrouter.ai) key in `.env` for the two cloud models (paid, see below)

## How to run
```sh
cp .env.example .env   # then fill in OPENROUTER_API_KEY
make run               # everything: every model x prompt x sampling on all 35 documents, then REPORT.md
make run DOCS="D001 D007"   # the same on two documents: a quick check (the REPORT.md tables keep full runs only)
make extract-one DOC=D001   # extract one document with the cloud model (one paid call)
make repair RUN=<run_id>    # one repair attempt for each invalid reply of a saved run
make metrics           # metrics table of all saved runs (rules: docs/METRICS.md)
make comparison        # rewrite the comparison tables in REPORT.md from saved runs (no model calls)
make test              # run tests
make lint              # lint and format check
```

### The full run
`make run` runs every configuration in `[comparison]` in `config.toml`: Haiku 5.5 with prompts v1, v2 and v3, then Haiku 4.5 and the local model, each with every prompt and each sampling configuration (`default`, `greedy`). That makes 15 runs. Each run sends every stage 1 document (the 35 safety data sheets with `stage` 1 in `data/manifest.csv`), one at a time. Before the first call it checks that the key is set, Ollama answers and the model is pulled. At the end it rewrites the comparison tables in `REPORT.md`, using the newest complete run of each configuration. Measured from the `run.json` files of the 15 runs in `REPORT.md`, a full run takes about 2 h 15 min on an Apple M4 Pro and costs about $3.62:
- each Haiku 5.5 run about 1.5 min and $0.07;
- each Haiku 4.5 run about 2.5 min and $0.55-0.58 (6 runs on 2026-10-28, 14.5 min and $3.41 in all; it costs 10x Haiku 5.5 per token);
- each local run about 19-20 min and free.

The 15 runs were made in two parts, so the total is the sum of the parts, not one timed run: the 9 Haiku 5.5 and local runs on 2026-10-27 (2 h 0 min, $0.21), then the 6 Haiku 4.5 runs added on 2026-10-28. On a laptop without a GPU the local runs take longer; that was not measured (`docs/ISSUES.md` I-004).

Each run writes `results/runs/<run_id>/`: one `<doc_id>.json` per document with status `valid`, `invalid` (reply not schema-valid) or `error` (call failed), and `run.json` with counts, cost, truncated documents and every failure. An invalid reply gets one repair attempt: the model sees its reply and the validation errors (`prompts/repair_v1.md`). The result is stored next to the first reply as `repaired` or `failed`, so first-response failures stay visible. Failed documents are saved, not hidden or retried. Every result file also records the configuration that produced it: the model (for a local model, also the Ollama digest of the exact build), the prompt version and file hash, every parameter sent (the sampling values that were really sent; `null` for Haiku 5.5, which takes none) and the input cap.

Budget guard: before each call, the cost recorded in all of `results/` is checked against `budget.stop_at_usd` in `config.toml` ($10; the committed results already account for about $3.83 of it); at that point the run stops and lists the remaining documents as `not_run`. Continue a stopped or interrupted run with `uv run python -m factory_docs --resume <run_id>`; it only fills in missing documents. Then `make comparison` rebuilds the tables.

### One configuration
`MODEL`, `SAMPLING` or `PROMPT` runs one configuration instead of all of them (without `MODEL`: the cloud model). Local models run through [Ollama](https://ollama.com) and are sized for an ordinary laptop (32 GB RAM): 4-12B parameters. Pass a model key from `config.toml`:
```sh
make run MODEL=haiku-5.5                        # the cloud model, all 35 documents
make run MODEL=qwen3.5-9b                       # all 35 documents, no API cost
make run MODEL=qwen3.5-9b DOCS="D001 D007"      # a subset
make run MODEL=qwen3.5-9b SAMPLING=greedy       # another sampling configuration
make run MODEL=qwen3.5-9b PROMPT=v2             # another prompt version (prompts/CHANGELOG.md)
```
Sampling (`temperature`, `top_p`) comes from the named `[sampling.*]` tables in `config.toml`: `default` (1.0 / 0.95, the model's own defaults) and `greedy` (0.0 / 1.0). Without `SAMPLING` a model uses the table named in its `sampling` key. The cloud models: Haiku 5.5 takes no sampling parameters, and Haiku 4.5 gets only the `temperature` of the table, because Claude 4+ models reject `temperature` and `top_p` in one request (`docs/ISSUES.md` I-003). `num_ctx` (the context window) is set explicitly per model. A prompt that does not fit is recorded as `error` and never as a reply, because some Ollama runners cut long prompts silently (`docs/ISSUES.md` I-004).

## Configuration
`config.toml` holds everything that changes a run: prompt version, model slug, sampling, timeout, retries and the input length cap. Keys are never in it; `OPENROUTER_API_KEY` comes from `.env`.

## Layout
- `src/factory_docs/`: application code
- `data/raw/`: source PDFs; `data/manifest.csv`: document list with source and terms
- `prompts/`: versioned prompts
- `results/runs/`: one folder per run, with every model output and its configuration
- `REPORT.md`: the comparison tables (generated) and the analysis
- `scripts/`: helper scripts
- `tests/`: tests
