# factory-docs-assistant

An AI assistant for industrial documentation: equipment manuals, safety data sheets, certificates and technical datasheets. It is built in stages: structured extraction to validated JSON (with a cloud vs local model comparison), then cited question answering (RAG), an agent with tools, and a production-style service.

Real factory documentation is closed, so this project uses only **public, openly available documents** as stand-ins. Each document's source URL and terms are listed in `data/manifest.csv`.

> Work in progress: stage 1 (structured extraction).

## Requirements
- [uv](https://docs.astral.sh/uv/) (installs the right Python version, 3.11, automatically)
- `make`

## How to run
```sh
cp .env.example .env   # then fill in OPENROUTER_API_KEY
make run               # extract all 35 stage 1 documents with the cloud model (paid, see below)
make extract-one DOC=D001   # extract one document with the cloud model (one paid call)
make metrics           # metrics table of all saved runs (rules: docs/METRICS.md)
make test              # run tests
make lint              # lint and format check
```

### The full run
`make run` sends every stage 1 document (the 35 safety data sheets with `stage` 1 in `data/manifest.csv`) to the cloud model, one at a time. One run with `anthropic/claude-haiku-5.5` cost $0.07 and took 90 seconds (first run, 2026-10-21, `results/runs/20261021T140902Z_haiku-5.5_v1/run.json`). It writes `results/runs/<run_id>/`: one `<doc_id>.json` per document with status `valid`, `invalid` (reply not schema-valid) or `error` (call failed), and `run.json` with counts, cost, truncated documents and every failure. Failed documents are saved, not hidden or retried.

Budget guard: before each call, the cost recorded in all of `results/` is checked against `budget.stop_at_usd` in `config.toml` ($4); at that point the run stops and lists the remaining documents as `not_run`. Continue a stopped or interrupted run with `uv run python -m factory_docs --resume <run_id>`; it only fills in missing documents.

### Local model (Ollama)
The same runner works with local models through [Ollama](https://ollama.com). They are sized for an ordinary laptop (32 GB RAM): 4-12B parameters. Install Ollama, pull a model and pass its key from `config.toml`:
```sh
ollama pull qwen3.5:9b
make run MODEL=qwen3.5-9b                       # all 35 documents, no API cost
make run MODEL=qwen3.5-9b DOCS="D001 D007"      # a subset
make run MODEL=qwen3.5-9b SAMPLING=greedy       # another sampling configuration
```
Sampling (`temperature`, `top_p`) comes from the named `[sampling.*]` tables in `config.toml`: `default` (1.0 / 0.95, the model's own defaults) and `greedy` (0.0 / 1.0). Without `SAMPLING` a model uses the table named in its `sampling` key. The cloud model takes no sampling parameters (`docs/ISSUES.md` I-003). `num_ctx` (the context window) is set explicitly per model. A prompt that does not fit is recorded as `error` and never as a reply, because some Ollama runners cut long prompts silently (`docs/ISSUES.md` I-004).

## Configuration
`config.toml` holds everything that changes a run: prompt version, model slug, sampling, timeout, retries and the input length cap. Keys are never in it; `OPENROUTER_API_KEY` comes from `.env`.

## Layout
- `src/factory_docs/`: application code
- `data/raw/`: source PDFs; `data/manifest.csv`: document list with source and terms
- `prompts/`: versioned prompts
- `results/`: model outputs and metrics
- `scripts/`: helper scripts
- `tests/`: tests
