# factory-docs-assistant

An AI assistant for industrial documentation: equipment manuals, safety data sheets, certificates and technical datasheets. It is built in stages: structured extraction to validated JSON (with a cloud vs local model comparison), then cited question answering (RAG), an agent with tools, and a production-style service.

Real factory documentation is closed, so this project uses only **public, openly available documents** as stand-ins. Each document's source URL and terms are listed in `data/manifest_draft.csv`.

> Work in progress: stage 1 (structured extraction).

## Requirements
- [uv](https://docs.astral.sh/uv/) (installs the right Python version, 3.11, automatically)
- `make`

## How to run
```sh
cp .env.example .env   # then fill in OPENROUTER_API_KEY
make run               # run the app
make extract-one DOC=D001   # extract one document with the cloud model (one paid call)
make test              # run tests
make lint              # lint and format check
```

## Configuration
`config.toml` holds everything that changes a run: prompt version, model slug, timeout, retries and the input length cap. Keys are never in it; `OPENROUTER_API_KEY` comes from `.env`.

## Layout
- `src/factory_docs/`: application code
- `data/raw/`: source PDFs; `data/manifest_draft.csv`: document list with source and terms
- `prompts/`: versioned prompts
- `results/`: model outputs and metrics
- `scripts/`: helper scripts
- `tests/`: tests
