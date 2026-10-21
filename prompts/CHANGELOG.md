# Prompt changelog

Every prompt version is a file `prompts/<name>_<version>.md`, loaded by
`factory_docs.prompts.load_prompt`. A file is never edited after it has been measured: a change
is a new version. Each entry states what changed, why, and the measured effect (metrics before
and after, from the metrics script).

## extract v1 (2026-10-21)
- What: baseline. Generic rules (use only the document, null when not stated, keep the
  document's language, JSON only) plus the full JSON schema with its field descriptions in the
  user message.
- Why: starting point. The schema goes into the prompt so the cloud and the local model see the
  same field rules; Ollama's `format` only constrains the output and cloud providers handle
  `response_format` differently.
- Measured effect: baseline; metrics come in S1-11/S1-12.
