# Issues and problems log

Problems met while building the project, what we considered, and what we chose. Newest first.
Each entry: date, task, problem, options considered, current choice and why, status.

## I-008: BM25 without stemming, and repeated common terms
- **Date / task:** 2026-11-20, S2-07 hybrid search
- **Problem:** The BM25 tokenizer is the same for all nine languages: no stemming and no stopwords (`src/factory_docs/lexical.py`). Two effects follow.
  - Inflected forms and compound parts do not match. Ukrainian `очі` does not match `очей`, and German `Punkt` does not match `Flammpunkt`.
  - A query word that is common but repeated many times in a short chunk can outscore one rare word. For `CAS 2634-33-5` the BM25 top 5 holds the 3 D060 chunks that contain the number, then 2 D064 table chunks that say `CAS` 8-9 times but do not contain it. The two D007 chunks that contain the number (and never say `CAS`) are at ranks 12 and below. For the bare query `2634-33-5`, all 5 chunks that contain it are the BM25 top 5. Seen with `make search MODE=bm25`; one query, not a measurement.
- **Options considered:**
  - A stemmer per language (Snowball, as in FastEmbed's Bm25): one more dependency, and each chunk would need its language's stemmer.
  - A lower k1, so repetitions count less.
  - Character n-grams, which match word parts in any language but give a much larger index.
  - Accept and record it: vector search covers inflections and paraphrases, and RRF merges both.
- **Current selection:** accept and record it, with the usual defaults k1 1.2 and b 0.75 untuned. S2-10 measures BM25 and hybrid recall, and tuning must not use the test questions.
- **Revisit if:** S2-10 shows BM25 misses on inflected uk/de/pl queries, or exact-code queries lose to repeated common terms.
- **Status:** accepted.

## I-007: Flat heading levels leave tiny structure chunks
- **Date / task:** 2026-11-19, S2-05 chunking
- **Problem:** The structure-aware chunker never puts two top-level sections in one chunk. In some documents the parser gives every heading the same level (D038, D029), or the first headings have no parent (D006, level 4 before any level 2). There, every heading counts as a top-level section, so small subsections such as `10.3 Möglichkeit gefährlicher Reaktionen : Keine Angaben` (D038) become chunks of their own. 89 of 2433 structure chunks (4 %) are under 100 characters (`data/chunks/chunking_comparison.md`). The front matter before the first heading is also a chunk of its own, such as `DATE: 22 March 2024` in D011.
- **Options considered:**
  - Let short chunks join a neighbour across a top-level section. This mixes SDS sections, and the chunk could then cite only `(no section heading)`.
  - Infer heading levels from numbering (`10` above `10.1`). This is a real fix, but it changes S2-03 output.
  - Prepend the document title and section path to each chunk before embedding (contextual chunk headers, a stretch item). A short chunk then still carries its context.
  - Accept and record it.
- **Current selection:** accept and record it. Short chunks are precise, and their section metadata is correct.
- **Seen in S2-06 (2026-11-19):** for the German smoke query `Wie hoch ist der Flammpunkt von RM 110?`, the 70-character chunk `D033-s-0001` (`1.1 Produktidentifikator`, only the trade name) is the top hit of e5-large, embeddinggemma and qwen3-0.6b-q. The chunk that states the flash point, `D033-s-0012` (section 9), is in no model's top 5, because it does not name the product (`data/index/embedding_comparison.md`). Vector search finds the product, not the property. Hybrid search (S2-07), reranking (S2-08) or contextual chunk headers may fix it; S2-10 measures it.
- **Seen in S2-07 (2026-11-20):** with hybrid search (`make search`, default mode), `D033-s-0012` is rank 2 for the same query (vector rank 18, BM25 rank 44), after `D033-s-0001`. It rises because it is in both candidate lists, not because BM25 ranks it high: it does not contain `RM 110`. One query, not a measurement; S2-10 measures it.
- **Seen in S2-08 (2026-11-20):** with reranking (`make search RERANK=1`), `D033-s-0012` is rank 1 for the same query, from hybrid rank 2, with both bge-reranker-v2-m3 builds (`data/index/reranker_comparison.md`). The cross-encoder reads `Flammpunkt` in the query and in the chunk together. One query, not a measurement; S2-10 measures it.
- **Revisit if:** S2-10 shows misses on short chunks, or if contextual chunk headers are added.
- **Status:** accepted.

## I-006: Known limits of the document cleaning rules
- **Date / task:** 2026-11-18, S2-04 cleaning and deduplication
- **Problem:** The cleaning rules leave a few known defects in `data/cleaned/`:
  - D010 and D038 have lines where the PDF parser mixed two columns. About 9 line-end hyphen repairs therefore involve unrelated word parts, such as `наслід-` + `пеки` in D010.
  - One word split across a page break (D010, pages 11-12) is not rejoined.
  - 18 `word- word` cases in lists and tables stay as they are. Most are real forms such as `Montage- und`, and a few are wrapped words such as `pre- approved`.
  - D026 keeps its repeated company address, because the line is at the page edge on only 2 of its 3 pages.
  - D063 page 2 had a `U+0001` character before `0,1%`. Its meaning cannot be recovered, so it is deleted and listed in `data/cleaned/removed.csv`.
- **Options considered:**
  - Fix the column mixing in the PDF parser. This is the real fix for D010 and D038, but it changes S2-03 output for every PDF.
  - Add document-specific exceptions. This hides the defects and does not carry over to new documents.
  - Accept and record the defects.
- **Current selection:** accept and record them. They affect a small share of the 69 documents, and `make clean-corpus` reports every removal.
- **Revisit if:** retrieval errors in S2-10 or the error analysis in S2-21 trace back to D010 or D038.
- **Status:** accepted.

## I-005: The JSON schema allows a concentration with neither min nor max
- **Date / task:** 2026-10-21, local model bake-off
- **Problem:** `ValueRange` requires at least one of `min` and `max` through a Pydantic validator. That rule is not in the JSON schema sent to the models, where both are simply nullable. So constrained decoding (Ollama `format`, OpenRouter `strict`) cannot enforce it. In the bake-off, `gemma4:e4b` and `gemma4:12b` both returned `{"min": null, "max": null}` for a D007 (uk) component, and the reply was marked `invalid`.
- **Options considered:**
  - Express the rule in the JSON schema (e.g. `anyOf` with one bound required). Strict structured output may not support it on every provider; it needs a test.
  - Say it plainly in the prompt (prompt v2): "if no value is stated, set concentration to null, not {min: null, max: null}".
  - A repair attempt with the validation error.
- **Current selection:** none yet. It is a measured first-response failure and stays counted as one. Handle it in prompt v2 and/or with a repair attempt, and record the effect.
- **Repair result (2026-10-27):** the repair attempt (`prompts/repair_v1.md`) made both invalid v3 replies schema-valid (qwen default D001: min/max both null; qwen greedy D037: min 10 > max 5). Both did it by setting the concentration to null, while the documents give 100 % and 1-5 %. So repair fixes validity, not the value; the first-response failures stay counted in "Valid (1st)" (`prompts/CHANGELOG.md`, repair v1).
- **Status:** open.

## I-004: Ollama context window, silent truncation and prompt cache
- **Date / task:** 2026-10-21, local model
- **Problem:** Ollama's FAQ says the context window (`num_ctx`) defaults to 4096 tokens, but our prompts are about 5k-50k tokens. The docs do not say what happens when a prompt does not fit. Many write-ups say Ollama cuts it off silently, so the model would answer from part of the document while the reply still looks valid.
- **What we measured** (Ollama 0.40.7, M4 Pro, `gemma4:e4b` default tag, an `nvfp4` build run by Ollama's MLX runner; probe script not committed):
  - D001 with `num_ctx = 2048`: `prompt_eval_count` was 8826 and the answer was correct. The MLX runner ignored `num_ctx` (`ollama ps` showed CONTEXT 2048) and read the whole prompt. Without `num_ctx`, Ollama 0.40 picks a default from available memory ("vram-based default context", 32768 here), not 4096.
  - The same D001 request sent twice: `prompt_eval_count` stayed 8826 and `prompt_eval_cached_count` was 8822. So `prompt_eval_count` is the full prompt, cached tokens included. The repeat took 1.5 s instead of 15 s, because Ollama reused the cached prompt.
  - On an Intel laptop, Ollama would use its llama.cpp runner with a `q4_K_M` build instead. We could not test truncation there, because no such build is installed; the user chose to keep the default tags.
- **Options considered:**
  - Trust the runner: fine on this Mac, unknown on a laptop.
  - Pre-count tokens before sending: Ollama has no documented tokenize endpoint.
  - Set `num_ctx` explicitly and check every reply: if `prompt_eval_count + num_predict > num_ctx`, the prompt did not fit the configured window, so it may have been cut on another runner.
- **Current selection:** the third option. `num_ctx` is set per model in `config.toml` and logged in every result. The Ollama client turns an overflow into a non-retryable error (status `error`), so a reply from a possibly cut prompt never counts as valid. Cached prompt tokens are recorded in `call.details`, so the metrics can show when latency benefited from the cache (the same document run twice in a row, as in the sampling test).
- **Live check (2026-10-21):**
  - D003, the longest prompt, on `qwen3.5:9b`: 30,627 prompt tokens fit in `num_ctx = 65536`, and the reply was valid. qwen's tokenizer counts about 38% fewer tokens than Claude's for the same prompt (49,477).
  - The same document with `num_ctx` forced to 16,384 was recorded as `error` ("prompt did not fit"), never as valid. Memory in `ollama ps` at the 64k context: 9.1 GB.
- **CPU-only timing not measured:** the MLX runner also ignores `num_gpu = 0` (`ollama ps` still showed 100% GPU, and one document took GPU time). A real CPU-only timing needs a llama.cpp build such as `qwen3.5:9b-q4_K_M` (6.6 GB). The user chose not to download it, so the stage 1 report (`docs/reports/stage-1.md`) gives only M4 Pro GPU latency and says laptop speed was not measured. The `num_gpu` option was removed from the config, because it had no effect here.
- **Revisit if:** a laptop run (llama.cpp runner) reports a `prompt_eval_count` capped at `num_ctx`. Then the check catches it as designed; confirm that and note it here.
- **Status:** decided.

## I-003: The cloud models on OpenRouter do not accept temperature or top_p
- **Date / task:** 2026-10-21, cloud model client
- **Problem:** The assignment asks for sampling parameters (temperature, top_p) in config and a comparison of at least 2 configurations. OpenRouter's model list (`GET /api/v1/models`, checked 2026-10-21) shows that the current mid-tier and small cloud models, including the chosen `anthropic/claude-haiku-5.5` and the earlier candidates `anthropic/claude-sonnet-5.5` and `openai/gpt-6.1-sol`, do not list `temperature` or `top_p` among their supported parameters. With `provider.require_parameters: true` (needed so structured output is honored), a request that sends them would find no provider.
- **Options considered:**
  - Compare sampling configurations on the local model only (Ollama accepts temperature and top_p).
  - Use reasoning effort as the cloud model's configuration axis instead of sampling.
  - Switch to an older cloud model that accepts temperature/top_p (e.g. `anthropic/claude-sonnet-4.6`, $3/$15 per 1M tokens).
- **Current selection:** sampling is compared on the local model only. The cloud model runs with fixed settings from `config.toml` (`reasoning_effort = "none"`), and the client sends only the parameters that are set there.
- **Revisit if:** the reviewer expects a sampling comparison on the cloud model too. Then add a model that accepts temperature as a second cloud entry in `config.toml` (a config change only).
- **Revisited 2026-10-28:** on the teacher's suggestion, `anthropic/claude-haiku-4.5` was added as a second cloud model with the same `default` and `greedy` configurations, on all three prompts. Haiku 5.5 stays as it is.
  - **Why this model:** OpenRouter lists `temperature`, `top_p` and `top_k` for it (`GET /api/v1/models`, 2026-10-28). It costs $1.00 / $5.00 per 1M tokens, 10x Haiku 5.5.
  - **Temperature only:** Anthropic's documentation says Claude 4+ models reject `temperature` and `top_p` in one request. So only `temperature` is sent (`sampling_params = ["temperature"]` in `config.toml`): 1.0 for `default` (also the API default) and 0.0 for `greedy`. The logged params show exactly what was sent.
  - **Smoke test:** D001 with both configurations, schema-valid, served by Amazon Bedrock.
  - **Risk:** on Anthropic API version `2026-10-21`, `temperature`, `top_p` and `top_k` are rejected on every model. If OpenRouter moves Haiku 4.5 to that version, `require_parameters` finds no provider and the calls fail as `error`; they are not silently sent without temperature.
  - It needed a small code change, not only config: a cloud model may now name a sampling table and the values it takes (`OpenRouterModel.sampling`, `sampling_params`).
- **Status:** decided.

## I-002: Stage 1 had 15 texts, below the course minimum of 30 (resolved: 35)
- **Date / task:** 2026-10-20, corpus manifest
- **Problem:** The course teacher advised focusing stage 1 on safety data sheets (SDS), because the other document types (manuals, certificates, declarations, datasheets) are hard to read. The corpus has only 15 SDS (D001, D002, D003, D006, D007, D008, D010, D032-D039; en, sv, pl, uk, de), while the assignment asks for at least 30 texts.
- **Options considered:**
  - Add about 15 more public SDS (same issuers as D006/D032 and D007/D033) to reach 30.
  - Ask the teacher whether fewer than 30 is acceptable for an SDS-only set.
  - Proceed with the 15 SDS we have.
- **Current selection:** proceed with the 15 SDS. The 20 non-SDS documents move to stage 2 (RAG) in `data/manifest.csv`; their texts stay in `data/texts/` for stage 2. The golden set keeps its 4 SDS (D001, D002, D006, D033) and gets 6 SDS replacements.
- **Revisit if:** the reviewer or rubric penalizes fewer than 30 texts, or the metrics on 15 documents are too noisy to compare models. Then add more SDS (only manifest rows and PDFs; the pipeline needs no change).
- **Update 2026-10-20:** the user chose to close the gap. 20 more public SDS were added (D046-D065: 13 English, 3 of them US format under OSHA HCS; 2 French, 2 Dutch, 2 Spanish, 1 Italian), so stage 1 has 35 SDS. Each was checked before adding: real PDF, text layer on every page, all 16 sections present, readable by `make texts` without new packages, under the runner cap of about 20k estimated tokens, and not a product already in the corpus. Rejected candidates: a scrambled text layer (Tytan, IT), AES encryption that would need the `cryptography` package (Novol, ES), a missing last page (older Nyrstar revision), and two documents of about 24k tokens that the runner would truncate. `tests/test_manifest.py` now requires at least 30 stage-1 SDS. The golden set stays at 10.
- **Status:** resolved (2026-10-20).

## I-001: Which PDF-to-text converter to use
- **Date / task:** 2026-10-19, text extraction
- **Problem:** The corpus is PDFs (EN, SV, PL, UK, DE, FR, IT; some scanned pages), but the models and the golden set work from plain text. The converter must keep page boundaries (stage 2 cites pages) and must work from a fresh clone with one command.
- **Options considered:**
  - `pypdf`: pure Python, BSD license, no system packages, page by page. Text quality is adequate; some layout noise.
  - `PyMuPDF`: faster, often cleaner text on complex layouts and tables. AGPL license.
  - `pdfplumber`: MIT license, good at tables, slower (built on pdfminer.six).
  - `pdftotext` (poppler): good quality, but needs a system package, so it breaks the one-command fresh-clone run.
- **Current selection:** `pypdf` (6.21.1). It is a boring, permissive choice with no system dependencies. Extraction noise is acceptable, even useful, because the course asks for "dirty" texts.
- **Observations from the first run:**
  - All 35 stage-1 documents extract in about 8 s, about 180k estimated tokens in total (4 characters per token, which undercounts Cyrillic). The earlier estimate of 0.4M came from layout-mode text and was an upper bound.
  - Typical noise: words split by stray spaces ("Nitroge n", "1907/200 6").
  - Pages without a text layer come out empty: D014 pp. 3-9 (scans), D019 pp. 37-39, D021 p. 47. No OCR in stage 1.
  - D002 (2026-10-20, user's visual check): a table is converted with some loss of its structure. Not critical; accepted as natural noise, and the text file is not edited by hand.
  - D038 (2026-10-20, while labeling): the PDF's column and box layout breaks the reading order. The section 3 table (line 77) comes after section 4 and the hazard-label box, and "Form : Verdichtetes Gas" from section 9 (line 192) comes right after the section 13 disposal text. All values are present, but not inside their sections. Accepted as noise; check it first if D038 components or physical_state fail.
  - Other noise seen while labeling the golden set: split digits in a value (D008 flash point "> 6 5 ° C") and words hyphenated across lines (D010 "наслід- ками"). The golden labels hold the document's real values ("> 65 °C", "наслідками").
  - D001 makes pypdf warn about 200 times that `fontTools` is needed for a font that holds only the space character. The text is unaffected, so the script logs pypdf errors only instead of adding `fontTools`.
- **Revisit if:** extraction errors (not model errors) show up in the error analysis in `docs/reports/stage-1.md`, or stage 2 needs table structure. Then try PyMuPDF or pdfplumber on the failing documents.
- **Status:** decided.
