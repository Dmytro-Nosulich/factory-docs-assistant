# How the metrics are measured

This file defines every number that `make metrics` prints and that `make comparison` writes into `docs/reports/stage-1.md` (code: `src/factory_docs/metrics.py`, tests: `tests/test_metrics.py`). Rules agreed 2026-10-21. When a rule changes, change it here, in the code and in the tests together.

The script reads only saved results in `results/runs/<run_id>/`, so every number can be re-computed without calling a model. The table has one row per run. A run is one model with one prompt version and one sampling configuration. `docs/reports/stage-1.md` shows one run per configuration in `[comparison]` in `config.toml`: the newest one with every stage 1 document, the current prompt file and the current parameters (`src/factory_docs/report.py`).

## Field accuracy on the golden set
Model outputs are compared with the hand-labeled files in `data/golden/` (10 documents). Each document has 9 fields (`src/factory_docs/schema.py`). Each field either matches or does not.

| Field | Match rule | Why |
|---|---|---|
| `language`, `signal_word`, `physical_state`, `un_number`, `revision_date` | exact | Enums, codes and dates: there is one right value, and the schema already fixes the format. |
| `product_name` | normalized exact (below) | Case, punctuation and spacing differ between PDF text and model output without changing the name. |
| `flash_point` | same unit; `min` and `max` each both null or within ±0.5 °C | Absorbs rounding when a model converts °F to °C. |
| `components` | the lists hold the same items, order ignored; item = normalized `name` + exact `cas_number` + `concentration` (as a range, ±0.01 %) | Order in the section 3 table carries no meaning; CAS is an identifier, so exact. |
| `hazard_statements` | the lists hold the same items, order ignored; item = exact `code` + normalized `text` | Both the code and its wording are checked (the user's choice). Codes are case-sensitive: `H360Fd` and `H360FD` mean different things. |

- **Normalized exact:** both strings are normalized the same way and then compared with `==`. Normalization applies Unicode NFKC, lowercases (`casefold`), replaces every punctuation and symbol character (`,` `.` `-` `®` `(` ...) with a space, and collapses runs of whitespace. This is the "exact match" score used by most extraction benchmarks (e.g. SQuAD). It is strict about content and ignores formatting. There is no fuzzy similarity, so there is no threshold to defend.
- **null:** a null golden value and a null output count as a match. Two empty lists count as a match.
- **Failed documents:** an `invalid` or `error` result has no usable output and misses all 9 fields.
- **Field acc.** = matched fields / (9 × golden documents in the run). **Field acc. (valid only)** uses only the golden documents with a valid output. It shows output quality separately from failures.
- **Lists, partial credit:** the headline counts a list field as right only if every item matches one-to-one. The per-field table adds item-level **F1** = 2 × matched / (predicted items + golden items), summed over all golden documents of the run. `hazard_codes F1` matches on the code only; it is a diagnostic that separates wrong codes from differing wording.

### Worked example
The test `test_score_document_matches_hand_count` scores this invented pair by hand:

| Field | Golden | Output | Match |
|---|---|---|---|
| product_name | `Example Degreaser 200` | `EXAMPLE  degreaser-200` | yes (both normalize to `example degreaser 200`) |
| signal_word | `danger` | `warning` | **no** |
| components | ethanol, propan-2-ol | propan-2-ol, ethanol | yes (order ignored) |
| hazard_statements | H225 "…vapour.", H319 "Causes serious eye irritation." | H319 "Causes eye irritation.", H225 "…vapour" | **no**: H225 matches (the `.` is dropped), H319 text differs; F1 = 2×1/(2+2) = 0.5, codes-only F1 = 1.0 |
| flash_point | 12 °C | 12.4 °C | yes (within 0.5) |
| language, physical_state, un_number, revision_date | | equal | yes |

Result: 7 of 9 fields.

## Valid responses
- **Valid (1st):** results with status `valid` (the first reply parsed and passed the schema) / all documents in the run. `invalid` and `error` both count as not valid.
- **Valid after repair:** results with `final_status` `valid` or `repaired` / all documents in the run. An `invalid` first reply gets one repair attempt (`prompts/repair_v1.md`); it is stored next to the first reply, never over it, so this column cannot hide first-response failures. A run with results from before the repair step (no `final_status`) shows `-`; `make repair RUN=<run_id>` adds the step to a saved run.
- **Field acc. (after repair):** the same as Field acc., but a `repaired` document is scored on its repaired output. A schema-valid repair is not a correct one: on D001 (qwen default, v3) the repair set the concentration to null to pass the schema, while the label says 100 %.

## Latency
- `call.latency_s` of every result that has a call (`valid` and `invalid`; an `error` has none). It is the time of the successful attempt and excludes retry waits.
- First calls only: a repair call is not in the latency numbers, so they stay step 2 (first-response) numbers. Its time is in `repair.call.latency_s`.
- p50 and p95 by **nearest rank**: sort the n values; the p-th percentile is the value at position ⌈p/100 × n⌉. It is always a real observed time. It is always shown with n; with n = 3, p95 is simply the slowest call.
- Cloud latency includes OpenRouter's routing and depends on which upstream provider served the call (recorded in each result).
- **Cached:** the number of local calls where Ollama took at least half of the prompt from its cache (`details.prompt_cached_tokens >= 0.5 × prompt_tokens`). A cached call is much faster than a cold one (D001: 23 s cold, 3 s cached), so a run with cached calls understates latency. Consecutive documents share the system prompt (about 114 tokens, seen on D033 in the local-model bake-off), and Ollama reuses that prefix; that saves almost no time, so it does not count (changed on 2026-10-21; the earlier rule was any cached token).
- Local timings are from an Apple M4 Pro with Ollama's MLX runner, not a CPU-only laptop (`docs/ISSUES.md` I-004).

## Cost per 1000 documents
- **$ / 1000 docs** = 1000 × mean over the run's documents with a reply of (prompt tokens × input price + completion tokens × output price) / 1,000,000, summed over the document's first call and its repair call, if any (both are paid). Token counts are the provider's own usage numbers in each result. Prices are per 1M tokens in `config.toml` (`price_in_usd_per_mtok`, `price_out_usd_per_mtok`, with the date checked in `price_checked`). Haiku 5.5: $0.10 / $0.50, checked 2026-10-21. Haiku 4.5: $1.00 / $5.00, checked 2026-10-28.
- **$ / 1000 (reported)** = the same mean from the cost that OpenRouter reports per call (first call plus repair call). It is a cross-check: on the first full run both give $1.96.
- `invalid` replies are included, since they were paid for.
- **Local models:** no API cost (`0 (local)`). Hardware, electricity and time are not counted. The report discusses them separately.

## Per language
Added 2026-10-27. Each language from `data/manifest.csv` gets one column, the languages with the most stage 1 documents first. A cell is two numbers, both on the first response:
- **Valid:** `valid` results / documents in that language (all 35 SDS count).
- **Fields:** matched fields / (9 × golden documents in that language), with the same rules as Field acc. `-` where the language has no golden document (es, fr, nl, it).

The counts are small (the golden set has 4 uk, 2 en, 2 de, 1 sv and 1 pl document), so one wrong field moves a language a lot. The table shows where errors cluster; it does not rank languages.

## Retrieval (stage 2)
Added 2026-11-21 (S2-10). `make eval-retrieval` (code: `src/factory_docs/retrieval_eval.py`, `scripts/eval_retrieval.py`; tests: `tests/test_retrieval_eval.py`) runs every setup in `[retrieval_eval]` in `config.toml` on both chunking strategies, for the questions in `evals/retrieval_gold.toml`. Results: `evals/results/retrieval.json` (every rank and returned chunk ID) and `evals/results/retrieval.md`, plus the generated section of `REPORT.md`. `--report` rebuilds both from the JSON without models.

- **Setups:** `vector`, `bm25`, `hybrid` (RRF of the two), and `hybrid+rerank` (the cross-encoder rescores the first `[reranking] candidates` hybrid hits and keeps the best). No filters. Every search setting comes from `config.toml` unchanged.
- **Hit:** a returned chunk from the evidence document that contains the evidence quote, whitespace ignored (`is_hit`, S2-09). When a question has several pieces of evidence, any one counts. A chunk with the same quote from another document is not a hit.
- **Rank** of a question: the 1-based position of the first hit among the top 10 chunks (10 = the largest k). No hit in the top 10 is a miss.
- **Recall@k** (k = 1, 3, 5, 10): questions with rank ≤ k / all questions. Each question has one answer, so this is the hit rate. Shown with the count (`13/15`), because with 15 questions one question moves it by 0.067.
- **MRR@10** (mean reciprocal rank): the mean of 1 / rank, with a miss counting 0. Rank 1 gives 1, rank 2 gives 0.5, rank 10 gives 0.1. It rewards putting the answer near the top, which matters because the answer prompt gets only a few chunks.
- **Per language:** recall@5 and MRR@10 per question language. 1 to 6 questions per language, so this shows where misses cluster, not a ranking of languages.
- **Timing:** median and slowest seconds per query, including the query embedding. Machine-dependent, so not part of the reproducibility check.
- **Default setup:** the rule was written down before the first run: on the `structure` collection, the highest MRR@10, then the higher recall@5, then the setup without a reranker. This is the only choice made on the gold set; no numeric setting is tuned on it.
- **Reproducibility:** two runs give identical ranks and returned chunk IDs. `test_the_committed_results_follow_from_their_own_rankings` re-derives every rank and summary in the committed JSON from the stored chunk IDs, the chunk files and the gold set.

## Section labels (stage 2)
Added 2026-11-21 (S2-12, decision 19). `make section-labels` (code: `src/factory_docs/section_labels.py`, `scripts/section_labels.py`; tests: `tests/test_section_labels.py`) checks whether a chunk's section label names the section its text comes from, because citations name it. It calls no model, and it writes `evals/results/section_labels.md` and the generated chunking table in `README.md`.

- **Label rule:** a chunk's label (`section_path`) is correct for a block when it is not empty and is a prefix of the block's section path. For a block under `SECTION 5 > 5.3`, `SECTION 5` and `SECTION 5 > 5.3` are correct, and `SECTION 4 > 4.3`, `SECTION 5 > 5.1` and `(no section heading)` are wrong. A parent counts, because a structure chunk that covers 5.1 to 5.3 is labelled with the section all its blocks share.
- **Gold set:** for each piece of evidence in `evals/retrieval_gold.toml`, the answer sections are the section paths of the cleaned blocks that contain the quote. The chunks checked are every chunk of the collection that contains the quote (the `is_hit` rule of S2-09). A chunk is correct when its label names one of the answer sections. A piece of evidence passes when it has at least one such chunk and all of them are correct. A question passes when any of its pieces of evidence passes. Ranking plays no part.
- **Corpus:** every (chunk, block) pair of every indexed document, where the block is one of the blocks the chunk was built from. A fixed chunk includes every block it overlaps, even a few characters of it. Pairs whose block has no section path are skipped and counted separately. Also reported: chunks whose label is correct for every block they hold.
- **Consistency:** before counting, the script rebuilds the chunks from `data/cleaned/` and stops if their texts differ from the chunk files.
