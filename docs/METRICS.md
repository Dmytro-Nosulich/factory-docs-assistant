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

## Refusal, "I don't know" (stage 2)
Added 2026-11-21 (S2-14). `make eval-refusal` (code: `src/factory_docs/refusal_eval.py`, `scripts/eval_refusal.py`; tests: `tests/test_refusal_eval.py`) answers every question in `evals/dev_questions.toml` with each prompt version in `[refusal_eval]`, using the search and answer settings `make ask` uses, no filter. Results: `evals/results/refusal.json` (every question's top scores, hits and answers) and `evals/results/refusal.md`, plus the generated section of `REPORT.md`. `--report` rebuilds both from the JSON without models. The model calls go to `results/answers/` and count toward the stage 2 guard.

- **Development questions:** answerable (with evidence, checked by `make dev-questions-check`), `near_miss` (on the corpus's topics, but no document states the fact: another substance, a sibling product, a property no document gives) and `off_topic`. They tune the threshold and the prompt, so the test set (S2-16) must not reuse them.
- **Two signals.** The score gate refuses when the best hit's reranker score (raw bge-reranker-v2-m3 score, the `[answer] min_rerank_score` scale) is below the threshold; a score equal to it passes. A question with no hit is refused too. The model refuses when its shown answer says `not_in_documents`, or when no answer can be shown (rejected after the repair, a call error, no hit).
- **Three decisions**, from one run with the gate off: the model only, the gate only, and both (refused when either refuses). This is how `make ask` decides with a threshold set.
- **Refusal rate:** unanswerable questions refused / all unanswerable questions, also per kind and per language. This is the "Done when" number of S2-14.
- **False-refusal rate:** answerable questions refused / all answerable questions. `partial` counts as answered.
- **Threshold rule** (written before the run): A = the top-1 scores of answerable questions, U = those of unanswerable ones. The threshold is the midpoint between min(A) and the highest score in U below min(A), rounded to 2 decimals (kept exact if rounding would cross either score). If no score in U is below min(A), there is no gate. So the gate refuses no answerable development question by construction; only fresh questions can show its false refusals. The threshold uses retrieval scores only, so it is the same for every prompt version.
- **Language match:** the claims and `missing` of a shown answer are in the question's language, by the heuristic in `src/factory_docs/language.py` (Ukrainian by script, other languages by common function words, `unknown` when unclear). Quotes are left out, since they stay in the chunk's language. Computed from the saved answers, so `--report` applies a changed detector. It is a heuristic: a short answer with no function word counts as `unknown`, so as a mismatch.
- **Right document:** an answered answerable question cites at least one chunk of an evidence document. It does not check the answer's content (S2-17).
- **Cost and latency:** the reported cost of all calls of a version, including repairs, and the nearest-rank median of seconds per question with a call.

## Cost per 1000 questions (stage 2)
Added 2026-11-21 (S2-15). `make cost` (code: `src/factory_docs/cost.py`, `scripts/cost_report.py`; tests: `tests/test_cost.py`) reads `evals/results/refusal.json` and makes no model calls, so it rebuilds the same way on a fresh clone. It writes `evals/results/cost.md` and the generated section of `REPORT.md`. `make run` runs it last.

- **Questions:** the development questions of the last `make eval-refusal` run, answered with `[answer] prompt_version` by `[answer] model` (the run must have both).
- **Gate applied afterwards.** The eval answers every question with the score gate off. Here a question whose top-1 reranker score is below `[answer] min_rerank_score`, or that has no hit, costs $0, because `make ask` makes no model call for it (the same rule as the refusal eval: a score equal to the threshold passes).
- **Cost of a question that reaches the model:** its first call plus its repair call, if any (both are paid). Prompt and completion tokens are the provider's own usage numbers.
- **$ / 1000 (tokens x price)**, the headline = 1000 × mean over the questions of (prompt tokens × input price + completion tokens × output price) / 1,000,000, with the prices per 1M tokens in `config.toml` (`price_in_usd_per_mtok`, `price_out_usd_per_mtok`, date in `price_checked`). The same rule as stage 1's cost per 1000 documents.
- **$ / 1000 (reported)** = the same mean from the cost OpenRouter reports per call. A cross-check.
- **Two rows:** over all development questions (gated ones count as $0), and over only those that reach the model. The first depends on the share of unanswerable questions in the development set, which is high on purpose. The second is the cost of a question the model answers or refuses itself.
- **Not counted:** embedding and reranking run on the local CPU, so they have no API cost. Hardware, electricity and time are not counted.

## Test set (stage 2)
Added 2026-11-23 (S2-16). `make test-set-check` (code: `src/factory_docs/test_set.py`, `scripts/check_test_set.py`; tests: `tests/test_test_set.py`) checks `evals/test_set.toml`. It makes no model calls. The scorers of S2-17 to S2-20 read the same file.

- **Held out.** No test question repeats one of `evals/dev_questions.toml` or `evals/retrieval_gold.toml` (whitespace and case ignored). Nothing is tuned on the test set.
- **Categories:** `answerable`, `similar_doc` (a sibling document holds a similar fact), `outdated_version` (the fact changed between versions) and `unanswerable` (expected status `not_in_documents`). The set needs at least 20 questions and every category.
- **Key facts:** what a correct answer contains. Each fact is a list of accepted spellings, and one spelling is enough. A spelling matches when it is a substring of the text with whitespace collapsed and case ignored (`fact_in`). At least one spelling of every key fact must be in an evidence quote, so every expected fact traces to a source. Other spellings (translations, number formats) are allowed.
- **Evidence:** the gold-set format and checks (document indexed, quote in one block of the cleaned text, on its page for a PDF, whole in a chunk of every collection).
- **Traps:** `wrong_docs` lists the documents whose citation is wrong, and these are never evidence. For `outdated_version`, each wrong document is superseded by an evidence document, directly or through a chain of `supersedes` in `data/manifest.csv`. Optional `forbidden_facts` hold the sibling's or the old version's value. Each must be in the cleaned text of a wrong document, in no evidence quote, and never equal to a key fact.
- **Near misses:** optional `absent_terms` (a product or grade name) must be in no chunk of any collection. This replaces a manual search.

## Answer scoring, deterministic (stage 2)
Added 2026-11-23 (S2-17). Code: `evals/scorers.py`; tests: `tests/test_scorers.py`. These scorers compare one shown answer with one question of `evals/test_set.toml` and call no model. S2-18 runs them on the answers to the test set.

- **Input:** a row `{final_status, answer}`, the same shape as the rows in `evals/results/refusal.json`. The doc ID of a citation is the prefix of its chunk ID (`D033-s-0012` → `D033`).
- **Refused:** the same rule as for refusal: the shown answer says `not_in_documents`, or no answer is shown (rejected after the repair, a call error, no hit, the score gate). `partial` counts as an answer. **Refusal correct:** refused exactly when the question is `unanswerable`. A refused answer is scored with no claims and no citations, even if the row still holds what the model wrote, so it misses every fact and every source.
- **Right source (headline):** at least one citation names a chunk of an evidence document. **Evidence cited (stricter):** at least one cited chunk holds an evidence quote (the gold-set hit rule `is_hit`: same document, quote contained, whitespace ignored). A cited ID missing from the chunk map counts as no hit. Neither is scored for unanswerable questions.
- **Key facts:** a fact is present when one of its spellings is in the claim texts (`fact_in`: substring, whitespace collapsed, case ignored; the S2-16 grounding rule). Quotes and `missing` are not searched, because a quote holds the fact whenever the right chunk is cited. Headline: all key facts present (binary). Detail: facts present out of all key facts.
- **Traps:** **wrong source** = a citation names a chunk of a `wrong_docs` document. It is reported but does not fail the question (user's choice). **Forbidden** = a forbidden fact is in the claim texts. That fails the question.
- **Passed:** an `unanswerable` question is refused. Any other question is not refused, cites an evidence document, and states every key fact. A trap must also state no forbidden fact.
- **Rates** (`counts`, also per category and per language in `summarize`): passed over all questions. Unanswerable refused over unanswerable questions. Answerable refused, right source, evidence cited and all facts over all other questions, traps included (a refusal counts as a miss). Facts present over all their key facts. Wrong source over traps (a refusal cites nothing). Forbidden over traps that have forbidden facts.
- **Known limits:** matching is by substring, so `102` also matches `1020`. An answer that names the sibling's value only to contrast it (`3 mm, not 1 mm as for SRM 613`) counts as forbidden. A correct fact in a spelling the labels do not list counts as missing. The LLM judge (S2-18) is there for such nuance.

## Answer scoring, LLM judge and faithfulness (stage 2)
Added 2026-11-23 (S2-18). `make eval` (code: `evals/judges.py`, `evals/answer_eval.py`, `scripts/eval_answers.py`; tests: `tests/test_judges.py`, `tests/test_eval_answers.py`) answers every question of `evals/test_set.toml` through the `make ask` pipeline (the `[answer]` settings, score gate on, no filter). It saves the answers in a new run file `evals/results/runs/<UTC time>.json` before any judge call, then scores them with the S2-17 scorers and runs both judges on the saved answers. `make judge RUN=<run file>` runs only the judges on a saved run's answers and writes a new run file that names the source run (`answers_from`). It stops if the chunk file or the test set changed since the answers were made. Both write `evals/results/eval.md` and the generated answer section of `REPORT.md`, and `--report` rebuilds them from the latest run file without models. Run files are committed. Answer records go to `results/answers/`, and each judge call goes to `results/judgements/` (gitignored). Both folders count toward the stage 2 guard `[answer] stop_at_usd` (decision 18), and the guard is checked before every call.

- **Judge model:** `[judge] model`, Haiku 5.5, the answer model too (user's choice). It takes no temperature, so a verdict can change between runs. Each judge is a versioned prompt (`prompts/judge_quality_v1.md`, `prompts/judge_faithfulness_v1.md`) and replies in a strict JSON schema, reasoning before the verdict.
- **Quality judge:** it sees the question, the test set's English `expected_answer` and the shown answer: its status, numbered claim texts and `missing`, but no citations or quotes. A question with no shown answer (score gate, rejected after the repair, call error) is shown as a refusal, as S2-17 scores it. Verdicts (user's choice: 3 levels): `correct` (every reference fact, nothing contradicting), `partially_correct` (some reference facts, nothing contradicting), `incorrect` (a contradiction, another product's or an older version's value, or the wrong refusal decision). The judge grades meaning, not language, spelling of numbers and units, or extra detail. Rates: each verdict over all questions, overall, per category and per language.
- **Faithfulness judge:** it sees every chunk sent to the answer model (user's choice; rendered as the answer model saw it, in the same order) and the numbered claim texts. It gives each claim `supported`, `partially_supported` (the claim adds something the context does not state) or `not_supported`. Outside knowledge and computed or converted values do not count as support. Chunk attributes (document, date, supersedes) do, so a "newer version" claim can be supported. Answers with no claims (refusals) get no call. Rates: claims at each level over all claims of judged answers, and answers whose claims are all supported over judged answers with claims.
- **Validation:** a judge reply must be JSON, match the schema, and for faithfulness give exactly one verdict per claim number 1..n. An invalid reply or a failed call gets no repair. It is counted as **not judged**: in the quality table, and for faithfulness in "answers not judged" over answers with claims. Its claims are left out of the claim rates.
- **Agreement with the deterministic scores:** a table of the S2-17 `passed` verdict against the quality verdict, in questions. This is not calibration: S2-19 compares the judge with hand grades.
- **Cost:** per phase (answers, quality judge, faithfulness judge): calls, the provider's prompt and completion tokens, the reported cost, and 1000 × cost / questions. A re-judged run spent nothing on answers, so "spent this run" counts only the judges.
- **Known limits:** the judge is the answer model, so it may grade its own answers too kindly (self-preference), which S2-19 measures. Each verdict comes from one call. The faithfulness judge does not see the quotes, so it checks the claim text against the context, not the citation (S2-13 already checks that each quote is in its chunk). n = 24, so one question moves a category rate by 12-20 points.

## Judge calibration (stage 2)
Added 2026-11-23 (S2-19), before the first judge call on the calibration answers. Code: `evals/calibration.py`, `scripts/calibrate_judge.py`; tests: `tests/test_calibration.py`.
- **Answers:** `make calibration-answers` answers all 24 test questions through the `make ask` pipeline with the score gate off (user's choice), so every question reaches the model and fewer answers are easy refusals. They are saved in `evals/results/calibration/answers.json` with no judge call. The answers come from the test set, but nothing in the answer system is tuned on them.
- **Blind hand grades:** the user grades every answer in `evals/judge_grades.toml` from `evals/results/calibration/sheet.md`. The sheet shows exactly what the quality judge is shown (the question, the English reference answer, and the shown answer as `render_answer` prints it, with no citations), plus the judge's rules from `prompts/judge_quality_v1.md`. `make calibrate-judge` calls the judge only when every answer has exactly one valid verdict, and records the grade file's sha256 before the first call. `--report` stops if the grade file changed after that.
- **Judge:** the quality judge of S2-18 (`[judge]` settings, prompt unchanged in S2-19), called 3 times per answer (user's choice), because the model takes no temperature and a verdict can change between calls. An invalid reply counts as the verdict `not_judged`.
- **Headline:** exact agreement on the 3 levels between the hand grade and the judge's **majority verdict** (the verdict of at least 2 of the 3 passes), over all graded answers. When the 3 passes all differ there is no majority, and the answer counts as a disagreement.
- **Also reported:** unweighted Cohen's kappa (3 levels; observed agreement corrected for the agreement two raters with these verdict counts would reach by chance; undefined when chance agreement is 1); binary agreement (correct against not correct); the confusion matrix of hand grade against judge majority; agreement of each pass alone; agreement on answers with claims and on refusals separately; agreement per category.
- **Self-preference, by direction (user's choice: no second judge model):** order the verdicts incorrect < partially_correct < correct. Among the disagreements, count the judge higher than the hand grade (lenient) against lower (strict), and with no verdict (no majority, or not judged). Many more lenient than strict disagreements would suggest the judge favours answers from its own model.
- **Judge consistency:** the share of answers where all 3 passes give the same verdict.
- **Grade spread (added 2026-11-24, after the run, as a reading note; no rule changed):** when every hand grade has the same level, chance agreement equals the judge's rate of that level, so kappa is 0 whatever the judge does. The report then says so and points to the exact agreement and the confusion matrix. When no answer is graded incorrect, it says that the sample does not show whether the judge agrees with a person on wrong answers.
- **Known limits:** n = 24, so one answer moves the agreement by about 4 points, and kappa is unstable at this size. One person graded. Many calibration answers are close to the S2-18 answers, whose judge verdicts the user had seen, so the grading was blind to these verdicts but not to earlier ones. If the judge prompt changes, calibrate again on new answers, because a prompt tuned on these disagreements would be measured on its own training data.
