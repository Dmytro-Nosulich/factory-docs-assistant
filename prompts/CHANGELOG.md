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
- Measured effect: baseline (`results/comparison.md`, golden set of 10 documents, all 35 SDS
  schema-valid on the first response in every run):

  | Run | Field accuracy | Latency p50 / p95 | $ / 1000 docs |
  | --- | --- | --- | --- |
  | Haiku 5.5 | 85/90 (94.4%) | 2.2 / 3.9 s | $1.96 |
  | qwen3.5-9b default | 77/90 (85.6%) | 26.0 / 84.5 s | 0 (local) |
  | qwen3.5-9b greedy | 79/90 (87.8%) | 27.0 / 85.8 s | 0 (local) |

  These numbers use the D006 golden label as corrected on 2026-10-27 (the user removed a
  `[18.12.2018]` that is not part of the trade name). With the old label, S1-12 reported 84, 76
  and 78 of 90.

## extract v2 (2026-10-27)
- What: v1 unchanged, plus a "Field rules" block in the system message:
  1. product_name: if section 1 gives both a product name and a trade name, use the trade name.
  2. language: ISO 639-1 code of the body text (`sv`, not `se`), not of the product name.
  3. signal_word: translations mapped to danger / warning (Achtung, Uwaga, Varning, Увага ->
     warning; Gefahr, Fara, Небезпечно -> danger).
  4. physical_state: solid, liquid or gas only; null for paste, cream or gel.
  5. min / max for `<`, `≤`, `>`, `≥`, ranges and single values, with examples.
  6. hazard_statements: every H and EUH statement for the product in section 2, each code once.
- Why: the v1 errors on the golden set (all three v1 runs):
  1. D001, D002: the "Product name" line ("Nitrogen, compressed") was used, not the
     "Trade name" line (all runs).
  2. D034: English body text, German trade name -> `de` (all runs). D002: `se` (greedy).
  3. D038: "Achtung" -> `danger` (Haiku).
  4. D010: "крем, паста" -> `liquid` (all runs; golden null).
  5. D003, D007, D008, D010: `< 0,05 %` read as a minimum, `1 - 10 %` without its maximum,
     `≥ 99 %` as a maximum (qwen).
  6. D003, D007, D008, D010: missing H315/H317/H318, H412, EUH066, EUH208, EUH210 (qwen).
- Measured effect (`results/comparison.md`; v1 -> v2, all runs 35/35 valid on the first
  response):

  | Run | Field accuracy | Latency p50 / p95 | $ / 1000 docs |
  | --- | --- | --- | --- |
  | Haiku 5.5 | 85 -> 89/90 (94.4% -> 98.9%) | 2.2 / 3.9 -> 2.4 / 4.5 s | $1.96 -> $2.02 |
  | qwen3.5-9b default | 77 -> 80/90 (85.6% -> 88.9%) | 26.0 / 84.5 -> 28.1 / 86.9 s | 0 |
  | qwen3.5-9b greedy | 79 -> 80/90 (87.8% -> 88.9%) | 27.0 / 85.8 -> 28.8 / 86.6 s | 0 |

  Per field, documents matched out of 10 (v1 -> v2):

  | Field | Haiku | qwen default | qwen greedy |
  | --- | --- | --- | --- |
  | product_name | 8 -> 9 | 8 -> 9 | 8 -> 8 |
  | language | 9 -> 10 | 9 -> 9 | 8 -> 8 |
  | signal_word | 9 -> 10 | 10 -> 10 | 10 -> 10 |
  | physical_state | 9 -> 10 | 9 -> **8** | 9 -> 10 |
  | components (F1) | 10 -> 10 (1.00) | 6 -> 7 (0.50 -> 0.69) | 7 -> 7 (0.69 -> 0.69) |
  | hazard_statements (F1) | 10 -> 10 (1.00) | 7 -> 8 (0.76 -> 0.87) | 7 -> 9 (0.82 -> 0.96) |
  | flash_point | 10 -> 10 | 9 -> 9 | 10 -> **8** |
  | revision_date | 10 -> 10 | 9 -> 10 | 10 -> 10 |

  - Helped: Haiku now misses 1 field of 90. The hazard-statement rule worked best on qwen
    (F1 0.76 -> 0.87 default, 0.82 -> 0.96 greedy).
  - Made worse: the physical_state rule was over-applied by qwen default. It fixed D010, but
    returned null for D002 ("Form: KOMPRIMERAD GAS") and D034 ("Physical state Fluid"), which
    are gas and liquid. A next version should map synonyms (compressed gas -> gas, fluid ->
    liquid) before falling back to null.
  - Made worse: flash_point on qwen greedy, from the min / max rule. Both golden values are
    lower bounds only, and v1 had both right. v2 wrote D008 (golden min 65) as min = max = 65
    and turned D010 round (golden min 190) to min null, max 190. The rule's examples are all
    concentrations; the model seems to have applied them to temperatures wrongly.
  - Little effect: the min / max rule on qwen (components 6 -> 7 and 7 -> 7). In a D007 smoke
    test, qwen still wrote `<0,025%` as min = max = 0.025, although the rule names this exact
    case.
  - Caveat: n = 10 golden documents and one run per configuration; qwen default samples at
    temperature 1.0, so a one-field change there is within run-to-run noise. The prompt is
    longer, so cloud cost rose by about 3%.

## extract v3 (2026-10-27)
- What: v2 with two rules rewritten; the other rules are unchanged.
  1. physical_state: map the section 9 form to solid, liquid or gas first, with synonyms in
     several languages (compressed gas, Verdichtetes Gas, Komprimerad gas -> gas; fluid,
     Flüssigkeit, vätska, рідина -> liquid; powder, crystals -> solid). Null only for other
     forms such as paste, cream or gel.
  2. min / max: the examples now cover temperatures too (`> 65 °C` -> min 65, max null), with
     the sentence "a value written with <, ≤, > or ≥ is never a single value: never set
     min = max for it".
- Why: the two v2 regressions. physical_state was null for "KOMPRIMERAD GAS" (D002) and
  "Fluid" (D034), on qwen default. flash_point lower bounds were wrong on qwen greedy: D008
  (min = max = 65), and D010 (max 190 instead of min 190).
- Measured effect (`results/comparison.md`; v2 -> v3):

  | Run | Valid (1st) | Field accuracy | Field accuracy, valid only |
  | --- | --- | --- | --- |
  | Haiku 5.5 | 35/35 -> 35/35 | 89 -> 88/90 (98.9% -> 97.8%) | same |
  | qwen3.5-9b default | 35/35 -> **34/35** | 80 -> 73/90 (88.9% -> 81.1%) | 80/90 -> 73/81 (90.1%) |
  | qwen3.5-9b greedy | 35/35 -> **34/35** | 80 -> 81/90 (88.9% -> 90.0%) | 80/90 -> 81/90 |

  Latency and cost changed little: Haiku $2.02 -> $2.04 per 1000 docs; qwen p50 about 28 s for
  both versions.

  - Fixed: physical_state. D002 gas, D034 liquid and D010 null are now right in all three
    configurations. qwen default shows 9/10 only because D001 failed (below).
  - Partly fixed: flash_point on qwen greedy, 8 -> 9/10. D008 is right (min 65); D010 is still
    max 190 instead of min 190. On qwen default, D010 came back null.
  - Made worse: validity on qwen, 100% -> 97.1% in both configurations, the first invalid
    first responses in any full run. Both are concentration ranges that the schema rejects:
    - D001 (default): nitrogen at 100 % came back as min null, max null. "Never set
      min = max" seems to have overridden "a single value -> min = max". A failed document
      counts as 9 missed fields, which is most of the drop from 80 to 73/90. On the valid
      documents, v3 (73/81, 90.1%) is close to v2 (88.9%).
    - D037 (greedy, not in the golden set): the section 3 table has "Spezifische
      Konzentrationsgrenze: … C ≥ 10 %" above the concentration "1-5%". qwen took min 10
      from the classification limit and max 5 from the concentration (min > max). The
      stronger "≥ x -> min" examples may have pulled the limit in.
  - Unchanged and not targeted: Haiku revision_date for D007 (v2 2022-10-10, right; v3
    null). v3 did not touch dates, so this counts as run-to-run variance.
  - Caveat: one run per configuration and 10 golden documents; qwen default samples at
    temperature 1.0. Both invalid responses are cases for S1-15 (one repair attempt).
  - Next step, if any: drop the "never min = max" sentence, which caused D001, and tell the
    model to ignore specific concentration limits ("C ≥ …"), which caused D037.

## repair v1 (2026-10-27)
- What: a new prompt, not an extraction version. When a reply fails validation, the model gets
  one repair attempt: the same conversation plus its own invalid reply and this user message,
  which lists the validation errors and asks for the complete corrected JSON, changing only what
  the errors require. No system section; it continues the extraction conversation (S1-15).
- Why: S1-14 produced the first invalid first responses (qwen v3: D001 default, D037 greedy).
  Both break Pydantic range rules that the JSON schema cannot express, so constrained decoding
  does not prevent them (`docs/ISSUES.md` I-005). The prompt is generic on purpose: it names no
  field and suggests no fix.
- Measured effect (`make repair` on the 9 comparison runs, then `make comparison`; first-response
  columns unchanged): the two invalid replies were both repaired, so every run is 35/35 valid
  after repair. Both repairs passed the schema by setting the concentration to null:
  - D001 (qwen default v3): nitrogen became `concentration: null`; the label is 100 %. The other
    8 fields match, so field accuracy goes 73/90 (81.1%) -> 81/90 (90.0%) after repair, but
    `components` is still wrong.
  - D037 (qwen greedy v3, not in the golden set): the first component became `null`; the text
    says "1-5%", so the right value is min 1, max 5.
  - Repair calls: 28.2 s and 23.5 s on the M4 Pro (local, $0).
  - Caveat: n = 2. Repair restores validity, not accuracy: both fixes chose the easy null over
    the value in the document.
