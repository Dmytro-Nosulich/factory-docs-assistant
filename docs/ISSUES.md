# Issues and problems log

Problems met while building the project, what we considered, and what we chose. Newest first.
Each entry: date, task, problem, options considered, current choice and why, status.

## I-003: The cloud models on OpenRouter do not accept temperature or top_p
- **Date / task:** 2026-10-21, S1-07 (cloud model client)
- **Problem:** The assignment asks for sampling parameters (temperature, top_p) in config and a comparison of at least 2 configurations (S1-10). OpenRouter's model list (`GET /api/v1/models`, checked 2026-10-21) shows that the current mid-tier and small cloud models, including the chosen `anthropic/claude-haiku-5.5` and the earlier candidates `anthropic/claude-sonnet-5.5` and `openai/gpt-6.1-sol`, do not list `temperature` or `top_p` among their supported parameters. With `provider.require_parameters: true` (needed so structured output is honored), a request that sends them would find no provider.
- **Options considered:**
  - Compare sampling configurations on the local model only (Ollama accepts temperature and top_p).
  - Use reasoning effort as the cloud model's configuration axis instead of sampling.
  - Switch to an older cloud model that accepts temperature/top_p (e.g. `anthropic/claude-sonnet-4.6`, $3/$15 per 1M tokens).
- **Current selection:** sampling is compared on the local model only (decision 11). The cloud model runs with fixed settings from `config.toml` (`reasoning_effort = "none"`), and the client sends only the parameters that are set there.
- **Revisit if:** the reviewer expects a sampling comparison on the cloud model too. Then add a model that accepts temperature as a second cloud entry in `config.toml` (a config change only).
- **Status:** decided.

## I-002: Stage 1 had 15 texts, below the course minimum of 30 (resolved: 35)
- **Date / task:** 2026-10-20, S1-02 (corpus manifest)
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
- **Date / task:** 2026-10-19, S1-03 (text extraction)
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
  - D038 (2026-10-20, S1-05 labeling): the PDF's column and box layout breaks the reading order. The section 3 table (line 77) comes after section 4 and the hazard-label box, and "Form : Verdichtetes Gas" from section 9 (line 192) comes right after the section 13 disposal text. All values are present, but not inside their sections. Accepted as noise; check it first if D038 components or physical_state fail.
  - Other noise seen while labeling the golden set (S1-05): split digits in a value (D008 flash point "> 6 5 ° C") and words hyphenated across lines (D010 "наслід- ками"). The golden labels hold the document's real values ("> 65 °C", "наслідками").
  - D001 makes pypdf warn about 200 times that `fontTools` is needed for a font that holds only the space character. The text is unaffected, so the script logs pypdf errors only instead of adding `fontTools`.
- **Revisit if:** extraction errors (not model errors) show up in the S1-18 error analysis, or stage 2 needs table structure. Then try PyMuPDF or pdfplumber on the failing documents.
- **Status:** decided.
