# Issues and problems log

Problems met while building the project, what we considered, and what we chose. Newest first.
Each entry: date, task, problem, options considered, current choice and why, status.

## I-002: Stage 1 has 15 texts, below the course minimum of 30
- **Date / task:** 2026-10-20, S1-02 (corpus manifest)
- **Problem:** The course teacher advised focusing stage 1 on safety data sheets (SDS), because the other document types (manuals, certificates, declarations, datasheets) are hard to read. The corpus has only 15 SDS (D001, D002, D003, D006, D007, D008, D010, D032-D039; en, sv, pl, uk, de), while the assignment asks for at least 30 texts.
- **Options considered:**
  - Add about 15 more public SDS (same issuers as D006/D032 and D007/D033) to reach 30.
  - Ask the teacher whether fewer than 30 is acceptable for an SDS-only set.
  - Proceed with the 15 SDS we have.
- **Current selection:** proceed with the 15 SDS. The 20 non-SDS documents move to stage 2 (RAG) in `data/manifest.csv`; their texts stay in `data/texts/` for stage 2. The golden set keeps its 4 SDS (D001, D002, D006, D033) and gets 6 SDS replacements.
- **Revisit if:** the reviewer or rubric penalizes fewer than 30 texts, or the metrics on 15 documents are too noisy to compare models. Then add more SDS (only manifest rows and PDFs; the pipeline needs no change).
- **Status:** accepted risk.

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
