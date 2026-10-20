# Issues and problems log

Problems met while building the project, what we considered, and what we chose. Newest first.
Each entry: date, task, problem, options considered, current choice and why, status.

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
  - D001 makes pypdf warn about 200 times that `fontTools` is needed for a font that holds only the space character. The text is unaffected, so the script logs pypdf errors only instead of adding `fontTools`.
- **Revisit if:** extraction errors (not model errors) show up in the S1-18 error analysis, or stage 2 needs table structure. Then try PyMuPDF or pdfplumber on the failing documents.
- **Status:** decided.
