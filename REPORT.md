# Stage 2 report: RAG with citations and evals

This report is in progress. Its sections are filled in as the stage 2 pipeline is built, and every number will come from a script in this repo.

The stage 1 report (structured extraction) is in [docs/reports/stage-1.md](docs/reports/stage-1.md).

## Retrieval metrics
Not measured yet (S2-10).

**Gold set (S2-09).** `evals/retrieval_gold.toml` holds 15 questions (en 6, uk 4, de 3, fr 1, it 1) over PDF, HTML, DOCX and Markdown documents. Each question names its evidence: document, section, page (PDF only) and a short exact quote (decision 16). A retrieved chunk counts as correct when it comes from the evidence document and contains the quote, with whitespace ignored. All 15 questions were drafted by Claude at the user's request (the user does not read German, French or Italian well) and reviewed by the user. `make retrieval-gold-check` confirms that every quote is in its document, on its page, and whole inside at least one chunk of both chunking strategies.

## Answer metrics
Not measured yet.

## Error analysis
Not written yet.

## What to improve next
Not written yet.
