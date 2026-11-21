"""Answer a question from the indexed documents with checked citations (S2-13).

Searches `[answer] collection` with the [search] settings (hybrid, reranked), refuses without a
model call when the best hit scores below `[answer] min_rerank_score` (S2-14), fits the best
`[answer] k` hits into `[answer] max_context_tokens`, asks `[answer] model`, checks every
citation, and writes the record to results/answers/<UTC time>.json.
Usage: uv run python scripts/ask.py "question" [--collection structure|fixed]
       [--doc-type sds,manual] [--language uk,de] [--since 2024-01-01] [--until 2025-12-31]
       [--latest-only]
"""

import argparse
from datetime import date

from factory_docs.answer import (
    NOT_CALLED,
    SHOWN,
    BudgetExceeded,
    answer_question,
    save_record,
    spent_on_answers,
)
from factory_docs.answer_display import print_record
from factory_docs.clients import make_client
from factory_docs.config import ROOT, load_config, load_env
from factory_docs.index import SearchFilter
from factory_docs.retriever import IndexNotReady, open_retriever
from factory_docs.run import record_cost


def values(text: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in text.split(",") if v.strip())


def main() -> None:
    config = load_config()
    settings = config.answer
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument(
        "--collection", default=settings.collection, choices=list(config.index.collections)
    )
    parser.add_argument("--doc-type", type=values, default=(), help="comma-separated doc_type")
    parser.add_argument("--language", type=values, default=(), help="comma-separated languages")
    parser.add_argument("--since", type=date.fromisoformat, help="document date on or after")
    parser.add_argument("--until", type=date.fromisoformat, help="document date on or before")
    parser.add_argument("--latest-only", action="store_true", help="skip superseded versions")
    args = parser.parse_args()
    if not args.question.strip():
        raise SystemExit('Give a question: make ask Q="..."')
    search_filter = SearchFilter(
        doc_types=args.doc_type,
        languages=args.language,
        date_from=args.since,
        date_to=args.until,
        latest_only=args.latest_only,
    )

    load_env()
    spent = spent_on_answers()
    if spent >= settings.stop_at_usd:
        raise SystemExit(f"Budget guard: {BudgetExceeded.__doc__} (${spent:.4f})")
    try:
        with open_retriever(config, args.collection) as retriever:
            hits = retriever.search(args.question, search_filter)
            retrieval = retriever.info(search_filter)
    except IndexNotReady as e:
        raise SystemExit(str(e)) from e

    client = make_client(settings.model, config)
    try:
        record = answer_question(args.question, hits, client, config, spent, retrieval)
    except BudgetExceeded as e:
        raise SystemExit(f"Budget guard: {e}") from e
    finally:
        client.close()
    path = save_record(record)
    mode = config.search.mode
    print(f"{args.collection}, {mode}, filter: {search_filter.describe()}: {args.question}")
    print_record(record)
    print(
        f"Saved: {path.relative_to(ROOT)}; stage 2 answers so far "
        f"${spent + record_cost(record):.4f} of ${settings.stop_at_usd:.2f}"
    )
    if record["final_status"] not in (*SHOWN, *NOT_CALLED):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
