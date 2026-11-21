"""Answer a question from the indexed documents with checked citations (S2-13).

Searches `[answer] collection` with the [search] settings (hybrid, reranked), fits the best
`[answer] k` hits into `[answer] max_context_tokens`, asks `[answer] model`, checks every
citation, and writes the record to results/answers/<UTC time>.json.
Usage: uv run python scripts/ask.py "question" [--collection structure|fixed]
       [--doc-type sds,manual] [--language uk,de] [--since 2024-01-01] [--until 2025-12-31]
       [--latest-only]
"""

import argparse
import textwrap
from datetime import date

from qdrant_client import QdrantClient

from factory_docs.answer import (
    SHOWN,
    BudgetExceeded,
    answer_question,
    save_record,
    shown_answer,
    spent_on_answers,
)
from factory_docs.clients import make_client
from factory_docs.config import ROOT, load_config, load_env
from factory_docs.embedding import FastEmbedder
from factory_docs.index import BM25, SearchFilter, search
from factory_docs.rerank import FastReranker
from factory_docs.run import record_cost

FILTER_FIELDS = "is_latest"  # a payload field every index built since S2-08 has


def values(text: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in text.split(",") if v.strip())


def source_line(source: dict) -> str:
    page = f", p. {source['pages']}" if source["pages"] else ""
    return f"{source['doc_id']} {source['title']}: {source['section']}{page}"


def print_record(record: dict) -> None:
    retrieval = record["retrieval"]
    sources = {s["chunk_id"]: s for s in retrieval["sent"]}
    print(
        f"Context: {len(retrieval['sent'])} chunks sent, {len(retrieval['dropped'])} dropped "
        f"by the budget, about {retrieval['est_context_tokens']} of "
        f"{retrieval['max_context_tokens']} tokens (estimate)\n"
    )
    answer = shown_answer(record)
    if answer is None:
        print(f"No answer: {record['final_status']}")
        for error in (record.get("repair") or record)["errors"]:
            print(f"  - {error}")
    else:
        print(f"Status: {answer['status']}")
        for number, claim in enumerate(answer["claims"], start=1):
            print(
                textwrap.fill(
                    claim["text"], 96, initial_indent=f"{number}. ", subsequent_indent="   "
                )
            )
            for citation in claim["citations"]:
                print(f"   [{citation['chunk_id']}] {source_line(sources[citation['chunk_id']])}")
                print(f'     "{" ".join(citation["quote"].split())}"')
        if answer["missing"]:
            print(textwrap.fill(f"Missing: {answer['missing']}", 96))
        if record["final_status"] == "repaired":
            print("\n(The first answer failed the checks and was repaired:)")
            for error in record["errors"]:
                print(f"  - {error}")
    calls = [c for c in (record["call"], (record["repair"] or {}).get("call")) if c]
    tokens_in = sum(c.get("prompt_tokens") or 0 for c in calls)
    tokens_out = sum(c.get("completion_tokens") or 0 for c in calls)
    latency = sum(c.get("total_s") or 0 for c in calls)
    model = record["model"]
    prompt = record["prompt"]
    print(
        f"\n{model['key']} ({model.get('provider') or '-'}), prompt {prompt['name']} "
        f"{prompt['version']}, {len(calls)} call(s), {tokens_in} prompt / {tokens_out} output "
        f"tokens, {latency:.1f} s, ${record_cost(record):.5f}"
    )


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
    store = ROOT / config.index.path
    if not store.exists():
        raise SystemExit(f"No index at {config.index.path}; run `make index` first")
    mode = config.search.mode
    embedder = None if mode == "bm25" else FastEmbedder.from_config(config)
    reranker = FastReranker.from_config(config) if config.search.rerank else None
    qdrant = QdrantClient(path=str(store))
    try:
        if BM25 not in (qdrant.get_collection(args.collection).config.params.sparse_vectors or {}):
            raise SystemExit("The index has no BM25 vectors (built before S2-07); run `make index`")
        sample = qdrant.scroll(args.collection, limit=1, with_payload=True)[0]
        if sample and FILTER_FIELDS not in (sample[0].payload or {}):
            raise SystemExit(
                "The index has no filter fields (built before S2-08); run `make index`"
            )
        hits = search(
            qdrant,
            args.collection,
            args.question,
            embedder,
            settings.k,
            config.search,
            mode,
            search_filter,
            reranker,
            config.reranking.candidates,
        )
    finally:
        qdrant.close()
    retrieval = {
        "collection": args.collection,
        "mode": mode,
        "embedding_model": embedder.model_name if embedder else None,
        "reranker": reranker.model_name if reranker else None,
        "rerank_candidates": config.reranking.candidates if reranker else None,
        "filter": search_filter.describe(),
    }

    client = make_client(settings.model, config)
    try:
        record = answer_question(args.question, hits, client, config, spent, retrieval)
    except BudgetExceeded as e:
        raise SystemExit(f"Budget guard: {e}") from e
    finally:
        client.close()
    path = save_record(record)
    print(f"{args.collection}, {mode}, filter: {search_filter.describe()}: {args.question}")
    print_record(record)
    print(
        f"Saved: {path.relative_to(ROOT)}; stage 2 answers so far "
        f"${spent + record_cost(record):.4f} of ${settings.stop_at_usd:.2f}"
    )
    if record["final_status"] not in SHOWN and record["final_status"] != "no_context":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
