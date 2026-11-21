"""Measure retrieval quality on the gold set: recall@k and MRR per setup and chunking (S2-10).

Runs every setup in `[retrieval_eval] setups` on every collection in `[index]` for each
question in evals/retrieval_gold.toml, with no filter, and scores the returned chunks with the
S2-09 hit rule (rules in docs/METRICS.md). Search settings come from config.toml unchanged;
nothing is tuned here.
Writes evals/results/retrieval.json (every rank and returned chunk ID, the settings, hashes of
the gold set and chunk files) and evals/results/retrieval.md, and rewrites the generated
"Retrieval metrics" section of REPORT.md.
Usage: uv run python scripts/eval_retrieval.py [--report]
  --report  rebuild the .md and the REPORT.md section from the saved JSON, no models
"""

import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version

from qdrant_client import QdrantClient

from factory_docs.config import ROOT, Config, load_config
from factory_docs.embedding import FastEmbedder
from factory_docs.index import BM25
from factory_docs.metrics import percentile
from factory_docs.rerank import FastReranker
from factory_docs.retrieval_eval import Setup, evaluate, pick_default, update_section
from factory_docs.retrieval_gold import GOLD, load_gold

OUT = ROOT / "evals" / "results"
JSON_OUT = OUT / "retrieval.json"
MD_OUT = OUT / "retrieval.md"
REPORT = ROOT / "REPORT.md"
INDEX_BUILD = ROOT / "data" / "index" / "index_build.json"
DEFAULT_COLLECTION = "structure"  # the collection `make search` uses; the default rule reads it
FILTER_FIELDS = "is_latest"  # a payload field every index built since S2-08 has


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_index(config: Config, client: QdrantClient) -> dict:
    """The index build record, after checking the index matches today's chunk files."""
    build = json.loads(INDEX_BUILD.read_text(encoding="utf-8"))
    for name, chunks_file in config.index.collections.items():
        recorded = build["collections"].get(name, {}).get("chunks_sha256")
        if recorded != sha256(ROOT / chunks_file):
            raise SystemExit(f"Index `{name}` was built from other chunks; run `make index`")
        if BM25 not in (client.get_collection(name).config.params.sparse_vectors or {}):
            raise SystemExit("The index has no BM25 vectors (built before S2-07); run `make index`")
        sample = client.scroll(name, limit=1, with_payload=True)[0]
        if sample and FILTER_FIELDS not in (sample[0].payload or {}):
            raise SystemExit(
                "The index has no filter fields (built before S2-08); run `make index`"
            )
    return build


def run(config: Config) -> dict:
    questions = load_gold()
    settings = config.retrieval_eval
    setups = [Setup.parse(name) for name in settings.setups]
    store = ROOT / config.index.path
    if not store.exists():
        raise SystemExit(f"No index at {config.index.path}; run `make index` first")
    embedder = FastEmbedder.from_config(config)
    reranker = FastReranker.from_config(config) if any(s.rerank for s in setups) else None
    # The first call of each model pays one-off ONNX set-up; keep it out of the timings
    embedder.embed_query("warm-up")
    if reranker:
        reranker.score("warm-up", ["warm-up"])

    client = QdrantClient(path=str(store))
    try:
        build = check_index(config, client)
        runs: dict[str, dict] = {}
        for collection in config.index.collections:
            runs[collection] = {}
            for setup in setups:
                print(f"{collection}, {setup.name} ...", flush=True)
                runs[collection][setup.name] = evaluate(
                    client,
                    collection,
                    questions,
                    setup,
                    settings.ks,
                    config.search,
                    embedder,
                    reranker,
                    config.reranking.candidates,
                )
    finally:
        client.close()

    return {
        "gold": {
            "path": str(GOLD.relative_to(ROOT)),
            "sha256": sha256(GOLD),
            "questions": [
                {
                    "id": q.id,
                    "language": q.language,
                    "question": q.question,
                    "evidence": [e.doc_id for e in q.evidence],
                }
                for q in questions
            ],
        },
        "settings": {
            "ks": settings.ks,
            "setups": settings.setups,
            "search": config.search.model_dump(exclude={"mode", "rerank"}),
            "rerank_candidates": config.reranking.candidates,
            "embedding_model": build["embedding"]["model"],
            "reranker_model": reranker.model_name if reranker else None,
        },
        "index": {
            name: {"chunks_file": c["chunks_file"], "chunks_sha256": c["chunks_sha256"]}
            for name, c in build["collections"].items()
            if name in config.index.collections
        },
        "versions": {name: version(name) for name in ("fastembed", "onnxruntime", "qdrant-client")},
        "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "runs": runs,
    }


def cell(summary: dict, k: str) -> str:
    return f"{summary['recall'][k]:.2f} ({summary['hits'][k]}/{summary['questions']})"


def rank_text(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def short(text: str, width: int = 70) -> str:
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= width else text[: width - 3] + "..."


def best_setup(data: dict, collection: str) -> str:
    runs = data["runs"][collection]
    summaries = {name: result["summary"] for name, result in runs.items()}
    rerank = {name: Setup.parse(name).rerank for name in runs}
    return pick_default(summaries, data["settings"]["ks"], rerank)


def summary_table(data: dict) -> list[str]:
    ks = [str(k) for k in data["settings"]["ks"]]
    depth = ks[-1]
    lines = [
        "| Chunking | Setup | " + " | ".join(f"Recall@{k}" for k in ks) + f" | MRR@{depth} |",
        "|---" * (len(ks) + 3) + "|",
    ]
    for collection, runs in data["runs"].items():
        for name, result in runs.items():
            s = result["summary"]
            lines.append(
                f"| {collection} | {name} | "
                + " | ".join(cell(s, k) for k in ks)
                + f" | {s['mrr']:.3f} |"
            )
    return lines


def default_lines(data: dict) -> list[str]:
    depth = data["settings"]["ks"][-1]
    chosen = best_setup(data, DEFAULT_COLLECTION)
    chosen_mrr = data["runs"][DEFAULT_COLLECTION][chosen]["summary"]["mrr"]
    lines = [
        f"Default-setup rule (fixed before the run): on the `{DEFAULT_COLLECTION}` collection, "
        f"the highest MRR@{depth}, then the higher recall@5, then the setup without a "
        f"reranker. Result: **{chosen}** (MRR@{depth} {chosen_mrr:.3f}). This choice is made "
        "on the gold set itself; no numeric setting was tuned on it.",
    ]
    for collection in data["runs"]:
        if collection == DEFAULT_COLLECTION:
            continue
        best = best_setup(data, collection)
        best_mrr = data["runs"][collection][best]["summary"]["mrr"]
        relation = "above" if best_mrr > chosen_mrr else "not above"
        lines.append("")
        lines.append(
            f"Best on `{collection}` by the same rule: {best} (MRR@{depth} {best_mrr:.3f}), "
            f"{relation} the `{DEFAULT_COLLECTION}` choice."
        )
    return lines


def language_tables(data: dict) -> list[str]:
    depth = data["settings"]["ks"][-1]
    k5 = "5" if 5 in data["settings"]["ks"] else str(depth)
    lines = []
    for collection, runs in data["runs"].items():
        languages = list(next(iter(runs.values()))["by_language"])
        lines += [
            f"**{collection}**: recall@{k5} (hits / questions) and MRR@{depth} per question "
            "language.",
            "",
            "| Setup | " + " | ".join(languages) + " |",
            "|---" * (len(languages) + 1) + "|",
        ]
        for name, result in runs.items():
            cells = []
            for language in languages:
                s = result["by_language"][language]
                cells.append(f"{s['hits'][k5]}/{s['questions']}, {s['mrr']:.2f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def question_table(data: dict) -> list[str]:
    columns = [(c, name) for c, runs in data["runs"].items() for name in runs]
    lines = [
        "| Q | Lang | Evidence | " + " | ".join(f"{c} {name}" for c, name in columns) + " |",
        "|---" * (len(columns) + 3) + "|",
    ]
    for q in data["gold"]["questions"]:
        ranks = [
            rank_text(data["runs"][c][name]["questions"][q["id"]]["rank"]) for c, name in columns
        ]
        lines.append(
            f"| {q['id']} | {q['language']} | {', '.join(q['evidence'])} | "
            + " | ".join(ranks)
            + " |"
        )
    lines += ["", "Questions:", ""]
    lines += [
        f"- {q['id']} ({q['language']}): {short(q['question'], 120)}"
        for q in data["gold"]["questions"]
    ]
    return lines


def timing_table(data: dict) -> list[str]:
    lines = ["| Chunking | Setup | p50 s | max s |", "|---" * 4 + "|"]
    for collection, runs in data["runs"].items():
        for name, result in runs.items():
            seconds = [r["seconds"] for r in result["questions"].values()]
            lines.append(
                f"| {collection} | {name} | {percentile(seconds, 50):.2f} | {max(seconds):.2f} |"
            )
    return lines


def intro(data: dict) -> str:
    s = data["settings"]
    search = s["search"]
    n = len(data["gold"]["questions"])
    reranker = f", reranker `{s['reranker_model']}`" if s["reranker_model"] else ""
    return (
        f"{n} questions from `{data['gold']['path']}`, no filters, top {s['ks'][-1]} chunks per "
        f"search. Embedding model `{s['embedding_model']}`{reranker}. Settings from "
        f"config.toml, none tuned on these questions: {search['prefetch_k']} candidates per "
        f"retriever, RRF k {search['rrf_k']}, BM25 k1 {search['bm25_k1']} and b "
        f"{search['bm25_b']}, {s['rerank_candidates']} hybrid candidates reranked. A chunk is "
        "a hit when it comes from the evidence document and contains its quote; rules in "
        f"docs/METRICS.md. One question is 1/{n} of a recall value."
    )


def markdown(data: dict) -> str:
    m = data["machine"]
    lines = [
        "# Retrieval measurement (S2-10)",
        "",
        "Generated by `make eval-retrieval`; do not edit by hand.",
        "",
        intro(data),
        "",
        "## Recall and MRR",
        "",
        *summary_table(data),
        "",
        *default_lines(data),
        "",
        "## By language",
        "",
        *language_tables(data),
        "## Per question",
        "",
        f"Rank of the first chunk that answers the question; `-` = not in the top "
        f"{data['settings']['ks'][-1]}.",
        "",
        *question_table(data),
        "",
        "## Timing",
        "",
        f"Seconds per query, including the embedding of the query (machine-dependent: "
        f"{m['platform']}, {m['cpu_count']} cores, CPU only). Not part of the reproducibility "
        "check.",
        "",
        *timing_table(data),
    ]
    return "\n".join(lines).rstrip() + "\n"


def report_section(data: dict) -> str:
    lines = [
        intro(data),
        "",
        *summary_table(data),
        "",
        *default_lines(data),
        "",
        *language_tables(data),
        "Per-question ranks and timings: [evals/results/retrieval.md](evals/results/retrieval.md).",
    ]
    return "\n".join(lines)


def write_outputs(data: dict) -> None:
    MD_OUT.write_text(markdown(data), encoding="utf-8")
    update_section(REPORT, report_section(data))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report", action="store_true", help=f"rebuild from {JSON_OUT.name}, no models"
    )
    args = parser.parse_args()
    if args.report:
        write_outputs(json.loads(JSON_OUT.read_text(encoding="utf-8")))
        print(f"Wrote {MD_OUT.relative_to(ROOT)} and REPORT.md from {JSON_OUT.name}")
        return
    data = run(load_config())
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_outputs(data)
    depth = data["settings"]["ks"][-1]
    for collection, runs in data["runs"].items():
        for name, result in runs.items():
            print(f"{collection:9} {name:14} MRR@{depth} {result['summary']['mrr']:.3f}")
    print(f"Default by the rule: {best_setup(data, DEFAULT_COLLECTION)}")
    print(f"Wrote {JSON_OUT.relative_to(ROOT)}, {MD_OUT.relative_to(ROOT)} and REPORT.md")


if __name__ == "__main__":
    main()
