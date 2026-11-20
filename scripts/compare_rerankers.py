"""Compare the multilingual FastEmbed cross-encoders on size, speed and truncation (S2-08).

The hybrid search of the current index gives the first `[reranking] candidates` structure
chunks for each smoke query. Each model in `[reranking] compare` then runs in its own process
(so load time and peak memory are its own): download it once, load it from the cache, count
each query + chunk pair's tokens, time reranking the candidates, and list the reranked top 5.
The smoke queries are a by-eye check, not a quality metric (S2-10 measures quality on the
hand-written gold set).
Writes data/index/reranker_comparison.json (raw numbers) and reranker_comparison.md.
Usage: uv run python scripts/compare_rerankers.py [--models key ...] [--report]
"""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from qdrant_client import QdrantClient

from factory_docs.config import ROOT, load_config
from factory_docs.embedding import FastEmbedder
from factory_docs.index import hybrid_search
from factory_docs.metrics import percentile
from factory_docs.rerank import FastReranker

OUT = ROOT / "data" / "index"
JSON_OUT = OUT / "reranker_comparison.json"
MD_OUT = OUT / "reranker_comparison.md"
COLLECTION = "structure"
TOP = 5
REPEATS = 3  # timed reranks per query, after one warm-up
# uk and de: the S2-06 smoke queries. en: added with the S2-08 plan (2026-11-20). Written by
# Claude, not gold labels.
SMOKE_QUERIES = {
    "uk": "Що робити, якщо засіб потрапив в очі?",
    "de": "Wie hoch ist der Flammpunkt von RM 110?",
    "en": "Which configuration styles does the Telegraf Modbus input plugin support?",
}
# The chunk that answers the de query (docs/ISSUES.md I-007), tracked in every list
WATCH = {"de": "D033-s-0012"}


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # Linux reports KiB


def weights_bytes(reranker: FastReranker) -> int:
    """The ONNX files this model loads. Both bge-m3 builds come from one Hugging Face repo, so
    the cache folder holds both, and its total would count the other build too."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    description = next(
        m for m in TextCrossEncoder.list_supported_models() if m["model"] == reranker.model_name
    )
    folder = Path(reranker.model.model._model_dir)
    files = [description["model_file"], *(description.get("additional_files") or [])]
    return sum((folder / name).stat().st_size for name in files)


def candidates() -> dict:
    """The hybrid candidates of each smoke query: what every reranker gets as input."""
    config = load_config()
    embedder = FastEmbedder.from_config(config)
    client = QdrantClient(path=str(ROOT / config.index.path))
    try:
        result = {}
        for language, query in SMOKE_QUERIES.items():
            hits = hybrid_search(
                client,
                COLLECTION,
                query,
                embedder,
                config.reranking.candidates,
                config.search.prefetch_k,
                config.search.rrf_k,
            )
            result[language] = [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "language": h.chunk.language,
                    "section": h.chunk.section,
                    "text": h.chunk.text,
                }
                for h in hits
            ]
        return result
    finally:
        client.close()


def measure(key: str, pools: dict) -> dict:
    """Every number for one model; run in a fresh process after the model is downloaded."""
    config = load_config()
    start = time.perf_counter()
    reranker = FastReranker.from_config(config, key)
    load_s = time.perf_counter() - start

    limit = reranker.token_limit
    lengths = []
    for language, query in SMOKE_QUERIES.items():
        lengths += reranker.token_lengths(query, [c["text"] for c in pools[language]])

    reranker.score("warm-up", ["warm-up"])  # the first call pays one-off ONNX set-up
    timings = []
    ranked = {}
    for language, query in SMOKE_QUERIES.items():
        texts = [c["text"] for c in pools[language]]
        for _ in range(REPEATS):
            start = time.perf_counter()
            scores = reranker.score(query, texts)
            timings.append(time.perf_counter() - start)
        order = sorted(
            range(len(texts)), key=lambda i: (-scores[i], pools[language][i]["chunk_id"])
        )
        ranked[language] = [
            {
                "chunk_id": pools[language][i]["chunk_id"],
                "first_stage_rank": i + 1,
                "score": round(scores[i], 4),
            }
            for i in order
        ]

    return {
        "key": key,
        "model": reranker.model_name,
        "size_bytes": weights_bytes(reranker),
        "token_limit": limit,
        "pairs": len(lengths),
        "pairs_over_limit": sum(limit is not None and n > limit for n in lengths),
        "tokens_p50": percentile(lengths, 50),
        "tokens_p95": percentile(lengths, 95),
        "tokens_max": max(lengths),
        "load_s": round(load_s, 2),
        "rerank_s_p50": round(percentile(timings, 50), 2),
        "rerank_s_max": round(max(timings), 2),
        "peak_rss_bytes": peak_rss_bytes(),
        "ranked": ranked,
    }


def run_in_subprocess(*args: str, stdin: str = "") -> str:
    command = [sys.executable, __file__, *args]
    result = subprocess.run(command, check=True, capture_output=True, text=True, input=stdin)
    return result.stdout


def machine() -> dict:
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "fastembed": version("fastembed"),
        "onnxruntime": version("onnxruntime"),
    }


def licences() -> dict[str, str]:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return {m["model"]: m["license"] for m in TextCrossEncoder.list_supported_models()}


def short(text: str, width: int = 60) -> str:
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= width else text[: width - 3] + "..."


def position(ids: list[str], chunk_id: str) -> str:
    return str(ids.index(chunk_id) + 1) if chunk_id in ids else "not in candidates"


def markdown(data: dict) -> str:
    m = data["machine"]
    n = data["candidates"]
    lines = [
        "# Reranker comparison (S2-08)",
        "",
        "Generated by `make compare-rerankers`; do not edit by hand.",
        "",
        f"Machine: {m['platform']}, {m['cpu_count']} cores; FastEmbed {m['fastembed']}, "
        f"ONNX Runtime {m['onnxruntime']} on CPU. Input: for each of the "
        f"{len(SMOKE_QUERIES)} smoke queries, the first {n} hybrid hits (S2-07) in the "
        f"`{COLLECTION}` collection; every model reranks the same candidates.",
        "",
        "Columns: *Weights* is the ONNX files the model loads (the tokenizer files, 17 MB, "
        "are not counted). *Limit* is the length in tokens "
        "FastEmbed truncates each query + chunk pair to; text past it is ignored without a "
        "warning. *Over limit* counts the pairs longer than that. *Load* is loading from the "
        f"cache, *Rerank* is the time to score {n} candidates for one query (median and "
        f"slowest of {REPEATS * len(SMOKE_QUERIES)} timed runs), *RAM* is the process's peak.",
        "",
        "| Model | Licence | Weights | Limit | Tokens p50 / p95 / max | Over limit | Load | "
        "Rerank p50 / max | RAM |",
        "|---" * 9 + "|",
    ]
    for r in data["results"]:
        lines.append(
            f"| `{r['model']}` | {r['licence']} | {r['size_bytes'] / 1e9:.2f} GB | "
            f"{r['token_limit']} | {r['tokens_p50']} / {r['tokens_p95']} / {r['tokens_max']} | "
            f"{r['pairs_over_limit']} of {r['pairs']} | {r['load_s']:.1f} s | "
            f"{r['rerank_s_p50']:.2f} / {r['rerank_s_max']:.2f} s | "
            f"{r['peak_rss_bytes'] / 1e9:.1f} GB |"
        )
    if len(data["results"]) == 2:
        a, b = data["results"]
        lines += ["", f"Top {TOP} agreement between `{a['key']}` and `{b['key']}`:", ""]
        for language in SMOKE_QUERIES:
            top_a = [h["chunk_id"] for h in a["ranked"][language][:TOP]]
            top_b = [h["chunk_id"] for h in b["ranked"][language][:TOP]]
            same_order = "same order" if top_a == top_b else "different order"
            lines.append(
                f"- {language}: {len(set(top_a) & set(top_b))} of {TOP} chunks shared, {same_order}"
            )
    lines += [
        "",
        "## Smoke queries",
        "",
        f"Hybrid top {TOP} and each model's reranked top {TOP}. *From* is the chunk's hybrid "
        "rank before reranking. A by-eye check, not a quality metric; search quality is "
        "measured in S2-10 on the hand-written gold set.",
        "",
    ]
    for language, query in SMOKE_QUERIES.items():
        pool = data["pools"][language]
        lines += [f"### {language}: {query}", "", "**hybrid (no reranking)**", ""]
        lines += ["| # | Chunk | Lang | Section |", "|---" * 4 + "|"]
        for rank, c in enumerate(pool[:TOP], start=1):
            lines.append(f"| {rank} | {c['chunk_id']} | {c['language']} | {short(c['section'])} |")
        by_id = {c["chunk_id"]: c for c in pool}
        for r in data["results"]:
            lines += ["", f"**{r['key']}**", ""]
            lines += ["| # | Chunk | Lang | Section | From | Score |", "|---" * 6 + "|"]
            for rank, h in enumerate(r["ranked"][language][:TOP], start=1):
                c = by_id[h["chunk_id"]]
                lines.append(
                    f"| {rank} | {h['chunk_id']} | {c['language']} | {short(c['section'])} | "
                    f"{h['first_stage_rank']} | {h['score']:.3f} |"
                )
        if language in WATCH:
            watch = WATCH[language]
            lines += [
                "",
                f"Answer chunk `{watch}` (docs/ISSUES.md I-007): hybrid rank "
                f"{position([c['chunk_id'] for c in pool], watch)}",
            ]
            for r in data["results"]:
                ids = [h["chunk_id"] for h in r["ranked"][language]]
                lines[-1] += f", {r['key']} rank {position(ids, watch)}"
            lines[-1] += "."
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="keys in [reranker_models]; default: compare")
    parser.add_argument(
        "--report", action="store_true", help=f"rebuild the .md from {JSON_OUT.name}, no models"
    )
    parser.add_argument("--download", help=argparse.SUPPRESS)  # internal: one model, then exit
    parser.add_argument("--measure", help=argparse.SUPPRESS)  # internal: print one result
    args = parser.parse_args()
    if args.report:
        MD_OUT.write_text(markdown(json.loads(JSON_OUT.read_text(encoding="utf-8"))), "utf-8")
        print(f"Wrote {MD_OUT.relative_to(ROOT)} from {JSON_OUT.name}")
        return
    if args.download:
        FastReranker.from_config(load_config(), args.download)
        return
    if args.measure:
        print(json.dumps(measure(args.measure, json.loads(sys.stdin.read()))))
        return

    config = load_config()
    if not (ROOT / config.index.path).exists():
        raise SystemExit(f"No index at {config.index.path}; run `make index` first")
    keys = args.models or config.reranking.compare
    print("hybrid candidates for the smoke queries ...", flush=True)
    pools = candidates()
    names = licences()
    results = []
    for key in keys:
        print(f"{key}: downloading if needed ...", flush=True)
        run_in_subprocess("--download", key)
        print(f"{key}: measuring ...", flush=True)
        out = run_in_subprocess("--measure", key, stdin=json.dumps(pools))
        result = json.loads(out.strip().splitlines()[-1])
        result["licence"] = names.get(result["model"], "?")
        print(f"{key}: rerank p50 {result['rerank_s_p50']} s", flush=True)
        results.append(result)
    data = {
        "machine": machine(),
        "candidates": config.reranking.candidates,
        # chunk text left out: it is in data/chunks/chunks.jsonl
        "pools": {
            lang: [{k: v for k, v in c.items() if k != "text"} for c in pool]
            for lang, pool in pools.items()
        },
        "results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    MD_OUT.write_text(markdown(data), encoding="utf-8")
    print(f"Wrote {MD_OUT.relative_to(ROOT)} and {JSON_OUT.name}")


if __name__ == "__main__":
    main()
