"""Compare the multilingual FastEmbed models on size, speed and truncation (S2-06).

For every model in [embedding] compare in config.toml, in its own process (so load time and
peak memory are its own): download it once, then load it from the cache, count each structure
chunk's tokens, embed all structure chunks, time query embedding, and list its top 5 for the
two smoke queries. The smoke queries are a by-eye check that cross-lingual search works, not a
quality metric (S2-10 measures quality on the hand-written gold set).
Writes data/index/embedding_comparison.json (raw numbers) and embedding_comparison.md.
Usage: uv run python scripts/compare_embeddings.py [--models key ...]
"""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from factory_docs.config import ROOT, load_config
from factory_docs.embedding import FastEmbedder
from factory_docs.index import load_chunks
from factory_docs.metrics import percentile

OUT = ROOT / "data" / "index"
JSON_OUT = OUT / "embedding_comparison.json"
MD_OUT = OUT / "embedding_comparison.md"
TOP = 5
QUERY_REPEATS = 20
# Approved with the S2-06 plan (2026-11-19); written by Claude, not gold labels
SMOKE_QUERIES = {
    "uk": "Що робити, якщо засіб потрапив в очі?",
    "de": "Wie hoch ist der Flammpunkt von RM 110?",
}
LANGUAGES = ["uk", "de"]  # truncation shares shown per language, besides the total


def model_size_bytes(embedder: FastEmbedder) -> int:
    """Files of the model in the cache; stat follows the Hugging Face cache symlinks."""
    folder = Path(embedder.model.model._model_dir)
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # Linux reports KiB


def measure(key: str) -> dict:
    """Every number for one model; run in a fresh process after the model is downloaded."""
    config = load_config()
    chunks = load_chunks(ROOT / config.index.collections["structure"])
    texts = [chunk.text for chunk in chunks]

    start = time.perf_counter()
    embedder = FastEmbedder.from_config(config, key)
    load_s = time.perf_counter() - start

    lengths = embedder.token_lengths(texts)
    limit = embedder.token_limit
    over: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for chunk, length in zip(chunks, lengths, strict=True):
        for group in ("all", chunk.language):
            over[group][1] += 1
            over[group][0] += limit is not None and length > limit

    embedder.embed_query("warm-up")  # the first call pays one-off ONNX set-up
    start = time.perf_counter()
    vectors = np.array(embedder.embed_passages(texts))
    embed_s = time.perf_counter() - start

    query_ms = []
    for _ in range(QUERY_REPEATS):
        start = time.perf_counter()
        embedder.embed_query(SMOKE_QUERIES["uk"])
        query_ms.append((time.perf_counter() - start) * 1000)

    norms = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    smoke = {}
    for language, query in SMOKE_QUERIES.items():
        q = np.array(embedder.embed_query(query))
        scores = norms @ (q / np.linalg.norm(q))
        order = np.argsort(-scores, kind="stable")[:TOP]
        smoke[language] = [
            {
                "chunk_id": chunks[i].chunk_id,
                "language": chunks[i].language,
                "section": chunks[i].section,
                "score": round(float(scores[i]), 4),
            }
            for i in order
        ]

    return {
        "key": key,
        "model": embedder.model_name,
        "dimension": embedder.dimension,
        "size_bytes": model_size_bytes(embedder),
        "token_limit": limit,
        "tokens_p50": percentile(lengths, 50),
        "tokens_p95": percentile(lengths, 95),
        "tokens_max": max(lengths),
        "over_limit": {group: counts for group, counts in sorted(over.items())},
        "chunks": len(chunks),
        "load_s": round(load_s, 2),
        "embed_s": round(embed_s, 1),
        "query_ms_p50": round(percentile(query_ms, 50), 1),
        "peak_rss_bytes": peak_rss_bytes(),
        "smoke": smoke,
    }


def run_in_subprocess(*args: str) -> str:
    command = [sys.executable, __file__, *args]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def machine() -> dict:
    import onnxruntime

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "fastembed": version("fastembed"),
        "onnxruntime": version("onnxruntime"),
        "onnx_providers": onnxruntime.get_available_providers(),
    }


def licences() -> dict[str, str]:
    from fastembed import TextEmbedding

    return {m["model"]: m["license"] for m in TextEmbedding.list_supported_models()}


def share(counts: list[int]) -> str:
    part, whole = counts
    return f"{100 * part / whole:.0f} %" if whole else "-"


def amount(value: float, unit: str) -> str:
    """Whole numbers, but one decimal under 10 so a fast model does not show as 0."""
    return f"{value:.1f} {unit}" if value < 10 else f"{value:.0f} {unit}"


def markdown(data: dict) -> str:
    m = data["machine"]
    lines = [
        "# Embedding model comparison (S2-06)",
        "",
        "Generated by `make compare-embeddings`; do not edit by hand.",
        "",
        f"Machine: {m['platform']}, {m['cpu_count']} cores; FastEmbed {m['fastembed']}, "
        f"ONNX Runtime {m['onnxruntime']} on CPU. Input: the {data['chunks']} structure chunks "
        "of `data/chunks/chunks.jsonl`.",
        "",
        "Columns: *Disk* is the model files in the cache. *Limit* is the input length in tokens "
        "FastEmbed truncates to; text past it is ignored without a warning. *Over limit* is the "
        "share of chunks longer than that, counted with the model's own tokenizer and its "
        "passage prefix. *Load* is loading from the cache, *Embed* is all chunks one after "
        "another, *Query* is the median time to embed one query, *RAM* is the process's peak.",
        "",
        "| Model | Licence | Dim | Disk | Limit | Tokens p50 / p95 / max | Over limit: all / "
        + " / ".join(LANGUAGES)
        + " | Load | Embed | Per chunk | Query | RAM |",
        "|---" * 12 + "|",
    ]
    for r in data["results"]:
        over = r["over_limit"]
        lines.append(
            f"| `{r['model']}` | {r['licence']} | {r['dimension']} | "
            f"{r['size_bytes'] / 1e9:.2f} GB | {r['token_limit']} | "
            f"{r['tokens_p50']} / {r['tokens_p95']} / {r['tokens_max']} | "
            + " / ".join(share(over.get(g, [0, 0])) for g in ["all", *LANGUAGES])
            + f" | {r['load_s']:.1f} s | {amount(r['embed_s'], 's')} | "
            f"{amount(1000 * r['embed_s'] / r['chunks'], 'ms')} | "
            f"{amount(r['query_ms_p50'], 'ms')} | {r['peak_rss_bytes'] / 1e9:.1f} GB |"
        )
    lines += [
        "",
        "## Smoke queries",
        "",
        "Top 5 structure chunks per model, by cosine similarity. A by-eye check that a "
        "Ukrainian and a German question reach the right kind of section, not a quality "
        "metric; search quality is measured in S2-10 on the hand-written gold set.",
        "",
    ]
    for language, query in SMOKE_QUERIES.items():
        lines += [f"### {language}: {query}", ""]
        for r in data["results"]:
            lines += [
                f"**{r['key']}**",
                "",
                "| # | Chunk | Lang | Section | Score |",
                "|---" * 5 + "|",
            ]
            for rank, hit in enumerate(r["smoke"][language], start=1):
                section = hit["section"].replace("|", "/")
                section = section if len(section) <= 70 else section[:67] + "..."
                lines.append(
                    f"| {rank} | {hit['chunk_id']} | {hit['language']} | {section} | "
                    f"{hit['score']:.3f} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="keys in [embedding_models]; default: compare")
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
        FastEmbedder.from_config(load_config(), args.download)
        return
    if args.measure:
        print(json.dumps(measure(args.measure)))
        return

    config = load_config()
    keys = args.models or config.embedding.compare
    names = licences()
    results = []
    for key in keys:
        print(f"{key}: downloading if needed ...", flush=True)
        run_in_subprocess("--download", key)
        print(f"{key}: measuring ...", flush=True)
        result = json.loads(run_in_subprocess("--measure", key).strip().splitlines()[-1])
        result["licence"] = names.get(result["model"], "?")
        print(
            f"{key}: embed {result['embed_s']} s, over limit {share(result['over_limit']['all'])}",
            flush=True,
        )
        results.append(result)
    data = {"machine": machine(), "chunks": results[0]["chunks"], "results": results}
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    MD_OUT.write_text(markdown(data), encoding="utf-8")
    print(f"Wrote {MD_OUT.relative_to(ROOT)} and {JSON_OUT.name}")


if __name__ == "__main__":
    main()
