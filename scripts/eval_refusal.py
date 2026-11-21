"""Measure "I don't know" on the development questions (S2-14).

Retrieves each question in evals/dev_questions.toml once, the way `make ask` does, answers it
with every `[refusal_eval] prompt_versions` version of the answer prompt with the score gate
off, and picks the gate threshold from the top-1 reranker scores by the rule in
`factory_docs.refusal_eval.pick_threshold` (docs/METRICS.md). Every answer record goes to
results/answers/ and counts toward the stage 2 budget guard (`[answer] stop_at_usd`).
Writes evals/results/refusal.json (every question's scores and outcomes) and
evals/results/refusal.md, and rewrites the generated refusal section of REPORT.md.
Usage: uv run python scripts/eval_refusal.py [--report]
  --report  rebuild the .md and the REPORT.md section from the saved JSON, no models
"""

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime

from factory_docs.answer import (
    BudgetExceeded,
    answer_question,
    fit_context,
    save_record,
    shown_answer,
    spent_on_answers,
)
from factory_docs.clients import make_client
from factory_docs.config import ROOT, Config, load_config, load_env
from factory_docs.dev_questions import DEV, DevQuestion, load_dev
from factory_docs.extract import prompt_info
from factory_docs.prompts import load_prompt
from factory_docs.refusal_eval import END, START, markdown, pick_threshold, report_section
from factory_docs.retrieval_eval import update_section
from factory_docs.retrieval_gold import is_hit
from factory_docs.retriever import IndexNotReady, open_retriever
from factory_docs.run import record_cost

OUT = ROOT / "evals" / "results"
JSON_OUT = OUT / "refusal.json"
MD_OUT = OUT / "refusal.md"
REPORT = ROOT / "REPORT.md"


def evidence_rank(question: DevQuestion, hits) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if any(is_hit(hit.chunk, evidence) for evidence in question.evidence):
            return rank
    return None


def run_row(question: DevQuestion, record: dict, path) -> dict:
    answer = shown_answer(record)
    calls = [c for c in (record["call"], (record.get("repair") or {}).get("call")) if c]
    claims = (answer or {}).get("claims", [])
    cited = sorted({c["chunk_id"].split("-")[0] for claim in claims for c in claim["citations"]})
    final_errors = (record.get("repair") or record)["errors"]
    return {
        "id": question.id,
        "final_status": record["final_status"],
        "answer": answer,
        "cited_docs": cited,
        "cost_usd": record_cost(record),
        "total_s": sum(c.get("total_s") or 0 for c in calls) if calls else None,
        "record": str(path.relative_to(ROOT)),
        "errors": [] if answer else final_errors,
    }


def run(config: Config) -> dict:
    questions = load_dev()
    settings = config.answer
    versions = config.refusal_eval.prompt_versions
    prompts = {v: prompt_info(load_prompt(settings.prompt, v)) for v in versions}
    load_env()
    spent = spent_on_answers()
    if spent >= settings.stop_at_usd:
        raise SystemExit(f"Budget guard: stage 2 answers have cost ${spent:.4f} already")

    try:
        with open_retriever(config) as retriever:
            retrieval = retriever.info()
            hits = {}
            for q in questions:
                print(f"search {q.id} ...", flush=True)
                hits[q.id] = retriever.search(q.question)
    except IndexNotReady as e:
        raise SystemExit(str(e)) from e

    question_rows = []
    for q in questions:
        sent, _ = fit_context(hits[q.id], settings.max_context_tokens)
        question_rows.append(
            {
                "id": q.id,
                "question": q.question,
                "language": q.language,
                "kind": q.kind.value,
                "evidence_docs": sorted({e.doc_id for e in q.evidence}),
                "top_score": hits[q.id][0].score if hits[q.id] else None,
                "evidence_rank": evidence_rank(q, hits[q.id]) if q.answerable else None,
                "evidence_sent": evidence_rank(q, sent) is not None if q.answerable else None,
                "hits": [[h.chunk.chunk_id, round(h.score, 4)] for h in hits[q.id]],
            }
        )

    runs: dict[str, list[dict]] = {}
    client = make_client(settings.model, config)
    try:
        for version in versions:
            runs[version] = []
            for q in questions:
                print(f"answer {version} {q.id} ...", flush=True)
                try:
                    record = answer_question(
                        q.question,
                        hits[q.id],
                        client,
                        config,
                        spent,
                        retrieval,
                        prompt_version=version,
                        gate=False,
                    )
                except BudgetExceeded as e:
                    raise SystemExit(f"Budget guard: {e}; nothing written") from e
                path = save_record(record)
                spent += record_cost(record)
                runs[version].append(run_row(q, record, path))
    finally:
        client.close()

    answerable = [r["top_score"] for r in question_rows if r["kind"] == "answerable"]
    unanswerable = [r["top_score"] for r in question_rows if r["kind"] != "answerable"]
    model = config.models[settings.model]
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": {"platform": platform.platform()},
        "dev": {
            "path": str(DEV.relative_to(ROOT)),
            "sha256": hashlib.sha256(DEV.read_bytes()).hexdigest(),
        },
        "settings": {
            "model": settings.model,
            "slug": model.slug,
            "prompts": prompts,
            "collection": retrieval["collection"],
            "mode": retrieval["mode"],
            "embedding_model": retrieval["embedding_model"],
            "reranker": retrieval["reranker"],
            "rerank_candidates": retrieval["rerank_candidates"],
            "k": settings.k,
            "max_context_tokens": settings.max_context_tokens,
        },
        "threshold": pick_threshold(answerable, unanswerable),
        "questions": question_rows,
        "runs": runs,
    }


def write_outputs(data: dict) -> None:
    MD_OUT.write_text(markdown(data), encoding="utf-8")
    update_section(REPORT, report_section(data), START, END)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="rebuild from the saved JSON")
    args = parser.parse_args()
    if args.report:
        data = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    else:
        data = run(load_config())
        OUT.mkdir(parents=True, exist_ok=True)
        JSON_OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_outputs(data)
    for path in (JSON_OUT, MD_OUT, REPORT):
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
