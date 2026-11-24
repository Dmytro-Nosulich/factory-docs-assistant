"""Answer the test set, score it and judge it (S2-18).

`make eval`: retrieves and answers every question of evals/test_set.toml through the `make ask`
pipeline (`[answer]` settings, score gate on, no filter) and saves the answers in a new run file,
evals/results/runs/<UTC time>.json. Then it scores them with the S2-17 deterministic scorers and
runs the two LLM judges of `evals/judges.py` on the saved answers (`[judge]` settings), and
writes the run file again with every score, judgement and cost.
`make judge RUN=<run file>`: only the judges, on the answers saved in an earlier run, written as
a new run file that names the run its answers come from. No answer is made again.
Both write evals/results/eval.md and the generated answer section of REPORT.md.

Every answer record goes to results/answers/ and every judge call to results/judgements/; both
count toward the stage 2 budget guard (`[answer] stop_at_usd`).
Usage: PYTHONPATH=. uv run python scripts/eval_answers.py [--judge RUN_FILE | --report]
  --report  rebuild eval.md and the REPORT.md section from the latest run file, no models
PYTHONPATH=. makes the `evals` package importable (it is not installed; pytest sets the same
path in pyproject.toml). The make targets set it.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from evals.answer_eval import (
    END,
    START,
    answer_row,
    judgement_summary,
    markdown,
    report_section,
    score_dict,
)
from evals.judges import FAITHFULNESS, QUALITY, judge_faithfulness, judge_quality
from evals.scorers import score_question
from factory_docs.answer import (
    JUDGEMENTS,
    BudgetExceeded,
    answer_question,
    save_record,
    spent_on_answers,
)
from factory_docs.clients import make_client
from factory_docs.config import ROOT, Config, load_config, load_env
from factory_docs.extract import prompt_info
from factory_docs.index import load_chunks
from factory_docs.prompts import load_prompt
from factory_docs.retrieval_eval import update_section
from factory_docs.retriever import IndexNotReady, open_retriever
from factory_docs.run import record_cost
from factory_docs.test_set import TEST_SET, TestQuestion, load_test_set

OUT = ROOT / "evals" / "results"
RUNS = OUT / "runs"
MD_OUT = OUT / "eval.md"
REPORT = ROOT / "REPORT.md"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def chunks_path(config: Config) -> Path:
    return ROOT / config.index.collections[config.answer.collection]


def judge_settings(config: Config) -> dict:
    settings = config.judge
    return {
        "model": settings.model,
        "slug": config.models[settings.model].slug,
        "prompts": {
            QUALITY: prompt_info(
                load_prompt(settings.quality_prompt, settings.quality_prompt_version)
            ),
            FAITHFULNESS: prompt_info(
                load_prompt(settings.faithfulness_prompt, settings.faithfulness_prompt_version)
            ),
        },
    }


def check_guard(config: Config) -> float:
    spent = spent_on_answers()
    if spent >= config.answer.stop_at_usd:
        raise SystemExit(f"Budget guard: stage 2 logs have cost ${spent:.4f} already")
    return spent


def new_run_path() -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    path, n = RUNS / f"{stamp}.json", 1
    while path.exists():
        n += 1
        path = RUNS / f"{stamp}_{n}.json"
    return path


def write_run(path: Path, run: dict) -> None:
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def answer_all(
    config: Config, questions: list[TestQuestion], spent: float, gate: bool = True
) -> tuple[dict, list]:
    """Retrieve and answer every question like `make ask`; (answer settings, rows).
    gate=False answers every question, whatever its top score (S2-19 calibration answers)."""
    settings = config.answer
    try:
        with open_retriever(config) as retriever:
            retrieval = retriever.info()
            hits = {}
            for q in questions:
                print(f"search {q.id} ...", flush=True)
                hits[q.id] = retriever.search(q.question)
    except IndexNotReady as e:
        raise SystemExit(str(e)) from e

    rows = []
    client = make_client(settings.model, config)
    try:
        for q in questions:
            print(f"answer {q.id} ...", flush=True)
            try:
                record = answer_question(
                    q.question, hits[q.id], client, config, spent, retrieval, gate=gate
                )
            except BudgetExceeded as e:
                raise SystemExit(f"Budget guard: {e}") from e
            path = save_record(record)
            spent += record_cost(record)
            rows.append(answer_row(q, record, rel(path)))
    finally:
        client.close()
    model = config.models[settings.model]
    answer_settings = {
        "model": settings.model,
        "slug": model.slug,
        "prompt": prompt_info(load_prompt(settings.prompt, settings.prompt_version)),
        "repair_prompt": prompt_info(
            load_prompt(settings.repair_prompt, settings.repair_prompt_version)
        ),
        "collection": retrieval["collection"],
        "mode": retrieval["mode"],
        "embedding_model": retrieval["embedding_model"],
        "reranker": retrieval["reranker"],
        "rerank_candidates": retrieval["rerank_candidates"],
        "k": settings.k,
        "max_context_tokens": settings.max_context_tokens,
        "min_rerank_score": settings.min_rerank_score if gate else None,
    }
    return answer_settings, rows


def judge_all(config: Config, run: dict, questions: list[TestQuestion], chunks: dict) -> None:
    """Score every saved answer and run both judges on it; fills the rows in place."""
    by_id = {q.id: q for q in questions}
    spent = check_guard(config)
    client = make_client(config.judge.model, config)
    try:
        for row in run["questions"]:
            q = by_id[row["id"]]
            row["score"] = score_dict(score_question(q, row, chunks))
            print(f"judge {q.id} ...", flush=True)
            try:
                quality = judge_quality(q, row, client, config, spent)
                spent += record_cost(quality)
                faithfulness = judge_faithfulness(
                    q.id, row, row["sent"], chunks, client, config, spent
                )
            except BudgetExceeded as e:
                raise SystemExit(f"Budget guard: {e}; run file left without judgements") from e
            spent += record_cost(faithfulness)
            for key, judgement in ((QUALITY, quality), (FAITHFULNESS, faithfulness)):
                path = save_record(judgement, JUDGEMENTS) if judgement["call"] else None
                row[key] = judgement_summary(judgement, rel(path) if path else None)
    finally:
        client.close()
    run["settings"]["judge"] = judge_settings(config)


def run_eval(config: Config) -> tuple[Path, dict]:
    questions = load_test_set()
    load_env()
    spent = check_guard(config)
    chunk_file = chunks_path(config)
    chunks = {c.chunk_id: c for c in load_chunks(chunk_file)}
    answer_settings, rows = answer_all(config, questions, spent)
    path = new_run_path()
    run = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": {"platform": platform.platform()},
        "git": git_state(),
        "answers_from": None,
        "test_set": {"path": rel(TEST_SET), "sha256": sha256(TEST_SET)},
        "chunks": {"path": rel(chunk_file), "sha256": sha256(chunk_file)},
        "settings": {"answer": answer_settings, "judge": None},
        "questions": rows,
    }
    write_run(path, run)  # the answers are saved before any judge call
    print(f"Saved answers: {rel(path)}")
    judge_all(config, run, questions, chunks)
    write_run(path, run)
    return path, run


def run_judges(config: Config, source: Path) -> tuple[Path, dict]:
    """The judges only, on the answers of a saved run, written as a new run file."""
    saved = json.loads(source.read_text(encoding="utf-8"))
    chunk_file = ROOT / saved["chunks"]["path"]
    if sha256(chunk_file) != saved["chunks"]["sha256"]:
        raise SystemExit(
            f"{saved['chunks']['path']} changed since {rel(source)}: the judges would not see "
            "the chunks the answers were made from; run `make eval`"
        )
    if sha256(TEST_SET) != saved["test_set"]["sha256"]:
        raise SystemExit(
            f"{rel(TEST_SET)} changed since {rel(source)}: the scores and reference answers "
            "would not match the answers; run `make eval`"
        )
    questions = load_test_set()
    load_env()
    chunks = {c.chunk_id: c for c in load_chunks(chunk_file)}
    keep = {k: v for k, v in saved.items() if k not in ("created_at", "machine", "git")}
    run = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": {"platform": platform.platform()},
        "git": git_state(),
        **keep,
        "answers_from": saved["answers_from"] or rel(source),
        "settings": {**saved["settings"], "judge": None},
        "questions": [
            {k: v for k, v in row.items() if k not in ("score", QUALITY, FAITHFULNESS)}
            for row in saved["questions"]
        ],
    }
    judge_all(config, run, questions, chunks)
    path = new_run_path()
    write_run(path, run)
    return path, run


def latest_run() -> Path:
    runs = sorted(RUNS.glob("*.json"))
    if not runs:
        raise SystemExit(f"no run file in {rel(RUNS)}; run `make eval`")
    return runs[-1]


def write_outputs(path: Path, run: dict) -> None:
    MD_OUT.write_text(markdown(run, rel(path)), encoding="utf-8")
    update_section(REPORT, report_section(run, rel(path)), START, END)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--judge", type=Path, metavar="RUN_FILE", help="judge a saved run again")
    group.add_argument("--report", action="store_true", help="rebuild from the latest run file")
    args = parser.parse_args()
    if args.report:
        path = latest_run()
        run = json.loads(path.read_text(encoding="utf-8"))
    elif args.judge:
        source = args.judge if args.judge.is_absolute() else Path.cwd() / args.judge
        if not source.exists():
            raise SystemExit(f"no run file {args.judge}")
        path, run = run_judges(load_config(), source.resolve())
    else:
        path, run = run_eval(load_config())
    write_outputs(path, run)
    for written in (path, MD_OUT, REPORT):
        print(f"Wrote {rel(written)}")


if __name__ == "__main__":
    main()
