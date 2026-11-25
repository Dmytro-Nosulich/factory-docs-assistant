"""Calibrate the quality judge against hand grades (S2-19). Rules: docs/METRICS.md.

`make calibration-answers` (--answers): answers every test question through the `make ask`
pipeline with the score gate off, so every question reaches the model, and saves the answers in
evals/results/calibration/answers.json. No judge is called. It also writes the blind grading
sheet evals/results/calibration/sheet.md and the grade file evals/judge_grades.toml with every
verdict empty. It refuses to overwrite either file, so grades always belong to their answers.
`make calibrate-judge`: stops unless every answer has exactly one valid hand grade. Then it
records the grade file's sha256 and runs the quality judge PASSES times on every saved answer,
and writes evals/results/calibration.json, calibration.md and the docs/reports/stage-2.md section.
  --report  rebuild calibration.md and the stage 2 report section from calibration.json, no models;
            stops if the grade file changed after the judge ran.

Judge calls are logged in results/judgements/ and count toward the stage 2 budget guard.
Usage: PYTHONPATH=. uv run python scripts/calibrate_judge.py [--answers | --report]
"""

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import eval_answers as ea  # scripts/eval_answers.py: the S2-18 run helpers

from evals.answer_eval import judgement_summary
from evals.calibration import (
    END,
    PASSES,
    START,
    grade_problems,
    grades_template,
    load_grades,
    markdown,
    report_section,
    result_row,
    sheet_markdown,
)
from evals.judges import judge_quality
from factory_docs.answer import JUDGEMENTS, BudgetExceeded, save_record
from factory_docs.clients import make_client
from factory_docs.config import ROOT, Config, load_config, load_env
from factory_docs.prompts import load_prompt
from factory_docs.retrieval_eval import update_section
from factory_docs.run import record_cost
from factory_docs.test_set import TEST_SET, load_test_set

CALIBRATION = ROOT / "evals" / "results" / "calibration"
ANSWERS = CALIBRATION / "answers.json"
SHEET = CALIBRATION / "sheet.md"
GRADES = ROOT / "evals" / "judge_grades.toml"
RESULTS = ROOT / "evals" / "results" / "calibration.json"
MD_OUT = ROOT / "evals" / "results" / "calibration.md"
REPORT = ROOT / "docs" / "reports" / "stage-2.md"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def quality_prompt(config: Config):
    return load_prompt(config.judge.quality_prompt, config.judge.quality_prompt_version)


def paths() -> dict:
    return {"answers": rel(ANSWERS), "sheet": rel(SHEET), "grades": rel(GRADES)}


def make_answers(config: Config) -> list[Path]:
    """Answer the test set with the gate off; write the answers, the sheet and the grade file."""
    for path in (ANSWERS, GRADES):
        if path.exists():
            raise SystemExit(
                f"{rel(path)} exists: the hand grades belong to the saved answers. Delete "
                f"{rel(ANSWERS)} and {rel(GRADES)} to answer again."
            )
    questions = load_test_set()
    load_env()
    spent = ea.check_guard(config)
    chunk_file = ea.chunks_path(config)
    answer_settings, rows = ea.answer_all(config, questions, spent, gate=False)
    run = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": {"platform": platform.platform()},
        "git": ea.git_state(),
        "purpose": "S2-19 judge calibration: answers for blind hand grading, score gate off",
        "test_set": {"path": rel(TEST_SET), "sha256": ea.sha256(TEST_SET)},
        "chunks": {"path": rel(chunk_file), "sha256": ea.sha256(chunk_file)},
        "settings": {"answer": {**answer_settings, "gate": False}},
        "questions": rows,
    }
    write_json(ANSWERS, run)
    by_id = {q.id: q for q in questions}
    SHEET.write_text(sheet_markdown(run, by_id, quality_prompt(config), paths()), encoding="utf-8")
    GRADES.write_text(grades_template(rows, rel(ANSWERS), rel(SHEET)), encoding="utf-8")
    return [ANSWERS, SHEET, GRADES]


def load_answers() -> dict:
    if not ANSWERS.exists():
        raise SystemExit(f"no {rel(ANSWERS)}; run `make calibration-answers`")
    run = json.loads(ANSWERS.read_text(encoding="utf-8"))
    for key, label in (("chunks", "the chunks"), ("test_set", "the test set")):
        if ea.sha256(ROOT / run[key]["path"]) != run[key]["sha256"]:
            raise SystemExit(
                f"{run[key]['path']} changed since {rel(ANSWERS)} was made: the judge would not "
                f"see {label} the answers were made from"
            )
    return run


def calibrate(config: Config) -> dict:
    """Check the hand grades, then judge every saved answer PASSES times."""
    run = load_answers()
    ids = [row["id"] for row in run["questions"]]
    if not GRADES.exists():
        raise SystemExit(f"no {rel(GRADES)}; run `make calibration-answers`")
    grades = load_grades(GRADES)
    problems = grade_problems(grades, ids)
    if problems:
        raise SystemExit(
            f"Grade every answer in {rel(GRADES)} first (no judge was called):\n  "
            + "\n  ".join(problems)
        )
    grades_sha = ea.sha256(GRADES)  # recorded before the first judge call
    questions = {q.id: q for q in load_test_set()}
    load_env()
    spent = ea.check_guard(config)
    judgements = {i: [] for i in ids}
    client = make_client(config.judge.model, config)
    try:
        for n in range(1, PASSES + 1):
            for row in run["questions"]:
                print(f"judge pass {n} {row['id']} ...", flush=True)
                try:
                    judgement = judge_quality(questions[row["id"]], row, client, config, spent)
                except BudgetExceeded as e:
                    raise SystemExit(f"Budget guard: {e}; no results written") from e
                spent += record_cost(judgement)
                path = save_record(judgement, JUDGEMENTS) if judgement["call"] else None
                judgements[row["id"]].append(
                    judgement_summary(judgement, rel(path) if path else None)
                )
    finally:
        client.close()
    by_grade = {g.id: g for g in grades}
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": {"platform": platform.platform()},
        "git": ea.git_state(),
        "answers": {"path": rel(ANSWERS), "sha256": ea.sha256(ANSWERS)},
        "grades": {"path": rel(GRADES), "sha256": grades_sha},
        "test_set": run["test_set"],
        "chunks": run["chunks"],
        "settings": {
            "answer": run["settings"]["answer"],
            "judge": ea.judge_settings(config),
            "passes": PASSES,
        },
        "questions": [
            result_row(row, questions[row["id"]], by_grade[row["id"]], judgements[row["id"]])
            for row in run["questions"]
        ],
    }


def load_results() -> dict:
    if not RESULTS.exists():
        raise SystemExit(f"no {rel(RESULTS)}; run `make calibrate-judge`")
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    if ea.sha256(GRADES) != result["grades"]["sha256"]:
        raise SystemExit(
            f"{rel(GRADES)} changed after the judge ran; run `make calibrate-judge` again"
        )
    return result


def write_outputs(result: dict) -> list[Path]:
    MD_OUT.write_text(markdown(result, rel(RESULTS)), encoding="utf-8")
    update_section(REPORT, report_section(result, rel(RESULTS)), START, END)
    return [MD_OUT, REPORT]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--answers", action="store_true", help="answer the test set, gate off")
    group.add_argument("--report", action="store_true", help="rebuild from calibration.json")
    args = parser.parse_args()
    if args.answers:
        written = make_answers(load_config())
        print("Grade every answer in the sheet, then run `make calibrate-judge`.")
    elif args.report:
        written = write_outputs(load_results())
    else:
        result = calibrate(load_config())
        write_json(RESULTS, result)
        written = [RESULTS, *write_outputs(result)]
    for path in written:
        print(f"Wrote {rel(path)}")


if __name__ == "__main__":
    main()
