"""Answer the `[run] sample_questions` and print them like `make ask` (S2-15).

The samples are development questions (evals/dev_questions.toml), answered through the same
pipeline as `make ask`: `[answer] collection`, hybrid + rerank, the score gate on, the context
budget, the answer model, and the citation checks. Each record goes to results/answers/ and
counts toward the stage 2 budget guard (`[answer] stop_at_usd`).
Usage: uv run python scripts/sample_answers.py
"""

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
from factory_docs.dev_questions import load_dev
from factory_docs.pipeline import sample_questions
from factory_docs.retriever import IndexNotReady, open_retriever
from factory_docs.run import record_cost


def main() -> None:
    config = load_config()
    settings = config.answer
    try:
        questions = sample_questions(config.run.sample_questions, load_dev())
    except ValueError as e:
        raise SystemExit(str(e)) from e
    load_env()
    spent = spent_on_answers()
    try:
        with open_retriever(config) as retriever:
            retrieval = retriever.info()
            hits = {q.id: retriever.search(q.question) for q in questions}
    except IndexNotReady as e:
        raise SystemExit(str(e)) from e

    failed = []
    client = make_client(settings.model, config)
    try:
        for q in questions:
            try:
                record = answer_question(q.question, hits[q.id], client, config, spent, retrieval)
            except BudgetExceeded as e:
                raise SystemExit(f"Budget guard: {e}") from e
            path = save_record(record)
            spent += record_cost(record)
            print(f"\n=== {q.id} ({q.language}, {q.kind.value}): {q.question}")
            print_record(record)
            print(f"Saved: {path.relative_to(ROOT)}")
            if record["final_status"] not in (*SHOWN, *NOT_CALLED):
                failed.append(q.id)
    finally:
        client.close()
    print(
        f"\n{len(questions)} sample questions; stage 2 answers so far ${spent:.4f} of "
        f"${settings.stop_at_usd:.2f}"
    )
    if failed:
        raise SystemExit(f"No answer could be shown for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
