"""Check the test set evals/test_set.toml (S2-16).

The file validates (fields match each category, IDs are unique, no question repeats one of
evals/dev_questions.toml or evals/retrieval_gold.toml), has at least 20 questions and every
category. Every question's labels are grounded in the corpus: evidence passes the gold-set checks,
key facts are in an evidence quote, forbidden facts are in a wrong document, an outdated version is
superseded by the evidence (data/manifest.csv), and absent terms are in no chunk.

Usage: uv run python scripts/check_test_set.py
Exits with status 1 if the file has a problem.
"""

import sys
from collections import Counter

from factory_docs.config import ROOT, load_config
from factory_docs.corpus import in_scope
from factory_docs.index import load_chunks
from factory_docs.retrieval_gold import load_documents
from factory_docs.test_set import (
    TEST_SET,
    check_question,
    coverage_problems,
    load_test_set,
    superseded_by,
)


def main() -> None:
    try:
        questions = load_test_set()
    except ValueError as error:
        print(error)
        sys.exit(1)

    config = load_config()
    collections = {
        name: load_chunks(ROOT / path) for name, path in config.index.collections.items()
    }
    indexed = {chunk.doc_id for chunks in collections.values() for chunk in chunks}
    wanted = {e.doc_id for q in questions for e in q.evidence} | {
        d for q in questions for d in q.wrong_docs
    }
    documents = load_documents(wanted & indexed)
    newer = superseded_by(in_scope())

    failed = 0
    for question in questions:
        check = check_question(question, documents, collections, newer)
        head = f"{question.id} {question.category.value} ({question.language})"
        sources = ", ".join(e.doc_id for e in question.evidence) or "no evidence"
        if question.wrong_docs:
            sources += f"; wrong: {', '.join(question.wrong_docs)}"
        if check.ok:
            print(f"{head}: OK  [{sources}]")
            continue
        failed += 1
        print(f"{head}: problems  [{sources}]")
        for evidence in check.evidence:
            for problem in evidence.problems:
                print(f"  evidence {evidence.number} ({evidence.evidence.doc_id}): {problem}")
        for problem in check.problems:
            print(f"  {problem}")

    coverage = coverage_problems(questions)
    for problem in coverage:
        print(f"set: {problem}")

    categories = Counter(q.category.value for q in questions)
    languages = Counter(q.language for q in questions)
    print(
        f"\n{len(questions)} questions: "
        + ", ".join(f"{c} {n}" for c, n in sorted(categories.items()))
        + "; languages: "
        + ", ".join(f"{lang} {n}" for lang, n in languages.most_common())
    )
    print(
        f"{len(questions) - failed} of {len(questions)} questions OK ({TEST_SET.relative_to(ROOT)})"
    )
    sys.exit(1 if failed or coverage else 0)


if __name__ == "__main__":
    main()
