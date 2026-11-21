"""Check the development questions evals/dev_questions.toml (S2-14).

The file validates (answerable questions have evidence, the others have none, IDs are unique,
no question repeats one of evals/retrieval_gold.toml). Every piece of evidence passes the same
checks as the retrieval gold set: the document is in the RAG index, the quote is in its cleaned
text (on the given page for a PDF), and a chunk of every collection in [index] contains it.

Usage: uv run python scripts/check_dev_questions.py
Exits with status 1 if the file has a problem.
"""

import sys
from collections import Counter

from factory_docs.config import ROOT, load_config
from factory_docs.dev_questions import DEV, load_dev
from factory_docs.index import load_chunks
from factory_docs.retrieval_gold import check_evidence, load_documents


def main() -> None:
    try:
        questions = load_dev()
    except ValueError as error:
        print(error)
        sys.exit(1)

    config = load_config()
    collections = {
        name: load_chunks(ROOT / path) for name, path in config.index.collections.items()
    }
    indexed = {chunk.doc_id for chunks in collections.values() for chunk in chunks}
    wanted = {evidence.doc_id for q in questions for evidence in q.evidence}
    documents = load_documents(wanted & indexed)

    failed = 0
    for question in questions:
        if not question.answerable:
            print(f"{question.id} {question.kind.value}: no evidence")
            continue
        for check in check_evidence(question, documents, collections):
            label = f"{check.question_id} evidence {check.number} ({check.evidence.doc_id})"
            if check.problems:
                failed += 1
                print(f"{label}: {len(check.problems)} problem(s)")
                for problem in check.problems:
                    print(f"  {problem}")
            else:
                found = "; ".join(f"{name}: {', '.join(ids)}" for name, ids in check.hits.items())
                print(f"{label}: OK  [{found}]")

    kinds = Counter(q.kind.value for q in questions)
    languages = Counter(q.language for q in questions)
    evidence_count = sum(len(q.evidence) for q in questions)
    print(
        f"\n{len(questions)} questions: "
        + ", ".join(f"{kind} {n}" for kind, n in sorted(kinds.items()))
        + "; languages: "
        + ", ".join(f"{lang} {n}" for lang, n in languages.most_common())
    )
    print(f"{evidence_count - failed} of {evidence_count} evidence OK ({DEV.relative_to(ROOT)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
