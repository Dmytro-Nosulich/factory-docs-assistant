"""Check the retrieval gold set evals/retrieval_gold.toml (S2-09).

For every piece of evidence: the document is in the RAG index, the quote is in its cleaned text
(on the given page for a PDF), and at least one chunk of every collection in [index] contains
it, so search can find it. Prints the chunks that hold each quote.

Usage: uv run python scripts/check_retrieval_gold.py
Exits with status 1 if the file has a problem.
"""

import sys
from collections import Counter

from factory_docs.config import ROOT, load_config
from factory_docs.index import load_chunks
from factory_docs.retrieval_gold import GOLD, check_evidence, load_documents, load_gold


def main() -> None:
    try:
        questions = load_gold()
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

    languages = Counter(q.language for q in questions)
    evidence_count = sum(len(q.evidence) for q in questions)
    print(
        f"\n{len(questions)} questions, {evidence_count} pieces of evidence, languages: "
        + ", ".join(f"{lang} {n}" for lang, n in languages.most_common())
    )
    print(f"{evidence_count - failed} of {evidence_count} OK ({GOLD.relative_to(ROOT)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
