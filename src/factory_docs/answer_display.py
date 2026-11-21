"""Print an answer record the way `make ask` shows it (S2-13; shared with `make samples`, S2-15)."""

import textwrap

from factory_docs.answer import shown_answer
from factory_docs.run import record_cost


def source_line(source: dict) -> str:
    page = f", p. {source['pages']}" if source["pages"] else ""
    return f"{source['doc_id']} {source['title']}: {source['section']}{page}"


def print_record(record: dict) -> None:
    retrieval = record["retrieval"]
    sources = {s["chunk_id"]: s for s in retrieval["sent"]}
    if record["final_status"] == "below_threshold":
        print(
            f"Not in the documents: the best of {len(retrieval['hits'])} chunks scores "
            f"{retrieval['top_score']:.2f}, below the threshold {retrieval['min_rerank_score']} "
            "([answer] min_rerank_score); no model was called."
        )
        return
    print(
        f"Context: {len(retrieval['sent'])} chunks sent, {len(retrieval['dropped'])} dropped "
        f"by the budget, about {retrieval['est_context_tokens']} of "
        f"{retrieval['max_context_tokens']} tokens (estimate)\n"
    )
    answer = shown_answer(record)
    if answer is None:
        print(f"No answer: {record['final_status']}")
        for error in (record.get("repair") or record)["errors"]:
            print(f"  - {error}")
    else:
        print(f"Status: {answer['status']}")
        for number, claim in enumerate(answer["claims"], start=1):
            print(
                textwrap.fill(
                    claim["text"], 96, initial_indent=f"{number}. ", subsequent_indent="   "
                )
            )
            for citation in claim["citations"]:
                print(f"   [{citation['chunk_id']}] {source_line(sources[citation['chunk_id']])}")
                print(f'     "{" ".join(citation["quote"].split())}"')
        if answer["missing"]:
            print(textwrap.fill(f"Missing: {answer['missing']}", 96))
        if record["final_status"] == "repaired":
            print("\n(The first answer failed the checks and was repaired:)")
            for error in record["errors"]:
                print(f"  - {error}")
    calls = [c for c in (record["call"], (record["repair"] or {}).get("call")) if c]
    tokens_in = sum(c.get("prompt_tokens") or 0 for c in calls)
    tokens_out = sum(c.get("completion_tokens") or 0 for c in calls)
    latency = sum(c.get("total_s") or 0 for c in calls)
    model = record["model"]
    prompt = record["prompt"]
    print(
        f"\n{model['key']} ({model.get('provider') or '-'}), prompt {prompt['name']} "
        f"{prompt['version']}, {len(calls)} call(s), {tokens_in} prompt / {tokens_out} output "
        f"tokens, {latency:.1f} s, ${record_cost(record):.5f}"
    )
