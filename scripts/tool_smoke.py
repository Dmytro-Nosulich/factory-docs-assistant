"""Live smoke test of tool calling: Haiku 5.5 asks for one tool, then answers (S3-02).

Usage: uv run python scripts/tool_smoke.py [--model haiku-5.5]
Two paid calls (well under $0.001). Call 1 sends a question and one toy tool,
`lookup_flash_point`; the model must ask for it exactly once, with JSON arguments. Call 2
sends back a canned tool result in the OpenAI message format the agent loop (S3-04) will use,
and the model must answer in text. The toy tool is not part of the agent: the real tools come
in S3-03. Writes results/agent/smoke/tool_call.json (gitignored; stage 3 budget, decision 29).
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from factory_docs.clients import make_client
from factory_docs.config import load_config, load_env
from factory_docs.openrouter import OpenRouterClient

OUT = Path(__file__).resolve().parent.parent / "results" / "agent" / "smoke" / "tool_call.json"
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_flash_point",
        "description": "Return the flash point of a product from its safety data sheet.",
        "parameters": {
            "type": "object",
            "properties": {"product_name": {"type": "string", "description": "trade name"}},
            "required": ["product_name"],
            "additionalProperties": False,
        },
    },
}
MESSAGES = [
    {"role": "system", "content": "Answer from tool results only. Use the tools you are given."},
    {"role": "user", "content": "What is the flash point of the product TEST-SOLVENT-1?"},
]
# A made-up product, so the canned value is never mistaken for data from a real SDS
CANNED_RESULT = json.dumps(
    {"product_name": "TEST-SOLVENT-1", "flash_point": {"min": 62, "max": None, "unit": "celsius"}}
)


def show(label: str, turn) -> None:
    r = turn.response
    print(
        f"{label}: finish {r.finish_reason}, provider {r.provider}, attempts {r.attempts}, "
        f"tokens {r.prompt_tokens}/{r.completion_tokens}, cost ${r.cost_usd}, {r.latency_s} s"
    )
    for call in turn.tool_calls:
        print(f"  tool call {call.id}: {call.name}({call.arguments})")
    if r.content:
        print(f"  text: {r.content}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="haiku-5.5", help="OpenRouter model key")
    args = parser.parse_args()

    load_env()
    client = make_client(args.model, load_config())
    if not isinstance(client, OpenRouterClient):
        parser.error(f"{args.model!r} is not an OpenRouter model")
    try:
        first = client.chat_with_tools(MESSAGES, [TOOL])
        show("call 1", first)
        problems = []
        if len(first.tool_calls) != 1 or first.tool_calls[0].name != "lookup_flash_point":
            problems.append(f"expected one lookup_flash_point call, got {first.tool_calls}")
        else:
            try:
                json.loads(first.tool_calls[0].arguments)
            except json.JSONDecodeError as e:
                problems.append(f"tool arguments are not JSON: {e}")
        second = None
        if not problems:
            result = {
                "role": "tool",
                "tool_call_id": first.tool_calls[0].id,
                "content": CANNED_RESULT,
            }
            second = client.chat_with_tools([*MESSAGES, first.message, result], [TOOL])
            show("call 2", second)
            if second.tool_calls or not second.response.content:
                problems.append("expected a text answer after the tool result")
    finally:
        client.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model_key": args.model,
        "slug": client.model.slug,
        "calls": [asdict(t) for t in (first, second) if t is not None],
        "problems": problems,
    }
    OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cost = sum(t.response.cost_usd or 0.0 for t in (first, second) if t is not None)
    print(f"total cost ${cost:.6f}; saved {OUT.relative_to(OUT.parents[3])}")
    if problems:
        raise SystemExit("smoke test failed: " + "; ".join(problems))
    print("smoke test OK: one tool call, then a text answer")


if __name__ == "__main__":
    main()
