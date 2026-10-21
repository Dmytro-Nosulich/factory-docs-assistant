"""Extract one document with one model and save the result (S1-07 smoke run).

Usage: uv run python scripts/extract_one.py --doc D001 [--model haiku-5.5]
Writes results/smoke/<model>_<doc>.json. Makes one model call (retries included); cloud
models are paid.
"""

import argparse
import json
from pathlib import Path

from factory_docs.clients import make_client
from factory_docs.config import load_config, load_env
from factory_docs.extract import extract_document

SMOKE = Path(__file__).resolve().parent.parent / "results" / "smoke"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", required=True, help="document ID, e.g. D001")
    parser.add_argument("--model", default="haiku-5.5", help="model key from config.toml")
    args = parser.parse_args()

    load_env()
    config = load_config()
    client = make_client(args.model, config)
    try:
        record = extract_document(args.doc, client, args.model, config)
    finally:
        client.close()

    SMOKE.mkdir(parents=True, exist_ok=True)
    path = SMOKE / f"{args.model}_{args.doc}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    call, inp = record["call"], record["input"]
    print(f"{args.doc} {args.model}: {record['status']}")
    print(f"  provider: {record['model'].get('provider')}, attempts: {call['attempts']}")
    if record["status"] != "error":
        print(
            f"  tokens in/out/reasoning: {call['prompt_tokens']}/{call['completion_tokens']}"
            f"/{call['reasoning_tokens']}, cost: ${call['cost_usd']}"
        )
        print(
            f"  latency: {call['latency_s']} s (total {call['total_s']} s), "
            f"finish: {call['finish_reason']}"
        )
    print(
        f"  pages sent: {inp['pages_sent']} of {inp['pages_total']}"
        + (" (truncated)" if inp["truncated"] else "")
    )
    for error in record["errors"]:
        print(f"  problem: {error}")
    print(f"  saved: {path.relative_to(SMOKE.parent.parent)}")


if __name__ == "__main__":
    main()
