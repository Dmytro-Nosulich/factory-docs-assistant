"""Check a cloud model's tool support and price on OpenRouter (S3-02).

Usage: uv run python scripts/model_info.py [--model haiku-5.5]
Calls the public `GET /api/v1/models` (no key, no cost), finds the model's slug and prints
its supported parameters, context length and price per 1M tokens next to the `price_*`
values in config.toml. Exits non-zero when the model does not list `tools` and `tool_choice`.
"""

import argparse

import httpx

from factory_docs.config import OpenRouterModel, load_config

NEEDED = ("tools", "tool_choice")


def per_mtok(per_token: str | None) -> float | None:
    """OpenRouter gives USD per token as a string; config.toml keeps USD per 1M tokens."""
    return None if per_token is None else round(float(per_token) * 1_000_000, 6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="haiku-5.5", help="model key from config.toml")
    args = parser.parse_args()

    config = load_config()
    model = config.models.get(args.model)
    if not isinstance(model, OpenRouterModel):
        parser.error(f"{args.model!r} is not an OpenRouter model in config.toml")
    url = f"{config.openrouter.base_url}/models"
    response = httpx.get(url, timeout=config.openrouter.timeout_s)
    response.raise_for_status()
    entry = next((m for m in response.json()["data"] if m["id"] == model.slug), None)
    if entry is None:
        raise SystemExit(f"{model.slug} is not listed on {url}")

    supported = entry.get("supported_parameters") or []
    pricing = entry.get("pricing") or {}
    price_in, price_out = per_mtok(pricing.get("prompt")), per_mtok(pricing.get("completion"))
    print(f"{args.model} = {model.slug} ({url})")
    print(f"  context length: {entry.get('context_length')}")
    print(f"  supported parameters: {', '.join(supported)}")
    print(
        f"  price per 1M tokens: in ${price_in}, out ${price_out} "
        f"(config.toml: in ${model.price_in_usd_per_mtok}, out ${model.price_out_usd_per_mtok}, "
        f"checked {model.price_checked})"
    )
    print(f"  pricing as listed: {pricing}")
    if (price_in, price_out) != (model.price_in_usd_per_mtok, model.price_out_usd_per_mtok):
        print("  WARNING: the listed price differs from config.toml")
    missing = [p for p in NEEDED if p not in supported]
    if missing:
        raise SystemExit(f"{model.slug} does not list {', '.join(missing)}: no tool calling")
    print(f"  tool calling: yes ({', '.join(NEEDED)} listed)")


if __name__ == "__main__":
    main()
