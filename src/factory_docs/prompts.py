"""Versioned prompt files (prompts/<name>_<version>.md), S1-06.

A prompt file has an optional header (e.g. an HTML comment, not sent to the model), then
exactly two sections under the headings `## system` and `## user`. Placeholders are written
`{{name}}` and filled by `Prompt.render`. The hash of the whole file is logged with every
result, so an edit that was not given a new version still shows.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
SECTION = re.compile(r"^## (\w+)[ \t]*$", re.MULTILINE)
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str
    user: str
    sha256: str  # first 12 hex characters of the file's hash

    @property
    def placeholders(self) -> set[str]:
        return set(PLACEHOLDER.findall(self.system + self.user))

    def render(self, **values: str) -> list[dict[str, str]]:
        """Chat messages with every `{{name}}` filled, e.g. render(document=..., schema=...)."""
        missing = self.placeholders - values.keys()
        unknown = values.keys() - self.placeholders
        if missing or unknown:
            raise ValueError(
                f"missing values: {sorted(missing)}; unknown values: {sorted(unknown)}"
            )

        # One pass, so a `{{...}}` inside a filled-in value is left as it is.
        def fill(template: str) -> str:
            return PLACEHOLDER.sub(lambda m: values[m.group(1)], template)

        return [
            {"role": "system", "content": fill(self.system)},
            {"role": "user", "content": fill(self.user)},
        ]


def parse_sections(text: str) -> dict[str, str]:
    """The `## system` and `## user` section bodies; anything before the first one is dropped."""
    parts = SECTION.split(text)
    names = parts[1::2]
    if sorted(names) != ["system", "user"]:
        raise ValueError(f"expected one '## system' and one '## user' section, found {names}")
    return dict(zip(names, (body.strip() for body in parts[2::2]), strict=True))


def load_prompt(name: str, version: str, prompts_dir: Path = PROMPTS) -> Prompt:
    path = prompts_dir / f"{name}_{version}.md"
    if not path.exists():
        available = sorted(
            p.stem.removeprefix(f"{name}_") for p in prompts_dir.glob(f"{name}_*.md")
        )
        raise FileNotFoundError(f"no prompt {path.name}; available versions: {available}")
    text = path.read_text(encoding="utf-8")
    sections = parse_sections(text)
    return Prompt(
        name=name,
        version=version,
        system=sections["system"],
        user=sections["user"],
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    )
