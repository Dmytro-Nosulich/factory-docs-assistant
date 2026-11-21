"""Which language an answer is written in: a small heuristic for the S2-14 language check.

Prompt answer v1 wrote English answers in German (`docs/ISSUES.md` I-009), so the refusal eval
counts answers whose claims and `missing` are in the question's language. This needs only the
nine corpus languages and short texts, so it is a heuristic, not a general detector:
- Latin script: each language scores one point per word of its list of common function words;
  the highest score wins.
- Cyrillic script, when the text has more Cyrillic than Latin letters, or has Cyrillic letters
  and no Latin function word (a Ukrainian sentence full of English chemical names): `uk`,
  unless the text has a letter Ukrainian does not use (ы, э, ъ, ё), which is `unknown`. The
  corpus has no other Cyrillic language; a short Russian text without those letters counts as uk.
- Otherwise (no function word, or a tie) `unknown`.
Quotes are not passed in: they stay in the chunk's language by design.
"""

import re
import unicodedata

WORD = re.compile(r"[^\W\d_]+")
NOT_UKRAINIAN = set("ыэъё")  # Russian letters Ukrainian does not have

# Frequent function words. A word common to two languages is left out of both, so it never
# decides a tie.
FUNCTION_WORDS = {
    "en": "the is are was were of to and in on for with this that it not no does do has have "
    "be by as at from or which what any only",
    "de": "der die das ist sind und nicht kein keine mit von zu im auf für bei ein eine wird "
    "werden den dem des es sich auch nur oder hat kann muss beträgt enthält",
    "fr": "le la les est sont et ne pas aucun aucune avec du des un une pour dans sur par il "
    "elle ce qui que être ou au aux",
    "it": "il lo gli è sono e non nessun nessuna con del della di un una per nel sulla che "
    "essere o al alla",
    "es": "el los las es son y no ningún ninguna con del un una para en por que ser o al",
    "nl": "het een is zijn en niet geen met van voor op bij dat wordt worden er ook",
    "pl": "jest są i nie brak z w na do dla że lub się przez",
    "sv": "och är inte ingen med av till för på som det att eller har",
}


def _words(language: str) -> set[str]:
    return set(FUNCTION_WORDS[language].split())


def _exclusive_lists() -> dict[str, set[str]]:
    lists = {language: _words(language) for language in FUNCTION_WORDS}
    shared = {
        word
        for language, words in lists.items()
        for word in words
        if any(word in other for o, other in lists.items() if o != language)
    }
    return {language: words - shared for language, words in lists.items()}


WORDS = _exclusive_lists()


def letters(text: str) -> tuple[int, int]:
    """(Cyrillic, Latin) letter counts."""
    cyrillic = latin = 0
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if name.startswith("CYRILLIC"):
            cyrillic += 1
        elif name.startswith("LATIN"):
            latin += 1
    return cyrillic, latin


def detect(text: str) -> str:
    """An ISO 639-1 code, or `unknown`."""
    cyrillic, latin = letters(text)
    if cyrillic == latin == 0:
        return "unknown"
    lowered = text.casefold()
    words = WORD.findall(lowered)
    scores = {language: sum(w in WORDS[language] for w in words) for language in WORDS}
    ranked = sorted(scores.values(), reverse=True)
    if cyrillic > latin or (cyrillic and ranked[0] == 0):
        return "unknown" if NOT_UKRAINIAN & set(lowered) else "uk"
    if ranked[0] == 0 or ranked[0] == ranked[1]:
        return "unknown"
    return max(scores, key=lambda language: scores[language])


def answer_text(answer: dict) -> str:
    """The parts of an answer written in the question's language: claims and `missing`."""
    parts = [claim["text"] for claim in answer.get("claims", [])]
    if answer.get("missing"):
        parts.append(answer["missing"])
    return "\n".join(parts)
