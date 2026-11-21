import pytest

from factory_docs.language import WORDS, answer_text, detect


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("The flash point of HLP 46 is 243 °C.", "en"),
        ("Der Flammpunkt ist in den Dokumenten nicht angegeben.", "de"),
        ("Le produit n'est pas classé comme dangereux.", "fr"),
        ("Il punto di infiammabilità non è indicato nei documenti.", "it"),
        ("Температура спалаху становить 243 °C.", "uk"),  # no і/ї/є/ґ, still uk
        ("Ціна в документах не вказана.", "uk"),
        # more Latin than Cyrillic letters, but no Latin function word (dev question V04, v1)
        ("Засіб STIHL Profi CP 200 містить sodium-p-cumene sulphonate у концентрації 1-<3%.", "uk"),
        ("The STIHL CC 100 shampoo is called автошампунь in the document.", "en"),
        ("Это не указано в документах.", "unknown"),  # Russian letters э, ы
        ("UN 1006", "unknown"),  # no function word
        ("243 °C", "unknown"),  # no letters
        ("", "unknown"),
    ],
)
def test_detect(text, language):
    assert detect(text) == language


def test_german_answer_to_an_english_question_is_caught():
    # the I-009 case: an English question answered in German
    assert detect("Die UN-Nummer für konzentrierte Schwefelsäure ist UN 1830.") == "de"


def test_shared_words_never_count():
    assert all(not (WORDS[a] & WORDS[b]) for a in WORDS for b in WORDS if a != b)


def test_answer_text_joins_claims_and_missing_but_not_quotes():
    answer = {
        "status": "partial",
        "claims": [{"text": "Flash point 243 °C.", "citations": [{"quote": "Flammpunkt"}]}],
        "missing": "No auto-ignition temperature.",
    }
    assert answer_text(answer) == "Flash point 243 °C.\nNo auto-ignition temperature."
