"""The BM25 tokenizer and term weights (S2-07), offline."""

import pytest

from factory_docs.lexical import Bm25Encoder, encode_query, term_id, tokenize, vocabulary


def test_codes_stay_whole_tokens():
    assert tokenize("CAS 64742-48-9, H226 and EUH066.") == [
        "cas",
        "64742-48-9",
        "h226",
        "and",
        "euh066",
    ]


def test_section_numbers_keep_their_dots_but_not_a_trailing_one():
    assert tokenize("Abschnitt 1.1. Ende.") == ["abschnitt", "1.1", "ende"]


def test_cyrillic_words_are_tokens():
    assert tokenize("Перша допомога: при ковтанні") == ["перша", "допомога", "при", "ковтанні"]


def test_casefold_matches_german_sharp_s_and_keeps_umlauts():
    assert tokenize("Größe STRASSE Ätzend") == ["grösse", "strasse", "ätzend"]
    assert tokenize("Straße") == tokenize("STRASSE")


def test_nfkc_unifies_fullwidth_letters_and_ligatures():
    assert tokenize("ＣＡＳ ﬁre") == ["cas", "fire"]


def test_unicode_dashes_become_hyphens():
    assert tokenize("64742–48–9") == tokenize("64742-48-9") == ["64742-48-9"]


def test_underscore_and_symbols_split_tokens():
    assert tokenize("a_b 23 °C; x") == ["a", "b", "23", "c", "x"]
    assert tokenize("?! …") == []


def test_term_ids_are_stable_32_bit_numbers():
    # Pinned: the index is built with these IDs, so a change here needs `make index`.
    assert term_id("h226") == 1524894692
    assert term_id("64742-48-9") == 228892732
    assert all(0 <= term_id(t) < 2**32 for t in ["a", "flammpunkt", "ковтанні"])
    assert term_id("h226") != term_id("h225")


def test_bm25_weights_match_hand_computed_values():
    # len 3, avg_len 4: norm = 1.2 · (1 − 0.75 + 0.75 · 3/4) = 0.975
    vector = Bm25Encoder(k1=1.2, b=0.75, avg_len=4).encode_passage("acid ACID water")
    assert dict(zip(vector.indices, vector.values, strict=True)) == {
        term_id("acid"): pytest.approx(2 * 2.2 / (2 + 0.975)),  # tf 2: 1.4790
        term_id("water"): pytest.approx(2.2 / (1 + 0.975)),  # tf 1: 1.1139
    }
    assert vector.indices == sorted(vector.indices)


def test_repeating_a_word_helps_less_each_time():
    encoder = Bm25Encoder(k1=1.2, b=0.75, avg_len=10)
    weights = [encoder.encode_passage(" ".join(["acid"] * tf)).values[0] for tf in (1, 2, 4, 8)]
    gains = [after - before for before, after in zip(weights, weights[1:], strict=False)]
    assert all(gain > 0 for gain in gains)
    assert gains == sorted(gains, reverse=True)
    assert weights[-1] < 1.2 + 1  # never above k1 + 1


def weight(vector, token: str) -> float:
    return dict(zip(vector.indices, vector.values, strict=True))[term_id(token)]


def test_a_longer_chunk_gets_a_lower_weight_for_the_same_term():
    encoder = Bm25Encoder(k1=1.2, b=0.75, avg_len=4)
    short = encoder.encode_passage("acid water")
    long = encoder.encode_passage("acid water " + "x " * 20)
    assert weight(long, "acid") < weight(short, "acid")
    assert weight(Bm25Encoder(1.2, b=0, avg_len=4).encode_passage("acid " + "x " * 20), "acid") == (
        pytest.approx(1.0)  # b = 0 ignores length: tf 1 weighs (k1 + 1) / (1 + k1) = 1
    )


def test_fit_takes_the_mean_token_count():
    assert Bm25Encoder.fit(["a b", "a b c d"], 1.2, 0.75).avg_len == 3
    assert Bm25Encoder.fit([], 1.2, 0.75).avg_len == 0
    assert Bm25Encoder(1.2, 0.75, avg_len=0).encode_passage("").indices == []


def test_a_query_weights_each_distinct_term_once():
    vector = encode_query("Flash point? flash POINT")
    assert vector.indices == sorted([term_id("flash"), term_id("point")])
    assert vector.values == [1.0, 1.0]
    assert encode_query("?").indices == []


def test_vocabulary_counts_distinct_tokens():
    assert vocabulary(["acid water", "Acid H226"]) == (3, 0)
