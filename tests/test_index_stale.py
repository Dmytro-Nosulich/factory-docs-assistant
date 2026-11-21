"""S2-15: `make run` rebuilds the index only when `stale_reasons` finds a reason."""

import copy
import json

from factory_docs.config import ROOT, load_config
from factory_docs.index import build_inputs, file_sha256, stale_reasons


def inputs():
    return {
        "embedding": {
            "key": "embeddinggemma",
            "model": "google/embeddinggemma-300m",
            "query_prefix": "q: ",
            "passage_prefix": "p: ",
        },
        "bm25": {"k1": 1.2, "b": 0.75},
        "store": "data/index/qdrant",
        "versions": {"fastembed": "0.9.1", "onnxruntime": "1.32.0", "qdrant-client": "1.20.0"},
        "collections": {
            "structure": {"chunks_file": "a.jsonl", "chunks_sha256": "aaa"},
            "fixed": {"chunks_file": "b.jsonl", "chunks_sha256": "bbb"},
        },
    }


def record():
    """A build record as build_index.py writes it: the inputs plus build facts."""
    saved = copy.deepcopy(inputs())
    saved["embedding"] |= {"dimension": 768, "token_limit": 2048, "batch_size": 32}
    saved["collections"]["structure"] |= {"points": 10, "build_s": 1.0}
    saved["collections"]["fixed"] |= {"points": 7, "build_s": 1.0}
    return saved


POINTS = {"structure": 10, "fixed": 7}


def test_matching_record_and_store_is_current():
    assert stale_reasons(record(), inputs(), POINTS) == []
    # an extra collection in the store does not matter
    assert stale_reasons(record(), inputs(), POINTS | {"old": 3}) == []


def test_missing_store_or_record_is_stale():
    assert stale_reasons(record(), inputs(), None) == ["no index at data/index/qdrant"]
    assert stale_reasons(None, inputs(), POINTS) == ["no build record"]


def test_changed_chunk_file_is_stale():
    now = inputs()
    now["collections"]["fixed"]["chunks_sha256"] = "changed"
    assert stale_reasons(record(), now, POINTS) == ["collection fixed: chunks_sha256 changed"]


def test_changed_model_prefix_bm25_or_library_is_stale():
    for section, key, value in [
        ("embedding", "model", "intfloat/multilingual-e5-large"),
        ("embedding", "query_prefix", "query: "),
        ("bm25", "k1", 1.5),
        ("versions", "fastembed", "0.10.0"),
    ]:
        now = inputs()
        now[section][key] = value
        [reason] = stale_reasons(record(), now, POINTS)
        assert reason.startswith(f"{section} {key}:"), reason


def test_missing_or_partial_collection_is_stale():
    assert stale_reasons(record(), inputs(), {"structure": 10}) == [
        "collection fixed: missing from the store"
    ]
    # a build that stopped half-way: the record is from the last full build
    [reason] = stale_reasons(record(), inputs(), {"structure": 4, "fixed": 7})
    assert reason == "collection structure: 4 points in the store, 10 in the record"


def test_new_collection_in_config_is_stale():
    now = inputs()
    now["collections"]["semantic"] = {"chunks_file": "c.jsonl", "chunks_sha256": "ccc"}
    assert stale_reasons(record(), now, POINTS | {"semantic": 1}) == [
        "collection semantic: not in the record"
    ]


def test_build_inputs_match_the_committed_build_record():
    """The committed index_build.json was built from the committed chunk files."""
    config = load_config()
    now = build_inputs(config)
    saved = json.loads((ROOT / "data/index/index_build.json").read_text(encoding="utf-8"))
    points = {name: c["points"] for name, c in saved["collections"].items()}
    assert stale_reasons(saved, now, points) == []
    for name, chunks_file in config.index.collections.items():
        assert now["collections"][name]["chunks_sha256"] == file_sha256(ROOT / chunks_file)
