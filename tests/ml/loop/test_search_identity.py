"""The policy id with search and the search identity (ADR-026 §5): ids agree with the TS twin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.loop.records import policy_id, search_identity_hash

ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads((ROOT / "config/search/rerank-1.v1.json").read_text())
EXISTING = json.loads((ROOT / "fixtures/policy-ids/existing.json").read_text())
H = "a" * 64
# Vectors shared with packages/showdown-bridge/test/search-config.test.ts.
CONFIG_HASH = "739f4b1f1a30411b3d5fb3d37749739a6673bcaf6195941f3a57f6dc1e70d49c"
# Moved by #248's format bumps (WORLD_BUILDER_FORMAT 2, world-facts/3), which are identity
# inputs, and by belief/5 and belief/6 through the belief identity hash named in the config.
SEARCH_HASH = "3508d46a98552f0898f662de77bfa30067c13eaefe085145f39766d0bc4e1ce6"
# The policy-id vectors take a search hash as an opaque string and keep the one
# they were computed with.
ID_VECTOR_SEARCH = "edd348bdf8b9f5069ada74afae22d3ba99e1b349e2748b245de73384c9d666b9"


def identity() -> dict[str, Any]:
    return {
        "search_config_hash": CONFIG_HASH,
        "belief_identity_hash": CONFIG["belief_identity_hash"],
        "mechanics_config_hash": "1" * 64,
        "mechanics_adapter_format": 2,
        "damage_aware_config_hash": "2" * 64,
        "damage_value_format": 1,
        "world_writers_hash": "3" * 64,
        "world_builder_format": 2,
        "world_facts_format": "world-facts/3",
        "reply_model_format": 1,
        "leaf_format": 1,
        "reader_sha256": "4" * 64,
        "version_manifest": {
            "showdown_commit": "c" * 40,
            "format_id": "gen9championsvgc2026regmc",
            "regulation": "M-C",
            "ruleset_hash": "5" * 64,
            "format_rules_hash": "6" * 64,
        },
    }


def test_committed_config_hash_is_the_ts_one() -> None:
    assert sha256_hex(jcs(CONFIG)) == CONFIG_HASH


def test_search_identity_hash_vector() -> None:
    assert search_identity_hash(identity()) == SEARCH_HASH


def test_search_identity_moves_with_every_input() -> None:
    base = search_identity_hash(identity())
    for key in identity():
        changed = identity()
        value = changed[key]
        if isinstance(value, int):
            changed[key] = value + 1
        elif isinstance(value, dict):
            changed[key] = {**value, "regulation": "M-D"}
        elif key == "world_facts_format":
            changed[key] = "world-facts/4"
        else:
            changed[key] = "9" * 64
        assert search_identity_hash(changed) != base, key


def test_search_identity_refuses_a_wrong_shape() -> None:
    with pytest.raises(ValueError, match="thirteen"):
        search_identity_hash({**identity(), "extra": 1})
    with pytest.raises(ValueError, match="SHA-256"):
        search_identity_hash({**identity(), "reader_sha256": "abc"})


@pytest.mark.parametrize("case", EXISTING)
def test_every_existing_policy_id_is_unchanged(case: dict[str, Any]) -> None:
    kwargs = {k: v for k, v in case.items() if k != "id"}
    assert policy_id(**kwargs) == case["id"]


def test_policy_id_gains_search_only_when_present() -> None:
    assert (
        policy_id(H, "b" * 64, "argmax", search=ID_VECTOR_SEARCH)
        == "policy:86611a1d2e6ce20f8e0a1a1f4b61cf3a88d27229bc18ef2a80ec4990b1797453"
    )
    full = policy_id(
        H,
        "b" * 64,
        "sample",
        temperature_ppm=700000,
        truncation={"kind": "threshold", "min_ppm": 30000},
        features="f" * 64,
        search=ID_VECTOR_SEARCH,
    )
    assert full == "policy:7cbab00c7c5aac1f95f622fec925b0ddf5bf0097d0d05352911b6ceb6174fb26"
    assert policy_id(H, "b" * 64, "sample", search=SEARCH_HASH) != policy_id(H, "b" * 64, "sample")
    assert policy_id(H, "b" * 64, "sample", search=SEARCH_HASH) != policy_id(
        H, "b" * 64, "argmax", search=SEARCH_HASH
    )


def test_policy_id_refuses_a_search_that_is_not_a_hash() -> None:
    with pytest.raises(ValueError, match="search"):
        policy_id(H, "b" * 64, search="nope")
