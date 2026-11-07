"""The policy id with the Team Preview solver (ADR-029 §4); ids agree with the TS twin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.loop.records import (
    policy_id,
    preview_solve_config_hash,
    preview_solve_identity_hash,
    read_preview_solve_config,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads(
    (ROOT / "config/preview/gen9championsvgc2026regmc/preview-solve-1.v1.json").read_text()
)
EXISTING = json.loads((ROOT / "fixtures/policy-ids/existing.json").read_text())
H = "a" * 64
# Vectors shared with packages/showdown-bridge/test/preview-solve-config.test.ts, computed with
# hashlib over the sorted-key compact JSON of these inputs.
CONFIG_HASH = "415714401460127e920daefdc52ad85707fef8fa3d1d98cdd5ff01fbba27ee0e"
IDENTITY_HASH = "89ae2b2359f6d1d272b03efc8d7959f9c0a2135d87ae4eefb48270b519caf932"
SEARCH_VECTOR = "edd348bdf8b9f5069ada74afae22d3ba99e1b349e2748b245de73384c9d666b9"


def identity() -> dict[str, Any]:
    return {
        "preview_solve_config_hash": CONFIG_HASH,
        "belief_identity_hash": CONFIG["belief_identity_hash"],
        "reader_sha256": "4" * 64,
        "version_manifest": {
            "showdown_commit": "c" * 40,
            "format_id": "gen9championsvgc2026regmc",
            "regulation": "M-C",
            "ruleset_hash": "5" * 64,
            "format_rules_hash": "6" * 64,
        },
        "solver_format": "preview-solver/1",
        "leaf_format": 1,
        "hypothetical_battle_format": 1,
    }


def test_committed_config_is_read_and_hashes_to_the_ts_vector() -> None:
    assert read_preview_solve_config(CONFIG) == CONFIG
    assert preview_solve_config_hash(CONFIG) == CONFIG_HASH
    assert sha256_hex(jcs(CONFIG)) == CONFIG_HASH


def test_config_reader_rules() -> None:
    over = json.loads(json.dumps(CONFIG))
    over["worlds"]["min_valid"] = over["worlds"]["n"] + 1
    with pytest.raises(ValueError, match="min_valid"):
        read_preview_solve_config(over)
    bare = json.loads(json.dumps(CONFIG))
    del bare["solver"]["iterations"]
    with pytest.raises(ValueError, match="iterations"):
        read_preview_solve_config(bare)
    exact = json.loads(json.dumps(CONFIG))
    exact["solver"] = {"kind": "exact-support", "iterations": 5}
    with pytest.raises(ValueError, match="iterations"):
        read_preview_solve_config(exact)
    exact["solver"] = {"kind": "exact-support"}
    assert read_preview_solve_config(exact)["solver"] == {"kind": "exact-support"}
    extra = json.loads(json.dumps(CONFIG))
    extra["note"] = "x"
    with pytest.raises(ValueError, match="PreviewSolveConfig"):
        read_preview_solve_config(extra)


def test_identity_hash_vector_and_every_input_moves_it() -> None:
    assert preview_solve_identity_hash(identity()) == IDENTITY_HASH
    for key in identity():
        changed = identity()
        value = changed[key]
        if isinstance(value, int):
            changed[key] = value + 1
        elif isinstance(value, dict):
            changed[key] = {**value, "regulation": "M-D"}
        elif key == "solver_format":
            changed[key] = "preview-solver/2"
        else:
            changed[key] = "9" * 64
        assert preview_solve_identity_hash(changed) != IDENTITY_HASH, key


def test_identity_refuses_a_wrong_shape() -> None:
    with pytest.raises(ValueError, match="seven"):
        preview_solve_identity_hash({**identity(), "extra": 1})
    with pytest.raises(ValueError, match="SHA-256"):
        preview_solve_identity_hash({**identity(), "reader_sha256": "abc"})


@pytest.mark.parametrize("case", EXISTING)
def test_every_existing_policy_id_is_unchanged(case: dict[str, Any]) -> None:
    kwargs = {k: v for k, v in case.items() if k != "id"}
    assert policy_id(**kwargs) == case["id"]


def test_policy_id_gains_preview_solve_only_when_present() -> None:
    assert (
        policy_id(H, "b" * 64, "argmax", preview_solve=IDENTITY_HASH)
        == "policy:917a7880a7a04b493d463d68fee28131bc7484f381de95e60329a8a3da8d5499"
    )
    full = policy_id(
        H,
        "b" * 64,
        "sample",
        temperature_ppm=700000,
        truncation={"kind": "threshold", "min_ppm": 30000},
        features="f" * 64,
        search=SEARCH_VECTOR,
        preview_solve=IDENTITY_HASH,
    )
    assert full == "policy:9e18393e4c548c72aa64192bb4c137f1ab4ce6ee163cfa552584863587bbdf92"
    # Independent of search: the solver alone, search alone and both are three different policies.
    solve = policy_id(H, "b" * 64, "argmax", preview_solve=IDENTITY_HASH)
    search = policy_id(H, "b" * 64, "argmax", search=SEARCH_VECTOR)
    both = policy_id(H, "b" * 64, "argmax", search=SEARCH_VECTOR, preview_solve=IDENTITY_HASH)
    assert len({solve, search, both, policy_id(H, "b" * 64, "argmax")}) == 4
    # An id without the field is the id the code gave before it existed.
    assert search == "policy:86611a1d2e6ce20f8e0a1a1f4b61cf3a88d27229bc18ef2a80ec4990b1797453"


def test_policy_id_refuses_a_preview_solve_that_is_not_a_hash() -> None:
    with pytest.raises(ValueError, match="preview_solve"):
        policy_id(H, "b" * 64, preview_solve="nope")
