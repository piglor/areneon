"""The usage prior and belief configuration readers (ADR-025 sections 2, 5; #1119).

The Python readers refuse exactly what the TS ones do: every reader-invalid corpus sample, the
schema-invalid ones, and an ``evs`` key, a paste or a team-shaped object anywhere in a prior. The
valid samples hash to the digests the TS side recorded.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_contracts.usage_prior import (
    UsagePriorError,
    belief_config_hash,
    read_belief_config,
    read_usage_prior,
    usage_prior_hash,
)

SAMPLES = Path(__file__).resolve().parents[2] / "fixtures/contracts"
HASHES: dict[str, str] = json.loads((SAMPLES / "hashes.json").read_text(encoding="utf-8"))


def load(path: str) -> Any:
    return json.loads((SAMPLES / path).read_text(encoding="utf-8"))


VOCABULARY = load("v1/team-vocabulary/valid/golden.json")


def names(contract: str, kind: str) -> list[str]:
    prefix = f"v1/{contract}/{kind}/"
    return sorted(p[len(prefix) :] for p in HASHES if p.startswith(prefix))


def test_the_three_source_kinds_read_and_hash_as_the_ts_side_does() -> None:
    for kind in ("smogon-chaos", "role-sets", "generator-matched"):
        path = f"v1/usage-prior/valid/{kind}.json"
        prior = read_usage_prior(load(path), VOCABULARY)
        assert prior["source"]["kind"] == kind
        assert usage_prior_hash(prior) == HASHES[path]


def test_the_belief_config_reads_and_hashes_as_the_ts_side_does() -> None:
    path = "v1/belief-config/valid/v1.json"
    config = read_belief_config(load(path), VOCABULARY)
    assert config["max_legality_redraws"] == 32
    assert belief_config_hash(config) == HASHES[path] == sha256_hex(jcs(load(path)))


@pytest.mark.parametrize("name", names("usage-prior", "reader-invalid"))
def test_the_prior_reader_refuses_each_reader_invalid_sample(name: str) -> None:
    with pytest.raises(UsagePriorError):
        read_usage_prior(load(f"v1/usage-prior/reader-invalid/{name}"), VOCABULARY)


@pytest.mark.parametrize("name", names("usage-prior", "invalid"))
def test_the_prior_reader_refuses_each_schema_invalid_sample(name: str) -> None:
    with pytest.raises(UsagePriorError):
        read_usage_prior(load(f"v1/usage-prior/invalid/{name}"), VOCABULARY)


@pytest.mark.parametrize("name", names("belief-config", "reader-invalid"))
def test_the_config_reader_refuses_each_reader_invalid_sample(name: str) -> None:
    with pytest.raises(UsagePriorError):
        read_belief_config(load(f"v1/belief-config/reader-invalid/{name}"), VOCABULARY)


@pytest.mark.parametrize("name", names("belief-config", "invalid"))
def test_the_config_reader_refuses_each_schema_invalid_sample(name: str) -> None:
    with pytest.raises(UsagePriorError):
        read_belief_config(load(f"v1/belief-config/invalid/{name}"), VOCABULARY)


def test_a_paste_or_an_evs_key_anywhere_is_refused() -> None:
    for intruder in ({"paste": "Pikachu @ Light Ball"}, {"evs": {"hp": 1}}):
        prior = load("v1/usage-prior/valid/smogon-chaos.json")
        prior["species"][0]["formes"][0].update(intruder)
        with pytest.raises(UsagePriorError, match="usage-prior"):
            read_usage_prior(prior, VOCABULARY)


def test_the_messages_name_the_rule() -> None:
    cases = {
        "unsorted-species": "species: .* repeated or out of order",
        "moves-sum": "slots times the denominator",
        "items-sum": "items: the weights do not sum",
        "inclusion-over-denominator": "included more often than there are sets",
        "spread-over-limit": "more Stat Points",
        "unknown-ability": "is not one the vocabulary lists",
        "stale-vocabulary-hash": "vocabulary_hash",
        "other-format": "format_id",
        "other-engine-data": "engine_data_hash",
        "unknown-forme": "forme aaaaaaaaa is not in the vocabulary roster",
        "unknown-species": "not in the vocabulary roster",
        "forme-weights-sum": "forme weights",
        "unsorted-inputs": "source inputs",
        "unknown-nature": "nature",
        "unknown-item": "item aaaaaaaaa",
        "unknown-move": "move aaaaaaaaa",
        "abilities-sum": "abilities: the weights",
        "spreads-sum": "spreads: the weights",
        "unsorted-spreads": "spread 1",
        "too-many-items": "k_items",
        "too-many-spreads": "k_spreads",
        "too-many-moves": "k_moves",
        "duplicate-item": "items:",
        "unsorted-moves": "moves:",
    }
    for name, message in cases.items():
        with pytest.raises(UsagePriorError, match=message):
            read_usage_prior(load(f"v1/usage-prior/reader-invalid/{name}.json"), VOCABULARY)
    assert set(cases) == {n[:-5] for n in names("usage-prior", "reader-invalid")}


def test_spreads_of_one_nature_order_by_stat_points_and_a_repeat_is_refused() -> None:
    prior = load("v1/usage-prior/valid/smogon-chaos.json")
    spreads = prior["species"][0]["formes"][0]["spreads"]["entries"]
    spreads[1]["nature"] = spreads[0]["nature"]
    read_usage_prior(copy.deepcopy(prior), VOCABULARY)
    prior["species"][0]["formes"][0]["spreads"]["entries"] = [spreads[1], spreads[0]]
    with pytest.raises(UsagePriorError, match="spread 1"):
        read_usage_prior(copy.deepcopy(prior), VOCABULARY)
    prior["species"][0]["formes"][0]["spreads"]["entries"] = [spreads[0], dict(spreads[0])]
    with pytest.raises(UsagePriorError, match="spread 1"):
        read_usage_prior(prior, VOCABULARY)


def test_a_mega_forme_may_use_any_roster_ability_and_no_other() -> None:
    mega = copy.deepcopy(VOCABULARY)
    mega["roster"][0]["required_item"] = mega["items"][0]
    prior = load("v1/usage-prior/valid/smogon-chaos.json")
    prior["vocabulary_hash"] = sha256_hex(jcs(mega))
    foreign = next(
        (a for a in mega["roster"][1]["abilities"] if a not in mega["roster"][0]["abilities"]),
        mega["roster"][1]["abilities"][0],
    )
    forme = prior["species"][0]["formes"][0]
    forme["abilities"] = {
        "denominator": 450,
        "entries": [{"id": foreign, "weight": 300}],
        "residual": 100,
        "unmapped": 50,
    }
    read_usage_prior(copy.deepcopy(prior), mega)
    forme["abilities"]["entries"][0]["id"] = "aaaaaaaaa"
    with pytest.raises(UsagePriorError, match="ability aaaaaaaaa"):
        read_usage_prior(prior, mega)


def test_a_vocabulary_that_is_not_a_vocabulary_is_refused() -> None:
    with pytest.raises(UsagePriorError, match="team-vocabulary"):
        read_usage_prior(load("v1/usage-prior/valid/smogon-chaos.json"), {"nope": 1})
