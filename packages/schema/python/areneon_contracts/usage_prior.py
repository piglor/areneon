"""Readers for the opponent usage prior and the belief configuration (ADR-025 sections 2, 5).

The schemas are closed and integer-only; these readers add what the contract profile cannot say,
as the TS readers do: sorted and unique ids, every id resolving in the team vocabulary the record
was built for (abilities and moves against that forme's roster entry), the exact integer sums
(review finding N4: every list states its denominator and its weights, no-item, empty-slot,
residual and unmapped masses add up to it), the truncation limits and the Stat Point limit.
``usage_prior_hash`` (the ``prior_hash`` of the belief identity) and ``belief_config_hash`` are
SHA-256 of the JCS bytes. No Smogon data and no belief code live here.
"""

from __future__ import annotations

import itertools
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, sha256_hex, validate

STATS = ("hp", "atk", "def", "spa", "spd", "spe")


class UsagePriorError(ValueError):
    """A usage prior or belief configuration the reader refuses."""


def _ordered(ids: list[str], what: str) -> None:
    for before, after in itertools.pairwise(ids):
        if not before < after:
            raise UsagePriorError(f"{what}: {after} is repeated or out of order")


def _check(name: str, value: object) -> None:
    try:
        validate(name, value)
    except ContractError as exc:
        raise UsagePriorError(str(exc)) from exc


def _spread_key(spread: dict[str, Any]) -> tuple[Any, ...]:
    return (spread["nature"], *(spread["stat_points"][stat] for stat in STATS))


def _check_abilities(forme: dict[str, Any], where: str, pool: set[str]) -> None:
    abilities = forme["abilities"]
    _ordered([e["id"] for e in abilities["entries"]], f"{where} abilities")
    for e in abilities["entries"]:
        if e["id"] not in pool:
            raise UsagePriorError(
                f"{where}: ability {e['id']} is not one the vocabulary lists for {forme['id']}"
            )
    mass = sum(e["weight"] for e in abilities["entries"]) + abilities["residual"]
    if mass + abilities["unmapped"] != abilities["denominator"]:
        raise UsagePriorError(f"{where} abilities: the weights do not sum to the denominator")


def _check_items(
    forme: dict[str, Any], where: str, k_items: int, vocabulary: dict[str, Any]
) -> None:
    items = forme["items"]
    _ordered([e["id"] for e in items["entries"]], f"{where} items")
    if len(items["entries"]) > k_items:
        raise UsagePriorError(f"{where} items: more entries than k_items")
    for e in items["entries"]:
        if e["id"] not in vocabulary["items"]:
            raise UsagePriorError(f"{where}: item {e['id']} is not in the vocabulary")
    mass = sum(e["weight"] for e in items["entries"]) + items["no_item"] + items["residual"]
    if mass + items["unmapped"] != items["denominator"]:
        raise UsagePriorError(f"{where} items: the weights do not sum to the denominator")


def _check_spreads(
    forme: dict[str, Any], where: str, k_spreads: int, vocabulary: dict[str, Any]
) -> None:
    spreads = forme["spreads"]
    if len(spreads["entries"]) > k_spreads:
        raise UsagePriorError(f"{where} spreads: more entries than k_spreads")
    natures = {n["id"] for n in vocabulary["natures"]}
    for k, spread in enumerate(spreads["entries"]):
        if spread["nature"] not in natures:
            raise UsagePriorError(f"{where}: nature {spread['nature']} is not in the vocabulary")
        if sum(spread["stat_points"].values()) > vocabulary["rules"]["ev_limit"]:
            raise UsagePriorError(
                f"{where}: spread {k} holds more Stat Points than the format's limit"
            )
        if k > 0 and _spread_key(spreads["entries"][k - 1]) >= _spread_key(spread):
            raise UsagePriorError(f"{where} spreads: spread {k} is repeated or out of order")
    mass = sum(s["weight"] for s in spreads["entries"]) + spreads["residual"]
    if mass + spreads["unmapped"] != spreads["denominator"]:
        raise UsagePriorError(f"{where} spreads: the weights do not sum to the denominator")


def _check_moves(forme: dict[str, Any], where: str, k_moves: int, pool: list[str]) -> None:
    moves = forme["moves"]
    _ordered([e["id"] for e in moves["entries"]], f"{where} moves")
    if len(moves["entries"]) > k_moves:
        raise UsagePriorError(f"{where} moves: more entries than k_moves")
    for e in moves["entries"]:
        if e["id"] not in pool:
            raise UsagePriorError(
                f"{where}: move {e['id']} is not in the roster move pool of {forme['id']}"
            )
        if e["inclusion"] > moves["denominator"]:
            raise UsagePriorError(
                f"{where}: move {e['id']} is included more often than there are sets"
            )
    mass = sum(e["inclusion"] for e in moves["entries"])
    mass += moves["empty_slots"] + moves["residual"] + moves["unmapped"]
    if mass != moves["slots"] * moves["denominator"]:
        raise UsagePriorError(
            f"{where} moves: the masses do not sum to slots times the denominator"
        )


def _check_forme(
    forme: dict[str, Any], where: str, prior: dict[str, Any], vocabulary: dict[str, Any]
) -> None:
    roster = {entry["id"]: entry for entry in vocabulary["roster"]}
    entry = roster.get(forme["id"])
    if entry is None:
        raise UsagePriorError(f"{where}: forme {forme['id']} is not in the vocabulary roster")
    # A Mega forme's abilities are post-Evolution ones, which its roster entry may lack.
    if "required_item" in entry:
        pool = {a for r in roster.values() for a in r["abilities"]}
    else:
        pool = set(entry["abilities"])
    limits = prior["derivation"]
    _check_abilities(forme, where, pool)
    _check_items(forme, where, limits["k_items"], vocabulary)
    _check_spreads(forme, where, limits["k_spreads"], vocabulary)
    _check_moves(forme, where, limits["k_moves"], entry["moves"])


def read_usage_prior(value: object, vocabulary: dict[str, Any]) -> dict[str, Any]:
    """The schema, then the reader rules, against ``vocabulary`` (a TeamVocabulary document).

    The prior must name the vocabulary by hash, and so its format and engine-data table.
    """
    _check("usage-prior", value)
    _check("team-vocabulary", vocabulary)
    prior: dict[str, Any] = value  # type: ignore[assignment]
    if prior["vocabulary_hash"] != sha256_hex(jcs(vocabulary)):
        raise UsagePriorError("vocabulary_hash is not the hash of the given vocabulary")
    if prior["format_id"] != vocabulary["format_id"]:
        raise UsagePriorError("format_id differs from the vocabulary's")
    if prior["engine_data_hash"] != vocabulary["engine_data_hash"]:
        raise UsagePriorError("engine_data_hash differs from the vocabulary's")
    if prior["source"]["kind"] == "smogon-chaos":
        _ordered([i["source_id"] for i in prior["source"]["inputs"]], "source inputs")

    roster = {entry["id"] for entry in vocabulary["roster"]}
    _ordered([s["id"] for s in prior["species"]], "species")
    for species in prior["species"]:
        if species["id"] not in roster:
            raise UsagePriorError(f"species {species['id']} is not in the vocabulary roster")
        _ordered([f["id"] for f in species["formes"]], f"species {species['id']} formes")
        if sum(f["weight"] for f in species["formes"]) != species["denominator"]:
            raise UsagePriorError(
                f"species {species['id']}: the forme weights do not sum to the denominator"
            )
        for forme in species["formes"]:
            where = f"species {species['id']} forme {forme['id']}"
            _check_forme(forme, where, prior, vocabulary)
    return prior


def usage_prior_hash(prior: dict[str, Any]) -> str:
    """SHA-256 of the prior's JCS bytes: ``prior_hash`` of the belief identity (ADR-025 §5)."""
    return sha256_hex(jcs(prior))


def read_belief_config(value: object, vocabulary: dict[str, Any]) -> dict[str, Any]:
    """The schema, strictly ascending ability ids, and each id an ability of the vocabulary."""
    _check("belief-config", value)
    _check("team-vocabulary", vocabulary)
    config: dict[str, Any] = value  # type: ignore[assignment]
    known = {a for entry in vocabulary["roster"] for a in entry["abilities"]}
    abilities = config["identity_hiding_abilities"]
    _ordered(abilities, "identity_hiding_abilities")
    for ability in abilities:
        if ability not in known:
            raise UsagePriorError(f"ability {ability} is not in the team vocabulary")
    return config


def belief_config_hash(config: dict[str, Any]) -> str:
    """SHA-256 of the config's JCS bytes: ``belief_config_hash`` of the belief identity."""
    return sha256_hex(jcs(config))
