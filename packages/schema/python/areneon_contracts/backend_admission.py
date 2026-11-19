"""Reader for the BackendAdmission ledger (ADR-030 common rules 2, 4, 6 and 7; ADR-028 section 5).

The schema is closed and integer-only; this reader recomputes what the profile cannot say, as
the TS reader of ``@areneon/differential`` does: the ledger's pin, sorted and unique corpora,
entities, revocations and pairs, at most one fresh corpus whose seed is the backend version's fresh
seed (SHA-256 over the JCS of ``{backend_commit, backend_format, purpose}``), nested counts, the
recomputed admission of every entity (at least 20 exercising fresh items, all equal, no difference
anywhere, no revocation, and only with a fresh corpus of the level's minimum size and no count
mismatch in any corpus; regression and shadow counts never admit), each revocation at the first
corpus that found a difference, and the listed pairs (both admitted, present together in at least
1 in 10^4 fresh items, admitted with at least 5 co-exercised items, all equal).
``admission_hash`` is SHA-256 of the JCS bytes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, sha256_hex, validate

N_EXERCISED = 20
N_PAIR = 5
PAIR_ODDS = 10_000
FRESH_PURPOSE = "differential-fresh/1"
MIN_FRESH_ITEMS = {"mechanics-query": 200_000, "battle-step": 200_000}


class AdmissionError(ValueError):
    """A ledger the reader refuses."""


def fresh_seed(backend_commit: str, backend_format: int) -> str:
    """The seed of a backend version's fresh admission corpus."""
    return sha256_hex(
        jcs(
            {
                "backend_commit": backend_commit,
                "backend_format": backend_format,
                "purpose": FRESH_PURPOSE,
            }
        )
    )


def admission_hash(ledger: dict[str, Any]) -> str:
    """SHA-256 of the ledger's JCS bytes (the hash ``generation.backends`` names)."""
    return sha256_hex(jcs(ledger))


def _admits(fresh: dict[str, Any] | None, differing: int, revoked: bool, evidence: bool) -> bool:
    return (
        evidence
        and fresh is not None
        and fresh["exercised"] >= N_EXERCISED
        and fresh["equal"] == fresh["exercised"]
        and differing == 0
        and not revoked
    )


def _sorted_unique(values: list[Any]) -> bool:
    return all(a < b for a, b in itertools.pairwise(values))


@dataclass(frozen=True)
class _Context:
    """What every entity row is checked against."""

    kinds: dict[str, str]
    items: dict[str, int]
    order: dict[str, int]
    revoked: dict[str, dict[str, Any]]
    known: set[str]
    evidence: bool


def _check_entity(entity: dict[str, Any], ctx: _Context) -> None:
    ranks = [ctx.order.get(c["corpus"], -1) for c in entity["counts"]]
    if any(r < 0 for r in ranks) or not _sorted_unique(ranks):
        raise AdmissionError(
            "an entity's counts must name known corpora, in corpus order, once each"
        )
    first_differing: str | None = None
    fresh: dict[str, Any] | None = None
    differing = 0
    for c in entity["counts"]:
        if c["present"] < 1 or c["present"] > ctx.items[c["corpus"]]:
            raise AdmissionError(
                "an entity is present in at least one and at most every item of a corpus"
            )
        if (
            c["exercised"] > c["present"]
            or c["equal"] > c["exercised"]
            or c["differing"] > c["present"]
        ):
            raise AdmissionError(
                "an entity's counts must nest: equal <= exercised <= present, differing <= present"
            )
        if c["exercised"] - c["equal"] > c["differing"]:
            raise AdmissionError("an unequal exercising item is a differing item")
        if c["differing"] > 0 and first_differing is None:
            first_differing = c["corpus"]
        differing += c["differing"]
        if ctx.kinds[c["corpus"]] == "fresh":
            fresh = c
    r = ctx.revoked.get(entity["id"])
    if (r["corpus"] if r is not None else None) != first_differing:
        raise AdmissionError(
            "an entity is revoked exactly when a difference named it,"
            " at the first corpus that found one"
        )
    _check_kinds(entity, fresh, ctx.known)
    ok = _admits(fresh, differing, r is not None, ctx.evidence)
    if entity["admitted"] != ok:
        if entity["admitted"] and _admits(fresh, differing, r is not None, True):
            raise AdmissionError(
                "nothing is admitted without a fresh corpus of the level's minimum size"
                " and with any count mismatch"
            )
        as_if_fresh = any(
            c["exercised"] >= N_EXERCISED and c["equal"] == c["exercised"]
            for c in entity["counts"]
            if ctx.kinds[c["corpus"]] != "fresh"
        )
        if entity["admitted"] and as_if_fresh and differing == 0 and r is None:
            raise AdmissionError(
                "admission evidence must come from the fresh corpus:"
                " regression and shadow counts never admit"
            )
        raise AdmissionError("an entity's admitted flag must equal the recomputed admission")


def _check_kinds(entity: dict[str, Any], fresh: dict[str, Any] | None, known: set[str]) -> None:
    rows = entity["kinds"]
    names = [k["kind"] for k in rows]
    if not _sorted_unique(names) or any(not n.startswith("kind:") or n not in known for n in names):
        raise AdmissionError(
            "an entity's kinds name query-kind entities of the ledger, sorted and unique"
        )
    if (entity["id"].startswith("kind:") or fresh is None) and rows:
        raise AdmissionError(
            "only an entity the fresh corpus holds, and no query kind, lists kinds"
        )
    limit = fresh["exercised"] if fresh is not None else 0
    if any(k["co_equal"] > k["co_exercised"] or k["co_exercised"] > limit for k in rows):
        raise AdmissionError("an entity's kind counts must nest within its fresh exercising items")


def _evidence_holds(ledger: dict[str, Any]) -> bool:
    fresh = [c for c in ledger["corpora"] if c["kind"] == "fresh"]
    return (
        bool(fresh)
        and fresh[0]["items"] >= MIN_FRESH_ITEMS[ledger["level"]]
        and all(c["count_mismatches"] == 0 for c in ledger["corpora"])
    )


def _check_pairs(pairs: list[dict[str, Any]], admitted: set[str], fresh_items: int) -> None:
    keys = [(p["a"], p["b"]) for p in pairs]
    if any(not a < b for a, b in keys) or not _sorted_unique(keys):
        raise AdmissionError("pairs must have a below b and be sorted and unique")
    for p in pairs:
        if p["a"] not in admitted or p["b"] not in admitted:
            raise AdmissionError("a pair lists two admitted entities")
        if (
            p["co_exercised"] > p["co_present"]
            or p["co_equal"] > p["co_exercised"]
            or p["co_present"] > fresh_items
        ):
            raise AdmissionError("a pair's counts must nest within the fresh corpus")
        if p["co_present"] == 0 or p["co_present"] * PAIR_ODDS < fresh_items:
            raise AdmissionError("a listed pair is present in at least 1 in 10^4 fresh items")
        if p["admitted"] != (p["co_exercised"] >= N_PAIR and p["co_equal"] == p["co_exercised"]):
            raise AdmissionError("a pair's admitted flag must equal the recomputed admission")


def read_backend_admission(value: object, showdown_commit: str) -> dict[str, Any]:
    """Read a ledger at the pin ``showdown_commit``: the schema, then every recomputed rule."""
    try:
        validate("backend-admission", value)
    except ContractError as exc:
        raise AdmissionError(str(exc)) from exc
    ledger: dict[str, Any] = value  # type: ignore[assignment]
    if ledger["showdown_commit"] != showdown_commit:
        raise AdmissionError(
            "the ledger is for another pinned engine: every query routes to the engine"
        )
    corpora = ledger["corpora"]
    if not _sorted_unique([c["id"] for c in corpora]):
        raise AdmissionError("corpora must be sorted by id and unique")
    fresh = [c for c in corpora if c["kind"] == "fresh"]
    if len(fresh) > 1:
        raise AdmissionError("a ledger holds at most one fresh corpus")
    backend = ledger["backend"]
    if fresh and fresh[0]["seed"] != fresh_seed(
        backend["backend_commit"], backend["backend_format"]
    ):
        raise AdmissionError("the fresh corpus's seed is not the backend version's fresh seed")
    entities = ledger["entities"]
    if not _sorted_unique([e["id"] for e in entities]):
        raise AdmissionError("entities must be sorted by id and unique")
    if not _sorted_unique([r["id"] for r in ledger["revoked"]]):
        raise AdmissionError("revocations must be sorted by id and unique")
    kinds = {c["id"]: c["kind"] for c in corpora}
    items = {c["id"]: c["items"] for c in corpora}
    order = {c["id"]: i for i, c in enumerate(corpora)}
    revoked = {r["id"]: r for r in ledger["revoked"]}
    known = {e["id"] for e in entities}
    evidence = _evidence_holds(ledger)
    for entity in entities:
        _check_entity(entity, _Context(kinds, items, order, revoked, known, evidence))
    for r in ledger["revoked"]:
        if r["id"] not in known or r["corpus"] not in order:
            raise AdmissionError("a revocation names a known entity and corpus")
        if r["position"] >= items[r["corpus"]]:
            raise AdmissionError("a revocation's position lies within its corpus")
    admitted = {e["id"] for e in entities if e["admitted"]}
    _check_pairs(ledger["pairs"], admitted, fresh[0]["items"] if fresh else 0)
    return ledger
