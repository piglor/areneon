"""The LadderRecord reader (ADR-021 sections 4, 6, 8): what the ladder decided in a generation.

The Python twin of ``packages/showdown-bridge/src/ladder-record.ts``: the schema checks the shape
and the same rules, in the same order and in the same words, check what the contract
profile cannot say.
Sorted and unique lists, table counts that agree with their sums, lower at most upper, the reference
anchor at its fixed value (and every anchor and frozen snapshot strength unchanged from the previous
record, when it is given), at most one copy, and the roots counted from the members. No strength or
bound is computed here: the record only holds them (slice 6 computes them).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, sha256_hex, validate

from areneon_ml.learner.config import RunConfig

REFERENCE_ANCHOR = "actor-rule@v1"
REFERENCE_STRENGTH = 1_000_000_000
# A pair score is the mean of two games of 0, 500000 or 1000000 ppm.
SCORE_STEP_PPM = 250_000
FULL_PPM = 1_000_000
# A copy needs a one-sided lower bound above even odds.
COPY_FLOOR_PPM = 500_000


class LadderRecordError(ValueError):
    """A LadderRecord breaks a rule."""


@dataclass(frozen=True)
class LadderRecord:
    """A validated record of one generation's ladder."""

    run_id: str
    run_config_hash: str
    generation: int
    raw: dict[str, Any]


def ladder_record_hash(raw: dict[str, Any]) -> str:
    """The hash a later record names this one by: SHA-256 of its JCS bytes."""
    return str(sha256_hex(jcs(raw)))


def _fail(message: str) -> None:
    raise LadderRecordError(message)


def _ascending(keys: Sequence[Any]) -> bool:
    return all(a < b for a, b in itertools.pairwise(keys))


def _check_settings(settings: dict[str, Any]) -> None:
    shares = settings["shares"]
    if shares["ladder_ppm"] + shares["self_play_ppm"] + shares["past_pool_ppm"] != FULL_PPM:
        _fail("a member's opponent shares must add up to 1000000")


def _anchor_minimum(raw: dict[str, Any], member: str) -> int:
    """The least ``score_sum_ppm // pairs`` over a member's anchor cells (one per anchor)."""
    return int(
        min(
            cell["score_sum_ppm"] // cell["pairs"]
            for cell in raw["cells"]
            if cell["member"] == member and cell["opponent"]["kind"] == "anchor"
        )
    )


def _check_selections(raw: dict[str, Any], member_ids: set[str]) -> None:
    first, *rest = raw["selections"]
    if first["kind"] == "health":
        _fail("the first selection must be the copy test's outcome (copy or none)")
    for entry in rest:
        if entry["kind"] != "health":
            _fail("at most one copy: every selection after the first must be a health replacement")
    for entry in rest:
        if entry["member"] not in member_ids:
            _fail("a health replacement names a member the record does not have")
    if not _ascending([entry["member"] for entry in rest]):
        _fail("health replacements must be sorted by member, each once")
    if first["kind"] == "copy":
        _check_copy(raw, first, rest, member_ids)


def _check_copy(
    raw: dict[str, Any], first: dict[str, Any], rest: list[dict[str, Any]], member_ids: set[str]
) -> None:
    loser, donor = first["loser"], first["donor"]
    if loser == donor:
        _fail("a copy's loser must differ from its donor")
    if loser not in member_ids or donor not in member_ids:
        _fail("a copy names a member the record does not have")
    if any(entry["member"] in (loser, donor) for entry in rest):
        _fail("a health replacement cannot name a copy's loser or donor")
    if first["lower_bound_ppm"] <= COPY_FLOOR_PPM + first["margin_ppm"]:
        _fail("a copy needs a lower bound above 500000 plus its margin")
    _check_settings(first["settings"])
    by_id = {member["member"]: member for member in raw["members"]}
    if any(
        m["strength"]["strength"] > by_id[donor]["strength"]["strength"] for m in raw["members"]
    ):
        _fail("a copy's donor must have the highest strength among the members")
    cell = next(
        (
            c
            for c in raw["cells"]
            if c["member"] == donor
            and c["opponent"]["kind"] == "member"
            and c["opponent"]["id"] == loser
        ),
        None,
    )
    if cell is None or cell["pairs"] != first["pairs"]:
        _fail("a copy's pairs must equal the donor-against-loser cell's pairs")
    donor_min = _anchor_minimum(raw, donor)
    loser_min = _anchor_minimum(raw, loser)
    if donor_min != first["donor_anchor_min_ppm"] or loser_min != first["loser_anchor_min_ppm"]:
        _fail("a copy's anchor minima must be the least anchor pair scores of its donor and loser")
    if donor_min < loser_min:
        _fail("a copy's donor cannot have a lower anchor minimum than its loser")
    before = len({m["root"] for m in raw["members"]})
    after = len(
        {by_id[donor]["root"] if m["member"] == loser else m["root"] for m in raw["members"]}
    )
    if first["roots_before"] != before or first["roots_after"] != after:
        _fail("a copy's roots must be counted from the members' roots")


def _check_cells(
    raw: dict[str, Any], members: set[str], snapshots: set[str], anchors: set[str]
) -> None:
    keys = [(c["member"], c["opponent"]["kind"], c["opponent"]["id"]) for c in raw["cells"]]
    if not _ascending(keys):
        _fail("cells must be sorted by member, opponent kind and opponent id, each once")
    for cell in raw["cells"]:
        if cell["member"] not in members:
            _fail("a cell names a member the record does not have")
        kind, ident = cell["opponent"]["kind"], cell["opponent"]["id"]
        known = members if kind == "member" else snapshots if kind == "snapshot" else anchors
        if ident not in known:
            _fail("a cell's opponent is not in the record")
        if kind == "member" and ident == cell["member"]:
            _fail("a member does not play itself in the table")
        if sum(cell["scores"]) != cell["pairs"]:
            _fail("a cell's pairs must equal the sum of its scores")
        weighted = sum(count * index for index, count in enumerate(cell["scores"]))
        if weighted * SCORE_STEP_PPM != cell["score_sum_ppm"]:
            _fail("a cell's score_sum_ppm must be 250000 times its weighted score counts")


def _check_previous(raw: dict[str, Any], previous: dict[str, Any]) -> None:
    if previous["run_id"] != raw["run_id"] or previous["run_config_hash"] != raw["run_config_hash"]:
        _fail("the previous record belongs to another run or config")
    if previous["generation"] + 1 != raw["generation"]:
        _fail("the previous record must be the record of the generation before")

    def anchor_key(anchors: list[dict[str, Any]]) -> str:
        return ",".join(f"{a['anchor_id']}={a['strength']}" for a in anchors)

    if anchor_key(previous["anchors"]) != anchor_key(raw["anchors"]):
        _fail("anchor strengths are frozen for the run: they differ from the previous record's")
    if [m["member"] for m in previous["members"]] != [m["member"] for m in raw["members"]]:
        _fail("the members must be the same set as in the previous record")
    now = {s["snapshot_id"]: s["strength"] for s in raw["snapshots"]}
    for snapshot in previous["snapshots"]:
        kept = now.get(snapshot["snapshot_id"])
        if kept is not None and kept != snapshot["strength"]:
            _fail(
                "a snapshot's strength and bounds are frozen at entry: "
                "they differ from the previous record's"
            )


def _check_window(raw: dict[str, Any]) -> None:
    window = raw["window"]
    if window["last_generation"] != raw["generation"]:
        _fail("the window must end at the record's generation")
    if window["first_generation"] != max(1, raw["generation"] - window["size"] + 1):
        _fail("the window must start size generations before its end, but not before generation 1")


def _check_anchors(raw: dict[str, Any]) -> None:
    if not _ascending([a["anchor_id"] for a in raw["anchors"]]):
        _fail("anchors must be sorted by anchor_id, each once")
    reference = next((a for a in raw["anchors"] if a["anchor_id"] == REFERENCE_ANCHOR), None)
    if reference is None or reference["strength"] != REFERENCE_STRENGTH:
        _fail("the reference anchor must be present at exactly 1000000000")


def _check_members(raw: dict[str, Any]) -> set[str]:
    if not _ascending([m["member"] for m in raw["members"]]):
        _fail("members must be sorted by member, each once")
    for member in raw["members"]:
        if member["strength"]["lower"] > member["strength"]["upper"]:
            _fail("a strength's lower bound cannot exceed its upper bound")
        _check_settings(member["settings"])
    if not _ascending([s["snapshot_id"] for s in raw["snapshots"]]):
        _fail("snapshots must be sorted by snapshot_id, each once")
    member_ids = {m["member"] for m in raw["members"]}
    for snapshot in raw["snapshots"]:
        if snapshot["snapshot_id"] != f"{snapshot['member']}@g{snapshot['generation']}":
            _fail("a snapshot's id must be its member and generation")
        if snapshot["generation"] >= raw["generation"]:
            _fail("a snapshot must be frozen in a generation before the record's")
        if snapshot["member"] not in member_ids:
            _fail("a snapshot names a member the record does not have")
        if snapshot["strength"]["lower"] > snapshot["strength"]["upper"]:
            _fail("a strength's lower bound cannot exceed its upper bound")
    return member_ids


def _check_config(raw: dict[str, Any], config: RunConfig) -> None:
    """The run key and hash, and the population rules the record was decided under."""
    if raw["run_id"] != config.run_key:
        _fail(f"run_id {raw['run_id']} is not the config's run key")
    if raw["run_config_hash"] != config.hash:
        _fail("run_config_hash is not the config's hash")
    population = config.raw.get("population")
    if population is None:
        _fail("the config has no population: a ladder record needs one")
    assert population is not None  # _fail raised otherwise
    # Exploiter seats are not rated members (ADR-021 sections 3, 6): the table holds members only.
    if len(raw["members"]) != population["members"]:
        _fail("the record's members must equal the config's population.members")
    if raw["window"]["size"] != population["rating"]["window"]:
        _fail("the window size must equal the config's population.rating.window")
    selection = population["selection"]
    first = raw["selections"][0]
    if first["kind"] == "copy":
        if first["margin_ppm"] != selection["margin_ppm"]:
            _fail("a copy's margin_ppm must equal the config's selection.margin_ppm")
        if first["pairs"] < selection["n_min"]:
            _fail("a copy needs at least the config's selection.n_min pairs")
        if first["roots_after"] < selection["min_roots"]:
            _fail("a copy must leave at least the config's selection.min_roots roots")
    elif first["kind"] == "none" and first["failed"] == "root_floor":
        # A copy changes the distinct roots by at most one, so the floor can only fail when the
        # members hold at most min_roots.
        if len({m["root"] for m in raw["members"]}) > selection["min_roots"]:
            _fail(
                "a root_floor refusal needs the members to hold at most "
                "the config's selection.min_roots roots"
            )


def read_ladder_record(
    raw: object,
    previous: object | None = None,
    config: RunConfig | None = None,
) -> LadderRecord:
    """Validate ``raw`` against the contract and the reader rules.

    ``previous`` (the record of the generation before, same run) adds the cross-generation rules;
    ``config`` ties the record to its run key and hash.
    """
    try:
        validate("ladder-record", raw)
    except ContractError as exc:
        raise LadderRecordError(f"invalid LadderRecord: {exc}") from exc
    assert isinstance(raw, dict)  # validate() accepted an object
    _check_window(raw)
    _check_anchors(raw)
    member_ids = _check_members(raw)
    _check_cells(
        raw,
        member_ids,
        {s["snapshot_id"] for s in raw["snapshots"]},
        {a["anchor_id"] for a in raw["anchors"]},
    )
    anchor_ids = [a["anchor_id"] for a in raw["anchors"]]
    for member in raw["members"]:
        mine = [
            c["opponent"]["id"]
            for c in raw["cells"]
            if c["member"] == member["member"] and c["opponent"]["kind"] == "anchor"
        ]
        if mine != anchor_ids:
            _fail("every member must have exactly one cell per anchor")
    m = len(raw["members"])
    if raw["cyclic_triples"] > m * (m - 1) * (m - 2) // 6:
        _fail("cyclic_triples cannot exceed the number of member triples")
    _check_selections(raw, member_ids)
    if previous is not None:
        assert isinstance(previous, dict)
        _check_previous(raw, previous)
    if config is not None:
        _check_config(raw, config)
    return LadderRecord(
        run_id=raw["run_id"],
        run_config_hash=raw["run_config_hash"],
        generation=raw["generation"],
        raw=raw,
    )
