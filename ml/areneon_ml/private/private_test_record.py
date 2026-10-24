"""The PrivateTestRecord, read and displayed in Python (ADR-015 erratum E1, item 2).

``read_private_test_record`` validates against the contract (closed objects, integers only,
``data_class`` the constant ``private``) and then applies the rules the contract profile cannot say.
They are the rules of the TS reader (``packages/showdown-bridge/src/private-test-record.ts``): the
counts add up, the pair-score histogram agrees with wins and ties, matches are sorted and unique and
each is named for its opponent and mode, the source fields agree with the source kind, and a pair
with private ancestry is tested only on a snapshot's hold-out teams or a public dataset's teams. A
message states a rule or a path and never repeats a value.

The interval is derived on display, never stored: a percentile bootstrap over the test set's
disjoint pairs (``areneon_ml.progress.stats.pair_interval``), seeded from the set's commitment
and the match id, so the same record always shows the same numbers. It is descriptive and always
carries ``LABEL``.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Any

from areneon_contracts.contracts import jcs, schema
from jsonschema import Draft202012Validator

from areneon_ml.progress.stats import PPM, PairInterval, pair_interval

#: What every display of a private test says it is.
LABEL = "private test set; not the Training gate; not validation"
#: What the interval is, in words.
INTERVAL_LABEL = "pair-level percentile bootstrap over the disjoint pairs, descriptive"
BOOTSTRAP_PURPOSE = "private-test-bootstrap/1"
SEED_DERIVATION = "private-test-seed/1"
PAIRING = "private-test-pairing/1"
BASELINES = frozenset({"random-legal", "actor-rule", "damage-aware"})
HALVES_PER_GAME = 2


class PrivateTestRecordError(ValueError):
    """A private test record breaks a rule."""


_validator = Draft202012Validator(schema("private-test-record"))


def opponent_name(policy_id: str) -> str:
    """The opponent's name in a match id: a baseline's name, or ``pair`` for another frozen pair."""
    return "pair" if policy_id.startswith("policy:") else policy_id.split("@", 1)[0]


def private_test_allowed(source_kind: str, subset: str) -> bool:
    """Whether a pair with private ancestry may be tested on this source (see the TS twin)."""
    return (source_kind == "snapshot" and subset == "holdout") or source_kind == "dataset"


def _fail(message: str) -> None:
    raise PrivateTestRecordError(message)


def read_private_test_record(raw: object) -> dict[str, Any]:
    """Validate a PrivateTestRecord: the schema, then the arithmetic, order and source rules."""
    paths = sorted({"/" + "/".join(map(str, e.absolute_path)) for e in _validator.iter_errors(raw)})
    if paths:
        raise PrivateTestRecordError(f"invalid PrivateTestRecord at {', '.join(paths)}")
    assert isinstance(raw, dict)  # the schema accepted an object
    record: dict[str, Any] = raw
    test_set, plan, matches = record["test_set"], record["plan"], record["matches"]
    kind, subset = test_set["source_kind"], test_set["subset"]
    if (kind == "snapshot") != ("snapshot_hash" in test_set):
        _fail("test_set.snapshot_hash must be present exactly for the snapshot source")
    if (kind == "dataset") != ("dataset_hash" in test_set):
        _fail("test_set.dataset_hash must be present exactly for the dataset source")
    if subset == "holdout" and kind != "snapshot":
        _fail("the holdout subset exists only in a snapshot")
    if test_set["pairs"] * 2 > test_set["team_count"]:
        _fail("test_set.pairs must not exceed half the teams")
    if plan["games_per_match"] != test_set["pairs"] * 2:
        _fail("plan.games_per_match must be twice test_set.pairs")
    if record["candidate_data_class"] == "private" and not private_test_allowed(kind, subset):
        _fail(
            "a pair with private ancestry is tested on hold-out teams of a snapshot "
            "or on a public dataset's teams only"
        )
    ids = [m["match_id"] for m in matches]
    if any(b <= a for a, b in itertools.pairwise(ids)):
        _fail("matches must be sorted by match_id, each once")
    for match in matches:
        _check_match(match, plan, test_set)
    return record


def _check_match(match: dict[str, Any], plan: dict[str, Any], test_set: dict[str, Any]) -> None:
    name = opponent_name(match["opponent_policy_id"])
    if match["match_id"] != f"{name}-{match['mode']}":
        _fail("a match is named <opponent>-<mode>")
    if name != "pair" and name not in BASELINES:
        _fail("an opponent that is not a pair must be a known scripted baseline")
    if (name == "pair") != ("opponent_checkpoints" in match):
        _fail("opponent_checkpoints are present exactly for a pair opponent")
    if match["games"] != plan["games_per_match"] or match["pairs"] != test_set["pairs"]:
        _fail("every match plays every pair from both seats")
    if match["wins"] + match["ties"] + match["losses"] != match["games"]:
        _fail("wins + ties + losses must equal games")
    if match["turn_cap_ties"] > match["ties"]:
        _fail("turn_cap_ties must not exceed ties")
    counts = match["pair_score_counts"]
    if sum(counts) != match["pairs"]:
        _fail("pair_score_counts must sum to the pairs")
    if sum(k * c for k, c in enumerate(counts)) != HALVES_PER_GAME * match["wins"] + match["ties"]:
        _fail("pair_score_counts must agree with wins and ties in half-games")
    keys = [(f["side"], f["cause"]) for f in match["forfeits"]]
    if any(b <= a for a, b in itertools.pairwise(keys)):
        _fail("forfeits must be sorted by side then cause, each pair once")

    def by(side: str) -> int:
        return sum(f["count"] for f in match["forfeits"] if f["side"] == side)

    if by("candidate") > match["losses"] or by("opponent") > match["wins"]:
        _fail("forfeits cannot exceed the losses or wins they decided")


def record_hash(record: dict[str, Any]) -> str:
    """SHA-256 of a record's JCS bytes."""
    return hashlib.sha256(jcs(record)).hexdigest()


def bootstrap_key(commitment: str, match_id: str) -> bytes:
    """The 32-byte key of one match's resampling stream: SHA-256 over the JCS of
    ``{commitment, match_id, purpose: "private-test-bootstrap/1"}``. It depends on the record alone
    (the commitment is a function of the salt), so any reader reproduces the interval."""
    return hashlib.sha256(
        jcs({"commitment": commitment, "match_id": match_id, "purpose": BOOTSTRAP_PURPOSE})
    ).digest()


def pair_scores(match: dict[str, Any]) -> list[int]:
    """The pairs' scores in half-games (0 to 4), ascending, expanded from the histogram."""
    return [k for k, count in enumerate(match["pair_score_counts"]) for _ in range(count)]


def match_interval(record: dict[str, Any], match: dict[str, Any]) -> PairInterval:
    """The candidate's mean pair score and its bootstrap interval in parts per million."""
    return pair_interval(
        pair_scores(match), bootstrap_key(record["test_set"]["commitment"], match["match_id"])
    )


def summary_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Counts, score and interval per match: everything a display shows."""
    out = []
    for match in record["matches"]:
        interval = match_interval(record, match)
        out.append(
            {
                "match_id": match["match_id"],
                "games": match["games"],
                "wins": match["wins"],
                "ties": match["ties"],
                "losses": match["losses"],
                "turn_cap_ties": match["turn_cap_ties"],
                "forfeits": match["forfeits"],
                "pairs": match["pairs"],
                "pair_score_sum": sum(pair_scores(match)),
                "score_ppm": interval.mean_ppm,
                "low_ppm": interval.low_ppm,
                "high_ppm": interval.high_ppm,
            }
        )
    return out


__all__ = [
    "INTERVAL_LABEL",
    "LABEL",
    "PPM",
    "PrivateTestRecordError",
    "bootstrap_key",
    "match_interval",
    "opponent_name",
    "pair_scores",
    "private_test_allowed",
    "read_private_test_record",
    "record_hash",
    "summary_of",
]
