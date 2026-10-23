"""The progress evaluation's records, read in Python (ADR-017 section 3).

``read_progress_record`` and ``read_game_record`` validate against the contracts (closed objects,
integers only, ``data_class`` the constant ``synthetic``) and then apply the rules the contract
profile cannot say. They are the same rules as the TS readers
(``packages/showdown-bridge/src/progress-records.ts``): the counts add up and agree with the
game rows, matches and hashes are sorted and unique, a stored game's choices are among its
candidates, and every hash and the seed recompute. ``read_progress_record`` also ties the record
to the run it claims (run key and config hash, the turn cap and latency budgets, each policy id
from its pair's checkpoints and the match's mode); the TS reader has no policy-id constants.
"""

from __future__ import annotations

import itertools
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, sha256_hex, validate

from areneon_ml.learner.config import RunConfig
from areneon_ml.loop.records import policy_id
from areneon_ml.progress.seeds import PAIRING, SEED_DERIVATION, progress_game_seed

SCHEMA_V2 = 2


class ProgressRecordError(ValueError):
    """A progress record breaks a rule."""


#: The counts a damage-aware baseline seat reports per game (ADR-018 section 4), in order.
BASELINE_COUNT_KEYS = (
    "decisions",
    "preview_decisions",
    "scored_decisions",
    "queries",
    "probe_runs",
    "unsupported_queries",
    "reader_fallbacks",
    "assumed_set_fallbacks",
    "epsilon_plays",
)


#: Says what a ``damage-aware-*`` row is: a scripted opponent the lineage has trained against.
DAMAGE_AWARE_LABEL = "seen scripted baseline: never held-out evidence"


def is_damage_aware(policy: str) -> bool:
    """Whether a policy id names the damage-aware baseline (``damage-aware@<commit>``)."""
    return policy.startswith("damage-aware@")


def sum_baseline_counts(counts: list[dict[str, int]]) -> dict[str, int]:
    """The per-game baseline counts summed, key by key."""
    return {key: sum(c[key] for c in counts) for key in BASELINE_COUNT_KEYS}


def _check_baseline_counts(counts: dict[str, int], what: str) -> None:
    if (
        counts["probe_runs"] > counts["queries"]
        or counts["unsupported_queries"] > counts["queries"]
    ):
        raise ProgressRecordError(
            f"{what}: probe runs and unsupported queries cannot exceed queries"
        )
    if counts["preview_decisions"] + counts["scored_decisions"] > counts["decisions"]:
        raise ProgressRecordError(f"{what}: preview and scored decisions cannot exceed decisions")


def _check_baseline_of_match(
    match: dict[str, Any], rows: list[dict[str, Any]], identities: set[str]
) -> None:
    """ADR-018 sections 4 and 5: a damage-aware match carries its identity hash and the sum of its
    rows' per-game counts, and no other match carries either; the counts recompute from the rows."""
    name = match["match_id"]
    opponent = match["opponent"]
    damage = opponent["kind"] == "baseline" and is_damage_aware(opponent["policy_id"])
    if not damage:
        if "baseline" in match or any("baseline_counts" in row for row in rows):
            raise ProgressRecordError(f"{name}: only a damage-aware match carries baseline counts")
        return
    if "baseline" not in match or any("baseline_counts" not in row for row in rows):
        raise ProgressRecordError(
            f"{name}: a damage-aware match carries baseline counts in its summary "
            "and in every game row"
        )
    identities.add(match["baseline"]["identity_hash"])
    per_game = [row["baseline_counts"] for row in rows]
    for index, counts in enumerate(per_game):
        _check_baseline_counts(counts, f"{name} game {index}")
    if match["baseline"]["counts"] != sum_baseline_counts(per_game):
        raise ProgressRecordError(
            f"{name}: baseline.counts must be the sum of the game rows' counts"
        )


def record_hash(record: object) -> str:
    """SHA-256 of a record's JCS bytes: the hash a record is named by."""
    return sha256_hex(jcs(record))


def log_hash(lines: list[str]) -> str:
    """SHA-256 of the JCS bytes of the omniscient log's lines, as a JSON array of strings."""
    return sha256_hex(jcs(lines))


def _ascending(values: list[str]) -> bool:
    return all(a < b for a, b in itertools.pairwise(values))


def _tally(rows: list[dict[str, Any]], max_turns: int) -> dict[str, Any]:
    """What a match's game rows say, recounted from the candidate's side."""
    won = tied = lost = cap_ties = turns = 0
    forfeits: dict[tuple[str, str], int] = {}
    for row in rows:
        outcome = row["outcome"]
        turns += outcome["turns"]
        if outcome["kind"] == "forfeit":
            key = (outcome["loser"], outcome["cause"])
            forfeits[key] = forfeits.get(key, 0) + 1
            lost += outcome["loser"] == "candidate"
            won += outcome["loser"] == "opponent"
        else:
            tied += outcome["winner"] == "tie"
            cap_ties += outcome["winner"] == "tie" and outcome["turns"] > max_turns
            won += outcome["winner"] == "candidate"
            lost += outcome["winner"] == "opponent"
    return {
        "wins": won,
        "ties": tied,
        "losses": lost,
        "turn_cap_ties": cap_ties,
        "turns_total": turns,
        "forfeits": forfeits,
    }


def _recount(match: dict[str, Any], rows: list[dict[str, Any]], max_turns: int) -> None:
    found = _tally(rows, max_turns)
    name = match["match_id"]
    if any(match[k] != found[k] for k in ("wins", "ties", "losses")):
        raise ProgressRecordError(f"{name}: wins, ties and losses must equal the game rows")
    if match["turn_cap_ties"] != found["turn_cap_ties"]:
        raise ProgressRecordError(f"{name}: turn_cap_ties must equal the tied games past max_turns")
    if match["turns_total"] != found["turns_total"]:
        raise ProgressRecordError(f"{name}: turns_total must equal the sum of the game rows' turns")
    listed = [(f["side"], f["cause"]) for f in match["forfeits"]]
    if not all(a < b for a, b in itertools.pairwise(listed)):
        raise ProgressRecordError(f"{name}: forfeits must be sorted by side then cause, each once")
    if {(f["side"], f["cause"]): f["count"] for f in match["forfeits"]} != found["forfeits"]:
        raise ProgressRecordError(f"{name}: forfeits must equal the forfeited game rows")


def _check_match(match: dict[str, Any], rows: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    name = match["match_id"]
    if match["games"] != plan["games_per_match"]:
        raise ProgressRecordError(f"{name}: games must equal plan.games_per_match")
    if match["wins"] + match["ties"] + match["losses"] != match["games"]:
        raise ProgressRecordError(f"{name}: wins + ties + losses must equal games")
    if len(rows) != match["games"]:
        raise ProgressRecordError(f"{name}: there must be one game row per game")
    for index, row in enumerate(rows):
        seat = "p1" if index % 2 == 0 else "p2"
        if (row["game"], row["pair"], row["candidate_seat"]) != (index, index, seat):
            raise ProgressRecordError(
                f"{name}: game rows must run 0..games-1, pair = game, seats alternating"
            )
    _recount(match, rows, plan["max_turns"])
    hashes = match["game_record_hashes"]
    if not _ascending(hashes) or len(hashes) != plan["recorded_games_per_match"]:
        raise ProgressRecordError(
            f"{name}: game_record_hashes must be sorted, unique, recorded_games_per_match long"
        )


def read_progress_record(raw: object, config: RunConfig | None = None) -> dict[str, Any]:
    """Validate a ProgressEvaluationRecord (schema version 1 or 2); with ``config`` also tie it to
    the run that made it. Version 1 is a record written before ADR-017 erratum E2 (one game per
    pair); version 2 plays every pair from both seats."""
    if isinstance(raw, dict) and raw.get("schema_version") == SCHEMA_V2:
        return _read_v2(raw, config)
    try:
        validate("progress-evaluation-record", raw)
    except ContractError as exc:
        raise ProgressRecordError(str(exc)) from exc
    assert isinstance(raw, dict)  # validate() accepted an object
    plan, matches, games = raw["plan"], raw["matches"], raw["games"]
    if raw["teams"]["pairs_used"] != plan["games_per_match"]:
        raise ProgressRecordError("teams.pairs_used must equal plan.games_per_match")
    if plan["recorded_games_per_match"] > plan["games_per_match"]:
        raise ProgressRecordError("recorded_games_per_match must not exceed games_per_match")
    ids = [m["match_id"] for m in matches]
    if not _ascending(ids):
        raise ProgressRecordError("matches must be sorted by match_id, each once")
    by_match: dict[str, list[dict[str, Any]]] = {i: [] for i in ids}
    for row in games:
        if row["match_id"] not in by_match:
            raise ProgressRecordError(f"a game row names the unknown match {row['match_id']}")
        by_match[row["match_id"]].append(row)
    if [row for i in ids for row in by_match[i]] != games:
        raise ProgressRecordError("game rows must be in match then game order")
    for match in matches:
        _check_match(match, by_match[match["match_id"]], plan)
    if plan["seed_derivation"] != SEED_DERIVATION or plan["pairing"] != PAIRING:
        raise ProgressRecordError("the plan names another seed derivation or pairing")
    if config is not None:
        _check_run(raw, config)
    return raw


def is_v2(record: dict[str, Any]) -> bool:
    """Whether a record is a both-seats ProgressEvaluationRecordV2."""
    return bool(record["schema_version"] == SCHEMA_V2)


#: Names the uniformly random Preview in a policy id and a stream label (ticket #940).
RANDOM_PREVIEW_POLICY = "random-preview@v1"


def candidate_preview_id(preview: dict[str, Any] | None, own: str) -> str:
    """The Preview part of a candidate's policy id: its own checkpoint, the random word or the
    checkpoint of the earlier generation whose Preview it used."""
    if preview is None or preview["kind"] == "learned":
        return own
    return RANDOM_PREVIEW_POLICY if preview["kind"] == "random" else str(preview["checkpoint"])


def preview_variant(preview: dict[str, Any]) -> str:
    """The word an ablation match id carries for its candidate's Preview."""
    return f"g{preview['generation']}" if preview["kind"] == "generation" else str(preview["kind"])


def _check_ablation_match(match: dict[str, Any], generation: int) -> None:
    """A Preview ablation match (ticket #940) plays a baseline and is named for its variant."""
    preview = match.get("candidate_preview")
    if preview is None:
        return
    name, opponent = match["match_id"], match["opponent"]
    if match["mode"] != "argmax":
        raise ProgressRecordError(f"{name}: an ablation match is played in argmax")
    if opponent["kind"] != "baseline":
        raise ProgressRecordError(f"{name}: an ablation match plays a scripted baseline")
    if preview["kind"] == "generation" and preview["generation"] >= generation:
        raise ProgressRecordError(f"{name}: an ablation Preview comes from an earlier generation")
    baseline = opponent["policy_id"].split("@")[0]
    if name != f"preview-{preview_variant(preview)}-{baseline}-{match['mode']}":
        raise ProgressRecordError(
            f"{name}: an ablation match is named preview-<variant>-<baseline>-<mode>"
        )


def _check_match_v2(
    match: dict[str, Any],
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    generation: int,
    identities: set[str],
) -> None:
    name = match["match_id"]
    opponent = match["opponent"]
    _check_ablation_match(match, generation)
    if opponent["kind"] == "anchor":
        if (
            opponent["generation"] >= generation
            or name != f"anchor-g{opponent['generation']}-{match['mode']}"
        ):
            raise ProgressRecordError(
                f"{name}: an anchor match is named anchor-g<generation>-<mode> "
                "and plays an earlier generation"
            )
        if match["games"] != plan["anchor_games_per_match"]:
            raise ProgressRecordError(f"{name}: games must equal plan.anchor_games_per_match")
    elif match["games"] != plan["games_per_match"]:
        raise ProgressRecordError(f"{name}: games must equal plan.games_per_match")
    if match["wins"] + match["ties"] + match["losses"] != match["games"]:
        raise ProgressRecordError(f"{name}: wins + ties + losses must equal games")
    if len(rows) != match["games"]:
        raise ProgressRecordError(f"{name}: there must be one game row per game")
    for index, row in enumerate(rows):
        seat = "p1" if index % 2 == 0 else "p2"
        if (row["game"], row["pair"], row["candidate_seat"]) != (index, index // 2, seat):
            raise ProgressRecordError(
                f"{name}: game rows must run 0..games-1, pair = game // 2, seats alternating"
            )
    _recount(match, rows, plan["max_turns"])
    _check_baseline_of_match(match, rows, identities)
    hashes = match["game_record_hashes"]
    if not _ascending(hashes) or len(hashes) != min(
        plan["recorded_games_per_match"], match["games"]
    ):
        raise ProgressRecordError(
            f"{name}: game_record_hashes must be sorted, unique, recorded_games_per_match long"
        )


def _check_matches_v2(
    matches: list[dict[str, Any]],
    by_match: dict[str, list[dict[str, Any]]],
    plan: dict[str, Any],
    generation: int,
) -> None:
    identities: set[str] = set()
    for match in matches:
        _check_match_v2(match, by_match[match["match_id"]], plan, generation, identities)
    if len(identities) > 1:
        raise ProgressRecordError(
            "every damage-aware match of a record must carry one baseline identity hash"
        )


def _check_not_mixed(matches: list[dict[str, Any]]) -> None:
    if len({m.get("candidate_preview") is None for m in matches}) > 1:
        raise ProgressRecordError("a Preview ablation record holds ablation matches only")


def _read_v2(raw: dict[str, Any], config: RunConfig | None) -> dict[str, Any]:
    try:
        validate("progress-evaluation-record-2", raw)
    except ContractError as exc:
        raise ProgressRecordError(str(exc)) from exc
    plan, matches, games = raw["plan"], raw["matches"], raw["games"]
    anchor_games = plan.get("anchor_games_per_match")
    anchored = any(m["opponent"]["kind"] == "anchor" for m in matches)
    if anchored != (anchor_games is not None):
        raise ProgressRecordError(
            "plan.anchor_games_per_match is present exactly when the record has anchor matches"
        )
    if plan["games_per_match"] % 2 or (anchor_games or 0) % 2:
        raise ProgressRecordError(
            "games_per_match and anchor_games_per_match must be even: "
            "every pair is played from both seats"
        )
    if raw["teams"]["pairs_used"] != max(plan["games_per_match"], anchor_games or 0) // 2:
        raise ProgressRecordError(
            "teams.pairs_used must be half the largest games_per_match or anchor_games_per_match"
        )
    if plan["recorded_games_per_match"] > plan["games_per_match"]:
        raise ProgressRecordError("recorded_games_per_match must not exceed games_per_match")
    ids = [m["match_id"] for m in matches]
    if not _ascending(ids):
        raise ProgressRecordError("matches must be sorted by match_id, each once")
    _check_not_mixed(matches)
    by_match: dict[str, list[dict[str, Any]]] = {i: [] for i in ids}
    for row in games:
        if row["match_id"] not in by_match:
            raise ProgressRecordError(f"a game row names the unknown match {row['match_id']}")
        by_match[row["match_id"]].append(row)
    if [row for i in ids for row in by_match[i]] != games:
        raise ProgressRecordError("game rows must be in match then game order")
    _check_matches_v2(matches, by_match, plan, raw["generation"])
    if plan["seed_derivation"] != SEED_DERIVATION or plan["pairing"] != PAIRING:
        raise ProgressRecordError("the plan names another seed derivation or pairing")
    if config is not None:
        _check_run(raw, config)
    return raw


def _check_run(raw: dict[str, Any], config: RunConfig) -> None:
    if raw["run_id"] != config.run_key or raw["run_config_hash"] != config.hash:
        raise ProgressRecordError("the record belongs to another run or run config")
    plan = raw["plan"]
    gen = config.generation
    if plan["max_turns"] != gen.max_turns or plan["latency_budget_ms"] != {
        "preview_ms": gen.preview_ms,
        "battle_ms": gen.battle_ms,
    }:
        raise ProgressRecordError(
            "the plan's turn cap or latency budgets differ from the run config's"
        )
    pair = raw["checkpoints"]
    for match in raw["matches"]:
        if match["policy_id"] != policy_id(
            candidate_preview_id(match.get("candidate_preview"), pair["preview"]),
            pair["battle"],
            match["mode"],
        ):
            raise ProgressRecordError(
                f"{match['match_id']}: policy_id is not the frozen pair's in this mode"
            )
        opponent = match["opponent"]
        if opponent["kind"] in ("previous", "anchor") and opponent["policy_id"] != policy_id(
            opponent["checkpoints"]["preview"], opponent["checkpoints"]["battle"], match["mode"]
        ):
            raise ProgressRecordError(
                f"{match['match_id']}: the {opponent['kind']} pair's policy_id "
                "is not its checkpoints'"
            )


def _check_game_baseline(raw: dict[str, Any]) -> None:
    baseline = raw.get("baseline")
    policies = raw["policies"]
    if baseline is None:
        if is_damage_aware(policies["p1"]) or is_damage_aware(policies["p2"]):
            raise ProgressRecordError(
                "a game with a damage-aware seat must carry the baseline counts"
            )
        return
    seat = baseline["seat"]
    other = "p2" if seat == "p1" else "p1"
    if not is_damage_aware(policies[seat]) or is_damage_aware(policies[other]):
        raise ProgressRecordError("baseline.seat must be the one seat whose policy is damage-aware")
    _check_baseline_counts(baseline["counts"], "baseline")
    # The actor counts every request it answered; a forfeit can stop it on one it never answered.
    answered = sum(1 for d in raw["decisions"] if d["player"] == seat)
    forfeited = 1 if raw["outcome"]["kind"] == "forfeit" else 0
    if not answered <= baseline["counts"]["decisions"] <= answered + forfeited:
        raise ProgressRecordError("baseline.counts.decisions must be the baseline seat's decisions")


def check_game_baseline(stored: dict[str, Any], match: dict[str, Any], row: dict[str, Any]) -> None:
    """A stored game's baseline fields are its match's and its row's: the seat the candidate does
    not hold, the match's identity hash and the row's counts (none outside a damage-aware match)."""
    found = stored.get("baseline")
    if "baseline" not in match:
        if found is not None:
            raise ProgressRecordError("a game outside a damage-aware match carries no baseline")
        return
    seat = "p2" if row["candidate_seat"] == "p1" else "p1"
    expected = {
        "seat": seat,
        "identity_hash": match["baseline"]["identity_hash"],
        "counts": row["baseline_counts"],
    }
    if found != expected:
        raise ProgressRecordError(
            "the stored game's baseline is not its match's identity and its row's counts"
        )


def read_game_record(raw: object) -> dict[str, Any]:
    """Validate a ProgressGameRecord: choices among candidates; hashes and seed recompute."""
    try:
        validate("progress-game-record", raw)
    except ContractError as exc:
        raise ProgressRecordError(str(exc)) from exc
    assert isinstance(raw, dict)  # validate() accepted an object
    for decision in raw["decisions"]:
        if decision["chosen"] not in [c["choice"] for c in decision["candidates"]]:
            raise ProgressRecordError(
                f"the choice of {decision['player']} at request {decision['request_seq']} "
                "is not among its candidates"
            )
    for seat in ("p1", "p2"):
        if sha256_hex(jcs(raw["teams"][seat]["team"])) != raw["teams"][seat]["team_hash"]:
            raise ProgressRecordError(f"the team hash of {seat} does not recompute")
    if log_hash(raw["omniscient_log"]) != raw["omniscient_log_sha256"]:
        raise ProgressRecordError("omniscient_log_sha256 does not recompute")
    _check_game_baseline(raw)
    # A game of a both-seats evaluation names its pair: both games of a pair share the pair's seed.
    pair = raw.get("pair")
    if pair is not None and pair != raw["game"] // 2:
        raise ProgressRecordError("pair must be the game index divided by two")
    if progress_game_seed(raw["run_id"], raw["game"] if pair is None else pair) != raw["seed"]:
        raise ProgressRecordError(
            "the seed is not the one progress-seed/1 gives this game's pair (or game index)"
        )
    return raw
