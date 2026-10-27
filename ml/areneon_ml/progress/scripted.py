"""The scripted head-to-head, in Python (ADR-025 section 13, ticket #1130).

Two scripted baselines (``damage-aware@v1`` and ``@v2`` here; no learned policy, no worker) are
played against each other on the ADR-017 E2 plan: every validation pair from both seats,
``progress-seed/1`` seeds, the pair as the unit. The Node side
(``packages/team-dataset/scripts/scripted-h2h.ts``) plays the games over parallel game processes and
writes a ``ScriptedHeadToHeadRecord``. This module runs it, reads the record back with its own
reader (the same rules as the TS reader ``packages/showdown-bridge/src/scripted-h2h-record.ts``),
ties it to the request, derives the pair-level interval of ``stats.pair_interval`` (never stored),
logs the look in the validation look log and only then commits the directory and shows the result.

It is a progress view on validation teams, never the Training gate, and a validation result is never
quoted as held-out evidence.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex, validate

from areneon_ml.learner.config import RunConfigError, read_run_config
from areneon_ml.loop.play import PlayError
from areneon_ml.progress.looks import Look, append_look, looks_path
from areneon_ml.progress.records import BASELINE_COUNT_KEYS, is_damage_aware, sum_baseline_counts
from areneon_ml.progress.stats import PPM, bootstrap_stream_key, pair_interval

CONTRACT = "scripted-head-to-head-record"
SCRIPT = "packages/team-dataset/scripts/scripted-h2h.ts"
RECORD = "head-to-head.json"
LABEL = "scripted head-to-head on validation teams; not the Training gate; no learned policy"
#: The scripted baselines a head-to-head can play (the TS side checks the same list).
ARMS = ("actor-rule@v1", "random-legal@v1", "damage-aware@v1", "damage-aware@v2")
#: How long past the declared wall-time cap the child is given before it is killed (the cap itself
#: is checked by the Node side between games; a game in flight is bounded by the battle budget).
_KILL_SLACK_S = 600.0
_STDERR_TAIL = 2000
_BELIEF_KEYS = (
    "decisions",
    "worlds_sampled",
    "redraws",
    "contradicted_worlds",
    "no_consistent_world",
)
_HALVES = {"a": 2, "tie": 1, "b": 0}


class ScriptedError(ValueError):
    """A scripted head-to-head is refused, or its record breaks a rule."""


def _fail(message: str) -> None:
    raise ScriptedError(message)


def plays_on_belief(policy: str) -> bool:
    """Whether a policy id plays on belief worlds (``damage-aware@v2``)."""
    return policy == "damage-aware@v2"


def _sum_belief(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {key: sum(r[key] for r in rows) for key in _BELIEF_KEYS}
    refusals: dict[str, int] = {}
    for row in rows:
        for item in row["refusals"]:
            refusals[item["code"]] = refusals.get(item["code"], 0) + item["count"]
    total["refusals"] = [{"code": c, "count": refusals[c]} for c in sorted(refusals)]
    return total


def _tally(rows: list[dict[str, Any]], max_turns: int) -> dict[str, Any]:
    won_a = won_b = tied = cap_ties = turns = 0
    forfeits: dict[tuple[str, str], int] = {}
    for row in rows:
        outcome = row["outcome"]
        turns += outcome["turns"]
        if outcome["kind"] == "forfeit":
            key = (outcome["loser"], outcome["cause"])
            forfeits[key] = forfeits.get(key, 0) + 1
            won_b += outcome["loser"] == "a"
            won_a += outcome["loser"] == "b"
        elif outcome["winner"] == "tie":
            tied += 1
            cap_ties += outcome["turns"] > max_turns
        else:
            won_a += outcome["winner"] == "a"
            won_b += outcome["winner"] == "b"
    return {
        "wins_a": won_a,
        "wins_b": won_b,
        "ties": tied,
        "turn_cap_ties": cap_ties,
        "turns_total": turns,
        "forfeits": [{"side": s, "cause": c, "count": n} for (s, c), n in sorted(forfeits.items())],
    }


def _check_counts(counts: dict[str, int], what: str) -> None:
    refused = counts.get("probe_refusals", 0)
    if max(counts["probe_runs"], counts["unsupported_queries"], refused) > counts["queries"]:
        _fail(f"{what}: probe runs, unsupported queries and probe refusals cannot exceed queries")
    if counts["preview_decisions"] + counts["scored_decisions"] > counts["decisions"]:
        _fail(f"{what}: preview and scored decisions cannot exceed decisions")


def _check_belief(counts: dict[str, Any], what: str) -> None:
    codes = [r["code"] for r in counts["refusals"]]
    if any(a >= b for a, b in itertools.pairwise(codes)):
        _fail(f"{what}: belief refusals must be sorted by code, each once")
    refused = sum(r["count"] for r in counts["refusals"])
    if counts["no_consistent_world"] + refused > counts["decisions"]:
        _fail(
            f"{what}: refusals and requests without a consistent world "
            "cannot exceed the belief's decisions"
        )


def _check_arm(side: str, arm: dict[str, Any], games: list[dict[str, Any]]) -> None:
    """One arm's presence rules and its sums over the game rows."""
    damage = is_damage_aware(arm["policy_id"])
    if damage != ("identity_hash" in arm) or damage != ("counts" in arm):
        _fail(f"arm {side}: identity_hash and counts are present exactly for a damage-aware arm")
    if plays_on_belief(arm["policy_id"]) != ("belief" in arm):
        _fail(f"arm {side}: belief is present exactly for an arm that plays on belief worlds")
    counts = [r.get(f"{side}_counts") for r in games]
    beliefs = [r.get(f"{side}_belief") for r in games]
    if any((c is not None) != damage for c in counts):
        _fail(f"arm {side}: every game row carries counts exactly when the arm is damage-aware")
    if any((b is not None) != ("belief" in arm) for b in beliefs):
        _fail(f"arm {side}: every game row carries belief counts exactly when the arm has a belief")
    if damage:
        for index, c in enumerate(counts):
            _check_counts(c, f"arm {side} game {index}")  # type: ignore[arg-type]
        _check_counts(arm["counts"], f"arm {side}")
        if arm["counts"] != sum_baseline_counts(counts):  # type: ignore[arg-type]
            _fail(f"arm {side}: counts must be the sum of the game rows' counts")
    if "belief" in arm:
        for index, b in enumerate(beliefs):
            _check_belief(b, f"arm {side} game {index}")  # type: ignore[arg-type]
        _check_belief(arm["belief"]["totals"], f"arm {side}")
        if arm["belief"]["totals"] != _sum_belief(beliefs):  # type: ignore[arg-type]
            _fail(f"arm {side}: belief totals must be the sum of the game rows' belief counts")


def read_scripted_record(value: object) -> dict[str, Any]:
    """Validate against the contract, then apply the rules the profile cannot say."""
    validate(CONTRACT, value)
    record: dict[str, Any] = json.loads(json.dumps(value))
    plan, teams, arms, games, result = (
        record["plan"],
        record["teams"],
        record["arms"],
        record["games"],
        record["result"],
    )
    if plan["games"] % 2:
        _fail("plan.games must be even: every pair is played from both seats")
    if teams["pairs_used"] != plan["games"] // 2:
        _fail("teams.pairs_used must be half of plan.games")
    if arms["a"]["policy_id"] == arms["b"]["policy_id"]:
        _fail("the two arms must be different baselines")
    if len(games) != plan["games"]:
        _fail("there must be one game row per game")
    for side in ("a", "b"):
        _check_arm(side, arms[side], games)
    for index, row in enumerate(games):
        if row["game"] != index or row["pair"] != index >> 1:
            _fail("game rows must run 0..games-1 with pair equal to the game divided by two")
        if row["a_seat"] != ("p1" if index % 2 == 0 else "p2"):
            _fail("arm a sits on p1 for an even game and p2 for an odd one")
    if result["wins_a"] + result["wins_b"] + result["ties"] != plan["games"]:
        _fail("wins_a + wins_b + ties must equal plan.games")
    listed = [(f["side"], f["cause"]) for f in result["forfeits"]]
    if any(a >= b for a, b in itertools.pairwise(listed)):
        _fail("forfeits must be sorted by arm then cause, each pair once")
    if result != _tally(games, plan["max_turns"]):
        _fail("the result must equal the game rows")
    return record


def record_hash(record: object) -> str:
    """SHA-256 of a record's JCS bytes."""
    return sha256_hex(jcs(record))


def pair_scores(record: dict[str, Any]) -> list[int]:
    """Per pair, arm a's score in half-games (0 to 4): a win is 2, a tie 1, a forfeit a loss."""
    halves = []
    for row in record["games"]:
        outcome = row["outcome"]
        if outcome["kind"] == "forfeit":
            halves.append(0 if outcome["loser"] == "a" else 2)
        else:
            halves.append(_HALVES[outcome["winner"]])
    return [halves[i] + halves[i + 1] for i in range(0, len(halves), 2)]


def _interval(values: list[int], key: bytes) -> dict[str, int]:
    found = pair_interval(values, key)
    return {"score_ppm": found.mean_ppm, "low_ppm": found.low_ppm, "high_ppm": found.high_ppm}


def summarise(record: dict[str, Any]) -> dict[str, Any]:
    """Descriptive summary: the pair-level percentile bootstrap interval of each arm's pair score.

    The resampling stream is seeded from the run key and the two policy ids, so a rerun shows the
    same interval. It is the pair-level interval of ``stats.pair_interval``, labelled descriptive
    and conditional on the fixed validation plan, with no multiplicity control.
    """
    arms = record["arms"]
    a, b = arms["a"]["policy_id"], arms["b"]["policy_id"]
    scores = pair_scores(record)
    key = bootstrap_stream_key(record["run_id"], 0, f"scripted-h2h:{a}:{b}")
    mirrored = [4 - s for s in scores]
    histogram = [scores.count(h) for h in range(5)]
    result = record["result"]
    out: dict[str, Any] = {
        "label": LABEL,
        "run_id": record["run_id"],
        "arms": {"a": a, "b": b},
        "pairs": len(scores),
        "games": record["plan"]["games"],
        "a": _interval(scores, key),
        "b": _interval(mirrored, key),
        "pair_score_counts": histogram,
        "wins_a": result["wins_a"],
        "wins_b": result["wins_b"],
        "ties": result["ties"],
        "turn_cap_ties": result["turn_cap_ties"],
        "forfeits": result["forfeits"],
        "forfeit_share_ppm": sum(f["count"] for f in result["forfeits"])
        * PPM
        // record["plan"]["games"],
    }
    out["a"]["diff_ppm"] = out["a"]["score_ppm"] - PPM // 2
    out["b"]["diff_ppm"] = out["b"]["score_ppm"] - PPM // 2
    for side in ("a", "b"):
        arm = arms[side]
        if "identity_hash" in arm:
            out[side]["identity_hash"] = arm["identity_hash"]
        if "counts" in arm:
            out[f"{side}_counts"] = {
                k: arm["counts"].get(k, 0) for k in (*BASELINE_COUNT_KEYS, "probe_refusals")
            }
        if "belief" in arm:
            out[f"{side}_belief"] = arm["belief"]
    return out


@dataclass(frozen=True)
class ScriptedRequest:
    """What one head-to-head asks for."""

    config: Path
    arm_a: str
    arm_b: str
    pairs: int
    out: Path
    #: An evaluation-only dataset's directory and hash, instead of the config's team dataset.
    dataset_dir: Path | None = None
    dataset_hash: str | None = None
    game_processes: int = 1
    wall_time_cap_minutes: int | None = None
    #: The look log; default the sibling ``<out>.looks.jsonl``.
    looks: Path | None = None


def _run_node(node: str, repo: Path, args: list[str], timeout_s: float | None) -> str:
    command = [node, str(repo / SCRIPT), *args]
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    except OSError as exc:
        raise PlayError(f"cannot start {node}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise PlayError(f"{SCRIPT} was killed after {timeout_s:g} s (the wall-time cap)") from exc
    if done.returncode != 0:
        raise PlayError(f"{SCRIPT} failed (exit {done.returncode}): {done.stderr[-_STDERR_TAIL:]}")
    return done.stdout


def match_id(arm_a: str, arm_b: str) -> str:
    """The name a look gives the head-to-head (an id token: ``@`` becomes ``-``)."""
    return f"{arm_a}-vs-{arm_b}".replace("@", "-")


def scripted_h2h(request: ScriptedRequest, *, repo: Path, node: str = "node") -> dict[str, Any]:
    """Run, read back, tie, log the look, commit; return the summary."""
    if request.arm_a not in ARMS or request.arm_b not in ARMS:
        raise ScriptedError(f"the arms are scripted baselines: {', '.join(ARMS)}")
    if request.arm_a == request.arm_b:
        raise ScriptedError("the two arms must be different baselines")
    if request.pairs < 1:
        raise ScriptedError("--pairs must be at least 1")
    if (request.dataset_dir is None) != (request.dataset_hash is None):
        raise ScriptedError("--dataset-dir and --dataset-hash go together")
    config = read_run_config(json.loads(request.config.read_text(encoding="utf-8")))
    if config.run_key.startswith("CHANGE-ME"):
        raise RunConfigError(
            "seeds.run_key is still the CHANGE-ME placeholder: give the run its own name"
        )
    out = request.out.resolve()
    if out.exists():
        raise ScriptedError("the output directory exists: a head-to-head is never overwritten")
    partial = out.with_name(f".{out.name}.partial")
    log = (request.looks if request.looks is not None else looks_path(out)).resolve()
    if log in (partial, out) or partial in log.parents or out in log.parents:
        raise ScriptedError("the look log must not be inside the output directory")
    cap_ms = (
        None if request.wall_time_cap_minutes is None else request.wall_time_cap_minutes * 60_000
    )
    shutil.rmtree(partial, ignore_errors=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    args = [
        *("--config", str(request.config.resolve())),
        *("--arm-a", request.arm_a),
        *("--arm-b", request.arm_b),
        *("--pairs", str(request.pairs)),
        *("--out", str(partial)),
        *(
            ("--dataset-dir", str(request.dataset_dir), "--dataset-hash", str(request.dataset_hash))
            if request.dataset_dir is not None
            else ()
        ),
        *(("--game-processes", str(request.game_processes)) if request.game_processes != 1 else ()),
        *(("--wall-time-cap-ms", str(cap_ms)) if cap_ms is not None else ()),
    ]
    timeout = None if cap_ms is None else cap_ms / 1000 + _KILL_SLACK_S
    try:
        line = _run_node(node, repo, args, timeout)
        try:
            printed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PlayError(f"{SCRIPT} did not print a JSON line") from exc
        record = read_scripted_record(json.loads((partial / RECORD).read_text(encoding="utf-8")))
        _tie(record, request, config, cap_ms)
        digest = record_hash(record)
        if not isinstance(printed, dict) or printed.get("record_hash") != digest:
            raise ScriptedError("the record on disk is not the one the play script reported")
        summary = summarise(record)
        # A validation result is about to be shown: log the look first (a refused run logs nothing).
        state = append_look(
            log,
            "scripted-h2h",
            config.run_key,
            [Look(0, match_id(request.arm_a, request.arm_b), record["plan"]["games"])],
        )
        os.replace(partial, out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return {
        "out": str(out),
        "record_hash": digest,
        "summary": summary,
        "looks": state.count,
        "look_log": str(log),
        "look_log_sha256": state.file_sha256,
        "wall_ms": printed.get("wall_ms"),
    }


def _tie(record: dict[str, Any], request: ScriptedRequest, config: Any, cap_ms: int | None) -> None:
    """The record must be the head-to-head that was asked for."""
    plan, arms = record["plan"], record["arms"]
    if record["run_id"] != config.run_key:
        raise ScriptedError("the record's run key is not the config's")
    if (arms["a"]["policy_id"], arms["b"]["policy_id"]) != (request.arm_a, request.arm_b):
        raise ScriptedError("the record's arms are not the ones asked for")
    if record["teams"]["pairs_used"] != request.pairs:
        raise ScriptedError("the record's pairs are not the number asked for")
    gen = config.generation
    if plan["max_turns"] != gen.max_turns or plan["latency_budget_ms"] != {
        "preview_ms": gen.preview_ms,
        "battle_ms": gen.battle_ms,
    }:
        raise ScriptedError("the record's turn cap and latency budgets are not the config's")
    if plan.get("wall_time_cap_ms") != cap_ms:
        raise ScriptedError("the record's wall-time cap is not the one asked for")
    if request.dataset_hash is not None and record["teams"]["dataset_hash"] != request.dataset_hash:
        raise ScriptedError("the record's dataset is not the one asked for")


__all__ = [
    "ARMS",
    "LABEL",
    "ScriptedError",
    "ScriptedRequest",
    "match_id",
    "pair_scores",
    "read_scripted_record",
    "record_hash",
    "scripted_h2h",
    "summarise",
]
