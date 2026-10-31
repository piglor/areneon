"""Best-response gain, the blind control and the room for the table (ADR-029 section 2, #1257).

An evaluation-only responder with white-box access to a frozen pair's weights, and only public team
information, chooses its Team Preview against the pair from a payoff table (no game outcome); the
evaluated pair's own Battle network plays the games. The Node side
(``packages/team-dataset/scripts/best-response.ts``) plays four arms on every validation pair from
both seats and writes a ``BestResponseRecord``: the mirror (the pair against itself), the responder
on team B against the pair on team A, the blind control (the same responder against the uniform
distribution over A's strategies) and the room arm (the responder on team A against the pair on B).

This module runs it, reads the record back with its own reader (the same rules as the TS reader
``packages/showdown-bridge/src/best-response-record.ts``), ties it to the request, derives the
pair-level intervals of ``stats.pair_interval`` (never stored), logs the look in the validation
look log and only then commits the directory and shows the result.

BRG is the mean over pairs of the responder's pair score minus the mirror's; ``BRG_blind`` the same
with the blind control; the exploitation margin ``BRG - BRG_blind`` is paired by pair; R is the room
arm's mean pair score minus one half. A conditional lower bound, a progress view on validation
teams: never the Training gate and never held-out evidence.
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

from areneon_ml.loop import layout
from areneon_ml.loop.play import PlayError
from areneon_ml.loop.records import policy_id
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.gamedata import GameData
from areneon_ml.progress import evaluation_key as ekey
from areneon_ml.progress.eval_dataset import EvalDataset
from areneon_ml.progress.looks import Look, append_look, looks_path
from areneon_ml.progress.run import ProgressError, _read_config, _source, max_pairs
from areneon_ml.progress.stats import PPM, bootstrap_stream_key, pair_interval

CONTRACT = "best-response-record"
SCRIPT = "packages/team-dataset/scripts/best-response.ts"
RECORD = "best-response.json"
LABEL = (
    "evaluation-only responder with white-box access to the evaluated policy's weights; "
    "a conditional lower bound on validation teams; not the Training gate; never held-out evidence"
)
ARMS = ("mirror", "responder", "blind", "room")
MODES = ("argmax", "sample")
OWN_RULES = ("prior-top-k", "distinct-leads")
_STDERR_TAIL = 2000
_HALVES = {"focal": 2, "tie": 1, "other": 0}
_COUNT_KEYS = (
    "worlds_asked",
    "worlds_used",
    "preview_queries",
    "entry_attempts",
    "entries",
    "value_queries",
    "rollout_choices",
    "back_draws",
    "clamped",
    "clamped_low",
    "clamped_high",
    "game_end_entries",
    "engine_refusals",
    "not_move_request",
    "worlds_dropped",
    "rollout_capped",
    "missing_entries",
)


class BestResponseError(ValueError):
    """An evaluation is refused, or its record breaks a rule."""


def _fail(message: str) -> None:
    raise BestResponseError(message)


def _tally(outcomes: list[dict[str, Any]], max_turns: int) -> dict[str, Any]:
    wins = losses = ties = cap_ties = turns = 0
    forfeits: dict[tuple[str, str], int] = {}
    for outcome in outcomes:
        turns += outcome["turns"]
        if outcome["kind"] == "forfeit":
            key = (outcome["loser"], outcome["cause"])
            forfeits[key] = forfeits.get(key, 0) + 1
            losses += outcome["loser"] == "focal"
            wins += outcome["loser"] == "other"
        elif outcome["winner"] == "tie":
            ties += 1
            cap_ties += outcome["turns"] > max_turns
        else:
            wins += outcome["winner"] == "focal"
            losses += outcome["winner"] == "other"
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "turn_cap_ties": cap_ties,
        "turns_total": turns,
        "forfeits": [{"side": s, "cause": c, "count": n} for (s, c), n in sorted(forfeits.items())],
    }


def _check_counts(
    counts: dict[str, int], decision: dict[str, Any], what: str, plan: dict[str, Any]
) -> None:
    worlds = counts["worlds_asked"] + decision["worlds_unstarted"] + decision["worlds_refused"]
    if worlds != plan["responder"]["worlds"]:
        _fail(f"{what}: worlds asked, unstarted and refused must add up to the plan's worlds")
    if counts["worlds_used"] > counts["worlds_asked"] or (
        counts["worlds_dropped"] > counts["worlds_used"]
    ):
        _fail(f"{what}: worlds used cannot exceed worlds asked, nor worlds dropped worlds used")
    if counts["entries"] > counts["entry_attempts"]:
        _fail(f"{what}: entries cannot exceed entry attempts")
    if counts["clamped"] != counts["clamped_low"] + counts["clamped_high"]:
        _fail(f"{what}: clamped must be the sum of clamped_low and clamped_high")


def _check_decision(
    decision: dict[str, Any], arms: list[str], what: str, plan: dict[str, Any]
) -> None:
    if decision["kind"] == "fallback":
        return
    counts, choices = decision["counts"], decision["choices"]
    _check_counts(counts, decision, what, plan)
    if decision["own"] > plan["responder"]["k_own"]:
        _fail(f"{what}: own strategies cannot exceed k_own")
    if decision["complete"] != (counts["missing_entries"] == 0):
        _fail(f"{what}: complete must hold exactly when no entry is missing")
    if counts["missing_entries"] > decision["own"] * decision["columns"]:
        _fail(f"{what}: missing entries cannot exceed the table's cells")
    if [c["arm"] for c in choices] != arms:
        _fail(f"{what}: the choices must be exactly {', '.join(arms)}, in that order")
    for choice in choices:
        if choice["rank"] >= decision["own"]:
            _fail(f"{what}: a chosen rank must be below the table's own strategies")
        if choice["columns_used"] > decision["columns"]:
            _fail(f"{what}: columns used cannot exceed the table's columns")


def read_best_response_record(value: object) -> dict[str, Any]:
    """Validate against the contract, then apply the rules the profile cannot say."""
    validate(CONTRACT, value)
    record: dict[str, Any] = json.loads(json.dumps(value))
    plan, teams, games, result = record["plan"], record["teams"], record["games"], record["result"]
    if plan["games"] % 2:
        _fail("plan.games must be even: every pair is played from both seats")
    if teams["pairs_used"] != plan["games"] // 2:
        _fail("teams.pairs_used must be half of plan.games")
    if teams.get("evaluation_dataset_hash") == teams["dataset_hash"]:
        _fail("the evaluation dataset cannot be the run's own dataset")
    if len(games) != plan["games"]:
        _fail("there must be one game row per game")
    for index, row in enumerate(games):
        if row["game"] != index or row["pair"] != index >> 1:
            _fail("game rows must run 0..games-1 with pair equal to the game divided by two")
        if row["a_seat"] != ("p1" if index % 2 == 0 else "p2"):
            _fail("team A sits on p1 for an even game and p2 for an odd one")
        _check_decision(row["b_decision"], ["responder", "blind"], "b_decision", plan)
        _check_decision(row["a_decision"], ["room"], "a_decision", plan)
    for arm in ARMS:
        listed = [(f["side"], f["cause"]) for f in result[arm]["forfeits"]]
        if any(a >= b for a, b in itertools.pairwise(listed)):
            _fail(f"{arm}: forfeits must be sorted by side then cause, each pair once")
        tally = result[arm]
        if tally["wins"] + tally["losses"] + tally["ties"] != plan["games"]:
            _fail(f"{arm}: wins + losses + ties must equal plan.games")
        if tally != _tally([row[arm] for row in games], plan["max_turns"]):
            _fail(f"{arm}: the tally must equal the game rows")
    return record


def record_hash(record: object) -> str:
    """SHA-256 of a record's JCS bytes."""
    return sha256_hex(jcs(record))


def pair_scores(record: dict[str, Any], arm: str) -> list[int]:
    """Per pair, the arm's focal-side score in half-games (0 to 4).

    A win is 2, a tie 1, a forfeit a loss for the side that forfeited.
    """
    halves = []
    for row in record["games"]:
        outcome = row[arm]
        if outcome["kind"] == "forfeit":
            halves.append(0 if outcome["loser"] == "focal" else 2)
        else:
            halves.append(_HALVES[outcome["winner"]])
    return [halves[i] + halves[i + 1] for i in range(0, len(halves), 2)]


def _interval(values: list[int], key: bytes, *, shift: int = 0) -> dict[str, int]:
    found = pair_interval(values, key)
    return {
        "mean_ppm": found.mean_ppm - shift,
        "low_ppm": found.low_ppm - shift,
        "high_ppm": found.high_ppm - shift,
    }


def _table_counts(record: dict[str, Any], side: str) -> dict[str, Any]:
    """Work done by one side's tables, summed, and how the decisions went."""
    totals = dict.fromkeys(_COUNT_KEYS, 0)
    tables = 0
    fallbacks: dict[str, int] = {}
    for row in record["games"]:
        decision = row[f"{side}_decision"]
        if decision["kind"] == "fallback":
            fallbacks[decision["code"]] = fallbacks.get(decision["code"], 0) + 1
            continue
        tables += 1
        for key in _COUNT_KEYS:
            totals[key] += decision["counts"][key]
    return {
        "tables": tables,
        "fallbacks": [{"code": c, "count": n} for c, n in sorted(fallbacks.items())],
        "counts": totals,
    }


def summarise(record: dict[str, Any], run_key: str | None = None) -> dict[str, Any]:
    """Descriptive summary: BRG, BRG_blind, the margin and R, each with a pair-level interval.

    Differences are per pair in half-games and read as shares of a game (the mean is
    ``sum / (4 n)``); the resampling streams are seeded from the run key, the generation and the
    statistic's label, so a rerun shows the same intervals. Labelled descriptive and conditional
    on the fixed validation plan, with no multiplicity control.
    """
    run_id = record["run_id"] if run_key is None else run_key
    generation = record["evaluated"]["generation"]
    mode = record["evaluated"]["mode"]
    scores = {arm: pair_scores(record, arm) for arm in ARMS}

    def key(label: str) -> bytes:
        return bootstrap_stream_key(run_id, generation, f"best-response:{label}:{mode}")

    brg = [r - m for r, m in zip(scores["responder"], scores["mirror"], strict=True)]
    brg_blind = [b - m for b, m in zip(scores["blind"], scores["mirror"], strict=True)]
    margin = [r - b for r, b in zip(scores["responder"], scores["blind"], strict=True)]
    same = sum(
        row["b_decision"]["kind"] == "table"
        and row["b_decision"]["choices"][0]["key"] == row["b_decision"]["choices"][1]["key"]
        for row in record["games"]
    )
    games = record["plan"]["games"]
    return {
        "label": LABEL,
        "run_id": record["run_id"],
        "generation": generation,
        "mode": mode,
        "policy_id": record["evaluated"]["policy_id"],
        "pairs": len(brg),
        "games": games,
        "scores": {arm: _interval(scores[arm], key(arm)) for arm in ARMS},
        "brg": _interval(brg, key("brg")),
        "brg_blind": _interval(brg_blind, key("brg-blind")),
        "margin": _interval(margin, key("margin")),
        "room": _interval(scores["room"], key("room"), shift=PPM // 2),
        "responder_choice_equals_blind": same,
        "forfeits": {arm: record["result"][arm]["forfeits"] for arm in ARMS},
        "forfeit_share_ppm": {
            arm: sum(f["count"] for f in record["result"][arm]["forfeits"]) * PPM // games
            for arm in ARMS
        },
        "b_tables": _table_counts(record, "b"),
        "a_tables": _table_counts(record, "a"),
    }


@dataclass(frozen=True)
class BestResponseRequest:
    """What one evaluation asks for."""

    run_dir: Path
    generation: int
    mode: str
    pairs: int
    worlds: int
    k_own: int
    own_rule: str
    leaf: str
    #: The Team Preview latency budget of every game, in milliseconds: the table is built inside it.
    preview_budget_ms: int
    out: Path
    engine_data: Path
    worker_python: str
    #: Another verified dataset (the eval-dataset option), instead of the run's own team dataset.
    eval_dataset: EvalDataset | None = None
    evaluation_key: str | None = None
    #: The look log; default the sibling ``<out>.looks.jsonl``.
    looks: Path | None = None


def _run_node(node: str, repo: Path, args: list[str]) -> str:
    try:
        done = subprocess.run(
            [node, str(repo / SCRIPT), *args],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise PlayError(f"cannot start {node}: {exc}") from exc
    if done.returncode != 0:
        raise PlayError(f"{SCRIPT} failed (exit {done.returncode}): {done.stderr[-_STDERR_TAIL:]}")
    return done.stdout


def match_id(mode: str) -> str:
    """The name a look gives the evaluation (an id token)."""
    return f"best-response-{mode}"


def _leaf(text: str) -> dict[str, Any]:
    if text == "first-request":
        return {"kind": "first-request"}
    if text.startswith("rollout:") and text[8:].isdigit() and int(text[8:]) >= 1:
        return {"kind": "rollout", "turns": int(text[8:])}
    raise BestResponseError(
        "--leaf must be first-request or rollout:<turns> with at least one turn"
    )


def _check_asked(request: BestResponseRequest) -> dict[str, Any]:
    """Refuse a malformed request before anything starts; return the leaf as the record names it."""
    if request.mode not in MODES:
        raise BestResponseError("--mode must be argmax or sample")
    if request.own_rule not in OWN_RULES:
        raise BestResponseError("--own-rule must be prior-top-k or distinct-leads")
    for name, value in (
        ("--pairs", request.pairs),
        ("--worlds", request.worlds),
        ("--k-own", request.k_own),
        ("--preview-budget-ms", request.preview_budget_ms),
    ):
        if value < 1:
            raise BestResponseError(f"{name} must be at least 1")
    return _leaf(request.leaf)


def best_response(
    request: BestResponseRequest, *, repo: Path, node: str = "node"
) -> dict[str, Any]:
    """Verify the run, play, read back, tie, log the look, commit; return the summary."""
    leaf = _check_asked(request)
    run_dir = request.run_dir.resolve()
    config = _read_config(run_dir)
    try:
        key = ekey.check(request.evaluation_key, config.run_key)
    except ekey.EvaluationKeyError as exc:
        raise BestResponseError(str(exc)) from exc
    if not config.team_dataset:
        raise ProgressError(
            "the run config names no team dataset: the interim pool has no validation split"
        )
    if (
        request.eval_dataset is not None
        and request.eval_dataset.dataset_hash == config.raw["team_dataset"]["dataset_hash"]
    ):
        raise ProgressError(
            "--eval-dataset-hash is the run's own training dataset: leave the option out"
        )
    limit = max_pairs(config, request.eval_dataset)
    if request.pairs > limit:
        raise ProgressError(
            f"{request.pairs} pairs need {request.pairs * 2} validation teams: the dataset has "
            f"{limit * 2}"
        )
    source = _source(run_dir, config, request.generation)
    mode_id = policy_id(source.preview, source.battle, request.mode)
    game = GameData.from_table(json.loads(request.engine_data.read_text(encoding="utf-8")))
    out = request.out.resolve()
    if out.exists():
        raise BestResponseError("the output directory exists: an evaluation is never overwritten")
    partial = out.with_name(f".{out.name}.partial")
    log = (request.looks if request.looks is not None else looks_path(out, run_dir)).resolve()
    if log in (partial, out) or partial in log.parents or out in log.parents:
        raise BestResponseError("the look log must not be inside the output directory")
    shutil.rmtree(partial, ignore_errors=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    args = [
        *("--config", str(run_dir / layout.CONFIG)),
        *("--generation", str(request.generation)),
        *("--preview", source.preview),
        *("--battle", source.battle),
        *("--mode", request.mode),
        *("--policy-id", mode_id),
        *("--generation-manifest-sha256", source.manifest_sha256),
        *("--team-set", str(source.team_set)),
        *("--run-dir", str(run_dir)),
        *("--engine-data", str(request.engine_data.resolve())),
        *("--encoder-version", encoder_version(game, config.model.encoder)),
        *("--worker-python", request.worker_python),
        *("--pairs", str(request.pairs)),
        *("--worlds", str(request.worlds)),
        *("--k-own", str(request.k_own)),
        *("--own-rule", request.own_rule),
        *("--leaf", request.leaf),
        *("--preview-budget-ms", str(request.preview_budget_ms)),
        *("--out", str(partial)),
        *(() if request.eval_dataset is None else request.eval_dataset.node_args()),
        *ekey.node_args(key),
    ]
    try:
        line = _run_node(node, repo, args)
        try:
            printed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PlayError(f"{SCRIPT} did not print a JSON line") from exc
        record = read_best_response_record(
            json.loads((partial / RECORD).read_text(encoding="utf-8"))
        )
        _tie(record, request, config, source, mode_id, key, leaf)
        digest = record_hash(record)
        if not isinstance(printed, dict) or printed.get("record_hash") != digest:
            raise BestResponseError("the record on disk is not the one the play script reported")
        summary = summarise(record, config.run_key)
        # A validation result is about to be shown: log the look first (a refused run logs nothing).
        state = append_look(
            log,
            "best-response",
            config.run_key,
            [Look(request.generation, match_id(request.mode), record["plan"]["games"], key)],
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


def _tie(  # noqa: PLR0913, PLR0917
    record: dict[str, Any],
    request: BestResponseRequest,
    config: Any,
    source: Any,
    mode_id: str,
    key: str | None,
    leaf: dict[str, Any],
) -> None:
    """The record must be the evaluation that was asked for."""
    plan, evaluated, teams = record["plan"], record["evaluated"], record["teams"]
    if record["run_id"] != config.run_key:
        raise BestResponseError("the record's run key is not the config's")
    expected = {
        "generation": request.generation,
        "generation_manifest_sha256": source.manifest_sha256,
        "checkpoints": {"preview": source.preview, "battle": source.battle},
        "mode": request.mode,
        "policy_id": mode_id,
    }
    if evaluated != expected:
        raise BestResponseError("the record's evaluated pair is not the one asked for")
    if teams["pairs_used"] != request.pairs:
        raise BestResponseError("the record's pairs are not the number asked for")
    gen = config.generation
    if plan["max_turns"] != gen.max_turns or plan["latency_budget_ms"] != {
        "preview_ms": gen.preview_ms,
        "battle_ms": gen.battle_ms,
    }:
        raise BestResponseError("the record's turn cap and latency budgets are not the config's")
    if plan["responder_preview_budget_ms"] != request.preview_budget_ms:
        raise BestResponseError("the record's Preview budget is not the one asked for")
    if plan.get("evaluation_key") != key:
        raise BestResponseError("the record's evaluation key is not the one asked for")
    responder = plan["responder"]
    if (
        responder["worlds"],
        responder["k_own"],
        responder["own_rule"],
        responder["leaf"],
    ) != (request.worlds, request.k_own, request.own_rule, leaf):
        raise BestResponseError("the record's responder settings are not the ones asked for")
    if request.eval_dataset is None:
        if "evaluation_dataset_hash" in teams:
            raise BestResponseError("the record names an evaluation dataset that was not asked for")
    elif teams.get("evaluation_dataset_hash") != request.eval_dataset.dataset_hash:
        raise BestResponseError("the record's dataset is not the one asked for")


__all__ = [
    "ARMS",
    "LABEL",
    "BestResponseError",
    "BestResponseRequest",
    "best_response",
    "match_id",
    "pair_scores",
    "read_best_response_record",
    "record_hash",
    "summarise",
]
