"""The best-response evaluation (ADR-029 section 2, #1257): reader, intervals and command.

The Node side is replaced by a stand-in that writes a record built from the contract corpus's
samples, so the reader, the ties to the request, the look log and the commit run for real while
no battle is played (the real script is driven by ``best-response.python.test.ts``).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from areneon_contracts.contracts import ContractError, jcs, validate
from areneon_ml.learner.config import read_run_config
from areneon_ml.loop import __main__ as module
from areneon_ml.loop.__main__ import EXIT_PLAY, EXIT_REFUSED, main
from areneon_ml.loop.play import PlayError
from areneon_ml.progress import best_response
from areneon_ml.progress.best_response import (
    BestResponseError,
    BestResponseRequest,
    match_id,
    pair_scores,
    read_best_response_record,
    record_hash,
    summarise,
)
from areneon_ml.progress.best_response import (
    best_response as run_best_response,
)
from areneon_ml.progress.eval_dataset import EvalDataset
from areneon_ml.progress.looks import LookError, read_looks, verify_looks
from areneon_ml.progress.stats import PPM, bootstrap_stream_key, pair_interval

REPO = Path(__file__).resolve().parents[3]
SAMPLES = REPO / "fixtures/contracts/v1/best-response-record"
TINY = REPO / "fixtures/contracts/v1/training-run-config/valid/tiny.json"
ENGINE_DATA = REPO / "fixtures/showdown/gen9championsvgc2026regmc/engine-data.json"
MANIFEST_SHA = "7" * 64
PREVIEW, BATTLE = "1" * 64, "2" * 64


def sample(kind: str, name: str) -> dict[str, Any]:
    value: dict[str, Any] = json.loads((SAMPLES / kind / f"{name}.json").read_text("utf-8"))
    return value


def files(kind: str) -> list[str]:
    return sorted(p.stem for p in (SAMPLES / kind).glob("*.json"))


def test_it_reads_every_valid_sample_and_refuses_the_others() -> None:
    assert files("valid") == ["evaluation-dataset-rollout", "two-pairs"]
    for name in files("valid"):
        assert read_best_response_record(sample("valid", name))["data_class"] == "synthetic"
    for name in files("invalid"):
        with pytest.raises(ContractError):
            read_best_response_record(sample("invalid", name))
    for name in files("reader-invalid"):
        # the schema passes it; a reader rule refuses it
        validate(best_response.CONTRACT, sample("reader-invalid", name))
        with pytest.raises(BestResponseError):
            read_best_response_record(sample("reader-invalid", name))


def test_pair_scores_count_a_forfeit_as_a_loss_for_its_side() -> None:
    record = sample("valid", "two-pairs")
    assert pair_scores(record, "mirror") == [0, 2]
    assert pair_scores(record, "responder") == [4, 4]  # the last is a forfeit by the other side
    assert pair_scores(record, "blind") == [1, 2]
    assert pair_scores(record, "room") == [3, 2]


def test_the_summary_is_the_pair_level_bootstrap_of_the_differences() -> None:
    record = sample("valid", "two-pairs")
    summary = summarise(record)
    key = bootstrap_stream_key("tiny-run", 5, "best-response:brg:argmax")
    mine = pair_interval([4, 2], key)  # responder minus mirror: [4 - 0, 4 - 2]
    assert summary["brg"] == {
        "mean_ppm": mine.mean_ppm,
        "low_ppm": mine.low_ppm,
        "high_ppm": mine.high_ppm,
    }
    assert summary["brg"]["mean_ppm"] == 6 * PPM // 8
    assert summary["brg_blind"]["mean_ppm"] == PPM // 8  # [1 - 0, 2 - 2]
    assert summary["margin"]["mean_ppm"] == 5 * PPM // 8  # [4 - 1, 4 - 2]
    # R is the room arm's mean pair score minus one half: (3 + 2) / 8 - 1/2
    assert summary["room"]["mean_ppm"] == 5 * PPM // 8 - PPM // 2
    assert summary["pairs"] == 2
    assert summary["games"] == 4
    assert summary["label"] == best_response.LABEL
    assert "white-box access" in summary["label"]
    assert summary["responder_choice_equals_blind"] == 0
    assert summary["b_tables"]["tables"] == 3
    assert summary["b_tables"]["fallbacks"] == [{"code": "no_world", "count": 1}]
    assert summary["a_tables"]["fallbacks"] == [{"code": "species_not_in_prior", "count": 1}]
    assert summary["forfeits"]["responder"] == [
        {"side": "other", "cause": "latency_overrun", "count": 1}
    ]
    assert summary["forfeit_share_ppm"]["responder"] == PPM // 4
    # the same record, the same intervals
    assert summarise(copy.deepcopy(record)) == summary


def _record(
    mirror: str, responder: str, blind: str, room: str = "tie", same: bool = False
) -> dict[str, Any]:
    """A record of two pairs whose arms win, tie or lose every game from the focal side."""
    record = sample("valid", "two-pairs")
    played = {
        "win": {"kind": "played", "winner": "focal", "turns": 5},
        "tie": {"kind": "played", "winner": "tie", "turns": 5},
        "loss": {"kind": "played", "winner": "other", "turns": 5},
    }
    for row in record["games"]:
        for arm, how in zip(
            ("mirror", "responder", "blind", "room"), (mirror, responder, blind, room), strict=True
        ):
            row[arm] = copy.deepcopy(played[how])
        row["b_decision"] = {
            "kind": "table",
            "counts": record["games"][0]["b_decision"]["counts"],
            "worlds_unstarted": 0,
            "worlds_refused": 0,
            "own": 4,
            "columns": 3,
            "complete": True,
            "choices": [
                {
                    "arm": "responder",
                    "rank": 1,
                    "key": "13|24",
                    "expected_ppm": 6,
                    "columns_used": 3,
                },
                {
                    "arm": "blind",
                    "rank": 1 if same else 0,
                    "key": "13|24" if same else "12|34",
                    "expected_ppm": 5,
                    "columns_used": 3,
                },
            ],
        }
        row["a_decision"] = record["games"][0]["a_decision"]
    record["result"] = {
        arm: best_response._tally([r[arm] for r in record["games"]], record["plan"]["max_turns"])
        for arm in best_response.ARMS
    }
    return read_best_response_record(record)


def test_a_fixed_strategy_policy_has_a_positive_margin_and_a_uniform_one_none() -> None:
    # Fixed strategy: the responder wins everything and the blind control loses like the mirror.
    fixed = summarise(_record("loss", "win", "loss"))
    assert fixed["brg"]["mean_ppm"] == PPM
    assert fixed["brg_blind"]["mean_ppm"] == 0
    assert fixed["margin"]["mean_ppm"] == PPM
    assert fixed["margin"]["low_ppm"] > 0
    # Uniform policy: the responder and the blind control choose alike and play the same games.
    uniform = summarise(_record("tie", "win", "win", same=True))
    assert uniform["brg"]["mean_ppm"] == PPM // 2
    assert uniform["margin"] == {"mean_ppm": 0, "low_ppm": 0, "high_ppm": 0}
    assert uniform["responder_choice_equals_blind"] == 4


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r["games"][0]["b_decision"]["counts"].update(worlds_used=5), "worlds used"),
        (lambda r: r["games"][0]["b_decision"]["counts"].update(worlds_dropped=5), "worlds used"),
        (lambda r: r["games"][0]["b_decision"]["counts"].update(entries=99), "entries cannot"),
        (lambda r: r["games"][0]["b_decision"].update(own=5), "exceed k_own"),
        (
            lambda r: (
                r["games"][0]["b_decision"].update(complete=False)
                or r["games"][0]["b_decision"]["counts"].update(missing_entries=13)
            ),
            "missing entries cannot",
        ),
        (lambda r: r["result"]["mirror"].update(wins=9), "must equal plan.games"),
        (lambda r: r["games"][1].update(game=5), "game rows must run"),
    ],
)
def test_the_reader_refuses_the_counts_and_tallies_a_sample_cannot_show(
    mutate: Any, message: str
) -> None:
    record = sample("valid", "two-pairs")
    mutate(record)
    with pytest.raises(BestResponseError, match=message):
        read_best_response_record(record)


def test_the_reader_rules_name_a_rule_not_a_value() -> None:
    with pytest.raises(BestResponseError, match="the tally must equal the game rows"):
        read_best_response_record(sample("reader-invalid", "tally-not-rows"))
    assert match_id("sample") == "best-response-sample"


# --- the command, with a stand-in for the Node side -------------------------------------------


def patched_request(req: BestResponseRequest, **change: Any) -> BestResponseRequest:
    return dataclasses.replace(req, **change)


def run_config(tmp_path: Path) -> Any:
    raw = json.loads(TINY.read_text("utf-8"))
    raw["team_dataset"] = {"path": str(tmp_path / "dataset"), "dataset_hash": "9" * 64}
    return read_run_config(raw)


def record_for(config: Any, req: BestResponseRequest) -> dict[str, Any]:
    record = sample("valid", "two-pairs")
    gen = config.generation
    record["run_id"] = config.run_key
    record["evaluated"] = {
        "generation": req.generation,
        "generation_manifest_sha256": MANIFEST_SHA,
        "checkpoints": {"preview": PREVIEW, "battle": BATTLE},
        "mode": req.mode,
        "policy_id": best_response.policy_id(PREVIEW, BATTLE, req.mode),
    }
    record["teams"] = {"dataset_hash": "9" * 64, "split": "validation", "pairs_used": 2}
    if req.eval_dataset is not None:
        record["teams"]["evaluation_dataset_hash"] = req.eval_dataset.dataset_hash
    record["plan"]["max_turns"] = gen.max_turns
    record["plan"]["latency_budget_ms"] = {"preview_ms": gen.preview_ms, "battle_ms": gen.battle_ms}
    record["plan"]["responder_preview_budget_ms"] = req.preview_budget_ms
    if req.evaluation_key is not None:
        record["plan"]["evaluation_key"] = req.evaluation_key
    record["plan"]["responder"].update(
        worlds=4,
        k_own=4,
        own_rule=req.own_rule,
        leaf={"kind": "first-request"}
        if req.leaf == "first-request"
        else {"kind": "rollout", "turns": 2},
    )
    record["result"] = {
        arm: best_response._tally([r[arm] for r in record["games"]], gen.max_turns)
        for arm in best_response.ARMS
    }
    return read_best_response_record(record)


class FakeNode:
    """Stands in for ``_run_node``: writes the record the real script would."""

    def __init__(self, tmp_path: Path, *, fail: bool = False, lie: bool = False) -> None:
        self.tmp_path, self.fail, self.lie = tmp_path, fail, lie
        self.record: dict[str, Any] = {}
        self.calls: list[list[str]] = []
        self.change: Any = None

    def __call__(self, node: str, repo: Path, args: list[str]) -> str:
        self.calls.append(args)
        if self.fail:
            raise PlayError("best-response.ts failed (exit 1): boom")
        out = Path(args[args.index("--out") + 1])
        out.mkdir(parents=True)
        (out / best_response.RECORD).write_bytes(jcs(self.record) + b"\n")
        digest = "0" * 64 if self.lie else record_hash(self.record)
        return json.dumps({"record_hash": digest, "wall_ms": 5}) + "\n"


def patched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any
) -> tuple[BestResponseRequest, FakeNode, Any]:
    config = run_config(tmp_path)
    monkeypatch.setattr(best_response, "_read_config", lambda run_dir: config)
    monkeypatch.setattr(
        best_response,
        "_source",
        lambda run_dir, cfg, generation: SimpleNamespace(
            preview=PREVIEW,
            battle=BATTLE,
            manifest_sha256=MANIFEST_SHA,
            team_set=tmp_path / "team-set.json",
        ),
    )
    monkeypatch.setattr(best_response, "max_pairs", lambda cfg, eval_dataset=None: 5)
    fields: dict[str, Any] = {
        "run_dir": tmp_path / "run",
        "generation": 5,
        "mode": "argmax",
        "pairs": 2,
        "worlds": 4,
        "k_own": 4,
        "own_rule": "prior-top-k",
        "leaf": "first-request",
        "preview_budget_ms": 3_600_000,
        "out": tmp_path / "br",
        "engine_data": ENGINE_DATA,
        "worker_python": "python",
    }
    fields.update(kwargs)
    req = BestResponseRequest(**fields)
    fake = FakeNode(tmp_path)
    try:
        fake.record = record_for(config, req)
    except (BestResponseError, ContractError):
        fake.record = {}  # a request the command refuses before it runs anything
    monkeypatch.setattr(best_response, "_run_node", fake)
    return req, fake, config


def test_it_runs_logs_the_look_and_commits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req, fake, _ = patched(monkeypatch, tmp_path)
    result = run_best_response(req, repo=REPO)
    args = fake.calls[0]
    assert args[args.index("--mode") + 1] == "argmax"
    assert args[args.index("--leaf") + 1] == "first-request"
    assert args[args.index("--generation-manifest-sha256") + 1] == MANIFEST_SHA
    assert "--evaluation-key" not in args
    assert (tmp_path / "br" / best_response.RECORD).exists()
    assert not (tmp_path / ".br.partial").exists()
    assert result["looks"] == 1
    log = tmp_path / "br.looks.jsonl"
    assert read_looks(log).count == 1
    entry = json.loads(log.read_text("utf-8"))
    assert entry["command"] == "best-response"
    assert entry["matches"] == [{"generation": 5, "match_id": "best-response-argmax", "games": 4}]
    assert result["record_hash"] == record_hash(fake.record)
    assert result["summary"]["pairs"] == 2
    verify_looks(log.read_bytes())
    with pytest.raises(BestResponseError, match="never overwritten"):
        run_best_response(req, repo=REPO)


def test_the_evaluation_key_and_the_evaluation_dataset_are_passed_and_tied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = EvalDataset(tmp_path / "eval", "c" * 64)
    req, fake, _ = patched(
        monkeypatch,
        tmp_path,
        evaluation_key="e0p-key",
        eval_dataset=dataset,
        mode="sample",
        leaf="rollout:2",
        own_rule="distinct-leads",
    )
    run_best_response(req, repo=REPO)
    args = fake.calls[0]
    assert args[args.index("--evaluation-key") + 1] == "e0p-key"
    assert args[args.index("--eval-dataset-hash") + 1] == "c" * 64
    entry = json.loads((tmp_path / "br.looks.jsonl").read_text("utf-8"))
    assert entry["matches"][0]["evaluation_key"] == "e0p-key"
    assert entry["matches"][0]["match_id"] == "best-response-sample"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mode": "greedy"}, "argmax or sample"),
        ({"own_rule": "random"}, "prior-top-k or distinct-leads"),
        ({"pairs": 0}, "--pairs must be at least 1"),
        ({"worlds": 0}, "--worlds"),
        ({"k_own": 0}, "--k-own"),
        ({"preview_budget_ms": 0}, "--preview-budget-ms"),
        ({"leaf": "rollout:0"}, "first-request or rollout"),
        ({"leaf": "second"}, "first-request or rollout"),
        ({"evaluation_key": "bad key"}, "id token"),
        ({"evaluation_key": "tiny-run"}, "run's own key"),
        ({"pairs": 6}, "validation teams"),
    ],
)
def test_it_refuses_a_bad_request_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any], message: str
) -> None:
    req, fake, _ = patched(monkeypatch, tmp_path, **change)
    with pytest.raises((BestResponseError, best_response.ProgressError), match=message):
        run_best_response(req, repo=REPO)
    assert fake.calls == []


def test_it_refuses_the_runs_own_dataset_and_a_log_inside_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req, fake, _ = patched(monkeypatch, tmp_path, eval_dataset=EvalDataset(tmp_path, "9" * 64))
    with pytest.raises(best_response.ProgressError, match="own training dataset"):
        run_best_response(req, repo=REPO)
    req, fake, _ = patched(monkeypatch, tmp_path, looks=tmp_path / "br" / "log")
    with pytest.raises(BestResponseError, match="look log must not be inside"):
        run_best_response(req, repo=REPO)
    assert fake.calls == []


def test_a_run_config_without_a_team_dataset_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req, fake, _ = patched(monkeypatch, tmp_path)
    monkeypatch.setattr(
        best_response,
        "_read_config",
        lambda run_dir: read_run_config(json.loads(TINY.read_text("utf-8"))),
    )
    with pytest.raises(best_response.ProgressError, match="no team dataset"):
        run_best_response(req, repo=REPO)
    assert fake.calls == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(run_id="other-run"), "run key"),
        (lambda r: r["evaluated"].update(generation=4), "evaluated pair"),
        (lambda r: r["evaluated"].update(mode="sample"), "evaluated pair"),
        (lambda r: r["teams"].update(pairs_used=1), "half of plan.games"),
        (lambda r: r["plan"].update(max_turns=r["plan"]["max_turns"] + 1), "turn cap"),
        (lambda r: r["plan"].update(responder_preview_budget_ms=5), "Preview budget"),
        (lambda r: r["plan"].update(evaluation_key="stray"), "evaluation key"),
        (lambda r: r["plan"]["responder"].update(k_own=5), "responder settings"),
        (lambda r: r["teams"].update(evaluation_dataset_hash="c" * 64), "not asked for"),
    ],
)
def test_a_record_that_is_not_the_one_asked_for_logs_nothing_and_leaves_no_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Any, message: str
) -> None:
    req, fake, _ = patched(monkeypatch, tmp_path)
    mutate(fake.record)
    with pytest.raises((BestResponseError, ContractError), match=message):
        run_best_response(req, repo=REPO)
    assert not (tmp_path / "br").exists()
    assert not (tmp_path / ".br.partial").exists()
    assert not (tmp_path / "br.looks.jsonl").exists()


def test_the_pairs_must_be_the_number_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req, fake, _ = patched(monkeypatch, tmp_path, pairs=1)
    fake.record = record_for(run_config(tmp_path), patched_request(req, pairs=2))
    with pytest.raises(BestResponseError, match="pairs are not"):
        run_best_response(req, repo=REPO)
    assert not (tmp_path / "br.looks.jsonl").exists()


def test_the_evaluation_dataset_must_be_the_one_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req, fake, _ = patched(
        monkeypatch, tmp_path, eval_dataset=EvalDataset(tmp_path / "e", "c" * 64)
    )
    fake.record["teams"]["evaluation_dataset_hash"] = "d" * 64
    with pytest.raises(BestResponseError, match="dataset is not"):
        run_best_response(req, repo=REPO)
    assert not (tmp_path / "br.looks.jsonl").exists()


def test_a_failed_or_lying_play_script_logs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req, fake, _ = patched(monkeypatch, tmp_path)
    fake.fail = True
    with pytest.raises(PlayError, match="boom"):
        run_best_response(req, repo=REPO)
    fake.fail, fake.lie = False, True
    with pytest.raises(BestResponseError, match="not the one the play script reported"):
        run_best_response(req, repo=REPO)
    assert not (tmp_path / "br").exists()
    assert not (tmp_path / "br.looks.jsonl").exists()


def test_a_play_script_that_prints_no_json_is_a_play_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req, fake, _ = patched(monkeypatch, tmp_path)
    monkeypatch.setattr(best_response, "_run_node", lambda *a: (fake(*a), "not json\n")[1])
    with pytest.raises(PlayError, match="did not print a JSON line"):
        run_best_response(req, repo=REPO)
    assert not (tmp_path / "br.looks.jsonl").exists()


def test_a_broken_look_log_stops_the_command_before_it_shows_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req, _, _ = patched(monkeypatch, tmp_path)
    (tmp_path / "br.looks.jsonl").write_text("not json\n", encoding="utf-8")
    with pytest.raises(LookError):
        run_best_response(req, repo=REPO)
    assert not (tmp_path / "br").exists()


def test_run_node_reports_how_the_child_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Done:
        def __init__(self, code: int) -> None:
            self.returncode, self.stdout, self.stderr = code, "out", "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done(0))
    assert best_response._run_node("node", REPO, ["--x"]) == "out"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done(1))
    with pytest.raises(PlayError, match=r"exit 1.*boom"):
        best_response._run_node("node", REPO, [])

    def missing(*a: Any, **k: Any) -> None:
        raise FileNotFoundError("no node")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(PlayError, match="cannot start"):
        best_response._run_node("nodex", REPO, [])


def test_the_main_command_prints_the_summary_and_maps_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    req, fake, _ = patched(monkeypatch, tmp_path)
    base = [
        "best-response",
        *("--run-dir", str(req.run_dir)),
        *("--generation", "5"),
        *("--mode", "argmax"),
        *("--pairs", "2"),
        *("--worlds", "4"),
        *("--k-own", "4"),
        *("--own-rule", "prior-top-k"),
        *("--leaf", "first-request"),
        *("--engine-data", str(ENGINE_DATA)),
    ]
    assert main([*base, *("--out", str(tmp_path / "ok"))]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["summary"]["mode"] == "argmax"
    assert printed["summary"]["pairs"] == 2
    fake.fail = True
    assert main([*base, *("--out", str(tmp_path / "fails"))]) == EXIT_PLAY
    fake.fail = False
    assert main([*base, *("--out", str(tmp_path / "ok"))]) == EXIT_REFUSED  # never overwritten
    assert module._SIMPLE["best-response"] is module._best_response
