"""The scripted head-to-head (ADR-025 section 13, #1130): the reader, the interval and the command.

The Node side is replaced by a stand-in that writes a record built from the contract corpus's
samples, so the reader, the ties to the request, the look log and the commit run for real while no
battle is played (the real script is driven by ``scripted-h2h.python.test.ts``).
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import ContractError, jcs, validate
from areneon_ml.learner.config import RunConfigError
from areneon_ml.loop import __main__ as module
from areneon_ml.loop.__main__ import EXIT_PLAY, EXIT_REFUSED, main
from areneon_ml.loop.play import PlayError
from areneon_ml.progress import scripted
from areneon_ml.progress.eval_dataset import EvalDataset
from areneon_ml.progress.looks import LookError, read_looks, verify_looks
from areneon_ml.progress.scripted import (
    ScriptedError,
    ScriptedRequest,
    match_id,
    pair_scores,
    read_scripted_record,
    record_hash,
    scripted_h2h,
    summarise,
)
from areneon_ml.progress.stats import PPM, bootstrap_stream_key, pair_interval

REPO = Path(__file__).resolve().parents[3]
SAMPLES = REPO / "fixtures/contracts/v1/scripted-head-to-head-record"
TINY = REPO / "fixtures/contracts/v1/training-run-config/valid/tiny.json"


def sample(kind: str, name: str) -> dict[str, Any]:
    value: dict[str, Any] = json.loads((SAMPLES / kind / f"{name}.json").read_text("utf-8"))
    return value


def files(kind: str) -> list[str]:
    return sorted(p.stem for p in (SAMPLES / kind).glob("*.json"))


def test_it_reads_every_valid_sample_and_refuses_the_others() -> None:
    assert files("valid") == ["actor-rule-against-random-legal", "v1-against-v2"]
    for name in files("valid"):
        assert read_scripted_record(sample("valid", name))["data_class"] == "synthetic"
    for name in files("invalid"):
        with pytest.raises(ContractError):
            read_scripted_record(sample("invalid", name))
    for name in files("reader-invalid"):
        # the schema passes it; a reader rule refuses it
        validate(scripted.CONTRACT, sample("reader-invalid", name))
        with pytest.raises(ScriptedError):
            read_scripted_record(sample("reader-invalid", name))


def test_pair_scores_count_a_forfeit_as_a_loss_for_its_arm() -> None:
    record = sample("valid", "v1-against-v2")
    # pair 0: b wins both; pair 1: a wins game 2, a forfeits game 3
    assert pair_scores(record) == [0, 2]
    tied = sample("valid", "actor-rule-against-random-legal")
    assert pair_scores(tied) == [3, 4]  # pair 0: a wins and ties; pair 1: a wins twice


def test_the_interval_is_the_pair_level_bootstrap_of_the_pair_scores() -> None:
    record = sample("valid", "v1-against-v2")
    summary = summarise(record)
    key = bootstrap_stream_key("tiny-run", 0, "scripted-h2h:damage-aware@v1:damage-aware@v2")
    mine = pair_interval(pair_scores(record), key)
    assert (summary["a"]["score_ppm"], summary["a"]["low_ppm"], summary["a"]["high_ppm"]) == (
        mine.mean_ppm,
        mine.low_ppm,
        mine.high_ppm,
    )
    theirs = pair_interval([4 - s for s in pair_scores(record)], key)
    assert (summary["b"]["low_ppm"], summary["b"]["high_ppm"]) == (theirs.low_ppm, theirs.high_ppm)
    assert summary["a"]["diff_ppm"] == mine.mean_ppm - PPM // 2
    assert summary["pairs"] == 2
    assert summary["pair_score_counts"] == [1, 0, 1, 0, 0]
    assert summary["forfeits"] == [{"side": "a", "cause": "latency_overrun", "count": 1}]
    assert summary["forfeit_share_ppm"] == PPM // 4
    assert summary["b_belief"]["totals"]["refusals"] == [
        {"code": "legality_redraws_exhausted", "count": 1},
        {"code": "reader_flagged", "count": 3},
    ]
    assert summary["a_counts"]["probe_refusals"] == 0
    assert "a_belief" not in summary
    assert "identity_hash" not in summarise(sample("valid", "actor-rule-against-random-legal"))["a"]
    assert "label" in summary
    # the same record, the same interval
    assert summarise(copy.deepcopy(record)) == summary


def test_the_reader_rules_name_a_rule_not_a_value() -> None:
    bad = sample("reader-invalid", "result-not-rows")
    with pytest.raises(ScriptedError, match="the result must equal the game rows"):
        read_scripted_record(bad)
    assert match_id("damage-aware@v1", "damage-aware@v2") == "damage-aware-v1-vs-damage-aware-v2"


# --- the command, with a stand-in for the Node side -------------------------------------------


def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(TINY.read_text("utf-8"), encoding="utf-8")
    return path


def record_for(config: Path, cap_ms: int | None = None) -> dict[str, Any]:
    gen = json.loads(config.read_text("utf-8"))["generation"]
    record = sample("valid", "v1-against-v2")
    record["plan"]["max_turns"] = gen["max_turns"]
    record["plan"]["latency_budget_ms"] = {
        "preview_ms": gen["latency_budget_ms"]["preview_ms"],
        "battle_ms": gen["latency_budget_ms"]["battle_ms"],
    }
    if cap_ms is None:
        record["plan"].pop("wall_time_cap_ms", None)
    else:
        record["plan"]["wall_time_cap_ms"] = cap_ms
    record["result"] = scripted._tally(record["games"], gen["max_turns"])
    return read_scripted_record(record)


class FakeNode:
    """Stands in for ``_run_node``: writes the record the real script would."""

    def __init__(self, record: dict[str, Any], *, fail: bool = False, lie: bool = False) -> None:
        self.record, self.fail, self.lie = record, fail, lie
        self.calls: list[tuple[list[str], float | None]] = []

    def __call__(self, node: str, repo: Path, args: list[str], timeout_s: float | None) -> str:
        self.calls.append((args, timeout_s))
        if self.fail:
            raise PlayError("scripted-h2h.ts failed (exit 1): the wall-time cap passed")
        out = Path(args[args.index("--out") + 1])
        out.mkdir(parents=True)
        (out / scripted.RECORD).write_bytes(jcs(self.record) + b"\n")
        digest = "0" * 64 if self.lie else record_hash(self.record)
        return json.dumps({"record_hash": digest, "wall_ms": 5}) + "\n"


def request(tmp_path: Path, **change: Any) -> ScriptedRequest:
    fields: dict[str, Any] = {
        "arm_a": "damage-aware@v1",
        "arm_b": "damage-aware@v2",
        "pairs": 2,
        "out": tmp_path / "h2h",
    }
    fields.update(change)
    fields.setdefault("config", config_file(tmp_path))
    return ScriptedRequest(**fields)


def test_it_runs_logs_the_look_and_commits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req = request(tmp_path, game_processes=3, wall_time_cap_minutes=480)
    fake = FakeNode(record_for(req.config, 480 * 60_000))
    monkeypatch.setattr(scripted, "_run_node", fake)
    result = scripted_h2h(req, repo=REPO)
    (args, timeout) = fake.calls[0]
    assert args[args.index("--game-processes") + 1] == "3"
    assert args[args.index("--wall-time-cap-ms") + 1] == str(480 * 60_000)
    assert timeout == 480 * 60 + scripted._KILL_SLACK_S
    assert (tmp_path / "h2h" / scripted.RECORD).exists()
    assert not (tmp_path / ".h2h.partial").exists()
    assert result["looks"] == 1
    log = tmp_path / "h2h.looks.jsonl"
    assert read_looks(log).count == 1
    entry = json.loads(log.read_text("utf-8"))
    assert entry["command"] == "scripted-h2h"
    assert entry["matches"] == [
        {"generation": 0, "match_id": "damage-aware-v1-vs-damage-aware-v2", "games": 4}
    ]
    assert result["record_hash"] == record_hash(fake.record)
    assert result["summary"]["pairs"] == 2
    verify_looks(log.read_bytes())


def test_a_second_run_appends_to_the_same_log_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = request(tmp_path)
    monkeypatch.setattr(scripted, "_run_node", FakeNode(record_for(first.config)))
    scripted_h2h(first, repo=REPO)
    with pytest.raises(ScriptedError, match="never overwritten"):
        scripted_h2h(first, repo=REPO)
    log = tmp_path / "shared.looks.jsonl"
    for name in ("one", "two"):
        scripted_h2h(request(tmp_path, out=tmp_path / name, looks=log), repo=REPO)
    assert read_looks(log).count == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"arm_a": "learned@v1"}, "scripted baselines"),
        ({"arm_b": "damage-aware@v1"}, "different baselines"),
        ({"pairs": 0}, "at least 1"),
    ],
)
def test_it_refuses_a_bad_request_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any], message: str
) -> None:
    fake = FakeNode(record_for(config_file(tmp_path)))
    monkeypatch.setattr(scripted, "_run_node", fake)
    with pytest.raises(ScriptedError, match=message):
        scripted_h2h(request(tmp_path, **change), repo=REPO)
    assert fake.calls == []


def test_it_refuses_the_placeholder_run_key_and_a_log_inside_the_output(tmp_path: Path) -> None:
    config = tmp_path / "placeholder.json"
    config.write_text(TINY.read_text("utf-8").replace("tiny-run", "CHANGE-ME-run"), "utf-8")
    with pytest.raises(RunConfigError, match="CHANGE-ME"):
        scripted_h2h(request(tmp_path, config=config), repo=REPO)
    good = config_file(tmp_path)
    with pytest.raises(ScriptedError, match="look log must not be inside"):
        scripted_h2h(request(tmp_path, config=good, looks=tmp_path / "h2h" / "log"), repo=REPO)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(run_id="other-run"), "run key"),
        (lambda r: r["plan"].update(max_turns=r["plan"]["max_turns"] + 1), "turn cap"),
        (lambda r: r["plan"].update(wall_time_cap_ms=5), "wall-time cap"),
    ],
)
def test_a_record_that_is_not_the_one_asked_for_logs_nothing_and_leaves_no_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Any, message: str
) -> None:
    req = request(tmp_path)
    record = record_for(req.config)
    mutate(record)
    monkeypatch.setattr(scripted, "_run_node", FakeNode(record))
    with pytest.raises((ScriptedError, ContractError), match=message if message else ""):
        scripted_h2h(req, repo=REPO)
    assert not (tmp_path / "h2h").exists()
    assert not (tmp_path / ".h2h.partial").exists()
    assert not (tmp_path / "h2h.looks.jsonl").exists()


def test_the_arms_must_be_the_ones_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req = request(tmp_path, arm_a="actor-rule@v1")
    monkeypatch.setattr(scripted, "_run_node", FakeNode(record_for(req.config)))
    with pytest.raises(ScriptedError, match="arms are not"):
        scripted_h2h(req, repo=REPO)
    assert not (tmp_path / "h2h.looks.jsonl").exists()


def test_pairs_and_dataset_must_match_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req = request(tmp_path, pairs=3)
    monkeypatch.setattr(scripted, "_run_node", FakeNode(record_for(req.config)))
    with pytest.raises(ScriptedError, match="pairs are not"):
        scripted_h2h(req, repo=REPO)
    req = request(tmp_path, eval_dataset=EvalDataset(tmp_path, "1" * 64))
    with pytest.raises(ScriptedError, match="dataset is not"):
        scripted_h2h(req, repo=REPO)
    assert not (tmp_path / "h2h.looks.jsonl").exists()


def test_a_failed_or_lying_play_script_logs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req = request(tmp_path)
    monkeypatch.setattr(scripted, "_run_node", FakeNode(record_for(req.config), fail=True))
    with pytest.raises(PlayError, match="wall-time cap"):
        scripted_h2h(req, repo=REPO)
    monkeypatch.setattr(scripted, "_run_node", FakeNode(record_for(req.config), lie=True))
    with pytest.raises(ScriptedError, match="not the one the play script reported"):
        scripted_h2h(req, repo=REPO)
    assert not (tmp_path / "h2h").exists()
    assert not (tmp_path / "h2h.looks.jsonl").exists()


def test_a_play_script_that_prints_no_json_is_a_play_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req = request(tmp_path)
    fake = FakeNode(record_for(req.config))
    monkeypatch.setattr(scripted, "_run_node", lambda *a, **k: (fake(*a, **k), "not json\n")[1])
    with pytest.raises(PlayError, match="did not print a JSON line"):
        scripted_h2h(req, repo=REPO)
    assert not (tmp_path / "h2h.looks.jsonl").exists()


def test_a_broken_look_log_stops_the_command_before_it_shows_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req = request(tmp_path)
    (tmp_path / "h2h.looks.jsonl").write_text("not json\n", encoding="utf-8")
    monkeypatch.setattr(scripted, "_run_node", FakeNode(record_for(req.config)))
    with pytest.raises(LookError):
        scripted_h2h(req, repo=REPO)
    assert not (tmp_path / "h2h").exists()


def test_run_node_reports_how_the_child_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Done:
        def __init__(self, code: int) -> None:
            self.returncode, self.stdout, self.stderr = code, "out", "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done(0))
    assert scripted._run_node("node", REPO, ["--x"], None) == "out"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done(1))
    with pytest.raises(PlayError, match=r"exit 1.*boom"):
        scripted._run_node("node", REPO, [], None)

    def timeout(*a: Any, **k: Any) -> None:
        raise subprocess.TimeoutExpired("node", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(PlayError, match="killed after"):
        scripted._run_node("node", REPO, [], 1.0)

    def missing(*a: Any, **k: Any) -> None:
        raise FileNotFoundError("no node")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(PlayError, match="cannot start"):
        scripted._run_node("nodex", REPO, [], None)


def test_the_main_command_prints_the_summary_and_maps_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = config_file(tmp_path)
    base = [
        "scripted-h2h",
        *("--config", str(config)),
        *("--arm-a", "damage-aware@v1"),
        *("--arm-b", "damage-aware@v2"),
        *("--pairs", "2"),
    ]
    monkeypatch.setattr(scripted, "_run_node", FakeNode(record_for(config, 60_000)))
    code = main([*base, *("--out", str(tmp_path / "ok")), *("--wall-time-cap-minutes", "1")])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["summary"]["arms"] == {"a": "damage-aware@v1", "b": "damage-aware@v2"}
    monkeypatch.setattr(scripted, "_run_node", FakeNode(record_for(config), fail=True))
    assert main([*base, *("--out", str(tmp_path / "fails"))]) == EXIT_PLAY
    assert main(
        [
            *base[:-2],
            "--pairs",
            "1",
            *("--arm-b", "damage-aware@v1"),
            "--out",
            str(tmp_path / "same"),
        ]
    ) in (
        EXIT_REFUSED,
        2,
    )
    assert module._SIMPLE["scripted-h2h"] is module._scripted
