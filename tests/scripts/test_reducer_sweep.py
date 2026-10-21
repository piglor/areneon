"""The reducer sweep's check side, on a committed random-play battle turned into a sweep line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import reducer_sweep
from tests.ml.model.support import RANDOM, candidates_for

ENCODER = reducer_sweep.build_encoder(reducer_sweep.ENGINE_DATA, reducer_sweep.DEFAULT_ENCODER)


def sweep_battle(path: Path = RANDOM[0], mix: str = "random-legal") -> dict[str, Any]:
    """The corpus battle in the shape `scripts/reducer-sweep.ts` writes."""
    source: dict[str, Any] = json.loads(path.read_text())
    channels = {p: source["channels"][p]["lines"] for p in ("p1", "p2")}
    decisions: dict[str, list[dict[str, Any]]] = {}
    for player, lines in channels.items():
        decisions[player] = []
        for position, line in enumerate(lines):
            if not line.startswith("|request|"):
                continue
            body = json.loads(line[len("|request|") :])
            if body.get("wait"):
                continue
            kind = "preview" if body.get("teamPreview") else "battle"
            decisions[player].append(
                {"n": position + 1, "kind": kind, "candidates": candidates_for(body)}
            )
    return {
        "index": 0, "mix": mix, "seed": source["seed"], "teams": source["teams"],
        "channels": channels, "decisions": decisions,
    }  # fmt: skip


def inject(battle: dict[str, Any], line: str) -> dict[str, Any]:
    """The battle with `line` put just before p1's second request (and the decisions moved)."""
    copy: dict[str, Any] = json.loads(json.dumps(battle))
    lines = copy["channels"]["p1"]
    at = [i for i, text in enumerate(lines) if text.startswith("|request|")][1]
    lines.insert(at, line)
    for decision in copy["decisions"]["p1"][1:]:
        decision["n"] += 1
    return copy


def report_of(*battles: dict[str, Any]) -> reducer_sweep.Report:
    report = reducer_sweep.Report()
    for battle in battles:
        reducer_sweep.check_battle(battle, ENCODER, report)
    return report


def test_a_clean_battle_is_checked_at_every_request() -> None:
    battle = sweep_battle()
    report = report_of(battle)
    assert report.unexplained == 0
    assert report.channels == 2
    assert report.queries == sum(len(d) for d in battle["decisions"].values())
    assert "unexplained: 0" in reducer_sweep.render(report)


def test_a_refusal_is_grouped_by_kind_and_keeps_the_shortest_example() -> None:
    report = report_of(
        inject(sweep_battle(), "|bogus|x"), inject(sweep_battle(RANDOM[1]), "|bogus|y")
    )
    assert report.unexplained == 2
    (group,) = report.groups.values()
    assert group.count == 2
    assert not group.excluded
    assert group.example is not None
    assert group.example["line"] in {"|bogus|x", "|bogus|y"}
    assert "REFUSAL x2" in reducer_sweep.render(report)


def test_the_illusion_replace_line_is_the_named_exclusion() -> None:
    report = report_of(inject(sweep_battle(), "|replace|p2a: Zoroark|Zoroark, L50, M"))
    assert report.unexplained == 0
    (group,) = report.groups.values()
    assert group.excluded
    assert "excluded x1" in reducer_sweep.render(report)


def test_a_battle_that_failed_to_play_is_a_finding() -> None:
    failed = {**sweep_battle(), "failure": {"kind": "Error", "message": "engine said 12 no"}}
    report = report_of(failed)
    assert report.unexplained == 1
    assert "BATTLE FAILURE x1: Error: engine said N no" in reducer_sweep.render(report)


def test_the_encoder_s_own_refusals_are_findings_too() -> None:
    battle = sweep_battle()
    battle["decisions"]["p1"][0]["kind"] = "battle"  # a kind that does not match the request
    report = report_of(battle)
    assert report.unexplained == 1
    (group,) = report.groups.values()
    assert group.kind.startswith("EncodingError")


def test_main_reads_files_and_reports_through_the_exit_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good, bad = tmp_path / "good.jsonl", tmp_path / "bad.jsonl"
    good.write_text(json.dumps(sweep_battle()) + "\n\n")
    bad.write_text(json.dumps(inject(sweep_battle(), "|bogus|x")) + "\n")
    assert reducer_sweep.main([str(good)]) == 0
    assert reducer_sweep.main([str(good), str(bad)]) == 1
    assert "battles 2" in capsys.readouterr().out
