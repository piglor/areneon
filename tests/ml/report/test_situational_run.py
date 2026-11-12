"""``progress --situational`` (#1286): the runner, the summary file and the printed block."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from areneon_ml.report import situational as s
from areneon_ml.report import situational_run as r

from tests.ml.report.test_situational import _history_raw, _raw


def _fake(raw: object, code: int = 0, seen: list[Sequence[str]] | None = None) -> r.Runner:
    def run(command: Sequence[str]) -> int:
        if seen is not None:
            seen.append(command)
        out = Path(command[command.index("--out") + 1])
        out.write_text(json.dumps(raw), encoding="utf8")
        return code

    return run


def test_writes_summary_beside_root_and_prints_rates(tmp_path: Path) -> None:
    root = tmp_path / "run.progress"
    root.mkdir()
    seen: list[Sequence[str]] = []
    out = r.situational_generation(
        root,
        20,
        tmp_path / "run",
        repo=tmp_path,
        node="node",
        max_games=5,
        run=_fake(_raw(), 0, seen),
    )
    command = seen[0]
    assert command[:4] == ["nice", "-n", "19", "node"]
    assert command[command.index("--max-games") + 1] == "5"
    target = tmp_path / "run.progress.situational" / "g000020.max5.json"
    assert out["summary"] == str(target)
    assert out["label"] == s.LABEL
    body = out["matches"]["m"]["policy:aa"]
    assert body["pairs"] == 8
    assert body["protect_threatened"] == 333333
    assert set(r.PRINTED) <= set(body)
    assert [p.name for p in target.parent.iterdir()] == ["g000020.max5.json"]  # no raw leftover
    first = target.read_bytes()
    r.situational_generation(
        root, 20, tmp_path / "run", repo=tmp_path, node="node", max_games=5, run=_fake(_raw())
    )
    assert target.read_bytes() == first
    assert not (root / "g000020.json").exists()  # nothing inside the progress root


def test_failures(tmp_path: Path) -> None:
    root = tmp_path / "p"
    root.mkdir()
    with pytest.raises(s.SituationalError, match="failed"):
        r.situational_generation(
            root, 1, tmp_path / "run", repo=tmp_path, node="n", run=_fake({}, 1)
        )
    assert list((tmp_path / "p.situational").iterdir()) == []
    with pytest.raises(s.SituationalError, match="inside the run"):
        r.situational_generation(root, 1, tmp_path, repo=tmp_path, node="n", run=_fake(_raw()))


def test_default_runner_returns_exit_code() -> None:
    assert r._run(["true"]) == 0
    assert r._run(["false"]) == 1


def test_a_capped_run_never_overwrites_the_full_summary(tmp_path: Path) -> None:
    root = tmp_path / "p"
    root.mkdir()
    run = tmp_path / "run"
    full = r.situational_generation(root, 3, run, repo=tmp_path, node="n", run=_fake(_raw()))
    before = Path(full["summary"]).read_bytes()
    capped = r.situational_generation(
        root, 3, run, repo=tmp_path, node="n", max_games=2, run=_fake(_raw())
    )
    assert Path(full["summary"]).name == "g000003.json"
    assert Path(capped["summary"]).name == "g000003.max2.json"
    assert Path(full["summary"]).read_bytes() == before
    assert sorted(p.name for p in Path(full["summary"]).parent.iterdir()) == [
        "g000003.json",
        "g000003.max2.json",
    ]


def test_history_rows_have_their_own_file_flag_and_printed_block(tmp_path: Path) -> None:
    root = tmp_path / "p"
    root.mkdir()
    seen: list[Sequence[str]] = []
    plain = r.situational_generation(
        root, 3, tmp_path / "run", repo=tmp_path, node="n", run=_fake(_raw())
    )
    out = r.situational_generation(
        root,
        3,
        tmp_path / "run",
        repo=tmp_path,
        node="n",
        history=True,
        run=_fake(_history_raw(), 0, seen),
    )
    assert "--history" in seen[0]
    assert out["summary"].endswith("g000003.history.json")
    assert plain["summary"].endswith("g000003.json")  # the plain summary is kept as it was
    body = out["matches"]["m"]["policy:aa"]
    assert body["history"]["stall_repeat"] == 500_000
    assert set(body["history"]) == set(s.HISTORY_STATISTIC_NAMES)
    assert "history" not in plain["matches"]["m"]["policy:aa"]
