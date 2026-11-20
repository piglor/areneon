"""``progress --allow-pruned``: a head to head between runs whose old generations were pruned."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from areneon_ml.loop.__main__ import EXIT_REFUSED, main

from tests.ml.loop.test_pruned_runs import prune_run
from tests.ml.progress.support import dataset_config, make_run
from tests.ml.progress.test_main import args, fake_node


@pytest.fixture
def runs(tmp_path: Path) -> tuple[Path, Path]:
    mine, other = tmp_path / "run", tmp_path / "other"
    make_run(mine, generations=3)
    make_run(other, generations=3, config=dataset_config("other-run", (21, 22)))
    for run_dir in (mine, other):
        prune_run(run_dir, 1)  # older than the evaluated (and newest) generation 2
    return mine, other


def versus(run: Path, other: Path, root: Path, *extra: str) -> int:
    return main(
        args(
            run,
            *("--generation", "2", "--games", "2", "--record-games", "1"),
            *("--versus-run-dir", str(other), "--versus-generation", "2", "--run-games", "4"),
            *("--progress-dir", str(root)),
            *("--anchors", "1", "--previous-matches"),  # opponents of the pruned generation 1
            *extra,
        )
    )


def test_a_head_to_head_of_pruned_runs_needs_the_flag(
    runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mine, other = runs
    fake_node(monkeypatch, mine)
    root = tmp_path / "p"
    assert versus(mine, other, root) == EXIT_REFUSED
    assert "g000001: its files differ from its manifest" in capsys.readouterr().err
    assert versus(mine, other, root, "--allow-pruned") == 0
    (done,) = json.loads(capsys.readouterr().out)["generations"]
    assert done["head_to_head"]["lineage"]["label"] == "independent lineage"
    assert {"anchor-g1-sample", "previous-sample", "run-sample"} <= {
        m["match_id"] for m in done["matches"]
    }
    # The committed directory is verified again on a rerun: the same rule applies.
    assert versus(mine, other, root) == EXIT_REFUSED
    capsys.readouterr()
    assert versus(mine, other, root, "--allow-pruned") == 0


def test_the_evaluated_generation_must_be_whole_even_with_the_flag(
    runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mine, other = runs
    fake_node(monkeypatch, mine)
    code = main(
        args(
            mine,
            *("--generation", "1", "--games", "2", "--record-games", "1"),
            *("--versus-run-dir", str(other), "--versus-generation", "2", "--run-games", "4"),
            *("--progress-dir", str(tmp_path / "p"), "--allow-pruned"),
        )
    )
    assert code == EXIT_REFUSED
    assert "g000001: its files differ from its manifest" in capsys.readouterr().err
