"""``python -m areneon_ml.loop inspect``: where it writes, what it refuses, the real Node pool."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.inspector import command
from areneon_ml.inspector.pool import SCRIPT, Pool, load_pool, read_pool_text
from areneon_ml.inspector.readers import InspectError
from areneon_ml.loop import __main__ as loop_main
from areneon_ml.loop import layout

from tests.ml.inspector.support import FIXTURE, ROOT, copy_run, fixture_pool

SIM = ROOT / "vendor" / "pokemon-showdown" / "dist" / "sim"


def loader(pool: Pool) -> command.PoolLoader:
    def load(repo: Path, node: str, config: Path) -> Pool:
        assert config.name == layout.CONFIG
        return pool

    return load


def test_it_writes_the_pages_beside_the_run_and_never_inside_it(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    assert command.default_out(run) == tmp_path / "run.inspect"
    done = command.inspect_run(
        run, None, repo=ROOT, node="node", pool_loader=loader(fixture_pool())
    )
    out = Path(done["out"])
    assert out == tmp_path / "run.inspect"
    names = sorted(p.name for p in out.iterdir())
    assert done["pages"] == len(names)
    assert "index.html" in names
    assert done["bytes"] == sum((out / n).stat().st_size for n in names)
    assert not (run / "run.inspect").exists()
    with pytest.raises(InspectError, match="inside the run directory"):
        command.inspect_run(
            run, run / "pages", repo=ROOT, node="node", pool_loader=loader(fixture_pool())
        )
    with pytest.raises(InspectError, match="inside the run directory"):
        command.inspect_run(run, run, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))


def test_it_changes_nothing_in_the_run_directory_and_replaces_an_earlier_output(
    tmp_path: Path,
) -> None:
    run = copy_run(tmp_path)
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    out = tmp_path / "pages"
    out.mkdir()
    (out / "teams-0099.html").write_text("old")
    command.inspect_run(run, out, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))
    assert not (out / "teams-0099.html").exists()
    assert before == {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    first = {p.name: p.read_bytes() for p in out.glob("*.html")}
    command.inspect_run(run, out, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))
    assert first == {p.name: p.read_bytes() for p in out.glob("*.html")}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pages", "run"]  # no partial or backup


def test_it_refuses_a_directory_that_is_not_an_inspector_output(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    makers: list[Callable[[Path], object]] = [
        lambda d: (d / "notes.txt").write_text("mine"),
        lambda d: (d / "sub").mkdir(),
        lambda d: (d / "index.html").symlink_to(tmp_path),
    ]
    for make in makers:
        out = tmp_path / "pages"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        make(out)
        before = sorted(p.name for p in out.iterdir())
        with pytest.raises(InspectError, match="something other than inspector pages"):
            command.inspect_run(
                run, out, repo=ROOT, node="node", pool_loader=loader(fixture_pool())
            )
        assert sorted(p.name for p in out.iterdir()) == before
    file = tmp_path / "afile"
    file.write_text("x")
    with pytest.raises(InspectError, match="something other than inspector pages"):
        command.inspect_run(run, file, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))


def test_a_failed_write_leaves_the_earlier_output_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = copy_run(tmp_path)
    out = tmp_path / "pages"
    command.inspect_run(run, out, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    real = layout.write_file
    calls: list[str] = []

    def failing(path: Path, data: bytes) -> None:
        calls.append(path.name)
        if len(calls) == 3:
            raise OSError("disk full")
        real(path, data)

    monkeypatch.setattr(layout, "write_file", failing)
    with pytest.raises(OSError, match="disk full"):
        command.inspect_run(run, out, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pages", "run"]
    # A failure after the old output was moved aside puts it back.
    monkeypatch.setattr(layout, "write_file", real)
    replace = os.replace

    def once(src: Path, dst: Path) -> None:
        if Path(dst) == out and Path(src).name.endswith(".partial"):
            raise OSError("rename failed")
        replace(src, dst)

    monkeypatch.setattr(os, "replace", once)
    with pytest.raises(OSError, match="rename failed"):
        command.inspect_run(run, out, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pages", "run"]


def test_it_refuses_before_writing_anything(tmp_path: Path) -> None:
    run = copy_run(tmp_path)
    raw = json.loads((run / layout.CONFIG).read_text())
    raw["lineage_id"] = "private-main"
    layout.write_json(run / layout.CONFIG, raw)
    out = tmp_path / "pages"
    with pytest.raises(InspectError, match="not a public one"):
        command.inspect_run(run, out, repo=ROOT, node="node", pool_loader=loader(fixture_pool()))
    assert not out.exists()


def test_main_prints_one_json_line_and_maps_a_refusal_to_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = copy_run(tmp_path)
    monkeypatch.setattr(command, "load_pool", loader(fixture_pool()))
    monkeypatch.setattr(
        loop_main,
        "inspect_run",
        lambda *a, **k: command.inspect_run(*a, **k, pool_loader=loader(fixture_pool())),
    )
    assert loop_main.main(["inspect", "--run-dir", str(run), "--out", str(tmp_path / "o")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["pages"] > 5
    assert printed["out"] == str(tmp_path / "o")
    shutil.rmtree(layout.generation_dir(run, 1))
    assert (
        loop_main.main(["inspect", "--run-dir", str(run), "--out", str(tmp_path / "p")])
        == loop_main.EXIT_REFUSED
    )
    assert "no gap" in capsys.readouterr().err


def test_a_pool_script_failure_or_bad_output_is_an_inspect_error(tmp_path: Path) -> None:
    with pytest.raises(InspectError, match="cannot start"):
        load_pool(ROOT, str(tmp_path / "no-such-node"), FIXTURE / layout.CONFIG)
    fake = tmp_path / "node"
    fake.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    fake.chmod(0o755)
    with pytest.raises(InspectError, match=r"failed \(exit 3\): boom"):
        load_pool(ROOT, str(fake), FIXTURE / layout.CONFIG)
    with pytest.raises(InspectError, match="did not print JSON"):
        read_pool_text("not json")


def test_the_pool_listing_refuses_what_it_does_not_know() -> None:
    raw = json.loads((FIXTURE / "pool.json").read_text())

    def bad(change: Callable[[dict[str, Any]], object]) -> Pool:
        value = json.loads(json.dumps(raw))
        change(value)
        return read_pool_text(json.dumps(value))

    for change, message in (
        (lambda v: v.update(extra=1), "extra"),
        (lambda v: v.update(source="other"), "schema 1"),
        (lambda v: v.update(dataset_hash="a" * 64), "only a dataset pool"),
        (lambda v: v["teams"][0].update(extra=1), "extra"),
        (lambda v: v["teams"][0]["sets"][0].update(mega_stone="no"), "mega_stone"),
        (lambda v: v["teams"][0]["sets"][0]["evs"].update(hp="x"), "evs hp"),
        (lambda v: v["teams"][0]["sets"][0]["evs"].update(luck=1), "luck"),
        (lambda v: v["teams"][0]["sets"][0]["evs"].pop("hp"), "lacks hp"),
        (lambda v: v["teams"][0].update(canonical_team_hash="x"), "canonical hash"),
        (lambda v: v["teams"][0].update(labels=[{"label_id": "a"}]), "lacks value"),
        (lambda v: v["teams"].reverse(), "sorted and unique"),
        (lambda v: v.update(teams=[]), "sorted and unique"),
        (lambda v: v["teams"][0].update(hash="z"), "SHA-256"),
    ):
        with pytest.raises(InspectError, match=message):
            bad(change)
    dataset = json.loads(json.dumps(raw))
    dataset.update(source="dataset", dataset_hash="a" * 64)
    dataset["teams"][0].update(
        canonical_team_hash="b" * 64, split="dev", labels=[{"label_id": "a", "value": "b"}]
    )
    assert read_pool_text(json.dumps(dataset)).dataset_hash == "a" * 64


@pytest.mark.skipif(not SIM.is_dir(), reason="the pinned engine is not built")
def test_the_real_pool_script_prints_the_committed_listing() -> None:
    """The Node side, run for real, gives the committed pool (the interim one, for this config)."""
    pool = load_pool(ROOT, "node", FIXTURE / layout.CONFIG)
    assert pool == fixture_pool()
    done = subprocess.run(
        ["node", str(ROOT / SCRIPT), "--config", str(FIXTURE / layout.CONFIG)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout == (FIXTURE / "pool.json").read_text()
