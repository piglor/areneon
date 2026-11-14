"""The teacher imitation mix through the loop (ADR-010 erratum A11.3, #1367): ``import-teacher``,
a Battle-only warm run that mixes the teacher's stream into its updates (report and sidecar keys,
provenance, the C6 consequence), a rerun and kills at four stages reaching the same hashes, and
each refusal."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.checkpoint.provenance import ProvenanceGraph
from areneon_ml.checkpoint.record import CheckpointRecord, read_record
from areneon_ml.learner.config import RunConfig
from areneon_ml.learner.data import LearnerDataError
from areneon_ml.loop import layout
from areneon_ml.loop.__main__ import main
from areneon_ml.loop.driver import export_provenance, sources_registry
from areneon_ml.loop.teacher import TeacherImportError, import_teacher, teacher_dir, teacher_paths
from areneon_ml.loop.warm_start import import_clone

from tests.ml.learner.support import edited_config, make_stream, tiny_config
from tests.ml.loop.support import Killed, killer, manifests
from tests.ml.loop.test_warm_start import clone_ids, clone_output, warm_loop

GENERATIONS = 3
BATTLE = ("battle",)
TEACHER_ID = "damage-aware@v3"


@pytest.fixture(scope="module")
def teacher_out(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """A teacher run's output directory (``stream.jsonl`` and ``training-set.json``)."""
    out = tmp_path_factory.mktemp("teacher")
    text, manifest = make_stream(policy=TEACHER_ID)
    (out / "stream.jsonl").write_text(text)
    layout.write_json(out / "training-set.json", manifest)
    return out, sha256_hex(jcs(manifest))


@pytest.fixture(scope="module")
def clone_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return clone_output(tmp_path_factory.mktemp("clone"))


def imitation_config(
    ids: dict[str, str], digest: str, *, key: str = "im", warmup: int | None = 1, init: int = 0
) -> RunConfig:
    def edit(raw: dict[str, Any]) -> None:
        raw["warm_start"] = {"battle": ids["battle"]}
        raw["seeds"]["run_key"] = key
        raw["seeds"]["init_preview"] += init  # another seeded Preview, so no shared checkpoint
        if warmup is not None:
            raw["ppo"]["value_warmup"] = {"generations": warmup}
        raw["ppo"]["imitation"] = {
            "training_set_hash": digest,
            "coef_micro": 100_000,
            "share_ppm": 500_000,
            "schedule": {"kind": "linear", "final_micro": 0, "generations": 2},
        }

    return edited_config(edit)


def started(root: Path, clone_dir: Path, teacher: Path, config: RunConfig, upto: int) -> Path:
    run_dir = root / "run"
    import_clone(clone_dir, run_dir, BATTLE)
    import_teacher(teacher, run_dir)
    warm_loop(run_dir, config).run(upto)
    return run_dir


@pytest.fixture(scope="module")
def finished(
    tmp_path_factory: pytest.TempPathFactory, clone_dir: Path, teacher_out: tuple[Path, str]
) -> tuple[Path, RunConfig]:
    config = imitation_config(clone_ids(clone_dir), teacher_out[1])
    return started(
        tmp_path_factory.mktemp("im"), clone_dir, teacher_out[0], config, GENERATIONS
    ), config


# -- import-teacher ------------------------------------------------------------------------------


def test_import_copies_the_stream_and_manifest_and_a_rerun_is_a_no_op(
    tmp_path: Path, teacher_out: tuple[Path, str]
) -> None:
    out, digest = teacher_out
    assert import_teacher(out, tmp_path / "run") == {"training_set_hash": digest}
    stream, manifest = teacher_paths(tmp_path / "run", digest)
    assert stream.read_bytes() == (out / "stream.jsonl").read_bytes()
    assert manifest.read_bytes() == (out / "training-set.json").read_bytes()
    stamp = stream.stat().st_mtime_ns
    assert import_teacher(out, tmp_path / "run") == {"training_set_hash": digest}
    assert stream.stat().st_mtime_ns == stamp
    assert not list(teacher_dir(tmp_path / "run", digest).glob(".*"))


def test_the_command_prints_the_hash(
    tmp_path: Path, teacher_out: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    out, digest = teacher_out
    args = ["import-teacher", "--stream-dir", str(out), "--run-dir", str(tmp_path / "r")]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == {"training_set_hash": digest}


def test_import_never_replaces_other_bytes(tmp_path: Path, teacher_out: tuple[Path, str]) -> None:
    out, digest = teacher_out
    run_dir = tmp_path / "run"
    import_teacher(out, run_dir)
    stream, _ = teacher_paths(run_dir, digest)
    stream.write_text("other\n")
    with pytest.raises(TeacherImportError, match="other bytes"):
        import_teacher(out, run_dir)
    assert stream.read_text() == "other\n"


def test_import_refuses_what_the_erratum_refuses(
    tmp_path: Path, teacher_out: tuple[Path, str]
) -> None:
    out, _ = teacher_out
    for change, message in (
        ({"data_class": "private"}, "not synthetic"),
        ({"learner_ready": False}, "not learner_ready"),
        ({"behaviour_policy_id": "policy:" + "ab" * 32}, "scripted teacher"),
    ):
        bad = tmp_path / message.replace(" ", "-")
        shutil.copytree(out, bad)
        layout.write_json(
            bad / "training-set.json", {**layout.read_json(out / "training-set.json"), **change}
        )
        with pytest.raises(LearnerDataError, match=message):
            import_teacher(bad, tmp_path / "run")
    torn = tmp_path / "torn"
    shutil.copytree(out, torn)
    (torn / "stream.jsonl").write_text("{}\n")
    with pytest.raises(TeacherImportError):
        import_teacher(torn, tmp_path / "run")
    missing = tmp_path / "missing"
    missing.mkdir()
    shutil.copy(out / "training-set.json", missing)
    with pytest.raises(TeacherImportError, match="not found"):
        import_teacher(missing, tmp_path / "run")
    assert not (tmp_path / "run" / "teacher").exists()


def test_import_checks_the_hash_against_the_runs_config(
    tmp_path: Path, clone_dir: Path, teacher_out: tuple[Path, str]
) -> None:
    out, _ = teacher_out
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir, BATTLE)
    config = imitation_config(clone_ids(clone_dir), "0" * 64)
    warm_loop(run_dir, config).run(0)  # writes run-config.json
    with pytest.raises(LearnerDataError, match=r"does not hash to ppo\.imitation"):
        import_teacher(out, run_dir)
    plain = tmp_path / "plain"
    warm_loop(plain, edited_config(lambda raw: raw["seeds"].update(run_key="plain"))).run(0)
    with pytest.raises(TeacherImportError, match=r"no ppo\.imitation"):
        import_teacher(out, plain)


# -- the run ---------------------------------------------------------------------------------------


def test_a_run_without_the_import_is_refused(
    tmp_path: Path, clone_dir: Path, teacher_out: tuple[Path, str]
) -> None:
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir, BATTLE)
    config = imitation_config(clone_ids(clone_dir), teacher_out[1])
    with pytest.raises(LearnerDataError, match="import-teacher"):
        warm_loop(run_dir, config).run(1)


def test_a_manifest_of_another_hash_is_refused_at_the_update(
    tmp_path: Path, clone_dir: Path, teacher_out: tuple[Path, str]
) -> None:
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir, BATTLE)
    import_teacher(teacher_out[0], run_dir)
    config = imitation_config(clone_ids(clone_dir), teacher_out[1], key="other")
    stream, manifest = teacher_paths(run_dir, teacher_out[1])
    tampered = {**layout.read_json(manifest), "learner_ready": False}
    layout.write_json(manifest, tampered)
    with pytest.raises(LearnerDataError, match="does not hash"):
        warm_loop(run_dir, config).run(1)
    assert stream.is_file()


def test_the_train_report_and_sidecar_carry_the_term(finished: tuple[Path, RunConfig]) -> None:
    run_dir, _ = finished
    # warm-up 1: update 0 is off; then t = 0, 1 of a linear schedule 100000 -> 0 over 2 updates
    expected = [0, 100_000, 50_000]
    for generation in range(1, GENERATIONS + 1):
        folder = layout.generation_dir(run_dir, generation)
        report = layout.read_json(folder / "train-report.json")
        battle = report["battle"]
        assert battle["imitation_coef_micro"] == expected[generation - 1]
        assert (battle["imitation_steps"] > 0) == (generation > 1)
        assert not any(k.startswith("imitation_") for k in report["preview"])
        sidecar = json.loads((folder / f"learner-{generation}.json").read_bytes())
        assert sidecar["imitation_coef_micro"] == expected[generation - 1]
    zero = json.loads((layout.generation_dir(run_dir, 0) / "learner-0.json").read_bytes())
    assert "imitation_coef_micro" not in zero


def test_an_independent_rerun_gives_identical_generations(
    finished: tuple[Path, RunConfig], tmp_path: Path, clone_dir: Path, teacher_out: tuple[Path, str]
) -> None:
    run_dir, config = finished
    again = started(tmp_path, clone_dir, teacher_out[0], config, GENERATIONS)
    assert manifests(again) == manifests(run_dir)


@pytest.mark.parametrize(
    ("generation", "stage"), [(0, "committed"), (1, "trained"), (2, "played"), (3, "written")]
)
def test_a_killed_run_resumes_with_identical_hashes(  # noqa: PLR0913, PLR0917
    finished: tuple[Path, RunConfig],
    tmp_path: Path,
    clone_dir: Path,
    teacher_out: tuple[Path, str],
    generation: int,
    stage: str,
) -> None:
    complete, config = finished
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir, BATTLE)
    import_teacher(teacher_out[0], run_dir)
    with pytest.raises(Killed):
        warm_loop(run_dir, config, killer(generation, stage)).run(GENERATIONS)
    warm_loop(run_dir, config).run(GENERATIONS)
    assert manifests(run_dir) == manifests(complete)
    for g in range(GENERATIONS + 1):
        for name in (f"learner-{g}.safetensors", f"learner-{g}.json"):
            assert (layout.generation_dir(run_dir, g) / name).read_bytes() == (
                layout.generation_dir(complete, g) / name
            ).read_bytes()


def test_an_absent_block_keeps_a_plain_run_free_of_the_term(tmp_path: Path) -> None:
    run_dir = tmp_path / "plain"
    config = tiny_config()
    warm_loop(run_dir, config).run(1)
    report = layout.read_json(layout.generation_dir(run_dir, 1) / "train-report.json")
    assert not any(k.startswith("imitation_") for role in report.values() for k in role)
    assert not (run_dir / "teacher").exists()


# -- provenance ----------------------------------------------------------------------------------


def test_the_records_list_the_teacher_set_and_the_registry_maps_it_to_nothing(
    finished: tuple[Path, RunConfig], teacher_out: tuple[Path, str], tmp_path: Path
) -> None:
    run_dir, _ = finished
    digest = teacher_out[1]
    assert sources_registry(run_dir)[digest] == []
    for generation in (1, GENERATIONS):
        for role in ("preview", "battle"):
            record = read_record(
                layout.read_json(
                    layout.generation_dir(run_dir, generation) / "records" / f"{role}.json"
                )
            )
            assert digest in record.training_set_hashes
            assert list(record.training_set_hashes) == sorted(set(record.training_set_hashes))
    zero = read_record(
        layout.read_json(layout.generation_dir(run_dir, 0) / "records" / "preview.json")
    )
    assert digest not in zero.training_set_hashes
    out = tmp_path / "export"
    export_provenance(run_dir, out)
    graph = ProvenanceGraph(
        [read_record(json.loads(p.read_text())) for p in (out / "records").iterdir()],
        json.loads((out / "sources.json").read_text()),
    )
    last = layout.latest_complete(run_dir)
    assert last is not None
    assert digest in graph.ancestry(last.manifest["checkpoints"]["battle"]).sources


def _graph(runs: list[Path], tmp_path: Path) -> tuple[ProvenanceGraph, list[str]]:
    records: list[CheckpointRecord] = []
    sources: dict[str, list[str]] = {}
    finals: list[str] = []
    for number, run in enumerate(runs):
        out = tmp_path / f"export-{number}"
        export_provenance(run, out)
        records += [read_record(json.loads(p.read_text())) for p in (out / "records").iterdir()]
        sources |= json.loads((out / "sources.json").read_text())
        final = layout.latest_complete(run)
        assert final is not None
        finals.append(final.manifest["checkpoints"]["battle"])
    return ProvenanceGraph(records, sources), finals


def test_two_runs_on_one_teacher_set_are_not_independent(
    finished: tuple[Path, RunConfig], tmp_path: Path, teacher_out: tuple[Path, str]
) -> None:
    run_dir, _ = finished
    # another clone (another teacher set behind it, other seeds) that mixes the SAME teacher stream
    other = clone_output(tmp_path / "clone", teacher="9" * 64, seed=300)
    config = imitation_config(clone_ids(other), teacher_out[1], key="im-2", init=500)
    second = started(tmp_path / "second", other, teacher_out[0], config, 1)
    graph, (first, again) = _graph([run_dir, second], tmp_path / "a")
    verdict = graph.independent(first, again)
    assert not verdict.independent
    assert teacher_out[1] in verdict.shared_sources
    # a run that mixes no teacher stream, on the same other clone, shares no source
    plain = tmp_path / "plain"
    import_clone(other, plain, BATTLE)

    def edit(raw: dict[str, Any]) -> None:
        raw["warm_start"] = {"battle": clone_ids(other)["battle"]}
        raw["seeds"]["run_key"] = "im-3"
        raw["seeds"]["init_preview"] += 900

    warm_loop(plain, edited_config(edit)).run(1)
    graph, (first, third) = _graph([run_dir, plain], tmp_path / "b")
    assert teacher_out[1] not in graph.independent(first, third).shared_sources
