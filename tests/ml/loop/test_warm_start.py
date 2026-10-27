"""A warm start: generation 0 from a declared cloned pair (ADR-010 erratum A2, ADR-019 §4, §6)."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.checkpoint.files import checkpoint_bytes, checkpoint_id, write_checkpoint
from areneon_ml.checkpoint.provenance import ProvenanceGraph
from areneon_ml.checkpoint.record import CheckpointRecord, read_record
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.__main__ import EXIT_REFUSED, main
from areneon_ml.loop.driver import (
    LoopError,
    export_provenance,
    sources_registry,
)
from areneon_ml.loop.warm_start import (
    WarmStartError,
    import_clone,
    record_path,
)
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.network import ROLES, init_network

from tests.ml.learner.support import edited_config, tiny_config
from tests.ml.learner.test_clone import teacher_stream, written
from tests.ml.loop.support import MANIFEST, FakePlayer, Killed, killer, make_loop, manifests
from tests.ml.model.support import game

TEACHER = "c" * 64


def clone_output(
    root: Path,
    *,
    teacher: str = TEACHER,
    seed: int = 100,
    model: dict[str, Any] | None = None,
    record_edit: Callable[[str, dict[str, Any]], None] | None = None,
) -> Path:
    """What ``learner clone`` writes (``checkpoints/``, ``records/``, ``sources.json``).

    Untrained networks stand in for the cloned ones: the loop reads the files, never the training.
    """
    config = tiny_config()
    shape = config.model
    if model is not None:
        shape = type(shape).from_dict(model)
    version = encoder_version(game(), shape.encoder)
    for offset, role in enumerate(ROLES):
        network = init_network(role, game(), shape, seed + offset)
        found = write_checkpoint(root, checkpoint_bytes(network, game()))
        raw: dict[str, Any] = {
            "schema_version": 1,
            "checkpoint_id": found,
            "role": role,
            "lineage_id": "tiny-clone",
            "league_opponents": [],
            "training_set_hashes": [teacher],
            "team_set_manifest_hash": "e" * 64,
            "run_config_hash": "f" * 64,
            "learner_state_hash": "d" * 64,
            "encoder_version": version,
            "data_class": "synthetic",
            "manifest": MANIFEST,
        }
        if record_edit is not None:
            record_edit(role, raw)
        layout.write_json(root / "records" / f"{found}.json", read_record(raw).to_json())
    (root / "sources.json").write_bytes(jcs({teacher: []}))
    return root


def clone_ids(clone_dir: Path) -> dict[str, str]:
    out = {}
    for path in (clone_dir / "records").glob("*.json"):
        record = read_record(json.loads(path.read_text()))
        out[record.role] = record.checkpoint_id
    return out


def warm_config(ids: dict[str, str], run_key: str = "tiny-run") -> RunConfig:
    def edit(raw: dict[str, Any]) -> None:
        raw["warm_start"] = {"preview": ids["preview"], "battle": ids["battle"]}
        raw["seeds"]["run_key"] = run_key

    return edited_config(edit)


def warm_loop(run_dir: Path, config: RunConfig, hook: Any = None) -> Any:
    return make_loop(run_dir, config, FakePlayer(config=config), hook)


@pytest.fixture(scope="module")
def clone_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return clone_output(tmp_path_factory.mktemp("clone"))


@pytest.fixture(scope="module")
def config(clone_dir: Path) -> RunConfig:
    return warm_config(clone_ids(clone_dir))


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory, clone_dir: Path, config: RunConfig) -> Path:
    """A warm-started run of two generations, never interrupted."""
    run_dir = tmp_path_factory.mktemp("warm")
    import_clone(clone_dir, run_dir)
    warm_loop(run_dir, config).run(2)
    return run_dir


def fresh_run_dir(tmp_path: Path, clone_dir: Path) -> Path:
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)
    return run_dir


# -- generation 0 ---------------------------------------------------------------------------------


def test_generation_zero_freezes_the_cloned_pair_with_the_clone_records(
    finished: Path, clone_dir: Path
) -> None:
    ids = clone_ids(clone_dir)
    zero = layout.verify_generation(layout.generation_dir(finished, 0), 0)
    assert zero["checkpoints"] == ids
    assert zero["training_set_hash"] is None
    for role in ROLES:
        stored = layout.generation_dir(finished, 0) / "records" / f"{role}.json"
        source = clone_dir / "records" / f"{ids[role]}.json"
        assert stored.read_bytes() == source.read_bytes()
        assert (
            checkpoint_id((finished / "checkpoints" / f"{ids[role]}.safetensors").read_bytes())
            == ids[role]
        )


def test_generation_one_descends_from_the_cloned_pair(finished: Path, clone_dir: Path) -> None:
    ids = clone_ids(clone_dir)
    for role in ROLES:
        record = read_record(
            layout.read_json(layout.generation_dir(finished, 1) / "records" / f"{role}.json")
        )
        assert record.parent == ids[role]
        assert (
            record.run_config_hash
            != read_record(
                layout.read_json(layout.generation_dir(finished, 0) / "records" / f"{role}.json")
            ).run_config_hash
        )


def test_the_learner_starts_fresh_from_the_cloned_weights(finished: Path, clone_dir: Path) -> None:
    sidecar = json.loads((layout.generation_dir(finished, 0) / "learner-0.json").read_text())
    assert sidecar["run_config_hash"] == warm_config(clone_ids(clone_dir)).hash
    assert sidecar["generation"] == 0
    # no learner step and no carried-over moments: the optimiser has taken no step
    assert sidecar["optimizer_steps"] == {"battle": {}, "preview": {}}


def test_a_run_without_warm_start_is_unchanged(tmp_path: Path) -> None:
    assert tiny_config().warm_start is None
    make_loop(tmp_path).run(0)
    zero = layout.verify_generation(layout.generation_dir(tmp_path, 0), 0)
    seeded = read_record(
        layout.read_json(layout.generation_dir(tmp_path, 0) / "records/preview.json")
    )
    assert seeded.checkpoint_id == zero["checkpoints"]["preview"]
    assert seeded.training_set_hashes == ()
    assert not (tmp_path / "warm-start").exists()


# -- determinism and resume ----------------------------------------------------------------------


def test_an_independent_warm_run_gives_identical_generations(
    finished: Path, tmp_path: Path, clone_dir: Path, config: RunConfig
) -> None:
    run_dir = fresh_run_dir(tmp_path, clone_dir)
    warm_loop(run_dir, config).run(2)
    assert manifests(run_dir) == manifests(finished)


@pytest.mark.parametrize(
    ("generation", "stage"),
    [(0, "written"), (0, "committed"), (1, "played"), (1, "trained"), (2, "written")],
)
def test_a_warm_run_killed_anywhere_resumes_with_identical_hashes(
    finished: Path, tmp_path: Path, clone_dir: Path, generation: int, stage: str
) -> None:
    config = warm_config(clone_ids(clone_dir))
    run_dir = fresh_run_dir(tmp_path, clone_dir)
    with pytest.raises(Killed):
        warm_loop(run_dir, config, killer(generation, stage)).run(2)
    result = warm_loop(run_dir, config).run(2)
    assert result.resumed_from == (
        generation if stage == "committed" else generation - 1 if generation else None
    )
    assert manifests(run_dir) == manifests(finished)
    for g in range(3):
        for name in (f"learner-{g}.safetensors", f"learner-{g}.json"):
            assert (layout.generation_dir(run_dir, g) / name).read_bytes() == (
                layout.generation_dir(finished, g) / name
            ).read_bytes()


def test_resuming_at_generation_zero_refuses_records_that_are_not_the_clone_records(
    tmp_path: Path, clone_dir: Path, config: RunConfig
) -> None:
    run_dir = fresh_run_dir(tmp_path, clone_dir)
    warm_loop(run_dir, config).run(0)
    path = record_path(run_dir, "battle")
    other = json.loads(path.read_text())
    other["lineage_id"] = "elsewhere"
    path.write_bytes(jcs(other) + b"\n")
    with pytest.raises(LoopError, match="not the warm-start clone records"):
        warm_loop(run_dir, config).run(1)


# -- refusals ------------------------------------------------------------------------------------


def test_a_pair_that_was_never_imported_is_refused(tmp_path: Path, config: RunConfig) -> None:
    with pytest.raises(LoopError, match="import-clone"):
        warm_loop(tmp_path, config).run(0)
    assert not list(layout.generations_dir(tmp_path).glob("g*"))


def test_a_checkpoint_of_the_other_role_is_refused(tmp_path: Path, clone_dir: Path) -> None:
    ids = clone_ids(clone_dir)
    run_dir = fresh_run_dir(tmp_path, clone_dir)
    swapped = warm_config({"preview": ids["battle"], "battle": ids["preview"]})
    with pytest.raises(LoopError, match="holds a preview network, not battle"):
        warm_loop(run_dir, swapped).run(0)


def test_a_checkpoint_whose_bytes_do_not_hash_to_its_id_is_refused(
    tmp_path: Path, clone_dir: Path, config: RunConfig
) -> None:
    run_dir = fresh_run_dir(tmp_path, clone_dir)
    ids = clone_ids(clone_dir)
    path = run_dir / "checkpoints" / f"{ids['preview']}.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(LoopError, match="hash to another id"):
        warm_loop(run_dir, config).run(0)


def test_a_checkpoint_made_under_another_model_section_is_refused(tmp_path: Path) -> None:
    other = copy.deepcopy(tiny_config().raw["model"])
    other["encoder"]["volatile_buckets"] += 1  # another encoder version and layout, a valid network
    clone_dir = clone_output(tmp_path / "clone", model=other)
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)
    with pytest.raises(LoopError, match="another model section"):
        warm_loop(run_dir, warm_config(clone_ids(clone_dir))).run(0)


def test_a_record_made_for_another_encoder_version_is_refused(tmp_path: Path) -> None:
    def edit(_role: str, raw: dict[str, Any]) -> None:
        raw["encoder_version"] = "enc9:" + "0" * 64

    clone_dir = clone_output(tmp_path / "clone", record_edit=edit)
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)
    with pytest.raises(LoopError, match="another encoder version"):
        warm_loop(run_dir, warm_config(clone_ids(clone_dir))).run(0)


def test_a_record_made_for_another_format_is_refused(tmp_path: Path) -> None:
    def edit(_role: str, raw: dict[str, Any]) -> None:
        raw["manifest"] = {**MANIFEST, "format_id": "gen9other"}

    clone_dir = clone_output(tmp_path / "clone", record_edit=edit)
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)
    with pytest.raises(LoopError, match="another format"):
        warm_loop(run_dir, warm_config(clone_ids(clone_dir))).run(0)


def test_a_record_that_is_not_a_clone_record_is_refused_at_the_start(
    tmp_path: Path, clone_dir: Path, config: RunConfig
) -> None:
    run_dir = fresh_run_dir(tmp_path, clone_dir)
    path = record_path(run_dir, "preview")
    plain = json.loads(path.read_text())
    plain["training_set_hashes"] = []
    path.write_bytes(jcs(plain) + b"\n")
    with pytest.raises(LoopError, match="not a clone record"):
        warm_loop(run_dir, config).run(0)


def test_a_record_that_names_another_checkpoint_or_is_missing_is_refused(
    tmp_path: Path, clone_dir: Path, config: RunConfig
) -> None:
    run_dir = fresh_run_dir(tmp_path, clone_dir)
    path = record_path(run_dir, "preview")
    kept = path.read_bytes()
    other = json.loads(kept)
    other["checkpoint_id"] = "9" * 64
    path.write_bytes(jcs(other) + b"\n")
    with pytest.raises(LoopError, match="another checkpoint"):
        warm_loop(run_dir, config).run(0)
    path.unlink()
    with pytest.raises(LoopError, match="missing or invalid"):
        warm_loop(run_dir, config).run(0)


# -- import-clone --------------------------------------------------------------------------------


def test_import_copies_checkpoints_and_records_and_is_idempotent(
    tmp_path: Path, clone_dir: Path
) -> None:
    ids = import_clone(clone_dir, tmp_path / "run")
    assert ids == clone_ids(clone_dir)
    for role, found in ids.items():
        assert (tmp_path / "run" / "checkpoints" / f"{found}.safetensors").is_file()
        assert (
            record_path(tmp_path / "run", role).read_bytes()
            == (clone_dir / "records" / f"{found}.json").read_bytes()
        )
    assert import_clone(clone_dir, tmp_path / "run") == ids


def test_import_never_replaces_a_record_with_different_bytes(
    tmp_path: Path, clone_dir: Path
) -> None:
    import_clone(clone_dir, tmp_path / "run")
    record_path(tmp_path / "run", "battle").write_bytes(b"{}\n")
    with pytest.raises(WarmStartError, match="already holds another battle record"):
        import_clone(clone_dir, tmp_path / "run")


def test_import_refuses_a_checkpoint_that_does_not_hash_to_its_id(
    tmp_path: Path, clone_dir: Path
) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(clone_dir, broken)
    victim = next((broken / "checkpoints").glob("*.safetensors"))
    victim.write_bytes(victim.read_bytes() + b" ")
    with pytest.raises(WarmStartError, match="does not hash to its id"):
        import_clone(broken, tmp_path / "run")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: r.update(training_set_hashes=[]), "not a clone record"),
        (lambda r: r.update(parent="1" * 64), "descends from another checkpoint"),
        (lambda r: r.update(data_class="private"), "not synthetic"),
    ],
)
def test_import_refuses_a_record_that_is_not_a_clone_record(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None], message: str
) -> None:
    clone_dir = clone_output(tmp_path / "clone", record_edit=lambda _role, raw: edit(raw))
    with pytest.raises(WarmStartError, match=message):
        import_clone(clone_dir, tmp_path / "run")


def test_import_refuses_an_output_without_one_record_per_role(
    tmp_path: Path, clone_dir: Path
) -> None:
    partial = tmp_path / "partial"
    shutil.copytree(clone_dir, partial)
    next((partial / "records").glob("*.json")).unlink()
    with pytest.raises(WarmStartError, match="exactly one preview and one battle"):
        import_clone(partial, tmp_path / "run")
    twice = tmp_path / "twice"
    shutil.copytree(clone_dir, twice)
    second = clone_output(tmp_path / "second", seed=7)
    for path in (second / "records").glob("*.json"):
        shutil.copy(path, twice / "records" / path.name)
    for path in (second / "checkpoints").glob("*.safetensors"):
        shutil.copy(path, twice / "checkpoints" / path.name)
    with pytest.raises(WarmStartError, match=r"two (preview|battle) records"):
        import_clone(twice, tmp_path / "run2")


def test_import_refuses_a_record_not_named_by_its_id_and_a_missing_checkpoint(
    tmp_path: Path, clone_dir: Path
) -> None:
    renamed = tmp_path / "renamed"
    shutil.copytree(clone_dir, renamed)
    path = next((renamed / "records").glob("*.json"))
    path.rename(path.with_name("x.json"))
    with pytest.raises(WarmStartError, match="named by its checkpoint id"):
        import_clone(renamed, tmp_path / "run")
    gone = tmp_path / "gone"
    shutil.copytree(clone_dir, gone)
    next((gone / "checkpoints").glob("*.safetensors")).unlink()
    with pytest.raises(WarmStartError, match="No such file"):
        import_clone(gone, tmp_path / "run")


@pytest.mark.parametrize("registry", [b"{}", b'{"%s": ["a"]}' % TEACHER.encode(), b"[", b""])
def test_import_refuses_a_sources_registry_that_does_not_map_the_teacher_set_to_nothing(
    tmp_path: Path, clone_dir: Path, registry: bytes
) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(clone_dir, broken)
    (broken / "sources.json").write_bytes(registry)
    with pytest.raises(WarmStartError, match=r"sources\.json"):
        import_clone(broken, tmp_path / "run")


def test_the_command_imports_and_prints_the_ids_or_exits_with_a_refusal(
    tmp_path: Path, clone_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["import-clone", "--clone-dir", str(clone_dir), "--run-dir", str(tmp_path / "r")])
    assert code == 0
    assert json.loads(capsys.readouterr().out) == {"checkpoints": clone_ids(clone_dir)}
    code = main(
        ["import-clone", "--clone-dir", str(tmp_path / "none"), "--run-dir", str(tmp_path / "s")]
    )
    assert code == EXIT_REFUSED
    assert "areneon_ml.loop:" in capsys.readouterr().err


# -- provenance ----------------------------------------------------------------------------------


def test_the_provenance_export_includes_the_clone_records_and_the_teacher_set(
    finished: Path, clone_dir: Path, tmp_path: Path
) -> None:
    ids = clone_ids(clone_dir)
    registry = sources_registry(finished)
    assert registry[TEACHER] == []
    out = tmp_path / "export"
    digest = export_provenance(finished, out)
    assert sha256_hex((out / "sources.json").read_bytes()) == digest
    for found in ids.values():
        assert (out / "records" / f"{found}.json").is_file()
    assert len(list((out / "records").iterdir())) == 6
    graph = ProvenanceGraph(
        [read_record(json.loads(p.read_text())) for p in (out / "records").iterdir()],
        json.loads((out / "sources.json").read_text()),
    )
    last = json.loads((layout.generation_dir(finished, 2) / "manifest.json").read_text())[
        "checkpoints"
    ]["battle"]
    assert TEACHER in graph.ancestry(last).sources
    assert set(ids.values()) <= graph.ancestry(last).checkpoints


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


def _second_run(tmp_path: Path, teacher: str, seed: int) -> Path:
    other = clone_output(tmp_path / f"clone-{seed}", teacher=teacher, seed=seed)
    cfg = warm_config(clone_ids(other), run_key=f"tiny-run-{seed}")
    run_dir = tmp_path / f"run-{seed}"
    import_clone(other, run_dir)
    warm_loop(run_dir, cfg).run(1)
    return run_dir


def test_independence_follows_the_shared_teacher_set(finished: Path, tmp_path: Path) -> None:
    sharing = _second_run(tmp_path, TEACHER, 200)
    distinct = _second_run(tmp_path, "9" * 64, 300)
    graph, (first, second) = _graph([finished, sharing], tmp_path / "a")
    verdict = graph.independent(first, second)
    assert not verdict.independent
    assert verdict.shared_sources == (TEACHER,)
    assert verdict.shared_checkpoints == ()
    graph, (first, second) = _graph([finished, distinct], tmp_path / "b")
    assert graph.independent(first, second).independent
    # a run with no warm start has no teacher set to share
    plain = tmp_path / "plain"
    unseeded = edited_config(lambda raw: raw["seeds"].update(run_key="plain"))
    warm_loop(plain, unseeded).run(1)
    graph, (first, second) = _graph([finished, plain], tmp_path / "c")
    assert graph.independent(first, second).independent


def test_a_real_cloning_run_output_starts_a_run(tmp_path: Path) -> None:
    report = written(tmp_path, "real", teacher_stream())
    run_dir = tmp_path / "run"
    ids = import_clone(tmp_path / "real", run_dir)
    assert ids == report["checkpoints"]
    cfg = warm_config(ids)
    warm_loop(run_dir, cfg).run(1)
    zero = layout.verify_generation(layout.generation_dir(run_dir, 0), 0)
    assert zero["checkpoints"] == ids
    assert sources_registry(run_dir)[report["teacher_set_hash"]] == []
