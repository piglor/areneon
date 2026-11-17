"""The teacher imitation mix inside a population (ADR-010 erratum A11.3, ADR-021, #1437).

One teacher stream is imported into the run directory and every member's learner draws from it
with its own learner seed: per-member ``imitation_*`` report keys, every member's records name the
teacher set and the registry maps it to nothing (an outside source), a rerun and a killed run reach
the same bytes, and a population without the block keeps its bytes.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.checkpoint.provenance import ProvenanceGraph
from areneon_ml.checkpoint.record import read_record
from areneon_ml.learner.config import RunConfig
from areneon_ml.learner.data import LearnerDataError
from areneon_ml.learner.imitation import DrawStream, TeacherSet
from areneon_ml.learner.trainer import Learner, TrainReport
from areneon_ml.loop import layout
from areneon_ml.loop.driver import export_provenance, sources_registry
from areneon_ml.loop.population import member_config
from areneon_ml.loop.teacher import import_teacher
from areneon_ml.loop.warm_start import import_clone

from tests.ml.learner.support import make_stream
from tests.ml.loop.support import Killed, killer
from tests.ml.loop.test_population import (
    LABELS,
    gen_path,
    make_population,
    population_config,
    tree_bytes,
)
from tests.ml.loop.test_warm_start import clone_ids, clone_output

GENERATIONS = 3


@pytest.fixture(scope="module")
def teacher_out(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    out = tmp_path_factory.mktemp("teacher")
    text, manifest = make_stream(policy="damage-aware@v3")
    (out / "stream.jsonl").write_text(text)
    layout.write_json(out / "training-set.json", manifest)
    return out, sha256_hex(jcs(manifest))


@pytest.fixture(scope="module")
def clone_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return clone_output(tmp_path_factory.mktemp("clone"))


def config_for(ids: dict[str, str], digest: str, key: str = "pop-im") -> RunConfig:
    def edit(raw: dict[str, Any]) -> None:
        raw["warm_start"] = {"battle": ids["battle"]}
        raw["ppo"]["value_warmup"] = {"generations": 1}
        raw["ppo"]["imitation"] = {
            "training_set_hash": digest,
            "coef_micro": 100_000,
            "share_ppm": 500_000,
            "schedule": {"kind": "linear", "final_micro": 0, "generations": 2},
        }

    return population_config(key=key, edit=edit)


def started(root: Path, clone_dir: Path, teacher: Path, config: RunConfig, upto: int) -> Path:
    run_dir = root / "run"
    import_clone(clone_dir, run_dir, ("battle",))
    import_teacher(teacher, run_dir)
    make_population(run_dir, config).run(upto)
    return run_dir


@pytest.fixture(scope="module")
def finished(
    tmp_path_factory: pytest.TempPathFactory, clone_dir: Path, teacher_out: tuple[Path, str]
) -> tuple[Path, RunConfig]:
    config = config_for(clone_ids(clone_dir), teacher_out[1])
    run_dir = started(tmp_path_factory.mktemp("im"), clone_dir, teacher_out[0], config, GENERATIONS)
    return run_dir, config


def test_every_member_reports_the_term_and_names_the_teacher_set(
    finished: tuple[Path, RunConfig], teacher_out: tuple[Path, str]
) -> None:
    run_dir, _ = finished
    digest = teacher_out[1]
    expected = [0, 100_000, 50_000]  # warm-up 1: update 0 is off, then t = 0, 1 of the schedule
    for generation in range(1, GENERATIONS + 1):
        for label in LABELS:
            member = gen_path(run_dir, generation) / "members" / label
            battle = layout.read_json(member / "train-report.json")["battle"]
            assert battle["imitation_coef_micro"] == expected[generation - 1]
            assert (battle["imitation_steps"] > 0) == (generation > 1)
            assert not any(
                k.startswith("imitation_")
                for k in layout.read_json(member / "train-report.json")["preview"]
            )
            for role in ("preview", "battle"):
                record = read_record(layout.read_json(member / "records" / f"{role}.json"))
                assert digest in record.training_set_hashes
                assert list(record.training_set_hashes) == sorted(set(record.training_set_hashes))
    # the Battle network is trained in every generation: a new checkpoint with a trained record
    # (its parent, its own generation's set and the teacher set), never the clone record again
    for label in LABELS:
        previous = None
        for generation in range(GENERATIONS + 1):
            member = gen_path(run_dir, generation) / "members" / label
            record = read_record(layout.read_json(member / "records" / "battle.json"))
            held = layout.read_json(gen_path(run_dir, generation) / "manifest.json")["members"][
                label
            ]
            if previous is not None:
                assert record.checkpoint_id != previous.checkpoint_id
                assert record.parent == previous.checkpoint_id
                assert set(record.training_set_hashes) == {held["training_set_hash"], digest}
            previous = record
    zero = read_record(
        layout.read_json(gen_path(run_dir, 0) / "members" / "m0" / "records" / "preview.json")
    )
    assert digest not in zero.training_set_hashes


def test_members_draw_their_own_teacher_steps(
    tmp_path: Path,
    clone_dir: Path,
    teacher_out: tuple[Path, str],
    finished: tuple[Path, RunConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config = finished
    made: list[tuple[int, int]] = []
    seeds: list[tuple[int, int]] = []  # the stream that drew each fetch
    fetched: list[list[int]] = []
    sizes: list[int] = []
    real_init, real_fetch = DrawStream.__init__, TeacherSet.fetch

    def init(self: DrawStream, learner_seed: int, update: int) -> None:
        made.append((learner_seed, update))
        real_init(self, learner_seed, update)

    def fetch(self: TeacherSet, indices: Sequence[int]) -> Any:
        seeds.append(made[-1])  # drawn from the stream built just before it
        fetched.append(list(indices))
        sizes.append(len(self))
        return real_fetch(self, indices)

    monkeypatch.setattr(DrawStream, "__init__", init)
    monkeypatch.setattr(TeacherSet, "fetch", fetch)
    started(tmp_path, clone_dir, teacher_out[0], config, GENERATIONS)
    assert len(fetched) == len(seeds) > 0
    own = {member_config(config, k).learner_seed for k in range(len(LABELS))}
    for (seed, update), indices, size in zip(seeds, fetched, sizes, strict=True):
        assert seed in own  # a member's own learner seed
        # the member's draw is a function of its seed and the update index alone
        assert indices == DrawStream(seed, update).draw(size, len(indices))
    assert len(own) == len(LABELS)  # the members have different seeds
    by_update: dict[int, list[list[int]]] = {}
    for (_, update), indices in zip(seeds, fetched, strict=True):
        by_update.setdefault(update, []).append(indices)
    assert any(len({tuple(i) for i in group}) > 1 for group in by_update.values())


def test_the_registry_maps_the_teacher_set_to_nothing_and_classify_sees_it(
    finished: tuple[Path, RunConfig], teacher_out: tuple[Path, str], tmp_path: Path
) -> None:
    run_dir, _ = finished
    digest = teacher_out[1]
    assert sources_registry(run_dir)[digest] == []
    out = tmp_path / "export"
    export_provenance(run_dir, out)
    graph = ProvenanceGraph(
        [read_record(json.loads(p.read_text())) for p in (out / "records").iterdir()],
        json.loads((out / "sources.json").read_text()),
    )
    last = layout.latest_complete(run_dir)
    assert last is not None
    for label in LABELS:
        found = last.manifest["members"][label]["checkpoints"]["battle"]
        assert digest in graph.ancestry(found).sources


def test_an_independent_rerun_is_byte_identical(
    finished: tuple[Path, RunConfig], tmp_path: Path, clone_dir: Path, teacher_out: tuple[Path, str]
) -> None:
    run_dir, config = finished
    again = started(tmp_path, clone_dir, teacher_out[0], config, GENERATIONS)
    assert tree_bytes(again) == tree_bytes(run_dir)


@pytest.mark.parametrize(
    ("generation", "stage"), [(1, "trained:m0"), (2, "played"), (3, "written")]
)
def test_a_killed_run_resumes_to_the_same_bytes(  # noqa: PLR0913, PLR0917
    finished: tuple[Path, RunConfig],
    tmp_path: Path,
    clone_dir: Path,
    teacher_out: tuple[Path, str],
    generation: int,
    stage: str,
) -> None:
    complete, config = finished
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir, ("battle",))
    import_teacher(teacher_out[0], run_dir)
    with pytest.raises(Killed):
        make_population(run_dir, config, on_stage=killer(generation, stage)).run(GENERATIONS)
    make_population(run_dir, config).run(GENERATIONS)
    assert tree_bytes(run_dir) == tree_bytes(complete)


def test_a_run_without_the_import_is_refused(
    tmp_path: Path, clone_dir: Path, teacher_out: tuple[Path, str]
) -> None:
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir, ("battle",))
    config = config_for(clone_ids(clone_dir), teacher_out[1])
    with pytest.raises(LearnerDataError, match="import-teacher"):
        make_population(run_dir, config).run(1)


def test_a_population_without_the_block_has_no_term_and_no_teacher(tmp_path: Path) -> None:
    run_dir = tmp_path / "plain"
    make_population(run_dir).run(1)
    for label in LABELS:
        report = layout.read_json(gen_path(run_dir, 1) / "members" / label / "train-report.json")
        assert not any(k.startswith("imitation_") for role in report.values() for k in role)
    assert not (run_dir / "teacher").exists()


def test_a_quarantined_then_copied_member_gets_no_teacher_hash_it_did_not_train_on(
    tmp_path: Path,
    clone_dir: Path,
    teacher_out: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = teacher_out[1]
    base = config_for(clone_ids(clone_dir), digest, key="pop-im-health")
    four = population_config(
        members=4,
        key="pop-im-health",
        edit=lambda raw: raw.update(warm_start=base.raw["warm_start"], ppo=base.raw["ppo"]),
    )
    real = Learner.train
    state: dict[str, Any] = {}

    def forced(self: Learner, steps: Any, labels: Any = None, teacher: Any = None) -> TrainReport:
        report = real(self, steps, labels, teacher)
        names = [m for m, learner in state["loop"]._learners.items() if learner is self]
        if names == ["m1"] and self.generation == 2:  # generation 2 (the counter equals it)
            roles = dict(report.roles)
            roles["battle"] = dataclasses.replace(roles["battle"], policy_loss=float("nan"))
            return TrainReport(roles)
        return report

    monkeypatch.setattr(Learner, "train", forced)
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir, ("battle",))
    import_teacher(teacher_out[0], run_dir)
    loop = make_population(run_dir, four)
    state["loop"] = loop
    loop.run(4)

    def record_bytes(generation: int, member: str, role: str) -> bytes:
        return (
            gen_path(run_dir, generation) / "members" / member / "records" / f"{role}.json"
        ).read_bytes()

    def hashes(generation: int, member: str) -> tuple[str, ...]:
        found = read_record(json.loads(record_bytes(generation, member, "preview")))
        return tuple(found.training_set_hashes)

    manifest = {g: layout.read_json(gen_path(run_dir, g) / "manifest.json") for g in (1, 2, 3)}
    assert digest in hashes(1, "m1")  # trained in generation 1
    assert digest in hashes(2, "m0")  # a healthy member keeps it
    # quarantined in generation 2: no training set, and the records it had, byte for byte
    assert manifest[2]["members"]["m1"]["training_set_hash"] is None
    for role in ("preview", "battle"):
        assert record_bytes(2, "m1", role) == record_bytes(1, "m1", role)
    assert hashes(2, "m1") == hashes(1, "m1")  # no teacher hash added that generation
    # the next generation's copy of the top member trains again: ordinary records, teacher named
    assert manifest[3]["members"]["m1"]["training_set_hash"] is not None
    assert digest in hashes(3, "m1")
    # a resumed driver re-checks every record against its manifest, copies included
    make_population(run_dir, four).run(4)
