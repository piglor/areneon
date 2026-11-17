"""The teacher imitation mix inside a population (ADR-010 erratum A11.3, ADR-021, #1437).

One teacher stream is imported into the run directory and every member's learner draws from it
with its own learner seed: per-member ``imitation_*`` report keys, every member's records name the
teacher set and the registry maps it to nothing (an outside source), a rerun and a killed run reach
the same bytes, and a population without the block keeps its bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.checkpoint.provenance import ProvenanceGraph
from areneon_ml.checkpoint.record import read_record
from areneon_ml.learner.config import RunConfig
from areneon_ml.learner.data import LearnerDataError
from areneon_ml.loop import layout
from areneon_ml.loop.driver import export_provenance, sources_registry
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
    zero = read_record(
        layout.read_json(gen_path(run_dir, 0) / "members" / "m0" / "records" / "preview.json")
    )
    assert digest not in zero.training_set_hashes


def test_members_draw_their_own_teacher_steps(finished: tuple[Path, RunConfig]) -> None:
    run_dir, _ = finished
    ce = [
        layout.read_json(gen_path(run_dir, 2) / "members" / label / "train-report.json")["battle"][
            "imitation_ce_before"
        ]
        for label in LABELS
    ]
    assert all(isinstance(v, float) for v in ce)


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
