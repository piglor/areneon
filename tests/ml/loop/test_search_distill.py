"""Search distillation through the loop (ADR-026 erratum E2, #1291): a warm-started run whose
generations from ``start_generation`` carry a label stream (fake labels here), reads the current and
the previous generation's labels, lists the label sets in the records, reruns byte-identically and
resumes after a kill to the same hashes."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.driver import LoopError, export_provenance, sources_registry
from areneon_ml.loop.play import PlayRequest
from areneon_ml.loop.warm_start import import_clone

from tests.ml.learner.support import edited_config, episode_lines, stream_of
from tests.ml.loop.support import FakePlayer, Killed, killer, manifests, reseal
from tests.ml.loop.test_warm_start import clone_ids, clone_output, warm_config, warm_loop

IDENTITY = "c" * 64
LABEL_CONFIG = "d" * 64
START = 2
RUN_GENERATIONS = 3


def distilling_config(ids: dict[str, str], aggregate: int = 2) -> RunConfig:
    base = warm_config(ids, "tiny-distill").raw

    def edit(raw: dict[str, Any]) -> None:
        raw.update(base)
        raw["team_dataset"] = {"path": "x", "dataset_hash": "5" * 64}
        raw["ppo"]["value_warmup"] = {"generations": 1}
        raw["ppo"]["search_distill"] = {
            "coef_micro": 5_000_000,
            "c_scale_micro": 4_000_000,
            "min_spread_ppm": 20_000,
            "aggregate_generations": aggregate,
        }
        raw["generation"]["search_labels"] = {
            "search_config_hash": LABEL_CONFIG,
            "label_share_ppm": 40_000,
            "start_generation": START,
        }

    return edited_config(edit)


class LabelPlayer(FakePlayer):
    """The fake simulator side, writing a label stream for every generation from ``START``."""

    def __init__(
        self, config: RunConfig, identity: str = IDENTITY, skip: bool = False, empty: bool = False
    ) -> None:
        super().__init__(config=config)
        self.identity = identity
        self.skip = skip
        self.empty = empty

    def identity_for(self, generation: int) -> str:
        return self.identity

    def play(self, request: PlayRequest) -> None:
        super().play(request)
        generation = request.generation
        if generation < START:
            return
        out = request.out_dir
        summary = layout.read_json(out / "generation.json")
        summary["search_labels"] = {
            "search_config_hash": LABEL_CONFIG,
            "search_identity_hash": self.identity_for(generation),
        }
        layout.write_json(out / "generation.json", summary)
        if self.skip:
            # One of the two label files only: an incomplete stream.
            layout.write_json(out / "label-set.json", {})
            return
        if self.empty:
            return  # a generation in which nothing was labelled writes no label files
        picker, rng = random.Random(generation), random.Random(100 + generation)
        examples = []
        for player in ("p1", "p2"):
            for n, base in enumerate(episode_lines(generation, player, picker)):
                open_ = [i for i, c in enumerate(base["candidates"]) if not c["pending"]]
                if base["observation"]["kind"] != "battle" or len(open_) < 2 or n % 2:
                    continue
                size = len(base["candidates"])
                prior = [rng.randrange(0, 1000) for _ in range(size)]
                prior[open_[0]] += 1000
                spread = 5_000 if n % 5 == 0 else 400_000  # one in five is under the minimum
                base["search_target"] = {
                    "prior_ppm": prior,
                    "entries": [
                        {"candidate": open_[0], "q_ppm": 500_000 + spread // 2},
                        {"candidate": open_[1], "q_ppm": 500_000 - spread // 2},
                    ],
                }
                examples.append(base)
        text, manifest = stream_of(examples, request.policy_id)
        manifest["search_labels"] = {
            "search_identity_hash": self.identity_for(generation),
            "labelled": len(examples),
            "refused": [],
        }
        (out / "labels.jsonl").write_text(text, encoding="utf-8")
        layout.write_json(out / "label-set.json", manifest)


def distill_loop(run_dir: Path, config: RunConfig, hook: Any = None, **player: Any) -> Any:
    from tests.ml.loop.support import make_loop  # noqa: PLC0415

    return make_loop(run_dir, config, LabelPlayer(config, **player), hook)


@pytest.fixture(scope="module")
def clone_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return clone_output(tmp_path_factory.mktemp("clone"))


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory, clone_dir: Path) -> Path:
    run_dir = tmp_path_factory.mktemp("distill")
    import_clone(clone_dir, run_dir)
    distill_loop(run_dir, distilling_config(clone_ids(clone_dir))).run(RUN_GENERATIONS)
    return run_dir


def set_hash(run_dir: Path, generation: int) -> str:
    path = layout.generation_dir(run_dir, generation) / "label-set.json"
    return sha256_hex(jcs(layout.read_json(path)))


def test_the_report_carries_the_distillation_keys_from_the_first_label_generation(
    finished: Path,
) -> None:
    for generation in range(1, RUN_GENERATIONS + 1):
        report = layout.read_json(layout.generation_dir(finished, generation) / "train-report.json")
        battle = report["battle"]
        assert "search_labels" not in report["preview"]
        if generation < START:
            assert battle["search_labels"] == 0  # no label stream yet (and a value warm-up)
        else:
            assert battle["search_labels"] > 0
            assert battle["search_kl_before"] > 0
    first = layout.read_json(layout.generation_dir(finished, START) / "train-report.json")["battle"]
    last = layout.read_json(layout.generation_dir(finished, 3) / "train-report.json")["battle"]
    # aggregate_generations 2: generation 3 reads its own labels and generation 2's
    assert last["search_labels"] > first["search_labels"]


def test_the_records_list_the_label_sets_and_the_registry_maps_them_to_the_pair(
    finished: Path,
) -> None:
    registry = sources_registry(finished)
    for generation in range(START, RUN_GENERATIONS + 1):
        held = {set_hash(finished, g) for g in range(max(START, generation - 1), generation + 1)}
        manifest = layout.verify_generation(layout.generation_dir(finished, generation), generation)
        assert set(manifest["label_set_hashes"]) == held
        for role in ("preview", "battle"):
            record = layout.read_json(
                layout.generation_dir(finished, generation) / "records" / f"{role}.json"
            )
            assert set(record["training_set_hashes"]) == held | {manifest["training_set_hash"]}
            assert record["training_set_hashes"] == sorted(record["training_set_hashes"])
        previous = layout.verify_generation(
            layout.generation_dir(finished, generation - 1), generation - 1
        )["checkpoints"]
        assert registry[set_hash(finished, generation)] == sorted(previous.values())
    first = layout.verify_generation(layout.generation_dir(finished, 1), 1)
    assert "label_set_hashes" not in first


def test_the_provenance_export_accepts_the_label_sets(finished: Path, tmp_path: Path) -> None:
    export_provenance(finished, tmp_path / "prov")


def test_a_rerun_is_byte_identical(finished: Path, tmp_path: Path, clone_dir: Path) -> None:
    run_dir = tmp_path / "again"
    import_clone(clone_dir, run_dir)
    distill_loop(run_dir, distilling_config(clone_ids(clone_dir))).run(RUN_GENERATIONS)
    assert manifests(run_dir) == manifests(finished)
    for g in range(1, RUN_GENERATIONS + 1):
        for name in (f"learner-{g}.safetensors", f"learner-{g}.json", "train-report.json"):
            assert (layout.generation_dir(run_dir, g) / name).read_bytes() == (
                layout.generation_dir(finished, g) / name
            ).read_bytes()


@pytest.mark.parametrize(
    ("generation", "stage"), [(2, "played"), (2, "trained"), (3, "played"), (3, "written")]
)
def test_a_killed_run_resumes_to_identical_hashes(
    finished: Path, tmp_path: Path, clone_dir: Path, generation: int, stage: str
) -> None:
    config = distilling_config(clone_ids(clone_dir))
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)
    with pytest.raises(Killed):
        distill_loop(run_dir, config, killer(generation, stage)).run(RUN_GENERATIONS)
    distill_loop(run_dir, config).run(RUN_GENERATIONS)
    assert manifests(run_dir) == manifests(finished)
    for g in range(RUN_GENERATIONS + 1):
        assert (layout.generation_dir(run_dir, g) / f"learner-{g}.json").read_bytes() == (
            layout.generation_dir(finished, g) / f"learner-{g}.json"
        ).read_bytes()


def test_a_label_stream_of_another_identity_stops_the_generation(
    tmp_path: Path, clone_dir: Path
) -> None:
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)
    config = distilling_config(clone_ids(clone_dir))

    class Mixed(LabelPlayer):
        def play(self, request: PlayRequest) -> None:
            super().play(request)
            if request.generation == START + 1:  # the stream says one identity, the summary another
                summary = layout.read_json(request.out_dir / "generation.json")
                summary["search_labels"]["search_identity_hash"] = "e" * 64
                layout.write_json(request.out_dir / "generation.json", summary)

    from tests.ml.loop.support import make_loop  # noqa: PLC0415

    with pytest.raises(Exception, match="another search identity"):
        make_loop(run_dir, config, Mixed(config)).run(RUN_GENERATIONS)


def test_a_generation_with_no_label_files_has_no_labels(tmp_path: Path, clone_dir: Path) -> None:
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)
    config = distilling_config(clone_ids(clone_dir))
    distill_loop(run_dir, config, empty=True).run(RUN_GENERATIONS)
    last = layout.generation_dir(run_dir, RUN_GENERATIONS)
    assert not (last / "labels.jsonl").exists()
    report = layout.read_json(last / "train-report.json")
    assert report["battle"]["search_labels"] == 0


def test_a_missing_label_stream_or_summary_block_stops_the_generation(
    tmp_path: Path, clone_dir: Path
) -> None:
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)
    config = distilling_config(clone_ids(clone_dir))
    with pytest.raises(LoopError, match="label stream is missing"):
        distill_loop(run_dir, config, skip=True).run(RUN_GENERATIONS)
    other = tmp_path / "run2"
    import_clone(clone_dir, other)

    class NoBlock(FakePlayer):
        pass

    from tests.ml.loop.support import make_loop  # noqa: PLC0415

    with pytest.raises(LoopError, match="does not state the run's label search config"):
        make_loop(other, config, NoBlock(config=config)).run(RUN_GENERATIONS)


def test_a_run_without_the_block_is_unchanged(tmp_path: Path, clone_dir: Path) -> None:
    run_dir = tmp_path / "plain"
    import_clone(clone_dir, run_dir)
    config = warm_config(clone_ids(clone_dir), "tiny-plain")
    warm_loop(run_dir, config).run(2)
    for g in (1, 2):
        base = layout.generation_dir(run_dir, g)
        assert not (base / "label-set.json").exists()
        assert "label_set_hashes" not in layout.verify_generation(base, g)
        report = layout.read_json(base / "train-report.json")
        assert not any(k.startswith("search_") for r in report.values() for k in r)


def test_each_aggregated_stream_is_checked_against_its_own_generation(
    tmp_path: Path, clone_dir: Path
) -> None:
    """Generation 3 reads generation 2's labels under generation 2's own stated identity."""
    config = distilling_config(clone_ids(clone_dir))
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)

    class PerGeneration(LabelPlayer):
        def identity_for(self, generation: int) -> str:
            return ("e" if generation == START else "f") * 64

    from tests.ml.loop.support import make_loop  # noqa: PLC0415

    make_loop(run_dir, config, PerGeneration(config)).run(RUN_GENERATIONS)
    assert layout.read_json(layout.generation_dir(run_dir, 3) / "train-report.json")["battle"][
        "search_labels"
    ]


def test_a_tampered_older_label_stream_is_refused(tmp_path: Path, clone_dir: Path) -> None:
    config = distilling_config(clone_ids(clone_dir), aggregate=3)
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)
    distill_loop(run_dir, config).run(RUN_GENERATIONS)
    older = layout.generation_dir(run_dir, START) / "labels.jsonl"
    older.write_text(older.read_text() + "\n")
    with pytest.raises(layout.RunDirError, match="files differ from its manifest"):
        distill_loop(run_dir, config).run(RUN_GENERATIONS + 1)


def test_an_older_generation_stating_another_config_is_refused(
    tmp_path: Path, clone_dir: Path
) -> None:
    config = distilling_config(clone_ids(clone_dir), aggregate=3)
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir)
    distill_loop(run_dir, config).run(RUN_GENERATIONS)
    path = layout.generation_dir(run_dir, START)
    summary = layout.read_json(path / "generation.json")
    summary["search_labels"]["search_config_hash"] = "9" * 64
    layout.write_json(path / "generation.json", summary)
    reseal(run_dir, START)
    with pytest.raises(LoopError, match=r"generation 2: generation\.json does not state"):
        distill_loop(run_dir, config).run(RUN_GENERATIONS + 1)
