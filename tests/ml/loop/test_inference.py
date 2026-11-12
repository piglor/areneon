"""``generation.inference`` (ADR-028 section 2 F2, #1195): the batched run's policy id, its config
rules, the claims that hold for it (plan, seeds and ids on rerun, every committed generation
verifies, retraining from a committed stream reproduces the learner state, a killed generation
resumes to a valid one) and the ones that do not (byte-identical reruns)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import validate
from areneon_ml.learner.config import RunConfigError, read_run_config
from areneon_ml.learner.data import trajectories
from areneon_ml.learner.state import pack_state, restore_state
from areneon_ml.loop import layout
from areneon_ml.loop.driver import Frozen, Loop
from areneon_ml.loop.metrics import MetricsError, read_run_metrics
from areneon_ml.loop.play import _divergence_line
from areneon_ml.loop.records import policy_id

from tests.ml.learner.support import edited_config, tiny_config
from tests.ml.loop.support import FakePlayer, Killed, killer, make_loop

GENERATIONS = 2
BATCH = {"kind": "batched", "max_batch": 2}
CORPUS = Path(__file__).resolve().parents[3] / "fixtures/contracts/v1"


def batched_config(batch: int = 2) -> Any:
    def edit(raw: dict[str, Any]) -> None:
        raw["generation"]["inference"] = {"kind": "batched", "max_batch": batch}

    return edited_config(edit)


class NoisyPlayer(FakePlayer):
    """Plays like a batched run: a replayed generation is other bytes (the scheduling differs)."""

    def perturbation(self) -> int:
        return 7 * len(self.requests)


def test_policy_id_vectors_are_pinned_and_inference_enters_only_when_set() -> None:
    # Computed outside the code with Python hashlib over the JCS bytes; the TS test pins the same.
    a, b = "a" * 64, "b" * 64
    assert (
        policy_id(a, b, inference={"kind": "batched", "max_batch": 4})
        == "policy:c1cd82a47b9ef4532ae97297f0b913625474a39e410224d496ac9b2b007c1fb0"
    )
    assert (
        policy_id(a, b, "sample", temperature_ppm=500_000, inference=BATCH)
        == "policy:93259e1f4d24200d9a36b721420a844599b303fe28f205534ca388db4c15c25d"
    )
    assert policy_id(a, b, inference=None) == policy_id(a, b)
    assert policy_id(a, b, inference=BATCH) != policy_id(
        a, b, inference={"kind": "batched", "max_batch": 3}
    )


@pytest.mark.parametrize(
    "inference",
    [
        {"kind": "batched", "max_batch": 1},
        {"kind": "batched"},
        {"kind": "async", "max_batch": 2},
        {"kind": "batched", "max_batch": 2, "device": "cuda"},
        {"kind": "batched", "max_batch": "2"},
    ],
)
def test_policy_id_refuses_an_inference_setting_it_cannot_encode(inference: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="inference"):
        policy_id("a" * 64, "b" * 64, inference=inference)


def test_the_config_reader_takes_the_batching_and_applies_its_rules() -> None:
    assert batched_config(2).generation.inference_max_batch == 2
    assert tiny_config().generation.inference_max_batch is None
    with pytest.raises(RunConfigError, match=r"cannot exceed generation\.n_parallel"):
        batched_config(3)


def test_a_batched_run_cannot_be_a_population() -> None:
    raw = json.loads((CORPUS / "training-run-config/valid/population.json").read_text())
    raw["generation"]["inference"] = {"kind": "batched", "max_batch": 2}
    with pytest.raises(RunConfigError, match="cannot be combined with a population"):
        read_run_config(raw)


def test_the_batching_enters_the_run_config_hash() -> None:
    assert batched_config().hash != tiny_config().hash


def test_the_pair_plays_under_the_batched_policy_id(tmp_path: Path) -> None:
    config = batched_config()
    player = FakePlayer(config=config)
    make_loop(tmp_path, config, player).run(GENERATIONS)
    assert len(player.requests) == GENERATIONS
    for request in player.requests:
        assert request.policy_id == policy_id(request.preview, request.battle, inference=BATCH)
        assert request.policy_id != policy_id(request.preview, request.battle)
    result = json.loads((layout.generation_dir(tmp_path, 1) / "league-run.json").read_bytes())
    assert result["pair_policy_id"] == player.requests[0].policy_id


def test_a_run_without_the_field_keeps_today_ids(tmp_path: Path) -> None:
    config = tiny_config()
    player = FakePlayer(config=config)
    make_loop(tmp_path, config, player).run(1)
    request = player.requests[0]
    assert request.policy_id == policy_id(request.preview, request.battle)


def test_frozen_policy_id_names_the_batching() -> None:
    pair = {"preview": "a" * 64, "battle": "b" * 64}
    plain = Frozen(0, pair, "c" * 64, None, ())
    batched = Frozen(0, pair, "c" * 64, None, (), None, None, 2)
    assert plain.policy_id == policy_id(pair["preview"], pair["battle"])
    assert batched.policy_id == policy_id(pair["preview"], pair["battle"], inference=BATCH)


def test_a_learner_refuses_a_stream_played_under_the_unbatched_id(tmp_path: Path) -> None:
    from areneon_ml.loop.driver import LoopError  # noqa: PLC0415

    config = batched_config()
    player = FakePlayer(config=config)

    class Unbatched(FakePlayer):
        def play(self, request: Any) -> None:
            import dataclasses  # noqa: PLC0415

            super().play(
                dataclasses.replace(request, policy_id=policy_id(request.preview, request.battle))
            )

    del player
    with pytest.raises(LoopError, match="another generation or pair"):
        make_loop(tmp_path, config, Unbatched(config=config)).run(1)


def noisy_run(run_dir: Path, on_stage: Any = None) -> Loop:
    config = batched_config()
    return make_loop(run_dir, config, NoisyPlayer(config=config), on_stage)


def test_a_rerun_keeps_plan_seeds_and_ids_but_not_the_bytes(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    noisy_run(first).run(GENERATIONS)
    # A second run whose games come out differently: byte-identical reruns do not hold.
    config = batched_config()
    make_loop(second, config, FakePlayer(config=config)).run(GENERATIONS)
    for g in range(1, GENERATIONS + 1):
        one, two = layout.generation_dir(first, g), layout.generation_dir(second, g)
        assert (one / "plan.json").read_bytes() == (two / "plan.json").read_bytes()
    # Generation 1 is played by the seeded pair of generation 0, so even its league record agrees.
    one, two = layout.generation_dir(first, 1), layout.generation_dir(second, 1)
    assert (one / "league-run.json").read_bytes() == (two / "league-run.json").read_bytes()
    assert (layout.generation_dir(first, 1) / "stream.jsonl").read_bytes() != (
        layout.generation_dir(second, 1) / "stream.jsonl"
    ).read_bytes()


def test_every_committed_generation_verifies(tmp_path: Path) -> None:
    noisy_run(tmp_path).run(GENERATIONS)
    for g in range(GENERATIONS + 1):
        layout.verify_generation(layout.generation_dir(tmp_path, g), g)


def test_retraining_from_a_committed_stream_reproduces_the_learner_state(tmp_path: Path) -> None:
    loop = noisy_run(tmp_path)
    loop.run(GENERATIONS)
    before, after = (layout.generation_dir(tmp_path, g) for g in (1, 2))
    manifest = json.loads((after / layout.MANIFEST).read_text())
    learner = loop._new_learner()
    stored = json.loads((before / layout.MANIFEST).read_text())["learner_state_hash"]
    restore_state(
        learner,
        (before / "learner-1.safetensors").read_bytes(),
        (before / "learner-1.json").read_bytes(),
        stored,
    )
    pair = json.loads((after / "league-run.json").read_text())["pair_policy_id"]
    found = trajectories(
        after / "stream.jsonl",
        json.loads((after / "training-set.json").read_text()),
        loop._encoders,
        pair,
    )
    learner.train(found, None)
    assert pack_state(learner).hash == manifest["learner_state_hash"]


@pytest.mark.parametrize(
    ("generation", "stage"), [(0, "committed"), (1, "trained"), (2, "played"), (2, "written")]
)
def test_a_killed_generation_resumes_to_a_valid_generation(
    tmp_path: Path, generation: int, stage: str
) -> None:
    with pytest.raises(Killed):
        noisy_run(tmp_path, killer(generation, stage)).run(GENERATIONS)
    noisy_run(tmp_path).run(GENERATIONS)
    for g in range(GENERATIONS + 1):
        path = layout.generation_dir(tmp_path, g)
        layout.verify_generation(path, g)
        manifest = json.loads((path / layout.MANIFEST).read_text())
        learner = noisy_run(tmp_path)._new_learner()
        restore_state(
            learner,
            (path / f"learner-{g}.safetensors").read_bytes(),
            (path / f"learner-{g}.json").read_bytes(),
            manifest["learner_state_hash"],
        )
        assert learner.generation == g


def divergence(**changes: Any) -> dict[str, Any]:
    return {
        "max_batch": 2,
        "decisions": 2000,
        "sampled": 256,
        "differing": 3,
        "max_prior_diff_ppm": 41,
        "target_met": True,
        **changes,
    }


def metrics_with(found: dict[str, Any]) -> dict[str, Any]:
    config = batched_config()
    record: dict[str, Any] = json.loads((CORPUS / "run-metrics/valid/cpu.json").read_text())
    record["run_config_hash"] = config.hash
    record["generation"] = 1
    record["threads"] = config.threads
    record["inference_divergence"] = found
    return record


def test_the_divergence_record_is_in_the_contract_and_ties_to_the_config() -> None:
    config = batched_config()
    validate("run-metrics", metrics_with(divergence()))
    assert read_run_metrics(metrics_with(divergence()), config, 1)["inference_divergence"]
    with pytest.raises(MetricsError, match="config's batching"):
        read_run_metrics(metrics_with(divergence(max_batch=3)), config, 1)
    with pytest.raises(MetricsError, match="differing <= sampled <= decisions"):
        read_run_metrics(metrics_with(divergence(differing=300)), config, 1)
    with pytest.raises(MetricsError, match="differing <= sampled <= decisions"):
        read_run_metrics(metrics_with(divergence(sampled=3000)), config, 1)


def test_the_play_script_line_is_read_and_anything_else_ignored() -> None:
    found = divergence()
    out = "\n".join(
        [
            "not json",
            json.dumps({"generation": 1}),
            json.dumps({"inference_divergence": found}),
        ]
    )
    assert _divergence_line(out) == found
    assert _divergence_line("") == {}
    assert _divergence_line(json.dumps({"inference_divergence": 3})) == {}
