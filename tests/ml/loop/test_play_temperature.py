"""A tiny loop with ``generation.play.temperature_ppm`` (ADR-023 section 1, #1046): the frozen
pair plays under the tempered policy id, the learner refuses any other id, the report names the
temperature, an independent rerun is byte-identical and a kill anywhere resumes to the same
hashes; a run without the field keeps today's ids."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.learner.data import LearnerDataError, trajectories
from areneon_ml.loop import layout
from areneon_ml.loop.driver import LoopError
from areneon_ml.loop.play import PlayRequest
from areneon_ml.loop.records import policy_id
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import edited_config, make_stream, tiny_config
from tests.ml.loop.support import FakePlayer, Killed, killer, make_loop, manifests
from tests.ml.model.support import game

GENERATIONS = 3
TAU = 500_000


def tempered_config() -> Any:
    def edit(raw: dict[str, Any]) -> None:
        raw["generation"]["play"] = {"temperature_ppm": TAU}

    return edited_config(edit)


def loop(run_dir: Path, on_stage: Any = None, player: Any = None) -> Any:
    config = tempered_config()
    return make_loop(run_dir, config, player or FakePlayer(config=config), on_stage)


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, FakePlayer]:
    run_dir = tmp_path_factory.mktemp("tempered")
    config = tempered_config()
    player = FakePlayer(config=config)
    loop(run_dir, player=player).run(GENERATIONS)
    return run_dir, player


def test_the_pair_plays_under_the_tempered_policy_id(finished: tuple[Path, FakePlayer]) -> None:
    run_dir, player = finished
    assert len(player.requests) == GENERATIONS
    for request in player.requests:
        assert request.policy_id == policy_id(request.preview, request.battle, temperature_ppm=TAU)
        assert request.policy_id != policy_id(request.preview, request.battle)
    result = json.loads((layout.generation_dir(run_dir, 1) / "league-run.json").read_bytes())
    assert result["pair_policy_id"] == player.requests[0].policy_id


def test_a_run_without_the_field_keeps_the_untempered_id(tmp_path: Path) -> None:
    config = tiny_config()
    player = FakePlayer(config=config)
    make_loop(tmp_path, config, player).run(1)
    request = player.requests[0]
    assert request.policy_id == policy_id(request.preview, request.battle)
    report = json.loads((layout.generation_dir(tmp_path, 1) / "train-report.json").read_bytes())
    assert all("temperature_ppm" not in role for role in report.values())


def test_the_report_names_the_temperature_of_each_role(finished: tuple[Path, FakePlayer]) -> None:
    run_dir, _ = finished
    for generation in range(1, GENERATIONS + 1):
        report = json.loads(
            (layout.generation_dir(run_dir, generation) / "train-report.json").read_bytes()
        )
        assert {role: v["temperature_ppm"] for role, v in report.items()} == {
            "battle": TAU,
            "preview": TAU,
        }


class WrongId(FakePlayer):
    """Plays under another id than the one asked for."""

    def __init__(self, config: Any, other: Any) -> None:
        super().__init__(config=config)
        self.other = other

    def play(self, request: PlayRequest) -> None:
        super().play(dataclasses.replace(request, policy_id=self.other(request)))


def test_a_tempered_run_refuses_a_set_played_at_temperature_one(tmp_path: Path) -> None:
    config = tempered_config()
    plain = WrongId(config, lambda r: policy_id(r.preview, r.battle))
    with pytest.raises(LoopError, match="another generation or pair"):
        make_loop(tmp_path, config, plain).run(1)


def test_a_run_at_temperature_one_refuses_a_set_played_at_the_tempered_id(tmp_path: Path) -> None:
    config = tiny_config()
    tempered = WrongId(config, lambda r: policy_id(r.preview, r.battle, temperature_ppm=TAU))
    with pytest.raises(LoopError, match="another generation or pair"):
        make_loop(tmp_path, config, tempered).run(1)


def test_the_learner_side_loader_refuses_the_other_id() -> None:
    encoders = {role: Encoder(game(), tiny_config().model.encoder) for role in ROLES}
    pair = "ab" * 32
    plain, tempered = policy_id(pair, pair), policy_id(pair, pair, temperature_ppm=TAU)
    text, manifest = make_stream(policy=tempered)
    assert trajectories(text, manifest, encoders, tempered)
    with pytest.raises(LearnerDataError, match="not played by the policy being trained"):
        trajectories(text, manifest, encoders, plain)
    text, manifest = make_stream(policy=plain)
    with pytest.raises(LearnerDataError, match="not played by the policy being trained"):
        trajectories(text, manifest, encoders, tempered)


def test_an_independent_rerun_gives_identical_generations(
    finished: tuple[Path, FakePlayer], tmp_path: Path
) -> None:
    loop(tmp_path).run(GENERATIONS)
    assert manifests(tmp_path) == manifests(finished[0])


@pytest.mark.parametrize(
    ("generation", "stage"), [(0, "committed"), (1, "trained"), (2, "played"), (3, "written")]
)
def test_a_run_killed_anywhere_resumes_with_identical_hashes(
    finished: tuple[Path, FakePlayer], tmp_path: Path, generation: int, stage: str
) -> None:
    with pytest.raises(Killed):
        loop(tmp_path, killer(generation, stage)).run(GENERATIONS)
    loop(tmp_path).run(GENERATIONS)
    assert manifests(tmp_path) == manifests(finished[0])
    for g in range(GENERATIONS + 1):
        for name in (f"learner-{g}.safetensors", f"learner-{g}.json"):
            assert (layout.generation_dir(tmp_path, g) / name).read_bytes() == (
                layout.generation_dir(finished[0], g) / name
            ).read_bytes()
