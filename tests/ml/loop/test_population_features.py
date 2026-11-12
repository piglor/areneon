"""A population run with candidate features, a shared trunk and a Battle-only warm start (#1333).

The play side (Node) builds the rows; here the driver's part: every member's pair id carries the
run's feature hash (the play request, the past pool's snapshots), an opponent sheet is still
refused, and a shared-trunk member keeps its two roles' trunk tied through the warm start, the
value warm-up and the anchor, resumable and byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.checkpoint.files import load_checkpoint
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import layout
from areneon_ml.loop.driver import LoopError
from areneon_ml.loop.past_pool import read_file as read_pool
from areneon_ml.loop.play import PlayError, PopulationPlayRequest
from areneon_ml.loop.population import PopulationLoop
from areneon_ml.loop.records import policy_id
from areneon_ml.loop.warm_start import import_clone
from areneon_ml.model.network import check_shared_trunk

from tests.ml.learner.support import tiny_config
from tests.ml.loop.support import Killed, killer
from tests.ml.loop.test_past_pool import SETTINGS
from tests.ml.loop.test_population import (
    LABELS,
    FakePopulationPlayer,
    gen_path,
    make_population,
    population_config,
    tree_bytes,
)
from tests.ml.loop.test_warm_start import clone_ids, clone_output
from tests.ml.model.support import game

FEATURES = "c" * 64
OPTIONS = {"candidate_features": {"feature_set": "probe-damage/1", "feature_set_hash": FEATURES}}
SHARED = {"shared_trunk": True}
MODEL = {**tiny_config().model.as_dict(), "options": SHARED}


class Stops(FakePopulationPlayer):
    """Records the play request and stops: the fake streams hold no feature rows."""

    def play_population(self, request: PopulationPlayRequest) -> None:
        self.requests.append(request)
        raise PlayError("stopped after the request")


def featured(
    key: str = "pop-features", past: bool = False
) -> RunConfig:  # a population whose pairs read a feature set
    def edit(raw: dict[str, Any]) -> None:
        raw["model"]["encoder"]["options"] = OPTIONS
        if past:
            raw["population"]["shares"] = {
                "ladder_ppm": 500_000,
                "self_play_ppm": 250_000,
                "past_pool_ppm": 250_000,
            }
            raw["population"]["seats_per_member"] = 4
            bounds = raw["population"]["explore"]["bounds"]
            bounds["self_play_ppm"] = {"min": 200_000, "max": 300_000}
            bounds["past_pool_ppm"] = {"min": 200_000, "max": 300_000}
            raw["population"]["past_pool"] = {
                **SETTINGS,
                "snapshot_every": 1,
                "max_size": 16,
                "max_past_per_member": 2,
            }
            raw["population"]["selection"]["n_min"] = 2

    return population_config(key=key, edit=edit)


def test_every_members_pair_id_carries_the_feature_hash(tmp_path: Path) -> None:
    config = featured()
    player = Stops(config)
    loop = make_population(tmp_path / "run", config, player)
    with pytest.raises(PlayError, match="stopped"):
        loop.run(1)
    request = player.requests[0]
    for pair in request.members:
        assert pair.policy_id == policy_id(pair.preview, pair.battle, features=FEATURES)
        assert pair.policy_id != policy_id(pair.preview, pair.battle)
    assert len({pair.policy_id for pair in request.members}) == len(LABELS)
    for member in loop.frozen.members:
        assert loop.frozen.policy_id(member) in {pair.policy_id for pair in request.members}


def test_a_run_without_features_keeps_its_ids(tmp_path: Path) -> None:
    config = population_config(key="pop-plain")
    player = Stops(config)
    with pytest.raises(PlayError, match="stopped"):
        make_population(tmp_path / "run", config, player).run(1)
    for pair in player.requests[0].members:
        assert pair.policy_id == policy_id(pair.preview, pair.battle)


def test_a_past_pools_snapshots_carry_the_feature_hash(tmp_path: Path) -> None:
    config = featured(key="pop-features-pool", past=True)
    run_dir = tmp_path / "run"
    make_population(run_dir, config).run(0)
    pool = read_pool(layout.read_json(gen_path(run_dir, 0) / "past-pool.json"), config)
    assert pool.snapshots
    for snapshot in pool.snapshots:
        assert snapshot.policy_id == policy_id(snapshot.preview, snapshot.battle, features=FEATURES)


def test_an_opponent_sheet_is_still_refused(tmp_path: Path) -> None:
    def edit(raw: dict[str, Any]) -> None:
        raw["model"]["encoder"]["options"] = {
            "opponent_sheet": {"feature_set": "opponent-sheet/1", "feature_set_hash": "d" * 64}
        }

    config = population_config(key="pop-sheet", edit=edit)
    with pytest.raises(LoopError, match="opponent sheet is refused"):
        PopulationLoop(config, tmp_path / "run", game(), FakePopulationPlayer(config))


# -- shared_trunk with a Battle-only warm start, a value warm-up and the anchor ------------------

GENERATIONS = 3
WARMUP = 2


def shared_config(ids: dict[str, str]) -> RunConfig:
    def edit(raw: dict[str, Any]) -> None:
        raw["model"]["options"] = dict(SHARED)
        raw["warm_start"] = {"battle": ids["battle"]}
        raw["ppo"]["value_warmup"] = {"generations": WARMUP}
        raw["ppo"]["anchor"] = {"kl_micro": 100_000, "schedule": {"kind": "constant"}}

    return population_config(key="pop-shared-warm", edit=edit)


@pytest.fixture(scope="module")
def clone_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return clone_output(tmp_path_factory.mktemp("clone"), model=MODEL)


@pytest.fixture(scope="module")
def shared(
    tmp_path_factory: pytest.TempPathFactory, clone_dir: Path
) -> tuple[Path, RunConfig, PopulationLoop]:
    config = shared_config(clone_ids(clone_dir))
    run_dir = tmp_path_factory.mktemp("shared") / "run"
    import_clone(clone_dir, run_dir, ("battle",))
    loop = make_population(run_dir, config)
    loop.run(GENERATIONS)
    return run_dir, config, loop


def network(run_dir: Path, generation: int, label: str, role: str) -> Any:
    manifest = layout.read_json(gen_path(run_dir, generation) / "manifest.json")
    checkpoint = manifest["members"][label]["checkpoints"][role]
    data = (run_dir / "checkpoints" / f"{checkpoint}.safetensors").read_bytes()
    return load_checkpoint(data, game(), checkpoint, role)


def test_each_members_trunk_stays_tied_in_every_generation(
    shared: tuple[Path, RunConfig, PopulationLoop],
) -> None:
    run_dir, _, loop = shared
    for g in range(GENERATIONS + 1):
        for label in LABELS:
            pair = {role: network(run_dir, g, label, role) for role in ("preview", "battle")}
            check_shared_trunk(pair)  # a pair with unequal trunk tensors is refused
    # The live learners hold ONE set of trunk parameters per member, not two equal copies.
    for label in LABELS:
        networks = loop._learners[label].networks
        keys = networks["battle"].trunk_keys()
        battle, preview = networks["battle"].state_dict(), networks["preview"].state_dict()
        assert keys
        for key in keys:
            assert torch.equal(battle[key], preview[key])
            assert (
                dict(networks["battle"].named_parameters())[key]
                is dict(networks["preview"].named_parameters())[key]
            )


def test_the_members_trunks_are_their_own_copies_of_the_clone(
    shared: tuple[Path, RunConfig, PopulationLoop],
) -> None:
    run_dir, _, _ = shared
    zero = [network(run_dir, 0, label, "battle").state_dict() for label in LABELS]
    # Generation 0: every member's Battle network is the clone, and its Preview trunk is that trunk.
    for state in zero[1:]:
        assert all(torch.equal(zero[0][k], v) for k, v in state.items())
    # Later the members have trained on their own streams: their trunks differ.
    last = [network(run_dir, GENERATIONS, label, "battle").state_dict() for label in LABELS]
    keys = network(run_dir, 0, "m0", "battle").trunk_keys()
    assert any(not torch.equal(last[0][k], last[1][k]) for k in keys)


def test_the_warm_start_and_the_anchor_act_on_the_warm_role_only(
    shared: tuple[Path, RunConfig, PopulationLoop],
) -> None:
    _, _, loop = shared
    for label in LABELS:
        learner = loop._learners[label]
        assert learner.warm_roles == frozenset({"battle"})
        assert set(learner.anchors) == {"battle"}


def test_a_shared_trunk_population_resumes_to_the_same_bytes(
    tmp_path: Path, clone_dir: Path, shared: tuple[Path, RunConfig, PopulationLoop]
) -> None:
    run_dir, config, _ = shared
    again = tmp_path / "again"
    import_clone(clone_dir, again, ("battle",))
    with pytest.raises(Killed):
        make_population(again, config, on_stage=killer(2, "trained:m0")).run(GENERATIONS)
    make_population(again, config).run(GENERATIONS)
    assert tree_bytes(again) == tree_bytes(run_dir)
