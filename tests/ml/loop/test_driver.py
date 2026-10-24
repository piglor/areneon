"""The generation loop: freeze, play, load, train, freeze; commit; resume (ADR-010 §1)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.checkpoint.files import checkpoint_bytes, checkpoint_id
from areneon_ml.checkpoint.provenance import HELD_OUT, SEEN
from areneon_ml.checkpoint.record import read_record
from areneon_ml.learner import parallel
from areneon_ml.learner.data import LearnerDataError
from areneon_ml.learner.state import LearnerStateError
from areneon_ml.loop import layout, records
from areneon_ml.loop.driver import (
    LoopError,
    export_provenance,
    provenance_graph,
    sources_registry,
)
from areneon_ml.loop.play import PlayRequest
from areneon_ml.loop.records import policy_id, team_set_hash, team_set_manifest
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import edited_config, tiny_config
from tests.ml.loop.support import (
    FakePlayer,
    Killed,
    killer,
    make_loop,
    manifests,
    reseal,
)
from tests.ml.model.support import game


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A run of two generations, never interrupted."""
    run_dir = tmp_path_factory.mktemp("uninterrupted")
    make_loop(run_dir).run(2)
    return run_dir


def test_generation_zero_is_the_seeded_initialisation_and_each_generation_is_committed_whole(
    finished: Path,
) -> None:
    names = sorted(p.name for p in layout.generations_dir(finished).iterdir())
    assert names == ["g000000", "g000001", "g000002"]
    for g in range(3):
        path = layout.generation_dir(finished, g)
        assert layout.verify_generation(path, g)["generation"] == g
        assert (path / f"learner-{g}.safetensors").is_file()
        assert (path / f"learner-{g}.json").is_file()
        assert sorted(p.name for p in (path / "records").iterdir()) == [
            "battle.json",
            "preview.json",
        ]
    assert not (layout.generation_dir(finished, 0) / "stream.jsonl").exists()
    assert (layout.generation_dir(finished, 1) / "stream.jsonl").is_file()
    assert (layout.generation_dir(finished, 1) / "train-report.json").is_file()


def test_the_checkpoints_are_in_the_store_under_their_ids_and_records_name_them(
    finished: Path,
) -> None:
    for g in range(3):
        path = layout.generation_dir(finished, g)
        facts = json.loads((path / "manifest.json").read_text())
        for role in ROLES:
            record = read_record(layout.read_json(path / "records" / f"{role}.json"))
            stored = (finished / "checkpoints" / f"{record.checkpoint_id}.safetensors").read_bytes()
            assert checkpoint_id(stored) == record.checkpoint_id == facts["checkpoints"][role]
            assert record.role == role


def test_generation_zero_has_no_parent_and_no_training_set_and_later_ones_chain(
    finished: Path,
) -> None:
    config = tiny_config()
    records = {
        g: {
            role: read_record(
                layout.read_json(layout.generation_dir(finished, g) / "records" / f"{role}.json")
            )
            for role in ROLES
        }
        for g in range(3)
    }
    for role in ROLES:
        assert records[0][role].parent is None
        assert records[0][role].training_set_hashes == ()
        assert records[1][role].parent == records[0][role].checkpoint_id
        assert records[2][role].parent == records[1][role].checkpoint_id
        assert len(records[2][role].training_set_hashes) == 1
        assert records[1][role].training_set_hashes != records[2][role].training_set_hashes
        assert records[2][role].run_config_hash == config.hash
        assert records[2][role].lineage_id == config.lineage_id
        assert records[2][role].data_class == "synthetic"
        assert records[2][role].league_opponents == ()
    assert records[2]["battle"].learner_state_hash == records[2]["preview"].learner_state_hash
    assert records[1]["battle"].learner_state_hash != records[2]["battle"].learner_state_hash


def test_the_team_set_manifest_grows_by_the_teams_each_generation_drew(finished: Path) -> None:
    first = layout.read_json(layout.generation_dir(finished, 1) / "team-set.json")
    second = layout.read_json(layout.generation_dir(finished, 2) / "team-set.json")
    assert layout.read_json(layout.generation_dir(finished, 0) / "team-set.json")["teams"] == []
    assert len(first["teams"]) == 2
    assert set(first["teams"]) < set(second["teams"])
    assert second["teams"] == sorted(second["teams"])
    record = read_record(
        layout.read_json(layout.generation_dir(finished, 2) / "records" / "battle.json")
    )
    assert record.team_set_manifest_hash == team_set_hash(second)


def test_the_player_is_asked_to_play_the_frozen_pair_as_one_policy(tmp_path: Path) -> None:
    player = FakePlayer()
    loop = make_loop(tmp_path, player=player)
    loop.run(2)
    zero, one = (
        json.loads((layout.generation_dir(tmp_path, g) / "manifest.json").read_text())[
            "checkpoints"
        ]
        for g in (0, 1)
    )
    first, second = player.requests
    assert (first.generation, first.preview, first.battle) == (1, zero["preview"], zero["battle"])
    assert first.policy_id == policy_id(zero["preview"], zero["battle"])
    assert second.policy_id == policy_id(one["preview"], one["battle"])
    assert first.policy_id != second.policy_id
    assert first.out_dir == layout.partial_dir(tmp_path, 1)
    assert loop.frozen.policy_id == policy_id(
        loop.frozen.checkpoints["preview"], loop.frozen.checkpoints["battle"]
    )


def test_policy_id_is_the_hash_of_the_pair_and_the_inference_settings() -> None:
    expected = "policy:" + sha256_hex(
        jcs(
            {
                "preview": "a" * 64,
                "battle": "b" * 64,
                "mode": "sample",
                "temperature": 1,
                "precision": "float32",
            }
        )
    )
    assert policy_id("a" * 64, "b" * 64) == expected
    assert policy_id("b" * 64, "a" * 64) != expected


def test_the_team_set_manifest_is_sorted_unique_and_hashed() -> None:
    manifest = team_set_manifest(["b" * 64, "a" * 64, "b" * 64])
    assert manifest == {"schema_version": 1, "teams": ["a" * 64, "b" * 64]}
    assert team_set_hash(manifest) == sha256_hex(jcs(manifest))


# -- determinism and resume ---------------------------------------------------


def test_an_independent_run_gives_identical_generations(finished: Path, tmp_path: Path) -> None:
    make_loop(tmp_path).run(2)
    assert manifests(tmp_path) == manifests(finished)


def test_load_workers_change_no_hash(
    finished: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fake streams are small: cut them anyway, so the load phase really is parallel (#1069).
    monkeypatch.setattr(parallel, "MIN_PIECE_BYTES", 1)
    calls: list[int] = []
    real = parallel.parallel_trajectories

    def spy(*args: Any, **kwargs: Any) -> Any:
        found = real(*args, **kwargs)
        calls.append(0 if found is None else len(found))
        return found

    monkeypatch.setattr(parallel, "parallel_trajectories", spy)
    loop = make_loop(tmp_path)
    loop.load_processes = 3
    loop.run(2)
    assert calls
    assert all(calls)  # every generation's stream was read by the workers
    assert manifests(tmp_path) == manifests(finished)
    # The learner state is the same bytes, not only the same manifest.
    state = "learner-2.safetensors"
    assert (layout.generation_dir(tmp_path, 2) / state).read_bytes() == (
        layout.generation_dir(finished, 2) / state
    ).read_bytes()


@pytest.mark.parametrize("generation", [1, 2])
@pytest.mark.parametrize("stage", ["played", "trained", "written"])
def test_a_run_killed_in_a_generation_resumes_with_identical_hashes(
    finished: Path, tmp_path: Path, generation: int, stage: str
) -> None:
    with pytest.raises(Killed):
        make_loop(tmp_path, on_stage=killer(generation, stage)).run(2)
    assert layout.partial_dir(tmp_path, generation).is_dir()
    assert not layout.generation_dir(tmp_path, generation).exists()
    resumed = make_loop(tmp_path)
    result = resumed.run(2)
    assert result.resumed_from == generation - 1
    assert result.discarded == [layout.partial_dir(tmp_path, generation).name]
    assert [g.generation for g in result.generations] == list(range(generation, 3))
    assert manifests(tmp_path) == manifests(finished)
    for g in range(3):
        for name in (f"learner-{g}.safetensors", f"learner-{g}.json"):
            assert (layout.generation_dir(tmp_path, g) / name).read_bytes() == (
                layout.generation_dir(finished, g) / name
            ).read_bytes()


def test_a_run_killed_after_a_commit_resumes_from_it(finished: Path, tmp_path: Path) -> None:
    with pytest.raises(Killed):
        make_loop(tmp_path, on_stage=killer(1, "committed")).run(2)
    result = make_loop(tmp_path).run(2)
    assert (result.resumed_from, result.discarded) == (1, [])
    assert manifests(tmp_path) == manifests(finished)


def test_a_finished_run_is_a_no_op_and_a_longer_target_continues(
    finished: Path, tmp_path: Path
) -> None:
    shutil.copytree(finished, tmp_path / "copy")
    run_dir = tmp_path / "copy"
    loop = make_loop(run_dir)
    again = loop.run(2)
    assert (again.resumed_from, again.generations) == (2, [])
    assert again.as_dict(loop.frozen)["generation"] == 2
    more = make_loop(run_dir).run(3)
    assert [g.generation for g in more.generations] == [3]
    assert set(manifests(run_dir)) == {"g000000", "g000001", "g000002", "g000003"}
    assert manifests(run_dir)["g000002"] == manifests(finished)["g000002"]


def test_zero_generations_commits_only_the_initialisation(tmp_path: Path) -> None:
    result = make_loop(tmp_path).run(0)
    assert result.generations == []
    assert set(manifests(tmp_path)) == {"g000000"}
    with pytest.raises(LoopError, match="cannot be negative"):
        make_loop(tmp_path).run(-1)


def test_the_loop_must_have_run_before_it_has_a_frozen_pair(tmp_path: Path) -> None:
    with pytest.raises(LoopError, match="has not run"):
        _ = make_loop(tmp_path).frozen


# -- refusals -----------------------------------------------------------------


def test_another_run_config_is_refused(finished: Path, tmp_path: Path) -> None:
    shutil.copytree(finished, tmp_path / "copy")
    other = edited_config(lambda raw: raw["ppo"].update(epochs=3))
    with pytest.raises(LoopError, match="another TrainingRunConfig"):
        make_loop(tmp_path / "copy", config=other).run(2)


def test_a_committed_generation_whose_file_changed_is_refused(
    finished: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "copy"
    shutil.copytree(finished, run_dir)
    path = layout.generation_dir(run_dir, 2) / "learner-2.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        make_loop(run_dir).run(2)


def _copy(finished: Path, tmp_path: Path) -> Path:
    shutil.copytree(finished, tmp_path / "copy")
    return tmp_path / "copy"


def test_a_state_that_does_not_hash_to_its_manifest_is_refused(
    finished: Path, tmp_path: Path
) -> None:
    run_dir = _copy(finished, tmp_path)
    shutil.copy(
        layout.generation_dir(run_dir, 1) / "learner-1.safetensors",
        layout.generation_dir(run_dir, 2) / "learner-2.safetensors",
    )
    reseal(run_dir, 2)
    with pytest.raises(LearnerStateError, match="does not hash"):
        make_loop(run_dir).run(2)


def edit_manifest(run_dir: Path, generation: int, change: Any) -> None:
    path = layout.generation_dir(run_dir, generation) / layout.MANIFEST
    manifest = json.loads(path.read_text())
    change(manifest)
    layout.write_json(path, manifest)


def test_weights_that_are_not_the_checkpoints_named_are_refused(
    finished: Path, tmp_path: Path
) -> None:
    run_dir = _copy(finished, tmp_path)
    edit_manifest(
        run_dir,
        2,
        lambda m: m["checkpoints"].update(
            battle=json.loads((layout.generation_dir(run_dir, 1) / layout.MANIFEST).read_text())[
                "checkpoints"
            ]["battle"]
        ),
    )
    with pytest.raises(LoopError, match="battle weights differ from its checkpoint"):
        make_loop(run_dir).run(2)


def test_a_manifest_missing_a_fact_is_refused(finished: Path, tmp_path: Path) -> None:
    run_dir = _copy(finished, tmp_path)
    edit_manifest(run_dir, 2, lambda m: m.pop("learner_state_hash"))
    with pytest.raises(LoopError, match="its manifest lacks"):
        make_loop(run_dir).run(2)


def test_a_record_that_disagrees_with_the_state_is_refused(finished: Path, tmp_path: Path) -> None:
    run_dir = _copy(finished, tmp_path)
    path = layout.generation_dir(run_dir, 2) / "records" / "preview.json"
    record = layout.read_json(path)
    record["learner_state_hash"] = "0" * 64
    layout.write_json(path, record)
    reseal(run_dir, 2)
    with pytest.raises(LoopError, match="preview record disagrees"):
        make_loop(run_dir).run(2)


def test_a_record_of_the_wrong_role_is_refused(finished: Path, tmp_path: Path) -> None:
    run_dir = _copy(finished, tmp_path)
    base = layout.generation_dir(run_dir, 2) / "records"
    shutil.copy(base / "battle.json", base / "preview.json")
    reseal(run_dir, 2)
    with pytest.raises(LoopError, match="holds a battle record"):
        make_loop(run_dir).run(2)


def test_a_team_set_that_differs_from_its_hash_is_refused(finished: Path, tmp_path: Path) -> None:
    run_dir = _copy(finished, tmp_path)
    layout.write_json(
        layout.generation_dir(run_dir, 2) / "team-set.json", team_set_manifest(["f" * 64])
    )
    reseal(run_dir, 2)
    with pytest.raises(LoopError, match="team-set manifest differs"):
        make_loop(run_dir).run(2)


def test_a_state_of_another_generation_than_its_directory_is_refused(
    finished: Path, tmp_path: Path
) -> None:
    run_dir = _copy(finished, tmp_path)
    path = layout.generation_dir(run_dir, 2)
    shutil.copy(
        layout.generation_dir(run_dir, 1) / "learner-1.safetensors", path / "learner-2.safetensors"
    )
    shutil.copy(layout.generation_dir(run_dir, 1) / "learner-1.json", path / "learner-2.json")

    def point_at_generation_one(manifest: dict[str, Any]) -> None:
        one = json.loads((layout.generation_dir(run_dir, 1) / layout.MANIFEST).read_text())
        manifest["learner_state_hash"] = one["learner_state_hash"]

    edit_manifest(run_dir, 2, point_at_generation_one)
    reseal(run_dir, 2)
    with pytest.raises(LoopError, match="another generation than its directory"):
        make_loop(run_dir).run(2)


# -- what the simulator side reports ------------------------------------------


def test_a_summary_of_another_generation_is_refused(tmp_path: Path) -> None:
    player = FakePlayer(lambda summary, _set: summary.update(generation=7))
    with pytest.raises(LoopError, match="another generation than asked"):
        make_loop(tmp_path, player=player).run(1)


class SpoiledLeague(FakePlayer):
    """A simulator side whose ``name`` file (league-run.json or plan.json) ``change`` edits."""

    def __init__(self, name: str, change: Any) -> None:
        super().__init__()
        self.name = name
        self.edit = change

    def play(self, request: PlayRequest) -> None:
        super().play(request)
        path = request.out_dir / self.name
        data = layout.read_json(path)
        self.edit(data)
        layout.write_json(path, data)


def test_a_league_record_that_does_not_belong_to_the_run_is_refused(tmp_path: Path) -> None:
    player = SpoiledLeague("league-run.json", lambda r: r.update(run_id="other"))
    with pytest.raises(LoopError, match=r"league-run\.json: run_id other"):
        make_loop(tmp_path, player=player).run(1)


def test_a_plan_with_a_seed_the_run_key_does_not_give_is_refused(tmp_path: Path) -> None:
    def forge(plan: list[dict[str, Any]]) -> None:
        plan[0]["seed"] = "sodium," + "0" * 32

    player = SpoiledLeague("plan.json", forge)
    with pytest.raises(LoopError, match=r"league-run\.json: .*plan_hash"):
        make_loop(tmp_path, player=player).run(1)


def test_a_record_of_another_generation_or_pair_is_refused(tmp_path: Path) -> None:
    def other_pair(record: dict[str, Any]) -> None:
        record["pair_policy_id"] = "policy:" + "0" * 64
        record["opponents"] = [{"opponent_policy_id": record["pair_policy_id"], "games": 2}]

    with pytest.raises(LoopError, match="another generation or pair"):
        make_loop(tmp_path, player=SpoiledLeague("league-run.json", other_pair)).run(1)


def test_a_record_that_counts_other_forfeits_than_the_summary_is_refused(tmp_path: Path) -> None:
    def forfeit(record: dict[str, Any]) -> None:
        record.update(played=1, forfeits=1, episodes=2)

    with pytest.raises(LoopError, match="other forfeits"):
        make_loop(tmp_path, player=SpoiledLeague("league-run.json", forfeit)).run(1)


def test_a_record_that_counts_other_forfeit_causes_than_the_summary_is_refused(
    tmp_path: Path,
) -> None:
    def record_says_latency(record: dict[str, Any]) -> None:
        record.update(
            played=1,
            forfeits=1,
            episodes=2,
            forfeit_causes=[{"cause": "latency_overrun", "count": 1}],
        )

    player = SpoiledLeague("league-run.json", record_says_latency)
    player.change = lambda summary, _set: summary.update(forfeits=1, forfeit_causes={"crash": 1})
    with pytest.raises(LoopError, match="other forfeit causes"):
        make_loop(tmp_path, player=player).run(1)


def test_a_training_set_that_is_not_the_one_reported_is_refused(tmp_path: Path) -> None:
    player = FakePlayer(lambda summary, _set: summary.update(training_set_hash="0" * 64))
    with pytest.raises(LoopError, match="differs from the hash"):
        make_loop(tmp_path, player=player).run(1)


def test_a_set_played_by_another_policy_than_the_frozen_pair_is_refused(tmp_path: Path) -> None:
    def other_policy(summary: dict[str, Any], training_set: dict[str, Any]) -> None:
        training_set["behaviour_policy_id"] = "random-legal@tiny"
        summary["training_set_hash"] = sha256_hex(jcs(training_set))

    with pytest.raises(LearnerDataError, match="not played by the policy being trained"):
        make_loop(tmp_path, player=FakePlayer(other_policy)).run(1)


def test_a_failed_generation_leaves_no_committed_generation_to_resume_from(tmp_path: Path) -> None:
    def not_ready(summary: dict[str, Any], training_set: dict[str, Any]) -> None:
        del training_set["learner_ready"]
        summary["training_set_hash"] = sha256_hex(jcs(training_set))

    with pytest.raises(LearnerDataError, match="not learner_ready"):
        make_loop(tmp_path, player=FakePlayer(not_ready)).run(1)
    assert set(manifests(tmp_path)) == {"g000000"}
    assert layout.partial_dir(tmp_path, 1).is_dir()


# -- provenance ---------------------------------------------------------------


def test_the_records_form_a_provenance_graph_whose_sources_are_the_runs_own(finished: Path) -> None:
    graph = provenance_graph(finished)
    zero, one, two = (
        json.loads((layout.generation_dir(finished, g) / "manifest.json").read_text())
        for g in range(3)
    )
    registry = sources_registry(finished)
    assert registry == {
        one["training_set_hash"]: sorted(zero["checkpoints"].values()),
        two["training_set_hash"]: sorted(one["checkpoints"].values()),
    }
    # the second pair descends from the first and from the data the first pair generated
    assert graph.classify(two["checkpoints"]["battle"], one["checkpoints"]["battle"]).label == SEEN
    assert (
        graph.classify(two["checkpoints"]["battle"], zero["checkpoints"]["preview"]).label == SEEN
    )
    # an unrelated checkpoint of another run is held out only if the records say it is independent
    assert HELD_OUT != SEEN


def test_the_export_writes_records_and_the_registry_for_the_checkpoint_cli(
    finished: Path, tmp_path: Path
) -> None:
    out = tmp_path / "export"
    digest = export_provenance(finished, out)
    assert len(list((out / "records").iterdir())) == 6
    assert sha256_hex((out / "sources.json").read_bytes()) == digest
    assert (out / "sources.json").read_bytes() == jcs(sources_registry(finished))
    export_provenance(finished, out)  # replaces what it wrote before
    assert len(list((out / "records").iterdir())) == 6


def test_no_provenance_without_a_committed_generation(tmp_path: Path) -> None:
    assert sources_registry(tmp_path) == {}
    with pytest.raises(LoopError, match="no committed generation"):
        provenance_graph(tmp_path)


def test_the_stored_pair_is_the_learners_weights(finished: Path) -> None:
    loop = make_loop(finished)
    loop.run(2)
    learner = loop._learner  # the loop's own learner, restored from generation 2
    assert learner is not None
    for role in ROLES:
        assert (
            checkpoint_id(checkpoint_bytes(learner.networks[role], game()))
            == loop.frozen.checkpoints[role]
        )


# -- review fixes: records versus manifest, the store, the lock, stale files ----------------


def test_a_record_naming_another_checkpoint_than_the_manifest_is_refused(
    finished: Path, tmp_path: Path
) -> None:
    run_dir = _copy(finished, tmp_path)
    path = layout.generation_dir(run_dir, 2) / "records" / "battle.json"
    record = layout.read_json(path)
    record["checkpoint_id"] = "f" * 64
    layout.write_json(path, record)
    reseal(run_dir, 2)
    with pytest.raises(LoopError, match="battle record names another checkpoint"):
        make_loop(run_dir).run(2)


def test_a_record_whose_training_sets_differ_from_the_manifest_is_refused(
    finished: Path, tmp_path: Path
) -> None:
    run_dir = _copy(finished, tmp_path)
    path = layout.generation_dir(run_dir, 2) / "records" / "battle.json"
    record = layout.read_json(path)
    record["training_set_hashes"] = ["e" * 64]
    layout.write_json(path, record)
    reseal(run_dir, 2)
    with pytest.raises(LoopError, match="training sets differ from its manifest"):
        make_loop(run_dir).run(2)
    # and generation 0 has none
    run_dir = tmp_path / "zero"
    make_loop(run_dir).run(0)
    path = layout.generation_dir(run_dir, 0) / "records" / "preview.json"
    record = layout.read_json(path)
    record["training_set_hashes"] = ["e" * 64]
    layout.write_json(path, record)
    reseal(run_dir, 0)
    with pytest.raises(LoopError, match="training sets differ"):
        make_loop(run_dir).run(0)


def test_a_checkpoint_missing_from_the_store_is_written_again_on_resume(
    finished: Path, tmp_path: Path
) -> None:
    run_dir = _copy(finished, tmp_path)
    loop = make_loop(run_dir)
    loop.run(2)
    lost = run_dir / "checkpoints" / f"{loop.frozen.checkpoints['battle']}.safetensors"
    original = lost.read_bytes()
    lost.unlink()
    make_loop(run_dir).run(2)
    assert lost.read_bytes() == original


def test_a_resume_discards_the_temp_files_of_a_killed_checkpoint_write(
    finished: Path, tmp_path: Path
) -> None:
    run_dir = _copy(finished, tmp_path)
    stale = run_dir / "checkpoints" / f".{'c' * 64}.partial"
    stale.write_bytes(b"half")
    make_loop(run_dir).run(2)
    assert not stale.exists()


def test_the_run_config_is_bound_atomically(tmp_path: Path) -> None:
    make_loop(tmp_path).run(0)
    assert layout.read_json(tmp_path / layout.CONFIG) == tiny_config().raw
    assert not (tmp_path / f".{layout.CONFIG}.partial").exists()


def test_a_run_directory_held_by_another_driver_is_refused(tmp_path: Path) -> None:
    with layout.run_lock(tmp_path), pytest.raises(layout.RunDirError, match="another process"):
        # flock locks belong to the open file, so a second open in this process conflicts too
        make_loop(tmp_path).run(0)
    make_loop(tmp_path).run(0)  # released with the first holder


def test_the_pair_always_plays_by_sampling_and_the_policy_id_says_so() -> None:
    assert (records.MODE, records.TEMPERATURE, records.PRECISION) == ("sample", 1, "float32")
