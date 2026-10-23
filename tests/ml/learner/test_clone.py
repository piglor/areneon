"""The supervised cloning learner (ADR-019 track 1): loss, split, determinism, provenance."""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import math
import random
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import ContractError, jcs, sha256_hex
from areneon_ml.checkpoint.__main__ import main as checkpoint_main
from areneon_ml.checkpoint.files import checkpoint_bytes, checkpoint_id, load_checkpoint
from areneon_ml.checkpoint.provenance import ProvenanceError, ProvenanceGraph
from areneon_ml.checkpoint.record import CheckpointRecord, read_record
from areneon_ml.learner.__main__ import main as learner_main
from areneon_ml.learner.clone import (
    CloneError,
    clone,
    cross_entropy,
    evaluate,
    held_back,
    top1_hits,
)
from areneon_ml.learner.clone_config import CloneConfig, CloneConfigError, read_clone_config
from areneon_ml.learner.clone_run import CloneOutputError, write_clone
from areneon_ml.learner.data import trajectories
from areneon_ml.learner.trainer import configure_learner
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import episode_lines, stream_of
from tests.ml.model.support import SAMPLES, game, sample, table

CLONES = SAMPLES / "clone-run-config"
TEACHER = "damage-aware@tiny"
KEY = "ab" * 32
# Episode hashes whose keyed hash (key ab*32, computed with openssl) falls at these fractions of
# 2^64: 5 -> 0.047, 6 -> 0.023, e -> 0.19, 3 -> 0.39, 2 -> 0.68, 1 -> 0.77, 0 -> 0.80.
HELD = "5" * 64
VERSION = sample("checkpoint-record/valid/battle.json")["manifest"]


class First(random.Random):
    """A teacher that always plays the first candidate in canonical order."""

    def choice(self, seq: Any) -> Any:
        return seq[0]


def raw_config(name: str = "valid/tiny.json") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((CLONES / name).read_text())
    return loaded


def teacher_stream(
    hashes: tuple[str, ...] = ("0" * 64, HELD, "2" * 64), policy: str = TEACHER
) -> tuple[str, dict[str, Any]]:
    """A learner-ready stream of the first-candidate teacher, one Episode per hash."""
    examples: list[dict[str, Any]] = []
    for position, episode in enumerate(hashes):
        for example in episode_lines(position % 3, "p1" if position % 2 == 0 else "p2", First()):
            example["episode_hash"] = episode
            examples.append(example)
    return stream_of(examples, policy)


def config_for(
    manifest: dict[str, Any], *, epochs: int = 12, ppm: int = 100000, learning_rate: int = 3000
) -> CloneConfig:
    raw = raw_config()
    raw["teacher"] = {
        "behaviour_policy_id": TEACHER,
        "training_set_hash": sha256_hex(jcs(manifest)),
    }
    raw["clone"].update(epochs=epochs, learning_rate_micro=learning_rate)
    raw["clone"]["holdout"] = {"key": KEY, "ppm": ppm}
    return read_clone_config(raw)


@pytest.fixture(autouse=True)
def deterministic() -> None:
    configure_learner(read_clone_config(raw_config()))


def run_main(main: Any, argv: list[str]) -> tuple[int, str, str]:
    """Run a command's ``main``: its status, what it wrote to stdout's buffer and to stderr."""
    buffer, err = io.BytesIO(), io.StringIO()

    class Stdout:
        def __init__(self) -> None:
            self.buffer = buffer

    original = sys.stdout
    sys.stdout = Stdout()
    try:
        with redirect_stderr(err):
            status = int(main(argv))
    finally:
        sys.stdout = original
    return status, buffer.getvalue().decode(), err.getvalue()


# -- the config reader --------------------------------------------------------------------------


def test_the_tiny_sample_is_read_with_every_real_value_exact() -> None:
    config = read_clone_config(raw_config())
    assert (config.lineage_id, config.run_key, config.device, config.threads) == (
        "tiny-clone",
        "tiny-clone-run",
        "cpu",
        1,
    )
    assert (config.init_preview, config.init_battle, config.learner_seed) == (21, 22, 23)
    assert (config.teacher_policy_id, config.teacher_set_hash) == (TEACHER, "c" * 64)
    settings = config.clone
    assert (settings.epochs, settings.minibatch_size, settings.learning_rate) == (3, 16, 0.003)
    assert (settings.adam_beta1, settings.adam_beta2, settings.adam_eps) == (0.9, 0.999, 1e-8)
    assert settings.max_grad_norm == 1.0
    assert (settings.holdout_key, settings.holdout_ppm) == (bytes.fromhex("d" * 64), 200000)
    assert config.model.d_model == 16
    assert config.format_id == "gen9championsvgc2026regmc"


def test_the_hash_is_the_sha256_of_the_jcs_bytes_and_ignores_key_order() -> None:
    config = read_clone_config(raw_config())
    assert config.hash == sha256_hex(jcs(raw_config()))
    assert read_clone_config(dict(reversed(raw_config().items()))).hash == config.hash
    other = raw_config()
    other["clone"]["epochs"] += 1
    assert read_clone_config(other).hash != config.hash


def test_no_holdout_is_a_valid_record() -> None:
    assert read_clone_config(raw_config("valid/no-holdout.json")).clone.holdout_ppm == 0


@pytest.mark.parametrize("path", sorted(p.name for p in (CLONES / "invalid").glob("*.json")))
def test_a_record_the_contract_refuses_is_refused(path: str) -> None:
    with pytest.raises(CloneConfigError):
        read_clone_config(json.loads((CLONES / "invalid" / path).read_text()))


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("heads-do-not-divide-width.json", "n_heads"),
        ("picks-exceed-team.json", "max_picks"),
    ],
)
def test_the_reader_refuses_what_the_profile_cannot_say(path: str, message: str) -> None:
    with pytest.raises(CloneConfigError, match=message):
        read_clone_config(json.loads((CLONES / "reader-invalid" / path).read_text()))


def test_a_non_integer_and_an_unknown_key_are_refused() -> None:
    fractional = raw_config()
    fractional["clone"]["learning_rate_micro"] = 0.5
    unknown = raw_config()
    unknown["clone"]["kl_micro"] = 1
    for bad in (fractional, unknown):
        with pytest.raises(CloneConfigError):
            read_clone_config(bad)


# -- the loss, the agreement and the split, by hand ---------------------------------------------


def test_cross_entropy_is_the_negative_log_of_the_softmax_over_the_real_candidates() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 9.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    # row 0: candidates 0 and 1 only, chosen 1: -log(e^2 / (e^1 + e^2)) = log(1 + e^-1)
    # row 1: padding's huge logit takes nothing, chosen 0: -log(1/2) = log 2
    loss = cross_entropy(logits, mask, torch.tensor([1, 0]))
    assert math.log(1 + math.exp(-1)) == pytest.approx(0.31326168751822286)
    assert loss.item() == pytest.approx((0.31326168751822286 + math.log(2)) / 2, rel=1e-6)


def test_top1_hits_counts_the_largest_real_logit_and_the_first_on_a_tie() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [5.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, True], [False, True, True]])
    # real arg-maxes: row 0 -> 1 (the 3.0 is padding), row 1 -> 0, row 2 -> 1 (tie of 1 and 2)
    assert top1_hits(logits, mask, torch.tensor([1, 0, 1])) == 3
    assert top1_hits(logits, mask, torch.tensor([2, 0, 2])) == 1
    assert top1_hits(logits, mask, torch.tensor([0, 1, 0])) == 0


@pytest.mark.parametrize(
    ("episode", "held"),
    [
        # HMAC-SHA256(key, hash as 64 hex ASCII characters), first 8 bytes: computed with openssl
        ("5" * 64, True),  # 0x0c32f06caef1160b = 0.047 of 2^64
        ("6" * 64, True),  # 0x05b9e651bbdfc636 = 0.023
        ("e" * 64, False),  # 0x3106fd810ce6c202 = 0.19
        ("0" * 64, False),  # 0xcdb95f0a92eb44d5 = 0.80
    ],
)
def test_the_held_back_part_is_a_keyed_hash_of_the_episode_hash(episode: str, held: bool) -> None:
    assert held_back(bytes.fromhex(KEY), episode, 100000) is held
    assert held_back(bytes.fromhex(KEY), episode, 0) is False


def test_held_back_compares_in_integers_at_the_boundary() -> None:
    key = bytes.fromhex(KEY)
    n = 0x0C32F06CAEF1160B  # the first 8 bytes for "5" * 64
    ppm = (n * 1_000_000) // (1 << 64)  # the largest ppm at which n is still not below the bound
    assert held_back(key, "5" * 64, ppm) is False
    assert held_back(key, "5" * 64, ppm + 1) is True


# -- cloning ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stream() -> tuple[str, dict[str, Any]]:
    return teacher_stream()


def test_the_loss_falls_and_the_cloned_policy_agrees_with_the_teacher_above_chance(
    stream: tuple[str, dict[str, Any]],
) -> None:
    text, manifest = stream
    cloned = clone(config_for(manifest), game(), text, manifest)
    for role in ROLES:
        report = cloned.roles[role]
        assert report.epoch_loss[-1] < report.epoch_loss[0] / 2
        assert report.train.hits > 1.5 * report.train.chance * report.train.steps
        assert report.train.hits == report.train.steps  # the first-candidate rule is learnt
    battle = cloned.roles["battle"]
    assert battle.optimiser_steps == 12 * math.ceil(battle.train.steps / 16)


def test_agreement_equals_a_hand_count_and_the_split_follows_the_keyed_hash(
    stream: tuple[str, dict[str, Any]],
) -> None:
    text, manifest = stream
    config = config_for(manifest)
    cloned = clone(config, game(), text, manifest)
    encoders = {role: Encoder(game(), config.model.encoder) for role in ROLES}
    found = trajectories(text, manifest, encoders, TEACHER)
    for role in ROLES:
        mine = [t for t in found if t.role == role]
        train = [s for t in mine if t.episode != HELD for s in t.steps]
        held = [s for t in mine if t.episode == HELD for s in t.steps]
        report = cloned.roles[role]
        assert report.holdout is not None
        assert (report.train.steps, report.holdout.steps) == (len(train), len(held))
        assert held  # the keyed hash of HELD puts that Episode, and only it, in the held-back part
        # one decision at a time: the arg-max real candidate against the teacher's choice
        for steps, agreement in ((train, report.train), (held, report.holdout)):
            hits = 0
            cloned.networks[role].eval()
            with torch.no_grad():
                for step in steps:
                    logits = cloned.networks[role](collate([step.query])).logits[0]
                    hits += int(int(logits.argmax()) == step.chosen)
            assert agreement.hits == hits
            chance = sum(1 / s.query.pending.shape[0] for s in steps) / len(steps)
            assert agreement.chance == pytest.approx(chance)
        assert evaluate(cloned.networks[role], held, 16, torch.device("cpu")) == report.holdout


def test_no_holdout_reports_none_and_trains_on_everything(
    stream: tuple[str, dict[str, Any]],
) -> None:
    text, manifest = stream
    cloned = clone(config_for(manifest, ppm=0, epochs=1), game(), text, manifest)
    assert all(report.holdout is None for report in cloned.roles.values())
    assert cloned.roles["preview"].train.steps == 3


def test_two_runs_give_byte_identical_checkpoints_and_outputs(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    text, manifest = stream
    config = config_for(manifest, epochs=3)
    for name in ("a", "b"):
        write_clone(tmp_path / name, config, game(), clone(config, game(), text, manifest), VERSION)
    names = sorted(
        p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file()
    )
    assert names == sorted(
        p.relative_to(tmp_path / "b") for p in (tmp_path / "b").rglob("*") if p.is_file()
    )
    assert len(names) == 9  # 2 checkpoints, 2 records, sources, state (2), report, config
    for relative in names:
        assert (tmp_path / "a" / relative).read_bytes() == (tmp_path / "b" / relative).read_bytes()


def test_the_seeds_and_the_settings_decide_the_weights(
    stream: tuple[str, dict[str, Any]],
) -> None:
    text, manifest = stream
    base = config_for(manifest, epochs=2)

    def battle_id(edit: Any) -> str:
        raw = copy.deepcopy(base.raw)
        edit(raw)
        cloned = clone(read_clone_config(raw), game(), text, manifest)
        return checkpoint_id(checkpoint_bytes(cloned.networks["battle"], game()))

    same = battle_id(lambda raw: None)
    assert same == battle_id(lambda raw: None)
    assert same != battle_id(lambda raw: raw["seeds"].update(init_battle=99))
    assert same != battle_id(lambda raw: raw["seeds"].update(learner=99))
    assert same != battle_id(lambda raw: raw["clone"].update(learning_rate_micro=1000))


# -- refusals -----------------------------------------------------------------------------------


def test_a_set_that_is_not_the_named_synthetic_learner_ready_teacher_set_is_refused(
    stream: tuple[str, dict[str, Any]],
) -> None:
    text, manifest = stream
    config = config_for(manifest)

    def refused(changed: dict[str, Any], match: str) -> None:
        with pytest.raises(CloneError, match=match):
            clone(config, game(), text, changed)

    refused({k: v for k, v in manifest.items() if k != "learner_ready"}, "not learner_ready")
    refused({**manifest, "data_class": "private"}, "not synthetic")
    refused({**manifest, "behaviour_policy_id": "actor-rule@tiny"}, "not played by the teacher")
    refused({**manifest, "behaviour_policy_id": "policy:" + "ab" * 32}, "not played by the teacher")
    refused(
        {**manifest, "episode_hashes": sorted({*manifest["episode_hashes"], "9" * 64})},
        "teacher.training_set_hash",
    )
    raw = copy.deepcopy(config.raw)
    raw["teacher"]["behaviour_policy_id"] = "policy:" + "ab" * 32
    with pytest.raises(CloneConfigError):
        read_clone_config(raw)  # the contract has no learned pair as a teacher
    learned = dataclasses.replace(config, teacher_policy_id="policy:" + "ab" * 32)
    with pytest.raises(CloneError, match="not a teacher"):
        clone(learned, game(), text, manifest)


def test_a_stream_that_differs_from_its_manifest_is_refused(
    stream: tuple[str, dict[str, Any]],
) -> None:
    text, manifest = stream
    with pytest.raises(CloneError):
        clone(config_for(manifest), game(), text + text.splitlines()[0] + "\n", manifest)


def test_a_role_with_nothing_left_to_train_on_is_refused() -> None:
    text, manifest = teacher_stream(hashes=(HELD,))
    with pytest.raises(CloneError, match="after the held-back part"):
        clone(config_for(manifest), game(), text, manifest)


def test_the_stream_may_be_given_as_a_file(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    text, manifest = stream
    (tmp_path / "stream.jsonl").write_text(text)
    config = config_for(manifest, epochs=1)
    from_text = clone(config, game(), text, manifest)
    from_file = clone(config, game(), tmp_path / "stream.jsonl", manifest)
    assert checkpoint_bytes(from_text.networks["battle"], game()) == checkpoint_bytes(
        from_file.networks["battle"], game()
    )
    assert from_text.team_set == from_file.team_set


# -- outputs and provenance ---------------------------------------------------------------------


def written(
    tmp_path: Path, name: str, stream: tuple[str, dict[str, Any]], seed: int = 21
) -> dict[str, Any]:
    text, manifest = stream
    raw = copy.deepcopy(config_for(manifest, epochs=2).raw)
    raw["seeds"]["init_preview"] = seed
    raw["seeds"]["run_key"] = f"clone-{name}"
    config = read_clone_config(raw)
    cloned = clone(config, game(), text, manifest)
    return write_clone(tmp_path / name, config, game(), cloned, VERSION)


def records(directory: Path) -> list[CheckpointRecord]:
    paths = sorted((directory / "records").glob("*.json"))
    return [read_record(json.loads(p.read_text())) for p in paths]


def test_the_records_follow_the_clone_rules_and_the_checkpoints_load(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    report = written(tmp_path, "one", stream)
    out = tmp_path / "one"
    found = records(out)
    teacher_set = report["teacher_set_hash"]
    assert {r.role for r in found} == {"preview", "battle"}
    for record in found:
        assert record.parent is None
        assert record.league_opponents == ()
        assert record.exploiter_target is None
        assert record.training_set_hashes == (teacher_set,)
        assert record.run_config_hash == report["run_config_hash"]
        assert record.learner_state_hash == report["learner_state_hash"]
        assert record.data_class == "synthetic"
        assert record.lineage_id == "tiny-clone"
        assert record.manifest == VERSION
        assert record.checkpoint_id == report["checkpoints"][record.role]
        data = (out / "checkpoints" / f"{record.checkpoint_id}.safetensors").read_bytes()
        load_checkpoint(data, game(), record.checkpoint_id, record.role)
    assert (out / "sources.json").read_bytes() == jcs({teacher_set: []})
    assert set(json.loads((out / "clone-report.json").read_text())["roles"]) == {
        "preview",
        "battle",
    }
    assert json.loads((out / "clone-state.json").read_text())["kind"] == "clone"
    assert (
        json.loads((out / "clone-run-config.json").read_text())["seeds"]["run_key"] == "clone-one"
    )
    assert len({r.team_set_manifest_hash for r in found}) == 1


def test_the_checker_takes_the_teacher_set_as_an_outside_source(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    written(tmp_path, "one", stream)
    written(tmp_path, "two", teacher_stream(hashes=("3" * 64, HELD, "2" * 64)), seed=31)
    first, second = records(tmp_path / "one"), records(tmp_path / "two")
    pair_one = [r.checkpoint_id for r in first]
    pair_two = [r.checkpoint_id for r in second]
    registry = {
        **json.loads((tmp_path / "one/sources.json").read_text()),
        **json.loads((tmp_path / "two/sources.json").read_text()),
    }
    assert len(registry) == 2  # two different teacher sets, each with no generator
    assert all(generators == [] for generators in registry.values())
    graph = ProvenanceGraph([*first, *second], registry)
    assert graph.ancestry(pair_one).checkpoints == frozenset(pair_one)
    assert graph.ancestry(pair_one).sources == frozenset({first[0].training_set_hashes[0]})
    # a run cloned from another teacher run shares nothing with this one (C6) ...
    assert graph.independent(pair_one, pair_two).independent is True
    assert graph.classify(pair_one, pair_two).label == "held_out"
    # ... the same run is related to itself through the source, and the teacher is a baseline
    assert graph.independent(pair_one, pair_one).shared_sources
    assert graph.classify(pair_one, TEACHER).label == "seen"
    # a record whose set is missing from the registry is refused
    with pytest.raises(ProvenanceError, match="not in the sources registry"):
        ProvenanceGraph(first, {})


def test_the_checkpoint_cli_accepts_the_written_directory(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    written(tmp_path, "one", stream)
    ids = [r.checkpoint_id for r in records(tmp_path / "one")]
    status, printed, error = run_main(
        checkpoint_main,
        [
            "independent",
            "--records",
            str(tmp_path / "one/records"),
            "--sources",
            str(tmp_path / "one/sources.json"),
            "--a",
            ids[0],
            "--b",
            ids[1],
        ],
    )
    assert status == 0, error
    result = json.loads(printed)
    assert result["independent"] is False  # the two roles of one run share its source
    assert result["shared_sources"]


def test_an_existing_output_is_never_overwritten_and_a_bad_record_commits_nothing(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    text, manifest = stream
    config = config_for(manifest, epochs=1)
    cloned = clone(config, game(), text, manifest)
    (tmp_path / "taken").mkdir()
    with pytest.raises(CloneOutputError, match="never overwrites"):
        write_clone(tmp_path / "taken", config, game(), cloned, VERSION)
    with pytest.raises(ContractError):  # the contract refuses the record
        write_clone(tmp_path / "bad", config, game(), cloned, {**VERSION, "format_id": "No Good"})
    assert not (tmp_path / "bad").exists()


# -- the command line ---------------------------------------------------------------------------


def command_files(tmp_path: Path, stream: tuple[str, dict[str, Any]]) -> list[str]:
    text, manifest = stream
    directory = tmp_path / "set"
    directory.mkdir()
    (directory / "stream.jsonl").write_text(text)
    (directory / "training-set.json").write_text(json.dumps(manifest))
    (tmp_path / "config.json").write_text(json.dumps(config_for(manifest, epochs=1).raw))
    (tmp_path / "engine.json").write_text(json.dumps(table()))
    (tmp_path / "version.json").write_text(json.dumps(VERSION))
    return [
        "clone",
        "--config", str(tmp_path / "config.json"),
        "--stream-dir", str(directory),
        "--engine-data", str(tmp_path / "engine.json"),
        "--version-manifest", str(tmp_path / "version.json"),
        "--out", str(tmp_path / "out"),
    ]  # fmt: skip


def test_the_clone_command_writes_the_outputs_and_prints_the_report(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    argv = command_files(tmp_path, stream)
    status, printed, error = run_main(learner_main, argv)
    assert status == 0, error
    assert json.loads(printed) == json.loads((tmp_path / "out/clone-report.json").read_text())
    status, _, error = run_main(learner_main, argv)  # the output exists now
    assert status == 3
    assert "never overwrites" in error


def test_the_clone_command_refuses_a_bad_set_or_config(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    argv = command_files(tmp_path, stream)
    manifest_path = tmp_path / "set/training-set.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["behaviour_policy_id"] = "actor-rule@tiny"
    manifest_path.write_text(json.dumps(manifest))
    status, _, error = run_main(learner_main, argv)
    assert status == 3
    assert "not played by the teacher" in error
    assert not (tmp_path / "out").exists()
    (tmp_path / "config.json").write_text("{}")
    assert run_main(learner_main, argv)[0] == 3
