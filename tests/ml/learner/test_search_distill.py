"""Search distillation in the learner (ADR-026 erratum E2 decisions 5 and 8, ADR-010 erratum A10,
#1291): the target and the KL against values worked out by hand, the label reader, the Battle-only
term, the report keys and an absent block that changes nothing."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.learner.config import (
    VALUE_FREE_FEATURE_SETS,
    RunConfig,
    RunConfigError,
    SearchDistillSettings,
    read_run_config,
)
from areneon_ml.learner.data import LearnerDataError
from areneon_ml.learner.labels import LabelReader, LabelStep, distill_target
from areneon_ml.learner.ppo import distill_kl
from areneon_ml.learner.state import pack_state
from areneon_ml.learner.trainer import LearnerError, to_device
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import (
    POLICY,
    edited_config,
    episode_lines,
    learner,
    stream_of,
    tiny_config,
)
from tests.ml.learner.test_config import CONFIGS, raw
from tests.ml.learner.test_trainer import trajs
from tests.ml.model.support import game

REPO = Path(__file__).resolve().parents[3]
IDENTITY = "c" * 64
SETTINGS = SearchDistillSettings(1_000_000, 4_000_000, 20_000, 4)


def distilled(**distill: Any) -> RunConfig:
    def edit(config: dict[str, Any]) -> None:
        config["warm_start"] = {"preview": "a" * 64, "battle": "b" * 64}
        config["team_dataset"] = {"path": "x", "dataset_hash": "5" * 64}
        config["ppo"]["value_warmup"] = {"generations": 1}
        config["ppo"]["search_distill"] = {
            "coef_micro": 1_000_000,
            "c_scale_micro": 4_000_000,
            "min_spread_ppm": 20_000,
            "aggregate_generations": 2,
            **distill,
        }
        config["generation"]["search_labels"] = {
            "search_config_hash": "d" * 64,
            "label_share_ppm": 40_000,
            "start_generation": 2,
        }

    return edited_config(edit)


# -- the config ---------------------------------------------------------------


def test_the_reader_reads_both_blocks_and_absent_is_none() -> None:
    config = read_run_config(raw("valid/search-distill.json"))
    assert config.ppo.search_distill == SearchDistillSettings(1_000_000, 4_000_000, 20_000, 4)
    assert config.generation.search_config_hash == "d" * 64
    assert config.generation.search_start_generation == 3
    plain = tiny_config()
    assert plain.ppo.search_distill is None
    assert plain.generation.search_config_hash is None
    assert plain.hash != config.hash


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("labels-only", "come together"),
        ("config-only", "come together"),
        ("no-warm-start", "needs warm_start"),
        ("no-value-warmup", r"needs ppo\.value_warmup"),
        ("start-in-warmup", "start_generation must be at least"),
        ("opponent-sheet", "opponent_sheet encoder option"),
        ("features-2", "its value reads candidate rows"),
        ("no-team-dataset", "needs a team_dataset"),
    ],
)
def test_the_reader_rules(name: str, message: str) -> None:
    with pytest.raises(RunConfigError, match=message):
        read_run_config(raw(f"reader-invalid/search-distill-{name}.json"))


@pytest.mark.parametrize("name", ["zero-share", "zero-aggregate", "extra-field", "short-hash"])
def test_the_schema_refuses_the_rest(name: str) -> None:
    with pytest.raises(RunConfigError):
        read_run_config(raw(f"invalid/search-distill-{name}.json"))
    assert (CONFIGS / "invalid" / f"search-distill-{name}.json").exists()


def test_the_allowed_feature_sets_are_the_ts_readers() -> None:
    source = (REPO / "packages/showdown-bridge/src/training-run-config.ts").read_text()
    found = re.search(r"const VALUE_FREE_FEATURE_SETS[^=]*= (\[[^\]]*\])", source)
    assert found is not None
    assert tuple(json.loads(found.group(1))) == VALUE_FREE_FEATURE_SETS


def test_the_value_free_feature_sets_are_allowed() -> None:
    for feature_set in VALUE_FREE_FEATURE_SETS:
        document = raw("valid/search-distill.json")
        document["model"]["encoder"]["options"] = {
            "candidate_features": {"feature_set": feature_set, "feature_set_hash": "c" * 64}
        }
        assert read_run_config(document).ppo.search_distill is not None


# -- the target, by hand ------------------------------------------------------

# Three candidates: 0 and 1 searched (q 0.8 and 0.2), 2 pending, never searched, zero prior.
PRIOR = [600_000, 400_000, 0]
ENTRIES = [(0, 800_000), (1, 200_000)]


def probabilities(logs: tuple[float, ...]) -> list[float]:
    return [math.exp(v) for v in logs]


def test_the_target_by_hand() -> None:
    # q_hat(2) = (0.6 * 0.8 + 0.4 * 0.2) / (0.6 + 0.4) = 0.56, so x = (0.56 - 0.2) / 0.6 = 0.6;
    # the zero prior is floored at 1 ppm. Weights: 0.6 e^4, 0.4 e^0, 1e-6 e^2.4.
    found = distill_target(PRIOR, ENTRIES, None, 4_000_000, 20_000)
    assert found is not None
    weights = [0.6 * math.exp(4), 0.4, 1e-6 * math.exp(2.4)]
    expected = [w / sum(weights) for w in weights]
    assert probabilities(found) == pytest.approx(expected, rel=1e-12)
    assert probabilities(found) == pytest.approx(
        [0.9879365416, 0.0120631260, 3.32434913e-07], rel=1e-8, abs=1e-12
    )
    assert sum(probabilities(found)) == pytest.approx(1.0, rel=1e-12)


def test_the_target_at_a_play_temperature() -> None:
    # tau = 0.5 doubles ln(prior): weights 0.36 e^4, 0.16, 1e-12 e^2.4.
    found = distill_target(PRIOR, ENTRIES, 500_000, 4_000_000, 20_000)
    assert found is not None
    assert probabilities(found) == pytest.approx(
        [0.9919254452, 0.0080745548, 5.5629526e-13], rel=1e-8, abs=1e-12
    )


def test_a_label_below_the_minimum_spread_has_no_target() -> None:
    assert distill_target(PRIOR, [(0, 510_000), (1, 500_000)], None, 4_000_000, 20_000) is None
    assert distill_target(PRIOR, [(0, 520_000), (1, 500_000)], None, 4_000_000, 20_000) is not None
    # no spread at all is never a target, even at a minimum of 0 (no division by zero)
    assert distill_target(PRIOR, [(0, 5), (1, 5)], None, 4_000_000, 0) is None


def test_an_unweighted_searched_set_is_refused() -> None:
    with pytest.raises(LearnerDataError, match="no prior mass"):
        distill_target([0, 0, 1_000_000], ENTRIES, None, 4_000_000, 0)


def test_the_kl_by_hand() -> None:
    target = distill_target(PRIOR, ENTRIES, None, 4_000_000, 20_000)
    assert target is not None
    p = probabilities(target)
    # a uniform policy over the three candidates, and one padded place that must not count
    logits = torch.zeros(1, 4)
    mask = torch.tensor([[True, True, True, False]])
    log_target = torch.tensor([[*target, 0.0]])
    expected = sum(x * math.log(3 * x) for x in p)
    assert float(distill_kl(logits, mask, log_target)[0]) == pytest.approx(expected, rel=1e-6)
    # at the target itself the KL is 0
    assert float(distill_kl(log_target.clone(), mask, log_target)[0]) == pytest.approx(0, abs=1e-6)


def test_the_kl_is_taken_at_the_play_temperature() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    target = torch.log(torch.tensor([[0.5, 0.25, 0.25]]))
    tempered = torch.softmax(logits * 2.0, dim=-1)[0]
    expected = sum(
        float(t) * math.log(float(t) / float(q))
        for t, q in zip((0.5, 0.25, 0.25), tempered, strict=False)
    )
    assert float(distill_kl(logits, mask, target, 500_000)[0]) == pytest.approx(expected, rel=1e-5)


# -- the label reader ---------------------------------------------------------


def label_example(
    base: dict[str, Any], rng: random.Random, spread: int = 400_000
) -> dict[str, Any]:
    n = len(base["candidates"])
    open_ = [i for i, c in enumerate(base["candidates"]) if not c["pending"]]
    first, second = open_[0], open_[1]
    example: dict[str, Any] = json.loads(json.dumps(base))
    example["search_target"] = {
        "prior_ppm": [rng.randrange(0, 1000) for _ in range(n)],
        "entries": [
            {"candidate": first, "q_ppm": 500_000 + spread // 2},
            {"candidate": second, "q_ppm": 500_000 - spread // 2},
        ],
    }
    example["search_target"]["prior_ppm"][first] += 1000  # the searched mass is positive
    return example


def label_stream(
    policy: str = POLICY, **change: Any
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(3)
    battles = [
        e
        for e in episode_lines(0, "p1", random.Random(1))
        if e["observation"]["kind"] == "battle"
        and sum(1 for c in e["candidates"] if not c["pending"]) >= 2
    ][:4]
    examples = [label_example(b, rng, 5_000 if i == 3 else 400_000) for i, b in enumerate(battles)]
    text, manifest = stream_of(examples, policy)
    manifest["search_labels"] = {
        "search_identity_hash": IDENTITY,
        "labelled": len(examples),
        "refused": [],
    }
    manifest.update(change)
    return text, manifest, examples


def reader(identity: str = IDENTITY) -> LabelReader:
    config = distilled()
    assert config.ppo.search_distill is not None
    return LabelReader(
        Encoder(game(), config.model.encoder), config.ppo.search_distill, None, identity
    )


def test_a_label_stream_is_read_into_targets(tmp_path: Path) -> None:
    text, manifest, examples = label_stream()
    path = tmp_path / "labels.jsonl"
    path.write_text(text)
    found = reader().read(path, manifest, POLICY)
    assert len(found.steps) == 3  # the fourth has a 5,000 ppm spread, under the 20,000 minimum
    assert found.below_spread == 1
    for step in found.steps:
        assert int(step.query.pending.shape[0]) == len(step.log_target)
        assert sum(probabilities(step.log_target)) == pytest.approx(1.0, rel=1e-9)
    assert examples


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"search_identity_hash": "e" * 64}, "another search identity"),
    ],
)
def test_a_label_stream_of_another_identity_is_refused(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    text, manifest, _ = label_stream()
    manifest["search_labels"] = {**manifest["search_labels"], **change}
    path = tmp_path / "labels.jsonl"
    path.write_text(text)
    with pytest.raises(LearnerDataError, match=message):
        reader().read(path, manifest, POLICY)


def test_a_label_stream_is_refused_when_not_what_it_claims(tmp_path: Path) -> None:
    text, manifest, _ = label_stream()
    path = tmp_path / "labels.jsonl"
    path.write_text(text)
    without = {k: v for k, v in manifest.items() if k != "search_labels"}
    with pytest.raises(LearnerDataError, match="needs a TrainingSet with search_labels"):
        reader().read(path, without, POLICY)
    with pytest.raises(LearnerDataError, match="not played by the policy being trained"):
        reader().read(path, manifest, "policy:" + "cd" * 32)
    with pytest.raises(LearnerDataError, match="not learner_ready"):
        reader().read(path, {**manifest, "learner_ready": False}, POLICY)
    with pytest.raises(LearnerDataError):  # the digest no longer matches
        reader().read(path, {**manifest, "stream_sha256": "0" * 64}, POLICY)


def test_a_decision_labelled_twice_is_refused(tmp_path: Path) -> None:
    text, manifest, _ = label_stream()
    path = tmp_path / "labels.jsonl"
    path.write_text(text)
    again = reader()
    again.read(path, manifest, POLICY)
    with pytest.raises(LearnerDataError, match="a decision twice"):
        again.read(path, manifest, POLICY)


def test_a_preview_label_is_refused(tmp_path: Path) -> None:
    rng = random.Random(3)
    preview = next(
        e for e in episode_lines(0, "p1", random.Random(1)) if e["observation"]["kind"] == "preview"
    )
    example = label_example(preview, rng)
    text, manifest = stream_of([example], POLICY)
    manifest["search_labels"] = {"search_identity_hash": IDENTITY, "labelled": 1, "refused": []}
    path = tmp_path / "labels.jsonl"
    path.write_text(text)
    with pytest.raises(LearnerDataError, match="not a Battle decision"):
        reader().read(path, manifest, POLICY)


# -- the term in the update ---------------------------------------------------


def battle_labels(count: int = 6) -> list[LabelStep]:
    rng = random.Random(7)
    out: list[LabelStep] = []
    for trajectory in trajs(tiny_config()):
        if trajectory.role != "battle":
            continue
        for step in trajectory.steps:
            n = int(step.query.pending.shape[0])
            open_ = [i for i in range(n) if float(step.query.pending[i]) < 0.5]
            if len(open_) < 2 or len(out) == count:
                continue
            prior = [rng.randrange(1, 1000) for _ in range(n)]
            target = distill_target(
                prior, [(open_[0], 900_000), (open_[1], 100_000)], None, 4_000_000, 0
            )
            assert target is not None
            out.append(LabelStep(step.query, target, "e", len(out)))
    assert len(out) == count
    return out


def trained(config: RunConfig, labels: list[LabelStep] | None) -> tuple[Any, Any]:
    student = learner(config)
    student.generation = 1  # past the one-update value warm-up
    return student, student.train(trajs(config), labels)


def test_a_distilling_update_reports_the_battle_role_only_and_the_kl_falls() -> None:
    config = distilled(coef_micro=20_000_000)
    _, report = trained(config, battle_labels())
    battle = report.as_dict()["battle"]
    assert battle["search_labels"] == 6
    assert battle["search_kl_after"] < battle["search_kl_before"]
    assert not {"search_labels", "search_kl_before", "search_kl_after"} & set(
        report.as_dict()["preview"]
    )


def test_no_labels_yet_reports_zero_and_trains_like_ppo() -> None:
    config = distilled()
    student, report = trained(config, [])
    assert report.as_dict()["battle"]["search_labels"] == 0
    assert report.as_dict()["battle"]["search_kl_before"] == 0.0
    plain = edited_config(lambda c: c["ppo"].update({}))
    # the same weights as a run without the block, from the same seed and the same data
    other = learner(plain)
    other.generation = 1
    other.train(trajs(plain))
    for role in ROLES:
        for (n, a), (_, b) in zip(
            student.networks[role].named_parameters(),
            other.networks[role].named_parameters(),
            strict=False,
        ):
            assert torch.equal(a, b), n


def test_labels_change_the_battle_weights_and_leave_the_preview_ones_alone() -> None:
    config = distilled(coef_micro=20_000_000)
    with_labels, _ = trained(config, battle_labels())
    without, _ = trained(config, [])
    battle_moved = any(
        not torch.equal(a, b)
        for a, b in zip(
            with_labels.networks["battle"].parameters(),
            without.networks["battle"].parameters(),
            strict=False,
        )
    )
    assert battle_moved
    for a, b in zip(
        with_labels.networks["preview"].parameters(),
        without.networks["preview"].parameters(),
        strict=False,
    ):
        assert torch.equal(a, b)


def test_labels_are_given_exactly_when_the_config_has_the_block() -> None:
    student = learner(tiny_config())
    with pytest.raises(LearnerError, match="exactly when"):
        student.train(trajs(tiny_config()), [])
    with pytest.raises(LearnerError, match="exactly when"):
        learner(distilled()).train(trajs(distilled()))


def test_an_absent_block_keeps_the_report_keys_and_the_state_bytes() -> None:
    config = tiny_config()
    student = learner(config)
    report = student.train(trajs(config)).as_dict()
    for role_report in report.values():
        assert not any(key.startswith("search_") for key in role_report)
    assert b"search_distill" not in pack_state(student).sidecar
    distilling, _ = trained(distilled(), [])
    assert b"search_distill" in pack_state(distilling).sidecar


def test_the_term_reaches_the_shared_trunk() -> None:
    def edit(config: dict[str, Any]) -> None:
        config["model"]["options"] = {"shared_trunk": True}
        config["warm_start"] = {"battle": "b" * 64}
        config["team_dataset"] = {"path": "x", "dataset_hash": "5" * 64}
        config["ppo"]["value_warmup"] = {"generations": 1}
        config["ppo"]["search_distill"] = {
            "coef_micro": 20_000_000,
            "c_scale_micro": 4_000_000,
            "min_spread_ppm": 0,
            "aggregate_generations": 1,
        }
        config["generation"]["search_labels"] = {
            "search_config_hash": "d" * 64,
            "label_share_ppm": 40_000,
            "start_generation": 2,
        }

    shared = edited_config(edit)
    labels = battle_labels()
    with_labels, report = trained(shared, labels)
    without, _ = trained(shared, [])
    trunk = with_labels.networks["battle"].trunk_keys()
    named = dict(with_labels.networks["battle"].named_parameters())
    base = dict(without.networks["battle"].named_parameters())
    assert any(not torch.equal(named[k], base[k]) for k in trunk)
    assert (
        report.as_dict()["battle"]["search_kl_after"]
        < report.as_dict()["battle"]["search_kl_before"]
    )


def test_the_distillation_gradient_reaches_the_battle_network() -> None:
    config = distilled()
    student = learner(config)
    labels = battle_labels(3)
    batch = to_device(collate([lab.query for lab in labels]), student.device)
    dense = student._label_dense(labels, batch)
    kl = distill_kl(student.networks["battle"](batch).logits, batch.cand_mask, dense).mean()
    torch.autograd.backward(kl)
    grads = [p.grad for p in student.networks["battle"].parameters() if p.grad is not None]
    assert grads
    assert any(float(g.abs().sum()) > 0 for g in grads)
    assert all(p.grad is None for p in student.networks["preview"].parameters())
