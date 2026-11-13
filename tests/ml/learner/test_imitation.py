"""Teacher imitation in the learner (ADR-010 erratum A11.3, #1367): the reader rules, the teacher
stream reader and each refusal, the draw (K, determinism, independence from the main generator),
the cross-entropy against values worked out by hand at temperature 1 and 0.5, the Battle-only term
in the update (report keys, off in the value warm-up, shared trunk), the sidecar and a resume."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.learner.config import ImitationSettings, RunConfig, RunConfigError, read_run_config
from areneon_ml.learner.data import LearnerDataError, Step
from areneon_ml.learner.imitation import (
    DrawStream,
    check_teacher_manifest,
    read_teacher,
)
from areneon_ml.learner.ppo import policy_terms
from areneon_ml.learner.state import LearnerStateError, pack_state, restore_state, state_hash
from areneon_ml.learner.trainer import LearnerError
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES

from tests.ml.learner.support import (
    POLICY,
    edited_config,
    learner,
    make_stream,
    stream_of,
    tiny_config,
)
from tests.ml.learner.test_config import raw
from tests.ml.learner.test_trainer import trajs
from tests.ml.model.support import game

TEACHER_ID = "damage-aware@v3"
SCHEDULE_LINEAR = {"kind": "linear", "final_micro": 0, "generations": 3}


def teacher_stream(policy: str = TEACHER_ID) -> tuple[str, dict[str, Any], str]:
    text, manifest = make_stream(policy=policy)
    return text, manifest, sha256_hex(jcs(manifest))


def imitating(  # noqa: PLR0913
    *,
    coef: int = 100_000,
    share: int = 500_000,
    schedule: dict[str, Any] | None = None,
    warmup: int | None = None,
    shared: bool = False,
    digest: str | None = None,
) -> RunConfig:
    digest = digest or teacher_stream()[2]

    def edit(config: dict[str, Any]) -> None:
        config["warm_start"] = {"battle": "b" * 64}
        if warmup is not None:
            config["ppo"]["value_warmup"] = {"generations": warmup}
        if shared:
            config["model"]["options"] = {"shared_trunk": True}
        config["ppo"]["imitation"] = {
            "training_set_hash": digest,
            "coef_micro": coef,
            "share_ppm": share,
            "schedule": schedule or {"kind": "constant"},
        }

    return edited_config(edit)


def teacher_steps(config: RunConfig, tmp_path: Path) -> tuple[Step, ...]:
    text, manifest, digest = teacher_stream()
    path = tmp_path / "stream.jsonl"
    path.write_text(text)
    encoders = {role: Encoder(game(), config.model.encoder) for role in ROLES}
    return read_teacher(path, manifest, digest, encoders)


# -- the config ---------------------------------------------------------------------------------


def test_the_reader_reads_the_block_and_absent_is_none() -> None:
    constant = read_run_config(raw("valid/imitation-constant.json"))
    assert constant.ppo.imitation == ImitationSettings("e" * 64, 100_000, 250_000, None)
    linear = read_run_config(raw("valid/imitation-linear.json")).ppo.imitation
    assert linear is not None
    assert linear.schedule is not None
    assert [linear.coef_micro_at(t) for t in (0, 5, 10, 11)] == [100_000, 50_000, 0, 0]
    assert tiny_config().ppo.imitation is None
    assert constant.hash != tiny_config().hash


def test_the_coefficient_rule_and_k() -> None:
    constant = ImitationSettings("e" * 64, 100_000, 250_000, None)
    assert constant.coef_micro_at(7) == 100_000
    with pytest.raises(RunConfigError, match="counts from 0"):
        constant.coef_micro_at(-1)
    assert constant.steps(55, 1000) == 13  # floor(0.25 * 55)
    assert constant.steps(55, 5) == 5  # capped at the set's size
    assert constant.steps(3, 1000) == 0


@pytest.mark.parametrize(
    ("name", "message"),
    [("no-warm-start", "needs warm_start"), ("population", "cannot be combined with a population")],
)
def test_the_reader_rules(name: str, message: str) -> None:
    with pytest.raises(RunConfigError, match=message):
        read_run_config(raw(f"reader-invalid/imitation-{name}.json"))


@pytest.mark.parametrize("name", ["zero-share", "short-hash", "extra-field"])
def test_the_schema_refuses_the_rest(name: str) -> None:
    with pytest.raises(RunConfigError):
        read_run_config(raw(f"invalid/imitation-{name}.json"))


# -- the teacher stream -------------------------------------------------------------------------


def test_the_teacher_stream_is_read_into_battle_steps_only(tmp_path: Path) -> None:
    config = imitating()
    steps = teacher_steps(config, tmp_path)
    assert len(steps) == 55  # the 55 battle decisions of the default stream; Previews are skipped
    assert all(isinstance(s, Step) for s in steps)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"data_class": "private"}, "not synthetic"),
        ({"learner_ready": False}, "not learner_ready"),
        ({"behaviour_policy_id": POLICY}, "scripted teacher"),
    ],
)
def test_the_manifest_refusals(change: dict[str, Any], message: str) -> None:
    _, manifest, _ = teacher_stream()
    with pytest.raises(LearnerDataError, match=message):
        check_teacher_manifest({**manifest, **change}, None)


def test_the_stream_refusals(tmp_path: Path) -> None:
    config = imitating()
    text, manifest, digest = teacher_stream()
    path = tmp_path / "stream.jsonl"
    path.write_text(text)
    encoders = {role: Encoder(game(), config.model.encoder) for role in ROLES}
    with pytest.raises(LearnerDataError, match="does not hash"):
        read_teacher(path, manifest, "0" * 64, encoders)
    broken = {**manifest, "stream_sha256": "0" * 64}
    with pytest.raises(LearnerDataError):  # the digest no longer matches the stream
        read_teacher(path, broken, sha256_hex(jcs(broken)), encoders)
    learned = {**manifest, "behaviour_policy_id": POLICY}
    with pytest.raises(LearnerDataError, match="scripted teacher"):
        read_teacher(path, learned, sha256_hex(jcs(learned)), encoders)
    assert digest


def test_an_example_the_encoder_refuses_is_a_data_error(tmp_path: Path) -> None:
    # A stream whose examples carry candidate rows the run's encoder (no feature set) refuses.
    config = imitating()
    text, manifest, _ = teacher_stream()
    lines = [json.loads(line) for line in text.splitlines()]
    for example in lines:
        if example["observation"]["kind"] == "battle":
            example["candidate_features"] = {
                "feature_set": "probe-damage/1",
                "feature_set_hash": "c" * 64,
                "rows": [],
            }
    refused, refused_manifest = stream_of(lines, TEACHER_ID)
    path = tmp_path / "stream.jsonl"
    path.write_text(refused)
    encoders = {role: Encoder(game(), config.model.encoder) for role in ROLES}
    with pytest.raises(LearnerDataError):
        read_teacher(path, refused_manifest, sha256_hex(jcs(refused_manifest)), encoders)
    assert manifest


# -- the draw -----------------------------------------------------------------------------------


def test_the_draw_is_deterministic_and_has_no_repeats() -> None:
    first = DrawStream(13, 4).draw(40, 10)
    assert first == DrawStream(13, 4).draw(40, 10)
    assert len(set(first)) == 10
    assert first != DrawStream(13, 5).draw(40, 10)  # another update, another draw
    assert first != DrawStream(14, 4).draw(40, 10)  # another learner seed
    stream = DrawStream(13, 4)
    stream.draw(40, 10)
    again = stream.shuffle(list(range(10)))
    assert sorted(again) == list(range(10))


def test_the_draw_leaves_the_main_generator_alone(tmp_path: Path) -> None:
    config = imitating(coef=1)
    steps = teacher_steps(config, tmp_path)
    work = trajs(config)
    student = learner(config)
    plain = learner(tiny_config())
    student.generation = plain.generation = 1
    student.train(work, None, steps)
    plain.train(work)
    # the generator's state advanced identically: the term never drew from it
    assert torch.equal(student.generator.get_state(), plain.generator.get_state())


# -- the cross-entropy, by hand -----------------------------------------------------------------


def test_the_cross_entropy_at_temperature_one_and_a_half() -> None:
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    mask = torch.tensor([[True, True, True]])
    chosen = torch.tensor([0])
    # ln(e + 1 + e^-1) = 1.4076059644 so the log-probability of candidate 0 is 1 - 1.4076
    one = policy_terms(logits, mask, chosen, None).log_prob
    assert float(one) == pytest.approx(1 - math.log(math.e + 1 + math.exp(-1)), abs=1e-6)
    assert float(-one) == pytest.approx(0.4076059644, abs=1e-6)
    # tau 0.5: logits double, 2 - ln(e^2 + 1 + e^-2) = -0.1429
    half = policy_terms(logits, mask, chosen, 500_000).log_prob
    assert float(-half) == pytest.approx(math.log(math.exp(2) + 1 + math.exp(-2)) - 2, abs=1e-6)
    # a padded fourth place changes nothing
    padded = policy_terms(
        torch.tensor([[1.0, 0.0, -1.0, 9.0]]), torch.tensor([[True, True, True, False]]), chosen
    ).log_prob
    assert float(padded) == pytest.approx(float(one), abs=1e-6)


@pytest.mark.parametrize("tau", [None, 500_000])
def test_the_learners_evaluation_equals_the_hand_formula(tau: int | None, tmp_path: Path) -> None:
    def edit(config: dict[str, Any]) -> None:
        config["warm_start"] = {"battle": "b" * 64}
        config["ppo"]["imitation"] = {
            "training_set_hash": "e" * 64,
            "coef_micro": 1,
            "share_ppm": 1,
            "schedule": {"kind": "constant"},
        }
        if tau is not None:
            config["generation"]["play"] = {"temperature_ppm": tau}

    config = edited_config(edit)
    steps = teacher_steps(config, tmp_path)[:6]
    student = learner(config)

    class Stub:
        """A network whose logits for a decision of n candidates are 0.3 * (n - i)."""

        def eval(self) -> None:
            return None

        def __call__(self, batch: Any) -> Any:
            logits = torch.zeros(batch.cand_mask.shape)
            for row in range(logits.shape[0]):
                n = int(batch.cand_mask[row].sum())
                logits[row, :n] = torch.tensor([0.3 * (n - i) for i in range(n)])
            return SimpleNamespace(logits=logits, value=torch.zeros(logits.shape[0]))

    student.networks["battle"] = Stub()  # type: ignore[assignment]
    scale = 1.0 if tau is None else 1_000_000 / tau
    expected = 0.0
    hits = 0
    for step in steps:
        n = int(step.query.pending.shape[0])
        z = [0.3 * (n - i) * scale for i in range(n)]
        norm = math.log(sum(math.exp(v) for v in z))
        expected += norm - z[step.chosen]
        hits += step.chosen == 0  # the stub ranks candidate 0 first
    ce, found = student._imitation_eval(steps)
    assert ce == pytest.approx(expected / len(steps), abs=1e-5)
    assert found == hits
    assert student._imitation_eval(()) == (0.0, 0)


# -- the term in the update ---------------------------------------------------------------------


def trained(config: RunConfig, tmp_path: Path, updates: int = 1, start: int = 0) -> Any:
    steps = teacher_steps(config, tmp_path)
    student = learner(config)
    student.generation = start
    report = None
    for _ in range(updates):
        report = student.train(trajs(config), None, steps)
    return student, report


def test_an_update_reports_the_battle_role_only_and_the_cross_entropy_falls(tmp_path: Path) -> None:
    config = imitating(coef=20_000_000, share=1_000_000)
    _, report = trained(config, tmp_path)
    battle = report.as_dict()["battle"]
    assert battle["imitation_steps"] == 55  # share 1.0 of the 55 battle steps, 55 available
    assert battle["imitation_coef_micro"] == 20_000_000
    assert battle["imitation_ce_after"] < battle["imitation_ce_before"]
    assert 0 <= battle["imitation_top1_ppm"] <= 1_000_000
    assert not any(k.startswith("imitation_") for k in report.as_dict()["preview"])


def test_k_follows_the_share(tmp_path: Path) -> None:
    config = imitating(share=250_000)
    _, report = trained(config, tmp_path)
    assert report.as_dict()["battle"]["imitation_steps"] == 13  # floor(0.25 * 55)


def test_the_term_is_off_in_the_value_warmup(tmp_path: Path) -> None:
    config = imitating(coef=20_000_000, warmup=1)
    steps = teacher_steps(config, tmp_path)
    student = learner(config)
    first = student.train(trajs(config), None, steps).as_dict()["battle"]
    assert first["value_warmup"] is True
    assert (first["imitation_steps"], first["imitation_coef_micro"]) == (0, 0)
    assert (first["imitation_ce_before"], first["imitation_ce_after"]) == (0.0, 0.0)
    second = student.train(trajs(config), None, steps).as_dict()["battle"]
    assert second["imitation_steps"] > 0
    assert second["imitation_coef_micro"] == 20_000_000


def test_the_term_changes_the_battle_weights_and_leaves_the_preview_ones_alone(
    tmp_path: Path,
) -> None:
    config = imitating(coef=20_000_000, share=1_000_000)
    mixed, _ = trained(config, tmp_path)
    plain = learner(tiny_config())
    plain.train(trajs(config))
    assert any(
        not torch.equal(a, b)
        for a, b in zip(
            mixed.networks["battle"].parameters(),
            plain.networks["battle"].parameters(),
            strict=True,
        )
    )
    for a, b in zip(
        mixed.networks["preview"].parameters(), plain.networks["preview"].parameters(), strict=True
    ):
        assert torch.equal(a, b)


def test_the_term_reaches_the_shared_trunk(tmp_path: Path) -> None:
    config = imitating(coef=20_000_000, share=1_000_000, shared=True)
    mixed, report = trained(config, tmp_path)
    zero = imitating(coef=1, share=1_000_000, shared=True)
    base, _ = trained(zero, tmp_path)
    trunk = mixed.networks["battle"].trunk_keys()
    named = dict(mixed.networks["battle"].named_parameters())
    other = dict(base.networks["battle"].named_parameters())
    assert any(not torch.equal(named[k], other[k]) for k in trunk)
    assert (
        report.as_dict()["battle"]["imitation_ce_after"]
        < (report.as_dict()["battle"]["imitation_ce_before"])
    )


def test_teacher_steps_are_given_exactly_when_the_config_has_the_block(tmp_path: Path) -> None:
    config = imitating()
    steps = teacher_steps(config, tmp_path)
    with pytest.raises(LearnerError, match="exactly when"):
        learner(config).train(trajs(config))
    plain = tiny_config()
    with pytest.raises(LearnerError, match="exactly when"):
        learner(plain).train(trajs(plain), None, steps)


def test_an_update_is_deterministic(tmp_path: Path) -> None:
    config = imitating(share=1_000_000)
    first, _ = trained(config, tmp_path)
    second, _ = trained(config, tmp_path)
    assert pack_state(first) == pack_state(second)


def test_an_absent_block_keeps_the_report_keys_and_the_state_bytes() -> None:
    config = tiny_config()
    student = learner(config)
    report = student.train(trajs(config)).as_dict()
    for role_report in report.values():
        assert not any(key.startswith("imitation_") for key in role_report)
    assert b"imitation" not in pack_state(student).sidecar


# -- the schedule, the sidecar and a resume -------------------------------------------------------


def test_the_coefficient_follows_the_schedule_from_the_first_policy_update(tmp_path: Path) -> None:
    config = imitating(coef=90_000, schedule=SCHEDULE_LINEAR, warmup=1)
    steps = teacher_steps(config, tmp_path)
    student = learner(config)
    coefs = []
    for _ in range(5):
        report = student.train(trajs(config), None, steps).as_dict()
        coefs.append(report["battle"]["imitation_coef_micro"])
    assert coefs == [0, 90_000, 60_000, 30_000, 0]  # the warm-up update is off, then t = 0..3
    assert json.loads(pack_state(student).sidecar)["imitation_coef_micro"] == 0


def test_the_sidecar_records_the_coefficient_for_a_linear_schedule_only(tmp_path: Path) -> None:
    constant, _ = trained(imitating(), tmp_path)
    assert "imitation_coef_micro" not in json.loads(pack_state(constant).sidecar)
    linear = imitating(schedule=SCHEDULE_LINEAR)
    fresh = learner(linear)
    assert "imitation_coef_micro" not in json.loads(pack_state(fresh).sidecar)
    student, _ = trained(linear, tmp_path)
    assert json.loads(pack_state(student).sidecar)["imitation_coef_micro"] == 100_000


@pytest.mark.parametrize("stop_after", [1, 2])
def test_a_resume_reaches_the_same_state_hash(stop_after: int, tmp_path: Path) -> None:
    config = imitating(schedule=SCHEDULE_LINEAR, warmup=1, share=1_000_000)
    steps = teacher_steps(config, tmp_path)
    continuous = learner(config)
    for _ in range(stop_after):
        continuous.train(trajs(config), None, steps)
    halfway = pack_state(continuous)
    resumed = learner(config)
    restore_state(resumed, halfway.tensors, halfway.sidecar, halfway.hash)
    assert resumed.last_imitation_coef_micro == continuous.last_imitation_coef_micro
    assert pack_state(resumed) == halfway
    for _ in range(3 - stop_after):
        continuous.train(trajs(config), None, steps)
        resumed.train(trajs(config), None, steps)
    assert pack_state(resumed) == pack_state(continuous)


def test_a_sidecar_with_another_coefficient_is_refused(tmp_path: Path) -> None:
    linear = imitating(schedule=SCHEDULE_LINEAR)
    student, _ = trained(linear, tmp_path)
    packed = pack_state(student)
    for change in ("set", "drop"):
        sidecar = json.loads(packed.sidecar)
        if change == "set":
            sidecar["imitation_coef_micro"] = 1
        else:
            del sidecar["imitation_coef_micro"]
        with pytest.raises(LearnerStateError, match="imitation weight"):
            restore_state(
                learner(linear), packed.tensors, jcs(sidecar), state_hash(packed.tensors, sidecar)
            )
    plain = learner(tiny_config())
    plain.train(trajs(tiny_config()))
    stored = pack_state(plain)
    sidecar = json.loads(stored.sidecar)
    sidecar["imitation_coef_micro"] = 5
    with pytest.raises(LearnerStateError, match="imitation weight"):
        restore_state(
            learner(tiny_config()),
            stored.tensors,
            jcs(sidecar),
            state_hash(stored.tensors, sidecar),
        )


def test_the_coefficient_rule_without_a_block() -> None:
    assert learner(tiny_config()).imitation_coef_micro(3) is None
