"""``clone.situation_weights`` and ``clone-eval`` (ADR-010 erratum A13, #1390): the config rules,
the groups by hand, the factors against a hand computation, the unchanged shuffle and step count,
bit-identical reruns, the report keys, and ``clone-eval`` equal to the clone's own report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.checkpoint.files import checkpoint_bytes, checkpoint_id
from areneon_ml.learner import data, parallel
from areneon_ml.learner.__main__ import main as learner_main
from areneon_ml.learner.clone import (
    Agreement,
    CloneError,
    GroupAgreement,
    SituationReport,
    clone,
    evaluate_groups,
    situation_factors,
    situation_groups,
    split_parts,
)
from areneon_ml.learner.clone_config import CloneConfigError, SituationWeights, read_clone_config
from areneon_ml.learner.clone_eval import clone_eval
from areneon_ml.learner.clone_run import write_clone
from areneon_ml.learner.data import LearnerDataError, Step, trajectories
from areneon_ml.learner.situation import (
    GROUPS,
    Groups,
    stall_candidates,
    streak_slots,
    streak_unmatched,
)
from areneon_ml.learner.trainer import configure_learner
from areneon_ml.model.batch import collate
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES, init_network
from areneon_ml.reducer.state import BattleState, PokemonState, SideState

from tests.ml.learner.test_clone import (
    CLONES,
    KEY,
    TEACHER,
    VERSION,
    raw_config,
    run_main,
    teacher_stream,
)
from tests.ml.model.support import game

WEIGHTS = {"streak_ppm": 5_000_000}


def weighted_raw(
    manifest: dict[str, Any], weights: dict[str, int] | None = None, epochs: int = 2
) -> dict[str, Any]:
    raw = raw_config()
    raw["model"]["encoder"]["options"] = {"recent_history": True}
    raw["teacher"] = {
        "behaviour_policy_id": TEACHER,
        "training_set_hash": sha256_hex(jcs(manifest)),
    }
    raw["clone"]["epochs"] = epochs
    raw["clone"]["holdout"] = {"key": KEY, "ppm": 100000}
    if weights is not None:
        raw["clone"]["situation_weights"] = weights
    return raw


@pytest.fixture(autouse=True)
def deterministic() -> None:
    configure_learner(read_clone_config(raw_config()))


@pytest.fixture(scope="module")
def stream() -> tuple[str, dict[str, Any]]:
    return teacher_stream()


@pytest.fixture
def flag_some_streaks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Call every other turn's decision a streak one (the corpus has no repeat Protect to find),
    with the stall flag on the even candidates."""
    monkeypatch.setattr(
        data, "streak_slots", lambda state, request, moves: (0,) if state.turn % 2 else ()
    )
    monkeypatch.setattr(
        data,
        "stall_candidates",
        lambda choices, slots, request, moves: tuple(i % 2 == 0 for i in range(len(choices))),
    )


# -- the config reader --------------------------------------------------------------------------


def test_the_block_is_read_and_absent_means_none() -> None:
    plain = read_clone_config(raw_config())
    assert plain.clone.situation_weights is None
    assert situation_groups(plain) is None
    both = read_clone_config(json.loads((CLONES / "valid/situation-weights.json").read_text()))
    assert both.clone.situation_weights == SituationWeights(5_000_000, 2_000_000)
    assert situation_groups(both) == Groups(threatened=True)
    only = read_clone_config(
        json.loads((CLONES / "valid/situation-weights-streak-only.json").read_text())
    )
    assert only.clone.situation_weights == SituationWeights(5_000_000, None)
    assert situation_groups(only) == Groups(threatened=False)
    assert both.hash != plain.hash


@pytest.mark.parametrize("name", ["float", "zero", "over-cap", "extra-field", "threatened-alone"])
def test_a_block_the_contract_refuses_is_refused(name: str) -> None:
    raw = json.loads((CLONES / f"invalid/situation-weights-{name}.json").read_text())
    with pytest.raises(CloneConfigError):
        read_clone_config(raw)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("no-recent-history", "needs model.encoder.options.recent_history"),
        ("no-threat-rows", "probe-damage/2 or probe-outcome/1"),
    ],
)
def test_the_reader_rules_the_profile_cannot_say(name: str, message: str) -> None:
    raw = json.loads((CLONES / f"reader-invalid/situation-weights-{name}.json").read_text())
    with pytest.raises(CloneConfigError, match=message):
        read_clone_config(raw)


# -- the groups, by hand ------------------------------------------------------------------------

STALL = {"protect", "detect"}


def member(name: str, *, fainted: bool = False) -> dict[str, Any]:
    return {"ident": f"p1: {name}", "condition": "0 fnt" if fainted else "100/100"}


def slot(*ids: str, disabled: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"moves": [{"id": i, "disabled": i in disabled} for i in ids]}


def battle_state(streaks: dict[str, int]) -> BattleState:
    roster = [PokemonState(own=True, name=n) for n in streaks]
    for mon, streak in zip(roster, streaks.values(), strict=True):
        mon.recent.protect_streak = streak
    return BattleState(player="p1", own=SideState(player="p1", roster=roster))


def test_a_living_active_with_a_streak_and_a_stalling_move_offered_is_a_streak_slot() -> None:
    state = battle_state({"A": 1, "B": 0, "C": 2})
    request = {
        "side": {"pokemon": [member("A"), member("B"), member("C")]},
        "active": [slot("protect", "tackle"), slot("protect")],
    }
    assert streak_slots(state, request, STALL) == (0,)  # B has no streak
    assert streak_slots(battle_state({"A": 1, "B": 3}), request, STALL) == (0, 1)


def test_no_stall_offered_a_disabled_stall_a_fainted_active_and_a_streak_of_zero_are_not() -> None:
    state = battle_state({"A": 1, "B": 1})
    base = {"side": {"pokemon": [member("A"), member("B")]}}
    assert streak_slots(state, {**base, "active": [slot("tackle"), slot("tackle")]}, STALL) == ()
    disabled = {
        **base,
        "active": [slot("protect", "tackle", disabled=("protect",)), slot("tackle")],
    }
    assert streak_slots(state, disabled, STALL) == ()
    fainted = {"side": {"pokemon": [member("A", fainted=True), member("B")]}}
    assert (
        streak_slots(state, {**fainted, "active": [slot("protect"), slot("tackle")]}, STALL) == ()
    )
    assert streak_slots(state, {**fainted, "active": [None, slot("detect")]}, STALL) == (1,)
    assert streak_slots(battle_state({"A": 0}), {**base, "active": [slot("protect")]}, STALL) == ()


def test_a_forced_switch_a_team_preview_and_an_unknown_name_have_no_streak_slot() -> None:
    state = battle_state({"A": 1})
    assert (
        streak_slots(state, {"side": {"pokemon": [member("A")]}, "forceSwitch": [True]}, STALL)
        == ()
    )
    assert streak_slots(state, {"teamPreview": True, "side": {"pokemon": []}}, STALL) == ()
    stranger = {"side": {"pokemon": [member("Z")]}, "active": [slot("protect")]}
    assert streak_slots(state, stranger, STALL) == ()


def test_streak_unmatched_counts_the_living_actives_the_roster_does_not_name() -> None:
    state = battle_state({"A": 1})
    base = {
        "side": {"pokemon": [member("A"), member("Z")]},
        "active": [slot("protect"), slot("tackle")],
    }
    assert streak_unmatched(state, base) == 1  # Z is no roster entry
    assert streak_slots(state, base, STALL) == (0,)
    assert streak_unmatched(battle_state({"A": 1, "Z": 0}), base) == 0
    fainted = {
        "side": {"pokemon": [member("A"), member("Z", fainted=True)]},
        "active": base["active"],
    }
    assert streak_unmatched(state, fainted) == 0  # a fainted active is not judged either way
    assert streak_unmatched(state, {**base, "active": [slot("protect"), None]}) == 0
    assert streak_unmatched(state, {"teamPreview": True, "side": {"pokemon": []}}) == 0
    assert streak_unmatched(state, {"side": {"pokemon": [member("A")]}, "forceSwitch": [True]}) == 0


def test_a_candidate_stalls_when_a_streak_slot_uses_a_stalling_move() -> None:
    request = {"active": [slot("protect", "tackle"), slot("detect", "tackle")]}
    choices = [
        "move 1, move 2",
        "move 2, move 1",
        "move 1, pass",
        "switch 3, move 1",
        "move 3, pass",
    ]
    assert stall_candidates(choices, (0,), request, STALL) == (True, False, True, False, False)
    assert stall_candidates(choices, (0, 1), request, STALL) == (True, True, True, True, False)
    assert stall_candidates(choices, (), request, STALL) == (False,) * 5
    assert stall_candidates(["team 1234"], (0,), request, STALL) == (False,)


class Session:
    """The two things ``_Collector._situation`` reads of an encoder session."""

    def __init__(self, state: BattleState | None, request: dict[str, Any] | None) -> None:
        self.state = state
        self.last_request = None if request is None else "|request|" + json.dumps(request)


def session(state: BattleState | None, request: dict[str, Any] | None) -> Any:
    return Session(state, request)


def collector(threatened: bool, stall_moves: Any = frozenset(STALL)) -> data._Collector:
    vocabulary = SimpleNamespace(stall_moves=stall_moves)
    encoders = {"battle": SimpleNamespace(game=SimpleNamespace(vocabulary=vocabulary))}
    return data._Collector(encoders, groups=Groups(threatened))  # type: ignore[arg-type]


def test_a_decision_is_streak_then_threatened_then_other(monkeypatch: pytest.MonkeyPatch) -> None:
    streaky = battle_state({"A": 1})
    request = {"side": {"pokemon": [member("A")]}, "active": [slot("protect", "tackle")]}
    quiet = {"side": {"pokemon": [member("A")]}, "active": [slot("tackle")]}
    flags = {"value": True}
    monkeypatch.setattr(data, "threatened_flag", lambda log, features: flags["value"])
    observation: dict[str, Any] = {"log": []}
    candidates = [{"choice": "move 1, pass"}, {"choice": "move 2, pass"}]
    both = collector(True)
    # streak first, even when the decision is Threatened too: weights do not multiply
    assert both._situation("e", session(streaky, request), observation, candidates, {}) == (
        "streak",
        (True, False),
        0,
    )
    assert both._situation("e", session(streaky, quiet), observation, candidates, {}) == (
        "threatened",
        None,
        0,
    )
    flags["value"] = False
    assert both._situation("e", session(streaky, quiet), observation, candidates, {}) == (
        "other",
        None,
        0,
    )
    # a request slot no roster entry names is counted, whatever the group
    stray = {
        "side": {"pokemon": [member("A"), member("Z")]},
        "active": [slot("tackle"), slot("tackle")],
    }
    assert both._situation("e", session(streaky, stray), observation, candidates, {})[2] == 1
    # without threatened_ppm a Threatened decision is `other`, and no flag is read
    flags["value"] = True
    assert collector(False)._situation(
        "e", session(streaky, quiet), observation, candidates, {}
    ) == ("other", None, 0)


def test_a_battle_example_without_threat_rows_or_a_stall_table_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quiet = {"side": {"pokemon": [member("A")]}, "active": [slot("tackle")]}
    held = session(battle_state({"A": 0}), quiet)
    with pytest.raises(LearnerDataError, match="needs the threats, speeds and foe_priorities"):
        collector(True)._situation("e", held, {"log": []}, [], None)
    with pytest.raises(LearnerDataError, match="stalling moves of this engine pin"):
        collector(True, stall_moves=None)._situation("e", held, {"log": []}, [], {})
    monkeypatch.setattr(data, "threatened_flag", lambda log, features: None)
    with pytest.raises(LearnerDataError, match="Episode e"):
        collector(True)._situation("e", held, {"log": []}, [], {})


# -- the loader: groups on every Battle step, through a worker ----------------------------------


def encoders_with_history() -> dict[str, Encoder]:
    config = read_clone_config(weighted_raw(teacher_stream()[1], WEIGHTS))
    return {role: Encoder(game(), config.model.encoder) for role in ROLES}


def test_every_battle_step_has_a_group_and_a_preview_step_none(
    stream: tuple[str, dict[str, Any]],
) -> None:
    text, manifest = stream
    found = trajectories(text, manifest, encoders_with_history(), TEACHER, groups=Groups(False))
    for trajectory in found:
        for step in trajectory.steps:
            assert step.group == (None if trajectory.role == "preview" else "other")
            assert step.stall is None
    plain = trajectories(text, manifest, encoders_with_history(), TEACHER)
    assert all(s.group is None for t in plain for s in t.steps)


def test_the_workers_carry_the_groups_and_stall_flags_like_one_process(
    stream: tuple[str, dict[str, Any]],
    flag_some_streaks: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    text, manifest = stream
    path = tmp_path / "stream.jsonl"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(parallel, "MIN_PIECE_BYTES", 1)
    encoders = encoders_with_history()
    serial = trajectories(path, manifest, encoders, TEACHER, groups=Groups(False))
    found = trajectories(path, manifest, encoders, TEACHER, processes=3, groups=Groups(False))
    got = [[(s.group, s.stall) for s in t.steps] for t in found]
    assert got == [[(s.group, s.stall) for s in t.steps] for t in serial]
    flat = [pair for row in got for pair in row]
    assert {group for group, _ in flat} >= {"streak", "other", None}
    assert all((stall is not None) == (group == "streak") for group, stall in flat)


def test_a_piece_with_groups_and_flags_round_trips() -> None:
    collected = data.Collected(["e"], {"e": 1.0}, {})
    step = _first_step()
    flagged = Step(step.query, 0, group="streak", stall=(True, False, True))
    plain = Step(step.query, 1)
    collected.steps[("e", "battle")] = [flagged, plain]
    back = parallel.unpack_piece(parallel.pack_piece(parallel._Piece(collected, set(), set())))
    steps = back.collected.steps[("e", "battle")]
    assert [(s.group, s.stall) for s in steps] == [("streak", (True, False, True)), (None, None)]


def _first_step() -> Step:
    text, manifest = teacher_stream()
    found = trajectories(text, manifest, encoders_with_history(), TEACHER)
    return found[0].steps[0]


# -- the factors and the loss -------------------------------------------------------------------


def fake_steps(groups: list[str]) -> list[Step]:
    step = _first_step()
    return [Step(step.query, 0, group=g) for g in groups]


def test_the_factors_are_the_weight_over_the_mean_weight_and_average_one_exactly() -> None:
    steps = fake_steps(["streak", "other", "other", "other"])
    factors, mean = situation_factors(steps, SituationWeights(5_000_000))
    # w = 5e6, 1e6, 1e6, 1e6: sum 8e6, mean 2e6; factors 2.5, 0.5, 0.5, 0.5
    assert factors.tolist() == [2.5, 0.5, 0.5, 0.5]
    assert mean == 2_000_000
    assert factors.dtype == torch.float32
    assert factors.sum().item() == len(steps)
    mixed = fake_steps(["streak", "threatened", "other"])
    factors, mean = situation_factors(mixed, SituationWeights(5_000_000, 2_000_000))
    assert mean == 8_000_000 // 3  # floor of the mean
    assert factors.tolist() == pytest.approx([5 * 3 / 8, 2 * 3 / 8, 3 / 8])
    third, _ = situation_factors(mixed, SituationWeights(1_000_000, 1_000_000))
    assert third.tolist() == [1.0, 1.0, 1.0]


def test_the_group_table_counts_by_hand(
    stream: tuple[str, dict[str, Any]], flag_some_streaks: None
) -> None:
    text, manifest = stream
    config = read_clone_config(weighted_raw(manifest, WEIGHTS, epochs=1))
    found = trajectories(text, manifest, encoders_with_history(), TEACHER, groups=Groups(False))
    steps = [s for t in found if t.role == "battle" for s in t.steps]
    cloned = clone(config, game(), text, manifest)
    network = cloned.networks["battle"]
    table = evaluate_groups(network, steps, 4, torch.device("cpu"))
    assert tuple(table) == GROUPS
    assert sum(t.agreement.steps for t in table.values()) == len(steps)
    assert table["threatened"].agreement.steps == 0
    assert table["threatened"].agreement.loss == 0.0
    streak = [s for s in steps if s.group == "streak"]
    assert table["streak"].agreement.steps == len(streak) > 0
    assert table["streak"].stall_teacher == sum(int(s.stall[s.chosen]) for s in streak if s.stall)
    network.eval()
    clone_stalls = hits = 0
    with torch.no_grad():
        for s in streak:
            from areneon_ml.model.batch import collate  # noqa: PLC0415

            pick = int(network(collate([s.query])).logits[0].argmax())
            clone_stalls += int(s.stall[pick]) if s.stall else 0
            hits += int(pick == s.chosen)
    assert table["streak"].stall_clone == clone_stalls
    assert table["streak"].agreement.hits == hits
    assert table["other"].stall_teacher is None


# -- clones -------------------------------------------------------------------------------------


def run(manifest: dict[str, Any], text: str, weights: dict[str, int] | None) -> Any:
    return clone(read_clone_config(weighted_raw(manifest, weights)), game(), text, manifest)


def test_the_plain_loss_path_is_the_mean_cross_entropy_of_the_initial_network(
    stream: tuple[str, dict[str, Any]],
) -> None:
    """A no-block clone of one epoch and one minibatch reports, as its epoch loss, the mean
    cross-entropy of the freshly initialised Battle network over its training steps. Computed here
    independently with ``log_softmax`` in float64 (a pinned literal would move with the torch build,
    as checkpoint ids do, so this compares to 1e-5 instead): drift of the plain path fails it."""
    text, manifest = stream
    raw = weighted_raw(manifest, None, epochs=1)
    raw["clone"]["minibatch_size"] = 4096
    config = read_clone_config(raw)
    encoders = {role: Encoder(game(), config.model.encoder) for role in ROLES}
    found = trajectories(text, manifest, encoders, config.teacher_policy_id, groups=None)
    steps = split_parts(config, found)["battle"][0]
    assert 0 < len(steps) <= 4096
    network = init_network("battle", game(), config.model, config.init_battle).eval()
    with torch.no_grad():
        batch = collate([s.query for s in steps])
        logits = network(batch).logits.double().masked_fill(~batch.cand_mask, float("-inf"))
        chosen = torch.tensor([s.chosen for s in steps])
        expected = -torch.log_softmax(logits, dim=-1).gather(1, chosen[:, None]).mean().item()
    got = clone(config, game(), text, manifest).roles["battle"].epoch_loss
    assert len(got) == 1
    assert got[0] == pytest.approx(expected, rel=1e-5)


def battle_id(cloned: Any) -> str:
    return checkpoint_id(checkpoint_bytes(cloned.networks["battle"], game()))


def test_equal_weights_give_the_plain_loss_and_the_shuffle_and_steps_do_not_change(
    stream: tuple[str, dict[str, Any]],
) -> None:
    text, manifest = stream
    plain = run(manifest, text, None)
    equal = run(manifest, text, {"streak_ppm": 1_000_000})
    assert battle_id(plain) == battle_id(equal)  # every factor is exactly 1.0
    assert plain.roles["battle"].epoch_loss == equal.roles["battle"].epoch_loss
    assert equal.roles["battle"].optimiser_steps == plain.roles["battle"].optimiser_steps
    assert torch.equal(equal.generator.get_state(), plain.generator.get_state())
    assert equal.roles["preview"].as_dict() == plain.roles["preview"].as_dict()


def test_weights_change_the_battle_clone_and_never_the_shuffle_or_the_steps(
    stream: tuple[str, dict[str, Any]], flag_some_streaks: None
) -> None:
    text, manifest = stream
    plain = run(manifest, text, None)
    heavy = run(manifest, text, WEIGHTS)
    assert battle_id(heavy) != battle_id(plain)
    assert heavy.roles["battle"].optimiser_steps == plain.roles["battle"].optimiser_steps
    assert torch.equal(heavy.generator.get_state(), plain.generator.get_state())
    # the Preview role is not weighted: its clone and report are the plain ones
    assert checkpoint_id(checkpoint_bytes(heavy.networks["preview"], game())) == checkpoint_id(
        checkpoint_bytes(plain.networks["preview"], game())
    )
    assert heavy.roles["preview"].as_dict() == plain.roles["preview"].as_dict()
    # train and holdout stay the unweighted agreements: the weighted objective is epoch_loss only
    assert heavy.roles["battle"].train.steps == plain.roles["battle"].train.steps


def test_the_report_has_the_group_keys_only_under_the_block(
    stream: tuple[str, dict[str, Any]], flag_some_streaks: None
) -> None:
    text, manifest = stream
    plain = run(manifest, text, None).roles["battle"].as_dict()
    assert not {"situation_weights", "weight_mean_ppm", "train_groups", "holdout_groups"} & set(
        plain
    )
    report = run(manifest, text, WEIGHTS).roles["battle"].as_dict()
    assert report["situation_weights"] == WEIGHTS
    assert report["weight_mean_ppm"] > 1_000_000
    for part in ("train_groups", "holdout_groups"):
        assert tuple(report[part]) == GROUPS
        assert set(report[part]["streak"]) == {
            "steps",
            "loss",
            "top1_hits",
            "chance",
            "stall_teacher",
            "stall_clone",
            "streak_unmatched",
        }
        assert "stall_teacher" not in report[part]["other"]
    both = run(manifest, text, {"streak_ppm": 5_000_000})
    assert "threatened_ppm" not in both.roles["battle"].as_dict()["situation_weights"]
    assert "situation_weights" not in run(manifest, text, None).roles["preview"].as_dict()


def test_no_holdout_has_no_held_back_group_table(stream: tuple[str, dict[str, Any]]) -> None:
    text, manifest = stream
    raw = weighted_raw(manifest, WEIGHTS, epochs=1)
    raw["clone"]["holdout"] = {"key": KEY, "ppm": 0}
    report = clone(read_clone_config(raw), game(), text, manifest).roles["battle"].as_dict()
    assert "holdout_groups" not in report
    assert "train_groups" in report


def test_a_run_with_weights_is_bit_identical_on_rerun(
    stream: tuple[str, dict[str, Any]], tmp_path: Path, flag_some_streaks: None
) -> None:
    text, manifest = stream
    config = read_clone_config(weighted_raw(manifest, WEIGHTS))
    for name in ("a", "b"):
        write_clone(tmp_path / name, config, game(), clone(config, game(), text, manifest), VERSION)
    files = sorted(
        p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file()
    )
    assert files
    for relative in files:
        assert (tmp_path / "a" / relative).read_bytes() == (tmp_path / "b" / relative).read_bytes()


def test_a_clone_without_the_block_has_the_keys_it_had_before(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    """The report of a clone without the block has exactly the earlier keys. (Checkpoint ids
    depend on the machine's torch build, so they are not pinned here; the ids and report hash
    were compared against the previous main on the development machine.)"""
    text, manifest = stream
    raw = raw_config()
    raw["teacher"] = {
        "behaviour_policy_id": TEACHER,
        "training_set_hash": sha256_hex(jcs(manifest)),
    }
    raw["clone"]["epochs"] = 2
    raw["clone"]["holdout"] = {"key": KEY, "ppm": 100000}
    config = read_clone_config(raw)
    report = write_clone(
        tmp_path / "o", config, game(), clone(config, game(), text, manifest), VERSION
    )
    for role in ROLES:
        assert set(report["roles"][role]) == {"optimiser_steps", "epoch_loss", "train", "holdout"}
    assert "train_groups" not in json.dumps(report)


# -- clone-eval ---------------------------------------------------------------------------------


def write_stream(tmp_path: Path, stream: tuple[str, dict[str, Any]]) -> Path:
    text, manifest = stream
    out = tmp_path / "stream"
    out.mkdir()
    (out / "stream.jsonl").write_text(text, encoding="utf-8")
    (out / "training-set.json").write_text(json.dumps(manifest), encoding="utf-8")
    return out


def tree(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def test_clone_eval_equals_the_clones_own_report_and_changes_nothing(
    stream: tuple[str, dict[str, Any]], tmp_path: Path, flag_some_streaks: None
) -> None:
    text, manifest = stream
    config = read_clone_config(weighted_raw(manifest, WEIGHTS))
    cloned = clone(config, game(), text, manifest)
    out = tmp_path / "clone"
    report = write_clone(out, config, game(), cloned, VERSION)
    before = tree(out)
    table = clone_eval(out, write_stream(tmp_path, stream), game())
    battle = report["roles"]["battle"]
    assert table["train"] == battle["train"]
    assert table["holdout"] == battle["holdout"]
    assert table["train_groups"] == battle["train_groups"]
    assert table["holdout_groups"] == battle["holdout_groups"]
    assert table["checkpoint"] == report["checkpoints"]["battle"]
    assert table["threatened_read"] is False
    assert tree(out) == before


def test_clone_eval_measures_a_clone_made_without_the_block(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    text, manifest = stream
    config = read_clone_config(weighted_raw(manifest, None))
    out = tmp_path / "plain"
    write_clone(out, config, game(), clone(config, game(), text, manifest), VERSION)
    table = clone_eval(out, write_stream(tmp_path, stream), game())
    assert tuple(table["train_groups"]) == GROUPS
    assert table["train_groups"]["other"]["steps"] == table["train"]["steps"]
    assert table["train_groups"]["streak"]["steps"] == 0
    assert table["holdout_groups"]["other"]["steps"] == table["holdout"]["steps"]


def test_clone_eval_refuses_a_clone_that_is_not_what_it_says(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    text, manifest = stream
    config = read_clone_config(weighted_raw(manifest, None))
    out = tmp_path / "plain"
    write_clone(out, config, game(), clone(config, game(), text, manifest), VERSION)
    streams = write_stream(tmp_path, stream)
    report = json.loads((out / "clone-report.json").read_text())
    report["run_config_hash"] = "0" * 64
    (out / "clone-report.json").write_text(json.dumps(report))
    with pytest.raises(CloneError, match="another run config"):
        clone_eval(out, streams, game())
    report["run_config_hash"] = config.hash
    report["checkpoints"]["battle"] = "1" * 64
    (out / "clone-report.json").write_text(json.dumps(report))
    with pytest.raises(CloneError, match="cannot be loaded"):
        clone_eval(out, streams, game())
    report["checkpoints"]["battle"] = report["checkpoints"]["preview"]
    (out / "clone-report.json").write_text(json.dumps(report))
    with pytest.raises(CloneError, match="cannot be loaded"):
        clone_eval(out, streams, game())  # the Preview checkpoint is not a Battle one
    other = json.loads((streams / "training-set.json").read_text())
    other["behaviour_policy_id"] = "actor-rule@tiny"
    (streams / "training-set.json").write_text(json.dumps(other))
    report["checkpoints"]["battle"] = checkpoint_id(b"x")
    with pytest.raises(CloneError, match="not played by the teacher"):
        clone_eval(out, streams, game())


def test_the_clone_eval_command_prints_and_writes_the_table_and_refuses_misuse(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    text, manifest = stream
    config = read_clone_config(weighted_raw(manifest, None))
    out = tmp_path / "plain"
    write_clone(out, config, game(), clone(config, game(), text, manifest), VERSION)
    streams = write_stream(tmp_path, stream)
    table_path = tmp_path / "table.json"
    engine = tmp_path / "engine-data.json"
    from tests.ml.model.support import table  # noqa: PLC0415

    engine.write_text(json.dumps(table()), encoding="utf-8")
    argv = [
        "clone-eval", "--clone-dir", str(out), "--stream-dir", str(streams),
        "--engine-data", str(engine),
    ]  # fmt: skip
    status, stdout, error = run_main(learner_main, [*argv, "--out", str(table_path)])
    assert (status, error) == (0, "")
    assert json.loads(table_path.read_text()) == json.loads(stdout)
    assert run_main(learner_main, argv)[0] == 0  # no --out: stdout only
    status, _, error = run_main(learner_main, [*argv, "--out", str(table_path)])
    assert status == 3
    assert "never overwrites" in error
    status, _, error = run_main(learner_main, [*argv, "--out", str(out / "table.json")])
    assert status == 3
    assert "changes nothing of the clone" in error
    dangling = tmp_path / "dangling.json"
    target = tmp_path / "elsewhere.json"
    dangling.symlink_to(target)  # exists() is False for it: only an exclusive create refuses
    assert run_main(learner_main, [*argv, "--out", str(dangling)])[0] == 3
    assert not target.exists()
    status, _, error = run_main(learner_main, [*argv[:2], str(tmp_path / "none"), *argv[3:]])
    assert status == 3


def test_the_report_names_the_threatened_weight_when_it_is_set() -> None:
    empty = {name: GroupAgreement(Agreement(0, 0.0, 0, 0.0)) for name in GROUPS}
    report = SituationReport(SituationWeights(5_000_000, 2_000_000), 3_000_000, empty, None)
    assert report.as_dict()["situation_weights"] == {
        "streak_ppm": 5_000_000,
        "threatened_ppm": 2_000_000,
    }


def test_clone_eval_refuses_a_stream_that_is_not_its_manifest(
    stream: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    text, manifest = stream
    config = read_clone_config(weighted_raw(manifest, None))
    out = tmp_path / "plain"
    write_clone(out, config, game(), clone(config, game(), text, manifest), VERSION)
    streams = write_stream(tmp_path, stream)
    (streams / "stream.jsonl").write_text(text[: text.index("\n") + 1], encoding="utf-8")
    with pytest.raises(CloneError):
        clone_eval(out, streams, game())
