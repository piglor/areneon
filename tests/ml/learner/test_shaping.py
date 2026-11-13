"""``ppo.shaping`` (ADR-010 erratum A11.1, #1365): Phi read from the player's own view, the shaped
rewards, the value-head meaning rules, the report keys and the refusals, against hand-worked
values."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
import torch
from areneon_ml.learner.config import MICRO, RunConfigError, read_run_config
from areneon_ml.learner.data import LearnerDataError, Trajectory, trajectories
from areneon_ml.learner.diagnostics import calibration
from areneon_ml.learner.ppo import gae, normalise, terminal_rewards
from areneon_ml.learner.shaping import ShapingError, differential, phi, shaped
from areneon_ml.learner.state import pack_state
from areneon_ml.learner.trainer import Learner, LearnerError
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES
from areneon_ml.progress.run import ProgressError, refuse_shaped_pair, refuse_shaped_value
from areneon_ml.reducer.state import BattleState, PokemonState, SideState

from tests.ml.learner.support import POLICY, edited_config, learner, make_stream, tiny_config
from tests.ml.learner.test_config import raw
from tests.ml.learner.test_trainer import trajs
from tests.ml.model.support import game

COEF = 1_000_000


def request(*conditions: str) -> str:
    side = {"pokemon": [{"condition": c, "active": i < 2} for i, c in enumerate(conditions)]}
    return "|request|" + json.dumps({"side": side})


def opponent(*mons: PokemonState) -> BattleState:
    state = BattleState(player="p1")
    state.opponent = SideState(player="p2", roster=list(mons))
    return state


def seen(hp: int, den: int = 100) -> PokemonState:
    return PokemonState(own=False, hp_num=hp, hp_den=den)


UNSEEN = PokemonState(own=False)
FAINTED = PokemonState(own=False, fainted=True, hp_num=0, hp_den=100)
OWN4 = request("100/200", "50/50", "0 fnt", "1/3 par")


def test_phi_is_worked_out_by_hand_from_the_own_request_and_the_shown_opponent() -> None:
    # own: HP shares 500000, 1000000, 0, 333333 -> 1833333 // 4 = 458333; standing 3/4 = 750000
    # opponent (n = 4): 50% seen, fainted, 100% seen, a phantom (not counted), one unseen at full
    # -> (500000 + 0 + 1000000 + 1000000) // 4 = 625000; standing (4 - 1) / 4 = 750000
    state = opponent(seen(50), FAINTED, seen(100), PokemonState(own=False, phantom=True), UNSEEN)
    state.opponent.roster[3] = dataclasses.replace(
        state.opponent.roster[3], hp_num=1, hp_den=100
    )  # the phantom has hit points too: still not counted
    d = differential(state, OWN4)
    assert d == (458333 - 625000 + 750000 - 750000) // 4 == -41667
    assert phi(d, COEF) == pytest.approx(-0.041667)
    assert phi(d, 500_000) == pytest.approx(-0.0208335)


def test_unseen_opponent_pokemon_count_full_and_an_even_start_is_zero() -> None:
    state = opponent(UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN, UNSEEN)
    assert differential(state, request("10/10", "20/20", "30/30", "40/40")) == 0


def test_more_seen_than_brought_and_a_missing_request_are_refused() -> None:
    state = opponent(*(seen(100) for _ in range(5)))
    with pytest.raises(ShapingError, match="5 opponent Pokemon were seen but a side brings 4"):
        differential(state, OWN4)
    with pytest.raises(ShapingError, match="no request"):
        differential(state, None)
    with pytest.raises(ShapingError, match="holds no request"):
        differential(state, "|turn|1")
    with pytest.raises(ShapingError, match="lists no own Pokemon"):
        differential(state, "|request|" + json.dumps({"side": {"pokemon": []}}))
    with pytest.raises(ShapingError, match="not a hit point count"):
        differential(state, request("lots"))


def test_the_shaped_rewards_follow_the_formula_and_telescope_at_gamma_one() -> None:
    potentials = [0.0, -0.25, 0.1, 0.4]
    rewards = [0.0, 0.0, 0.0, 1.0]
    out, magnitude = shaped(rewards, potentials, 1.0)
    # r'_t = r_t + Phi(t+1) - Phi(t), Phi after the last decision 0
    assert out == pytest.approx([-0.25, 0.35, 0.3, 1.0 - 0.4])
    assert sum(out) == pytest.approx(1.0 - potentials[0])
    assert magnitude == pytest.approx(0.25 + 0.35 + 0.3 + 0.4)
    out, _ = shaped([0.0, 1.0], [0.2, 0.5], 0.9)
    assert out == pytest.approx([0.9 * 0.5 - 0.2, 1.0 - 0.5])
    with pytest.raises(ShapingError, match="a reward and a potential"):
        shaped([0.0], [0.0, 0.0], 1.0)


# -- the trainer ---------------------------------------------------------------


def shaping_config(coef: int | None = COEF, **extra: Any) -> Any:
    def edit(config: dict[str, Any]) -> None:
        if coef is not None:
            config["ppo"]["shaping"] = {"coef_micro": coef}
        config["ppo"].update(extra)

    return edited_config(edit)


def with_potentials(config: Any) -> list[Trajectory]:
    """The default trajectories with a worked ``d`` on every Battle step (1000 * i - 4000)."""
    out = []
    for t in trajs(config):
        steps = tuple(
            dataclasses.replace(s, potential_d=1000 * i - 4000) if t.role == "battle" else s
            for i, s in enumerate(t.steps)
        )
        out.append(dataclasses.replace(t, steps=steps))
    return out


def captured(student: Learner, tr: list[Trajectory]) -> tuple[dict[str, Any], Any]:
    seen_tables: dict[str, Any] = {}
    original = Learner._epochs

    def spy(self: Learner, role: str, table: Any, warm: bool) -> Any:
        seen_tables[role] = table
        return original(self, role, table, warm)

    Learner._epochs = spy  # type: ignore[method-assign]
    try:
        report = student.train(tr)
    finally:
        Learner._epochs = original  # type: ignore[method-assign]
    return seen_tables, report


def test_battle_returns_are_gae_on_the_shaped_rewards_and_preview_is_untouched() -> None:
    config = shaping_config()
    tr = with_potentials(config)
    tables, report = captured(learner(config), tr)
    fresh = learner(config)
    battle = [t for t in tr if t.role == "battle"]
    _, values = fresh._score("battle", [s for t in battle for s in t.steps])
    advantages, returns, start, magnitude, count = [], [], 0, 0.0, 0
    for t in battle:
        end = start + len(t.steps)
        phis = [phi(s.potential_d or 0, COEF) for s in t.steps]
        moved, step_magnitude = shaped(
            terminal_rewards(len(t.steps), t.score).tolist(), phis, config.ppo.discount
        )
        magnitude += step_magnitude
        count += len(t.steps)
        a, r = gae(
            torch.tensor(moved), values[start:end], config.ppo.discount, config.ppo.gae_lambda
        )
        advantages.append(a)
        returns.append(r)
        start = end
    assert torch.allclose(tables["battle"].returns.cpu(), torch.cat(returns), rtol=0, atol=1e-5)
    assert torch.allclose(
        tables["battle"].advantages.cpu(), normalise(torch.cat(advantages)), rtol=0, atol=1e-5
    )
    stored = report.as_dict()["battle"]
    assert stored["shaping_coef_micro"] == COEF
    assert stored["shaping_mean_abs_micro"] == int(magnitude / count * MICRO)
    assert stored["shaping_mean_abs_micro"] > 0
    # the Preview role: no key, and its table equals a run without the block
    assert "shaping_coef_micro" not in report.as_dict()["preview"]
    plain_tables, _ = captured(learner(tiny_config()), trajs(tiny_config()))
    for name in ("returns", "advantages", "old_log_prob"):
        assert torch.equal(getattr(tables["preview"], name), getattr(plain_tables["preview"], name))


def test_the_calibration_bins_compare_the_value_plus_phi_with_the_score() -> None:
    config = shaping_config()
    tr = with_potentials(config)
    tables, _ = captured(learner(config), tr)
    fresh = learner(config)
    battle = [t for t in tr if t.role == "battle"]
    steps = [s for t in battle for s in t.steps]
    _, values = fresh._score("battle", steps)
    scores = [t.score for t in battle for _ in t.steps]
    plus = [
        v + phi(s.potential_d or 0, COEF)
        for v, s in zip(values.double().tolist(), steps, strict=True)
    ]
    assert tables["battle"].value_calibration == calibration(plus, scores)
    assert tables["battle"].value_calibration != calibration(values.double().tolist(), scores)


def test_the_preview_bootstrap_reads_the_battle_value_plus_phi_of_the_first_decision() -> None:
    config = shaping_config(preview_bootstrap_micro=MICRO)
    tr = with_potentials(config)
    tables, _ = captured(learner(config), tr)
    fresh = learner(config)
    battle = [t for t in tr if t.role == "battle"]
    _, bvalues = fresh._score("battle", [s for t in battle for s in t.steps])
    first, start = {}, 0
    for t in battle:
        first[t.episode] = float(bvalues[start]) + phi(t.steps[0].potential_d or 0, COEF)
        start += len(t.steps)
    preview = [t for t in tr if t.role == "preview"]
    _, pvalues = fresh._score("preview", [s for t in preview for s in t.steps])
    targets = torch.tensor([first[t.episode] for t in preview], dtype=torch.float32)
    assert torch.allclose(tables["preview"].returns.cpu(), targets, rtol=0, atol=1e-6)
    assert float(targets.abs().sum()) > 0
    assert pvalues.shape[0] == len(preview)


def test_absent_keeps_todays_weights_and_report_and_the_block_changes_the_battle_weights() -> None:
    absent = learner()
    absent_report = absent.train(trajs(tiny_config())).as_dict()
    assert all("shaping_coef_micro" not in r for r in absent_report.values())
    # potential_d on the steps is ignored without the block
    ignoring = learner()
    ignoring.train(with_potentials(tiny_config()))
    assert pack_state(ignoring).tensors == pack_state(absent).tensors
    config = shaping_config()
    on = learner(config)
    on.train(with_potentials(config))
    assert pack_state(on).tensors != pack_state(absent).tensors
    assert json.loads(pack_state(on).sidecar)["ppo"]["shaping"] == {"coef_micro": COEF}


def test_a_battle_step_without_its_potential_is_refused() -> None:
    config = shaping_config()
    with pytest.raises(LearnerError, match="needs the Battle steps read with their potential"):
        learner(config).train(trajs(config))


def test_the_collector_reads_the_potential_of_every_battle_step_only_under_shaping() -> None:
    config = tiny_config()
    encoders = {role: Encoder(game(), config.model.encoder) for role in ROLES}
    text, manifest = make_stream()
    plain = trajectories(text, manifest, encoders, POLICY)
    on = trajectories(text, manifest, encoders, POLICY, shaping=True)
    assert [t.role for t in plain] == [t.role for t in on]
    battle_steps = 0
    for before, after in zip(plain, on, strict=True):
        for a, b in zip(before.steps, after.steps, strict=True):
            assert a.potential_d is None
            assert (b.potential_d is not None) == (after.role == "battle")
            assert b.chosen == a.chosen
            if b.potential_d is not None:
                battle_steps += 1
                assert -500_000 <= b.potential_d <= 500_000
    assert battle_steps == 55


def test_a_potential_that_cannot_be_read_is_a_data_error_naming_the_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from areneon_ml.learner import data  # noqa: PLC0415

    def refuse(state: Any, request: Any) -> int:
        raise ShapingError("5 opponent Pokemon were seen but a side brings 4")

    monkeypatch.setattr(data, "differential", refuse)
    config = tiny_config()
    encoders = {role: Encoder(game(), config.model.encoder) for role in ROLES}
    text, manifest = make_stream()
    with pytest.raises(LearnerDataError, match=r"Episode .*: 5 opponent Pokemon were seen"):
        trajectories(text, manifest, encoders, POLICY, shaping=True)


# -- the readers and the refusals ----------------------------------------------


def test_the_reader_reads_the_block_and_absent_is_none() -> None:
    assert read_run_config(raw()).ppo.shaping is None
    config = read_run_config(raw("valid/potential.json"))
    assert config.ppo.shaping is not None
    assert config.ppo.shaping.coef_micro == COEF
    assert config.hash != read_run_config(raw()).hash


def test_a_run_with_shaping_is_refused_by_the_value_consumers() -> None:
    config = shaping_config()
    with pytest.raises(ProgressError, match="reads the value head as a score"):
        refuse_shaped_value(config, "the best-response responder")
    with pytest.raises(ProgressError, match="the search candidate"):
        refuse_shaped_pair(config, "argmax", None)
    with pytest.raises(ProgressError, match="the Team Preview solver"):
        refuse_shaped_pair(config, None, "sample")
    refuse_shaped_pair(config, None, None)
    refuse_shaped_pair(tiny_config(), "argmax", "sample")


def test_the_reader_rule_names_the_pairing() -> None:
    bad = raw("reader-invalid/potential-search-distill.json")
    with pytest.raises(RunConfigError, match=r"ppo\.shaping cannot be combined"):
        read_run_config(bad)
