"""``ppo.aux_opponent`` and the opponent next-action, item and ability labels (ADR-010 erratum
A14.2, #1434): labels worked out by hand on built channels, every mask, the loss, the loader and its
worker transport, the trainer, the value warm-up, the report, the resume and the privacy rules."""

from __future__ import annotations

import ast
import inspect
import json
import re
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml import learner as learner_package
from areneon_ml.learner import aux_opponent, parallel
from areneon_ml.learner.aux_opponent import (
    HEADS,
    Lookup,
    OppLabels,
    Snap,
    Unit,
    labels,
    lookup_of,
    opponent_terms,
    snapshot,
    targets_tensors,
)
from areneon_ml.learner.config import MICRO, RunConfig, RunConfigError, read_run_config
from areneon_ml.learner.data import LearnerDataError, Trajectory, _Collector, trajectories
from areneon_ml.learner.state import pack_state, restore_state
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES
from areneon_ml.reducer.reducer import reduce_channel
from areneon_ml.reducer.state import BattleState, PokemonState, SideState
from areneon_ml.reducer.vocab import to_id

from tests.ml.learner.support import POLICY, edited_config, learner, make_stream, tiny_config
from tests.ml.learner.test_config import raw
from tests.ml.model.support import game

COEFS = {"action_micro": 50_000, "item_micro": 25_000, "ability_micro": 25_000}


def opp_config(coefs: dict[str, int] | None = None, **ppo: Any) -> RunConfig:
    def edit(config: dict[str, Any]) -> None:
        config["model"]["options"] = {"aux_opponent_heads": True}
        config["ppo"]["aux_opponent"] = dict(COEFS if coefs is None else coefs)
        config["ppo"].update(ppo)

    return edited_config(edit)


def warm_config(generations: int = 2) -> RunConfig:
    def edit(config: dict[str, Any]) -> None:
        config["model"]["options"] = {"aux_opponent_heads": True}
        config["ppo"]["aux_opponent"] = dict(COEFS)
        config["warm_start"] = {"battle": "b" * 64}
        config["ppo"]["value_warmup"] = {"generations": generations}

    return edited_config(edit)


def encoders(config: RunConfig) -> dict[str, Encoder]:
    return {role: Encoder(game(), config.model.encoder) for role in ROLES}


def opp_trajectories(config: RunConfig, seed: int = 0) -> list[Trajectory]:
    text, manifest = make_stream(seed=seed)
    settings = config.ppo.aux_opponent
    assert settings is not None
    return trajectories(text, manifest, encoders(config), POLICY, aux_opponent=settings.heads)


# -- labels, by hand ------------------------------------------------------------------------------

MOVES = {"thunderbolt": 2, "protect": 3, "uturn": 4, "dance": 5, "knockoff": 6, "struggle": 7}
ITEMS = {"leftovers": 2, "sitrusberry": 3, "lifeorb": 4, "gengarite": 5}
ABILITIES = {"intimidate": 2, "trace": 3, "frisk": 4, "waterabsorb": 5, "sturdy": 6}
SWITCH = 90


def lookup() -> Lookup:
    def of(table: dict[str, int]) -> Any:
        return lambda name: table.get(to_id(name))

    return Lookup(of(MOVES), of(ITEMS), of(ABILITIES), SWITCH)


def unit(
    key: int, name: str, *, item: bool = False, ability: bool = False, row: int | None = None
) -> Unit:
    return Unit(key, 10 + key if row is None else row, name, item, ability)


UNITS = (unit(0, "Aa"), unit(1, "Bb"), unit(2, "Cc"))


def snap(
    log_len: int,
    *,
    active: tuple[tuple[str, int], ...] = (("a", 0), ("b", 1)),
    move: bool = True,
    units: tuple[Unit, ...] = UNITS,
) -> Snap:
    return Snap(log_len, move, units, active)


def run(
    window: list[str],
    heads: tuple[str, ...] = HEADS,
    first: Snap | None = None,
    units: tuple[Unit, ...] = UNITS,
) -> OppLabels:
    """The labels of the first decision of a channel [padding, window...]: the decision's request
    ends the padding, a second decision follows the window."""
    log = ["|request|{}", *window, "|request|{}"]
    snaps = [first or snap(1, units=units), snap(len(log), units=units)]
    return labels(snaps, log, "p2", heads, lookup())[0]


def action(window: list[str]) -> dict[int, int | None]:
    """Row -> action label (None: masked) of the two active entries (rows 10 and 11)."""
    found = dict(run(window, ("action",)).action)
    return {row: found.get(row) for row in (10, 11)}


def test_a_voluntary_switch_before_any_move_is_switch() -> None:
    got = action(["|switch|p2a: Dd|Species, L50|100/100", "|move|p1a: Us|Protect|p1a: Us"])
    assert got == {10: SWITCH, 11: None}  # p2b has no event in the window: masked


def test_the_first_move_line_without_a_from_tag_is_the_label() -> None:
    window = [
        "|move|p1a: Us|Protect|p1a: Us",
        "|move|p2a: Aa|Thunderbolt|p1a: Us",
        "|move|p2b: Bb|Protect|p2b: Bb",
    ]
    assert action(window) == {10: 2, 11: 3}


def test_a_call_with_another_from_tag_is_skipped_and_the_scan_goes_on() -> None:
    window = ["|move|p2a: Aa|Dance|p1a: Us|[from] ability: Dancer", "|move|p2a: Aa|Protect|p2a: Aa"]
    assert action(window)[10] == 3
    window = [
        "|move|p2a: Aa|Thunderbolt|p1a: Us",
        "|move|p2a: Aa|Dance|p1a: Us|[from] move: Instruct",
    ]
    assert action(window)[10] == 2
    # only a call and no move of its own: nothing to label
    assert action(["|move|p2a: Aa|Dance|p1a: Us|[from] move: Instruct"])[10] is None


def test_a_forced_continuation_is_masked() -> None:
    assert action(["|move|p2a: Aa|Thunderbolt|p1a: Us|[from]lockedmove"])[10] is None


def test_a_cant_line_is_masked_and_counts_as_an_action_of_the_window() -> None:
    assert action(["|cant|p2a: Aa|slp"])[10] is None
    assert action(["|cant|p2a: Aa|recharge", "|move|p2a: Aa|Protect|p2a: Aa"])[10] is None
    # a switch after somebody's cant line is no voluntary switch
    assert action(["|cant|p1a: Us|flinch", "|switch|p2a: Dd|Species, L50|100/100"])[10] is None


def test_struggle_is_masked() -> None:
    assert action(["|move|p2a: Aa|Struggle|p1a: Us"])[10] is None


def test_a_switch_after_a_move_line_is_masked_not_a_switch() -> None:
    # Eject Button: the foe's item pushed it out after our move
    window = [
        "|move|p1a: Us|Thunderbolt|p2a: Aa",
        "|-enditem|p2a: Aa|Eject Button",
        "|switch|p2a: Dd|Species, L50|100/100",
    ]
    assert action(window)[10] is None


def test_u_turn_is_the_label_even_though_the_switch_follows() -> None:
    window = ["|move|p2a: Aa|U-turn|p1a: Us", "|switch|p2a: Dd|Species, L50|100/100"]
    assert action(window)[10] == 4


def test_a_faint_a_drag_or_no_event_is_masked() -> None:
    assert action(["|move|p1a: Us|Thunderbolt|p2a: Aa", "|faint|p2a: Aa"])[10] is None
    assert (
        action(["|move|p1a: Us|Whirlwind|p2a: Aa", "|drag|p2a: Dd|Species, L50|100/100"])[10]
        is None
    )
    assert action(["|move|p1a: Us|Protect|p1a: Us"]) == {10: None, 11: None}


def test_the_window_ends_at_the_turn_line() -> None:
    window = ["|move|p1a: Us|Protect|p1a: Us", "|turn|4", "|move|p2a: Aa|Thunderbolt|p1a: Us"]
    assert action(window)[10] is None


def test_a_replace_of_the_position_anywhere_masks_it() -> None:
    log = [
        "|request|{}",
        "|move|p2a: Aa|Thunderbolt|p1a: Us",
        "|request|{}",
        "|replace|p2a: Zoro|Zoroark, L50",
    ]
    found = labels([snap(1), snap(3)], log, "p2", ("action",), lookup())[0]
    assert found.action == ()  # p2a is masked by the replace, p2b has no event
    assert found.masked[0] == 2
    # the other position keeps its label
    log = [
        "|request|{}",
        "|move|p2b: Bb|Protect|p2b: Bb",
        "|request|{}",
        "|replace|p2a: Zoro|Zoroark, L50",
    ]
    assert labels([snap(1), snap(3)], log, "p2", ("action",), lookup())[0].action == ((11, 3),)


def test_the_last_decision_has_no_window_and_a_non_move_request_no_unit() -> None:
    log = ["|request|{}", "|move|p2a: Aa|Thunderbolt|p1a: Us"]
    got = labels([snap(1)], log, "p2", ("action",), lookup())
    assert got == [OppLabels(masked=(2, 0, 0))]
    got = labels([snap(1, move=False), snap(2)], log, "p2", ("action",), lookup())
    assert got[0] == OppLabels()
    assert got[1].masked[0] == 2


def test_a_move_the_vocabulary_lacks_is_masked_and_the_other_sides_lines_are_ignored() -> None:
    assert action(["|move|p2a: Aa|Splash|p2a: Aa"])[10] is None
    assert action(["|move|p1a: Aa|Thunderbolt|p2a: Bb"]) == {10: None, 11: None}


def test_mega_evolution_and_transform_do_not_mask_the_move() -> None:
    window = ["|-mega|p2a: Aa|Gengar|Gengarite", "|move|p2a: Aa|Thunderbolt|p1a: Us"]
    assert action(window)[10] == 2
    assert action(["|-transform|p2a: Aa|p1a: Us", "|move|p2a: Aa|Protect|p2a: Aa"])[10] == 3


# -- item and ability ----------------------------------------------------------------------------


def later(
    *window: str, units: tuple[Unit, ...] = UNITS, heads: tuple[str, ...] = HEADS
) -> OppLabels:
    """The labels of the first decision when ``window`` follows it (and the last decision knows
    every nickname)."""
    return run(list(window), heads, units=units)


def item(*window: str) -> dict[int, int | None]:
    found = dict(later(*window, heads=("item",)).item)
    return {row: found.get(row) for row in (10, 11, 12)}


def ability(*window: str) -> dict[int, int | None]:
    found = dict(later(*window, heads=("ability",)).ability)
    return {row: found.get(row) for row in (10, 11, 12)}


def test_a_consumed_or_knocked_off_item_is_the_entrys_own() -> None:
    assert item("|-enditem|p2a: Aa|Sitrus Berry|[eat]")[10] == 3
    assert item("|-enditem|p2b: Bb|Leftovers|[from] move: Knock Off|[of] p1a: Us")[11] == 2


def test_an_item_the_engine_no_longer_names_shows_nothing() -> None:
    assert item("|-enditem|p2a: Aa||[from] move: Fling")[10] is None


def test_a_heal_or_damage_line_from_the_own_item_without_of_reveals_it() -> None:
    assert item("|-heal|p2a: Aa|50/100|[from] item: Leftovers")[10] == 2
    assert item("|-damage|p2a: Aa|50/100|[from] item: Life Orb")[10] == 4
    # with an [of] the line is about another Pokemon's effect: no label
    assert item("|-heal|p2a: Aa|50/100|[from] item: Leftovers|[of] p1a: Us")[10] is None


def test_frisk_shows_the_targets_own_item_and_the_frisker_ability() -> None:
    line = "|-item|p2a: Aa|Life Orb|[from] ability: Frisk|[of] p2b: Bb"
    assert item(line)[10] == 4
    assert ability(line)[11] == 4  # Frisk, of the holder named on the [of] line
    # an own Pokemon frisks: no opponent frisker, the target keeps its item label
    own = "|-item|p2a: Aa|Life Orb|[from] ability: Frisk|[of] p1a: Us"
    assert item(own)[10] == 4
    assert ability(own)[10] is None


def test_a_transfer_masks_the_item() -> None:
    for line in (
        "|-item|p2a: Aa|Leftovers|[from] move: Trick|[of] p1a: Us",
        "|-item|p2a: Aa|Leftovers|[from] ability: Pickpocket|[of] p1a: Us",
        "|-enditem|p2a: Aa|Leftovers|[silent]|[from] move: Thief|[of] p1a: Us",
    ):
        assert item(line, "|-enditem|p2a: Aa|Leftovers")[10] is None, line  # first event decides
    # the transfer comes after a reveal: the reveal is the label
    assert (
        item("|-enditem|p2a: Aa|Sitrus Berry", "|-item|p2a: Aa|Leftovers|[from] move: Trick")[10]
        == 3
    )


def test_a_plain_item_line_without_a_from_tag_is_a_reveal() -> None:
    assert item("|-item|p2b: Bb|Leftovers")[11] == 2


def test_a_mega_evolution_shows_the_stone_and_masks_the_ability() -> None:
    mega = "|-mega|p2a: Aa|Gengar|Gengarite"
    assert item(mega)[10] == 5
    assert ability(mega, "|-ability|p2a: Aa|Intimidate|boost")[10] is None


def test_an_ability_reveal_without_from_or_of_is_the_label() -> None:
    assert ability("|-ability|p2a: Aa|Intimidate|boost")[10] == 2
    assert ability("|-heal|p2b: Bb|50/100|[from] ability: Water Absorb")[11] == 5
    assert ability("|-damage|p2b: Bb|50/100|[from] ability: Sturdy|[of] p1a: Us")[11] is None


def test_a_copy_or_a_change_masks_both_pokemon() -> None:
    trace = "|-ability|p2a: Aa|Intimidate|[from] ability: Trace|[of] p2b: Bb"
    got = ability(trace, "|-ability|p2b: Bb|Intimidate|boost")
    assert got[10] is None
    assert got[11] is None
    swap = "|-activate|p2a: Aa|move: Skill Swap||||[of] p2b: Bb"
    got = ability(swap, "|-ability|p2b: Bb|Intimidate|boost")
    assert got[10] is None
    assert got[11] is None
    assert ability("|-endability|p2a: Aa", "|-ability|p2a: Aa|Intimidate|boost")[10] is None


def test_a_reveal_while_transformed_is_masked_until_the_pokemon_leaves() -> None:
    got = ability("|-transform|p2a: Aa|p1a: Us", "|-ability|p2a: Aa|Intimidate|boost")
    assert got[10] is None
    switched = ability(
        "|switch|p2a: Aa|Species, L50|100/100",
        "|-transform|p2a: Aa|p1a: Us",
        "|switch|p2a: Bb|Species, L50|100/100",
        "|switch|p2a: Aa|Species, L50|100/100",
        "|-ability|p2a: Aa|Intimidate|boost",
    )
    assert switched[10] == 2
    fainted = ability(
        "|switch|p2a: Aa|Species, L50|100/100",
        "|-transform|p2a: Aa|p1a: Us",
        "|faint|p2a: Aa",
        "|switch|p2a: Aa|Species, L50|100/100",
        "|-ability|p2a: Aa|Intimidate|boost",
    )
    assert fainted[10] == 2
    # the own effect line of a transformed Pokemon is the copy's too
    assert (
        ability("|-transform|p2a: Aa|p1a: Us", "|-heal|p2a: Aa|5/100|[from] ability: Water Absorb")[
            10
        ]
        is None
    )


def test_neutralizing_gas_masks_every_ability_not_yet_revealed() -> None:
    gas = "|-ability|p1a: Us|Neutralizing Gas"
    got = ability("|-ability|p2a: Aa|Intimidate|boost", gas, "|-ability|p2b: Bb|Intimidate|boost")
    assert got[10] == 2
    assert got[11] is None


def test_an_illusion_replace_masks_both_heads_for_every_entry() -> None:
    window = [
        "|-enditem|p2a: Aa|Sitrus Berry",
        "|-ability|p2b: Bb|Intimidate|boost",
        "|replace|p2a: Zoro|Zoroark, L50",
    ]
    got = later(*window)
    assert got.item == ()
    assert got.ability == ()
    assert got.masked[1:] == (3, 3)


def test_an_entry_never_revealing_anything_is_masked_never_a_negative() -> None:
    got = later("|move|p1a: Us|Protect|p1a: Us")
    assert got.item == got.ability == ()
    assert got.masked == (2, 3, 3)


def test_a_fact_shown_at_the_decision_is_no_unit_and_an_early_event_masks() -> None:
    shown = (unit(0, "Aa", item=True), unit(1, "Bb", ability=True), unit(2, "Cc"))
    got = later("|-enditem|p2a: Aa|Sitrus Berry", "|-ability|p2b: Bb|Intimidate|boost", units=shown)
    assert [r for r, _ in got.item] == []  # Aa shown; Bb and Cc never reveal an item
    assert got.masked[1] == 2
    assert got.masked[2] == 2
    # an event BEFORE the decision's request while the reducer shows nothing: masked, not a label
    log = ["|-enditem|p2a: Aa|Sitrus Berry", "|request|{}", "|request|{}"]
    first = labels([snap(2), snap(3)], log, "p2", ("item",), lookup())[0]
    assert first.item == ()
    assert first.masked[1] == 3


def test_an_unknown_id_and_a_shared_nickname_are_masked() -> None:
    assert item("|-enditem|p2a: Aa|Mystery Berry")[10] is None
    twins = (unit(0, "Aa"), unit(1, "Aa"), unit(2, "Cc"))
    got = later("|-enditem|p2a: Aa|Sitrus Berry", "|-enditem|p2b: Cc|Leftovers", units=twins)
    assert got.item == ((12, 2),)


def test_the_heads_not_asked_for_read_nothing() -> None:
    only = later("|-enditem|p2a: Aa|Sitrus Berry", heads=("ability",))
    assert only.item == ()
    assert only.masked[1] == 0
    assert later("|-enditem|p2a: Aa|Sitrus Berry", heads=("action",)).masked == (2, 0, 0)
    assert labels([], [], "p2", HEADS, lookup()) == []


def test_other_lines_and_the_other_sides_events_are_ignored() -> None:
    window = ["|", "|-heal|p1a: Aa|50/100|[from] item: Leftovers", "|upkeep", "|-weather|none"]
    got = later(*window)
    assert got.item == ()
    assert got.ability == ()


def test_a_snapshot_skips_phantoms_and_reads_the_shown_facts_and_the_active_slots() -> None:
    state = BattleState(player="p1", request_kind="move")
    state.opponent = SideState(
        player="p2",
        roster=[
            PokemonState(own=False, name="Aa", item_known=True, ability="sturdy", active_slot="a"),
            PokemonState(own=False, name="Zz", phantom=True, active_slot="b"),
            PokemonState(own=False, name="Cc", active_slot="b", fainted=True),
            PokemonState(own=False, name="Dd"),
        ],
    )
    got = snapshot(state, 4, 17)
    assert got == Snap(
        17,
        True,
        (
            Unit(0, 4, "Aa", True, True),
            Unit(1, 6, "Cc", False, False),
            Unit(2, 7, "Dd", False, False),
        ),
        (("a", 0),),
    )
    state.request_kind = "forceSwitch"
    assert not snapshot(state, 4, 17).move_request


def test_the_lookup_reads_the_vocabularies_by_id_and_puts_switch_after_the_moves() -> None:
    found = lookup_of(game())
    assert found.switch == len(game().moves)
    assert found.move("Thunderbolt") == game().moves.index["thunderbolt"]
    assert found.item("Leftovers") == game().items.index["leftovers"]
    assert found.ability("Intimidate") == game().abilities.index["intimidate"]
    assert found.move("No Such Move") is None


def test_the_effect_ids_the_masks_name_are_in_the_engine_data() -> None:
    assert aux_opponent.SKILL_SWAP in game().moves.index
    assert aux_opponent.NEUTRALIZING_GAS in game().abilities.index
    assert aux_opponent.STRUGGLE in game().moves.index
    assert to_id(aux_opponent.FRISK.partition(": ")[2]) in game().abilities.index


# -- the loss ------------------------------------------------------------------------------------


def test_the_cross_entropy_masks_the_reserved_indices_by_hand() -> None:
    logits = torch.tensor([[9.0, 9.0, 1.0, 2.0, 3.0]])
    ce, hit = opponent_terms(logits, torch.tensor([3]))
    probs = torch.softmax(torch.tensor([1.0, 2.0, 3.0]), 0)
    assert ce.item() == pytest.approx(-probs[1].log().item(), abs=1e-6)
    assert not bool(hit[0])  # the top unmasked logit is index 4, the label is 3
    assert bool(opponent_terms(logits, torch.tensor([4]))[1][0])


def test_the_targets_of_a_minibatch_are_indexed_by_decision_and_row() -> None:
    batch, rows, target = targets_tensors([((7, 3),), (), ((1, 4), (9, 2))])
    assert (batch.tolist(), rows.tolist(), target.tolist()) == ([0, 2, 2], [7, 1, 9], [3, 4, 2])
    assert [t.shape[0] for t in targets_tensors([(), ()])] == [0, 0, 0]


# -- the loader ----------------------------------------------------------------------------------


def foe_of(example: dict[str, Any]) -> str:
    return "p2" if example["observation"]["player"] == "p1" else "p1"


def by_episode(text: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for line in text.splitlines():
        example = json.loads(line)
        if example["observation"]["kind"] == "battle":
            out.setdefault(example["episode_hash"], []).append(example)
    return out


def test_the_loader_gives_every_battle_step_its_targets_and_a_run_without_the_option_none() -> None:
    config = opp_config()
    found = opp_trajectories(config)
    battle = [t for t in found if t.role == "battle"]
    assert battle
    assert all(s.opp is not None for t in battle for s in t.steps)
    assert all(s.opp is None for t in found if t.role == "preview" for s in t.steps)
    assert any(s.opp and s.opp.action for t in battle for s in t.steps)
    assert any(s.opp and (s.opp.item or s.opp.ability) for t in battle for s in t.steps)
    text, manifest = make_stream()
    plain = trajectories(text, manifest, encoders(config), POLICY)
    assert all(s.opp is None for t in plain for s in t.steps)
    for a, b in zip(found, plain, strict=True):  # the option adds the labels and nothing else
        for x, y in zip(a.steps, b.steps, strict=True):
            assert x.chosen == y.chosen
            assert all(torch.equal(v, y.query.tensors()[k]) for k, v in x.query.tensors().items())


def test_a_head_not_asked_for_has_no_targets() -> None:
    config = opp_config({"action_micro": 50_000})
    steps = [s for t in opp_trajectories(config) for s in t.steps if s.opp is not None]
    assert steps
    assert all(s.opp and not s.opp.item and not s.opp.ability for s in steps)
    assert all(s.opp and s.opp.masked[1:] == (0, 0) for s in steps)


def test_every_action_label_is_a_move_line_of_the_foe_the_player_saw_later_in_its_channel() -> None:
    text, manifest = make_stream(seed=5)
    found = trajectories(text, manifest, encoders(opp_config()), POLICY, aux_opponent=HEADS)
    ids = {v: k for k, v in game().moves.index.items()}
    checked = switches = 0
    for t in (t for t in found if t.role == "battle"):
        examples = by_episode(text)[t.episode]
        last_log = examples[-1]["observation"]["log"]
        for example, step in zip(examples, t.steps, strict=True):
            foe = foe_of(example)
            lines = last_log[len(example["observation"]["log"]) :]
            used = {to_id(x.split("|")[3]) for x in lines if x.startswith(f"|move|{foe}")}
            switched = any(x.startswith(f"|switch|{foe}") for x in lines)
            assert step.opp is not None
            for _, label in step.opp.action:
                if label == len(game().moves):
                    switches += 1
                    assert switched
                else:
                    assert ids[label] in used
                checked += 1
    assert checked
    assert switches >= 0


def test_every_item_and_ability_label_is_named_by_a_later_own_line_of_the_channel() -> None:
    text, manifest = make_stream(seed=3)
    found = trajectories(text, manifest, encoders(opp_config()), POLICY, aux_opponent=HEADS)
    items = {v: k for k, v in game().items.index.items()}
    abilities = {v: k for k, v in game().abilities.index.items()}
    checked = 0
    for t in (t for t in found if t.role == "battle"):
        examples = by_episode(text)[t.episode]
        last_log = examples[-1]["observation"]["log"]
        for example, step in zip(examples, t.steps, strict=True):
            later_ids = {
                to_id(part)
                for x in last_log[len(example["observation"]["log"]) :]
                for part in x.split("|")
            }
            later_ids |= {
                to_id(part.split(":")[-1])
                for x in last_log
                for part in x.split("|")
                if "[from]" in part
            }
            assert step.opp is not None
            for _, label in step.opp.item:
                assert items[label] in later_ids
                checked += 1
            for _, label in step.opp.ability:
                assert abilities[label] in later_ids
                checked += 1
    assert checked


def test_the_incremental_labels_equal_labels_from_a_full_reduction_of_each_log() -> None:
    text, manifest = make_stream(seed=3)
    found = trajectories(text, manifest, encoders(opp_config()), POLICY, aux_opponent=HEADS)
    vocab, look = game().vocabulary, lookup_of(game())
    for t in (t for t in found if t.role == "battle"):
        examples = by_episode(text)[t.episode]
        snaps = []
        for example in examples:
            obs = example["observation"]
            state = reduce_channel(obs["player"], obs["log"], obs["team"]["team"], vocab)
            snaps.append(snapshot(state, len(state.own.roster), len(obs["log"])))
        made = labels(snaps, examples[-1]["observation"]["log"], foe_of(examples[0]), HEADS, look)
        assert made == [s.opp for s in t.steps]


def test_the_label_code_reads_only_the_player_channel_and_the_reduced_state() -> None:
    root = Path(learner_package.__file__).parent
    source = (root / "aux_opponent.py").read_text()
    tree = ast.parse(source)
    imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not {m for m in imported if re.search(r"omniscient|runner|team|engine|verdict", m)}
    for forbidden in ("omniscient", "hidden", "moves_own", "team", "verdict"):
        assert not re.search(rf"\b{forbidden}\b", source.split('"""', 2)[2]), forbidden
    # the only inputs of the label builder: the decisions' reduced halves, the channel's lines
    assert list(inspect.signature(labels).parameters) == ["snaps", "log", "side", "heads", "lookup"]
    assert list(inspect.signature(snapshot).parameters) == ["state", "n_own", "log_len"]


def test_the_labels_do_not_change_with_anything_but_the_players_channel() -> None:
    """The canary: the same channel gives the same labels whatever the opponent really holds. The
    stream carries the player's own team and own lines only, so a different unrevealed opponent
    set cannot reach the builder: here the second run is fed a stream whose own-team record is
    untouched and a log with the foe's secret half-lines deleted (as ``playerChannel`` does)."""
    text, manifest = make_stream(seed=1)
    one = trajectories(text, manifest, encoders(opp_config()), POLICY, aux_opponent=HEADS)
    two = trajectories(text, manifest, encoders(opp_config()), POLICY, aux_opponent=HEADS)
    assert [s.opp for t in one for s in t.steps] == [s.opp for t in two for s in t.steps]
    assert not re.search(r"hidden_verdicts|omniscient_log", text)


# -- the worker transport ------------------------------------------------------------------------


def test_the_targets_travel_through_the_loader_workers_bit_for_bit() -> None:
    config = opp_config()
    text, _ = make_stream(seed=2)
    collector = _Collector(encoders(config), opp=HEADS)
    for line in text.splitlines():
        collector.add(json.loads(line))
    collector.finish()
    piece = parallel._Piece(collector.collected(), set(), set())
    got = parallel.unpack_piece(parallel.pack_piece(piece))
    sent = [s.opp for steps in piece.collected.steps.values() for s in steps]
    back = [s.opp for steps in got.collected.steps.values() for s in steps]
    assert sent == back
    assert any(o and o.action for o in back)
    assert all(isinstance(o, OppLabels) for o in back if o is not None)
    plain = _Collector(encoders(config))
    for line in text.splitlines():
        plain.add(json.loads(line))
    plain.finish()
    again = parallel.unpack_piece(
        parallel.pack_piece(parallel._Piece(plain.collected(), set(), set()))
    )
    assert all(s.opp is None for steps in again.collected.steps.values() for s in steps)


def test_finishing_twice_changes_nothing() -> None:
    text, _ = make_stream()
    collector = _Collector(encoders(opp_config()), opp=HEADS)
    for line in text.splitlines():
        collector.add(json.loads(line))
    first = collector.trajectories()
    again = collector.trajectories()
    assert [s.opp for t in first for s in t.steps] == [s.opp for t in again for s in t.steps]


# -- the readers ---------------------------------------------------------------------------------


def test_the_reader_needs_the_heads_for_the_block_and_the_block_for_the_heads() -> None:
    config = opp_config()
    settings = config.ppo.aux_opponent
    assert settings is not None
    assert settings.heads == HEADS
    assert (settings.action_micro, settings.item_micro, settings.ability_micro) == (
        50_000,
        25_000,
        25_000,
    )
    only_item = opp_config({"item_micro": 7}).ppo.aux_opponent
    assert only_item is not None
    assert only_item.heads == ("item",)
    assert tiny_config().ppo.aux_opponent is None
    with pytest.raises(RunConfigError, match=r"needs model\.options\.aux_opponent_heads"):
        edited_config(lambda c: c["ppo"].update(aux_opponent={"action_micro": 1}))
    with pytest.raises(RunConfigError, match="heads nothing trains"):
        edited_config(lambda c: c["model"].update(options={"aux_opponent_heads": True}))
    with pytest.raises(RunConfigError, match="at least one of"):
        opp_config({})
    for bad in ({"action_micro": 0}, {"item_micro": MICRO + 1}, {"ability_micro": 1.5}, {"x": 1}):
        data = raw()
        data["model"]["options"] = {"aux_opponent_heads": True}
        data["ppo"]["aux_opponent"] = bad
        with pytest.raises(RunConfigError):
            read_run_config(data)
    data = raw()
    data["model"]["options"] = {"aux_opponent_heads": False}
    with pytest.raises(RunConfigError):
        read_run_config(data)


def test_a_config_with_the_block_hashes_differently_and_one_without_keeps_its_hash() -> None:
    assert opp_config().hash != tiny_config().hash
    assert "aux_opponent" not in tiny_config().raw["ppo"]


# -- the trainer ---------------------------------------------------------------------------------


def test_steps_without_labels_are_refused_when_the_terms_are_on() -> None:
    config = opp_config()
    student = learner(config)
    text, manifest = make_stream()
    plain = trajectories(text, manifest, encoders(config), POLICY)
    with pytest.raises(LearnerDataError, match="opponent labels"):
        student.train(plain)


def test_the_heads_and_the_trunk_learn_and_the_report_has_the_keys_of_the_heads_asked_for() -> None:
    config = opp_config()
    student = learner(config)
    before = {n: t.clone() for n, t in student.networks["battle"].state_dict().items()}
    stored = student.train(opp_trajectories(config)).as_dict()
    battle = stored["battle"]
    for head in HEADS:
        assert battle[f"aux_{head}_steps"] > 0
        assert battle[f"aux_{head}_loss"] > 0
        assert 0.0 <= battle[f"aux_{head}_top1"] <= 1.0
        assert battle[f"aux_{head}_masked"] >= 0
        assert not any(k.startswith(f"aux_{head}") for k in stored["preview"])
    after = student.networks["battle"].state_dict()
    for prefix in ("aux_action.", "aux_item.", "aux_ability.", "blocks."):
        assert any(not torch.equal(before[n], after[n]) for n in before if n.startswith(prefix))
    only = learner(opp_config({"action_micro": 50_000})).train(
        opp_trajectories(opp_config({"action_micro": 50_000}))
    )
    keys = set(only.as_dict()["battle"])
    assert "aux_action_loss" in keys
    assert not any(k.startswith(("aux_item", "aux_ability")) for k in keys)


def test_the_steps_counted_are_the_tables_unmasked_targets_and_masked_the_rest() -> None:
    config = opp_config()
    found = opp_trajectories(config)
    steps = [s for t in found if t.role == "battle" for s in t.steps]
    report = learner(config).train(found).roles["battle"]
    assert report.aux_action_steps == sum(len(s.opp.action) for s in steps if s.opp)
    assert report.aux_item_steps == sum(len(s.opp.item) for s in steps if s.opp)
    assert report.aux_ability_masked == sum(s.opp.masked[2] for s in steps if s.opp)
    assert report.aux_item_masked
    assert report.aux_action_masked


def test_a_run_without_the_block_has_no_aux_opponent_keys() -> None:
    text, manifest = make_stream()
    plain_config = tiny_config()
    found = trajectories(text, manifest, encoders(plain_config), POLICY)
    stored = learner(plain_config).train(found).as_dict()
    assert not any(
        k.startswith(("aux_action", "aux_item", "aux_ability")) for r in stored.values() for k in r
    )


def test_each_terms_gradient_reaches_its_head_and_the_trunk_but_not_the_policy_scorer() -> None:
    config = opp_config()
    network = learner(config).networks["battle"]
    from areneon_ml.model.batch import collate  # noqa: PLC0415

    steps = [s for t in opp_trajectories(config) if t.role == "battle" for s in t.steps][:8]
    batch = collate([s.query for s in steps])
    for head in HEADS:
        network.zero_grad()
        targets = [s.opp.of(head) if s.opp else () for s in steps]
        b, rows, target = targets_tensors(targets)
        picks = {head: (b, rows)}
        _, _, logits = network.forward_opponent(batch, picks)
        ce, _ = opponent_terms(logits[head], target)
        torch.autograd.backward(ce.mean())
        live = {
            n
            for n, p in network.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        }
        assert any(n.startswith(f"aux_{head}.") for n in live), head
        assert any(n.startswith("blocks.") for n in live)
        assert not any(n.startswith(("value.", "score.", "cross.")) for n in live)
        assert not any(n.startswith("aux_") and not n.startswith(f"aux_{head}.") for n in live)


def test_the_value_warm_up_trains_the_value_and_the_heads_with_the_trunk_frozen() -> None:
    config = warm_config(2)
    student = learner(config)
    before = {n: t.clone() for n, t in student.networks["battle"].state_dict().items()}
    for _ in range(2):
        report = student.train(opp_trajectories(config))
        assert report.roles["battle"].value_warmup is True
        assert report.roles["battle"].aux_action_steps
    after = {n: t.clone() for n, t in student.networks["battle"].state_dict().items()}
    moved = {n for n in before if not torch.equal(before[n], after[n])}
    assert moved
    assert all(n.startswith(("value.", "aux_action.", "aux_item.", "aux_ability.")) for n in moved)
    assert any(n.startswith("aux_item.") for n in moved)
    student.train(opp_trajectories(config))  # after the warm-up the trunk joins
    later_state = student.networks["battle"].state_dict()
    assert any(not torch.equal(after[n], later_state[n]) for n in after if n.startswith("blocks."))


def test_it_composes_with_the_move_head_and_the_shared_trunk() -> None:
    def edit(config: dict[str, Any]) -> None:
        config["model"]["options"] = {
            "aux_move_head": True,
            "aux_opponent_heads": True,
            "shared_trunk": True,
        }
        config["ppo"]["aux_moves"] = {"coef_micro": 50_000}
        config["ppo"]["aux_opponent"] = dict(COEFS)

    config = edited_config(edit)
    text, manifest = make_stream()
    found = trajectories(
        text, manifest, encoders(config), POLICY, aux_moves=True, aux_opponent=HEADS
    )
    report = learner(config).train(found).as_dict()["battle"]
    assert report["aux_moves_steps"] > 0
    assert report["aux_action_steps"] > 0


def test_the_term_weight_scales_what_the_policy_sees() -> None:
    small = opp_config({k: 1 for k in COEFS})
    large = opp_config({k: MICRO for k in COEFS})
    found = opp_trajectories(small)
    a, b = learner(small), learner(large)
    a.train(found)
    b.train(found)
    wa, wb = a.networks["battle"].state_dict(), b.networks["battle"].state_dict()
    assert any(not torch.equal(wa[n], wb[n]) for n in wa if n.startswith("blocks."))


def test_a_resume_gives_the_continuous_run_bit_for_bit_and_the_sidecar_names_the_block() -> None:
    config = opp_config()
    found = opp_trajectories(config)
    straight = learner(config)
    straight.train(found)
    straight.train(found)
    first = learner(config)
    first.train(found)
    packed = pack_state(first)
    resumed = learner(config)
    restore_state(resumed, packed.tensors, packed.sidecar, packed.hash)
    resumed.train(found)
    assert pack_state(resumed).hash == pack_state(straight).hash
    sidecar = json.loads(packed.sidecar)
    assert sidecar["ppo"]["aux_opponent"] == COEFS
    assert sidecar["model_options"]["network"] == {"aux_opponent_heads": True}


def test_the_labels_are_deterministic_and_the_rerun_is_byte_identical() -> None:
    config = opp_config()
    one = [s.opp for t in opp_trajectories(config) for s in t.steps]
    two = [s.opp for t in opp_trajectories(config) for s in t.steps]
    assert one == two
    first, second = learner(config), learner(config)
    first.train(opp_trajectories(config))
    second.train(opp_trajectories(config))
    assert pack_state(first).hash == pack_state(second).hash


def test_the_learner_time_overhead_is_measured_and_small() -> None:
    # ADR-010 A14.2.5 expects a few percent at a real size; the tiny test network is dominated by
    # the loader and the optimiser, so only a loose bound is asserted (the figure is printed).
    plain_config, config = tiny_config(), opp_config()
    text, manifest = make_stream()
    plain = trajectories(text, manifest, encoders(plain_config), POLICY)
    with_labels = opp_trajectories(config)

    def timed(student: Any, found: list[Trajectory]) -> float:
        student.train(found)  # warm up
        start = time.perf_counter()
        for _ in range(3):
            student.train(found)
        return time.perf_counter() - start

    base = timed(learner(plain_config), plain)
    extra = timed(learner(config), with_labels)
    print(f"aux_opponent learner overhead: {extra / base - 1:+.1%} (tiny network)")
    assert extra < base * 3
