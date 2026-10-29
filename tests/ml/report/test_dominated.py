"""The statically dominated damaging-move rate (#1025, ADR-022 M2): hand-built games, the page."""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import areneon_ml.report.dominated as module
import pytest
from areneon_ml.learner.config import read_run_config
from areneon_ml.loop import layout
from areneon_ml.report.dominated import (
    INTERVAL_LABEL,
    LABEL,
    DominatedError,
    DominatedTable,
    committed_dominated_table,
    config_file,
    dominated_rates,
    game_dominated,
    interval_of,
    progress_diagnostics,
    progress_dominated,
    read_dominated_table,
)
from areneon_ml.report.model import read_report_model
from areneon_ml.report.move_use import (
    StoredGame,
    class_files,
    committed_move_classes,
    move_use,
    progress_move_use,
    stored_games,
)
from areneon_ml.report.render import build_report

from tests.ml.report.support import make_run, progress_root
from tests.ml.report.test_move_use import (
    A_OPTIONS,
    B_OPTIONS,
    HEAD,
    P2_OPTIONS,
    PREVIEW,
    TURN_1,
    decision,
    slot_options,
)
from tests.ml.report.test_move_use import (
    game as shared_game,
)
from tests.ml.report.test_report import HOSTILE, assert_inert

FORMAT = "gen9championsvgc2026regmc"
_FOUND = committed_dominated_table(FORMAT)
assert _FOUND is not None
TABLE: DominatedTable = _FOUND
OFFERED = slot_options(A_OPTIONS, B_OPTIONS)


def game(*args: Any, **kwargs: Any) -> StoredGame:
    """The shared hand-built game with Pound, a move of the format, in place of Tackle."""
    stored = shared_game(*args, **kwargs)
    teams = json.loads(json.dumps(stored.record["teams"]).replace("Tackle", "Pound"))
    return dataclasses.replace(stored, record={**stored.record, "teams": teams})


def used(log: list[str], *picks: dict[str, Any], seat: str = "p1") -> dict[str, Any]:
    stored = game(log, [*PREVIEW, *picks])
    return game_dominated(stored, TABLE)[seat].as_dict()


def at_turn_1(*extra: str) -> list[str]:
    """The log up to the first move decision, with ``extra`` lines applied before it."""
    return [*HEAD, *extra, "|turn|1"]


def p1_pick(chosen: str, options: list[str] = OFFERED, seq: int = 2) -> dict[str, Any]:
    return decision("p1", seq, options, chosen)


def counts(seat: dict[str, Any]) -> dict[str, int]:
    keys = ("scored", "dominated", "immune", "status", "ally_target", "no_power", "unscored")
    return {k: seat[k] for k in (*keys, "unresolved")}


# Ann (Charizard, fire/flying) holds Flamethrower (fire 90, same type), Icy Wind (ice 55, spread
# over the foes); the foes are Eve (Venusaur, grass/poison) and Fay (Gengar, ghost/poison).
# Flamethrower: 90 * 1.5 = 135; on Venusaur x2 (grass) x1 (poison) = 270; on Gengar x1 x1 = 135.
# Icy Wind: 55; Venusaur x2 = 110, Gengar x1 = 55; summed 165 * 0.75 = 123.75.
# Best 270; the threshold is half of it, 135.


def test_the_best_move_on_its_best_target_is_not_dominated() -> None:
    seat = used(TURN_1, p1_pick("move 1 1, move 1"))
    # slot a: Flamethrower on Eve (270, the best); slot b is Tailwind, a status move
    assert (seat["scored"], seat["dominated"], seat["status"]) == (1, 0, 1)


def test_a_score_equal_to_the_threshold_is_not_dominated() -> None:
    # Flamethrower on Fay is 135, exactly half of 270: not below it
    seat = used(TURN_1, p1_pick("move 1 2, move 1"))
    assert (seat["scored"], seat["dominated"], seat["immune"]) == (1, 0, 0)


def test_a_spread_move_below_the_threshold_is_dominated() -> None:
    # Icy Wind (123.75) is below 135
    seat = used(TURN_1, p1_pick("move 4, move 1"))
    assert (seat["scored"], seat["dominated"], seat["immune"]) == (1, 1, 0)
    assert seat["status"] == 1  # slot b is Tailwind


def test_an_immune_target_is_dominated_and_counted_apart() -> None:
    # Cy (Pikachu) holds Pound only; Pound on Gengar (ghost) is 0 while on Venusaur it is 40
    log = [*TURN_1, "|faint|p1a: Ann", "|switch|p1a: Cy|Pikachu, L50|100/100", "|turn|2"]
    turn_1 = p1_pick("move 2, move 1")  # Protect; Tailwind
    forced = decision("p1", 3, ["switch 3, pass", "switch 4, pass"], "switch 3, pass")
    cy = ["move 1 1", "move 1 2", "move 2", "move 3", "move 4", "switch 3"]
    options = slot_options(cy, B_OPTIONS)
    seat = used(log, turn_1, forced, p1_pick("move 1 2, move 1", options, 4))
    assert (seat["scored"], seat["dominated"], seat["immune"]) == (1, 1, 1)
    again = used(log, turn_1, forced, p1_pick("move 1 1, move 1", options, 4))
    assert (again["scored"], again["dominated"], again["immune"]) == (1, 0, 0)


def test_a_spread_move_is_summed_over_the_alive_foes() -> None:
    # Bo (Blastoise) holds Surf (water 90, same type, over all adjacent) and Psychic (90, 180 on
    # either foe). With both foes alive Surf is (67.5 + 135) * 0.75 = 151.875 against 180: kept.
    # With only Eve alive it is 67.5 * 0.75 = 50.625, below the 90 threshold: dominated.
    both = used(TURN_1, p1_pick("move 1 1, move 3"))
    assert (both["scored"], both["dominated"]) == (2, 0)
    log = at_turn_1("|faint|p2b: Fay")
    alone = used(log, p1_pick("move 1 1, move 3"))
    assert (alone["scored"], alone["dominated"]) == (2, 1)


def test_the_slot_with_a_fainted_foe_in_it_has_no_score() -> None:
    log = at_turn_1("|faint|p2b: Fay")
    seat = used(log, p1_pick("move 1 2, move 1"))
    assert (seat["scored"], seat["unscored"]) == (0, 1)


def test_status_ally_target_and_unresolved_are_counted_apart() -> None:
    seat = used(TURN_1, p1_pick("move 3, move 1"))  # Swords Dance; Tailwind
    assert counts(seat) == {
        "scored": 0,
        "dominated": 0,
        "immune": 0,
        "status": 2,
        "ally_target": 0,
        "no_power": 0,
        "unscored": 0,
        "unresolved": 0,
    }
    ally = used(
        TURN_1, p1_pick("move 1 -2, move 1", slot_options([*A_OPTIONS, "move 1 -2"], B_OPTIONS))
    )
    assert (ally["ally_target"], ally["scored"]) == (1, 0)
    locked = slot_options(["move 1 1", "move 1 2"], B_OPTIONS)
    seat = used(TURN_1, p1_pick("move 1 1, move 1", locked))
    assert (seat["unresolved"], seat["scored"]) == (1, 0)


def test_a_switch_a_pass_and_an_empty_candidate_list_are_not_read() -> None:
    seat = used(TURN_1, p1_pick("switch 3, switch 3", slot_options(["switch 3"], ["switch 3"])))
    assert not any(seat[k] for k in ("scored", "status", "unresolved", "unscored"))
    wait = decision("p1", 2, [], "")
    stored = game(TURN_1, [*PREVIEW, wait])
    assert game_dominated(stored, TABLE)["p1"].scored == 0


def test_the_shown_species_follows_replace_swap_and_detailschange() -> None:
    # Ann becomes a Rock-type Mega-like forme: Flamethrower is then not the same type
    log = at_turn_1("|detailschange|p1a: Ann|Charizard-Mega-X, L50")
    seat = used(log, p1_pick("move 1 1, move 1"))
    assert seat["scored"] == 1  # the new forme is in the table (fire/dragon, still same type)
    malformed = at_turn_1("|detailschange|nobody|Pikachu, L50")  # no slot: changes nothing
    assert used(malformed, p1_pick("move 1 1, move 1"))["scored"] == 1
    unknown = at_turn_1("|detailschange|p1a: Ann|Notaspecies, L50")
    assert used(unknown, p1_pick("move 1 1, move 1"))["unscored"] == 1
    swapped = at_turn_1("|swap|p1b: Bo|0")
    seat = used(swapped, p1_pick("move 1 1, move 1", slot_options(B_OPTIONS, A_OPTIONS)))
    assert seat["scored"] == 0  # Bo's Tailwind is a status move... in slot a now
    assert seat["status"] == 1
    p2 = game_dominated(
        game(
            TURN_1,
            [
                *PREVIEW,
                decision("p2", 2, slot_options(P2_OPTIONS, P2_OPTIONS), "move 1 1, move 2 1"),
            ],
        ),
        TABLE,
    )["p2"].as_dict()
    assert p2["scored"] == 2


def test_a_move_the_table_lacks_and_a_missing_foe_are_counted_apart() -> None:
    # Splash is not a move of the format's engine-data table: it cannot be scored
    p2 = decision("p2", 2, slot_options(P2_OPTIONS, P2_OPTIONS), "move 4, move 4")
    seat = game_dominated(game(TURN_1, [*PREVIEW, p2]), TABLE)["p2"].as_dict()
    assert (seat["unscored"], seat["scored"]) == (2, 0)
    bare = ["|teampreview|4", "|start", "|switch|p1a: Ann|Charizard, L50|100/100", "|turn|1"]
    seat = used(bare, p1_pick("move 1 1, move 1"))
    assert seat["unscored"] == 1  # no foe is on the field


def with_ann(moves: list[str]) -> StoredGame:
    """TURN_1 with Ann holding ``moves``; p1 chooses her move 1 at Eve and Bo's Tailwind."""
    stored = game(TURN_1, [*PREVIEW, p1_pick("move 1 1, move 1")])
    teams = copy.deepcopy(stored.record["teams"])
    teams["p1"]["team"][0]["moves"] = moves
    return dataclasses.replace(stored, record={**stored.record, "teams": teams})


def test_a_random_target_move_is_the_mean_over_the_alive_foes() -> None:
    # Raging Fury (fire 120, same type, random target): 180 * 2 = 360 on Venusaur, 180 on Gengar,
    # mean 270. Flamethrower on Eve is 270 as well, so neither is dominated.
    stored = with_ann(["Raging Fury", "Flamethrower", "Protect", "Swords Dance"])
    seat = game_dominated(stored, TABLE)["p1"]
    assert (seat.scored, seat.dominated) == (1, 0)
    # a table that lists no random-target kind cannot score Raging Fury, and skips it as the best
    bare = dataclasses.replace(TABLE, random=frozenset())
    seat = game_dominated(stored, bare)["p1"]
    assert (seat.scored, seat.unscored) == (0, 1)
    # with Flamethrower chosen, Raging Fury is left out of the best
    other = with_ann(["Flamethrower", "Raging Fury", "Protect", "Swords Dance"])
    seat = game_dominated(other, bare)["p1"]
    assert (seat.scored, seat.dominated) == (1, 0)


def test_a_damaging_move_with_no_base_power_is_counted_apart() -> None:
    stored = with_ann(["Seismic Toss", "Flamethrower", "Protect", "Swords Dance"])
    seat = game_dominated(stored, TABLE)["p1"]
    assert (seat.no_power, seat.scored) == (1, 0)
    # the same move enabled but not chosen does not enter the best
    other = with_ann(["Flamethrower", "Seismic Toss", "Protect", "Swords Dance"])
    seat = game_dominated(other, TABLE)["p1"]
    assert (seat.scored, seat.dominated, seat.no_power) == (1, 0, 0)


# -- the aggregation, the intervals and the config record ---------------------------------------


def test_matches_split_candidate_and_opponent() -> None:
    pick = p1_pick("move 4, move 1")
    a = game(TURN_1, [*PREVIEW, pick], policies=("policy:" + "a" * 64, "damage-aware@v1"))
    b = game(
        TURN_1,
        [*PREVIEW, pick],
        policies=("damage-aware@v1", "policy:" + "a" * 64),
        candidate="p2",
        number=1,
    )
    found = dominated_rates([a, b], TABLE)
    (match,) = found["matches"]
    assert found["label"] == LABEL
    assert found["interval_label"] == INTERVAL_LABEL
    assert found["config_hash"] == TABLE.config_hash
    assert match["games"] == 2
    assert match["reference"] is True
    assert match["candidate"]["dominated"] == 1
    assert match["opponent"]["dominated"] == 1


def test_the_interval_is_binomial_and_empty_without_trials() -> None:
    assert interval_of(0, 0) is None
    found = interval_of(1, 2)
    assert found is not None
    assert found["ppm"] == 500_000
    assert found["low_ppm"] < 500_000 < found["high_ppm"]


def test_the_committed_config_is_pinned() -> None:
    assert TABLE.effectiveness == (Fraction(1), Fraction(2), Fraction(1, 2), Fraction(0))
    assert (TABLE.same_type, TABLE.spread, TABLE.threshold) == (
        Fraction(3, 2),
        Fraction(3, 4),
        Fraction(1, 2),
    )
    assert "allAdjacent" in TABLE.spread_kinds
    assert committed_dominated_table("gen9nonesuch") is None


def edited_tables(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    path, engine = tmp_path / "config.json", tmp_path / "engine-data.json"
    engine.write_bytes(class_files(FORMAT)[1].read_bytes())
    raw = json.loads(config_file(FORMAT).read_text())
    return path, engine, raw


def test_the_config_is_refused_unless_it_fits_the_engine_data(tmp_path: Path) -> None:
    path, engine, raw = edited_tables(tmp_path)
    path.write_text(json.dumps(raw))
    assert read_dominated_table(path, engine).config_hash == TABLE.config_hash

    def refuse(change: Any, message: str) -> None:
        edited = copy.deepcopy(raw)
        change(edited)
        path.write_text(json.dumps(edited))
        with pytest.raises(DominatedError, match=message):
            read_dominated_table(path, engine)

    refuse(lambda c: c.update(extra=1), "missing or unknown keys")
    refuse(lambda c: c.update(kind="other"), "not a version 1")
    refuse(lambda c: c.update(engine_data_hash="f" * 64), "another engine-data table")
    refuse(lambda c: c.update(effectiveness_ppm=[1_000_000]), "every code")
    refuse(lambda c: c.update(effectiveness_ppm=[1_000_000, 2.5, 1, 1]), "integer")
    refuse(lambda c: c.update(same_type_ppm=0), "same_type_ppm")
    refuse(lambda c: c.update(threshold_ppm=2_000_000), "threshold_ppm")
    refuse(lambda c: c["target_kinds"].pop("random"), "single, spread and random")
    refuse(lambda c: c["target_kinds"].update(single=["normal", "any"]), "sorted, unique")
    refuse(lambda c: c["target_kinds"].update(single=["nosuch"]), "lacks")
    refuse(lambda c: c["target_kinds"].update(random=["normal"]), "two groups")
    with pytest.raises(DominatedError, match="cannot be read"):
        read_dominated_table(tmp_path / "missing.json", engine)


# -- a progress generation and the page -----------------------------------------------------------


@pytest.fixture
def run(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    make_run(path)
    return path


@pytest.fixture
def root(run: Path, tmp_path: Path) -> Path:
    return progress_root(run, tmp_path / "progress")


def config_of(run: Path) -> Any:
    return read_run_config(layout.read_json(run / layout.CONFIG))


def test_the_stored_games_of_a_generation_give_one_entry_per_match(run: Path, root: Path) -> None:
    found = read_report_model(run, root).progress[1].dominated
    assert found is not None
    assert found["stored_games"] == sum(m["games"] for m in found["matches"]) > 0
    for entry in found["matches"]:
        for role in ("candidate", "opponent"):
            seat = entry[role]
            assert seat["scored"] + seat["unscored"] + seat["status"] + seat["unresolved"] > 0
            assert 0 <= seat["immune"] <= seat["dominated"] <= seat["scored"]


def test_the_page_shows_the_label_the_config_hash_and_the_interval_label(
    run: Path, root: Path
) -> None:
    model = read_report_model(run, root)
    text = build_report(model)
    assert LABEL in text
    assert "Statically dominated damaging moves" in text
    assert TABLE.config_hash in text
    assert TABLE.engine_data_hash in text
    assert INTERVAL_LABEL in text
    assert f"config/diagnostics/{FORMAT}/dominated-move.v1.json" in model.inputs
    assert_inert(text, ())


def test_no_games_or_no_config_means_no_table(run: Path, root: Path) -> None:
    model = read_report_model(run, root)
    bare = dataclasses.replace(
        model,
        progress={g: dataclasses.replace(f, dominated=None) for g, f in model.progress.items()},
    )
    assert "Statically dominated" not in build_report(bare)
    assert progress_dominated(root, 1, config_of(run)) is not None
    other = dataclasses.replace(config_of(run), format_id="gen9nonesuch")
    assert progress_dominated(root, 1, other) is None
    for game_file in (root / "g000001" / "games").glob("*.json"):
        game_file.unlink()
    assert progress_dominated(root, 1, config_of(run)) is None


def test_hostile_strings_in_the_table_are_inert(run: Path, root: Path) -> None:
    model = read_report_model(run, root)
    facts = model.progress[1]
    rates = copy.deepcopy(facts.dominated)
    assert rates is not None
    rates["matches"][0]["candidate"]["policy_id"] = HOSTILE
    rates["matches"][0]["match_id"] = HOSTILE
    rates["config_hash"] = HOSTILE
    changed = dataclasses.replace(facts, dominated=rates)
    text = build_report(dataclasses.replace(model, progress={**model.progress, 1: changed}))
    assert_inert(text, (HOSTILE,))


def test_the_one_pass_diagnostics_equal_the_diagnostics_of_the_held_list(
    run: Path, root: Path
) -> None:
    """Streaming the stored games changes no byte (#1245): same output as folding a held list."""
    config = config_of(run)
    streamed = stored_games(root, 1, config)
    assert isinstance(streamed, Iterator)  # one parsed game at a time, not a list
    held = list(streamed)
    assert held
    classes = committed_move_classes(config.format_id)
    table = committed_dominated_table(config.format_id)
    assert classes is not None
    assert table is not None
    reordered = list(reversed(held))  # the sums do not depend on the order of the games
    expected = (move_use(reordered, classes), dominated_rates(reordered, table))
    assert progress_diagnostics(root, 1, config) == expected
    assert progress_diagnostics(root, 1, config, None)[0] == progress_move_use(root, 1, config)
    assert progress_dominated(root, 1, config) == expected[1]
    assert json.dumps(expected, sort_keys=True) == json.dumps(
        progress_diagnostics(root, 1, config), sort_keys=True
    )


def test_with_neither_record_the_diagnostics_are_none_and_nothing_is_read(
    run: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No move-class table and no dominated config: no game is opened, not even the manifest."""

    def unread(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a stored game was read")

    monkeypatch.setattr(module, "committed_move_classes", lambda _format: None)
    monkeypatch.setattr(module, "committed_dominated_table", lambda _format: None)
    monkeypatch.setattr(module, "stored_games", unread)
    assert progress_diagnostics(root, 1, config_of(run)) == (None, None)
