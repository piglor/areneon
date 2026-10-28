"""The move-use diagnostic (#1164): hand-built games, the class table and the page."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.inspector.progress_replay import decision_points
from areneon_ml.learner.config import read_run_config
from areneon_ml.loop import layout
from areneon_ml.progress.run import progress_generation
from areneon_ml.report.model import read_report_model
from areneon_ml.report.move_use import (
    CLASS_ORDER,
    LABEL,
    MoveClasses,
    MoveUseError,
    StoredGame,
    committed_move_classes,
    game_use,
    move_use,
    progress_move_use,
    read_move_classes,
    share_ppm,
    to_id,
)
from areneon_ml.report.render import build_report

from tests.ml.progress.support import FakeProgressPlayer, committed_manifest
from tests.ml.report.support import make_run, progress_root
from tests.ml.report.test_report import HOSTILE, assert_inert

FORMAT = "gen9championsvgc2026regmc"
FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "showdown" / FORMAT
_FOUND = committed_move_classes(FORMAT)
assert _FOUND is not None
CLASSES: MoveClasses = _FOUND


def mon(name: str, moves: list[str], species: str | None = None) -> dict[str, Any]:
    return {"name": name, "species": species or name, "moves": moves}


# p1: A holds a plain attack, Protect, Swords Dance and Icy Wind; B holds Tailwind, Trick Room,
# Surf and Psychic. p2 plays the same two Pokemon under other names.
TEAM_1 = [
    mon("Ann", ["Flamethrower", "Protect", "Swords Dance", "Icy Wind"], "Charizard"),
    mon("Bo", ["Tailwind", "Trick Room", "Surf", "Psychic"], "Blastoise"),
    mon("Cy", ["Tackle", "Protect", "Splash", "Roar"], "Pikachu"),
    mon("Di", ["Tackle", "Protect", "Splash", "Roar"], "Snorlax"),
]
TEAM_2 = [
    mon("Eve", ["Flamethrower", "Surf", "Tackle", "Splash"], "Venusaur"),
    mon("Fay", ["Flamethrower", "Surf", "Tackle", "Splash"], "Gengar"),
    mon("Gus", ["Flamethrower", "Surf", "Tackle", "Splash"], "Snorlax"),
    mon("Hal", ["Flamethrower", "Surf", "Tackle", "Splash"], "Pikachu"),
]

SWITCHES = [
    "|switch|p1a: Ann|Charizard, L50|100/100",
    "|switch|p1b: Bo|Blastoise, L50|100/100",
    "|switch|p2a: Eve|Venusaur, L50|100/100",
    "|switch|p2b: Fay|Gengar, L50|100/100",
]


def slot_options(first: list[str], second: list[str]) -> list[str]:
    return [f"{a}, {b}" for a in first for b in second]


def decision(
    player: str, seq: int, options: list[str], chosen: str, **extra: str
) -> dict[str, Any]:
    return {
        "player": player,
        "request_seq": seq,
        "candidates": [{"choice": c, "pending": False} for c in options],
        "chosen": chosen,
    } | extra


def game(  # noqa: PLR0913
    log: list[str],
    decisions: list[dict[str, Any]],
    *,
    policies: tuple[str, str] = ("policy:" + "a" * 64, "actor-rule@v1"),
    candidate: str = "p1",
    match: str = "m-argmax",
    number: int = 0,
) -> StoredGame:
    record = {
        "teams": {"p1": {"team": TEAM_1}, "p2": {"team": TEAM_2}},
        "policies": {"p1": policies[0], "p2": policies[1]},
        "omniscient_log": log,
        "decisions": decisions,
    }
    return StoredGame(match, number, candidate, "baseline", record, decision_points(record))


PREVIEW = [
    decision("p1", 1, ["team 1234"], "team 1234"),
    decision("p2", 1, ["team 1234"], "team 1234"),
]
HEAD = ["|teampreview|4", "|start", *SWITCHES]
A_OPTIONS = ["move 1 1", "move 1 2", "move 2", "move 3", "move 4", "switch 3", "switch 4"]
B_OPTIONS = ["move 1", "move 2", "move 3 1", "move 3 2", "move 4 1", "move 4 2", "switch 3"]
P2_OPTIONS = ["move 1 1", "move 2 1", "move 3 1", "move 4"]
TURN_1 = [*HEAD, "|turn|1"]


def counts(use: dict[str, Any], name: str) -> tuple[int, int, int]:
    c = use["classes"][name]
    return c["actions"], c["offered"], c["passed"]


def test_known_counts_for_one_turn() -> None:
    # Ann plays Swords Dance (status, raises own stat) and Bo switches out; p2 attacks twice.
    first = decision("p1", 2, slot_options(A_OPTIONS, B_OPTIONS), "move 3, switch 3")
    second = decision("p2", 2, slot_options(P2_OPTIONS, P2_OPTIONS), "move 1 1, move 2 1")
    used = game_use(game(TURN_1, [*PREVIEW, first, second]), CLASSES)
    p1 = used["p1"].as_dict()
    assert (p1["slot_actions"], p1["unresolved"]) == (2, 0)
    # slot a: Ann chose Swords Dance; slot b: Bo chose a switch
    assert counts(p1, "status") == (1, 2, 1)  # offered in both slots; chosen in slot a only
    assert counts(p1, "raises_own_stat") == (1, 1, 0)  # only Swords Dance, only Ann
    assert counts(p1, "lowers_target_stat") == (0, 1, 1)  # Icy Wind offered to Ann, not chosen
    assert counts(p1, "protection") == (0, 1, 1)  # Protect offered to Ann
    assert counts(p1, "side_condition") == (0, 1, 1)  # Tailwind offered to Bo
    assert counts(p1, "field_condition") == (0, 1, 1)  # Trick Room offered to Bo
    assert counts(p1, "switch") == (1, 2, 1)  # offered in both slots, taken in slot b
    p2 = used["p2"].as_dict()
    assert p2["slot_actions"] == 2
    assert all(counts(p2, name) == (0, 0, 0) for name in CLASS_ORDER)  # no class move in the pool


def test_each_class_is_counted_from_the_move_the_slot_names() -> None:
    log = [*TURN_1]
    options = slot_options(A_OPTIONS, B_OPTIONS)
    chosen = {
        "move 1 1, move 4 1": {"lowers_target_stat": 0, "status": 0},  # a plain move; Psychic
        "move 2, move 2": {"protection": 1, "status": 2, "field_condition": 1},
        "move 4, move 1": {"lowers_target_stat": 1, "side_condition": 1, "status": 1},
    }
    for pick, expected in chosen.items():
        used = game_use(game(log, [*PREVIEW, decision("p1", 2, options, pick)]), CLASSES)
        p1 = used["p1"].as_dict()
        for name in CLASS_ORDER:
            assert p1["classes"][name]["actions"] == expected.get(name, 0), (pick, name)


def test_a_refused_choice_and_a_forced_switch_are_not_counted() -> None:
    options = slot_options(A_OPTIONS, B_OPTIONS)
    refused = decision("p1", 2, options, "move 3, move 1", refused="[Unavailable choice] x")
    retry = decision("p1", 3, options, "move 2, move 1")
    forced = decision("p1", 4, ["switch 3, pass", "switch 4, pass"], "switch 3, pass")
    log = [*TURN_1, "|faint|p1a: Ann", "|switch|p1a: Cy|Pikachu, L50|100/100", "|turn|2"]
    used = game_use(game(log, [*PREVIEW, refused, retry, forced]), CLASSES)
    p1 = used["p1"].as_dict()
    assert p1["slot_actions"] == 2  # only the retry
    assert counts(p1, "protection") == (1, 1, 0)


def test_the_slot_follows_switches_faints_and_swaps() -> None:
    turn_1 = decision("p1", 2, slot_options(A_OPTIONS, B_OPTIONS), "move 1 1, move 1")
    # Ann faints, Cy replaces her; then Ann's and Bo's places swap: Bo is slot a at turn 3.
    cy_options = ["move 1 1", "move 2", "move 3", "move 4", "switch 3"]
    turn_2 = decision("p1", 4, slot_options(cy_options, B_OPTIONS), "move 2, move 1")
    log = [
        *TURN_1,
        "|faint|p1a: Ann",
        "|switch|p1a: Cy|Pikachu, L50|100/100",
        "|turn|2",
        "|swap|p1b: Bo|0",
        "|turn|3",
    ]
    forced = decision("p1", 3, ["switch 3, pass", "switch 4, pass"], "switch 3, pass")
    swapped = decision("p1", 5, slot_options(B_OPTIONS, cy_options), "move 2, move 3")
    used = game_use(game(log, [*PREVIEW, turn_1, forced, turn_2, swapped]), CLASSES)
    p1 = used["p1"].as_dict()
    assert p1["slot_actions"] == 6
    # turn 1: Ann Flamethrower, Bo Tailwind; turn 2: Cy Protect, Bo Tailwind (Cy took slot a);
    # turn 3: Bo is slot a and plays Trick Room, Cy slot b plays Splash
    assert counts(p1, "protection") == (1, 3, 2)  # Protect offered to Ann, then twice to Cy
    assert counts(p1, "side_condition") == (2, 3, 1)  # Tailwind: Bo in slot b twice, slot a once
    assert counts(p1, "field_condition") == (1, 3, 2)  # Trick Room: chosen only on turn 3


def test_a_locked_or_overflowing_slot_is_unresolved_never_guessed() -> None:
    locked = decision(
        "p1", 2, slot_options(["move 1 1", "move 1 2"], B_OPTIONS), "move 1 1, move 1"
    )
    overflow = decision(
        "p1", 3, slot_options(["move 1 1", "move 5 1"], B_OPTIONS), "move 5 1, move 1"
    )
    log = [*TURN_1, "|turn|2"]
    used = game_use(game(log, [*PREVIEW, locked, overflow]), CLASSES)
    p1 = used["p1"].as_dict()
    assert p1["unresolved"] == 2  # slot a both times
    assert p1["slot_actions"] == 2  # slot b both times (Tailwind is Bo's move 1)
    assert counts(p1, "side_condition") == (2, 2, 0)


def test_a_seat_whose_log_moves_are_not_in_its_set_is_unresolved_for_the_game() -> None:
    """Transform copies the target's moves: the seat's numbers cannot be read from its team."""
    options = slot_options(A_OPTIONS, B_OPTIONS)
    pick = decision("p1", 2, options, "move 3, move 1")
    copied = "|move|p1a: Ann|Quiver Dance|p1a: Ann"
    used = game_use(game([*TURN_1, copied], [*PREVIEW, pick]), CLASSES)
    assert (used["p1"].slot_actions, used["p1"].unresolved) == (0, 2)
    # the same line, called by another effect, does not count
    called = copied + "|[from]move: Sleep Talk"
    used = game_use(game([*TURN_1, called], [*PREVIEW, pick]), CLASSES)
    assert (used["p1"].slot_actions, used["p1"].unresolved) == (2, 0)
    # Struggle is no slot
    used = game_use(game([*TURN_1, "|move|p1a: Ann|Struggle|p2a: Eve"], [*PREVIEW, pick]), CLASSES)
    assert used["p1"].unresolved == 0
    # a name the team does not hold
    used = game_use(game([*TURN_1, "|move|p1a: Zed|Protect|p1a: Zed"], [*PREVIEW, pick]), CLASSES)
    assert used["p1"].unresolved == 2


def test_an_unnamed_occupant_is_unresolved() -> None:
    pick = decision("p1", 2, slot_options(A_OPTIONS, B_OPTIONS), "move 3, move 1")
    log = ["|teampreview|4", "|start", "|switch|p1a: Nobody|Charizard, L50|1/1", "|turn|1"]
    used = game_use(game(log, [*PREVIEW, pick]), CLASSES)
    assert used["p1"].unresolved == 2  # slot a unnamed, slot b empty


@pytest.mark.parametrize("baseline", ["damage-aware@v1", "damage-aware@v3"])
def test_matches_split_candidate_and_opponent_and_mark_the_damage_aware_reference(
    baseline: str,
) -> None:
    options = slot_options(A_OPTIONS, B_OPTIONS)
    pick = decision("p1", 2, options, "move 2, move 1")
    other = decision("p2", 2, slot_options(P2_OPTIONS, P2_OPTIONS), "move 1 1, move 1 1")
    a = game(TURN_1, [*PREVIEW, pick, other], policies=("policy:" + "a" * 64, baseline))
    b = game(
        TURN_1,
        [*PREVIEW, pick, other],
        policies=(baseline, "policy:" + "a" * 64),
        candidate="p2",
        number=1,
    )
    # candidate p2 is the frozen pair here and p1 the damage-aware seat that played the picks
    found = move_use([b, a], CLASSES)
    (match,) = found["matches"]
    assert found["label"] == LABEL
    assert match["games"] == 2
    assert match["reference"] is True
    assert match["candidate"]["policy_id"] == "policy:" + "a" * 64
    assert match["opponent"]["policy_id"] == baseline
    # game 0 (candidate p1) gave the candidate the picks, game 1 (candidate p2) gave them to the
    # opponent: each side has one Protect and one plain move in its two games
    assert counts(match["candidate"], "protection") == (1, 1, 0)
    assert counts(match["opponent"], "protection") == (1, 1, 0)
    assert found["class_table_sha256"] == CLASSES.table_sha256
    plain = move_use(
        [dataclasses.replace(a, record={**a.record, "policies": {"p1": "x@1", "p2": "y@2"}})],
        CLASSES,
    )
    assert plain["matches"][0]["reference"] is False


def test_shares_are_integers_and_empty_denominators_are_none() -> None:
    assert share_ppm(1, 3) == 333_333
    assert share_ppm(0, 0) is None
    assert to_id("Swords Dance") == "swordsdance"


# --- the class table ------------------------------------------------------------------------------


def copy_tables(tmp_path: Path) -> tuple[Path, Path]:
    classes, engine = tmp_path / "move-classes.json", tmp_path / "engine-data.json"
    classes.write_bytes((FIXTURES / "move-classes.json").read_bytes())
    engine.write_bytes((FIXTURES / "engine-data.json").read_bytes())
    return classes, engine


def test_the_classes_the_export_derives_are_pinned() -> None:
    summary = json.loads((FIXTURES / "move-classes.summary.json").read_text())
    raw = json.loads((FIXTURES / "move-classes.json").read_text())
    assert sha256_hex(jcs(raw)) == summary["move_class_table_hash"] == CLASSES.table_sha256
    assert {k: len(v) for k, v in raw["classes"].items()} == summary["counts"]
    engine = json.loads((FIXTURES / "engine-data.json").read_text())
    assert sha256_hex(jcs(engine)) == raw["engine_data_hash"] == CLASSES.engine_data_hash
    for name, inside, outside in (
        ("protection", {"protect", "detect", "kingsshield", "endure"}, {"quickguard", "icywind"}),
        ("side_condition", {"tailwind", "reflect", "lightscreen"}, {"trickroom", "protect"}),
        ("field_condition", {"trickroom", "raindance", "electricterrain"}, {"tailwind"}),
        ("lowers_target_stat", {"charm", "icywind", "snarl"}, {"closecombat", "swordsdance"}),
        ("raises_own_stat", {"swordsdance", "coaching", "flamecharge"}, {"closecombat", "charm"}),
        ("status", {"protect", "tailwind", "charm"}, {"icywind", "flamethrower"}),
    ):
        assert inside <= CLASSES.classes[name]
        assert not outside & CLASSES.classes[name]
    assert CLASSES.of("protect") == ("status", "protection")
    assert CLASSES.of("flamethrower") == ()


def test_the_table_is_refused_unless_it_fits_the_engine_data(tmp_path: Path) -> None:
    classes, engine = copy_tables(tmp_path)
    assert read_move_classes(classes, engine).table_sha256 == CLASSES.table_sha256
    raw = json.loads(classes.read_text())

    def refuse(change: Any, message: str) -> None:
        edited = copy.deepcopy(raw)
        change(edited)
        classes.write_text(json.dumps(edited))
        with pytest.raises(MoveUseError, match=message):
            read_move_classes(classes, engine)

    refuse(lambda t: t.update(engine_data_hash="f" * 64), "another engine-data table")
    refuse(lambda t: t["classes"].update(protection=["protect", "detect"]), "sorted, unique")
    refuse(lambda t: t["classes"].update(protection=["zzzzzzzz"]), "sorted, unique")
    refuse(lambda t: t["classes"]["status"].pop(), "not the Status moves")
    refuse(lambda t: t.update(extra=1), "cannot be read")
    changed = json.loads(engine.read_text())
    changed["mod"] = "other"
    engine.write_text(json.dumps(changed))
    classes.write_bytes((FIXTURES / "move-classes.json").read_bytes())
    with pytest.raises(MoveUseError, match="another engine-data table"):
        read_move_classes(classes, engine)
    with pytest.raises(MoveUseError, match="cannot be read"):
        read_move_classes(tmp_path / "missing.json", engine)


def test_a_format_without_a_table_has_no_diagnostic() -> None:
    assert committed_move_classes("gen9nonesuch") is None
    assert isinstance(CLASSES, MoveClasses)


# --- a progress generation and the page ----------------------------------------------------------


@pytest.fixture
def run(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    make_run(path)
    return path


@pytest.fixture
def root(run: Path, tmp_path: Path) -> Path:
    return progress_root(run, tmp_path / "progress")


def test_the_stored_games_of_a_generation_give_one_entry_per_match(run: Path, root: Path) -> None:
    model = read_report_model(run, root)
    used = model.progress[1].move_use
    assert used is not None
    assert used["label"] == LABEL
    assert [m["match_id"] for m in used["matches"]] == sorted(
        m.match_id for m in model.progress[1].matches
    )
    assert used["stored_games"] == sum(m["games"] for m in used["matches"]) > 0
    for entry in used["matches"]:
        for role in ("candidate", "opponent"):
            seat = entry[role]
            assert seat["slot_actions"] > 0
            assert set(seat["classes"]) == set(CLASS_ORDER)
            for c in seat["classes"].values():
                assert 0 <= c["passed"] <= c["offered"]
                assert c["actions"] <= seat["slot_actions"]


def test_the_page_shows_the_table_the_label_and_the_class_table_hash(run: Path, root: Path) -> None:
    model = read_report_model(run, root)
    text = build_report(model)
    assert LABEL in text
    assert "Move-use diagnostic" in text
    assert CLASSES.table_sha256 in text
    assert CLASSES.engine_data_hash in text
    assert "Protect family" in text
    assert "fixtures/showdown/gen9championsvgc2026regmc/move-classes.json" in model.inputs
    assert_inert(text, ())


def test_a_report_without_games_or_with_an_unknown_format_has_no_table(
    run: Path, root: Path, tmp_path: Path
) -> None:
    model = read_report_model(run, root)
    facts = model.progress[1]
    bare = dataclasses.replace(
        model, progress={**model.progress, 1: dataclasses.replace(facts, move_use=None)}
    )
    assert "Move-use diagnostic" not in build_report(
        dataclasses.replace(
            bare,
            progress={g: dataclasses.replace(f, move_use=None) for g, f in bare.progress.items()},
        )
    )
    assert progress_move_use(root, 1, config_of(run)) is not None
    for game_file in (root / "g000001" / "games").glob("*.json"):
        game_file.unlink()
    assert progress_move_use(root, 1, config_of(run)) is None


def config_of(run: Path) -> Any:
    return read_run_config(layout.read_json(run / layout.CONFIG))


def test_hostile_strings_in_the_diagnostic_are_inert(run: Path, root: Path) -> None:
    model = read_report_model(run, root)
    facts = model.progress[1]
    used = copy.deepcopy(facts.move_use)
    assert used is not None
    used["matches"][0]["candidate"]["policy_id"] = HOSTILE
    used["matches"][0]["match_id"] = HOSTILE
    used["class_table_sha256"] = HOSTILE
    changed = dataclasses.replace(facts, move_use=used)
    text = build_report(dataclasses.replace(model, progress={**model.progress, 1: changed}))
    assert_inert(text, (HOSTILE,))
    broken = text.replace("&lt;/script&gt;", "</script>", 1)
    with pytest.raises(AssertionError):
        assert_inert(broken, (HOSTILE,))


def test_a_game_that_does_not_verify_refuses_the_diagnostic(run: Path, root: Path) -> None:
    cfg = config_of(run)
    target = root / "g000001" / "games"
    stored = sorted(target.glob("*.json"))[0]
    raw = json.loads(stored.read_text())
    raw["omniscient_log"].append("|extra")
    stored.write_text(json.dumps(raw))
    with pytest.raises(MoveUseError, match=r"omniscient_log_sha256|manifest|hash"):
        progress_move_use(root, 1, cfg)
    # a record whose manifest is fine but which the progress record does not list
    raw["omniscient_log_sha256"] = sha256_hex(jcs(raw["omniscient_log"]))
    stored.write_text(json.dumps(raw))
    with pytest.raises(MoveUseError):
        progress_move_use(root, 1, cfg)


def test_the_progress_command_prints_the_diagnostic(run: Path, tmp_path: Path) -> None:
    player = FakeProgressPlayer(config_of(run), committed_manifest(run, 0))
    progress_generation(run, 1, tmp_path / "p", player, games=2, record_games=1)
    used = progress_move_use(tmp_path / "p", 1, config_of(run))
    assert used is not None
    assert used["stored_games"] == len(used["matches"])  # one stored game per match
    assert json.loads(jcs(used))["label"] == LABEL
