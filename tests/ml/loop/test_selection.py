"""The copy and explore step (ADR-021 section 6, Figure 2, slice 8)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import ladder_write as lw
from areneon_ml.loop import rating, selection
from areneon_ml.loop.ladder import read_ladder_record
from areneon_ml.loop.population_seed import population_draw

from tests.ml.loop.test_ladder_write import ANCHOR_STRENGTHS, generation_rows
from tests.ml.loop.test_population import population_config

MEMBERS = ("m0", "m1", "m2")
ANCHOR_IDS = ("actor-rule@v1", "random-legal@v1")
BOOT = rating.BootstrapPlan(rating.BootstrapSeed("pop-tiny", 4, "m0"), 40, 50_000)
STRONG = [0, 0, 0, 0, 8]
GENERATION = 4


def config_for(
    members: int = 3, selection_edit: dict[str, int] | None = None, **ppo: int
) -> RunConfig:
    def edit(raw: dict[str, Any]) -> None:
        raw["population"]["selection"].update(
            {"ready_generations": 3, "n_min": 4, "margin_ppm": 50_000, "min_roots": 2}
            | (selection_edit or {})
        )
        raw["ppo"].update(ppo)

    return population_config(members=members, edit=edit)


def forged(
    versus: list[int] | None = None,
    anchors: dict[str, list[int]] | None = None,
    pairs: tuple[str, str] = ("m0", "m2"),
) -> dict[tuple[str, str, str], list[int]]:
    """A window table: ``pairs[0]`` against ``pairs[1]`` with ``versus``, and each one's anchors."""
    table: dict[tuple[str, str, str], list[int]] = {}
    for member in MEMBERS:
        for anchor in ANCHOR_IDS:
            table[(member, "anchor", anchor)] = [0, 0, 1, 2, 3]
    for member, scores in (anchors or {}).items():
        for anchor in ANCHOR_IDS:
            table[(member, "anchor", anchor)] = list(scores)
    first, second = pairs
    scores = STRONG if versus is None else versus
    table[(first, "member", second)] = list(scores)
    table[(second, "member", first)] = list(reversed(scores))
    return table


def decide(  # noqa: PLR0913
    config: RunConfig,
    table: selection.Table,
    *,
    strengths: dict[str, int] | None = None,
    roots: dict[str, str] | None = None,
    last_copy: dict[str, int] | None = None,
    generation: int = GENERATION,
) -> dict[str, Any]:
    found = strengths or {"m0": 3_000_000_000, "m1": 2_000_000_000, "m2": 1_000_000_000}
    return selection.decide(
        config,
        generation,
        strengths=found,
        table=table,
        anchors=ANCHOR_IDS,
        roots=roots or {m: m for m in found},
        settings={m: selection.member_settings(config, k) for k, m in enumerate(found)},
        last_copy=last_copy or {},
        bootstrap=BOOT,
    )


# -- Figure 2, branch by branch -----------------------------------------------------------------


def test_every_branch_of_figure_2_on_forged_histories() -> None:
    config = config_for()
    cases: list[tuple[str, dict[str, Any], str | None]] = [
        # (name, keyword arguments of decide, the failed condition; None is a copy)
        ("copy", {}, None),
        ("nobody ready", {"last_copy": {"m0": 3, "m1": 3, "m2": 2}}, "no_ready_member"),
        ("ready at exactly the threshold", {"last_copy": {"m0": 1}}, None),
        ("just short of ready", {"last_copy": {"m2": 2, "m1": 2, "m0": 2}}, "no_ready_member"),
        ("too few pairs", {"table": forged([0, 0, 0, 0, 3])}, "too_few_pairs"),
        ("exactly n_min pairs", {"table": forged([0, 0, 0, 0, 4])}, None),
        ("no head to head", {"table": forged(pairs=("m0", "m1"))}, "too_few_pairs"),
        (
            "the only ready member is the strongest",
            {"last_copy": {"m1": GENERATION, "m2": GENERATION}},
            "too_few_pairs",
        ),
        ("bound below the margin", {"table": forged([4, 0, 0, 0, 4])}, "lower_bound"),
        ("bound at an even score", {"table": forged([0, 0, 8, 0, 0])}, "lower_bound"),
        (
            "donor weaker on anchors",
            {"table": forged(anchors={"m0": [0, 4, 3, 0, 0], "m2": [0, 0, 0, 0, 7]})},
            "anchor_minimum",
        ),
        (
            "equal anchor minima",
            {"table": forged(anchors={"m0": [0, 0, 0, 0, 7], "m2": [0, 0, 0, 0, 7]})},
            None,
        ),
        ("roots floor", {"roots": {"m0": "m0", "m1": "m0", "m2": "m2"}}, "root_floor"),
        ("roots floor held", {"roots": {"m0": "m0", "m1": "m1", "m2": "m1"}}, None),
    ]
    for name, arguments, failed in cases:
        outcome = decide(config, **({"table": forged()} | arguments))
        if failed is None:
            assert outcome["kind"] == "copy", name
            assert (outcome["loser"], outcome["donor"]) == ("m2", "m0"), name
        else:
            assert outcome == {"kind": "none", "failed": failed}, name


def test_the_copy_carries_its_evidence() -> None:
    config = config_for()
    outcome = decide(config, forged())
    assert outcome["pairs"] == 8
    assert outcome["margin_ppm"] == 50_000
    assert outcome["lower_bound_ppm"] == 1_000_000  # every resample holds only perfect pairs
    assert outcome["donor_anchor_min_ppm"] == outcome["loser_anchor_min_ppm"] == 833_333
    assert (outcome["roots_before"], outcome["roots_after"]) == (3, 2)
    assert outcome["learner_seed_index"] == 0
    assert outcome["settings"] == selection.explore(
        config, GENERATION, "m2", selection.member_settings(config, 0)
    )
    # The bound is the copy lane of the generation's bootstrap seed.
    assert outcome["lower_bound_ppm"] == rating.head_to_head_lower(STRONG, BOOT, rating.LANE_COPY)


def test_at_most_one_copy_and_the_loser_is_the_weakest_ready_member() -> None:
    config = config_for(members=4)
    strengths = {"m0": 4_000_000_000, "m1": 3_000_000_000, "m2": 2_000_000_000, "m3": 1_000_000_000}
    table = forged(pairs=("m0", "m3"))
    for member in ("m3", "m1"):
        table.update({(member, "anchor", a): [0, 0, 1, 2, 3] for a in ANCHOR_IDS})
    table[("m3", "member", "m2")] = list(STRONG)
    outcome = decide(config, table, strengths=strengths)
    assert (outcome["kind"], outcome["loser"]) == ("copy", "m3")
    # m3 in its grace period: the next weakest ready member is the one at stake.
    other = decide(config, table, strengths=strengths, last_copy={"m3": GENERATION})
    assert other == {"kind": "none", "failed": "too_few_pairs"}


def test_ties_go_to_the_lower_member_index() -> None:
    config = config_for()
    even = {"m0": 2_000_000_000, "m1": 2_000_000_000, "m2": 2_000_000_000}
    outcome = decide(config, forged(pairs=("m0", "m2")), strengths=even)
    assert outcome["kind"] == "none"  # donor and loser are both m0: no head to head with itself
    outcome = decide(config, forged(pairs=("m0", "m1")), strengths=even, last_copy={"m0": 4})
    assert (outcome["kind"], outcome["loser"]) == ("copy", "m1")


# -- the ladder record carries the decision ------------------------------------------------------


def window_rows(generations: int) -> list[lw.GenerationRows]:
    return [lw.GenerationRows(g, generation_rows()) for g in range(1, generations + 1)]


def lowered() -> RunConfig:
    return config_for(
        members=2, selection_edit={"ready_generations": 1, "n_min": 2, "min_roots": 1}
    )


def test_a_copy_in_a_built_record_passes_the_reader_and_the_next_record_applies_it() -> None:
    config = lowered()
    # Two generations give two pairs of m0 against m1, and m1 is ready (2 - 0 >= 1): the weak m1
    # is copied.
    first = lw.build_ladder_record(config, 2, window_rows(2), ANCHOR_STRENGTHS)
    assert first["selections"][0]["kind"] == "copy"
    copy_ = first["selections"][0]
    assert (copy_["loser"], copy_["donor"]) == ("m1", "m0")
    assert read_ladder_record(first, None, config).raw is first
    rows = window_rows(3)
    rows[1] = lw.GenerationRows(2, rows[1].rows, ("m1",))  # the driver names the copied slot
    second = lw.build_ladder_record(config, 3, rows, ANCHOR_STRENGTHS, first, last_copy={"m1": 2})
    assert read_ladder_record(second, first, config).generation == 3
    by_member = {m["member"]: m for m in second["members"]}
    assert by_member["m1"]["root"] == "m0"  # the loser descends from the donor now
    assert by_member["m1"]["settings"] == copy_["settings"]
    assert by_member["m0"]["root"] == "m0"
    # m1 was copied in generation 2: its results of generations 1 and 2 are gone, 3's remain.
    cell = next(c for c in second["cells"] if (c["member"], c["opponent"]["id"]) == ("m0", "m1"))
    assert cell["pairs"] == 1
    # m1 is in its grace period and the decision is made on one pair.
    assert second["selections"][0] == {"kind": "none", "failed": "too_few_pairs"}
    # The record's own members are the state at its decision: the copy is in its selection only.
    assert [m["root"] for m in first["members"]] == ["m0", "m1"]


def test_a_copy_that_would_end_below_the_root_floor_is_none() -> None:
    config = config_for(
        members=2, selection_edit={"ready_generations": 1, "n_min": 2, "min_roots": 2}
    )
    record = lw.build_ladder_record(config, 2, window_rows(2), ANCHOR_STRENGTHS)
    assert record["selections"] == [{"kind": "none", "failed": "root_floor"}]
    assert read_ladder_record(record, None, config).generation == 2


def test_the_default_thresholds_need_many_pairs() -> None:
    config = population_config()
    record = lw.build_ladder_record(config, 3, window_rows(3), ANCHOR_STRENGTHS)
    assert record["selections"] == [{"kind": "none", "failed": "too_few_pairs"}]
    record = lw.build_ladder_record(config, 2, window_rows(2), ANCHOR_STRENGTHS)
    assert record["selections"] == [{"kind": "none", "failed": "no_ready_member"}]


# -- explore -------------------------------------------------------------------------------------


def with_kl() -> RunConfig:
    def edit(raw: dict[str, Any]) -> None:
        raw["ppo"]["target_kl_micro"] = 20_000
        raw["population"]["explore"]["bounds"]["target_kl_micro"] = {"min": 10_000, "max": 40_000}
        raw["population"]["explore"]["bounds"]["self_play_ppm"] = {"min": 400_000, "max": 600_000}
        raw["population"]["explore"]["bounds"]["past_pool_ppm"] = {"min": 0, "max": 100_000}

    return population_config(edit=edit)


def test_explore_follows_the_documented_draw_order() -> None:
    config = population_config()
    start = selection.member_settings(config, 0)
    got = selection.explore(config, 7, "m1", start)
    # The first draw alone decides whether the learning rate changed; recompute the rest by index.
    first = population_draw("pop-tiny", 7, "m1", "explore", 0) % 2
    assert (got["learning_rate_micro"] != start["learning_rate_micro"]) == bool(first)
    if first:
        up = population_draw("pop-tiny", 7, "m1", "explore", 1) % 2
        value = start["learning_rate_micro"]
        assert got["learning_rate_micro"] == (value * 5 // 4 if up else value * 4 // 5)


@pytest.mark.parametrize("config_of", [population_config, with_kl])
def test_explore_stays_inside_the_bounds_and_changes_only_what_it_may(
    config_of: Callable[[], RunConfig],
) -> None:
    config = config_of()
    bounds = config.raw["population"]["explore"]["bounds"]
    start = selection.member_settings(config, 0)
    seen: set[tuple[int, int, int, int]] = set()
    for generation in range(0, 120):
        for loser in ("m0", "m1", "m2", "m3"):
            got = selection.explore(config, generation, loser, start)
            assert got == selection.explore(config, generation, loser, start)  # pure
            assert set(got) == set(start)
            for name in ("learning_rate_micro", "entropy_coef_micro", "epochs", "clip_micro"):
                assert bounds[name]["min"] <= got[name] <= bounds[name]["max"], name
            if "target_kl_micro" in start:
                assert (
                    bounds["target_kl_micro"]["min"]
                    <= got["target_kl_micro"]
                    <= bounds["target_kl_micro"]["max"]
                )
            assert abs(got["epochs"] - start["epochs"]) <= 1
            lr, base = got["learning_rate_micro"], start["learning_rate_micro"]
            assert lr in {base, base * 5 // 4, base * 4 // 5}
            shares = got["shares"]
            assert sum(shares.values()) == 1_000_000
            for kind in selection.SHARE_KINDS:
                assert bounds[kind]["min"] <= shares[kind] <= bounds[kind]["max"], kind
                assert abs(shares[kind] - start["shares"][kind]) in (0, selection.SHARE_STEP_PPM)
            seen.add((lr, got["epochs"], got["clip_micro"], got["entropy_coef_micro"]))
    assert len(seen) > 6  # the draws really move the settings
    assert start == selection.member_settings(config, 0)  # the input is never changed


def test_explore_clamps_at_the_bounds() -> None:
    def edit(raw: dict[str, Any]) -> None:
        bounds = raw["population"]["explore"]["bounds"]
        bounds["learning_rate_micro"] = {"min": 1000, "max": 1000}
        bounds["epochs"] = {"min": 2, "max": 2}
        bounds["clip_micro"] = {"min": 200_000, "max": 200_000}
        bounds["entropy_coef_micro"] = {"min": 10_000, "max": 10_000}
        bounds["ladder_ppm"] = {"min": 500_000, "max": 500_000}
        bounds["self_play_ppm"] = {"min": 250_000, "max": 250_000}
        bounds["past_pool_ppm"] = {"min": 250_000, "max": 250_000}
        raw["population"]["shares"] = {
            "ladder_ppm": 500_000,
            "self_play_ppm": 250_000,
            "past_pool_ppm": 250_000,
        }

    config = population_config(edit=edit)
    start = selection.member_settings(config, 0)
    assert all(selection.explore(config, g, "m1", start) == start for g in range(40))


def test_a_target_kl_without_a_bound_is_left_alone() -> None:
    config = population_config(edit=lambda raw: raw["ppo"].update(target_kl_micro=20_000))
    start = selection.member_settings(config, 0)
    assert start["target_kl_micro"] == 20_000
    assert all(
        selection.explore(config, g, "m0", start)["target_kl_micro"] == 20_000 for g in range(40)
    )


def test_the_learner_seed_is_the_member_learner_draw() -> None:
    config = population_config()
    got = selection.learner_seed(config, 5, "m1", 0, (1 << 63) - 1)
    assert got == population_draw("pop-tiny", 5, "m1", "member-learner", 0) & ((1 << 63) - 1)


# -- the state a generation starts from ------------------------------------------------------------


def test_the_slot_state_is_the_configs_then_the_previous_records_with_its_copy_applied() -> None:
    config = population_config(
        edit=lambda raw: raw["population"].update(
            member_ppo=[{"member": 1, "learning_rate_micro": 700}]
        )
    )
    roots, settings = selection.slot_state(config, None)
    assert roots == {"m0": "m0", "m1": "m1"}
    assert settings["m1"]["learning_rate_micro"] == 700
    assert settings["m0"]["learning_rate_micro"] == 1000
    record = lw.build_ladder_record(config, 1, window_rows(1), ANCHOR_STRENGTHS)
    assert selection.slot_state(config, record) == (roots, settings)
    explored = {**settings["m0"], "epochs": 3}
    record["selections"][0] = {"kind": "copy", "loser": "m1", "donor": "m0", "settings": explored}
    now_roots, now_settings = selection.slot_state(config, record)
    assert now_roots == {"m0": "m0", "m1": "m0"}
    assert now_settings["m1"] == explored
    assert now_settings["m1"] is not explored  # a copy of it


def test_ppo_with_replaces_the_schedule_starts_and_leaves_the_rest() -> None:
    def edit(raw: dict[str, Any]) -> None:
        raw["ppo"]["learning_rate_schedule"] = {
            "kind": "linear",
            "final_micro": 100,
            "generations": 10,
        }

    config = population_config(edit=edit)
    got = selection.ppo_with(
        config.ppo, {"learning_rate_micro": 1250, "epochs": 3, "clip_micro": 160_000}
    )
    assert got.learning_rate == 0.00125
    assert got.schedule is not None
    assert got.schedule.start_micro == 1250
    assert (got.epochs, got.clip) == (3, 0.16)
    assert got.value_coef == config.ppo.value_coef
    assert selection.ppo_part({"epochs": 3, "shares": {}}) == {"epochs": 3}
