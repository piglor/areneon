"""Health stops and the quarantine record (ADR-021 sections 6 and 12, slice 9), on forged data."""

from __future__ import annotations

import math
from typing import Any

import pytest
from areneon_ml.loop import health, selection
from areneon_ml.loop import ladder_write as lw
from areneon_ml.loop.ladder import LadderRecordError, read_ladder_record

from tests.ml.loop.test_ladder_write import ANCHOR_STRENGTHS
from tests.ml.loop.test_selection import BOOT, config_for, decide, forged, lowered, window_rows

LIMITS = health.Limits(clip_ppm=500_000, kl_ppm=100_000, forfeit_ppm=50_000)


def report(clip: float = 0.1, kl: float = 0.01, **battle: Any) -> dict[str, Any]:
    return {
        "battle": {"clip_fraction": clip, "approx_kl": kl, "policy_loss": 0.5, **battle},
        "preview": {"clip_fraction": 0.0, "approx_kl": 0.0, "policy_loss": 0.5},
    }


def over_clip() -> dict[str, Any]:
    return report(clip=0.6)


def over_kl() -> dict[str, Any]:
    return report(kl=0.2)


def games(total: int, lost: int, member: str = "m1") -> list[dict[str, Any]]:
    rows = []
    for n in range(total):
        row: dict[str, Any] = {
            "member": member,
            "outcome": "played",
            "seats": {"p1": member, "p2": "m0"},
            "winner": "p1",
        }
        if n < lost:
            row.update(
                outcome="forfeit", winner="p2", forfeit={"loser": "p1", "cause": "invalid_reply"}
            )
        rows.append(row)
    return rows


def cause(
    now: dict[str, Any],
    earlier: list[dict[str, Any] | None] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> str | None:
    found: str | None = health.stop_cause(
        now, earlier or [None, None], rows or games(20, 0), "m1", LIMITS
    )
    return found


# -- the stops -------------------------------------------------------------------------------------


def test_a_healthy_report_is_not_a_stop() -> None:
    assert cause(report()) is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_number_anywhere_in_the_report_is_a_stop(bad: float) -> None:
    assert cause(report(value_loss=bad)) == "non_finite_loss"
    nested = report()
    nested["preview"]["value_calibration"] = [{"mean_value": bad}]
    assert cause(nested) == "non_finite_loss"


def test_a_non_finite_stop_comes_before_the_others() -> None:
    assert cause(report(clip=0.9, policy_loss=math.nan), [over_clip(), over_clip()]) == (
        "non_finite_loss"
    )


@pytest.mark.parametrize(
    ("make", "expected"), [(over_clip, "clip_fraction"), (over_kl, "approx_kl")]
)
def test_clip_and_kl_need_three_generations_in_a_row(make: Any, expected: str) -> None:
    assert cause(make(), [make(), make()]) == expected
    assert cause(make(), [make(), None]) is None  # a generation that does not count breaks it
    assert cause(make(), [None, make()]) is None
    assert cause(make(), [make(), report()]) is None  # a healthy one between
    assert cause(make()) is None  # two generations are not three
    assert cause(report(), [make(), make()]) is None  # the current generation must be over too


def test_the_limits_are_strict_and_exact_in_integers() -> None:
    assert cause(report(clip=0.5), [report(clip=0.5), report(clip=0.5)]) is None
    assert cause(report(clip=0.500001), [report(clip=0.500001)] * 2) == "clip_fraction"
    assert cause(report(kl=0.1), [report(kl=0.1)] * 2) is None
    assert cause(report(kl=0.100001), [report(kl=0.100001)] * 2) == "approx_kl"


def test_a_value_only_report_and_a_report_with_holes_never_count() -> None:
    warm = report(clip=0.9, value_warmup=True)
    assert cause(over_clip(), [warm, over_clip()]) is None
    sanitized = report()
    sanitized["battle"]["clip_fraction"] = None  # a stored report whose NaN was nulled
    assert cause(over_clip(), [sanitized, over_clip()]) is None
    assert cause(over_clip(), [{"preview": {}}, over_clip()]) is None


def test_the_clip_stop_comes_before_the_kl_stop() -> None:
    both = report(clip=0.6, kl=0.2)
    assert cause(both, [both, both]) == "clip_fraction"


def test_forfeits_over_the_share_of_the_members_own_games_are_a_stop() -> None:
    assert cause(report(), rows=games(20, 1)) is None  # exactly 50,000 ppm is allowed
    assert cause(report(), rows=games(20, 2)) == "forfeits"
    assert cause(report(), rows=games(1, 1)) == "forfeits"
    assert cause(report(), rows=[]) is None


def test_only_the_members_own_games_and_losses_count() -> None:
    mine = games(10, 0)
    # Another member's forfeits do not count, nor a forfeit the member's opponent made.
    assert cause(report(), rows=mine + games(10, 10, member="m2")) is None
    won = games(10, 0)
    won[0].update(outcome="forfeit", winner="p1", forfeit={"loser": "p2", "cause": "crash"})
    assert cause(report(), rows=won) is None


def test_the_stored_report_has_no_nan_and_a_healthy_one_is_unchanged() -> None:
    ok = report()
    assert health.sanitize(ok) is ok
    bad = report(value_loss=math.nan)
    bad["preview"]["value_calibration"] = ({"mean_value": math.inf},)
    cleaned = health.sanitize(bad)
    assert cleaned["battle"]["value_loss"] is None
    assert cleaned["preview"]["value_calibration"] == [{"mean_value": None}]
    assert cleaned["battle"]["policy_loss"] == 0.5
    assert not health.non_finite(cleaned)


def test_half_or_more_quarantined_is_an_incident() -> None:
    health.check_incident({}, 4)
    health.check_incident({"m1": "forfeits"}, 4)
    health.check_incident({"m1": "forfeits"}, 3)
    for stopped, members in (
        ({"m0": "forfeits", "m1": "approx_kl"}, 4),
        ({"m0": "forfeits", "m1": "approx_kl"}, 3),
        ({"m0": "forfeits"}, 2),
        ({"m0": "forfeits"}, 1),
    ):
        with pytest.raises(health.HealthIncident, match="half or more"):
            health.check_incident(stopped, members)


# -- the ladder record ------------------------------------------------------------------------


def test_a_quarantined_member_is_neither_loser_nor_donor_of_a_copy() -> None:
    config = config_for()
    table = forged(pairs=("m0", "m2"))
    assert decide(config, table)["kind"] == "copy"  # m2 is the weakest ready member
    outcome = selection.decide(
        config,
        4,
        strengths={"m0": 3_000_000_000, "m1": 2_000_000_000, "m2": 1_000_000_000},
        table=forged(pairs=("m0", "m1")),
        anchors=("actor-rule@v1", "random-legal@v1"),
        roots={m: m for m in ("m0", "m1", "m2")},
        settings={
            m: selection.member_settings(config, k) for k, m in enumerate(("m0", "m1", "m2"))
        },
        last_copy={},
        bootstrap=BOOT,
        excluded=frozenset({"m2", "m0"}),
    )
    # Only m1 is eligible: it is its own donor, so there is no head to head.
    assert outcome == {"kind": "none", "failed": "too_few_pairs"}


def test_a_built_record_lists_each_health_replacement_after_the_copy_outcome() -> None:
    config = lowered()
    record = lw.build_ladder_record(
        config, 2, window_rows(2), ANCHOR_STRENGTHS, stopped={"m1": "approx_kl"}
    )
    # m1 is quarantined, so m0 has nobody to be copied onto: the copy test fails, health stays.
    assert record["selections"] == [
        {"kind": "none", "failed": "too_few_pairs"},
        {"kind": "health", "member": "m1", "cause": "approx_kl"},
    ]
    assert read_ladder_record(record, None, config).raw is record
    assert selection.quarantined(record) == {"m1": "approx_kl"}
    assert selection.replaced_slots(record) == ("m1",)


def test_the_health_donor_is_the_top_live_member_and_the_settings_are_explored() -> None:
    config = lowered()
    record = lw.build_ladder_record(
        config, 2, window_rows(2), ANCHOR_STRENGTHS, stopped={"m1": "forfeits"}
    )
    (found,) = selection.replacements(config, record)
    assert (found.loser, found.donor, found.cause) == ("m1", "m0", "forfeits")
    assert found.settings == selection.explore(config, 2, "m1", record["members"][0]["settings"])
    assert found.seed_index == selection.LEARNER_SEED_INDEX
    # The next record starts from it: the slot descends from the donor under the explored settings.
    rows = window_rows(3)
    rows[1] = lw.GenerationRows(2, rows[1].rows, ("m1",))
    after = lw.build_ladder_record(config, 3, rows, ANCHOR_STRENGTHS, record, last_copy={"m1": 2})
    members = {m["member"]: m for m in after["members"]}
    assert members["m1"]["root"] == "m0"
    assert members["m1"]["settings"] == found.settings
    assert read_ladder_record(after, record, config).generation == 3


def test_the_donor_is_never_a_quarantined_member_and_ties_go_to_the_lower_index() -> None:
    def record(strengths: dict[str, int], held: dict[str, str]) -> dict[str, Any]:
        return {
            "members": [
                {"member": m, "strength": {"strength": s}} for m, s in sorted(strengths.items())
            ],
            "selections": [
                {"kind": "none", "failed": "lower_bound"},
                *({"kind": "health", "member": m, "cause": c} for m, c in sorted(held.items())),
            ],
        }

    top = {"m0": 9, "m1": 5, "m2": 9, "m10": 9}
    assert selection.health_donor(record(top, {})) == "m0"
    assert selection.health_donor(record(top, {"m0": "forfeits"})) == "m2"
    assert selection.health_donor(record(top, {"m0": "forfeits", "m2": "approx_kl"})) == "m10"


def test_the_reader_refuses_a_health_replacement_of_a_copys_slot() -> None:
    config = lowered()
    record = lw.build_ladder_record(config, 2, window_rows(2), ANCHOR_STRENGTHS)
    assert record["selections"][0]["kind"] == "copy"
    record["selections"].append({"kind": "health", "member": "m1", "cause": "forfeits"})
    with pytest.raises(LadderRecordError, match="cannot name a copy's loser or donor"):
        read_ladder_record(record, None, config)
