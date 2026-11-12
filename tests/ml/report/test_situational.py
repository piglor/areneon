"""The situational move-use summary (#1283): hand-built counts, expected rates and intervals."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from areneon_ml.report import situational as s
from areneon_ml.report.render import _situational_section


def _seat(policy: str, counts: dict[str, int]) -> dict[str, object]:
    return {"policy": policy, "counts": counts}


def _raw() -> dict[str, object]:
    games = []
    # Eight pairs, two games each; the learned seat alternates between p1 and p2.
    for pair in range(8):
        for half in range(2):
            learned = {
                "protect.threatened.fresh.offered": 2,
                "protect.threatened.fresh.chosen": 1,
                "protect.safe.fresh.offered": 2,
                "protect.safe.fresh.chosen": 0,
                "protect.threatened.repeat.offered": 1,
                "protect.safe.repeat.offered": 1,
                "switch.bad_matchup.offered": 1,
                "switch.bad_matchup.chosen": 1,
                "switch.safe.offered": 3,
                "switch.exposed_slower.offered": 1,
                "switch.safe.chosen": 1,
            }
            seats = {
                "p1": _seat("policy:aa", learned)
                if half == 0
                else _seat("actor-rule@v1", {"decisions": 1}),
                "p2": _seat("actor-rule@v1", {"decisions": 1})
                if half == 0
                else _seat("policy:aa", learned),
            }
            games.append({"match_id": "m", "game": 2 * pair + half, "pair": pair, "seats": seats})
    return {
        "format": s.FORMAT,
        "generation": 20,
        "run_id": "r",
        "evaluation_key": "k",
        "pin": "x",
        "unreplayable_games": 0,
        "games": games,
    }


def test_rates_differences_and_intervals() -> None:
    out = s.summarise(_raw(), b_boot=200)
    body = out["matches"]["m"]["policy:aa"]
    assert body["pairs"] == 8
    assert body["games"] == 16
    assert body["counts"]["protect.threatened.fresh.offered"] == 32
    stats = body["statistics"]
    assert stats["protect_threatened"]["mean_ppm"] == 333_333  # 16 of 48 offered
    assert stats["protect_safe"]["mean_ppm"] == 0
    assert stats["protect_threatened_minus_safe"]["mean_ppm"] == 333_333
    assert stats["repeat_protect"]["mean_ppm"] == 0  # offered 32, never chosen
    assert stats["switch_bad_matchup"]["mean_ppm"] == 1_000_000
    assert stats["switch_not_threatened"]["mean_ppm"] == 250_000  # 1 of 4 offered in each game
    assert stats["switch_bad_matchup_minus_not_threatened"]["mean_ppm"] == 750_000
    # Every pair is identical, so every resample gives the same value: a point interval.
    assert stats["protect_threatened_minus_safe"]["interval"]["low_ppm"] == 333_333
    assert stats["protect_threatened_minus_safe"]["interval"]["high_ppm"] == 333_334
    # The baseline never had a Protect offered: no rate, no interval.
    base = out["matches"]["m"]["actor-rule@v1"]["statistics"]
    assert base["protect_threatened"] == {"mean_ppm": None, "interval": None}
    assert base["protect_threatened_minus_safe"]["mean_ppm"] is None


def test_interval_has_width_when_pairs_differ_and_is_byte_stable() -> None:
    raw = _raw()
    for game in raw["games"]:  # type: ignore[attr-defined]
        if game["pair"] % 2 == 0:
            for seat in game["seats"].values():
                if seat["policy"] == "policy:aa":
                    seat["counts"]["protect.threatened.fresh.chosen"] = 2
    a = s.summarise(raw, b_boot=200)
    b = s.summarise(raw, b_boot=200)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    iv = a["matches"]["m"]["policy:aa"]["statistics"]["protect_threatened"]["interval"]
    assert iv["low_ppm"] < 500_000 < iv["high_ppm"]  # mean is 24 of 48 offered
    assert iv["resamples"] == 200


def test_undefined_resamples_and_refusals(tmp_path: Path) -> None:
    raw = _raw()
    # One pair carries the only safe-Protect offers: resamples without it have no safe rate.
    for game in raw["games"]:  # type: ignore[attr-defined]
        for seat in game["seats"].values():
            if game["pair"] != 0:
                seat["counts"].pop("protect.safe.fresh.offered", None)
                seat["counts"].pop("protect.safe.repeat.offered", None)
    out = s.summarise(raw, b_boot=200)
    stat = out["matches"]["m"]["policy:aa"]["statistics"]["protect_safe"]
    assert stat["interval"] is None or stat["interval"]["resamples"] < 200
    with pytest.raises(s.SituationalError):
        s.summarise({"format": "other"})
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(_raw()), encoding="utf8")
    lines: list[str] = []
    assert s.main([str(path)], lines.append) == 0
    assert s.LABEL in lines[0]
    out_path = tmp_path / "out.json"
    assert s.main([str(path), "--out", str(out_path)]) == 0
    assert json.loads(out_path.read_bytes())["format"] == s.FORMAT


def test_a_game_without_a_pair_is_its_own_unit() -> None:
    raw = _raw()
    games = raw["games"]
    # Two unpaired games whose indices equal paired ones must not merge with pair 0 and 1.
    for game in games[:2]:  # type: ignore[index]
        game["pair"] = None
    out = s.summarise(raw, b_boot=50)
    assert (
        out["matches"]["m"]["policy:aa"]["pairs"] == 9
    )  # games 0 and 1 are units g0 and g1, apart from pairs 1 to 7
    for game in games:  # type: ignore[attr-defined]
        game["pair"] = None
    assert s.summarise(raw, b_boot=50)["matches"]["m"]["policy:aa"]["pairs"] == 16


def test_summary_reader_and_dashboard_section() -> None:
    summary = s.summarise(_raw(), b_boot=50)
    assert s.read_summary(json.loads(json.dumps(summary)))["generation"] == 20
    section = _situational_section(SimpleNamespace(situational=summary))  # type: ignore[arg-type]
    assert "Situational move-use" in section
    assert s.LABEL in section
    assert "333333 ppm" in section
    assert _situational_section(SimpleNamespace(situational=None)) == ""  # type: ignore[arg-type]
    with pytest.raises(s.SituationalError):
        s.read_summary({"format": s.FORMAT})
    with pytest.raises(s.SituationalError):
        s.read_summary({**summary, "label": "other"})
    broken = json.loads(json.dumps(summary))
    del broken["matches"]["m"]["policy:aa"]["statistics"]["protect_safe"]
    with pytest.raises(s.SituationalError):
        s.read_summary(broken)


def _history_raw() -> dict[str, object]:
    raw = _raw()
    for game in raw["games"]:  # type: ignore[attr-defined]
        for seat in game["seats"].values():
            if seat["policy"] == "policy:aa":
                seat["history"] = {
                    "history.decisions": 4,
                    "history.feature_errors": 1,
                    "history.feature_error.reader_gap": 1,
                    "history.stall.repeat.offered": 2,
                    "history.stall.repeat.chosen": 1,
                    "history.stall.repeat.chosen_success_ppm": 333_333,
                    "history.stall.repeat.chosen_success_n": 1,
                    "history.stall.fresh.offered": 2,
                    "history.stall.fresh.chosen": 2,
                    "history.power.up.offered": 1,
                    "history.power.up.chosen": 1,
                    "history.power.down.offered": 3,
                    "history.fails.all.offered": 2,
                    "history.fails.all.chosen": 1,
                    "history.fails.acted.offered": 2,
                    "history.fails.acted.chosen": 1,
                    "history.conditional.offered": 2,
                    "history.conditional.chosen": 1,
                    "history.conditional.chosen_branches": 3,
                    "history.conditional.chosen_passing": 2,
                }
    return raw


def test_history_rows_rates_and_intervals() -> None:
    out = s.summarise(_history_raw(), b_boot=100)
    entry = out["matches"]["m"]["policy:aa"]
    stats = entry["history"]["statistics"]
    assert set(stats) == set(s.HISTORY_STATISTIC_NAMES)
    assert entry["history"]["counts"]["history.decisions"] == 64
    assert stats["stall_repeat"]["mean_ppm"] == 500_000
    assert stats["stall_fresh"]["mean_ppm"] == 1_000_000
    assert stats["stall_repeat_minus_fresh"]["mean_ppm"] == -500_000
    assert stats["stall_repeat_success"]["mean_ppm"] == 333_333  # a share of one: 333333 / 10^6
    assert stats["stall_fresh_success"] == {"mean_ppm": None, "interval": None}
    assert stats["power_up"]["mean_ppm"] == 1_000_000
    assert stats["power_down"]["mean_ppm"] == 0
    assert stats["power_up_minus_down"]["mean_ppm"] == 1_000_000
    assert stats["fails_all"]["mean_ppm"] == 500_000
    assert stats["fails_all_ppm_of_decisions"]["mean_ppm"] == 250_000  # one per four decisions
    assert stats["history_build_failed"]["mean_ppm"] == 200_000  # 1 failed of 5 attempted
    assert stats["fails_all_fresh"]["mean_ppm"] is None
    assert stats["conditional"]["mean_ppm"] == 500_000
    assert stats["conditional_chosen_pass_share"]["mean_ppm"] == 666_666
    # every pair is equal, so the interval is a point
    iv = stats["stall_repeat"]["interval"]
    assert iv["low_ppm"] == 500_000
    assert iv["resamples"] == 100
    # the baseline seat carries no history
    assert "history" not in out["matches"]["m"]["actor-rule@v1"]
    again = s.summarise(_history_raw(), b_boot=100)
    assert json.dumps(out, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_history_leaves_the_existing_summary_unchanged() -> None:
    plain = s.summarise(_raw(), b_boot=50)
    with_history = s.summarise(_history_raw(), b_boot=50)
    for policy, body in with_history["matches"]["m"].items():
        stripped = {k: v for k, v in body.items() if k != "history"}
        assert stripped == plain["matches"]["m"][policy]
    assert all("history" not in b for b in plain["matches"]["m"].values())


def test_history_reader_render_and_dashboard(tmp_path: Path) -> None:
    summary = s.summarise(_history_raw(), b_boot=50)
    assert s.read_summary(json.loads(json.dumps(summary)))["generation"] == 20
    broken = json.loads(json.dumps(summary))
    del broken["matches"]["m"]["policy:aa"]["history"]["statistics"]["power_up"]
    with pytest.raises(s.SituationalError):
        s.read_summary(broken)
    text = s.render(summary)
    assert "history.stall_repeat: 500000 ppm" in text
    section = _situational_section(SimpleNamespace(situational=summary))  # type: ignore[arg-type]
    assert "History groups" in section
    assert "Fails every branch" in section
    assert "500000 ppm" in section
    plain = _situational_section(SimpleNamespace(situational=s.summarise(_raw(), b_boot=50)))  # type: ignore[arg-type]
    assert "History groups" not in plain
    # the plain-text table has no history lines without them
    assert "history." not in s.render(s.summarise(_raw(), b_boot=50))


def test_history_hostile_policy_string_is_escaped() -> None:
    raw = _history_raw()
    for game in raw["games"]:  # type: ignore[attr-defined]
        for seat in game["seats"].values():
            if seat["policy"] == "policy:aa":
                seat["policy"] = "<script>alert(1)</script>"
    section = _situational_section(SimpleNamespace(situational=s.summarise(raw, b_boot=20)))  # type: ignore[arg-type]
    assert "<script>alert" not in section
    assert "&lt;script&gt;" in section


def test_a_root_without_games_has_no_history_table_and_the_summary_bytes_are_stable(
    tmp_path: Path,
) -> None:
    empty = {**_raw(), "games": []}
    summary = s.summarise(empty, b_boot=20)
    assert summary["matches"] == {}
    assert "History groups" not in _situational_section(SimpleNamespace(situational=summary))  # type: ignore[arg-type]
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(_history_raw()), encoding="utf8")
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert s.main([str(path), "--out", str(a)]) == 0
    assert s.main([str(path), "--out", str(b)]) == 0
    assert a.read_bytes() == b.read_bytes()
