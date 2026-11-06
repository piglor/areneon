"""The situational move-use summary (#1283): hand-built counts, expected rates and intervals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from areneon_ml.report import situational as s


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
