"""The profile in the metrics record and the loop share L (ADR-028 section 1, #1191)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import layout
from areneon_ml.loop.__main__ import main
from areneon_ml.loop.metrics import (
    Counts,
    MetricsError,
    Peaks,
    build_metrics,
    metrics_path,
    profile_record,
    read_run_metrics,
    write_metrics,
)
from areneon_ml.loop.share import loop_share, run_loop_share

from tests.ml.learner.support import tiny_config
from tests.ml.loop.support import FakePlayer, make_loop, manifests
from tests.ml.loop.test_metrics import ticking

PLAY: dict[str, Any] = {
    "decisions": 30,
    "game_processes": {
        "processes": 2,
        "cpu_ms": 1000,
        "e_sim_ms": 200,
        "e_judge_ms": 100,
        "e_probe_ms": 300,
        "o_obs_ms": 50,
        "ipc_wait_ms": 400,
        "pool_load_ms": 60,
        "merge_ms": 40,
        "per_process": [
            {"cpu_ms": 600, "peak_rss_bytes": 700, "decisions": 20},
            {"cpu_ms": 400, "decisions": 10},
        ],
    },
    "policy_workers": {
        "workers": 1,
        "queries": 30,
        "cpu_ms": 500,
        "w_enc_ms": 120,
        "w_fwd_ms": 260,
        "w_enc_cpu_ms": 100,
        "w_fwd_cpu_ms": 250,
        "per_process": [{"cpu_ms": 500, "peak_rss_bytes": 9}],
    },
}
LOAD = {
    "validate": {"wall_ns": 3_000_000, "cpu_ns": 2_000_000, "count": 4},
    "reduce": {"wall_ns": 5_999_999, "cpu_ns": 1_000_000, "count": 4},
}


def record(profile: dict[str, Any] | None = None, **phases: int) -> dict[str, Any]:
    phase_ms = {"play": 1000, "load": 200, "train": 300, "write": 40, "seal": 60, **phases}
    return build_metrics(
        tiny_config(),
        1,
        phase_ms,
        Counts(2, 3, 4),
        Peaks(profile=profile_record(PLAY, LOAD) if profile is None else profile),
    )


def test_the_load_split_is_whole_milliseconds_and_absent_timers_count_zero() -> None:
    found = profile_record(PLAY, LOAD)
    assert found is not None
    assert found["load"] == {
        "validate_ms": 3,
        "reduce_ms": 5,
        "encode_ms": 0,
        "validate_cpu_ms": 2,
        "reduce_cpu_ms": 1,
        "encode_cpu_ms": 0,
    }
    assert profile_record({}, LOAD) is None


def test_a_record_without_a_profile_keeps_its_bytes_and_one_with_it_reads_back() -> None:
    plain = build_metrics(
        tiny_config(),
        1,
        {"play": 1},
        Counts(),
        Peaks(),
    )
    assert "profile" not in plain
    full = record()
    assert read_run_metrics(full, tiny_config(), 1)["profile"]["decisions"] == 30


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda p: p["game_processes"].update(processes=3), "game process count"),
        (lambda p: p["policy_workers"].update(workers=2), "policy worker count"),
        (lambda p: p.update(decisions=31), "decisions"),
        (lambda p: p["game_processes"].update(cpu_ms=1), "game process CPU"),
        (lambda p: p["policy_workers"].update(cpu_ms=1), "policy worker CPU"),
    ],
)
def test_a_profile_whose_lists_disagree_with_its_totals_is_refused(edit: Any, message: str) -> None:
    profile = profile_record(copy.deepcopy(PLAY), LOAD)
    assert profile is not None
    edit(profile)
    with pytest.raises(MetricsError, match=message):
        read_run_metrics(record(profile), tiny_config(), 1)


def test_a_profile_off_the_contract_is_refused() -> None:
    profile = profile_record(copy.deepcopy(PLAY), LOAD)
    assert profile is not None
    profile["game_processes"]["extra"] = 1
    with pytest.raises(MetricsError, match="run-metrics"):
        read_run_metrics({**record(), "profile": profile}, tiny_config(), 1)


def test_the_loop_share_from_cpu_and_from_wall_uses_the_smaller() -> None:
    out = loop_share(record())
    # On the game side E_sim + E_judge + O_obs = 350. CPU: workers add 100 + 250 over
    # (1000 - 60 - 40) + 500: (350 + 350) / 1400 = 50%. Wall: (350 + 400) / (750 + 300) = 71.4%.
    assert out["c_cpu_ppm"] == 500_000
    assert out["c_wall_ppm"] == 714_285
    assert out["c_ppm"] == 500_000
    assert out["c_source"] == "cpu"
    assert out["t_ms"] == 1600
    assert out["l_ppm"] == 500_000 * 1000 // 1600
    assert out["load_ms"] == {"validate": 3, "reduce": 5, "encode": 0}
    assert out["processes"] == {"game": 2, "workers": 1}
    # Wall can be the smaller: a latency-bound play spends its time waiting on a worker.
    waiting = copy.deepcopy(PLAY)
    waiting["game_processes"]["e_probe_ms"] = 3000
    wall = loop_share(record(profile_record(waiting, LOAD)))
    assert wall["c_source"] == "wall"
    assert wall["c_wall_ppm"] == 750 * 1_000_000 // 3750


def test_a_run_without_cpu_or_time_reports_zero_shares() -> None:
    empty = copy.deepcopy(PLAY)
    for key in list(empty["game_processes"]):
        if key.endswith("_ms"):
            empty["game_processes"][key] = 0
    empty["game_processes"]["per_process"] = [{"cpu_ms": 0, "decisions": 30}]
    empty["game_processes"]["processes"] = 1
    for key in list(empty["policy_workers"]):
        if key.endswith("_ms"):
            empty["policy_workers"][key] = 0
    empty["policy_workers"]["per_process"] = [{"cpu_ms": 0}]
    out = loop_share(record(profile_record(empty, {}), play=0, load=0, train=0, write=0, seal=0))
    assert (out["c_cpu_ppm"], out["c_wall_ppm"], out["l_ppm"]) == (0, 0, 0)


def test_the_command_reads_the_metrics_files_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_metrics(tmp_path, record())
    bare = build_metrics(tiny_config(), 2, {"play": 1}, Counts(), Peaks())
    write_metrics(tmp_path, bare)
    before = sorted(p.name for p in tmp_path.rglob("*"))
    out = run_loop_share(tmp_path)
    assert [g["generation"] for g in out["generations"]] == [1]
    assert out["without_profile"] == [2]
    assert run_loop_share(tmp_path, (2,))["generations"] == []
    assert main(["loop-share", "--run-dir", str(tmp_path), "--generation", "1"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["generations"][0]["generation"] == 1
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


def test_the_command_refuses_a_run_without_metrics_and_a_damaged_file(tmp_path: Path) -> None:
    with pytest.raises(MetricsError, match="no metrics directory"):
        run_loop_share(tmp_path)
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "g000000.json").write_text("[1]\n")  # not a record: skipped
    (tmp_path / "metrics" / "g000001.json").write_text('{"generation": 1}\n')
    with pytest.raises(MetricsError, match=r"g000001\.json"):
        run_loop_share(tmp_path)


def test_the_driver_records_the_profile_the_simulator_side_reports_and_nothing_else_changes(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain"
    profiled = tmp_path / "profiled"
    make_loop(plain).run(2)
    player = FakePlayer()
    player.last_profile = copy.deepcopy(PLAY)  # type: ignore[attr-defined]
    loop = make_loop(profiled, player=player)
    loop.clock = ticking()
    loop.profile = True
    loop.run(2)
    assert manifests(profiled) == manifests(plain)  # the generation files are byte-identical
    for g in (1, 2):
        raw = layout.read_json(metrics_path(profiled, g))
        found = read_run_metrics(raw, tiny_config(), g)["profile"]
        assert found["decisions"] == 30
        assert set(found["load"]) >= {"validate_ms", "reduce_ms", "encode_ms"}
        assert "profile" not in layout.read_json(metrics_path(plain, g))
    # Generation 0 plays nothing and has none.
    assert "profile" not in layout.read_json(metrics_path(profiled, 0))
