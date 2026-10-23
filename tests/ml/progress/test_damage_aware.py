"""The damage-aware baseline as a progress-evaluation opponent (ADR-018 slice 4)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import layout
from areneon_ml.progress.records import (
    ProgressRecordError,
    check_game_baseline,
    read_game_record,
    read_progress_record,
    sum_baseline_counts,
)
from areneon_ml.progress.run import (
    DAMAGE_AWARE_LABEL,
    ProgressError,
    verify_progress_dir,
)
from areneon_ml.progress.run import progress_generation as _progress_generation

from tests.ml.progress.support import (
    IDENTITY,
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
    sample,
)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("run") / "run"
    make_run(run_dir, generations=2)
    return run_dir


def player(run: Path, change: Any = None) -> FakeProgressPlayer:
    return FakeProgressPlayer(dataset_config(), committed_manifest(run, 0), change)


def evaluate(run: Path, root: Path, fake: Any, **kwargs: Any) -> dict[str, Any]:
    found: dict[str, Any] = _progress_generation(run, 1, root, fake, 2, 1, (), 4, **kwargs)
    return found


def rewrite(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    raw = layout.read_json(path)
    change(raw)
    layout.write_json(path, raw)


def test_the_two_matches_are_played_on_request_and_say_what_they_are(
    run: Path, tmp_path: Path
) -> None:
    fake = player(run)
    done = evaluate(run, tmp_path / "p", fake, damage_aware_budget_minutes=30)
    assert fake.requests[0].damage_aware_budget_ms == 30 * 60_000
    ids = [m["match_id"] for m in done["matches"]]
    assert ids == [
        "actor-rule-argmax",
        "actor-rule-sample",
        "damage-aware-argmax",
        "damage-aware-sample",
        "random-legal-argmax",
        "random-legal-sample",
    ]
    assert done["damage_aware"] == "played"
    by_id = {m["match_id"]: m for m in done["matches"]}
    assert by_id["damage-aware-sample"]["comparison"] == DAMAGE_AWARE_LABEL
    assert by_id["damage-aware-sample"]["baseline"]["identity_hash"] == IDENTITY
    assert "baseline" not in by_id["actor-rule-sample"]
    record = read_progress_record(layout.read_json(tmp_path / "p" / "g000001" / "progress.json"))
    match = next(m for m in record["matches"] if m["match_id"] == "damage-aware-argmax")
    rows = [r for r in record["games"] if r["match_id"] == "damage-aware-argmax"]
    assert match["baseline"]["counts"] == sum_baseline_counts([r["baseline_counts"] for r in rows])


def test_it_is_off_by_default_and_a_committed_root_without_it_is_kept(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    fake = player(run)
    done = evaluate(run, root, fake)
    assert fake.requests[0].damage_aware_budget_ms is None
    assert done["damage_aware"].startswith("none: damage-aware matches are opt-in")
    again = player(run)
    kept = evaluate(run, root, again, damage_aware_budget_minutes=30)
    assert kept["skipped"] is True
    assert again.requests == []
    assert kept["damage_aware"].startswith("none: damage-aware matches are opt-in")


def test_a_committed_root_with_the_matches_reports_them_played(run: Path, tmp_path: Path) -> None:
    root = tmp_path / "p"
    evaluate(run, root, player(run), damage_aware_budget_minutes=30)
    done = evaluate(run, root, player(run))
    assert done["skipped"] is True
    assert done["damage_aware"] == "played"
    assert verify_progress_dir(root / "g000001", dataset_config(), run, 1)["matches"]


def test_two_runs_give_byte_identical_files(run: Path, tmp_path: Path) -> None:
    for name in ("a", "b"):
        evaluate(run, tmp_path / name, player(run), damage_aware_budget_minutes=30)
    a, b = (sorted((tmp_path / n / "g000001").rglob("*.json")) for n in ("a", "b"))
    assert [p.relative_to(tmp_path / "a") for p in a] == [p.relative_to(tmp_path / "b") for p in b]
    assert all(x.read_bytes() == y.read_bytes() for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("minutes", [0, -3])
def test_a_budget_below_one_minute_is_refused(run: Path, tmp_path: Path, minutes: int) -> None:
    with pytest.raises(ProgressError, match="whole number of minutes"):
        evaluate(run, tmp_path / "p", player(run), damage_aware_budget_minutes=minutes)


def test_the_ablation_plays_no_damage_aware_match(run: Path, tmp_path: Path) -> None:
    with pytest.raises(ProgressError, match="plays no damage-aware match"):
        evaluate(
            run,
            tmp_path / "p",
            player(run),
            preview_ablation=True,
            damage_aware_budget_minutes=30,
        )


def refuse_in_record(
    run: Path, tmp_path: Path, change: Callable[[dict[str, Any]], None], message: str, **kw: Any
) -> None:
    def hook(out: Path, _request: Any) -> None:
        rewrite(out / "progress.json", change)

    root = tmp_path / "p"
    with pytest.raises((ProgressError, ProgressRecordError), match=message):
        evaluate(run, root, player(run, hook), **kw)
    assert sorted(p.name for p in root.iterdir()) == [".lock"]  # nothing committed


def test_matches_nobody_asked_for_are_refused(run: Path, tmp_path: Path) -> None:
    # A simulator side that plays them although they were not requested.
    class Eager(FakeProgressPlayer):
        def progress(self, request: Any) -> dict[str, Any]:
            import dataclasses  # noqa: PLC0415

            return super().progress(dataclasses.replace(request, damage_aware_budget_ms=1))

    with pytest.raises(ProgressError, match="asks for no damage-aware matches"):
        evaluate(run, tmp_path / "p", Eager(dataset_config(), committed_manifest(run, 0)))


def test_half_a_set_or_another_opponent_is_refused(run: Path, tmp_path: Path) -> None:
    def drop_sample(raw: dict[str, Any]) -> None:
        raw["matches"] = [m for m in raw["matches"] if m["match_id"] != "damage-aware-sample"]
        raw["games"] = [g for g in raw["games"] if g["match_id"] != "damage-aware-sample"]

    refuse_in_record(
        run,
        tmp_path / "a",
        drop_sample,
        "damage-aware-argmax and damage-aware-sample",
        damage_aware_budget_minutes=30,
    )

    def misname(raw: dict[str, Any]) -> None:
        for m in raw["matches"]:
            if m["match_id"] == "damage-aware-sample":
                m["opponent"]["policy_id"] = "damage-aware@v2"

    refuse_in_record(
        run, tmp_path / "b", misname, "not damage-aware@v1", damage_aware_budget_minutes=30
    )


def test_a_stored_game_must_carry_its_matchs_identity_and_its_rows_counts(
    run: Path, tmp_path: Path
) -> None:
    def hook(out: Path, _request: Any) -> None:
        rewrite(
            out / "games" / "damage-aware-argmax-0.json",
            lambda g: g["baseline"].update(identity_hash="8" * 64),
        )
        record = layout.read_json(out / "progress.json")
        from areneon_ml.progress.records import record_hash  # noqa: PLC0415

        stored = layout.read_json(out / "games" / "damage-aware-argmax-0.json")
        for m in record["matches"]:
            if m["match_id"] == "damage-aware-argmax":
                m["game_record_hashes"] = [record_hash(stored)]
        layout.write_json(out / "progress.json", record)

    with pytest.raises(ProgressError, match="not its match's identity"):
        evaluate(run, tmp_path / "p", player(run, hook), damage_aware_budget_minutes=30)


def test_a_committed_stored_game_is_tied_to_its_row_on_verification(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    evaluate(run, root, player(run), damage_aware_budget_minutes=30)
    target = root / "g000001"
    rewrite(
        target / "games" / "damage-aware-argmax-0.json",
        lambda g: g["baseline"]["counts"].update(queries=g["baseline"]["counts"]["queries"] + 1),
    )
    manifest = layout.read_json(target / layout.MANIFEST)
    manifest["files"] = layout.hash_tree(target)
    layout.write_json(target / layout.MANIFEST, manifest)
    with pytest.raises(ProgressError, match="not its match's identity"):
        verify_progress_dir(target, dataset_config(), run, 1)


def test_the_reader_ties_a_game_to_its_row() -> None:
    game = sample("progress-game-record/valid/damage-aware")
    match = {"baseline": {"identity_hash": game["baseline"]["identity_hash"]}}
    row = {"candidate_seat": "p1", "baseline_counts": game["baseline"]["counts"]}
    check_game_baseline(game, match, row)
    with pytest.raises(ProgressRecordError, match="not its match's identity"):
        check_game_baseline(game, match, {**row, "candidate_seat": "p2"})
    with pytest.raises(ProgressRecordError, match="carries no baseline"):
        check_game_baseline(game, {}, row)
    plain = sample("progress-game-record/valid/played")
    check_game_baseline(plain, {}, row)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("baseline-counts-disagree", "baseline.counts must be the sum"),
        ("baseline-row-missing", "carries baseline counts in its summary"),
        ("baseline-on-other-match", "only a damage-aware match carries baseline counts"),
        (
            "baseline-probe-runs-exceed-queries",
            "probe runs, unsupported queries and probe refusals",
        ),
        ("baseline-identities-differ", "one baseline identity hash"),
    ],
)
def test_the_evaluation_corpus_samples_are_refused(name: str, message: str) -> None:
    with pytest.raises(ProgressRecordError, match=message):
        read_progress_record(sample(f"progress-evaluation-record-2/reader-invalid/{name}"))


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("baseline-seat-wrong", "baseline.seat must be the one seat"),
        ("baseline-missing", "must carry the baseline counts"),
        ("baseline-decisions-differ", "must be the baseline seat's decisions"),
    ],
)
def test_the_game_corpus_samples_are_refused(name: str, message: str) -> None:
    with pytest.raises(ProgressRecordError, match=message):
        read_game_record(sample(f"progress-game-record/reader-invalid/{name}"))


def test_the_valid_samples_read_and_a_forfeit_may_leave_one_request_unanswered() -> None:
    assert read_progress_record(sample("progress-evaluation-record-2/valid/damage-aware-match"))
    game = copy.deepcopy(sample("progress-game-record/valid/damage-aware"))
    read_game_record(game)
    game["outcome"] = {
        "kind": "forfeit",
        "loser": "p1",
        "cause": "crash",
        "request_seq": 9,
        "turns": 3,
    }
    game["baseline"]["counts"]["decisions"] += 1
    read_game_record(game)
    game["baseline"]["counts"]["preview_decisions"] = game["baseline"]["counts"]["decisions"]
    with pytest.raises(ProgressRecordError, match="preview and scored decisions"):
        read_game_record(game)


def test_probe_refusals_are_summed_and_bounded() -> None:
    record = copy.deepcopy(sample("progress-evaluation-record-2/valid/damage-aware-match"))
    for row in record["games"]:
        row["baseline_counts"]["probe_refusals"] = 1
    record["matches"][0]["baseline"]["counts"]["probe_refusals"] = len(record["games"])
    assert read_progress_record(record)
    record["matches"][0]["baseline"]["counts"]["probe_refusals"] += 1
    with pytest.raises(ProgressRecordError, match="must be the sum"):
        read_progress_record(record)
    over = copy.deepcopy(sample("progress-evaluation-record-2/valid/damage-aware-match"))
    big = over["games"][0]["baseline_counts"]["queries"] + 1
    over["games"][0]["baseline_counts"]["probe_refusals"] = big
    over["matches"][0]["baseline"]["counts"]["probe_refusals"] = big
    with pytest.raises(ProgressRecordError, match="probe refusals cannot exceed queries"):
        read_progress_record(over)


def test_the_policy_id_constant_is_the_one_the_corpus_and_the_game_samples_name() -> None:
    from areneon_ml.progress.run import DAMAGE_AWARE_POLICY  # noqa: PLC0415

    record = sample("progress-evaluation-record-2/valid/damage-aware-match")
    assert record["matches"][0]["opponent"]["policy_id"] == DAMAGE_AWARE_POLICY
    assert (
        sample("progress-game-record/valid/damage-aware")["policies"]["p2"] == DAMAGE_AWARE_POLICY
    )
