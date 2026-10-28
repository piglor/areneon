"""Extra play modes (ADR-023 section 1, #1044) and the play-mode gap (#1042)."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs
from areneon_ml.loop import layout
from areneon_ml.loop.records import features_hash, policy_id
from areneon_ml.progress import run as run_module
from areneon_ml.progress.gap import GAP_LABEL, SAME_MODE_NOTE, V1_NOT_AVAILABLE, play_mode_gap
from areneon_ml.progress.modes import (
    ExtraMode,
    ModeError,
    match_extra,
    match_stem,
    mode_label,
    parse_mode,
    parse_modes,
    ppm_decimal,
)
from areneon_ml.progress.records import ProgressRecordError, read_progress_record
from areneon_ml.progress.run import ProgressError, VersusSpec, summary_of
from areneon_ml.progress.stats import PPM, pair_interval

from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
)

T5 = ExtraMode(500_000, None)
TH3 = ExtraMode(None, 30_000)
BOTH = ExtraMode(500_000, 30_000)


# --- the grammar -----------------------------------------------------------------------------


def test_the_grammar_names_extra_modes_and_nothing_else() -> None:
    assert parse_mode("argmax") is None
    assert parse_mode("sample") is None
    assert parse_mode("sample:t500000") == T5
    assert parse_mode("sample:th30000") == TH3
    assert parse_mode("sample:t500000:th30000") == BOTH
    assert (T5.name, TH3.name, BOTH.name) == (
        "sample:t500000",
        "sample:th30000",
        "sample:t500000:th30000",
    )
    assert (T5.suffix, TH3.suffix, BOTH.suffix) == ("-t500000", "-th30000", "-t500000-th30000")
    assert BOTH.truncation == {"kind": "threshold", "min_ppm": 30000}
    assert T5.truncation is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Sample",
        "argmax:t5",
        "sample:t0",
        "sample:t1000000",
        "sample:t1000000:th5",
        "sample:t05",
        "sample:th0",
        "sample:th1000000",
        "sample:th30000:t500000",
        "sample:t500000:",
        "sample:t5.5",
        "sample:t-5",
        "sample t500000",
        "sample:t500000:th30000:x",
    ],
)
def test_a_mode_outside_the_grammar_is_refused(text: str) -> None:
    with pytest.raises(ModeError):
        parse_mode(text)


def test_a_mode_list_has_each_mode_once_and_ignores_the_defaults() -> None:
    assert parse_modes(["argmax", "sample", "sample:t500000", "sample:th30000"]) == (T5, TH3)
    assert parse_modes([]) == ()
    with pytest.raises(ModeError, match="twice"):
        parse_modes(["sample:t500000", "sample:t500000"])
    with pytest.raises(ModeError, match="twice"):
        parse_modes(["argmax", "argmax"])


def test_an_extra_mode_sets_something_in_range() -> None:
    with pytest.raises(ModeError):
        ExtraMode()
    with pytest.raises(ModeError):
        ExtraMode(1_000_000, None)
    with pytest.raises(ModeError):
        ExtraMode(None, 0)


def test_labels_say_the_mode_in_words() -> None:
    assert ppm_decimal(500_000) == "0.5"
    assert ppm_decimal(30_000) == "0.03"
    assert ppm_decimal(1) == "0.000001"
    plain = {"mode": "sample", "match_id": "x-sample"}
    assert mode_label(plain) == "sample"
    assert mode_label({"mode": "argmax", "match_id": "x-argmax"}) == "argmax"
    assert mode_label({"mode": "sample", "temperature_ppm": 500_000}) == "sample, T 0.5"
    assert (
        mode_label({"mode": "sample", "truncation": {"kind": "threshold", "min_ppm": 30_000}})
        == "sample, T 1, threshold 0.03"
    )
    assert (
        mode_label(
            {
                "mode": "sample",
                "temperature_ppm": 500_000,
                "truncation": {"kind": "threshold", "min_ppm": 30_000},
            }
        )
        == "sample, T 0.5, threshold 0.03"
    )
    assert match_extra(plain) is None


def test_the_stem_drops_the_mode_part() -> None:
    assert match_stem({"mode": "argmax", "match_id": "damage-aware-v3-argmax"}) == "damage-aware-v3"
    assert match_stem({"mode": "sample", "match_id": "random-legal-sample"}) == "random-legal"
    assert (
        match_stem({"mode": "sample", "match_id": "random-legal-sample-t500000-th30000"})
        == "random-legal"
    )
    assert match_stem({"mode": "sample", "match_id": "odd"}) == "odd"


# --- the gap, from hand-built records ---------------------------------------------------------

#: A pair score in half-games and a pair of game outcomes (candidate on p1 first) that make it.
OUTCOMES = {
    4: ("candidate", "candidate"),
    3: ("candidate", "tie"),
    2: ("candidate", "opponent"),
    1: ("tie", "opponent"),
    0: ("opponent", "opponent"),
}


def match_of(
    match_id: str, mode: str, scores: list[int], **more: Any
) -> tuple[dict[str, Any], Any]:
    rows = []
    wins = ties = 0
    for pair, score in enumerate(scores):
        for offset, winner in enumerate(OUTCOMES[score]):
            wins += winner == "candidate"
            ties += winner == "tie"
            rows.append(
                {
                    "match_id": match_id,
                    "game": 2 * pair + offset,
                    "pair": pair,
                    "candidate_seat": "p1" if offset == 0 else "p2",
                    "outcome": {"kind": "played", "winner": winner, "turns": 4},
                }
            )
    games = 2 * len(scores)
    return {
        "match_id": match_id,
        "mode": mode,
        "opponent": {"kind": "baseline", "policy_id": "x@v1"},
        "games": games,
        "wins": wins,
        "ties": ties,
        "losses": games - wins - ties,
        **more,
    }, rows


def record_of(specs: list[tuple[str, str, list[int], dict[str, Any]]]) -> dict[str, Any]:
    matches, games = [], []
    for match_id, mode, scores, more in specs:
        match, rows = match_of(match_id, mode, scores, **more)
        matches.append(match)
        games.extend(rows)
    return {
        "schema_version": 2,
        "run_id": "hand-run",
        "generation": 7,
        "matches": sorted(matches, key=lambda m: str(m["match_id"])),
        "games": games,
    }


ARGMAX = [4, 3, 3, 2, 2, 4, 1, 3]
SAMPLE = [2, 3, 1, 2, 0, 4, 2, 1]
COLD = [4, 2, 3, 2, 2, 4, 0, 3]


def test_the_gap_is_argmax_minus_the_mode_pair_by_pair() -> None:
    record = record_of(
        [
            ("random-legal-argmax", "argmax", ARGMAX, {}),
            ("random-legal-sample", "sample", SAMPLE, {}),
            (
                "random-legal-sample-t500000",
                "sample",
                COLD,
                {"temperature_ppm": 500_000},
            ),
        ]
    )
    gap = play_mode_gap(record)
    assert gap["label"] == GAP_LABEL
    rows = {r["match_id"]: r for r in gap["rows"]}
    assert sorted(rows) == ["random-legal-sample", "random-legal-sample-t500000"]
    for match_id, theirs in (
        ("random-legal-sample", SAMPLE),
        ("random-legal-sample-t500000", COLD),
    ):
        row = rows[match_id]
        diffs = [a - b for a, b in zip(ARGMAX, theirs, strict=True)]
        key = hashlib.sha256(
            jcs(
                {
                    "run_key": "hand-run",
                    "generation": 7,
                    "label": f"gap:random-legal:argmax:"
                    f"{'sample' if match_id == 'random-legal-sample' else 'sample:t500000'}",
                    "purpose": "progress-bootstrap/1",
                }
            )
        ).digest()
        interval = pair_interval(diffs, key)
        assert row["pairs"] == 8
        assert row["mean_ppm"] == sum(diffs) * PPM // (4 * 8) == interval.mean_ppm
        assert (row["low_ppm"], row["high_ppm"]) == (interval.low_ppm, interval.high_ppm)
        assert row["low_ppm"] <= row["mean_ppm"] <= row["high_ppm"]
        assert (row["better"], row["equal"], row["worse"]) == (
            sum(d > 0 for d in diffs),
            sum(d == 0 for d in diffs),
            sum(d < 0 for d in diffs),
        )
        assert row["argmax_ppm"] == sum(ARGMAX) * PPM // 32
        assert row["mode_ppm"] == sum(theirs) * PPM // 32
        # the gap is the difference of the scores, to the rounding of the pair mean
        assert abs(row["mean_ppm"] - (row["argmax_ppm"] - row["mode_ppm"])) <= 1
    assert rows["random-legal-sample"]["mode"] == "sample"
    assert rows["random-legal-sample-t500000"]["mode"] == "sample, T 0.5"
    assert rows["random-legal-sample"]["opponent"] == "random-legal"


def test_the_stream_names_the_opponent_and_both_modes() -> None:
    record = record_of(
        [
            ("a-argmax", "argmax", ARGMAX, {}),
            ("a-sample", "sample", SAMPLE, {}),
            ("b-argmax", "argmax", ARGMAX, {}),
            ("b-sample", "sample", SAMPLE, {}),
        ]
    )
    rows = {r["opponent"]: r for r in play_mode_gap(record)["rows"]}
    # equal scores, but another opponent is another stream: the intervals are not shared
    assert (rows["a"]["mean_ppm"], rows["a"]["pairs"]) == (rows["b"]["mean_ppm"], 8)
    assert (rows["a"]["low_ppm"], rows["a"]["high_ppm"]) != (
        rows["b"]["low_ppm"],
        rows["b"]["high_ppm"],
    )
    assert play_mode_gap(record) == play_mode_gap(copy.deepcopy(record))


def test_only_baselines_and_the_other_runs_pair_have_a_gap() -> None:
    run_opponent = {
        "kind": "run",
        "run_id": "other",
        "run_config_hash": "a" * 64,
        "generation": 1,
        "policy_id": "policy:" + "b" * 64,
        "checkpoints": {"preview": "c" * 64, "battle": "d" * 64},
    }
    record = record_of(
        [
            ("run-argmax", "argmax", ARGMAX, {"opponent": run_opponent}),
            ("run-sample", "sample", SAMPLE, {"opponent": run_opponent}),
            (
                "anchor-g0-argmax",
                "argmax",
                ARGMAX,
                {"opponent": {**run_opponent, "kind": "anchor"}},
            ),
            (
                "anchor-g0-sample",
                "sample",
                SAMPLE,
                {"opponent": {**run_opponent, "kind": "anchor"}},
            ),
            (
                "previous-argmax",
                "argmax",
                ARGMAX,
                {"opponent": {**run_opponent, "kind": "previous"}},
            ),
            (
                "previous-sample",
                "sample",
                SAMPLE,
                {"opponent": {**run_opponent, "kind": "previous"}},
            ),
            ("lonely-sample", "sample", SAMPLE, {}),
            (
                "search-x-argmax",
                "argmax",
                ARGMAX,
                {"search": {"identity_hash": "e" * 64, "counts": {}}},
            ),
            (
                "search-x-sample",
                "sample",
                SAMPLE,
                {"search": {"identity_hash": "e" * 64, "counts": {}}},
            ),
        ]
    )
    gap = play_mode_gap(record)
    assert [r["match_id"] for r in gap["rows"]] == ["run-sample"]  # no argmax for lonely
    assert gap["rows"][0]["opponent"] == "run"
    notes = {s["match_id"]: s.get("play_mode_gap") for s in _summary(record)}
    assert notes["anchor-g0-sample"] == notes["previous-argmax"] == SAME_MODE_NOTE
    assert notes["run-sample"] is None


def _summary(record: dict[str, Any]) -> list[dict[str, Any]]:
    """``summary_of`` on a hand-built record: it needs the fields the reader would add."""
    full = copy.deepcopy(record)
    for match in full["matches"]:
        match.setdefault("turn_cap_ties", 0)
        match.setdefault("forfeits", [])
        match.setdefault("turns_total", 0)
    return summary_of(full)


def test_a_version_1_record_has_no_gap() -> None:
    record = record_of([("a-argmax", "argmax", ARGMAX, {})])
    record["schema_version"] = 1
    assert play_mode_gap(record) == {"label": GAP_LABEL, "status": V1_NOT_AVAILABLE}
    assert V1_NOT_AVAILABLE == "not available: version 1 record"


# --- the record readers -----------------------------------------------------------------------


def corpus(name: str) -> dict[str, Any]:
    root = (
        Path(__file__).resolve().parents[3] / "fixtures/contracts/v1/progress-evaluation-record-2"
    )
    found: dict[str, Any] = json.loads((root / f"{name}.json").read_text())
    return found


def test_the_corpus_extra_mode_samples_read_and_the_bad_ones_do_not() -> None:
    for name in ("extra-mode-match", "extra-mode-temperature-match", "extra-mode-run-match"):
        read_progress_record(corpus(f"valid/{name}"))
    for name in (
        "extra-mode-misnamed",
        "extra-mode-name-without-settings",
        "extra-mode-on-anchor",
        "extra-mode-on-argmax",
        "extra-mode-on-previous",
        "extra-mode-run-misnamed",
    ):
        with pytest.raises(ProgressRecordError):
            read_progress_record(corpus(f"reader-invalid/{name}"))
    for name in (
        "extra-mode-temperature-default",
        "extra-mode-temperature-zero",
        "extra-mode-temperature-fractional",
        "extra-mode-truncation-kind",
        "extra-mode-truncation-no-threshold",
    ):
        with pytest.raises(ProgressRecordError):
            read_progress_record(corpus(f"invalid/{name}"))


def test_a_match_with_a_search_or_a_preview_variant_takes_no_extra_mode() -> None:
    record = corpus("valid/extra-mode-match")
    searched = copy.deepcopy(record)
    searched["matches"][0]["candidate_preview"] = {"kind": "learned"}
    with pytest.raises(ProgressRecordError, match="ablation"):
        read_progress_record(searched)


# --- the evaluation: the request, the checks, the bytes ----------------------------------------

ENCODER = "enc-other"


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    base = tmp_path_factory.mktemp("modes")
    make_run(base / "a", generations=2)
    make_run(base / "b", generations=2, config=dataset_config("other-run", (21, 22)))
    return base / "a", base / "b"


def evaluate(
    run: Path, root: Path, modes: tuple[ExtraMode, ...], versus: Path | None = None, **more: Any
) -> tuple[dict[str, Any], FakeProgressPlayer]:
    fake = FakeProgressPlayer(dataset_config(), committed_manifest(run, 0))
    spec = None if versus is None else VersusSpec(versus, 1, 4, ENCODER)
    done: dict[str, Any] = run_module.progress_generation(
        run, 1, root, fake, 4, 1, (), 4, versus=spec, modes=modes, **more
    )
    return done, fake


def tree(path: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()
    }


def test_extra_modes_are_played_against_the_baselines_with_their_policy_ids(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, _ = runs
    done, fake = evaluate(a, tmp_path / "p", (T5, BOTH))
    (request,) = fake.requests
    config = dataset_config()
    assert [e.mode for e in request.extra_modes] == [T5, BOTH]
    assert request.extra_modes[0].policy_id == policy_id(
        request.preview,
        request.battle,
        "sample",
        temperature_ppm=500_000,
        features=features_hash(config),
    )
    assert request.extra_modes[1].policy_id == policy_id(
        request.preview,
        request.battle,
        "sample",
        temperature_ppm=500_000,
        truncation={"kind": "threshold", "min_ppm": 30000},
        features=features_hash(config),
    )
    assert request.extra_modes[0].versus_policy_id is None
    record = layout.read_json(tmp_path / "p" / "g000001" / "progress.json")
    ids = [m["match_id"] for m in record["matches"]]
    assert ids == [
        f"{actor}-{mode}"
        for actor in ("actor-rule", "random-legal")
        for mode in ("argmax", "sample", "sample-t500000", "sample-t500000-th30000")
    ]
    extra = next(m for m in record["matches"] if m["match_id"] == "actor-rule-sample-t500000")
    assert extra["temperature_ppm"] == 500_000
    assert "truncation" not in extra
    assert extra["policy_id"] == request.extra_modes[0].policy_id
    assert extra["opponent"] == {"kind": "baseline", "policy_id": "actor-rule@v1"}
    # the summary names every mode; the gap rows are derived, argmax minus the mode
    by_id = {s["match_id"]: s for s in done["matches"]}
    assert by_id["actor-rule-sample-t500000-th30000"]["mode"] == "sample, T 0.5, threshold 0.03"
    assert by_id["actor-rule-argmax"]["mode"] == "argmax"
    assert by_id["actor-rule-sample"]["mode"] == "sample"
    gap = done["play_mode_gap"]
    assert gap["label"] == GAP_LABEL
    assert [r["match_id"] for r in gap["rows"]] == [
        f"{actor}-sample{suffix}"
        for actor in ("actor-rule", "random-legal")
        for suffix in ("", "-t500000", "-t500000-th30000")
    ]
    # the stand-in wins every even game in argmax and sample and one game in four in an extra
    # mode: the two pairs score [2, 2] and [2, 0] half-games, so the gap is 0 and 1/4
    assert [r["mean_ppm"] for r in gap["rows"][:3]] == [0, 250_000, 250_000]
    assert all(r["pairs"] == 2 for r in gap["rows"])


def test_without_modes_the_bytes_are_todays(runs: tuple[Path, Path], tmp_path: Path) -> None:
    a, _ = runs
    evaluate(a, tmp_path / "x", ())
    evaluate(a, tmp_path / "y", (T5,))
    plain = tree(tmp_path / "x")
    assert not any("-t500000" in name for name in plain)
    record = layout.read_json(tmp_path / "x" / "g000001" / "progress.json")
    assert not any("temperature_ppm" in m or "truncation" in m for m in record["matches"])
    again = tmp_path / "z"
    evaluate(a, again, ())
    assert tree(again) == plain


def test_two_evaluations_with_an_extra_mode_give_the_same_bytes(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, _ = runs
    for name in ("x", "y"):
        evaluate(a, tmp_path / name, (BOTH, TH3))
    assert tree(tmp_path / "x") == tree(tmp_path / "y")


def test_the_head_to_head_plays_both_sides_in_the_mode(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs
    done, fake = evaluate(a, tmp_path / "p", (T5,), versus=b)
    (request,) = fake.requests
    other = request.versus
    assert other is not None
    mine = request.extra_modes[0]
    assert mine.versus_policy_id == policy_id(
        other.preview, other.battle, "sample", temperature_ppm=500_000, features=other.features
    )
    assert mine.versus_policy_id != other.policy_id_sample
    assert mine.versus_policy_id != mine.policy_id
    record = layout.read_json(tmp_path / "p" / "g000001" / "progress.json")
    run_match = next(m for m in record["matches"] if m["match_id"] == "run-sample-t500000")
    assert run_match["temperature_ppm"] == 500_000
    assert run_match["policy_id"] == mine.policy_id
    assert run_match["opponent"]["policy_id"] == mine.versus_policy_id
    assert run_match["games"] == 4
    assert "run-sample-t500000" in {r["match_id"] for r in done["play_mode_gap"]["rows"]}
    # a committed directory is verified again without the flag, then with the very same modes
    again, _ = evaluate(a, tmp_path / "p", (T5,), versus=b)
    assert again["skipped"] is True
    plain, _ = evaluate(a, tmp_path / "p", (), versus=b)
    assert plain["skipped"] is True
    assert "run-sample-t500000" in {m["match_id"] for m in plain["matches"]}


def test_extra_modes_nobody_asked_for_or_half_a_set_are_refused(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs

    def drop(match_id: str) -> Any:
        def change(out: Path, _request: Any) -> None:
            record = layout.read_json(out / "progress.json")
            record["matches"] = [m for m in record["matches"] if m["match_id"] != match_id]
            record["games"] = [g for g in record["games"] if g["match_id"] != match_id]
            layout.write_json(out / "progress.json", record)

        return change

    fake = FakeProgressPlayer(
        dataset_config(), committed_manifest(a, 0), drop("random-legal-sample-t500000")
    )
    with pytest.raises(ProgressError, match="extra play modes are not the modes asked for"):
        run_module.progress_generation(a, 1, tmp_path / "p", fake, 4, 1, (), 4, modes=(T5,))

    # a record with extra modes nobody asked for: the player ignores the request's modes
    class Ignoring(FakeProgressPlayer):
        def progress(self, request: Any) -> dict[str, Any]:
            return super().progress(dataclasses.replace(request, extra_modes=self.extra))

        extra: tuple[Any, ...] = ()

    sneaky = Ignoring(dataset_config(), committed_manifest(a, 0))
    request = evaluate(a, tmp_path / "q", (T5,))[1].requests[0]
    sneaky.extra = request.extra_modes
    with pytest.raises(ProgressError, match="asks for no extra play modes"):
        run_module.progress_generation(a, 1, tmp_path / "r", sneaky, 4, 1, (), 4, modes=())

    # and a head-to-head whose other side does not play the mode
    def other_side(out: Path, _request: Any) -> None:
        record = layout.read_json(out / "progress.json")
        for m in record["matches"]:
            if m["match_id"] == "run-sample-t500000":
                m["opponent"]["policy_id"] = "policy:" + "9" * 64
        layout.write_json(out / "progress.json", record)

    fake = FakeProgressPlayer(dataset_config(), committed_manifest(a, 0), other_side)
    with pytest.raises(ProgressError, match="does not play the same mode"):
        run_module.progress_generation(
            a,
            1,
            tmp_path / "s",
            fake,
            4,
            1,
            (),
            4,
            versus=VersusSpec(b, 1, 4, ENCODER),
            modes=(T5,),
        )


def test_a_policy_id_that_is_not_the_modes_is_refused(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, _ = runs

    def wrong(out: Path, _request: Any) -> None:
        record = layout.read_json(out / "progress.json")
        for m in record["matches"]:
            if m["match_id"] == "actor-rule-sample-t500000":
                m["policy_id"] = policy_id(m["policy_id"][7:71], "0" * 64, "sample")
        layout.write_json(out / "progress.json", record)

    fake = FakeProgressPlayer(dataset_config(), committed_manifest(a, 0), wrong)
    with pytest.raises(ProgressRecordError, match="policy_id is not the frozen pair's"):
        run_module.progress_generation(a, 1, tmp_path / "p", fake, 4, 1, (), 4, modes=(T5,))


def test_extra_modes_are_not_played_with_the_ablation_or_the_search(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, _ = runs
    fake = FakeProgressPlayer(dataset_config(), committed_manifest(a, 0))
    with pytest.raises(ProgressError, match="not with --preview-ablation"):
        run_module.progress_generation(
            a, 1, tmp_path / "p", fake, 4, 1, (), 4, preview_ablation=True, modes=(T5,)
        )
    with pytest.raises(ProgressError, match="not with --preview-ablation"):
        run_module.progress_generation(
            a, 1, tmp_path / "p", fake, 4, 1, (), 4, search_mode="argmax", modes=(T5,)
        )
    with pytest.raises(ProgressError, match="listed once"):
        run_module.progress_generation(a, 1, tmp_path / "p", fake, 4, 1, (), 4, modes=(T5, T5))
    assert fake.requests == []


def test_a_committed_generation_without_the_asked_modes_is_kept_and_says_so(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, _ = runs
    evaluate(a, tmp_path / "p", ())
    done, _ = evaluate(a, tmp_path / "p", (T5,))
    assert done["skipped"] is True
    assert "missing: sample:t500000" in done["modes"]
    assert "t500000" not in "".join(m["match_id"] for m in done["matches"])
    evaluate(a, tmp_path / "q", (T5,))
    kept, _ = evaluate(a, tmp_path / "q", (T5,))
    assert "modes" not in kept
