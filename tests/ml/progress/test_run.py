"""The progress command's run: reads, refusals, the atomic commit, the progress root (ADR-017)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import sha256_hex
from areneon_ml.learner.config import read_run_config
from areneon_ml.loop import layout
from areneon_ml.loop.play import PlayError
from areneon_ml.loop.records import policy_id
from areneon_ml.progress import run as run_module
from areneon_ml.progress.records import ProgressRecordError, read_progress_record
from areneon_ml.progress.run import (
    SEEN_LABEL,
    PreviousPair,
    ProgressError,
    ProgressRequest,
    committed_generations,
    max_pairs,
    progress_lock,
    progress_root,
    summary_of,
    verify_progress_dir,
)
from areneon_ml.progress.seeds import progress_game_seed

from tests.ml.learner.support import tiny_config
from tests.ml.loop.support import make_loop, reseal
from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
)


def progress_generation(  # noqa: PLR0913, PLR0917
    run_dir: Path,
    generation: int,
    root: Path,
    player: Any,
    games: int = 2,
    record_games: int = 1,
    anchors: tuple[int, ...] = (),
    anchor_games: int = 4,
    **kwargs: Any,
) -> dict[str, Any]:
    """The command's function with the fixture dataset's size: 2 games, no anchors unless asked."""
    found: dict[str, Any] = run_module.progress_generation(
        run_dir, generation, root, player, games, record_games, anchors, anchor_games, **kwargs
    )
    return found


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A committed run of two generations; the tests only read it (each copies what it edits)."""
    run_dir = tmp_path_factory.mktemp("run") / "run"
    make_run(run_dir, generations=2)
    return run_dir


def copied(run: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run"
    shutil.copytree(run, target)
    return target


def player(run: Path, change: Any = None) -> FakeProgressPlayer:
    return FakeProgressPlayer(dataset_config(), committed_manifest(run, 0), change)


def test_the_progress_root_is_the_sibling_and_never_inside_the_run(tmp_path: Path) -> None:
    run = tmp_path / "r"
    run.mkdir()
    assert progress_root(run, None) == (tmp_path / "r.progress").resolve()
    assert progress_root(run, tmp_path / "elsewhere") == (tmp_path / "elsewhere").resolve()
    with pytest.raises(ProgressError, match="inside the run"):
        progress_root(run, run / "progress")
    with pytest.raises(ProgressError, match="inside the run"):
        progress_root(run, run)
    with pytest.raises(ProgressError, match="contain the run"):
        progress_root(run, tmp_path)


def test_a_generation_is_evaluated_and_committed_whole(run: Path, tmp_path: Path) -> None:
    root = tmp_path / "p"
    fake = player(run)
    done = progress_generation(run, 1, root, fake, games=2, record_games=1)
    assert done["skipped"] is False
    assert fake.requests[0].previous is None  # the previous-generation matches are opt-in
    assert [m["match_id"] for m in done["matches"]] == [
        "actor-rule-argmax",
        "actor-rule-sample",
        "random-legal-argmax",
        "random-legal-sample",
    ]
    target = root / "g000001"
    assert sorted(p.name for p in target.iterdir()) == ["games", "manifest.json", "progress.json"]
    assert sorted(p.name for p in (target / "games").iterdir()) == [
        f"{m}-0.json" for m in done_ids(done)
    ]
    assert sorted(p.name for p in root.iterdir()) == [".lock", "g000001"]  # no partial left
    manifest = layout.read_json(target / "manifest.json")
    assert manifest["files"] == layout.hash_tree(target)
    (request,) = fake.requests
    assert request.out_dir.name == ".g000001.partial"
    assert request.games == 2
    assert request.generation == 1
    assert request.policy_id_sample != request.policy_id_argmax
    assert request.manifest_sha256 == sha256_hex(
        (layout.generation_dir(run, 1) / layout.MANIFEST).read_bytes()
    )
    # The run directory was only read.
    assert not any(p.name.startswith("progress") for p in run.iterdir())


def done_ids(done: dict[str, Any]) -> list[str]:
    return [m["match_id"] for m in done["matches"]]


def test_a_committed_directory_is_verified_and_never_rewritten(run: Path, tmp_path: Path) -> None:
    root = tmp_path / "p"
    progress_generation(run, 1, root, player(run), games=2, record_games=1)
    before = (root / "g000001" / "manifest.json").read_bytes()
    again = player(run)
    done = progress_generation(run, 1, root, again, games=2, record_games=1)
    assert done["skipped"] is True
    assert again.requests == []
    assert (root / "g000001" / "manifest.json").read_bytes() == before
    # A changed committed file is caught when it is verified, not skipped.
    progress = root / "g000001" / "progress.json"
    progress.write_bytes(progress.read_bytes().replace(b'"turns":5', b'"turns":6', 1))
    with pytest.raises(layout.RunDirError, match="differ from its manifest"):
        progress_generation(run, 1, root, again, games=2, record_games=1)


def test_a_refusal_commits_nothing_and_removes_the_partial_directory(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"

    class Failing(FakeProgressPlayer):
        def progress(self, request: ProgressRequest) -> dict[str, Any]:
            super().progress(request)
            raise PlayError("incident")

    with pytest.raises(PlayError):
        progress_generation(
            run, 1, root, Failing(dataset_config(), committed_manifest(run, 0)), 2, 1
        )
    assert sorted(p.name for p in root.iterdir()) == [".lock"]


def test_a_stale_partial_directory_is_deleted_at_start(run: Path, tmp_path: Path) -> None:
    root = tmp_path / "p"
    (root / ".g000009.partial").mkdir(parents=True)
    (root / ".g000009.partial" / "x").write_text("half")
    progress_generation(run, 1, root, player(run), games=2, record_games=1)
    assert sorted(p.name for p in root.iterdir()) == [".lock", "g000001"]


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        (
            "another generation's manifest hash",
            lambda out, r: rewrite(
                out / "progress.json", lambda d: d.update(generation_manifest_sha256="0" * 64)
            ),
            "is not the record asked for",
        ),
        (
            "another frozen pair",
            lambda out, r: rewrite(
                out / "progress.json", lambda d: d["checkpoints"].update(preview="0" * 64)
            ),
            "frozen pair's",
        ),
        (
            "another game count",
            lambda out, r: rewrite(
                out / "progress.json", lambda d: d["plan"].update(recorded_games_per_match=0)
            ),
            "recorded_games_per_match long",
        ),
        (
            "another run",
            lambda out, r: rewrite(out / "progress.json", lambda d: d.update(run_id="other-run")),
            "another run",
        ),
        (
            "an extra game file",
            lambda out, r: (out / "games" / "stray.json").write_text("{}"),
            "other files than the record lists",
        ),
        (
            "a missing game file",
            lambda out, r: (out / "games" / "actor-rule-argmax-0.json").unlink(),
            "No such file",
        ),
        (
            "a game of another match",
            lambda out, r: rewrite(
                out / "games" / "actor-rule-argmax-0.json",
                lambda d: d.update(match_id="random-legal-sample"),
            ),
            "is not the game it is named for",
        ),
        (
            "a game whose bytes differ from the listed hash",
            lambda out, r: rewrite(
                out / "games" / "actor-rule-argmax-0.json",
                lambda d: d["decisions"][0].update(refused="x"),
            ),
            "differ from game_record_hashes",
        ),
        (
            "a game with a bad log hash",
            lambda out, r: rewrite(
                out / "games" / "actor-rule-argmax-0.json",
                lambda d: d["omniscient_log"].append("|x"),
            ),
            "omniscient_log_sha256",
        ),
    ],
)
def test_the_files_the_simulator_side_wrote_are_checked_before_anything_commits(
    run: Path, tmp_path: Path, name: str, change: Any, message: str
) -> None:
    root = tmp_path / "p"
    with pytest.raises((ProgressError, ProgressRecordError, OSError), match=message):
        progress_generation(run, 1, root, player(run, change), games=2, record_games=1)
    assert sorted(p.name for p in root.iterdir()) == [".lock"], name


def rewrite(path: Path, change: Any) -> None:
    value = json.loads(path.read_text())
    change(value)
    layout.write_json(path, value)


def test_a_run_without_a_team_dataset_has_no_validation_split(tmp_path: Path) -> None:
    run_dir = tmp_path / "plain"
    make_loop(run_dir, tiny_config()).run(1)
    with pytest.raises(ProgressError, match="no team dataset"):
        progress_generation(run_dir, 1, tmp_path / "p", player(run_dir), 2, 1)


def test_a_generation_that_is_not_committed_is_refused(run: Path, tmp_path: Path) -> None:
    for generation in (3, -1):
        with pytest.raises(ProgressError, match="is not committed"):
            progress_generation(run, generation, tmp_path / "p", player(run), 2, 1)


def test_more_games_than_pairs_is_refused_before_the_worker_starts(
    run: Path, tmp_path: Path
) -> None:
    fake = player(run)
    with pytest.raises(ProgressError, match="at most 60"):
        progress_generation(run, 1, tmp_path / "p", fake, 122, 1)  # 61 pairs
    assert fake.requests == []
    assert sorted(p.name for p in (tmp_path / "p").iterdir()) == [".lock"]
    progress_generation(run, 1, tmp_path / "p", fake, 120, 0)  # 60 pairs: the limit is allowed


def test_a_dataset_manifest_that_cannot_be_read_refuses(run: Path, tmp_path: Path) -> None:
    copy = copied(run, tmp_path)
    config = copy / "run-config.json"
    rewrite(config, lambda d: d["team_dataset"].update(path=str(tmp_path / "nowhere")))
    with pytest.raises(ProgressError, match="cannot read the validation split"):
        max_pairs(read_run_config(layout.read_json(config)))


def test_sizes_are_checked_first(run: Path, tmp_path: Path) -> None:
    for games, record in ((0, 0), (2, 3), (2, -1), (3, 1)):
        with pytest.raises(ProgressError, match="even number of at least 2"):
            progress_generation(run, 1, tmp_path / "p", player(run), games, record)


def test_a_run_that_is_not_synthetic_is_refused(run: Path, tmp_path: Path) -> None:
    copy = copied(run, tmp_path)
    path = layout.generation_dir(copy, 2) / "records" / "battle.json"
    rewrite(path, lambda d: d.update(data_class="private"))
    reseal(copy, 2)
    with pytest.raises(ProgressError, match="synthetic"):
        progress_generation(copy, 2, tmp_path / "p", player(copy), 2, 1)


def test_records_that_name_another_checkpoint_than_the_manifest_are_refused(
    run: Path, tmp_path: Path
) -> None:
    copy = copied(run, tmp_path)
    path = layout.generation_dir(copy, 2) / "records" / "preview.json"
    rewrite(path, lambda d: d.update(checkpoint_id="0" * 64))
    reseal(copy, 2)
    with pytest.raises(ProgressError, match="names another checkpoint"):
        progress_generation(copy, 2, tmp_path / "p", player(copy), 2, 1)


def test_a_generation_under_another_run_config_is_refused(run: Path, tmp_path: Path) -> None:
    copy = copied(run, tmp_path)
    manifest = layout.generation_dir(copy, 2) / layout.MANIFEST
    rewrite(manifest, lambda d: d.update(run_config_hash="0" * 64))
    with pytest.raises(ProgressError, match="another run config"):
        progress_generation(copy, 2, tmp_path / "p", player(copy), 2, 1)


def test_a_manifest_without_a_frozen_pair_is_refused(run: Path, tmp_path: Path) -> None:
    copy = copied(run, tmp_path)
    manifest = layout.generation_dir(copy, 2) / layout.MANIFEST
    rewrite(manifest, lambda d: d.update(checkpoints={"battle": "a" * 64}))
    with pytest.raises(ProgressError, match="names no frozen pair"):
        progress_generation(copy, 2, tmp_path / "p", player(copy), 2, 1)


def test_a_progress_seed_equal_to_a_training_seed_is_refused(run: Path, tmp_path: Path) -> None:
    copy = copied(run, tmp_path)
    config = dataset_config()
    plan_file = layout.generation_dir(copy, 1) / "plan.json"
    plan = layout.read_json(plan_file)
    plan[0]["seed"] = progress_game_seed(config.run_key, 0)  # the one pair of 2 games
    layout.write_json(plan_file, plan)
    reseal(copy, 1)
    with pytest.raises(ProgressError, match="game 0 equals a seed"):
        progress_generation(copy, 2, tmp_path / "p", player(copy), 2, 1)


def test_the_team_set_must_be_the_one_the_manifest_names(run: Path, tmp_path: Path) -> None:
    copy = copied(run, tmp_path)
    path = layout.generation_dir(copy, 2) / "team-set.json"
    rewrite(path, lambda d: d["teams"].append("f" * 64))
    manifest = layout.generation_dir(copy, 2) / layout.MANIFEST
    rewrite(manifest, lambda d: d.update(files=layout.hash_tree(path.parent)))
    with pytest.raises(ProgressError, match=r"team-set\.json differs"):
        progress_generation(copy, 2, tmp_path / "p", player(copy), 2, 1)


def test_two_commands_never_share_a_root(tmp_path: Path) -> None:
    root = tmp_path / "p"
    with progress_lock(root), pytest.raises(ProgressError, match="holds"), progress_lock(root):
        pass
    with progress_lock(root):  # released when the holder leaves
        assert (root / ".lock").exists()


def test_a_committed_directory_of_another_generation_or_run_is_refused(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    progress_generation(run, 1, root, player(run), games=2, record_games=1)
    # g000001's files presented as g000002: its manifest and record are not generation 2's.
    shutil.copytree(root / "g000001", root / "g000002")
    with pytest.raises(layout.RunDirError, match="not generation 2"):
        progress_generation(run, 2, root, player(run), games=2, record_games=1)
    # Its record names another committed generation.
    other = tmp_path / "q"
    progress_generation(run, 2, other, player(run), games=2, record_games=1)
    target = other / "g000002"
    rewrite(target / "progress.json", lambda d: d.update(generation=1))
    rewrite(target / layout.MANIFEST, lambda d: d.update(files=layout.hash_tree(target)))
    with pytest.raises(ProgressError, match="not of the committed generation"):
        verify_progress_dir(target, dataset_config(), run, 2)
    rewrite(target / layout.MANIFEST, lambda d: d.update(run_id="other"))
    with pytest.raises((ProgressError, ProgressRecordError)):
        verify_progress_dir(target, dataset_config(), run, 2)


def test_every_committed_generation_is_listed_and_summarised(run: Path, tmp_path: Path) -> None:
    assert committed_generations(run) == [0, 1, 2]
    assert committed_generations(tmp_path) == []
    root = tmp_path / "p"
    done = progress_generation(run, 0, root, player(run), games=4, record_games=0)
    first = done["matches"][0]
    assert first["interval"] == "conditional on the fixed validation plan"
    assert first["wilson_low_ppm"] <= first["score_ppm"] <= first["wilson_high_ppm"]
    record = read_progress_record(layout.read_json(root / "g000000" / "progress.json"))
    assert summary_of(record) == done["matches"]
    assert not list((root / "g000000").rglob("*-0.json"))  # --record-games 0 stores no game


def previous_of(run: Path, generation: int) -> dict[str, str]:
    checkpoints = layout.read_json(layout.generation_dir(run, generation) / layout.MANIFEST)[
        "checkpoints"
    ]
    assert isinstance(checkpoints, dict)
    return dict(checkpoints)


def test_generation_one_plays_the_previous_pair_and_the_rows_carry_its_checkpoint_ids(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    fake = player(run)
    done = progress_generation(
        run, 1, root, fake, games=2, record_games=1, anchors=(), previous_matches=True
    )
    (request,) = fake.requests
    zero = previous_of(run, 0)
    assert request.previous is not None
    assert (request.previous.preview, request.previous.battle) == (zero["preview"], zero["battle"])
    assert request.previous.policy_id_sample == policy_id(zero["preview"], zero["battle"], "sample")
    assert request.previous.policy_id_argmax == policy_id(zero["preview"], zero["battle"], "argmax")
    assert done["previous"] == "played"
    record = read_progress_record(layout.read_json(root / "g000001" / "progress.json"))
    rows = {m["match_id"]: m for m in record["matches"] if m["opponent"]["kind"] == "previous"}
    assert sorted(rows) == ["previous-argmax", "previous-sample"]
    for mode, match in (("sample", rows["previous-sample"]), ("argmax", rows["previous-argmax"])):
        assert match["mode"] == mode
        assert match["opponent"]["checkpoints"] == zero
        assert match["opponent"]["policy_id"] == request.previous.__dict__[f"policy_id_{mode}"]
        assert match["policy_id"] == policy_id(request.preview, request.battle, mode)
        assert match["policy_id"] != match["opponent"]["policy_id"]
    # The summary labels the previous rows as the seen, same-lineage comparison they are.
    labelled = {m["match_id"]: m.get("comparison") for m in done["matches"]}
    assert labelled["previous-sample"] == SEEN_LABEL
    assert labelled["actor-rule-sample"] is None


def test_generation_zero_has_no_previous_generation_and_says_so(run: Path, tmp_path: Path) -> None:
    fake = player(run)
    done = progress_generation(run, 0, tmp_path / "p", fake, games=2, record_games=1)
    assert fake.requests[0].previous is None
    assert done["previous"].startswith("none: generation 0")
    assert not [m for m in done["matches"] if m["match_id"].startswith("previous")]


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        (
            "another previous pair",
            lambda d: [
                m["opponent"]["checkpoints"].update(preview="0" * 64)
                for m in d["matches"]
                if m["opponent"]["kind"] == "previous"
            ],
            "previous generation's pair|policy_id is not its checkpoints",
        ),
        (
            "a consistent substitution: another pair, its own policy ids",
            lambda d: [
                m["opponent"].update(
                    checkpoints={"preview": "0" * 64, "battle": "1" * 64},
                    policy_id=policy_id("0" * 64, "1" * 64, m["mode"]),
                )
                for m in d["matches"]
                if m["opponent"]["kind"] == "previous"
            ],
            "not the previous generation's frozen pair",
        ),
        (
            "only one previous mode",
            lambda d: d.update(
                matches=[m for m in d["matches"] if m["match_id"] != "previous-argmax"],
                games=[g for g in d["games"] if g["match_id"] != "previous-argmax"],
            ),
            "are previous-sample and previous-argmax",
        ),
    ],
)
def test_wrong_previous_matches_are_refused_before_anything_commits(
    run: Path, tmp_path: Path, name: str, change: Any, message: str
) -> None:
    def hook(out: Path, request: ProgressRequest) -> None:
        rewrite(out / "progress.json", change)

    root = tmp_path / "p"
    with pytest.raises((ProgressError, ProgressRecordError), match=message):
        progress_generation(
            run,
            1,
            root,
            FakeProgressPlayer(dataset_config(), committed_manifest(run, 0), hook),
            games=2,
            record_games=1,
            previous_matches=True,
        )
    assert sorted(p.name for p in root.iterdir()) == [".lock"], name


def test_previous_matches_at_generation_zero_are_refused(run: Path, tmp_path: Path) -> None:
    class Eager(FakeProgressPlayer):
        def progress(self, request: ProgressRequest) -> dict[str, Any]:
            first = ProgressRequest(**{**request.__dict__, "previous": None})
            two = FakeProgressPlayer(self.config, self.manifest)
            two.progress(
                ProgressRequest(**{**first.__dict__, "previous": fake_previous(request.preview)})
            )
            return {}

    with pytest.raises(ProgressError, match="asks for no previous-generation matches"):
        progress_generation(
            run, 0, tmp_path / "p", Eager(dataset_config(), committed_manifest(run, 0)), 2, 1
        )


def fake_previous(preview: str) -> PreviousPair:
    return PreviousPair(preview, preview, policy_id(preview, preview, "sample"), "x")


def test_a_root_without_previous_matches_is_verified_and_kept_whatever_the_flag(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    progress_generation(run, 1, root, player(run), games=2, record_games=1)
    before = (root / "g000001" / "manifest.json").read_bytes()
    for flag in (False, True):
        again = player(run)
        done = progress_generation(
            run, 1, root, again, games=2, record_games=1, previous_matches=flag
        )
        assert done["skipped"] is True
        assert again.requests == []
        assert done["previous"].startswith("none: previous-generation matches are opt-in")
    assert (root / "g000001" / "manifest.json").read_bytes() == before


def test_the_flag_at_generation_zero_changes_nothing(run: Path, tmp_path: Path) -> None:
    fake = player(run)
    done = progress_generation(
        run, 0, tmp_path / "p", fake, games=2, record_games=1, previous_matches=True
    )
    assert fake.requests[0].previous is None
    assert done["previous"].startswith("none: generation 0")


def test_a_committed_root_with_previous_matches_is_skipped_and_reported_played(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    progress_generation(run, 1, root, player(run), 2, 1, previous_matches=True)
    done = progress_generation(run, 1, root, player(run), 2, 1)
    assert done["skipped"] is True
    assert done["previous"] == "played"


def test_two_runs_give_byte_identical_records_with_the_previous_matches(
    run: Path, tmp_path: Path
) -> None:
    for name in ("a", "b"):
        progress_generation(
            run, 1, tmp_path / name, player(run), games=2, record_games=1, previous_matches=True
        )
    a, b = (sorted((tmp_path / n / "g000001").rglob("*.json")) for n in ("a", "b"))
    assert [p.relative_to(tmp_path / "a") for p in a] == [p.relative_to(tmp_path / "b") for p in b]
    assert all(x.read_bytes() == y.read_bytes() for x, y in zip(a, b, strict=True))


def anchor_rows(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["match_id"]: m for m in record["matches"] if m["opponent"]["kind"] == "anchor"}


def test_an_anchor_is_the_fixed_earlier_generations_pair_played_from_both_seats(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    fake = player(run)
    # Generation 1 is not earlier than itself: asking for it as an anchor leaves it out.
    done = progress_generation(
        run, 1, root, fake, games=2, record_games=1, anchors=(0, 1), anchor_games=4
    )
    (request,) = fake.requests
    zero = previous_of(run, 0)
    assert [(a.generation, a.preview, a.battle) for a in request.anchors] == [
        (0, zero["preview"], zero["battle"])
    ]
    assert request.anchors[0].policy_id_sample == policy_id(
        zero["preview"], zero["battle"], "sample"
    )
    assert request.anchor_games == 4
    assert done["anchors"] == "played: g0; skipped (not earlier than this generation): g1"
    record = read_progress_record(layout.read_json(root / "g000001" / "progress.json"))
    found = anchor_rows(record)
    assert sorted(found) == ["anchor-g0-argmax", "anchor-g0-sample"]
    for mode, match in (
        ("sample", found["anchor-g0-sample"]),
        ("argmax", found["anchor-g0-argmax"]),
    ):
        assert match["games"] == 4
        assert match["opponent"] == {
            "kind": "anchor",
            "generation": 0,
            "policy_id": policy_id(zero["preview"], zero["battle"], mode),
            "checkpoints": zero,
        }
    rows = [r for r in record["games"] if r["match_id"] == "anchor-g0-sample"]
    assert [(r["game"], r["pair"], r["candidate_seat"]) for r in rows] == [
        (0, 0, "p1"),
        (1, 0, "p2"),
        (2, 1, "p1"),
        (3, 1, "p2"),
    ]
    assert record["plan"]["anchor_games_per_match"] == 4
    assert record["plan"]["games_per_match"] == 2
    assert record["teams"]["pairs_used"] == 2
    summary = {m["match_id"]: m for m in done["matches"]}["anchor-g0-sample"]
    assert summary["comparison"] == SEEN_LABEL
    assert summary["score_ppm"] == 500_000
    baseline = {m["match_id"]: m for m in done["matches"]}["actor-rule-sample"]
    assert "comparison" not in baseline


def test_no_anchor_is_played_at_or_below_its_own_generation(run: Path, tmp_path: Path) -> None:
    fake = player(run)
    done = progress_generation(
        run, 0, tmp_path / "p", fake, games=2, record_games=1, anchors=(0, 5), anchor_games=4
    )
    assert fake.requests[0].anchors == ()
    assert fake.requests[0].anchor_games == 0
    assert done["anchors"].startswith("none requested or played; skipped")
    assert not [m for m in done["matches"] if m["match_id"].startswith("anchor")]


@pytest.mark.parametrize(
    ("anchors", "anchor_games", "message"),
    [
        ((0,), 3, "even number of at least 2"),
        ((0,), 0, "even number of at least 2"),
        ((-1,), 4, "generation numbers of 0 or more"),
        ((0,), 10_000, "validation team pairs"),
    ],
)
def test_wrong_anchor_arguments_are_refused_before_the_worker_starts(
    run: Path, tmp_path: Path, anchors: tuple[int, ...], anchor_games: int, message: str
) -> None:
    fake = player(run)
    with pytest.raises(ProgressError, match=message):
        progress_generation(
            run, 1, tmp_path / "p", fake, 2, 1, anchors=anchors, anchor_games=anchor_games
        )
    assert fake.requests == []


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        (
            "another anchor pair",
            lambda d: [
                m["opponent"]["checkpoints"].update(preview="0" * 64)
                for m in d["matches"]
                if m["opponent"]["kind"] == "anchor"
            ],
            "policy_id is not its checkpoints",
        ),
        (
            "a consistent substitution: another pair, its own policy ids",
            lambda d: [
                m["opponent"].update(
                    checkpoints={"preview": "0" * 64, "battle": "1" * 64},
                    policy_id=policy_id("0" * 64, "1" * 64, m["mode"]),
                )
                for m in d["matches"]
                if m["opponent"]["kind"] == "anchor"
            ],
            "not generation 0's pair",
        ),
        (
            "only one anchor mode",
            lambda d: d.update(
                matches=[m for m in d["matches"] if m["match_id"] != "anchor-g0-argmax"],
                games=[g for g in d["games"] if g["match_id"] != "anchor-g0-argmax"],
            ),
            "both sample and argmax",
        ),
        (
            "anchors that were not asked for",
            lambda d: d.update(
                matches=[m for m in d["matches"] if m["opponent"]["kind"] != "anchor"],
                games=[g for g in d["games"] if not g["match_id"].startswith("anchor")],
                plan={k: v for k, v in d["plan"].items() if k != "anchor_games_per_match"},
                teams={**d["teams"], "pairs_used": 1},
            ),
            "not the anchors asked for",
        ),
    ],
)
def test_wrong_anchor_matches_are_refused_before_anything_commits(
    run: Path, tmp_path: Path, name: str, change: Any, message: str
) -> None:
    def hook(out: Path, request: ProgressRequest) -> None:
        rewrite(out / "progress.json", change)

    root = tmp_path / "p"
    with pytest.raises((ProgressError, ProgressRecordError), match=message):
        progress_generation(
            run,
            1,
            root,
            FakeProgressPlayer(dataset_config(), committed_manifest(run, 0), hook),
            games=2,
            record_games=1,
            anchors=(0,),
            anchor_games=4,
        )
    assert sorted(p.name for p in root.iterdir()) == [".lock"], name


def test_a_committed_anchor_directory_is_verified_kept_and_a_missing_anchor_is_said(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    progress_generation(run, 1, root, player(run), 2, 1, anchors=(0,), anchor_games=4)
    before = (root / "g000001" / "manifest.json").read_bytes()
    again = player(run)
    done = progress_generation(run, 1, root, again, 2, 1, anchors=(0,), anchor_games=4)
    assert done["skipped"] is True
    assert again.requests == []
    assert done["anchors"] == "played: g0"
    # An anchor asked for later that the directory lacks is reported, never added.
    none = tmp_path / "q"
    progress_generation(run, 1, none, player(run), 2, 1)
    said = progress_generation(run, 1, none, player(run), 2, 1, anchors=(0,), anchor_games=4)
    assert said["skipped"] is True
    assert said["anchors"].startswith("none requested or played; missing: g0")
    assert (root / "g000001" / "manifest.json").read_bytes() == before


def test_two_runs_give_byte_identical_records_with_the_anchor_matches(
    run: Path, tmp_path: Path
) -> None:
    for name in ("a", "b"):
        progress_generation(
            run, 1, tmp_path / name, player(run), 2, 1, anchors=(0,), anchor_games=4
        )
    a, b = (sorted((tmp_path / n / "g000001").rglob("*.json")) for n in ("a", "b"))
    assert all(x.read_bytes() == y.read_bytes() for x, y in zip(a, b, strict=True))
    assert any("anchor-g0-sample-3" in p.name or "anchor-g0-sample-0" in p.name for p in a)


def test_a_stored_anchor_record_with_a_wrong_opponent_is_refused_on_verification(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    progress_generation(run, 1, root, player(run), 2, 1, anchors=(0,), anchor_games=4)
    config = read_run_config(layout.read_json(run / layout.CONFIG))
    record = layout.read_json(root / "g000001" / "progress.json")
    for match in record["matches"]:
        if match["opponent"]["kind"] == "anchor":
            match["opponent"]["checkpoints"]["battle"] = "0" * 64
            match["opponent"]["policy_id"] = policy_id(
                match["opponent"]["checkpoints"]["preview"], "0" * 64, match["mode"]
            )
    layout.write_json(root / "g000001" / "progress.json", record)
    with pytest.raises(layout.RunDirError):
        verify_progress_dir(root / "g000001", config, run, 1)  # the manifest hash no longer holds
    target = root / "g000001"
    manifest = layout.read_json(target / layout.MANIFEST)
    manifest["files"] = layout.hash_tree(target)
    layout.write_json(target / layout.MANIFEST, manifest)
    with pytest.raises(ProgressError, match="not generation 0's pair"):
        verify_progress_dir(root / "g000001", config, run, 1)
