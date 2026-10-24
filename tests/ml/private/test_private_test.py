"""The owner's private test command: guard, root, refusals, the record and the privacy canary.

The simulator side is a stand-in that builds a record from the request, so the command's own
checks (the guard, the private root, the run directory, the class rules, the tie of the record to
the request, the files it writes) run for real while no engine, worker or team is involved. The
"teams" are marker strings in a throwaway ``private-data`` staging directory that the Python
side never opens; nothing here is a real team.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import layout
from areneon_ml.loop.records import policy_id
from areneon_ml.private.owner_session import OwnerSession, OwnerSessionError
from areneon_ml.private.private_root import PrivateRootError
from areneon_ml.private.private_test import (
    LOOKS_NAME,
    REFUSALS,
    ChildRequest,
    NodePrivateTestPlayer,
    PrivateTestError,
    PrivateTestRequest,
    private_test_main,
    run_private_test,
)
from areneon_ml.private.private_test_record import read_private_test_record
from areneon_ml.progress.looks import read_looks, verify_looks

from tests.ml.loop.support import reseal
from tests.ml.progress.support import committed_manifest, make_run, sample

REPO = Path(__file__).resolve().parents[3]
ENGINE_DATA = REPO / "fixtures/showdown/gen9championsvgc2026regmc/engine-data.json"
MARKERS = ("CanaryNickAlpha", "CanaryNickBeta")
TERMINAL = OwnerSession(env={}, is_terminal=lambda: (True, True, True), open_terminal=lambda: None)


def mode_policy(child: ChildRequest, mode: str) -> str:
    return policy_id(child.preview, child.battle, mode)


class FakePlayer:
    """The Node child without a simulator: a record built from the request."""

    def __init__(self, config: Any, manifest: dict[str, Any], change: Any = None) -> None:
        self.config, self.manifest, self.change = config, manifest, change
        self.children: list[ChildRequest] = []

    def play(self, child: ChildRequest) -> dict[str, Any]:
        self.children.append(child)
        r = child.request
        salt = child.salt_path.read_text(encoding="ascii").strip()
        record = copy.deepcopy(sample("private-test-record/valid/staging"))
        generation = self.config.generation
        record.update(
            run_id=self.config.run_key,
            run_config_hash=self.config.hash,
            generation=r.generation,
            generation_manifest_sha256=child.generation_manifest_sha256,
            checkpoints={"preview": child.preview, "battle": child.battle},
            candidate_data_class=child.candidate_data_class,
            version_manifest=self.manifest,
        )
        record["plan"]["max_turns"] = generation.max_turns
        record["plan"]["latency_budget_ms"] = {
            "preview_ms": generation.preview_ms,
            "battle_ms": generation.battle_ms,
        }
        test_set = record["test_set"]
        test_set.update(
            commitment=hashlib.sha256(salt.encode()).hexdigest(),
            source_kind=r.source,
            subset=r.subset,
        )
        if r.source == "snapshot":
            test_set["snapshot_hash"] = r.snapshot_hash
        if r.source == "dataset":
            test_set["dataset_hash"] = self.config.raw["team_dataset"]["dataset_hash"]
        names = ["actor-rule", "random-legal"]
        names += ["damage-aware"] if r.damage_aware_budget_ms is not None else []
        names += ["pair"] if child.opponent is not None else []
        template = record["matches"][0]
        matches = []
        for name in names:
            for mode in ("argmax", "sample"):
                m = copy.deepcopy(template)
                m["match_id"] = f"{name}-{mode}"
                m["mode"] = mode
                m["policy_id"] = mode_policy(child, mode)
                m["opponent_policy_id"] = f"{name}@v1"
                m.pop("opponent_checkpoints", None)
                if name == "pair" and child.opponent is not None:
                    m["opponent_policy_id"] = policy_id(child.opponent[0], child.opponent[1], mode)
                    m["opponent_checkpoints"] = {
                        "preview": child.opponent[0],
                        "battle": child.opponent[1],
                    }
                matches.append(m)
        record["matches"] = sorted(matches, key=lambda m: m["match_id"])
        if r.max_pairs is not None:
            # Every pair split one game each: consistent counts for any number of pairs.
            pairs = r.max_pairs
            record["test_set"]["pairs"] = pairs
            record["test_set"]["team_count"] = max(7, 2 * pairs)
            record["plan"]["games_per_match"] = 2 * pairs
            for m in record["matches"]:
                m.update(games=2 * pairs, wins=pairs, ties=0, losses=pairs, turn_cap_ties=0)
                m.update(forfeits=[], pairs=pairs, pair_score_counts=[0, 0, pairs, 0, 0])
        if self.change is not None:
            self.change(record)
        return {"record": record, "wall_ms": {"fake": 1}}


class Setup:
    def __init__(self, tmp_path: Path) -> None:
        tmp_path.chmod(0o700)
        self.home = tmp_path / "private-data"
        self.root = self.home / "tests"
        self.teams = self.home / "teams"
        for d in (self.home, self.root, self.teams):
            d.mkdir(mode=0o700)
        for i, marker in enumerate(MARKERS):
            (self.teams / f"{marker}-{i}.txt").write_text(f"{marker} (Pikachu) @ Light Ball\n")
        self.run_dir = tmp_path / "run"
        self.config = make_run(self.run_dir, generations=2)
        self.manifest = committed_manifest(self.run_dir, 1)

    def request(self, **over: Any) -> PrivateTestRequest:
        base: dict[str, Any] = {
            "run_dir": self.run_dir,
            "generation": 1,
            "out_root": self.root,
            "engine_data": ENGINE_DATA,
            "source": "staging",
            "subset": "all",
            "teams_dir": self.teams,
            "snapshots_dir": self.home / "snapshots",
            "repo": REPO,
        }
        return PrivateTestRequest(**{**base, **over})

    def player(self, change: Any = None) -> FakePlayer:
        return FakePlayer(self.config, self.manifest, change)

    def make_private(self, run_dir: Path | None = None, generation: int = 1) -> None:
        directory = run_dir or self.run_dir
        for role in ("preview", "battle"):
            path = layout.generation_dir(directory, generation) / "records" / f"{role}.json"
            record = layout.read_json(path)
            record["data_class"] = "private"
            layout.write_json(path, record)
        reseal(directory, generation)


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


def code_of(run: Any) -> str:
    try:
        run()
    except PrivateTestError as error:
        shown = f"private-test: {error.code}"
        return error.code if str(error) in (shown, f"{shown} {error.position}") else "unfixed"
    return "ok"


def files_of(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir())


def test_a_run_writes_one_record_a_salt_and_a_look_and_shows_counts_only(setup: Setup) -> None:
    player = setup.player()
    result = run_private_test(setup.request(), session=TERMINAL, player=player)
    names = files_of(setup.root)
    assert names == sorted([LOOKS_NAME, result["record"], result["salt_file"]])
    for name in names:
        assert stat.S_IMODE((setup.root / name).stat().st_mode) == 0o600
    record = read_private_test_record(json.loads((setup.root / result["record"]).read_text()))
    assert record["data_class"] == "private"
    assert result["label"] == "private test set; not the Training gate; not validation"
    assert [m["match_id"] for m in result["matches"]] == [m["match_id"] for m in record["matches"]]
    assert result["looks"] == 1
    assert read_looks(setup.root / LOOKS_NAME).count == 1
    assert result["candidate_data_class"] == "synthetic"
    shown = json.dumps(result)
    assert "pair_score_sum" in shown
    assert "low_ppm" in shown
    child = player.children[0]
    assert child.candidate_data_class == "synthetic"
    assert child.opponent is None


def test_the_same_salt_and_inputs_give_the_same_record_bytes_and_another_salt_another_commitment(
    setup: Setup,
) -> None:
    first = run_private_test(setup.request(), session=TERMINAL, player=setup.player())
    before = (setup.root / first["record"]).read_bytes()
    second = run_private_test(
        setup.request(salt_file=first["salt_file"]), session=TERMINAL, player=setup.player()
    )
    assert second["record"] == first["record"]
    assert second["record_hash"] == first["record_hash"]
    assert (setup.root / first["record"]).read_bytes() == before
    assert second["looks"] == 2  # a rerun is a look of its own
    other = run_private_test(setup.request(), session=TERMINAL, player=setup.player())
    assert other["test_set"]["commitment"] != first["test_set"]["commitment"]
    assert other["record"] != first["record"]
    assert len(verify_looks((setup.root / LOOKS_NAME).read_bytes()).head) == 64


def test_refuses_in_an_agent_session_and_on_non_terminal_stdio_before_anything_is_read(
    setup: Setup, tmp_path: Path
) -> None:
    player = setup.player()
    missing = setup.request(run_dir=tmp_path / "nowhere", out_root=tmp_path / "no-such-root")
    for session, code in (
        (
            OwnerSession(
                env={"CLAUDECODE": "1"},
                is_terminal=lambda: (True, True, True),
                open_terminal=lambda: None,
            ),
            "agent_session",
        ),
        (
            OwnerSession(
                env={"AI_AGENT": "x"},
                is_terminal=lambda: (True, True, True),
                open_terminal=lambda: None,
            ),
            "agent_session",
        ),
        (
            OwnerSession(
                env={}, is_terminal=lambda: (True, False, True), open_terminal=lambda: None
            ),
            "not_a_terminal",
        ),
        (
            OwnerSession(
                env={}, is_terminal=lambda: (False, True, True), open_terminal=lambda: None
            ),
            "not_a_terminal",
        ),
    ):
        with pytest.raises(OwnerSessionError) as caught:
            run_private_test(missing, session=session, player=player)
        assert caught.value.code == code
    assert player.children == []
    assert files_of(setup.root) == []


def test_main_shows_only_fixed_words_and_exit_codes(setup: Setup) -> None:
    shown: list[str] = []
    out: list[bytes] = []
    agent = OwnerSession(
        env={"CLAUDECODE": "1"}, is_terminal=lambda: (True, True, True), open_terminal=lambda: None
    )
    assert (
        private_test_main(
            setup.request(), session=agent, player=setup.player(), out=out.append, err=shown.append
        )
        == 3
    )
    assert shown == ["refusing: an agent session is running this; use your own terminal"]
    assert out == []
    assert (
        private_test_main(
            setup.request(),
            session=TERMINAL,
            player=setup.player(),
            out=out.append,
            err=shown.append,
        )
        == 0
    )
    assert out
    assert out[0].endswith(b"\n")
    assert json.loads(out[0])["record"]
    # A usage error exits 2; a refusal of the root 3; the child's failure 4.
    shown.clear()
    assert (
        private_test_main(
            setup.request(subset="holdout"),
            session=TERMINAL,
            player=setup.player(),
            out=out.append,
            err=shown.append,
        )
        == 2
    )
    assert shown == ["private-test: usage"]
    bad_root = setup.request(out_root=setup.home.parent)
    assert (
        private_test_main(
            bad_root, session=TERMINAL, player=setup.player(), out=out.append, err=shown.append
        )
        == 3
    )

    class Failing:
        def __init__(self, error: Exception) -> None:
            self.error = error

        def play(self, child: ChildRequest) -> dict[str, Any]:
            raise self.error

    shown.clear()
    assert (
        private_test_main(
            setup.request(),
            session=TERMINAL,
            player=Failing(PrivateTestError("team_invalid", 3)),
            out=out.append,
            err=shown.append,
        )
        == 4
    )
    assert shown == ["private-test: team_invalid 3"]
    shown.clear()
    secret = RuntimeError(f"validator said {MARKERS[0]} {setup.teams}")
    assert (
        private_test_main(
            setup.request(),
            session=TERMINAL,
            player=Failing(secret),
            out=out.append,
            err=shown.append,
        )
        == 4
    )
    assert shown == ["private-test: failed"]


def test_refuses_a_root_that_is_not_private(setup: Setup, tmp_path: Path) -> None:
    for root in (tmp_path / "out", tmp_path / "private-data" / "absent", Path("relative")):
        if root == tmp_path / "out":
            root.mkdir(mode=0o700)
        with pytest.raises(PrivateRootError):
            run_private_test(setup.request(out_root=root), session=TERMINAL, player=setup.player())
    loose = setup.home / "loose"
    loose.mkdir(mode=0o755)
    loose.chmod(0o755)
    with pytest.raises(PrivateRootError):
        run_private_test(setup.request(out_root=loose), session=TERMINAL, player=setup.player())


def test_refuses_a_bad_combination_of_flags(setup: Setup) -> None:
    cases = [
        {"source": "staging", "subset": "holdout"},
        {"source": "dataset", "subset": "holdout"},
        {"source": "snapshot", "snapshot_hash": None},
        {"source": "staging", "snapshot_hash": "a" * 64},
        {"source": "snapshot", "snapshot_hash": "xyz"},
        {"source": "other"},
        {"subset": "some"},
        {"generation": -1},
        {"max_pairs": 0},
        {"damage_aware_budget_ms": 0},
        {"opponent_run_dir": setup.run_dir},
        {"opponent_generation": 0},
        {"salt_file": "../salt"},
    ]
    for over in cases:
        assert (
            code_of(
                lambda over=over: run_private_test(
                    setup.request(**over), session=TERMINAL, player=setup.player()
                )
            )
            == "usage"
        ), over


def test_refuses_a_run_that_does_not_verify(setup: Setup, tmp_path: Path) -> None:
    for over in ({"generation": 7}, {"run_dir": tmp_path / "nowhere"}):
        assert (
            code_of(
                lambda over=over: run_private_test(
                    setup.request(**over), session=TERMINAL, player=setup.player()
                )
            )
            == "run_unverified"
        )
    (layout.generation_dir(setup.run_dir, 1) / "records" / "battle.json").write_text("{}")
    assert (
        code_of(lambda: run_private_test(setup.request(), session=TERMINAL, player=setup.player()))
        == "run_unverified"
    )
    assert files_of(setup.root) == []


def test_a_pair_with_private_ancestry_is_tested_on_holdout_or_public_teams_only(
    setup: Setup,
) -> None:
    setup.make_private()
    for over in (
        {"source": "staging"},
        {"source": "snapshot", "snapshot_hash": "a" * 64, "subset": "all"},
    ):
        player = setup.player()
        assert (
            code_of(
                lambda over=over, player=player: run_private_test(
                    setup.request(**over), session=TERMINAL, player=player
                )
            )
            == "private_pair_source"
        )
        assert player.children == []
    assert files_of(setup.root) == []
    held = run_private_test(
        setup.request(source="snapshot", snapshot_hash="a" * 64, subset="holdout"),
        session=TERMINAL,
        player=setup.player(),
    )
    assert held["candidate_data_class"] == "private"
    public = run_private_test(
        setup.request(source="dataset"), session=TERMINAL, player=setup.player()
    )
    assert public["test_set"]["source_kind"] == "dataset"
    assert (
        read_private_test_record(json.loads((setup.root / public["record"]).read_text()))[
            "candidate_data_class"
        ]
        == "private"
    )


def test_plays_another_frozen_pair_and_the_damage_aware_baseline_when_asked(
    setup: Setup, tmp_path: Path
) -> None:
    player = setup.player()
    result = run_private_test(
        setup.request(
            opponent_run_dir=setup.run_dir,
            opponent_generation=0,
            damage_aware_budget_ms=60_000,
            max_pairs=2,
        ),
        session=TERMINAL,
        player=player,
    )
    ids = [m["match_id"] for m in result["matches"]]
    assert ids == [
        "actor-rule-argmax",
        "actor-rule-sample",
        "damage-aware-argmax",
        "damage-aware-sample",
        "pair-argmax",
        "pair-sample",
        "random-legal-argmax",
        "random-legal-sample",
    ]
    child = player.children[0]
    assert child.opponent is not None
    assert child.opponent[:2] != (child.preview, child.battle)
    # The opponent may not be the pair under test, and a private opponent makes the record private.
    same = setup.request(opponent_run_dir=setup.run_dir, opponent_generation=1)
    assert (
        code_of(lambda: run_private_test(same, session=TERMINAL, player=setup.player()))
        == "opponent_unverified"
    )
    gone = setup.request(opponent_run_dir=tmp_path / "nowhere", opponent_generation=0)
    assert (
        code_of(lambda: run_private_test(gone, session=TERMINAL, player=setup.player()))
        == "opponent_unverified"
    )
    setup.make_private(generation=0)
    private_opponent = setup.request(opponent_run_dir=setup.run_dir, opponent_generation=0)
    assert (
        code_of(lambda: run_private_test(private_opponent, session=TERMINAL, player=setup.player()))
        == "private_pair_source"
    )


def test_refuses_a_record_that_is_not_about_the_request(setup: Setup) -> None:
    def other_generation(record: dict[str, Any]) -> None:
        record["generation"] = 0

    def other_checkpoint(record: dict[str, Any]) -> None:
        record["checkpoints"]["battle"] = "e" * 64

    def other_policy(record: dict[str, Any]) -> None:
        record["matches"][0]["policy_id"] = "policy:" + "e" * 64

    def other_matches(record: dict[str, Any]) -> None:
        record["matches"] = record["matches"][:2]

    def other_source(record: dict[str, Any]) -> None:
        record["test_set"]["source_kind"] = "dataset"
        record["test_set"]["dataset_hash"] = "1" * 64

    def other_turns(record: dict[str, Any]) -> None:
        record["plan"]["max_turns"] += 1

    def other_pairs(record: dict[str, Any]) -> None:
        record["test_set"]["pairs"] = 1
        record["plan"]["games_per_match"] = 2

    def broken(record: dict[str, Any]) -> None:
        record["matches"][0]["wins"] += 1

    for change in (
        other_generation,
        other_checkpoint,
        other_policy,
        other_matches,
        other_source,
        other_turns,
        broken,
    ):
        assert (
            code_of(
                lambda change=change: run_private_test(
                    setup.request(), session=TERMINAL, player=setup.player(change)
                )
            )
            == "record_mismatch"
        )
    assert (
        code_of(
            lambda: run_private_test(
                setup.request(max_pairs=2), session=TERMINAL, player=setup.player(other_pairs)
            )
        )
        == "record_mismatch"
    )
    assert (
        code_of(lambda: run_private_test(setup.request(), session=TERMINAL, player=_Garbage()))
        == "record_mismatch"
    )
    assert [
        n for n in files_of(setup.root) if n.startswith("private-test-") and n.endswith(".json")
    ] == []
    assert not (setup.root / LOOKS_NAME).exists()


class _Garbage:
    def play(self, child: ChildRequest) -> dict[str, Any]:
        return {"record": {"not": "a record"}}


def test_a_different_record_of_the_same_name_is_never_overwritten(setup: Setup) -> None:
    first = run_private_test(setup.request(), session=TERMINAL, player=setup.player())
    path = setup.root / first["record"]
    path.chmod(0o600)
    path.write_text("tampered")
    assert (
        code_of(
            lambda: run_private_test(
                setup.request(salt_file=first["salt_file"]), session=TERMINAL, player=setup.player()
            )
        )
        == "record_exists"
    )
    assert path.read_text() == "tampered"


def test_the_salt_file_must_be_a_private_file_of_the_root(setup: Setup) -> None:
    first = run_private_test(setup.request(), session=TERMINAL, player=setup.player())
    salt = setup.root / first["salt_file"]
    for text, mode in (("nothex\n", 0o600), ("ab" * 32, 0o644)):
        salt.chmod(0o600)
        salt.write_text(text)
        salt.chmod(mode)
        assert (
            code_of(
                lambda: run_private_test(
                    setup.request(salt_file=first["salt_file"]),
                    session=TERMINAL,
                    player=setup.player(),
                )
            )
            == "salt_unreadable"
        )
    assert (
        code_of(
            lambda: run_private_test(
                setup.request(salt_file="absent.key"), session=TERMINAL, player=setup.player()
            )
        )
        == "salt_unreadable"
    )


def test_no_marker_reaches_any_output_error_or_file(
    setup: Setup, capsys: pytest.CaptureFixture[str]
) -> None:
    shown: list[str] = []
    out: list[bytes] = []
    assert (
        private_test_main(
            setup.request(),
            session=TERMINAL,
            player=setup.player(),
            out=out.append,
            err=shown.append,
        )
        == 0
    )
    written = {p.name: p.read_text() for p in setup.root.iterdir()}
    surfaces = (
        [b.decode() for b in out]
        + shown
        + list(written)
        + list(written.values())
        + [capsys.readouterr().out, capsys.readouterr().err]
    )
    needles = [*MARKERS, str(setup.teams), *(p.name for p in setup.teams.iterdir())]
    assert [n for n in needles for s in surfaces if n in s] == []
    # Negative control: a leaking surface would be seen.
    assert [n for n in needles if n in "".join(surfaces) + f" {MARKERS[0]}"]
    # Nothing but the three files, and no page, game record or log of a game.
    assert sorted(os.listdir(setup.root)) == sorted(written)
    assert not any(n.endswith((".html", ".jsonl")) and n != LOOKS_NAME for n in written)


def stand_in_node(tmp_path: Path, stderr: str, stdout: str, code: int) -> str:
    script = tmp_path / f"node-{abs(hash((stderr, stdout, code)))}"
    log = tmp_path / "argv.log"
    script.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {log}\n"
        f"printf '%s' '{stdout}'\n"
        f"printf '%s\\n' '{stderr}' >&2\n"
        f"exit {code}\n"
    )
    script.chmod(0o700)
    return str(script)


def test_the_node_child_shows_only_fixed_refusals(setup: Setup, tmp_path: Path) -> None:
    child = ChildRequest(
        setup.request(
            damage_aware_budget_ms=5,
            max_pairs=3,
            source="snapshot",
            snapshot_hash="a" * 64,
            subset="holdout",
        ),
        setup.run_dir / layout.CONFIG,
        "enc-test",
        "3" * 64,
        "1" * 64,
        "2" * 64,
        "private",
        tmp_path / "salt.key",
        ("5" * 64, "6" * 64, "policy:" + "c" * 64, "policy:" + "d" * 64, tmp_path, "enc-other"),
    )

    def play(stderr: str, stdout: str, code: int) -> str:
        player = NodePrivateTestPlayer(
            REPO, stand_in_node(tmp_path, stderr, stdout, code), "python-x"
        )
        try:
            player.play(child)
        except PrivateTestError as error:
            return str(error)
        return "ok"

    assert play("", '{"record": {}}', 0) == "ok"
    argv = (tmp_path / "argv.log").read_text().split("\n")
    for flag in (
        "--candidate-data-class",
        "--snapshot-hash",
        "--max-pairs",
        "--damage-aware-budget-ms",
        "--opponent-pair",
        "--opponent-run-dir",
        "--opponent-encoder-version",
        "--source",
        "--subset",
        "--salt-file",
    ):
        assert flag in argv
    assert argv[argv.index("--policy-id-sample") + 1] == policy_id("1" * 64, "2" * 64, "sample")
    # The flags the child is given are the flags the Node child parses, no more and no fewer.
    source = (REPO / "packages/showdown-bridge/src/private-test-cli.ts").read_text(encoding="utf-8")
    start = source.index("const FLAGS = {")
    flags_block = source[start : source.index("} as const;", start)]
    node_flags = set(re.findall(r'^\s+"?([a-z0-9-]+)"?: \{ type: "string" \}', flags_block, re.M))
    assert {a[2:] for a in argv if a.startswith("--")} == node_flags
    assert play("private-test: team_invalid 2", "", 1) == "private-test: team_invalid 2"
    assert play("private-test: play_failed", "", 1) == "private-test: play_failed"
    for text in (
        f"validator said {MARKERS[0]}",
        "private-test: team_invalid 0",
        "private-test: unheard_of",
        "private-test: team_invalid 2 extra",
        "",
    ):
        shown = play(text, "", 1)
        assert MARKERS[0] not in shown
        assert shown in (
            "private-test: failed",
            "private-test: unheard_of",
        )  # an unknown code is shown as failed
    assert play("", "not json", 0) == "private-test: failed"
    assert play("", "[1]", 0) == "private-test: failed"
    with pytest.raises(PrivateTestError) as caught:
        NodePrivateTestPlayer(REPO, str(tmp_path / "no-such-node"), "python-x").play(child)
    assert str(caught.value) == "private-test: failed"


def test_every_refusal_has_fixed_words_without_a_path_or_value() -> None:
    for code, words in REFUSALS.items():
        assert str(PrivateTestError(code)) == f"private-test: {code}"
        assert "/" not in words
        assert "{" not in words
    assert str(PrivateTestError("no-such-code")) == "private-test: failed"
    assert str(PrivateTestError("team_invalid", 4)) == "private-test: team_invalid 4"
