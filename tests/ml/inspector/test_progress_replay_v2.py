"""Progress-game replays of version 2 records: pairs, anchors, the ablation, version 1 (#951)."""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.inspector import progress_replay as pr
from areneon_ml.inspector import progress_replay_command as command
from areneon_ml.inspector import progress_replay_page as page_module
from areneon_ml.inspector.readers import InspectError
from areneon_ml.loop import layout
from areneon_ml.progress.records import DAMAGE_AWARE_LABEL
from areneon_ml.progress.run import VersusSpec, progress_generation

from tests.ml.inspector.test_progress_replay import (
    GAMES,
    ROOT,
    make_consistent,
    problems,
    table,
)
from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    downgrade_to_v1,
    make_run,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A progress root: generation 0 version 1; generation 1 version 2 with an anchor (g0);
    generation 2 a Preview ablation. Every game is stored (four per match)."""
    base = tmp_path_factory.mktemp("progress-v2")
    run = base / "run"
    config = make_run(run, generations=3)
    root = base / "run.progress"
    player = FakeProgressPlayer(config, committed_manifest(run, 0), make_consistent, previous=False)
    progress_generation(run, 0, root, player, GAMES, GAMES, (), 4)
    downgrade_to_v1(root / "g000000")
    progress_generation(run, 1, root, player, GAMES, GAMES, (0,), 4)
    progress_generation(run, 2, root, player, GAMES, GAMES, (), 4, preview_ablation=True)
    return root


def games(root: Path, generation: int, **kwargs: Any) -> list[pr.ProgressGame]:
    return pr.read_progress_games(root, generation, **kwargs)


def page(game: pr.ProgressGame) -> str:
    return html.unescape(page_module.replay_page(game, page_module.dex_for(table(), game)))


def test_the_games_of_a_pair_are_read_together_and_know_each_other(built: Path) -> None:
    found = games(built, 1, match="actor-rule-sample", games=[2])
    # asking for game 2 brings game 3, its other half, with it
    assert [(g.game, g.pair, g.twin) for g in found] == [(2, 1, 3), (3, 1, 2)]
    assert [g.candidate_seat for g in found] == ["p1", "p2"]


def test_a_game_whose_twin_is_not_stored_has_no_twin(built: Path, tmp_path: Path) -> None:
    root = tmp_path / "run.progress"
    shutil.copytree(built, root)
    target = root / "g000001"
    (target / "games" / "actor-rule-sample-3.json").unlink()
    manifest = layout.read_json(target / layout.MANIFEST)
    manifest["files"] = layout.hash_tree(target)
    layout.write_json(target / layout.MANIFEST, manifest)
    (only,) = games(root, 1, match="actor-rule-sample", games=[2])
    assert (only.game, only.twin) == (2, None)
    assert "The other game of this pair" not in page(only)


def test_a_pair_page_links_to_the_other_game_and_says_which_it_is(built: Path) -> None:
    first, second = games(built, 1, match="random-legal-argmax", games=[0])
    one, two = page(first), page(second)
    assert 'href="random-legal-argmax-1.html"' in one
    assert 'href="random-legal-argmax-0.html"' in two
    assert "Pair 0, game 1 of 2" in one
    assert "Pair 0, game 2 of 2" in two
    assert "each pair is played from both seats with the same teams and battle seed" in one
    # the link is a name made of ids only, never a data string
    assert re.findall(r'href="([^"]*)"', one.split("The other game")[1])[0] == (
        "random-legal-argmax-1.html"
    )


def test_an_anchor_game_names_the_anchor_and_the_label(built: Path) -> None:
    (g, _twin) = games(built, 1, match="anchor-g0-argmax", games=[0])
    text = page(g)
    assert "generation 0's frozen pair, an anchor" in text
    assert page_module.SEEN in text
    assert g.opponent_generation == 0
    # p2 is the anchor in game 0 (the candidate sits in p1)
    assert "p2: the opponent, generation 0's frozen pair, an anchor" in text


def test_a_cross_run_game_names_the_other_run_and_the_label(tmp_path: Path) -> None:
    run, other = tmp_path / "run", tmp_path / "other"
    config = make_run(run, generations=2)
    make_run(other, generations=2, config=dataset_config("other-run", (21, 22)))
    root = tmp_path / "run.progress"
    player = FakeProgressPlayer(config, committed_manifest(run, 0), make_consistent, previous=False)
    progress_generation(
        run,
        1,
        root,
        player,
        GAMES,
        GAMES,
        (),
        4,
        damage_aware_budget_minutes=30,
        versus=VersusSpec(other, 1, 4, "enc-other"),
    )
    (g, _twin) = games(root, 1, match="run-sample", games=[0])
    (dg, _dtwin) = games(root, 1, match="damage-aware-sample", games=[0])
    assert DAMAGE_AWARE_LABEL in page(dg)
    assert page_module.CROSS_RUN not in page(dg)
    text = page(g)
    assert g.opponent_kind == "run"
    assert "another run's frozen pair" in text
    assert page_module.CROSS_RUN in text
    assert page_module.SEEN not in text
    index = tmp_path / "pages"
    command.replay_progress(root, 1, index, repo=ROOT, match="run-sample", games=[0])
    assert f"({page_module.CROSS_RUN})" in (index / "progress-replays" / "index.html").read_text()


def test_an_ablation_game_names_the_candidates_preview(built: Path) -> None:
    found = games(built, 2, match="preview-random-actor-rule-argmax", games=[0])
    assert {g.preview_variant for g in found} == {"random"}
    assert "Team Preview ablation: random Preview" in page(found[0])
    plain = games(built, 2, match="preview-learned-actor-rule-argmax", games=[1])
    assert {g.preview_variant for g in plain} == {"learned"}
    generation = games(built, 2, match="preview-g0-actor-rule-argmax", games=[0])
    assert generation[0].preview_variant == "g0"


def test_a_version_1_game_keeps_its_page_wording_and_has_no_pair(built: Path) -> None:
    (g,) = games(built, 0, match="actor-rule-sample", games=[1])
    assert (g.pair, g.twin) == (None, None)
    text = page(g)
    assert "sample mode against baseline. Synthetic, public-lineage game" in text
    assert "The other game of this pair" not in text
    assert "Pair " not in text.split('id="replay-root"')[0]


def test_the_index_lists_a_version_2_generation_by_pair_and_leaves_version_1_alone(
    built: Path, tmp_path: Path
) -> None:
    out = tmp_path / "pages"
    command.replay_progress(built, 0, out, repo=ROOT, match="actor-rule-sample")
    command.replay_progress(built, 1, out, repo=ROOT, match="anchor-g0-sample", games=[0])
    index = (out / "progress-replays" / "index.html").read_text()
    text = html.unescape(index)
    # version 1: one link per game, the wording the index always had
    assert ">actor-rule-sample game 0</a> · <a" in text
    # version 2: the two games of a pair side by side, the anchor labelled
    assert (
        f"anchor-g0-sample pair 0 ({page_module.SEEN}): "
        '<a href="g000001/anchor-g0-sample-0.html">game 0</a>, '
        '<a href="g000001/anchor-g0-sample-1.html">game 1</a>'
    ) in text
    assert command.both_seat_generations(built, [0, 1, 2, 9]) == {1, 2}


def test_an_unreadable_record_counts_as_version_1_for_the_index(tmp_path: Path) -> None:
    (tmp_path / "g000003").mkdir()
    (tmp_path / "g000003" / "progress.json").write_text("not json")
    assert command.both_seat_generations(tmp_path, [3, 4]) == set()


def test_a_v2_page_stays_inside_the_page_security_rules(built: Path) -> None:
    (g, _twin) = games(built, 1, match="anchor-g0-argmax", games=[0])
    raw = page_module.replay_page(g, page_module.dex_for(table(), g))
    assert problems(raw) == []


@pytest.fixture(scope="module")
def damage_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generation 1 with the damage-aware matches (ADR-018 slice 4); every game stored."""
    base = tmp_path_factory.mktemp("progress-damage")
    run = base / "run"
    config = make_run(run, generations=2)
    root = base / "run.progress"
    player = FakeProgressPlayer(config, committed_manifest(run, 0), make_consistent, previous=False)
    progress_generation(run, 1, root, player, GAMES, GAMES, (), 4, damage_aware_budget_minutes=30)
    return root


def test_a_damage_aware_game_is_named_and_carries_its_counts_on_the_page(
    damage_root: Path,
) -> None:
    found = games(damage_root, 1, match="damage-aware-sample", games=[0])
    assert [g.baseline["seat"] for g in found if g.baseline] == ["p2", "p1"]
    first, second = found
    text = page(first)
    assert "the damage-aware scripted baseline" in text
    assert "seen scripted baseline: never held-out evidence" in text
    counts = first.baseline["counts"]  # type: ignore[index]
    assert f"{counts['decisions']} decisions, {counts['queries']} probe queries" in text
    assert "damage-aware@v1" in text
    assert "the damage-aware scripted baseline" in page(second)
    other = games(damage_root, 1, match="actor-rule-sample", games=[0])[0]
    assert other.baseline is None
    assert "damage-aware" not in page(other)


def test_the_index_labels_the_damage_aware_match(damage_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    command.replay_progress(damage_root, 1, out, repo=ROOT, match="damage-aware-argmax")
    index = html.unescape((out / "progress-replays" / "index.html").read_text())
    assert "seen scripted baseline: never held-out evidence" in index


def test_a_stored_game_that_disagrees_with_its_row_is_refused(
    damage_root: Path, tmp_path: Path
) -> None:
    root = tmp_path / "run.progress"
    shutil.copytree(damage_root, root)
    target = root / "g000001"
    name = target / "games" / "damage-aware-sample-0.json"
    stored = layout.read_json(name)
    stored["baseline"]["counts"]["queries"] += 1
    layout.write_json(name, stored)
    manifest = layout.read_json(target / layout.MANIFEST)
    manifest["files"] = layout.hash_tree(target)
    layout.write_json(target / layout.MANIFEST, manifest)
    with pytest.raises(InspectError):
        games(root, 1, match="damage-aware-sample")
