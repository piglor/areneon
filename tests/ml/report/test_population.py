"""The dashboard's ladder view of a population run (ADR-021 slice 11)."""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import math
import re
import shutil
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import sha256_hex
from areneon_ml.learner.trainer import Learner
from areneon_ml.loop import ladder_write, layout
from areneon_ml.loop.__main__ import EXIT_REFUSED, main
from areneon_ml.loop.selection import health_donor
from areneon_ml.report.command import report_run
from areneon_ml.report.ladder_view import (
    LADDER_LABEL,
    _health_stats,
    events,
    rating,
    shown,
)
from areneon_ml.report.model import ReportError, read_report_model
from areneon_ml.report.render import build_report

from tests.ml.loop import test_population_health as health
from tests.ml.loop.test_past_pool import loop_config_edit
from tests.ml.loop.test_population import (
    gen_path,
    ladder_of,
    make_population,
    population_config,
    reseal,
)
from tests.ml.loop.test_population_copy import run_to
from tests.ml.report.support import fixture_copy
from tests.ml.report.test_report import TEMPLATE_TAGS, assert_inert, parsed

HOSTILE = "x</script><img src=x onerror=alert(1)> \"q\" 's' & <!-- " + chr(0x2028)


@pytest.fixture(scope="module")
def copying(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("copying")
    run_to(path, 3)
    return path


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("plain")
    make_population(path, population_config()).run(2)
    return path


def clone(run: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run"
    shutil.copytree(run, target)
    return target


def page(run: Path) -> str:
    return build_report(read_report_model(run, None))


def display(strength: int) -> str:
    """The display rating by an independent formula (binary floats; the page uses decimals)."""
    return f"{400 * math.log10(strength / 10**9):+.1f}"


# -- numbers equal a recomputation from the files ---


def test_the_display_rating_is_a_decimal_log_and_is_zero_at_the_reference() -> None:
    assert shown(10**9) == "+0.0"
    assert shown(10**10) == "+400.0"
    assert shown(10**8) == "-400.0"
    assert str(rating(10**9)) == "0.0"  # never "-0.0"
    for strength in (1, 7, 186_951_069, 13_441_525_362, 10**15):
        assert shown(strength) == display(strength)


def test_ladder_numbers_equal_the_files(copying: Path) -> None:
    text = page(copying)
    for number in (1, 2, 3):
        ladder = ladder_of(copying, number)
        for m in ladder["members"]:
            s = m["strength"]
            assert display(s["strength"]) in text
            assert f"{display(s['lower'])} to {display(s['upper'])}" in text
            assert str(s["strength"]) in text
        for a in ladder["anchors"]:
            assert a["anchor_id"] in text
        for c in ladder["cells"]:
            mean = c["score_sum_ppm"] // c["pairs"]
            tenths = (mean + 500) // 1000
            assert f'{tenths // 10}.{tenths % 10}% <span class="muted">' in text
            assert f"{c['pairs']} pairs" in text
    model = read_report_model(copying, None)
    assert model.population is not None
    assert [g.number for g in model.population.generations] == [0, 1, 2, 3]
    assert [g.number for g in model.generations] == [0, 1, 2, 3]
    for g in model.population.generations[1:]:
        league = layout.read_json(gen_path(copying, g.number) / "league-run.json")
        assert sum(m.games for m in g.members) == sum(m["games"] for m in league["members"])
        assert g.ladder == ladder_of(copying, g.number)
        assert f"Generation {g.number}: ladder table and pair-score heatmap" in text


def test_the_page_says_what_the_ladder_is_and_is_not(copying: Path) -> None:
    text = page(copying)
    assert LADDER_LABEL in text
    assert LADDER_LABEL == "Training-pool teams, sampling mode; not validation, not held out."
    assert text.count(LADDER_LABEL) >= 8  # the top and every section
    assert "never a measure of strength" in text
    assert "Not the Training gate" in text
    assert "validation teams" not in text.replace("not validation", "")


def test_a_copy_and_its_evidence_are_shown(copying: Path) -> None:
    text = page(copying)
    ladder = ladder_of(copying, 1)
    first = ladder["selections"][0]
    assert first["kind"] == "copy"
    assert "2 pairs of the donor against the loser" in text
    assert f"roots {first['roots_before']} to {first['roots_after']}" in text
    found = events(read_report_model(copying, None).population)  # type: ignore[arg-type]
    assert [(e["kind"], e["at"], e["donor"], e["loser"]) for e in found] == [
        ("copy", g, "m0", "m1") for g in (1, 2, 3)
    ]
    assert "Copy tree" in text
    assert "solid edge: copy" in text
    assert re.search(r"<b>3</b>copies", text)


def test_a_generation_without_a_copy_states_the_failed_condition(plain: Path) -> None:
    text = page(plain)
    first = ladder_of(plain, 1)["selections"][0]
    assert first == {"kind": "none", "failed": "no_ready_member"}
    assert "no member was ready" in text
    assert re.search(r"<b>0</b>copies", text)
    assert "No cyclic triple at any generation." in text


def test_cyclic_triples_with_three_members(tmp_path: Path) -> None:
    config = population_config(members=3, key="pop-three")
    run = tmp_path / "run"
    make_population(run, config).run(2)
    path = gen_path(run, 2) / "ladder.json"
    ladder = layout.read_json(path)
    ladder["cyclic_triples"] = 1
    layout.write_json(path, ladder)
    reseal(run, 2, ladder_hash=ladder_write.json_hash(ladder))
    text = page(run)
    assert "Generation 2: 1 cyclic triple: non-transitive at this generation" in text


def test_the_settings_and_the_per_member_counts_are_shown(copying: Path) -> None:
    text = page(copying)
    for m in ladder_of(copying, 2)["members"]:
        s = m["settings"]
        assert str(s["learning_rate_micro"]) in text
    counts = layout.read_json(copying / "metrics" / "members" / "g000002.json")
    for label, values in counts["members"].items():
        assert str(values["decisions"]) in text
        assert label in text
    assert "Wall-clock time and throughput" in text
    assert "metrics/members/g000002.json" in read_report_model(copying, None).inputs


def test_every_new_input_is_listed_with_its_sha256(copying: Path) -> None:
    model = read_report_model(copying, None)
    text = build_report(model)
    expected = {
        "generations/g000000/anchors.json",
        "generations/g000000/calibration.json",
        "generations/g000002/ladder.json",
        "generations/g000002/league-run.json",
        "generations/g000002/generation.json",
        "generations/g000002/plan.json",
        "generations/g000002/members/m0/train-report.json",
        "generations/g000002/members/m1/train-report.json",
        "metrics/g000002.json",
    }
    assert expected <= set(model.inputs)
    for rel, digest in model.inputs.items():
        assert digest == hashlib.sha256((copying / rel).read_bytes()).hexdigest()
        assert rel in text
        assert digest in text


# -- security, determinism ---


def test_csp_hashes_and_no_remote_reference(copying: Path) -> None:
    text = page(copying)
    found = parsed(text)
    assert found.first_in_head == "meta"
    content = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', text)
    assert content is not None
    policy = content[1].replace("&#x27;", "'")
    styles = [b for t, b in found.bodies if t == "style"]
    scripts = [b for t, b in found.bodies if t == "script"]
    assert len(styles) == 1
    assert len(scripts) == 2
    for kind, body in (("style", styles[0]), ("script", scripts[1])):
        digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
        assert f"{kind}-src 'sha256-{digest}'" in policy
    assert "default-src 'none'" in policy
    assert "img-src 'none'" in policy
    assert re.search(r"https?:|//[a-z]|url\(|@import|<link|<img|<iframe", text) is None
    assert "style=" not in text
    assert set(found.tags) <= TEMPLATE_TAGS
    for svg in re.findall(r"<svg.*?</svg>", text, re.S):
        assert 'role="group"' in svg
        assert 'aria-label="' in svg
        assert "<title>" in svg
    for table in re.findall(r"<table[ >].*?</table>", text, re.S):
        assert 'scope="col"' in table
        assert 'scope="row"' in table


def test_the_build_needs_no_network(copying: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_a: object, **_k: object) -> None:
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert page(copying)


def test_the_page_is_deterministic_and_the_command_writes_one_file(
    copying: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out" / "a.html"
    first = report_run(copying, None, out)
    data = out.read_bytes()
    assert first["generations"] == 4
    assert data == page(copying).encode()
    report_run(copying, None, out)
    assert out.read_bytes() == data
    assert out.stat().st_mode & 0o777 == 0o644


def hostile(run: Path) -> Any:
    model = read_report_model(run, None)
    facts = model.population
    assert facts is not None
    gens = []
    for g in facts.generations:
        if g.ladder is None:
            gens.append(g)
            continue
        ladder = copy.deepcopy(g.ladder)
        ladder["anchors"][0]["anchor_id"] = HOSTILE
        ladder["snapshots"] = [
            {
                "snapshot_id": HOSTILE,
                "member": "m0",
                "generation": 0,
                "strength": {"strength": 10**9, "lower": 10**9, "upper": 10**9},
            }
        ]
        for cell in ladder["cells"]:
            if cell["opponent"]["kind"] == "anchor" and cell["opponent"]["id"] == "actor-rule@v1":
                cell["opponent"]["id"] = HOSTILE
        members = tuple(
            dataclasses.replace(m, forfeit_causes=((HOSTILE, 1),), forfeits=1) for m in g.members
        )
        gens.append(
            dataclasses.replace(g, ladder=ladder, members=members, pool_snapshots=(HOSTILE,))
        )
    return dataclasses.replace(
        model,
        run_key=HOSTILE,
        lineage_id=HOSTILE,
        format_id=HOSTILE,
        inputs={HOSTILE: "0" * 64},
        population=dataclasses.replace(facts, generations=tuple(gens), anchors=((HOSTILE, 1),)),
    )


def test_hostile_strings_everywhere_are_inert(copying: Path) -> None:
    text = build_report(hostile(copying))
    assert_inert(text, (HOSTILE,))
    assert "&lt;/script&gt;" in text


def test_canary_negative_control(copying: Path) -> None:
    text = build_report(hostile(copying))
    broken = text.replace("&lt;/script&gt;", "</script>", 1)
    with pytest.raises(AssertionError):
        assert_inert(broken, (HOSTILE,))
    with pytest.raises(AssertionError):
        assert_inert(text.replace("</main>", "<b onclick=x>y</b></main>", 1), ())


# -- quarantined members: a null metric and a null training set are read as what they are ---


def quarantine(run: Path) -> None:
    """Forge generation 2 of a no-copy run as slice 9 writes a quarantine."""
    path = gen_path(run, 2)
    ladder = layout.read_json(path / "ladder.json")
    ladder["selections"].append({"kind": "health", "member": "m1", "cause": "non_finite_loss"})
    layout.write_json(path / "ladder.json", ladder)
    report = layout.read_json(path / "members" / "m1" / "train-report.json")
    report["battle"]["clip_fraction"] = None
    report["battle"]["policy_loss"] = None
    layout.write_json(path / "members" / "m1" / "train-report.json", report)
    manifest = layout.read_json(path / "manifest.json")
    manifest["members"]["m1"]["training_set_hash"] = None
    reseal(run, 2, ladder_hash=ladder_write.json_hash(ladder), members=manifest["members"])


def test_a_null_metric_and_a_null_training_set_are_accepted(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    quarantine(run)
    model = read_report_model(run, None)
    assert model.population is not None
    member = next(m for m in model.population.generations[2].members if m.member == "m1")
    assert member.training_set_hash is None
    assert member.nulls == (("battle", ("clip_fraction", "policy_loss")),)
    text = build_report(model)
    assert "not finite" in text
    assert "update discarded (no training set)" in text
    assert "quarantine and replacement" in text
    assert "a non-finite number in its update" in text
    assert re.search(r"<b>1</b>quarantines", text)
    # The derived donor is the strongest member that is not quarantined.
    assert "m0 (derived: the strongest member not quarantined)" in text
    found = events(model.population)
    assert [(e["kind"], e["member"], e["donor"]) for e in found] == [("health", "m1", "m0")]


# -- refusals ---


def refused(run: Path, message: str) -> None:
    with pytest.raises(ReportError, match=message):
        read_report_model(run, None)


def test_a_ladder_record_of_another_run_is_refused(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    path = gen_path(run, 2) / "ladder.json"
    ladder = layout.read_json(path)
    ladder["run_id"] = "another-run"
    layout.write_json(path, ladder)
    reseal(run, 2, ladder_hash=ladder_write.json_hash(ladder))
    refused(run, "g000002: ladder.json: the previous record belongs to another run or config")
    # The first record has no previous one: the config tie names the run key.
    path = gen_path(run, 1) / "ladder.json"
    first = layout.read_json(path)
    first["run_id"] = "another-run"
    layout.write_json(path, first)
    reseal(run, 1, ladder_hash=ladder_write.json_hash(first))
    refused(run, "g000001: ladder.json: run_id another-run is not the config's run key")


def test_a_ladder_record_the_manifest_does_not_name_is_refused(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    path = gen_path(run, 2) / "ladder.json"
    ladder = layout.read_json(path)
    ladder["cyclic_triples"] = 0
    ladder["cells"][0]["scores"] = [0, 0, 0, 0, ladder["cells"][0]["pairs"]]
    ladder["cells"][0]["score_sum_ppm"] = 1_000_000 * ladder["cells"][0]["pairs"]
    layout.write_json(path, ladder)
    reseal(run, 2)  # the files match the manifest, but its ladder_hash names the old record
    refused(run, "g000002: its ladder.json is not the one its manifest names")


def test_a_ladder_record_of_other_members_than_the_config_is_refused(
    plain: Path, tmp_path: Path
) -> None:
    run = clone(plain, tmp_path)
    path = gen_path(run, 2) / "ladder.json"
    ladder = layout.read_json(path)
    ladder["window"]["size"] = 9
    layout.write_json(path, ladder)
    reseal(run, 2, ladder_hash=ladder_write.json_hash(ladder))
    refused(run, "g000002: ladder.json")


def test_anchor_strengths_other_than_the_frozen_ones_are_refused(
    plain: Path, tmp_path: Path
) -> None:
    run = clone(plain, tmp_path)
    path = gen_path(run, 2) / "ladder.json"
    ladder = layout.read_json(path)
    ladder["anchors"][1]["strength"] += 1
    layout.write_json(path, ladder)
    reseal(run, 2, ladder_hash=ladder_write.json_hash(ladder))
    refused(run, "g000002: ladder.json")


def test_a_manifest_naming_other_anchors_is_refused(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    reseal(run, 2, anchors_hash=sha256_hex(b"other"))
    refused(run, "g000002: its manifest names other anchor strengths than generation 0 froze")


def test_a_league_record_of_another_generation_is_refused(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    source = gen_path(run, 1) / "league-run.json"
    (gen_path(run, 2) / "league-run.json").write_bytes(source.read_bytes())
    reseal(run, 2)
    refused(run, "g000002")


def test_a_pair_that_is_not_the_one_the_generation_before_froze_is_refused(
    plain: Path, tmp_path: Path
) -> None:
    run = clone(plain, tmp_path)
    manifest = layout.read_json(gen_path(run, 1) / "manifest.json")
    manifest["members"]["m0"]["checkpoints"]["battle"] = "0" * 64
    layout.write_json(gen_path(run, 1) / "manifest.json", manifest)
    refused(run, "g000002: m0's pair is not the one the generation before froze")


def test_a_manifest_of_other_members_is_refused(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    manifest = layout.read_json(gen_path(run, 2) / "manifest.json")
    del manifest["members"]["m1"]
    layout.write_json(gen_path(run, 2) / "manifest.json", manifest)
    refused(run, "g000002: its manifest does not hold exactly m0..m1")


def test_a_changed_file_is_refused_by_the_manifest(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    with (gen_path(run, 1) / "ladder.json").open("ab") as handle:
        handle.write(b" ")
    refused(run, "differ")


def test_a_train_report_that_is_not_strict_is_refused(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    path = gen_path(run, 2) / "members" / "m0" / "train-report.json"
    report = layout.read_json(path)
    report["battle"]["entropy"] = None
    report["battle"]["steps"] = None  # only a metric may be null
    layout.write_json(path, report)
    reseal(run, 2)
    refused(run, "steps")


def test_the_command_reads_a_population_run_and_the_inspector_still_refuses_it(
    plain: Path, tmp_path: Path
) -> None:
    out = tmp_path / "r.html"
    assert main(["report", "--run-dir", str(plain), "--out", str(out)]) == 0
    assert out.is_file()
    code = main(["inspect", "--run-dir", str(plain), "--out", str(tmp_path / "inspect")])
    assert code == EXIT_REFUSED


# -- the past pool, the per-member counts and the remaining refusals ---


@pytest.fixture(scope="module")
def pooled(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("pooled")
    make_population(path, loop_config_edit()).run(3)
    return path


def test_the_past_pool_and_the_frozen_rated_snapshots_are_shown(pooled: Path) -> None:
    model = read_report_model(pooled, None)
    assert model.population is not None
    text = build_report(model)
    assert "<h2>Past pool</h2>" in text
    for g in model.population.generations:
        file = layout.read_json(gen_path(pooled, g.number) / "past-pool.json")
        assert g.pool_snapshots == tuple(s["id"] for s in file["snapshots"])
        assert f"generations/g{g.number:06d}/past-pool.json" in model.inputs
    snapshots = [s for g in model.population.generations if g.ladder for s in g.ladder["snapshots"]]
    assert snapshots, "the tiny run freezes a snapshot's strength"
    assert "fixed: frozen when it had enough pairs" in text
    assert snapshots[0]["snapshot_id"] in text
    assert "snapshot" in text


def test_a_past_pool_the_manifest_does_not_name_or_that_differs_is_refused(
    pooled: Path, plain: Path, tmp_path: Path
) -> None:
    run = clone(pooled, tmp_path / "a")
    path = gen_path(run, 2) / "past-pool.json"
    file = layout.read_json(path)
    file["cells"] = file["cells"][1:] or file["cells"]
    file["generation"] = 99
    layout.write_json(path, file)
    reseal(run, 2)
    refused(run, "g000002: its past-pool.json is not the one its manifest names")
    reseal(run, 2, past_pool_hash=ladder_write.json_hash(file))
    refused(run, "g000002: past-pool.json")
    stray = clone(plain, tmp_path / "b")
    (gen_path(stray, 2) / "past-pool.json").write_bytes(
        (gen_path(pooled, 2) / "past-pool.json").read_bytes()
    )
    reseal(stray, 2)
    refused(stray, "a past-pool.json its manifest does not name")


def test_the_member_counts_are_read_strictly(plain: Path, tmp_path: Path) -> None:
    def path(run: Path) -> Path:
        return run / "metrics" / "members" / "g000002.json"

    cases: list[tuple[str, Callable[[dict[str, Any]], object]]] = [
        ("it is another generation's", lambda b: b.update(generation=1)),
        ("names other members", lambda b: b["members"].pop("m1")),
        ("games", lambda b: b["members"]["m0"].update(games=-1)),
        ("does not know: extra", lambda b: b["members"]["m0"].update(extra=1)),
    ]
    for index, (message, change) in enumerate(cases):
        run = clone(plain, tmp_path / str(index))
        body = layout.read_json(path(run))
        change(body)
        layout.write_json(path(run), body)
        refused(run, message)
    run = clone(plain, tmp_path / "stray")
    (run / "metrics" / "members" / "notes.txt").write_text("x")
    refused(run, "is not a members file of this run")
    run = clone(plain, tmp_path / "gap")
    layout.write_json(
        run / "metrics" / "members" / "g000009.json", {"generation": 9, "members": {}}
    )
    refused(run, "is not a members file of this run")
    run = clone(plain, tmp_path / "none")
    shutil.rmtree(run / "metrics" / "members")
    assert read_report_model(run, None).population.member_counts == {}  # type: ignore[union-attr]


def test_a_run_without_member_counts_says_nothing_about_them(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    shutil.rmtree(run / "metrics" / "members")
    assert "Per-member counts" not in page(run)


def test_generation_zero_files_are_checked(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path / "a")
    path = gen_path(run, 0) / "anchors.json"
    anchors = layout.read_json(path)
    anchors["run_id"] = "another-run"
    layout.write_json(path, anchors)
    reseal(run, 0)
    refused(run, "g000000: anchors.json")
    other = clone(plain, tmp_path / "b")
    anchors = layout.read_json(gen_path(other, 0) / "anchors.json")
    anchors["anchors"][1]["strength"] += 1
    layout.write_json(gen_path(other, 0) / "anchors.json", anchors)
    reseal(other, 0)
    refused(other, "other anchor strengths")


def test_generation_json_must_be_the_generations_version_2_file(
    plain: Path, tmp_path: Path
) -> None:
    changes: list[Callable[[dict[str, Any]], object]] = [
        lambda g: g.update(generation=9),
        lambda g: g["members"].pop(),
        lambda g: g["members"][0].update(training_set_hash=7),
        lambda g: g["members"][0].update(examples=-1),
        lambda g: g.update(extra=1),
    ]
    for index, change in enumerate(changes):
        run = clone(plain, tmp_path / str(index))
        path = gen_path(run, 2) / "generation.json"
        body = layout.read_json(path)
        change(body)
        layout.write_json(path, body)
        reseal(run, 2)
        with pytest.raises(ReportError):
            read_report_model(run, None)


def test_a_member_with_two_different_training_sets_is_refused(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    manifest = layout.read_json(gen_path(run, 2) / "manifest.json")
    manifest["members"]["m0"]["training_set_hash"] = "1" * 64
    reseal(run, 2, members=manifest["members"])
    refused(run, "m0's training set differs between its files")


def test_a_non_synthetic_member_record_is_refused(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    path = gen_path(run, 2) / "members" / "m0" / "records" / "battle.json"
    record = layout.read_json(path)
    record["data_class"] = "private"
    layout.write_json(path, record)
    reseal(run, 2)
    with pytest.raises(ReportError):
        read_report_model(run, None)
    record["data_class"] = "synthetic"
    del record["role"]
    layout.write_json(path, record)
    reseal(run, 2)
    refused(run, "records/battle.json")


def test_a_train_report_that_is_not_an_object_or_lacks_a_role_is_refused(
    plain: Path, tmp_path: Path
) -> None:
    bodies: list[Any] = [[], {"battle": 1, "preview": 2}, {"battle": {}, "extra": 1}]
    for index, body in enumerate(bodies):
        run = clone(plain, tmp_path / str(index))
        path = gen_path(run, 2) / "members" / "m0" / "train-report.json"
        layout.write_json(path, body)
        reseal(run, 2)
        with pytest.raises(ReportError):
            read_report_model(run, None)


def test_an_unreadable_file_is_a_report_error(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    (gen_path(run, 2) / "plan.json").write_text("not json")
    reseal(run, 2)
    refused(run, "plan.json")


def test_a_member_without_a_train_report_is_shown_as_not_recorded(
    plain: Path, tmp_path: Path
) -> None:
    run = clone(plain, tmp_path)
    (gen_path(run, 2) / "members" / "m1" / "train-report.json").unlink()
    reseal(run, 2)
    assert "not recorded" in page(run)


def test_events_and_the_tree_without_a_played_generation(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    shutil.rmtree(run / "generations" / "g000001")
    shutil.rmtree(run / "generations" / "g000002")
    shutil.rmtree(run / "metrics")
    text = page(run)
    assert "No generation has played yet" in text
    assert "Copy tree" not in text


# -- a real quarantine written by the loop (slice 9) ---


def test_a_real_quarantine_run_is_reported(tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Learner, "train", health.forced_train)
        run = tmp_path / "health"
        health.run_to(run, health.reset())
    model = read_report_model(run, None)
    assert model.population is not None
    text = build_report(model)
    found = events(model.population)
    assert [(e["kind"], e["at"], e["member"]) for e in found if e["kind"] == "health"] == [
        ("health", health.BAD_GENERATION, "m1")
    ]
    stopped = next(m for m in model.population.generations[2].members if m.member == "m1")
    assert stopped.training_set_hash is None
    assert stopped.nulls
    assert "not finite" in text
    assert "update discarded (no training set)" in text
    assert "quarantine and replacement" in text


# -- nulls only where the ladder record quarantines ---


def test_a_null_in_a_member_the_ladder_does_not_quarantine_is_refused(
    plain: Path, tmp_path: Path
) -> None:
    run = clone(plain, tmp_path / "report")
    path = gen_path(run, 2) / "members" / "m1" / "train-report.json"
    report = layout.read_json(path)
    report["battle"]["entropy"] = None
    layout.write_json(path, report)
    reseal(run, 2)
    refused(run, "m1's train report holds a null but the ladder record does not quarantine it")
    other = clone(plain, tmp_path / "hash")
    manifest = layout.read_json(gen_path(other, 2) / "manifest.json")
    manifest["members"]["m1"]["training_set_hash"] = None
    reseal(other, 2, members=manifest["members"])
    refused(other, "manifest training set is null exactly when the ladder record quarantines it")


def test_a_quarantined_member_has_a_null_training_set(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    path = gen_path(run, 2) / "ladder.json"
    ladder = layout.read_json(path)
    ladder["selections"].append({"kind": "health", "member": "m1", "cause": "forfeits"})
    layout.write_json(path, ladder)
    reseal(run, 2, ladder_hash=ladder_write.json_hash(ladder))
    refused(run, "manifest training set is null exactly when the ladder record quarantines it")


def test_every_float_health_sanitize_can_null_is_accepted_for_a_quarantined_member(
    plain: Path, tmp_path: Path
) -> None:
    run = clone(plain, tmp_path)
    quarantine(run)
    path = gen_path(run, 2) / "members" / "m1" / "train-report.json"
    report = layout.read_json(path)
    battle = report["battle"]
    for key in ("approx_kl", "entropy", "value_loss", "explained_variance"):
        battle[key] = None
    battle["anchor_kl"] = None
    battle["anchor_beta_micro"] = 5
    for entry in battle["value_calibration"][:2]:
        entry["mean_value"] = None
        entry["mean_score"] = None
    report["preview"]["explained_variance"] = None
    layout.write_json(path, report)
    reseal(run, 2)
    model = read_report_model(run, None)
    assert model.population is not None
    member = next(m for m in model.population.generations[2].members if m.member == "m1")
    assert dict(member.nulls)["battle"] == (
        "anchor_kl",
        "approx_kl",
        "clip_fraction",
        "entropy",
        "explained_variance",
        "policy_loss",
        "value_calibration",
        "value_loss",
    )
    assert dict(member.nulls)["preview"] == ("explained_variance",)
    assert "not finite" in build_report(model)
    # An integer is never null, whoever the member is.
    battle["steps"] = None
    layout.write_json(path, report)
    reseal(run, 2)
    refused(run, "steps")


def forge_all_quarantined(run: Path) -> None:
    path = gen_path(run, 2)
    ladder = layout.read_json(path / "ladder.json")
    ladder["selections"] += [
        {"kind": "health", "member": "m0", "cause": "non_finite_loss"},
        {"kind": "health", "member": "m1", "cause": "non_finite_loss"},
    ]
    layout.write_json(path / "ladder.json", ladder)
    manifest = layout.read_json(path / "manifest.json")
    for label in ("m0", "m1"):
        manifest["members"][label]["training_set_hash"] = None
    reseal(run, 2, ladder_hash=ladder_write.json_hash(ladder), members=manifest["members"])


def test_a_generation_where_every_member_is_quarantined_has_no_donor(
    plain: Path, tmp_path: Path
) -> None:
    run = clone(plain, tmp_path)
    forge_all_quarantined(run)
    model = read_report_model(run, None)
    assert model.population is not None
    assert [e["donor"] for e in events(model.population)] == [None, None]
    text = build_report(model)
    assert "none (derived: the strongest member not quarantined)" in text
    assert "Copy tree" in text


def test_the_donor_is_the_loops_own_function(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    quarantine(run)
    model = read_report_model(run, None)
    assert model.population is not None
    ladder = model.population.generations[2].ladder
    assert ladder is not None
    assert [e["donor"] for e in events(model.population)] == [health_donor(ladder)]


def test_a_member_without_a_row_or_a_report_has_no_health_numbers(plain: Path) -> None:
    model = read_report_model(plain, None)
    assert model.population is not None
    assert _health_stats(model.population, 0, "m1") == "not recorded"


def test_generation_zero_must_name_the_anchors_exactly(plain: Path, tmp_path: Path) -> None:
    run = clone(plain, tmp_path)
    manifest = layout.read_json(gen_path(run, 0) / "manifest.json")
    del manifest["anchors_hash"]
    layout.write_json(gen_path(run, 0) / "manifest.json", manifest)
    refused(run, "other anchor strengths than anchors.json")


def test_a_single_lineage_run_does_not_skip_a_members_metrics_directory(tmp_path: Path) -> None:
    run = fixture_copy(tmp_path, train=True)
    (run / "metrics" / "members").mkdir(parents=True)
    with pytest.raises(ReportError, match="is not a metrics file of this run"):
        read_report_model(run, None)
