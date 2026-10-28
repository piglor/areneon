"""The chaos derivation on a hand-computed toy file (invented numbers) and its refusals (#1120).

The toy has Rillaboom (truncated by patched limits), Salamence with its Mega forme, a species the
engine spells differently, a semantic ``nothing`` and ``''``, a spread over the Stat Point limit and
a move outside the forme's pool. Every expected integer below was worked on paper from the unit rule
(a list's denominator is its summed weight times 10^6; the weights are exact shares of it).
"""

# ruff: noqa: E501  (toy files and expected records are kept whole so they read as written)

from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_contracts.usage_prior import read_usage_prior

from scripts import derive_usage_prior as derive_mod
from scripts.derive_usage_prior import derive, main, run
from scripts.usage_common import DerivationError, parse_source

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "gen9championsvgc2026regmc"
FIXTURES = ROOT / "fixtures" / "showdown" / FORMAT
VOCABULARY = json.loads((FIXTURES / "team-vocabulary.json").read_text())
ENGINE_DATA = json.loads((FIXTURES / "engine-data.json").read_text())
SHOWS_AS = {"rillaboom": "rillaboom", "salamence": "salamence", "salamencemega": "salamence"}

CHAOS_TEXT = """
{"info": {"metagame": "gen9championsvgc2026regmc", "cutoff": 1630, "cutoff deviation": 0,
          "team type": null, "number of battles": 1000},
 "data": {
  "Rillaboom": {
    "Raw count": 5000, "usage": 0.5,
    "Abilities": {"grassysurge": 7, "overgrow": 1},
    "Items": {"leftovers": 4, "lifeorb": 2, "nothing": 1, "sitrusberry": 1, "notanitem": 0.5},
    "Moves": {"fakeout": 8.5, "grassyglide": 8.0, "woodhammer": 6.0, "uturn": 4.0, "knockoff": 3.0,
              "protect": 1.5, "surf": 0.5, "nomove": 1.0, "": 1.5},
    "Spreads": {"Adamant:32/0/0/0/34/0": 0.333333333333, "Jolly:0/32/0/0/2/32": 0.333333333333,
                "Timid:0/0/0/32/2/32": 0.333333333333},
    "Teammates": {"Nobody": 99}, "Tera Types": {}, "Happiness": {}},
  "Salamence": {
    "Raw count": 100, "usage": 0.1,
    "Abilities": {"intimidate": 1.5, "moxie": 0.5},
    "Items": {"lifeorb": 1, "nothing": 1},
    "Moves": {"protect": 2, "dragondance": 2, "roost": 1, "tailwind": 1, "earthquake": 1, "": 1},
    "Spreads": {"Adamant:32/32/2/0/0/0": 1.5, "Adamant:40/40/40/0/0/0": 0.5}},
  "Salamence-Mega": {
    "Raw count": 5000, "usage": 0.4,
    "Abilities": {"aerilate": 98},
    "Items": {"salamencite": 98},
    "Moves": {"protect": 98, "dragondance": 98, "hyperbeam": 98, "doubleedge": 98},
    "Spreads": {"Adamant:0/32/0/0/0/34": 98}},
  "Farfetch'd": {
    "Raw count": 1, "usage": 0.001,
    "Abilities": {"keeneye": 1}, "Items": {"nothing": 1, "stick": 1},
    "Moves": {"": 1, "fly": 3}, "Spreads": {"Adamant:0/0/0/0/0/0": 1}}
 }}
"""

INPUTS: dict[str, Any] = {
    "month": "2026-09",
    "cutoff": 1630,
    "format_id": FORMAT,
    "files": [
        {"source_id": "summary", "url": "https://example.test/s.txt", "retrieved_at": "2026-10-28T11:00:00Z", "size": 10, "sha256": "1" * 64},
        {"source_id": "chaos", "url": "https://example.test/c.json", "retrieved_at": "2026-10-28T11:00:00Z", "size": 20, "sha256": "2" * 64},
    ],
}  # fmt: skip


DEFAULT_TOY = derive(parse_source(CHAOS_TEXT), INPUTS, VOCABULARY, ENGINE_DATA, SHOWS_AS)[0]


@pytest.fixture
def small_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(derive_mod, "K_ITEMS", 2)
    monkeypatch.setattr(derive_mod, "K_MOVES", 3)
    monkeypatch.setattr(derive_mod, "K_SPREADS", 3)


def toy(chaos_text: str = CHAOS_TEXT) -> tuple[dict[str, Any], dict[str, Any]]:
    return derive(parse_source(chaos_text), INPUTS, VOCABULARY, ENGINE_DATA, SHOWS_AS)


def forme(record: dict[str, Any], species: str, forme_id: str) -> dict[str, Any]:
    found = next(s for s in record["species"] if s["id"] == species)
    found_forme: dict[str, Any] = next(f for f in found["formes"] if f["id"] == forme_id)
    return found_forme


def test_rillaboom_truncated_toy_gives_the_hand_computed_integers(small_limits: None) -> None:
    record, _ = toy()
    r = forme(record, "rillaboom", "rillaboom")
    assert r["weight"] == 8_500_000  # summed item weight 8.5
    assert r["abilities"] == {
        "denominator": 8_000_000,
        "entries": [
            {"id": "grassysurge", "weight": 7_000_000},
            {"id": "overgrow", "weight": 1_000_000},
        ],
        "residual": 0,
        "unmapped": 0,
    }
    # k_items 2 keeps leftovers and life orb; sitrus is the residual; `nothing` is no_item, never unmapped;
    # the unknown item is the 0.5 of unmapped mass.
    assert r["items"] == {
        "denominator": 8_500_000,
        "entries": [
            {"id": "leftovers", "weight": 4_000_000},
            {"id": "lifeorb", "weight": 2_000_000},
        ],
        "no_item": 1_000_000,
        "residual": 1_000_000,
        "unmapped": 500_000,
    }
    # Three equal thirds of 0.999999999999: the denominator is 10^6, each quota 333333.33, and the spare unit
    # goes to the first spread by id order of the keys.
    assert r["spreads"] == {
        "denominator": 1_000_000,
        "entries": [
            {
                "nature": "adamant",
                "stat_points": dict(hp=32, atk=0, **{"def": 0}, spa=0, spd=34, spe=0),
                "weight": 333_334,
            },
            {
                "nature": "jolly",
                "stat_points": dict(hp=0, atk=32, **{"def": 0}, spa=0, spd=2, spe=32),
                "weight": 333_333,
            },
            {
                "nature": "timid",
                "stat_points": dict(hp=0, atk=0, **{"def": 0}, spa=32, spd=2, spe=32),
                "weight": 333_333,
            },
        ],
        "residual": 0,
        "unmapped": 0,
    }
    # Moves sum to 34 = 4 slots * 8.5: inclusion is a share of 4 * 8_500_000; k_moves 3 keeps fakeout (forced, equal
    # to the denominator), grassyglide and woodhammer; uturn, knockoff and protect (4 + 3 + 1.5) and the move
    # outside the pool (surf 0.5) leave 8.5 + 1.5 residual / unmapped; `''` is the 1.5 empty-slot mass.
    assert r["moves"] == {
        "denominator": 8_500_000,
        "slots": 4,
        "entries": [
            {"id": "fakeout", "inclusion": 8_500_000},
            {"id": "grassyglide", "inclusion": 8_000_000},
            {"id": "woodhammer", "inclusion": 6_000_000},
        ],
        "empty_slots": 1_500_000,
        "residual": 8_500_000,
        "unmapped": 1_500_000,
    }


def test_mega_forme_weight_is_the_stone_holding_share() -> None:
    record, _ = toy()
    species = next(s for s in record["species"] if s["id"] == "salamence")
    assert species["denominator"] == 100_000_000
    assert [(f["id"], f["weight"]) for f in species["formes"]] == [
        ("salamence", 2_000_000),
        ("salamencemega", 98_000_000),
    ]
    mega = forme(record, "salamence", "salamencemega")
    assert mega["abilities"]["entries"] == [
        {"id": "aerilate", "weight": 98_000_000}
    ]  # post-Evolution ability
    assert mega["items"] == {
        "denominator": 98_000_000,
        "entries": [{"id": "salamencite", "weight": 98_000_000}],
        "no_item": 0,
        "residual": 0,
        "unmapped": 0,
    }
    assert "holder_abilities" not in mega
    assert [m["id"] for m in mega["moves"]["entries"]] == [
        "doubleedge",
        "dragondance",
        "hyperbeam",
        "protect",
    ]
    assert mega["moves"]["empty_slots"] == 0
    base = forme(record, "salamence", "salamence")
    assert base["abilities"]["entries"] == [
        {"id": "intimidate", "weight": 1_500_000},
        {"id": "moxie", "weight": 500_000},
    ]
    assert base["items"]["no_item"] == 1_000_000
    assert base["items"]["unmapped"] == 0
    assert base["moves"]["empty_slots"] == 1_000_000
    # The 120-point spread is over the format's limit: unmapped, not remapped.
    assert base["spreads"]["unmapped"] == 500_000
    assert [s["weight"] for s in base["spreads"]["entries"]] == [1_500_000]


def test_unmapped_species_and_the_report(small_limits: None) -> None:
    record, report = toy()
    assert [s["id"] for s in record["species"]] == [
        "rillaboom",
        "salamence",
    ]  # sorted, no Farfetch'd
    assert report["unmapped_species"] == ["Farfetch'd"]
    assert report["source_entries"] == 4
    assert report["species_in_record"] == 2
    assert report["formes_in_record"] == 3
    # Items: mapped 4+2+1 (Rillaboom) + 1 (life orb) + 98 (stone) = 106; unmapped 0.5 (unknown item) + 1.5 (all of
    # the species with no id); `nothing` is semantic and in neither term. share = 106 / 107.5.
    assert report["mapping_share_ppm"]["items"] == int(106 * 10**6 / 107.5)
    assert (
        report["mapping_detail_ppm"]["items"]["semantic_mass_excluded_micro"] == 3_000_000
    )  # 1 + 1 + 1
    assert (
        report["mapping_detail_ppm"]["moves"]["semantic_mass_excluded_micro"] == 3_500_000
    )  # 1.5 + 1 + 1
    assert report["gate_passed"] is True
    assert report["n8"]["reading"].startswith("usage sums to neither")


def test_n8_reading_follows_the_usage_sum() -> None:
    assert derive_mod.n8_reading(Fraction(5998273, 10**6)).endswith("not the four brought")
    assert derive_mod.n8_reading(Fraction(4)).endswith("the four brought")
    assert "undetermined" in derive_mod.n8_reading(Fraction(5))


def test_the_toy_record_passes_the_reader_and_is_byte_stable() -> None:
    first, _ = toy()
    again, _ = toy()
    assert jcs(first) == jcs(again)
    read_usage_prior(first, VOCABULARY)
    assert first["vocabulary_hash"] == sha256_hex(jcs(VOCABULARY))
    assert [i["source_id"] for i in first["source"]["inputs"]] == ["chaos", "summary"]
    assert first["derivation"] == {
        "script_version": "usage-prior-chaos.v1",
        "k_items": 16,
        "k_moves": 24,
        "k_spreads": 32,
    }


def test_limits_are_inside_the_hash(small_limits: None) -> None:
    truncated, _ = toy()
    assert truncated["derivation"] == {
        "script_version": "usage-prior-chaos.v1",
        "k_items": 2,
        "k_moves": 3,
        "k_spreads": 3,
    }
    assert truncated["species"] != DEFAULT_TOY["species"]


def changed(edit: Any) -> dict[str, Any]:
    chaos: dict[str, Any] = parse_source(CHAOS_TEXT)
    edit(chaos)
    return chaos


def derive_chaos(chaos: dict[str, Any]) -> None:
    derive(chaos, INPUTS, VOCABULARY, ENGINE_DATA, SHOWS_AS)


def test_refusals() -> None:
    with pytest.raises(DerivationError, match="another format or cutoff"):
        derive_chaos(changed(lambda c: c["info"].update(cutoff=1500)))
    with pytest.raises(DerivationError, match="another format or cutoff"):
        derive_chaos(changed(lambda c: c["info"].update(metagame="gen9ou")))
    with pytest.raises(DerivationError, match="lacks a list"):
        derive_chaos(changed(lambda c: c["data"]["Salamence-Mega"].pop("Items")))
    with pytest.raises(DerivationError, match="whole number of slots"):
        derive_chaos(changed(lambda c: c["data"]["Salamence-Mega"]["Moves"].update(doubleedge=90)))
    with pytest.raises(DerivationError, match="one spread"):
        derive_chaos(
            changed(
                lambda c: c["data"]["Salamence-Mega"].update(
                    Spreads={"Adamant:0/32/0/0/0/34": 90, "Adamant:00/32/0/0/0/34": 8}
                )
            )
        )


def test_a_forme_with_no_weight_is_dropped_and_counted() -> None:
    text = CHAOS_TEXT.replace(
        '"Items": {"lifeorb": 1, "nothing": 1}', '"Items": {"lifeorb": 0.0000001, "nothing": 0}'
    )
    text = text.replace(
        '"Moves": {"protect": 2, "dragondance": 2, "roost": 1, "tailwind": 1, "earthquake": 1, "": 1}',
        '"Moves": {"protect": 0}',
    )
    _, report = toy(text)
    assert report["dropped_formes"] == ["salamence"]


# -- the command ---------------------------------------------------------------------------------


def make_inputs(tmp: Path, chaos_text: str = CHAOS_TEXT, battles: int = 1000) -> Path:
    files = {
        "chaos": ("chaos.json", chaos_text.encode()),
        "summary": ("summary.txt", f"Total battles: {battles}\nAvg. weight/team: 0.1\n".encode()),
        "leads": (
            "leads.txt",
            b"Total leads: 1\n| 1    | Rillaboom          |  6.0% | 1 | 1% |\n| 2    | Sirfetch'd | 0% | 1 | 1% |\n",
        ),
    }
    entries = []
    for source_id, (name, data) in files.items():
        (tmp / name).write_bytes(data)
        entries.append({"source_id": source_id, "file": name, "url": f"https://example.test/{name}", "retrieved_at": "2026-10-28T11:00:00Z", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})  # fmt: skip
    record = {
        "month": "2026-09",
        "cutoff": 1630,
        "format_id": FORMAT,
        "preview_map": "generator-matched.uniform-v1.json",
        "files": entries,
    }
    path = tmp / "inputs.json"
    path.write_text(json.dumps(record))
    return path


def test_run_writes_the_record_and_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = make_inputs(tmp_path)
    out, report = tmp_path / "prior.json", tmp_path / "report.json"
    # The toy has only three species of the 300: the real preview map is used, its other species simply have no entry.
    assert run(inputs, tmp_path, ROOT, out, report) == 0
    record = json.loads(out.read_text())
    assert out.read_bytes() == jcs(record) + b"\n"
    assert json.loads(report.read_text())["leads"] == {"names": 2, "unmapped": ["Sirfetch'd"]}
    assert json.loads(report.read_text())["prior_hash"] == sha256_hex(jcs(record))
    assert "prior_hash" in capsys.readouterr().out
    out2 = tmp_path / "prior2.json"
    assert run(inputs, tmp_path, ROOT, out2, None) == 0
    assert out2.read_bytes() == out.read_bytes()


def test_run_refuses_a_file_that_differs_from_its_record(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    (tmp_path / "chaos.json").write_bytes(CHAOS_TEXT.replace("5000", "5001").encode())
    with pytest.raises(DerivationError, match="differs from its record"):
        run(inputs, tmp_path, ROOT, tmp_path / "o.json", None)


def test_run_refuses_a_summary_with_another_battle_count(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, battles=999)
    with pytest.raises(DerivationError, match="different battle counts"):
        run(inputs, tmp_path, ROOT, tmp_path / "o.json", None)


def test_summary_without_a_count_is_refused() -> None:
    with pytest.raises(DerivationError):
        derive_mod.summary_battles("Avg. weight/team: 1\n")


def test_a_record_below_the_mapping_gate_is_not_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chaos = parse_source(CHAOS_TEXT)
    chaos["data"]["Rillaboom"]["Abilities"] = {"grassysurge": 1, "zzability": 99}
    inputs = make_inputs(tmp_path, json.dumps(chaos, default=float))
    out = tmp_path / "prior.json"
    assert run(inputs, tmp_path, ROOT, out, tmp_path / "report.json") == 1
    assert not out.exists()
    assert json.loads((tmp_path / "report.json").read_text())["gate_passed"] is False
    assert "gate failed" in capsys.readouterr().err


def test_run_refuses_a_map_built_for_another_vocabulary(tmp_path: Path) -> None:
    root = tmp_path / "root"
    shutil_tree = root / "config" / "opponent-model" / FORMAT
    shutil_tree.mkdir(parents=True)
    (root / "fixtures" / "showdown").mkdir(parents=True)
    (root / "fixtures" / "showdown" / FORMAT).symlink_to(FIXTURES)
    other = json.loads(
        (
            ROOT / "config" / "opponent-model" / FORMAT / "generator-matched.uniform-v1.json"
        ).read_text()
    )
    other["vocabulary_hash"] = "0" * 64
    (shutil_tree / "generator-matched.uniform-v1.json").write_text(json.dumps(other))
    inputs = make_inputs(tmp_path)
    with pytest.raises(DerivationError, match="another vocabulary"):
        run(inputs, tmp_path, root, tmp_path / "o.json", None)


def test_run_refuses_an_engine_table_the_vocabulary_does_not_name(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "fixtures" / "showdown" / FORMAT).mkdir(parents=True)
    (root / "fixtures" / "showdown" / FORMAT / "team-vocabulary.json").write_text(
        json.dumps(VOCABULARY)
    )
    changed = copy.deepcopy(ENGINE_DATA)
    changed["species"][0]["name"] = "Changed"
    (root / "fixtures" / "showdown" / FORMAT / "engine-data.json").write_text(json.dumps(changed))
    inputs = make_inputs(tmp_path)
    with pytest.raises(DerivationError, match="engine-data"):
        run(inputs, tmp_path, root, tmp_path / "o.json", None)


def test_main_reports_a_refusal_with_status_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = make_inputs(tmp_path)
    (tmp_path / "chaos.json").write_bytes(b"{}")
    code = main(
        ["--inputs", str(inputs), "--raw-dir", str(tmp_path), "--out", str(tmp_path / "o.json")]
    )
    assert code == 2
    assert "derive_usage_prior:" in capsys.readouterr().err


def test_names_with_no_id_or_outside_the_forme_are_unmapped_never_remapped() -> None:
    def edit(chaos: dict[str, Any]) -> None:
        entry = chaos["data"]["Rillaboom"]
        entry["Abilities"].update(
            intimidate=1, zzability=2
        )  # another species' ability; no id at all
        entry["Spreads"].update(
            {"Wobbly:1/1/1/1/1/1": 1, "not a spread": 2, "Adamant:300/0/0/0/0/0": 3}
        )

    record, report = derive(changed(edit), INPUTS, VOCABULARY, ENGINE_DATA, SHOWS_AS)
    r = forme(record, "rillaboom", "rillaboom")
    assert [e["id"] for e in r["abilities"]["entries"]] == ["grassysurge", "overgrow"]
    assert r["abilities"]["unmapped"] == r["abilities"]["denominator"] * 3 // 11
    assert r["spreads"]["unmapped"] > 0
    assert report["mapping_detail_ppm"]["abilities"]["outside_vocabulary"] > 0
    assert report["mapping_detail_ppm"]["spreads"]["no_engine_id"] > 0
