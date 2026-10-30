"""The usage weight derivation on a hand-computed toy chaos file (invented numbers) (#976).

The toy has Rillaboom (lists truncated by patched limits), Salamence, a species the roster lacks, a
semantic ``nothing`` and ``''``, a move outside the species' pool, a spread over the Stat Point limit
and a malformed one, a teammate outside the roster and non-positive teammate weights. Every expected
integer below was worked on paper from the rule (each list is split into 10^6 parts by largest
remainder, ties to the named ids first; ``w`` is the integer square root of ``ppm * 10^6``).
"""

# ruff: noqa: E501  (toy files and expected tables are kept whole so they read as written)

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_contracts.usage_weight_table import read_usage_weight_table

from scripts import derive_usage_weights as mod
from scripts.derive_usage_weights import derive, main, run
from scripts.usage_common import DerivationError, parse_source

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "gen9championsvgc2026regmc"
FIXTURES = ROOT / "fixtures" / "showdown" / FORMAT
VOCABULARY = json.loads((FIXTURES / "team-vocabulary.json").read_text())
ENGINE_DATA = json.loads((FIXTURES / "engine-data.json").read_text())

CHAOS_TEXT = """
{"info": {"metagame": "gen9championsvgc2026regmc", "cutoff": 1630, "number of battles": 1000},
 "data": {
  "Rillaboom": {
    "Abilities": {"grassysurge": 7, "overgrow": 1},
    "Items": {"leftovers": 4, "lifeorb": 2, "nothing": 1, "sitrusberry": 1, "notanitem": 0.5},
    "Moves": {"fakeout": 8.5, "grassyglide": 8.0, "woodhammer": 6.0, "uturn": 4.0, "knockoff": 3.0,
              "protect": 1.5, "surf": 0.5, "nomove": 1.0, "": 1.5},
    "Spreads": {"Adamant:32/0/0/0/34/0": 0.5, "Jolly:0/32/0/0/2/32": 0.25, "Timid:0/0/0/32/2/32": 0.125,
                "Bold:0/0/32/0/34/0": 0.0625, "Adamant:40/40/40/0/0/0": 0.0625, "Weird:1": 0.25},
    "Teammates": {"Salamence": 3, "Nobody": 2, "Rillaboom": 1, "Salamence-Mega": -2}},
  "Salamence": {
    "Abilities": {"intimidate": 1.5, "moxie": 0.5},
    "Items": {"lifeorb": 1, "nothing": 1},
    "Moves": {"protect": 2, "dragondance": 2, "roost": 1, "tailwind": 1, "earthquake": 1, "": 1},
    "Spreads": {"Adamant:32/32/2/0/0/0": 1.5, "Adamant:0/0/0/0/0/0": 0.5},
    "Teammates": {"Rillaboom": -1}},
  "Nobodymon": {
    "Abilities": {"x": 1}, "Items": {"nothing": 1.5}, "Moves": {"": 1},
    "Spreads": {"Adamant:0/0/0/0/0/0": 1}, "Teammates": {"Rillaboom": 5}}
 }}
"""

CLEAN_TEXT = """
{"info": {"metagame": "gen9championsvgc2026regmc", "cutoff": 1630, "number of battles": 1000},
 "data": {"Rillaboom": {
    "Abilities": {"grassysurge": 3, "overgrow": 1},
    "Items": {"leftovers": 1, "lifeorb": 1, "nothing": 2},
    "Moves": {"fakeout": 4, "grassyglide": 4, "woodhammer": 4, "uturn": 4},
    "Spreads": {"Adamant:32/0/0/0/34/0": 1, "Jolly:0/32/0/0/2/32": 1},
    "Teammates": {"Salamence": 1}}}}
"""

FILES = {
    "chaos": ("chaos.json", "https://example.test/c.json"),
    "summary": ("summary.txt", "https://example.test/s.txt"),
}


def inputs_for(raw: Path, chaos_text: str, battles: int = 1000) -> dict[str, Any]:
    """Write the chaos and summary files under ``raw`` and the input record that names them."""
    texts = {"chaos": chaos_text, "summary": f"Total battles: {battles}\nrest\n"}
    files = []
    for source_id, (name, url) in FILES.items():
        data = texts[source_id].encode()
        (raw / name).write_bytes(data)
        files.append(
            {
                "file": name,
                "source_id": source_id,
                "url": url,
                "retrieved_at": "2026-10-28T11:00:00Z",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {"month": "2026-09", "cutoff": 1630, "format_id": FORMAT, "files": files}


INPUTS: dict[str, Any] = {
    "month": "2026-09",
    "cutoff": 1630,
    "format_id": FORMAT,
    "files": [
        {"file": "s", "source_id": "summary", "url": "https://example.test/s.txt", "retrieved_at": "2026-10-28T11:00:00Z", "size": 10, "sha256": "1" * 64},
        {"file": "c", "source_id": "chaos", "url": "https://example.test/c.json", "retrieved_at": "2026-10-28T11:00:00Z", "size": 20, "sha256": "2" * 64},
    ],
}  # fmt: skip


@pytest.fixture
def small_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "K_ITEMS", 2)
    monkeypatch.setattr(mod, "K_MOVES", 3)
    monkeypatch.setattr(mod, "K_SPREADS", 3)
    monkeypatch.setattr(mod, "K_TEAMMATES", 1)


def toy(text: str = CHAOS_TEXT) -> tuple[dict[str, Any], dict[str, Any]]:
    return derive(parse_source(text), INPUTS, VOCABULARY, ENGINE_DATA)


def species(table: dict[str, Any], sid: str) -> dict[str, Any]:
    found: dict[str, Any] = next(s for s in table["species"] if s["id"] == sid)
    return found


def test_species_shares_and_roots_are_the_hand_computed_integers(small_limits: None) -> None:
    table, _ = toy()
    # Item weights 8.5, 2 and the unknown species' 1.5 (total 12): quotas 708333.33, 166666.67, 125000;
    # the spare unit goes to the larger remainder (Salamence). w is the integer square root of ppm * 10^6.
    assert [(s["id"], s["usage_ppm"], s["w_species"]) for s in table["species"]] == [
        ("rillaboom", 708_333, 841_625),
        ("salamence", 166_667, 408_248),
    ]
    assert table["unmapped_species_ppm"] == 125_000
    assert 841_625**2 <= 708_333 * 10**6 < 841_626**2


def test_rillaboom_lists_truncated_by_patched_limits(small_limits: None) -> None:
    table, _ = toy()
    r = species(table, "rillaboom")
    assert r["abilities"] == {
        "entries": [{"id": "grassysurge", "ppm": 875_000}, {"id": "overgrow", "ppm": 125_000}],
        "unmapped_ppm": 0,
    }
    # k_items 2 keeps leftovers and life orb; sitrus is truncated; `nothing` is no_item, never unmapped;
    # the unknown item is the 0.5 of unmapped mass. Quotas of 8.5: 470588.24, 235294.12, 117647.06 (twice),
    # 58823.53; the spare unit goes to the largest remainder (the unmapped mass).
    assert r["items"] == {
        "entries": [{"id": "leftovers", "ppm": 470_588}, {"id": "lifeorb", "ppm": 235_294}],
        "no_item_ppm": 117_647,
        "truncated_ppm": 117_647,
        "unmapped_ppm": 58_824,
    }
    # Total 34: fake out 8.5, grassy glide 8, wood hammer 6 are kept; U-turn, Knock Off, Protect and Surf
    # (8.5 of them in the pool, Surf's 0.5 is outside it) are truncated; `nomove` (1) and Surf (0.5) are
    # unmapped; the empty slot weighs 1.5. The two spare units go to the .647 remainders.
    assert r["moves"] == {
        "entries": [
            {"id": "fakeout", "ppm": 250_000},
            {"id": "grassyglide", "ppm": 235_294},
            {"id": "woodhammer", "ppm": 176_470},
        ],
        "empty_ppm": 44_118,
        "truncated_ppm": 250_000,
        "unmapped_ppm": 44_118,
    }
    # Total 1.25: .5, .25, .125 are kept, Bold's .0625 truncated, the over-limit spread's .0625 and the
    # malformed key's .25 unmapped (.3125).
    assert r["spreads"] == {
        "entries": [
            {
                "nature": "adamant",
                "stat_points": dict(hp=32, atk=0, **{"def": 0}, spa=0, spd=34, spe=0),
                "ppm": 400_000,
            },
            {
                "nature": "jolly",
                "stat_points": dict(hp=0, atk=32, **{"def": 0}, spa=0, spd=2, spe=32),
                "ppm": 200_000,
            },
            {
                "nature": "timid",
                "stat_points": dict(hp=0, atk=0, **{"def": 0}, spa=32, spd=2, spe=32),
                "ppm": 100_000,
            },
        ],
        "truncated_ppm": 50_000,
        "unmapped_ppm": 250_000,
    }
    # Positive weights 3, 2 (Nobody, unmapped) and 1 (Rillaboom); the -2 is not weight. k_teammates 1.
    assert r["teammates"] == {
        "entries": [{"id": "salamence", "ppm": 500_000, "weight": 707_106}],
        "truncated_ppm": 166_667,
        "unmapped_ppm": 333_333,
    }


def test_salamence_lists_and_the_empty_teammate_list(small_limits: None) -> None:
    table, _ = toy()
    s = species(table, "salamence")
    assert s["items"] == {
        "entries": [{"id": "lifeorb", "ppm": 500_000}],
        "no_item_ppm": 500_000,
        "truncated_ppm": 0,
        "unmapped_ppm": 0,
    }
    # Weights 2, 2, 1, 1, 1 and an empty slot of 1; k_moves 3 keeps the two 2s and, of the tied 1s, the
    # lowest id (earthquake); roost and tailwind (2 of 8) are truncated.
    assert s["moves"]["entries"] == [
        {"id": "dragondance", "ppm": 250_000},
        {"id": "earthquake", "ppm": 125_000},
        {"id": "protect", "ppm": 250_000},
    ]
    assert (s["moves"]["empty_ppm"], s["moves"]["truncated_ppm"], s["moves"]["unmapped_ppm"]) == (
        125_000,
        250_000,
        0,
    )
    assert [e["ppm"] for e in s["spreads"]["entries"]] == [250_000, 750_000]  # all-zero first
    assert s["teammates"] == {"entries": [], "truncated_ppm": 0, "unmapped_ppm": 0}


def test_the_report_counts_the_unmapped_mass_and_fails_the_gate(small_limits: None) -> None:
    table, report = toy()
    assert report["unmapped_species"] == ["Nobodymon"]
    assert report["nonpositive_teammate_entries"] == 2
    assert report["species_in_table"] == 2
    # abilities: 10 of 11 map (the unknown species' 1 is unmapped); spreads and teammates fall below 95%.
    assert report["mapping_share_ppm"]["abilities"] == 909_090
    assert report["mapping_share_ppm"]["spreads"] == 691_176
    assert report["gate_passed"] is False
    assert report["mapping_detail_ppm"]["items"]["semantic_mass_excluded_micro"] == 3_500_000
    read_usage_weight_table(table, VOCABULARY)


def test_the_table_carries_its_sources_and_the_derivation_limits(small_limits: None) -> None:
    table, _ = toy()
    assert table["derivation"] == {
        "script_version": "usage-weights-chaos.v1",
        "k_items": 2,
        "k_moves": 3,
        "k_spreads": 3,
        "k_teammates": 1,
        "species_root": 2,
        "teammate_root": 2,
    }
    assert [s["source_id"] for s in table["sources"]] == ["chaos", "summary"]
    for source in table["sources"]:
        assert source["basis"] == "owner-accepted-risk"
        assert source["basis_reference"] == "ADR-009 erratum 1"
        assert source["url"] in source["attribution"]
    assert table["vocabulary_hash"] == sha256_hex(jcs(VOCABULARY))
    assert table["number_of_battles"] == 1000


def test_no_evs_key_is_ever_written(small_limits: None) -> None:
    table, _ = toy()
    assert b'"evs"' not in jcs(table)


def test_the_same_input_gives_the_same_bytes() -> None:
    assert jcs(toy()[0]) == jcs(toy()[0])
    assert jcs(toy(CLEAN_TEXT)[0]) == jcs(toy(CLEAN_TEXT)[0])


def test_a_species_below_one_ppm_is_omitted_and_counted() -> None:
    text = """
    {"info": {"metagame": "gen9championsvgc2026regmc", "cutoff": 1630, "number of battles": 1},
     "data": {
      "Rillaboom": {"Abilities": {"grassysurge": 1}, "Items": {"nothing": 1}, "Moves": {"": 1},
                    "Spreads": {"Adamant:0/0/0/0/0/0": 1}, "Teammates": {}},
      "Salamence": {"Abilities": {"intimidate": 1}, "Items": {"nothing": 0.0000001}, "Moves": {"": 1},
                    "Spreads": {"Adamant:0/0/0/0/0/0": 1}, "Teammates": {}}}}
    """
    table, report = toy(text)
    assert [s["id"] for s in table["species"]] == ["rillaboom"]
    assert table["species"][0]["usage_ppm"] == 1_000_000
    assert report["species_below_one_ppm"] == 1
    read_usage_weight_table(table, VOCABULARY)


def test_a_passing_toy_is_written_through_the_command(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    inputs = inputs_for(raw, CLEAN_TEXT)
    record = tmp_path / "inputs.json"
    record.write_text(json.dumps(inputs))
    out = tmp_path / "usage" / "table.json"
    report_path = tmp_path / "report.json"
    assert run(record, raw, ROOT, out, report_path) == 0
    table = json.loads(out.read_text())
    assert out.read_bytes() == jcs(table) + b"\n"
    read_usage_weight_table(table, VOCABULARY)
    report = json.loads(report_path.read_text())
    assert report["gate_passed"] is True
    assert report["usage_table_hash"] == sha256_hex(jcs(table))
    assert table["species"][0]["usage_ppm"] == 1_000_000
    assert table["species"][0]["w_species"] == 1_000_000
    # Moves: four of weight 4 are each a quarter of the list; K_MOVES 32 truncates nothing.
    assert [e["ppm"] for e in table["species"][0]["moves"]["entries"]] == [250_000] * 4
    assert main(["--inputs", str(record), "--raw-dir", str(raw), "--out", str(out)]) == 0


def test_a_failing_gate_writes_no_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    inputs = inputs_for(raw, CHAOS_TEXT)
    record = tmp_path / "inputs.json"
    record.write_text(json.dumps(inputs))
    out = tmp_path / "table.json"
    report_path = tmp_path / "report.json"
    assert run(record, raw, ROOT, out, report_path) == 1
    assert not out.exists()
    assert json.loads(report_path.read_text())["gate_passed"] is False
    assert "mapping gate failed" in capsys.readouterr().err


def test_a_file_that_differs_from_its_record_is_refused(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    inputs = inputs_for(raw, CLEAN_TEXT)
    (raw / "chaos.json").write_bytes((raw / "chaos.json").read_bytes() + b" ")
    record = tmp_path / "inputs.json"
    record.write_text(json.dumps(inputs))
    argv = ["--inputs", str(record), "--raw-dir", str(raw), "--out", str(tmp_path / "t.json")]
    with pytest.raises(DerivationError, match="differs from its record"):
        run(record, raw, ROOT, tmp_path / "t.json", None)
    assert main(argv) == 2


def test_the_summary_must_state_the_files_battle_count(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    record = tmp_path / "inputs.json"
    record.write_text(json.dumps(inputs_for(raw, CLEAN_TEXT, battles=999)))
    with pytest.raises(DerivationError, match="different battle counts"):
        run(record, raw, ROOT, tmp_path / "t.json", None)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c["info"].update(cutoff=1760), "another format or cutoff"),
        (lambda c: c["info"].update(metagame="gen9ou"), "another format or cutoff"),
        (lambda c: c["data"]["Rillaboom"].pop("Teammates"), "lacks a list"),
        (
            lambda c: c["data"].update(rillaboom=c["data"]["Rillaboom"]),
            "two source entries name one species",
        ),
        (lambda c: c["data"]["Rillaboom"].update(Items={"nothing": 0}), "no item weight"),
        (lambda c: c["data"]["Rillaboom"].update(Abilities={}), "no weight cannot be scaled"),
        (
            lambda c: c["data"]["Rillaboom"]["Spreads"].update({"adamant:32/0/0/0/34/0": 1}),
            "two spread keys name one spread",
        ),
        (lambda c: c["data"]["Rillaboom"].update(Abilities={"grassysurge": -1}), "negative"),
        (lambda c: c["data"]["Rillaboom"].update(Teammates={"Salamence": "x"}), "not a number"),
    ],
)
def test_refusals(change: Any, message: str) -> None:
    chaos = json.loads(CLEAN_TEXT)
    change(chaos)
    with pytest.raises(DerivationError, match=message):
        derive(parse_source(json.dumps(chaos)), INPUTS, VOCABULARY, ENGINE_DATA)


def test_the_source_reader_refuses_duplicate_keys_and_nan() -> None:
    with pytest.raises(DerivationError, match="duplicate object key"):
        parse_source('{"a": 1, "a": 2}')
    with pytest.raises(DerivationError, match="NaN or Infinity"):
        parse_source('{"a": NaN}')


def test_an_unmapped_species_with_non_positive_teammates_still_counts_its_mass() -> None:
    chaos = json.loads(CHAOS_TEXT)
    chaos["data"]["Nobodymon"]["Teammates"] = {"Rillaboom": -5, "Salamence": 2}
    table, report = derive(parse_source(json.dumps(chaos)), INPUTS, VOCABULARY, ENGINE_DATA)
    assert report["mapping_detail_ppm"]["teammates"]["no_engine_id"] > 0
    assert copy.deepcopy(table["unmapped_species_ppm"]) == 125_000
