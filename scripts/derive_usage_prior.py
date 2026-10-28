"""Derive a ``UsagePrior`` of kind ``smogon-chaos`` from a recorded Smogon chaos file.

ADR-025 section 2, #1120.

Offline and deterministic: standard library only (plus the contracts package for RFC 8785 and the
reader), decimal parsing, no network. The raw files stay in a gitignored directory; an input
record names each by file name, URL, retrieval time, size and SHA-256, and a file whose size or
hash differs is refused. Only aggregate numbers are read and written (the keys ``Teammates``,
``Checks and Counters``, ``Viability Ceiling``, ``Tera Types`` and ``Happiness`` are never read).

    PYTHONPATH=.:packages/schema/python python -m scripts.derive_usage_prior \\
        --inputs config/opponent-model/<format>/smogon-chaos.<month>-<cutoff>.inputs.json \\
        --raw-dir .tmp/chaos \\
        --out config/opponent-model/<format>/usage-prior.<month>-<cutoff>.json \\
        [--report REPORT.json] [--fallback]

Rules (ADR-025 section 2; each is tested on a hand-computed toy file):

* Every name maps to an engine id by exact id or exact name through the engine-data table. A name
  that maps to no id, or to an id the vocabulary does not allow for that forme (an item outside
  the format's item list, an ability or move outside the forme's roster entry, a spread over the
  Stat Point limit), is ``unmapped``: counted, reported, never remapped, never redistributed.
* Semantic keys: ``Items`` ``nothing`` is ``no_item`` (a legal option) and ``Moves`` ``''`` is
  ``empty_slots`` (an empty move slot). Neither is unmapped mass or counts against the gate.
* A preview species is the form Team Preview shows (taken from the generator-matched prior of the
  same vocabulary, which derives it from the engine's Mega stone data). Its formes are the chaos
  entries that show as it; a forme's weight is its entry's summed item weight, so for a Mega
  forme it is the stone-holding share. A Mega forme's item is its stone and its ability list the
  post-Evolution ability, both as the entry says; the pre-Evolution ability of a stone holder is the
  consumer's approximation from the base forme's list (this record carries no ``holder_abilities``).
* Truncation: items ``k_items`` 16, moves ``k_moves`` 24, spreads ``k_spreads`` 32, by weight,
  ties by id. Dropped mapped mass is ``residual``. Weights are scaled to an integer denominator by
  ``usage_common`` (the unit rule) and largest remainder, ties by id.
* A move list has ``slots`` = the ratio of its summed weight to the item weight (4, refused if not
  an integer within 10^-6), each move's inclusion is its share of ``slots * denominator``, and the
  empty-slot mass is the ``''`` weight.
* Names are mapped by exact id, exact name, or the engine's own ``toID`` of the name (so
  ``Farfetch'd`` finds the engine's ``farfetchd``); no spelling is listed or guessed.
* Fallback (``--fallback``, #1209): a vocabulary preview species with no chaos entry (it was never
  seen at the cutoff) would make the belief refuse any roster holding it. With the flag its entry is
  copied from the preview-species map, the committed ``generator-matched`` prior of the same
  vocabulary, truncated to this record's limits (the dropped weights go to ``residual``, which the
  belief expands equally over the options not named, exactly the map's equal weights for moves and
  within the vocabulary's items for items), and the record's ``source.fallback`` names the prior by
  hash and lists the species taken from it. Those species carry no usage information.
* Gate (ADR-020 P1): at least 95% of each list kind's mass maps, else no record is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex, validate
from areneon_contracts.usage_prior import read_usage_prior

from scripts.usage_common import (
    UNIT,
    DerivationError,
    NameTable,
    fraction,
    largest_remainder,
    parse_source,
    to_units,
    top_k,
)

SCRIPT_VERSION = "usage-prior-chaos.v2"
# Derivation parameters, not game rules: ADR-025 section 2 (truncation) and ADR-020 P1 (gate).
K_ITEMS, K_MOVES, K_SPREADS = 16, 24, 32
GATE_PPM = 950_000
MAX_POINTS = 255  # the contract's bound of one Stat Point value
MAX_SLOTS = 8  # the contract's bound of the move slots
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
KINDS = ("abilities", "items", "moves", "spreads")
SPREAD_KEY = re.compile(r"^([A-Za-z]+):([0-9]{1,3})(?:/([0-9]{1,3})){5}$")
# The only keys of an entry that are read; the others (teammates, counters, ...) are never touched.
READ_KEYS = ("Abilities", "Items", "Moves", "Spreads")
# Order of the special groups after the named ids when a tie is broken.
SPECIAL = {"@no_item": 1, "@empty": 2, "@residual": 3, "@unmapped": 4}


def _order(key: str) -> tuple[int, str]:
    return (SPECIAL.get(key, 0), key)


class Mass:
    """Mass per list kind (exact fractions), for the mapping report and the gate."""

    def __init__(self) -> None:
        self.mapped = Fraction(0)
        self.no_id = Fraction(0)
        self.outside = Fraction(0)
        self.semantic = Fraction(0)

    def total(self) -> Fraction:
        return self.mapped + self.no_id + self.outside

    def ppm(self) -> int | None:
        return None if self.total() == 0 else int(self.mapped * UNIT / self.total())


def _grouped(
    mapped: Mapping[str, Fraction], kept_ids: Sequence[str], special: Mapping[str, Fraction]
) -> dict[str, Fraction]:
    groups = {f"i:{k}": mapped[k] for k in kept_ids}
    groups.update(special)
    return groups


def _scale(
    total: int, groups: Mapping[str, Fraction], caps: Mapping[str, int] | None = None
) -> dict[str, int]:
    return largest_remainder(
        total, groups, caps, order=lambda k: _order(k[2:] if k.startswith("i:") else k)
    )


class Deriver:
    """One derivation: the tables, the vocabulary and the accumulated report."""

    def __init__(
        self,
        vocabulary: Mapping[str, Any],
        engine_data: Mapping[str, Any],
        shows_as: Mapping[str, str],
    ) -> None:
        self.vocabulary = vocabulary
        self.roster = {e["id"]: e for e in vocabulary["roster"]}
        self.items_ok = set(vocabulary["items"])
        self.natures = {n["id"] for n in vocabulary["natures"]}
        self.ev_limit = vocabulary["rules"]["ev_limit"]
        self.species = NameTable(engine_data["species"])
        self.moves = NameTable(engine_data["moves"])
        self.items = NameTable(engine_data["items"])
        self.abilities = NameTable(engine_data["abilities"])
        self.shows_as = shows_as
        self.mass = {kind: Mass() for kind in KINDS}
        self.unmapped_species: dict[str, Fraction] = {}
        self.dropped_formes: list[str] = []

    # -- lists ---------------------------------------------------------------------------------

    def _abilities(self, entry: Mapping[str, Any], forme: str) -> dict[str, Any]:
        pool = set(self.roster[forme]["abilities"])
        return self._plain(
            entry["Abilities"], "abilities", self.abilities, lambda i: i in pool, None
        )

    def _items(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        return self._plain(
            entry["Items"], "items", self.items, lambda i: i in self.items_ok, K_ITEMS
        )

    def _plain(
        self,
        raw: Mapping[str, Any],
        kind: str,
        table: NameTable,
        allowed: Any,
        limit: int | None,
    ) -> dict[str, Any]:
        mass = self.mass[kind]
        mapped: dict[str, Fraction] = {}
        no_item = Fraction(0)
        unmapped = Fraction(0)
        for key, value in raw.items():
            w = fraction(value)
            if kind == "items" and key == "nothing":
                no_item += w
                mass.semantic += w
                continue
            ident = table.resolve(key)
            if ident is None:
                mass.no_id += w
                unmapped += w
            elif not allowed(ident):
                mass.outside += w
                unmapped += w
            else:
                mass.mapped += w
                mapped[ident] = mapped.get(ident, Fraction(0)) + w
        kept = top_k(mapped, limit if limit is not None else len(mapped))
        residual = sum((mapped[k] for k in mapped if k not in kept), Fraction(0))
        denominator = to_units(_sum(raw))
        special = {"@residual": residual, "@unmapped": unmapped}
        if kind == "items":
            special["@no_item"] = no_item
        scaled = _scale(denominator, _grouped(mapped, kept, special))
        out: dict[str, Any] = {
            "denominator": denominator,
            "entries": [
                {"id": k, "weight": scaled[f"i:{k}"]} for k in sorted(kept) if scaled[f"i:{k}"] > 0
            ],
            "residual": scaled["@residual"],
            "unmapped": scaled["@unmapped"],
        }
        if kind == "items":
            out["no_item"] = scaled["@no_item"]
            out = {k: out[k] for k in ("denominator", "entries", "no_item", "residual", "unmapped")}
        return out

    def _spreads(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        mass = self.mass["spreads"]
        mapped: dict[str, Fraction] = {}
        parsed: dict[str, tuple[str, tuple[int, ...]]] = {}
        unmapped = Fraction(0)
        for key, value in raw.items():
            w = fraction(value)
            match = SPREAD_KEY.match(key)
            if match is None:
                mass.no_id += w
                unmapped += w
                continue
            nature, _, points = key.partition(":")
            values = tuple(int(p) for p in points.split("/"))
            if nature.lower() not in self.natures or max(values) > MAX_POINTS:
                mass.no_id += w
                unmapped += w
            elif sum(values) > self.ev_limit:
                mass.outside += w
                unmapped += w
            else:
                mass.mapped += w
                mapped[key] = w
                parsed[key] = (nature.lower(), values)
        if len(set(parsed.values())) != len(parsed):
            raise DerivationError("two spread keys name one spread")
        kept = top_k(mapped, K_SPREADS)
        residual = sum((mapped[k] for k in mapped if k not in kept), Fraction(0))
        denominator = to_units(_sum(raw))
        scaled = _scale(
            denominator, _grouped(mapped, kept, {"@residual": residual, "@unmapped": unmapped})
        )
        entries: list[dict[str, Any]] = [
            {
                "nature": parsed[k][0],
                "stat_points": dict(zip(STATS, parsed[k][1], strict=True)),
                "weight": scaled[f"i:{k}"],
            }
            for k in kept
            if scaled[f"i:{k}"] > 0
        ]
        entries.sort(key=lambda e: (e["nature"], *(e["stat_points"][s] for s in STATS)))
        return {
            "denominator": denominator,
            "entries": entries,
            "residual": scaled["@residual"],
            "unmapped": scaled["@unmapped"],
        }

    def _moves(
        self, raw: Mapping[str, Any], forme: str, denominator: int, set_weight: Fraction
    ) -> dict[str, Any]:
        mass = self.mass["moves"]
        pool = set(self.roster[forme]["moves"])
        total = _sum(raw)
        ratio = total / set_weight
        slots = round(ratio)
        if not 1 <= slots <= MAX_SLOTS or abs(ratio - slots) >= Fraction(1, UNIT):
            raise DerivationError("a move list does not sum to a whole number of slots")
        mapped: dict[str, Fraction] = {}
        empty = Fraction(0)
        unmapped = Fraction(0)
        for key, value in raw.items():
            w = fraction(value)
            if key == "":
                empty += w
                mass.semantic += w
                continue
            ident = self.moves.resolve(key)
            if ident is None:
                mass.no_id += w
                unmapped += w
            elif ident not in pool:
                mass.outside += w
                unmapped += w
            else:
                mass.mapped += w
                mapped[ident] = mapped.get(ident, Fraction(0)) + w
        kept = top_k(mapped, K_MOVES)
        residual = sum((mapped[k] for k in mapped if k not in kept), Fraction(0))
        groups = _grouped(
            mapped, kept, {"@empty": empty, "@residual": residual, "@unmapped": unmapped}
        )
        caps = {f"i:{k}": denominator for k in kept}
        caps["@empty"] = (slots - 1) * denominator
        scaled = _scale(slots * denominator, groups, caps)
        return {
            "denominator": denominator,
            "slots": slots,
            "entries": [
                {"id": k, "inclusion": scaled[f"i:{k}"]}
                for k in sorted(kept)
                if scaled[f"i:{k}"] > 0
            ],
            "empty_slots": scaled["@empty"],
            "residual": scaled["@residual"],
            "unmapped": scaled["@unmapped"],
        }

    # -- formes and species -----------------------------------------------------------------------

    def forme(self, name: str, entry: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
        """One chaos entry as a forme of its preview species, or None without usable weight."""
        for key in READ_KEYS:
            if key not in entry:
                raise DerivationError("a source entry lacks a list")
        sid = self.species.resolve(name)
        ident = sid if sid in self.roster and sid in self.shows_as else None
        if ident is None:
            weight = _sum(entry["Items"])
            self.unmapped_species[name] = weight
            self._whole_unmapped(entry)
            return None
        set_weight = _sum(entry["Items"])
        denominator = to_units(set_weight)
        if denominator == 0:
            self.dropped_formes.append(ident)
            return None
        items = self._items(entry)
        body = {
            "id": ident,
            "weight": items["denominator"],
            "abilities": self._abilities(entry, ident),
            "items": items,
            "spreads": self._spreads(entry["Spreads"]),
            "moves": self._moves(entry["Moves"], ident, denominator, set_weight),
        }
        return self.shows_as[ident], body

    def _whole_unmapped(self, entry: Mapping[str, Any]) -> None:
        """A species that maps to no usable id: all its mass is unmapped (semantic keys aside)."""
        for kind, key, semantic in (
            ("abilities", "Abilities", None),
            ("items", "Items", "nothing"),
            ("moves", "Moves", ""),
            ("spreads", "Spreads", None),
        ):
            for k, v in entry[key].items():
                w = fraction(v)
                if k == semantic:
                    self.mass[kind].semantic += w
                else:
                    self.mass[kind].no_id += w


def _sum(raw: Mapping[str, Any]) -> Fraction:
    return sum((fraction(v) for v in raw.values()), Fraction(0))


def _truncated(entries: Sequence[Mapping[str, Any]], key: str, limit: int) -> tuple[list[Any], int]:
    """The ``limit`` heaviest entries (ties by id), sorted by id, and the weight dropped."""
    kept = sorted(entries, key=lambda e: (-e[key], e["id"]))[:limit]
    ids = {e["id"] for e in kept}
    dropped = sum(e[key] for e in entries if e["id"] not in ids)
    return sorted(kept, key=lambda e: e["id"]), dropped


def fallback_species(species: Mapping[str, Any]) -> dict[str, Any]:
    """One species of the fallback prior as an entry of this record: lists cut to the limits."""
    out = json.loads(json.dumps(species))
    for forme in out["formes"]:
        forme["items"]["entries"], dropped = _truncated(
            forme["items"]["entries"], "weight", K_ITEMS
        )
        forme["items"]["residual"] += dropped
        moves = forme["moves"]
        moves["entries"], dropped = _truncated(moves["entries"], "inclusion", K_MOVES)
        moves["residual"] += dropped
        if len(forme["spreads"]["entries"]) > K_SPREADS:
            raise DerivationError("a fallback forme names more spreads than the record allows")
    return out  # type: ignore[no-any-return]


def derive(  # noqa: PLR0913, PLR0917 - the five inputs of the derivation and the optional fallback
    chaos: Mapping[str, Any],
    inputs: Mapping[str, Any],
    vocabulary: Mapping[str, Any],
    engine_data: Mapping[str, Any],
    shows_as: Mapping[str, str],
    fallback: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The record and the report of one derivation (pure; the gate is the caller's)."""
    info = chaos["info"]
    if info["metagame"] != vocabulary["format_id"] or info["cutoff"] != inputs["cutoff"]:
        raise DerivationError("the file is for another format or cutoff than the input record")
    deriver = Deriver(vocabulary, engine_data, shows_as)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(chaos["data"]):
        made = deriver.forme(name, chaos["data"][name])
        if made is not None:
            grouped.setdefault(made[0], []).append(made[1])
    species = []
    for sid in sorted(grouped):
        formes = sorted(grouped[sid], key=lambda f: f["id"])
        species.append(
            {"id": sid, "denominator": sum(f["weight"] for f in formes), "formes": formes}
        )
    filled: list[str] = []
    if fallback is not None:
        have = {s["id"] for s in species}
        for entry in fallback["species"]:
            if entry["id"] not in have:
                species.append(fallback_species(entry))
                filled.append(entry["id"])
        species.sort(key=lambda s: s["id"])
        filled.sort()
    record: dict[str, Any] = {
        "schema_version": 1,
        "format_id": vocabulary["format_id"],
        "engine_data_hash": vocabulary["engine_data_hash"],
        "vocabulary_hash": sha256_hex(jcs(vocabulary)),
        "source": {
            "kind": "smogon-chaos",
            "month": inputs["month"],
            "cutoff": inputs["cutoff"],
            "number_of_battles": info["number of battles"],
            "inputs": sorted(
                (
                    {k: f[k] for k in ("source_id", "url", "retrieved_at", "size", "sha256")}
                    for f in inputs["files"]
                ),
                key=lambda f: f["source_id"],
            ),
        },
        "derivation": {
            "script_version": SCRIPT_VERSION,
            "k_items": K_ITEMS,
            "k_moves": K_MOVES,
            "k_spreads": K_SPREADS,
        },
        "species": species,
    }
    if fallback is not None:
        record["source"]["fallback"] = {
            "kind": fallback["source"]["kind"],
            "prior_hash": sha256_hex(jcs(fallback)),
            "species": filled,
        }
    return record, report_of(chaos, deriver, record)


def n8_reading(usage: Fraction) -> str:
    """N8 from the file's own fields: ``usage`` sums over species to the size counted over."""
    if abs(usage - 6) < Fraction(1, 100):
        return (
            "usage sums to six per team: counts are over the six registered, not the four brought"
        )
    if abs(usage - 4) < Fraction(1, 100):
        return "usage sums to four per team: counts are over the four brought"
    return "usage sums to neither six nor four: undetermined"


def report_of(
    chaos: Mapping[str, Any], deriver: Deriver, record: Mapping[str, Any]
) -> dict[str, Any]:
    """The derivation report: integers and strings only (shares in parts per million)."""
    data = chaos["data"]
    battles = chaos["info"]["number of battles"]
    usage = sum((fraction(e["usage"]) for e in data.values()), Fraction(0))
    raw = sum((fraction(e["Raw count"]) for e in data.values()), Fraction(0))
    shares = {kind: m.ppm() for kind, m in deriver.mass.items()}
    return {
        "script_version": SCRIPT_VERSION,
        "source_entries": len(data),
        "species_in_record": len(record["species"]),
        "fallback_species": len(record["source"].get("fallback", {}).get("species", [])),
        "formes_in_record": sum(len(s["formes"]) for s in record["species"]),
        "mapping_share_ppm": shares,
        "mapping_detail_ppm": {
            kind: {
                "no_engine_id": int(m.no_id * UNIT / m.total()) if m.total() else 0,
                "outside_vocabulary": int(m.outside * UNIT / m.total()) if m.total() else 0,
                "semantic_mass_excluded_micro": int(m.semantic * UNIT),
            }
            for kind, m in deriver.mass.items()
        },
        "gate_ppm": GATE_PPM,
        "gate_passed": all(s is not None and s >= GATE_PPM for s in shares.values()),
        "unmapped_species": sorted(deriver.unmapped_species),
        "dropped_formes": sorted(deriver.dropped_formes),
        # N8: which count the file's numbers are over, from its own fields.
        "n8": {
            "number_of_battles": battles,
            "sum_usage_micro": int(usage * UNIT),
            "sum_raw_count": int(raw),
            "raw_count_per_team_micro": int(raw * UNIT / (2 * battles)),
            "reading": n8_reading(usage),
        },
    }


def leads_report(text: str, deriver: Deriver) -> dict[str, Any]:
    """O3: the ``leads`` names checked against the engine-data species (reported only)."""
    names = re.findall(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|", text, flags=re.M)
    unmapped = sorted(n for n in names if deriver.species.resolve(n) is None)
    return {"names": len(names), "unmapped": unmapped}


def summary_battles(text: str) -> int:
    match = re.match(r"^Total battles: ([0-9]+)$", text.splitlines()[0])
    if match is None:
        raise DerivationError("the summary file does not start with a battle count")
    return int(match.group(1))


def verified_files(inputs: Mapping[str, Any], raw_dir: Path) -> dict[str, bytes]:
    """Each input file's bytes, refused unless the size and the SHA-256 are the record's."""
    out = {}
    for f in inputs["files"]:
        data = (raw_dir / f["file"]).read_bytes()
        if len(data) != f["size"] or hashlib.sha256(data).hexdigest() != f["sha256"]:
            raise DerivationError(f"input {f['source_id']} differs from its record")
        out[f["source_id"]] = data
    return out


def run(  # noqa: PLR0913, PLR0917 - the command's paths and its one flag
    inputs_path: Path,
    raw_dir: Path,
    root: Path,
    out: Path,
    report_path: Path | None,
    use_fallback: bool = False,
) -> int:
    inputs = json.loads(inputs_path.read_text())
    fmt = inputs["format_id"]
    fixtures = root / "fixtures" / "showdown" / fmt
    vocabulary = json.loads((fixtures / "team-vocabulary.json").read_text())
    engine_data = json.loads((fixtures / "engine-data.json").read_text())
    if sha256_hex(jcs(engine_data)) != vocabulary["engine_data_hash"]:
        raise DerivationError("the engine-data table is not the one the vocabulary names")
    preview = json.loads(
        (root / "config" / "opponent-model" / fmt / inputs["preview_map"]).read_text()
    )
    if preview["vocabulary_hash"] != sha256_hex(jcs(vocabulary)):
        raise DerivationError("the preview-species map was built for another vocabulary")
    shows_as = {f["id"]: s["id"] for s in preview["species"] for f in s["formes"]}
    files = verified_files(inputs, raw_dir)
    chaos = parse_source(files["chaos"].decode())
    if summary_battles(files["summary"].decode()) != chaos["info"]["number of battles"]:
        raise DerivationError("the summary and the chaos file state different battle counts")
    record, report = derive(
        chaos, inputs, vocabulary, engine_data, shows_as, preview if use_fallback else None
    )
    deriver = Deriver(vocabulary, engine_data, shows_as)
    report["leads"] = leads_report(files["leads"].decode(), deriver)
    validate("usage-prior", record)
    read_usage_prior(record, vocabulary)
    report["prior_hash"] = sha256_hex(jcs(record))
    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["gate_passed"]:
        print("mapping gate failed: no record written", file=sys.stderr)
        return 1
    out.write_bytes(jcs(record) + b"\n")
    print(json.dumps(report, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="fill the species the source lacks from the preview-species map (generator-matched)",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        return run(args.inputs, args.raw_dir, args.root, args.out, args.report, args.fallback)
    except DerivationError as exc:
        print(f"derive_usage_prior: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
