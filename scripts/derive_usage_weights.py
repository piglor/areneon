"""Derive a ``UsageWeightTable`` from a recorded Smogon chaos file (ADR-020 section 4, #976).

Offline and deterministic: standard library only (plus the contracts package for RFC 8785 and the
reader), decimal parsing, no network. It reads the same input record as ``derive_usage_prior`` (the
raw files stay in a gitignored directory; each is named by file, URL, retrieval time, size and
SHA-256, and a file whose size or hash differs is refused) and the same ``usage_common`` pieces.

    PYTHONPATH=.:packages/schema/python python -m scripts.derive_usage_weights \\
        --inputs config/opponent-model/<format>/smogon-chaos.<month>-<cutoff>.inputs.json \\
        --raw-dir .tmp/chaos \\
        --out config/teams/<format>/usage/<month>-<cutoff>.json \\
        [--report REPORT.json]

Rules (each is tested on a hand-computed toy file):

* A chaos entry is a species of the table when its name maps (exact id, exact name or the engine's
  ``toID`` of the name) to a roster entry of the team vocabulary. Its weight is its summed item
  weight (the number of sets, as in ``derive_usage_prior``). Species weights become parts per
  million of the summed species weight by largest remainder, ties by id; an entry that maps to no
  roster species is ``unmapped_species_ppm``. A species whose share rounds to 0 ppm is omitted.
* Every other list is in ppm of that species' own list total (largest remainder, ties by id;
  named ids before the special groups): the top-K entries by weight (ties by id),
  ``truncated_ppm`` (mapped mass the cut dropped), ``unmapped_ppm`` (a name with no engine id, or an
  id the vocabulary does not allow here: an item outside the item list, an ability or move outside
  the species' roster entry, a spread over the Stat Point limit or with an unknown nature, a
  teammate outside the roster; counted, never remapped) and the semantic key (``nothing`` in Items
  is ``no_item_ppm``, ``''`` in Moves is ``empty_ppm``; neither is unmapped mass nor counts against
  the gate).
* ``Teammates`` is read only for names and weights, and only positive weights count (a non-positive
  value carries no co-occurrence: it is reported, not weighted). A species with no positive teammate
  weight has an empty list with every mass 0.
* ``w_species`` and a teammate's ``weight`` are the exact integer root of the ppm
  (``floor((ppm * 10^(6 * (r - 1)))^(1/r))``, so r = 1 leaves the ppm and a full share stays a
  million), with the table's ``species_root`` and ``teammate_root``.
* The key ``evs`` is never used; spreads are ``Nature:hp/atk/def/spa/spd/spe`` Stat Points.
* Gate (ADR-020 P1): at least 95% of each list kind's mass maps, else no table is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex, validate
from areneon_contracts.usage_weight_table import read_usage_weight_table, rooted_weight

from scripts.derive_usage_prior import (
    GATE_PPM,
    MAX_POINTS,
    SPREAD_KEY,
    STATS,
    Mass,
    summary_battles,
    verified_files,
)
from scripts.usage_common import (
    UNIT,
    DerivationError,
    NameTable,
    fraction,
    largest_remainder,
    parse_source,
    top_k,
)

SCRIPT_VERSION = "usage-weights-chaos.v1"
# Derivation parameters, not game rules: ADR-020 section 4 (truncation and roots) and P1 (gate).
K_ITEMS, K_MOVES, K_SPREADS, K_TEAMMATES = 16, 32, 32, 48
SPECIES_ROOT, TEAMMATE_ROOT = 2, 2
KINDS = ("abilities", "items", "moves", "spreads", "teammates")
READ_KEYS = ("Abilities", "Items", "Moves", "Spreads", "Teammates")
BASIS = "owner-accepted-risk"
BASIS_REFERENCE = "ADR-009 erratum 1"
# Rank of a group when a remainder ties: named ids first, then the special groups.
SPECIAL = {"@semantic": 1, "@truncated": 2, "@unmapped": 3}


def _order(key: str) -> tuple[int, str]:
    return (SPECIAL.get(key, 0), key)


def _sum(raw: Mapping[str, Any]) -> Fraction:
    return sum((fraction(v) for v in raw.values()), Fraction(0))


class Deriver:
    """One derivation: the tables, the vocabulary and the accumulated report."""

    def __init__(self, vocabulary: Mapping[str, Any], engine_data: Mapping[str, Any]) -> None:
        self.vocabulary = vocabulary
        self.roster = {e["id"]: e for e in vocabulary["roster"]}
        self.items_ok = set(vocabulary["items"])
        self.natures = {n["id"] for n in vocabulary["natures"]}
        self.ev_limit = vocabulary["rules"]["ev_limit"]
        self.species = NameTable(engine_data["species"])
        self.moves = NameTable(engine_data["moves"])
        self.items = NameTable(engine_data["items"])
        self.abilities = NameTable(engine_data["abilities"])
        self.mass = {kind: Mass() for kind in KINDS}
        self.unmapped_species: dict[str, Fraction] = {}
        self.nonpositive_teammates = 0

    # -- lists ---------------------------------------------------------------------------------

    def _list(  # noqa: PLR0913, PLR0917 - the list kind, its raw weights, the two name rules and the cut
        self,
        kind: str,
        raw: Mapping[str, Any],
        resolve: Callable[[str], str | None],
        allowed: Callable[[str], bool],
        limit: int | None,
        semantic_key: str | None = None,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """The ppm of every kept id and of the special groups of one list (summing to 10^6)."""
        mass = self.mass[kind]
        mapped: dict[str, Fraction] = {}
        semantic = unmapped = Fraction(0)
        for key, value in raw.items():
            w = fraction(value)
            if key == semantic_key:
                semantic += w
                mass.semantic += w
                continue
            ident = resolve(key)
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
        groups = {k: mapped[k] for k in kept}
        groups["@truncated"] = sum((w for k, w in mapped.items() if k not in kept), Fraction(0))
        groups["@unmapped"] = unmapped
        groups["@semantic"] = semantic
        scaled = largest_remainder(UNIT, groups, order=_order)
        entries = {k: scaled[k] for k in sorted(kept) if scaled[k] > 0}
        return entries, {k: scaled[k] for k in SPECIAL}

    def _abilities(self, raw: Mapping[str, Any], sid: str) -> dict[str, Any]:
        pool = set(self.roster[sid]["abilities"])
        entries, special = self._list(
            "abilities", raw, self.abilities.resolve, pool.__contains__, None
        )
        return {
            "entries": [{"id": k, "ppm": v} for k, v in entries.items()],
            "unmapped_ppm": special["@unmapped"],
        }

    def _items(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        entries, special = self._list(
            "items", raw, self.items.resolve, self.items_ok.__contains__, K_ITEMS, "nothing"
        )
        return {
            "entries": [{"id": k, "ppm": v} for k, v in entries.items()],
            "no_item_ppm": special["@semantic"],
            "truncated_ppm": special["@truncated"],
            "unmapped_ppm": special["@unmapped"],
        }

    def _moves(self, raw: Mapping[str, Any], sid: str) -> dict[str, Any]:
        pool = set(self.roster[sid]["moves"])
        entries, special = self._list(
            "moves", raw, self.moves.resolve, pool.__contains__, K_MOVES, ""
        )
        return {
            "entries": [{"id": k, "ppm": v} for k, v in entries.items()],
            "empty_ppm": special["@semantic"],
            "truncated_ppm": special["@truncated"],
            "unmapped_ppm": special["@unmapped"],
        }

    def _spreads(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        def resolve(key: str) -> str | None:
            if SPREAD_KEY.match(key) is None:
                return None
            nature, _, points = key.partition(":")
            values = [int(p) for p in points.split("/")]
            return key if nature.lower() in self.natures and max(values) <= MAX_POINTS else None

        def allowed(key: str) -> bool:
            total: int = sum(int(p) for p in key.partition(":")[2].split("/"))
            return total <= int(self.ev_limit)

        entries, special = self._list("spreads", raw, resolve, allowed, K_SPREADS)
        parsed: dict[tuple[str, tuple[int, ...]], int] = {}
        for key, ppm in entries.items():
            nature, _, points = key.partition(":")
            spread = (nature.lower(), tuple(int(p) for p in points.split("/")))
            if spread in parsed:
                raise DerivationError("two spread keys name one spread")
            parsed[spread] = ppm
        return {
            "entries": [
                {"nature": n, "stat_points": dict(zip(STATS, p, strict=True)), "ppm": ppm}
                for (n, p), ppm in sorted(parsed.items())
            ],
            "truncated_ppm": special["@truncated"],
            "unmapped_ppm": special["@unmapped"],
        }

    def _teammates(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        positive: dict[str, Any] = {}
        for key, value in raw.items():
            if fraction_signed(value) > 0:
                positive[key] = value
            else:
                self.nonpositive_teammates += 1
        if not positive:
            return {"entries": [], "truncated_ppm": 0, "unmapped_ppm": 0}
        entries, special = self._list(
            "teammates", positive, self.species.resolve, self.roster.__contains__, K_TEAMMATES
        )
        return {
            "entries": [
                {"id": k, "ppm": v, "weight": rooted_weight(v, TEAMMATE_ROOT)}
                for k, v in entries.items()
            ],
            "truncated_ppm": special["@truncated"],
            "unmapped_ppm": special["@unmapped"],
        }

    # -- species -------------------------------------------------------------------------------

    def unmapped_whole(self, entry: Mapping[str, Any]) -> None:
        """A species that maps to no roster entry: all its mass is unmapped, semantic keys aside."""
        for kind, key, semantic in (
            ("abilities", "Abilities", None),
            ("items", "Items", "nothing"),
            ("moves", "Moves", ""),
            ("spreads", "Spreads", None),
            ("teammates", "Teammates", None),
        ):
            for k, v in entry[key].items():
                if kind == "teammates" and fraction_signed(v) <= 0:
                    continue
                w = fraction(v)
                if k == semantic:
                    self.mass[kind].semantic += w
                else:
                    self.mass[kind].no_id += w

    def species_lists(self, sid: str, entry: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "items": self._items(entry["Items"]),
            "abilities": self._abilities(entry["Abilities"], sid),
            "moves": self._moves(entry["Moves"], sid),
            "spreads": self._spreads(entry["Spreads"]),
            "teammates": self._teammates(entry["Teammates"]),
        }


def fraction_signed(value: Any) -> Fraction:
    """An exact fraction of a parsed source number that may be negative (teammate deviations)."""
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise DerivationError("a source weight is not a number")
    return Fraction(value)


def derive(
    chaos: Mapping[str, Any],
    inputs: Mapping[str, Any],
    vocabulary: Mapping[str, Any],
    engine_data: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The table and the report of one derivation (pure; the gate is the caller's)."""
    info = chaos["info"]
    if info["metagame"] != vocabulary["format_id"] or info["cutoff"] != inputs["cutoff"]:
        raise DerivationError("the file is for another format or cutoff than the input record")
    deriver = Deriver(vocabulary, engine_data)
    weights: dict[str, Fraction] = {}
    lists: dict[str, dict[str, Any]] = {}
    unmapped_weight = Fraction(0)
    for name in sorted(chaos["data"]):
        entry = chaos["data"][name]
        for key in READ_KEYS:
            if key not in entry:
                raise DerivationError("a source entry lacks a list")
        weight = _sum(entry["Items"])
        sid = deriver.species.resolve(name)
        if sid is None or sid not in deriver.roster:
            deriver.unmapped_species[name] = weight
            deriver.unmapped_whole(entry)
            unmapped_weight += weight
        elif sid in weights:
            raise DerivationError("two source entries name one species")
        elif weight == 0:
            raise DerivationError("a species has no item weight")
        else:
            weights[sid] = weight
            lists[sid] = deriver.species_lists(sid, entry)
    groups: dict[str, Fraction] = dict(weights)
    groups["@unmapped"] = unmapped_weight
    shares = largest_remainder(UNIT, groups, order=_order)
    species = [
        {
            "id": sid,
            "usage_ppm": shares[sid],
            "w_species": rooted_weight(shares[sid], SPECIES_ROOT),
            **lists[sid],
        }
        for sid in sorted(weights)
        if shares[sid] > 0
    ]
    table: dict[str, Any] = {
        "schema_version": 1,
        "format_id": vocabulary["format_id"],
        "engine_data_hash": vocabulary["engine_data_hash"],
        "vocabulary_hash": sha256_hex(jcs(vocabulary)),
        "month": inputs["month"],
        "cutoff": inputs["cutoff"],
        "number_of_battles": info["number of battles"],
        "derivation": {
            "script_version": SCRIPT_VERSION,
            "k_items": K_ITEMS,
            "k_moves": K_MOVES,
            "k_spreads": K_SPREADS,
            "k_teammates": K_TEAMMATES,
            "species_root": SPECIES_ROOT,
            "teammate_root": TEAMMATE_ROOT,
        },
        "sources": sorted(
            (
                {
                    **{k: f[k] for k in ("source_id", "url", "retrieved_at", "size", "sha256")},
                    "attribution": (
                        f"Smogon University usage statistics, {inputs['month']}, "
                        f"{inputs['format_id']}, rating cutoff {inputs['cutoff']}: {f['url']}"
                    ),
                    "basis": BASIS,
                    "basis_reference": BASIS_REFERENCE,
                }
                for f in inputs["files"]
            ),
            key=lambda s: s["source_id"],
        ),
        "unmapped_species_ppm": shares["@unmapped"],
        "species": species,
    }
    return table, report_of(chaos, deriver, table, shares)


def report_of(
    chaos: Mapping[str, Any],
    deriver: Deriver,
    table: Mapping[str, Any],
    shares: Mapping[str, int],
) -> dict[str, Any]:
    """The derivation report: integers and strings only (shares in parts per million)."""
    kinds = {kind: m.ppm() for kind, m in deriver.mass.items()}
    return {
        "script_version": SCRIPT_VERSION,
        "source_entries": len(chaos["data"]),
        "species_in_table": len(table["species"]),
        "species_below_one_ppm": sum(1 for k, v in shares.items() if k != "@unmapped" and v == 0),
        "unmapped_species_ppm": table["unmapped_species_ppm"],
        "unmapped_species": sorted(deriver.unmapped_species),
        "nonpositive_teammate_entries": deriver.nonpositive_teammates,
        "mapping_share_ppm": kinds,
        "mapping_detail_ppm": {
            kind: {
                "no_engine_id": int(m.no_id * UNIT / m.total()) if m.total() else 0,
                "outside_vocabulary": int(m.outside * UNIT / m.total()) if m.total() else 0,
                "semantic_mass_excluded_micro": int(m.semantic * UNIT),
            }
            for kind, m in deriver.mass.items()
        },
        "gate_ppm": GATE_PPM,
        "gate_passed": all(s is not None and s >= GATE_PPM for s in kinds.values()),
    }


def run(inputs_path: Path, raw_dir: Path, root: Path, out: Path, report_path: Path | None) -> int:
    inputs = json.loads(inputs_path.read_text())
    fmt = inputs["format_id"]
    fixtures = root / "fixtures" / "showdown" / fmt
    vocabulary = json.loads((fixtures / "team-vocabulary.json").read_text())
    engine_data = json.loads((fixtures / "engine-data.json").read_text())
    if sha256_hex(jcs(engine_data)) != vocabulary["engine_data_hash"]:
        raise DerivationError("the engine-data table is not the one the vocabulary names")
    files = verified_files(inputs, raw_dir)
    chaos = parse_source(files["chaos"].decode())
    if summary_battles(files["summary"].decode()) != chaos["info"]["number of battles"]:
        raise DerivationError("the summary and the chaos file state different battle counts")
    table, report = derive(chaos, inputs, vocabulary, engine_data)
    validate("usage-weight-table", table)
    read_usage_weight_table(table, vocabulary)
    report["usage_table_hash"] = sha256_hex(jcs(table))
    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["gate_passed"]:
        print("mapping gate failed: no table written", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(jcs(table) + b"\n")
    print(json.dumps(report, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        return run(args.inputs, args.raw_dir, args.root, args.out, args.report)
    except DerivationError as exc:
        print(f"derive_usage_weights: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
