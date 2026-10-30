/* eslint-disable @typescript-eslint/no-explicit-any */
// The UsageWeightTable's share of the sample corpus (ADR-008 section 2, ADR-020 section 4): a small
// table built from the corpus's slim team vocabulary (the numbers are invented for the sample and
// derived from no source), schema-invalid mutations, and reader-invalid ones that pass the schema
// and that `readUsageWeightTable` refuses against that vocabulary.

import type { TeamVocabulary } from "./team-vocabulary.ts";
import { teamVocabularyHash } from "./team-vocabulary.ts";
import { rootedWeight } from "./usage-weight-table.ts";

type Cases = Record<string, unknown>;

function mutated(value: unknown, change: (copy: any) => void): Record<string, unknown> {
  const copy = structuredClone(value) as Record<string, unknown>;
  change(copy);
  return copy;
}

const hex = (digit: string): string => digit.repeat(64);
const SPECIES_ROOT = 2;
const TEAMMATE_ROOT = 2;

export function usageWeightTableSamples(vocabulary: TeamVocabulary): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const [nature0, nature1] = [vocabulary.natures[0]?.id as string, vocabulary.natures[1]?.id as string];
  const ids = vocabulary.roster.map((r) => r.id);
  const split = (total: number, n: number): number[] => Array.from({ length: n }, (_, k) => Math.floor(total / n) + (k === 0 ? total % n : 0));
  // Every list adds up to 1,000,000 ppm.
  const species = vocabulary.roster.map((r, k) => {
    const usage = k === 0 ? 950_000 - 450_000 * (ids.length - 1) : 450_000;
    const teammate = ids.find((id) => id !== r.id) as string;
    return {
      id: r.id,
      usage_ppm: usage,
      w_species: rootedWeight(usage, SPECIES_ROOT),
      items: { entries: vocabulary.items.map((id, j) => ({ id, ppm: split(700_000, vocabulary.items.length)[j] as number })), truncated_ppm: 100_000, unmapped_ppm: 50_000, no_item_ppm: 150_000 },
      abilities: { entries: r.abilities.map((id, j) => ({ id, ppm: split(900_000, r.abilities.length)[j] as number })), unmapped_ppm: 100_000 },
      moves: { entries: r.moves.slice(0, 3).map((id, j, kept) => ({ id, ppm: split(600_000, kept.length)[j] as number })), truncated_ppm: 150_000, unmapped_ppm: 50_000, empty_ppm: 200_000 },
      spreads: {
        entries: [
          { nature: nature0, stat_points: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 1 }, ppm: 500_000 },
          { nature: nature1, stat_points: { hp: 1, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, ppm: 300_000 },
        ],
        truncated_ppm: 150_000,
        unmapped_ppm: 50_000,
      },
      teammates: { entries: [{ id: teammate, ppm: 600_000, weight: rootedWeight(600_000, TEAMMATE_ROOT) }], truncated_ppm: 300_000, unmapped_ppm: 100_000 },
    };
  });
  const source = (id: string, digit: string, name: string) => ({
    source_id: id,
    url: `https://example.invalid/stats/2026-09/${name}`,
    retrieved_at: "2026-10-27T00:00:00Z",
    size: 1000,
    sha256: hex(digit),
    attribution: `Sample attribution for ${name}; invented for the corpus`,
    basis: "owner-accepted-risk",
    basis_reference: "ADR-009 erratum 1",
  });
  const base = {
    schema_version: 1,
    format_id: vocabulary.format_id,
    engine_data_hash: vocabulary.engine_data_hash,
    vocabulary_hash: teamVocabularyHash(vocabulary),
    month: "2026-09",
    cutoff: 1630,
    number_of_battles: 1_630_000,
    derivation: { script_version: "sample-1", k_items: 16, k_moves: 32, k_spreads: 32, k_teammates: 48, species_root: SPECIES_ROOT, teammate_root: TEAMMATE_ROOT },
    sources: [source("chaos", "1", "chaos.json"), source("summary", "2", "summary.txt")],
    unmapped_species_ppm: 50_000,
    species,
  };
  const withTable = (change: (t: any) => void) => mutated(base, change);
  const first = (t: any) => t.species[0];
  // Two teammates (ids in the order given) with equal shares; the masses still add up to 1,000,000.
  const both = (order: readonly string[]) => ({
    entries: order.slice(0, 2).map((id) => ({ id, ppm: 300_000, weight: rootedWeight(300_000, TEAMMATE_ROOT) })),
    truncated_ppm: 300_000,
    unmapped_ppm: 100_000,
  });
  const aaa = "aaaaaaaaa";

  return {
    valid: {
      "usage-weight-table/v1": base,
      "usage-weight-table/root-1": withTable((t) => {
        t.derivation.species_root = 1;
        t.derivation.teammate_root = 1;
        for (const s of t.species) {
          s.w_species = s.usage_ppm;
          for (const m of s.teammates.entries) m.weight = m.ppm;
        }
      }),
      "usage-weight-table/no-teammates": withTable((t) => (first(t).teammates = { entries: [], truncated_ppm: 0, unmapped_ppm: 0 })),
    },
    invalid: {
      "usage-weight-table/extra-field": withTable((t) => (t.note = "x")),
      "usage-weight-table/fractional-ppm": withTable((t) => (first(t).items.entries[0].ppm = 0.5)),
      "usage-weight-table/zero-ppm": withTable((t) => (first(t).items.entries[0].ppm = 0)),
      "usage-weight-table/ppm-over-total": withTable((t) => (first(t).items.entries[0].ppm = 1_000_001)),
      "usage-weight-table/missing-truncated": withTable((t) => delete first(t).moves.truncated_ppm),
      "usage-weight-table/missing-no-item": withTable((t) => delete first(t).items.no_item_ppm),
      "usage-weight-table/missing-empty": withTable((t) => delete first(t).moves.empty_ppm),
      "usage-weight-table/abilities-truncated": withTable((t) => (first(t).abilities.truncated_ppm = 0)),
      "usage-weight-table/missing-spread-point": withTable((t) => delete first(t).spreads.entries[0].stat_points.spe),
      "usage-weight-table/missing-teammate-weight": withTable((t) => delete first(t).teammates.entries[0].weight),
      "usage-weight-table/bad-month": withTable((t) => (t.month = "2026-13")),
      "usage-weight-table/no-sources": withTable((t) => (t.sources = [])),
      "usage-weight-table/other-basis": withTable((t) => (t.sources[0].basis = "granted")),
      "usage-weight-table/no-attribution": withTable((t) => delete t.sources[0].attribution),
      "usage-weight-table/no-species": withTable((t) => (t.species = [])),
      "usage-weight-table/root-zero": withTable((t) => (t.derivation.species_root = 0)),
      "usage-weight-table/evs-field": withTable((t) => (first(t).spreads.entries[0].evs = { hp: 1 })),
      "usage-weight-table/next-version": withTable((t) => (t.schema_version = 2)),
    },
    readerInvalid: {
      "usage-weight-table/unsorted-species": withTable((t) => t.species.reverse()),
      "usage-weight-table/unsorted-sources": withTable((t) => t.sources.reverse()),
      "usage-weight-table/species-sum": withTable((t) => (t.unmapped_species_ppm += 1)),
      "usage-weight-table/species-root": withTable((t) => (first(t).w_species += 1)),
      "usage-weight-table/unknown-species": withTable((t) => (first(t).id = aaa)),
      "usage-weight-table/duplicate-item": withTable((t) => (first(t).items.entries[1].id = first(t).items.entries[0].id)),
      "usage-weight-table/unsorted-moves": withTable((t) => first(t).moves.entries.reverse()),
      "usage-weight-table/unsorted-spreads": withTable((t) => first(t).spreads.entries.reverse()),
      "usage-weight-table/unsorted-teammates": withTable((t) => (first(t).teammates = both([...ids].reverse()))),
      "usage-weight-table/unknown-item": withTable((t) => (first(t).items.entries[0].id = aaa)),
      "usage-weight-table/unknown-move": withTable((t) => (first(t).moves.entries[0].id = aaa)),
      "usage-weight-table/unknown-ability": withTable((t) => (first(t).abilities.entries[0].id = aaa)),
      "usage-weight-table/unknown-nature": withTable((t) => (first(t).spreads.entries[0].nature = aaa)),
      "usage-weight-table/unknown-teammate": withTable((t) => (first(t).teammates.entries[0].id = aaa)),
      "usage-weight-table/abilities-sum": withTable((t) => (first(t).abilities.unmapped_ppm += 1)),
      "usage-weight-table/items-sum": withTable((t) => (first(t).items.no_item_ppm += 1)),
      "usage-weight-table/moves-sum": withTable((t) => (first(t).moves.empty_ppm += 1)),
      "usage-weight-table/spreads-sum": withTable((t) => (first(t).spreads.truncated_ppm -= 1)),
      "usage-weight-table/teammates-sum": withTable((t) => (first(t).teammates.unmapped_ppm += 1)),
      "usage-weight-table/teammate-root": withTable((t) => (first(t).teammates.entries[0].weight += 1)),
      "usage-weight-table/spread-over-limit": withTable((t) => (first(t).spreads.entries[0].stat_points = { hp: 255, atk: 255, def: 255, spa: 255, spd: 255, spe: 255 })),
      "usage-weight-table/too-many-items": withTable((t) => (t.derivation.k_items = 1)),
      "usage-weight-table/too-many-moves": withTable((t) => (t.derivation.k_moves = 1)),
      "usage-weight-table/too-many-spreads": withTable((t) => (t.derivation.k_spreads = 1)),
      "usage-weight-table/too-many-teammates": withTable((t) => {
        t.derivation.k_teammates = 1;
        first(t).teammates = both(ids);
      }),
      "usage-weight-table/stale-vocabulary-hash": withTable((t) => (t.vocabulary_hash = hex("f"))),
      "usage-weight-table/other-format": withTable((t) => (t.format_id = "gen9ou")),
      "usage-weight-table/other-engine-data": withTable((t) => (t.engine_data_hash = hex("e"))),
    },
  };
}
