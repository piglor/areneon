/* eslint-disable @typescript-eslint/no-explicit-any */
// The UsagePrior's and BeliefConfig's share of the sample corpus (ADR-008 section 2, ADR-025
// sections 2 and 5): small priors built from the corpus's slim team vocabulary (the numbers are
// invented for the sample and derived from no source), schema-invalid mutations, and reader-invalid
// ones that pass the schema and that `readUsagePrior` / `readBeliefConfig` refuse against that
// vocabulary.

import type { TeamVocabulary } from "./team-vocabulary.ts";
import { teamVocabularyHash } from "./team-vocabulary.ts";

type Cases = Record<string, unknown>;

function mutated(value: unknown, change: (copy: any) => void): Record<string, unknown> {
  const copy = structuredClone(value) as Record<string, unknown>;
  change(copy);
  return copy;
}

const hex = (digit: string): string => digit.repeat(64);

export function usagePriorSamples(vocabulary: TeamVocabulary): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const [nature0, nature1] = [vocabulary.natures[0]?.id as string, vocabulary.natures[1]?.id as string];
  const itemEntries = vocabulary.items.map((id, k) => ({ id, weight: k === 0 ? 400 : 200 }));
  const forme = (id: string, abilities: readonly string[], moves: readonly string[]) => ({
    id,
    weight: 100,
    abilities: { denominator: 300 * abilities.length + 150, entries: abilities.map((a) => ({ id: a, weight: 300 })), residual: 100, unmapped: 50 },
    items: { denominator: 900, entries: itemEntries, no_item: 150, residual: 100, unmapped: 50 },
    spreads: {
      denominator: 1000,
      entries: [
        { nature: nature0, stat_points: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 1 }, weight: 500 },
        { nature: nature1, stat_points: { hp: 1, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, weight: 300 },
      ],
      residual: 150,
      unmapped: 50,
    },
    // 900 sets of 4 slots: the inclusions, the empty slots (two per set on average), the residual and the unmapped mass fill 3600.
    moves: {
      denominator: 900,
      slots: 4,
      entries: moves.map((m, k) => ({ id: m, inclusion: k === 0 ? 900 : 450 })),
      empty_slots: 1800,
      residual: 3600 - 900 - 450 * (moves.length - 1) - 1800 - 150,
      unmapped: 150,
    },
  });
  const species = vocabulary.roster.map((r) => ({ id: r.id, denominator: 100, formes: [forme(r.id, r.abilities, r.moves)] }));
  const base = {
    schema_version: 1,
    format_id: vocabulary.format_id,
    engine_data_hash: vocabulary.engine_data_hash,
    vocabulary_hash: teamVocabularyHash(vocabulary),
    source: {
      kind: "smogon-chaos",
      month: "2026-09",
      cutoff: 1630,
      number_of_battles: 1630000,
      inputs: [{ source_id: "chaos-2026-09-1630", url: "https://example.invalid/stats/2026-09/chaos/format-1630.json", retrieved_at: "2026-10-27T00:00:00Z", size: 1000, sha256: hex("1") }],
    },
    derivation: { script_version: "sample-1", k_items: 16, k_moves: 24, k_spreads: 32 },
    species,
  };
  const withPrior = (change: (p: any) => void) => mutated(base, change);
  const first = (p: any) => p.species[0].formes[0];
  const config = { schema_version: 1, identity_hiding_abilities: [vocabulary.roster[0]?.abilities[0] as string], max_legality_redraws: 32, max_disguise_assignments: 64 };
  const configWith = (change: (c: any) => void) => mutated(config, change);
  const second = { source_id: "chaos-2026-09-extra", url: "https://example.invalid/stats/2026-09/extra.json", retrieved_at: "2026-10-27T00:00:00Z", size: 1, sha256: hex("2") };

  return {
    valid: {
      "usage-prior/smogon-chaos": base,
      "usage-prior/role-sets": withPrior((p) => (p.source = { kind: "role-sets", role_set_table_hash: hex("3"), fill_prior_hash: hex("4") })),
      "usage-prior/generator-matched": withPrior((p) => (p.source = { kind: "generator-matched", generator_config_hash: hex("5"), generator_commit: "a".repeat(40) })),
      "belief-config/v1": config,
    },
    invalid: {
      "usage-prior/extra-field": withPrior((p) => (p.note = "x")),
      "usage-prior/fractional-weight": withPrior((p) => (first(p).items.entries[0].weight = 0.5)),
      "usage-prior/negative-residual": withPrior((p) => (first(p).items.residual = -1)),
      "usage-prior/zero-denominator": withPrior((p) => (first(p).abilities.denominator = 0)),
      "usage-prior/zero-weight": withPrior((p) => (first(p).items.entries[0].weight = 0)),
      "usage-prior/missing-unmapped": withPrior((p) => delete first(p).moves.unmapped),
      "usage-prior/missing-no-item": withPrior((p) => delete first(p).items.no_item),
      "usage-prior/ppm-share": withPrior((p) => (first(p).items.entries[0].share_ppm = 400000)),
      "usage-prior/unknown-source-kind": withPrior((p) => (p.source = { kind: "limitless" })),
      "usage-prior/chaos-no-inputs": withPrior((p) => (p.source.inputs = [])),
      "usage-prior/bad-month": withPrior((p) => (p.source.month = "2026-13")),
      "usage-prior/missing-spread-point": withPrior((p) => delete first(p).spreads.entries[0].stat_points.spe),
      "usage-prior/no-species": withPrior((p) => (p.species = [])),
      "usage-prior/no-formes": withPrior((p) => (p.species[0].formes = [])),
      "usage-prior/next-version": withPrior((p) => (p.schema_version = 2)),
      "belief-config/extra-field": configWith((c) => (c.note = "x")),
      "belief-config/fractional-bound": configWith((c) => (c.max_legality_redraws = 0.5)),
      "belief-config/zero-assignments": configWith((c) => (c.max_disguise_assignments = 0)),
      "belief-config/missing-abilities": configWith((c) => delete c.identity_hiding_abilities),
    },
    readerInvalid: {
      "usage-prior/unsorted-species": withPrior((p) => p.species.reverse()),
      "usage-prior/duplicate-item": withPrior((p) => (first(p).items.entries[1].id = first(p).items.entries[0].id)),
      "usage-prior/unsorted-moves": withPrior((p) => first(p).moves.entries.reverse()),
      "usage-prior/unsorted-spreads": withPrior((p) => first(p).spreads.entries.reverse()),
      "usage-prior/unsorted-inputs": withPrior((p) => (p.source.inputs = [second, ...p.source.inputs])),
      "usage-prior/unknown-species": withPrior((p) => (p.species[0].id = "aaaaaaaaa")),
      "usage-prior/unknown-forme": withPrior((p) => (first(p).id = "aaaaaaaaa")),
      "usage-prior/unknown-item": withPrior((p) => (first(p).items.entries[0].id = "aaaaaaaaa")),
      "usage-prior/unknown-move": withPrior((p) => (first(p).moves.entries[0].id = "aaaaaaaaa")),
      "usage-prior/unknown-ability": withPrior((p) => (first(p).abilities.entries[0].id = "aaaaaaaaa")),
      "usage-prior/unknown-nature": withPrior((p) => (first(p).spreads.entries[0].nature = "aaaaaaaaa")),
      "usage-prior/forme-weights-sum": withPrior((p) => (p.species[0].denominator = 101)),
      "usage-prior/abilities-sum": withPrior((p) => (first(p).abilities.unmapped += 1)),
      "usage-prior/items-sum": withPrior((p) => (first(p).items.denominator += 1)),
      "usage-prior/spreads-sum": withPrior((p) => (first(p).spreads.residual -= 1)),
      "usage-prior/moves-sum": withPrior((p) => (first(p).moves.empty_slots += 1)),
      "usage-prior/inclusion-over-denominator": withPrior((p) => {
        first(p).moves.entries[0].inclusion = 901;
        first(p).moves.residual -= 1;
      }),
      "usage-prior/spread-over-limit": withPrior((p) => (first(p).spreads.entries[0].stat_points = { hp: 255, atk: 255, def: 255, spa: 255, spd: 255, spe: 255 })),
      "usage-prior/too-many-items": withPrior((p) => (p.derivation.k_items = 1)),
      "usage-prior/too-many-spreads": withPrior((p) => (p.derivation.k_spreads = 1)),
      "usage-prior/too-many-moves": withPrior((p) => (p.derivation.k_moves = 1)),
      "usage-prior/stale-vocabulary-hash": withPrior((p) => (p.vocabulary_hash = hex("f"))),
      "usage-prior/other-format": withPrior((p) => (p.format_id = "gen9ou")),
      "usage-prior/other-engine-data": withPrior((p) => (p.engine_data_hash = hex("e"))),
      "belief-config/unknown-ability": configWith((c) => (c.identity_hiding_abilities = ["aaaaaaaaa"])),
      "belief-config/unsorted-abilities": configWith((c) => (c.identity_hiding_abilities = [...vocabulary.roster.flatMap((r) => r.abilities)].sort().reverse().filter((a, k, all) => all.indexOf(a) === k))),
      "belief-config/duplicate-ability": configWith((c) => (c.identity_hiding_abilities = [c.identity_hiding_abilities[0], c.identity_hiding_abilities[0]])),
    },
  };
}
