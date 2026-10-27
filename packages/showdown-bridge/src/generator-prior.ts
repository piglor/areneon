// The `generator-matched` usage prior (ADR-025 section 10, slice 3): the distribution the synthetic
// opponents are actually drawn from, derived from a `TeamGeneratorConfig` and the team vocabulary by
// the proposal's own rules (ADR-013 section 2; the code is `packages/team-dataset/src/proposal.ts`),
// with every number an exact integer under the contract's denominators.
//
//   forme    a team member is a roster entry drawn uniformly, so within a preview species each roster
//            entry that shows as it (the species itself, and each Mega forme whose stone changes the
//            species into it) weighs 1;
//   ability  uniform over the roster entry's abilities;
//   item     the config's `item_weights` over the vocabulary's items (`default_item_weight` for an item
//            the list does not name), a zero weight excluded; a forme that requires an item (a Mega
//            forme) holds it. The Item Clause is not in the marginal: it couples the slots, and the
//            belief's exact subset DP applies it (with equal item weights, as in the committed config,
//            the DP's joint equals the proposal's sequential draw; with unequal weights it is an
//            approximation, stated here);
//   moves    `min(moves_per_set, pool)` distinct moves without replacement, which is the conditional
//            Poisson design with equal weights: every move of the pool has inclusion k / n, and a set
//            with fewer moves than slots counts its empty slots;
//   spread   one "other spread" entry holding all the mass: the nature and Stat Points are drawn by the
//            generator's own routine through the `SpreadSampler` port (packages/team-dataset).
//
// Not modelled, and measured by the slice's test instead: the stratum quotas and the validator's
// rejections change the real team distribution a little. The builder is pure: the Mega-to-preview map
// is an input (`megaPreviewSpecies` derives it from the pinned engine's stone data).

import { createHash } from "node:crypto";
import type { TeamGeneratorConfig } from "@areneon/schema/types/v1/team-generator-config.ts";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { type Json, jcs } from "./canonical-json.ts";
import type { EngineData } from "./engine-data.ts";
import type { Engine } from "./engine.ts";
import { type TeamVocabulary, teamVocabularyHash } from "./team-vocabulary.ts";

export class GeneratorPriorError extends Error {
  override name = "GeneratorPriorError";
}

function bad(message: string): never {
  throw new GeneratorPriorError(message);
}

/** Bumped when the derivation changes how a number is computed. */
export const GENERATOR_PRIOR_SCRIPT = "generator-prior.v1";

const MAX = 1_000_000_000_000;

type Forme = UsagePrior["species"][number]["formes"][number];
type RosterEntry = TeamVocabulary["roster"][number];

/** SHA-256 of the JCS bytes of a generator config: the prior's `generator_config_hash` (and the dataset manifests' `config_hash`). */
export const generatorConfigHash = (config: TeamGeneratorConfig): string => createHash("sha256").update(jcs(config as unknown as Json), "utf8").digest("hex");

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));
const cmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/**
 * For each roster entry that requires an item and shows at Team Preview as another roster entry (a Mega
 * forme before it evolves), the id of the species it shows as: the forme the entry's stone changes into
 * it, from the pinned engine's item data (`megaStone`: pre-Mega forme name to Mega forme name). An entry
 * the stone map does not name (a forme that is its own preview species) is absent from the map. Refuses a
 * stone that names two pre-Mega formes of the roster, and a pre-Mega forme outside the roster.
 */
export function megaPreviewSpecies(engine: Engine, vocabulary: TeamVocabulary, table: EngineData): ReadonlyMap<string, string> {
  const dex = (engine.sim as unknown as { Dex: { mod(name: string): { items: { get(id: string): { megaStone?: unknown } } } } }).Dex.mod(vocabulary.mod);
  const names = new Map(table.species.map((s) => [s.id, s.name]));
  const ids = new Map(table.species.map((s) => [s.name, s.id]));
  const roster = new Set(vocabulary.roster.map((r) => r.id));
  const shows = new Map<string, string>();
  for (const entry of vocabulary.roster) {
    if (entry.required_item === undefined) continue;
    const stone = dex.items.get(entry.required_item).megaStone;
    if (typeof stone !== "object" || stone === null || Array.isArray(stone)) continue;
    const mega = names.get(entry.id);
    const from = Object.entries(stone as Record<string, unknown>)
      .filter(([, to]) => to === mega)
      .map(([name]) => ids.get(name));
    if (from.length === 0) continue;
    if (from.length > 1 || from[0] === undefined || !roster.has(from[0])) bad(`roster entry ${entry.id}: its stone does not name exactly one pre-Mega forme of the roster`);
    shows.set(entry.id, from[0]);
  }
  return shows;
}

export interface GeneratorPriorInput {
  readonly vocabulary: TeamVocabulary;
  readonly config: TeamGeneratorConfig;
  /** The commit of the generator code whose draw rules this reproduces (40 lower-case hex digits). */
  readonly generator_commit: string;
  /** `megaPreviewSpecies`: Mega roster entry to the preview species it shows as. */
  readonly shows_as: ReadonlyMap<string, string>;
}

function forme(entry: RosterEntry, weights: ReadonlyMap<string, number>, slots: number): Forme {
  const abilities = [...new Set(entry.abilities)].sort(cmp);
  const pool = [...new Set(entry.moves)].sort(cmp);
  const n = pool.length;
  const k = Math.min(slots, n);

  // Items: the stone a forme requires, else the positive weights; scaled so the move inclusions are integers.
  const named: { id: string; weight: number }[] =
    entry.required_item !== undefined
      ? [{ id: entry.required_item, weight: 1 }]
      : [...weights.keys()].sort(cmp).flatMap((id) => ((weights.get(id) as number) > 0 ? [{ id, weight: weights.get(id) as number }] : []));
  const sum = named.reduce((t, e) => t + e.weight, 0);
  const base = sum === 0 ? 1 : sum;
  const scale = n / gcd(base, n);
  const denominator = base * scale;
  if (denominator > MAX) bad(`${entry.id}: the item weights are too large for the contract's denominators`);
  return {
    id: entry.id,
    weight: 1,
    abilities: { denominator: abilities.length, entries: abilities.map((id) => ({ id, weight: 1 })), residual: 0, unmapped: 0 },
    items: { denominator, entries: named.map((e) => ({ id: e.id, weight: e.weight * scale })), no_item: sum === 0 ? scale : 0, residual: 0, unmapped: 0 },
    spreads: { denominator: 1, entries: [], residual: 1, unmapped: 0 },
    moves: { denominator, slots, entries: pool.map((id) => ({ id, inclusion: (denominator / n) * k })), empty_slots: (slots - k) * denominator, residual: 0, unmapped: 0 },
  };
}

/** Derive the `generator-matched` prior of `config` on `vocabulary`. */
export function deriveGeneratorPrior(input: GeneratorPriorInput): UsagePrior {
  const { vocabulary, config, generator_commit, shows_as } = input;
  if (!/^[0-9a-f]{40}$/.test(generator_commit)) bad("generator_commit is not 40 lower-case hex digits");
  const slots = config.moves_per_set ?? bad("the generator config has no moves_per_set");
  const roster = new Map(vocabulary.roster.map((r) => [r.id, r]));
  const items = new Set(vocabulary.items);
  const weights = new Map<string, number>();
  const override = new Map(config.item_weights.map((w) => [w.item, w.weight]));
  for (const id of override.keys()) if (!items.has(id)) bad(`the config weighs item ${id}, which the vocabulary does not list`);
  for (const id of vocabulary.items) weights.set(id, override.get(id) ?? config.default_item_weight);

  const groups = new Map<string, RosterEntry[]>();
  for (const entry of vocabulary.roster) {
    const shown = shows_as.get(entry.id) ?? entry.id;
    if (!roster.has(shown)) bad(`roster entry ${entry.id} shows as ${shown}, which is not in the roster`);
    if (shows_as.has(shown)) bad(`roster entry ${entry.id} shows as ${shown}, which itself shows as another species`);
    groups.set(shown, [...(groups.get(shown) ?? []), entry]);
  }
  const species = [...groups.keys()].sort(cmp).map((id) => {
    const formes = (groups.get(id) as RosterEntry[]).sort((a, b) => cmp(a.id, b.id)).map((e) => forme(e, weights, slots));
    return { id, denominator: formes.length, formes: formes as [Forme, ...Forme[]] };
  });
  const all = species.flatMap((s) => s.formes);
  return {
    schema_version: 1,
    format_id: vocabulary.format_id,
    engine_data_hash: vocabulary.engine_data_hash,
    vocabulary_hash: teamVocabularyHash(vocabulary),
    source: { kind: "generator-matched", generator_config_hash: generatorConfigHash(config), generator_commit },
    derivation: {
      script_version: GENERATOR_PRIOR_SCRIPT,
      k_items: Math.max(1, ...all.map((f) => f.items.entries.length)),
      k_moves: Math.max(1, ...all.map((f) => f.moves.entries.length)),
      k_spreads: 1,
    },
    species: species as UsagePrior["species"],
  };
}
