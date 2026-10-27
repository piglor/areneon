// The `role-sets` usage prior (ADR-025 section 2, the committed licence-clean fallback; ADR-026 section 3's
// secondary arm): the committed `RoleSetTable` (ADR-020 section 3, MIT) read as a `UsagePrior`, with the
// item, nature and Stat Points of a set taken from a named fill prior, the `generator-matched` record of
// slice 3 (`fill_prior_hash`). One implementation: the belief and the sampler read it like any other prior.
//
//   forme    a table entry (a roster entry) weighs its number of sets, so a preview species is uniform over
//            ALL its roles (the roles of a Mega forme count for the species), then over a role's moves and
//            abilities;
//   moves    a role is a move pool, not a set: a set holds `min(slots, pool)` distinct moves of its role,
//            drawn uniformly (the conditional Poisson design with equal weights, as in the generator's
//            prior), so a move of a role has inclusion k / n in that role; the forme's inclusion is the
//            mean over its roles and the empty slots are counted. A move outside every role of the forme
//            has no mass and no residual: the belief gives it probability zero (the known mismatch with the
//            uniform-team generator, whose sets draw from the whole roster pool);
//   ability  uniform over a role's abilities, the roles equal. The table names a Mega forme's PRE-Evolution
//            abilities (the form Team Preview shows). The belief reads a stone holder's pre-Evolution
//            ability from the BASE forme's list (ADR-025 B1) and its own list only after the Evolution, so
//            the base forme's list is the mixture over every role of the preview species, the roles of its
//            stone-holding formes included, and a Mega forme's own list is the roster's Mega abilities
//            (the post-Evolution ability, uniform: the table says nothing about it). A species whose table
//            entries are all stone holders has no base forme here, and the belief then reads the
//            pre-Evolution ability uniformly over the vocabulary's list;
//   item, nature and Stat Points
//            the fill prior's, forme by forme: its item list scaled to this record's denominator (a Mega
//            forme holds its stone) and its "other spread" entry, which the belief's `SpreadSampler` port
//            fills with the generator's own routine. Nothing turns a role into an item or a spread.
//
// The builder is pure: the Mega-to-preview map is an input (`megaPreviewSpecies`).

import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { type RoleSetTable, roleSetTableHash } from "./role-sets.ts";
import { type TeamVocabulary, teamVocabularyHash } from "./team-vocabulary.ts";
import { readUsagePrior, usagePriorHash } from "./usage-prior.ts";

export class RoleSetsPriorError extends Error {
  override name = "RoleSetsPriorError";
}

function bad(message: string): never {
  throw new RoleSetsPriorError(message);
}

/** Bumped when the derivation changes how a number is computed. */
export const ROLE_SETS_PRIOR_SCRIPT = "role-sets-prior.v1";

const MAX = 1_000_000_000_000;

type Forme = UsagePrior["species"][number]["formes"][number];
type RoleSet = RoleSetTable["species"][number]["sets"][number];

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));
const lcm = (a: number, b: number): number => {
  const v = (a / gcd(a, b)) * b;
  return v > MAX ? bad("a denominator is too large for the contract") : v;
};
const cmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

export interface RoleSetsPriorInput {
  readonly vocabulary: TeamVocabulary;
  /** The role-set table read against `vocabulary` (`readRoleSetTable`). */
  readonly table: RoleSetTable;
  /** The `generator-matched` prior that supplies items, natures and Stat Points (read with `readUsagePrior`). */
  readonly fill: UsagePrior;
  /** `megaPreviewSpecies`: Mega roster entry to the preview species it shows as. */
  readonly shows_as: ReadonlyMap<string, string>;
}

/** The ability list of a mixture over roles: each role weighs 1 and is uniform over its abilities. */
function abilityMixture(roles: readonly (readonly string[])[]): Forme["abilities"] {
  const unit = roles.reduce((a, r) => lcm(a, r.length), 1);
  const weight = new Map<string, number>();
  for (const role of roles) for (const id of role) weight.set(id, (weight.get(id) ?? 0) + unit / role.length);
  const g = [...weight.values()].reduce(gcd, unit * roles.length);
  return {
    denominator: (unit * roles.length) / g,
    entries: [...weight.keys()].sort(cmp).map((id) => ({ id, weight: (weight.get(id) as number) / g })),
    residual: 0,
    unmapped: 0,
  };
}

function forme(id: string, roles: readonly RoleSet[], abilities: Forme["abilities"], fill: Forme): Forme {
  const w = roles.length;
  const slots = fill.moves.slots;
  const k = roles.map((r) => Math.min(slots, r.moves.length));

  // The fill's item list in lowest terms, then scaled until every move inclusion is an integer.
  const il = fill.items;
  const g = [il.denominator, il.no_item, il.residual, il.unmapped, ...il.entries.map((e) => e.weight)].reduce(gcd, 0);
  const reduced = il.denominator / g;
  let denominator = lcm(reduced, w);
  roles.forEach((r, i) => {
    const unit = w * r.moves.length;
    denominator = lcm(denominator, unit / gcd(unit, k[i] as number));
  });
  const factor = denominator / reduced;

  const inclusion = new Map<string, number>();
  let empty = 0;
  roles.forEach((r, i) => {
    const each = (denominator * (k[i] as number)) / (w * r.moves.length);
    for (const m of r.moves) inclusion.set(m, (inclusion.get(m) ?? 0) + each);
    empty += (denominator * (slots - (k[i] as number))) / w;
  });
  return {
    id,
    weight: w,
    abilities,
    items: {
      denominator,
      entries: il.entries.map((e) => ({ id: e.id, weight: (e.weight / g) * factor })),
      no_item: (il.no_item / g) * factor,
      residual: (il.residual / g) * factor,
      unmapped: (il.unmapped / g) * factor,
    },
    spreads: fill.spreads,
    moves: { denominator, slots, entries: [...inclusion.keys()].sort(cmp).map((m) => ({ id: m, inclusion: inclusion.get(m) as number })), empty_slots: empty, residual: 0, unmapped: 0 },
  };
}

/** Derive the `role-sets` prior of `table` with the fills of `fill`. */
export function deriveRoleSetsPrior(input: RoleSetsPriorInput): UsagePrior {
  const { vocabulary, table, fill, shows_as } = input;
  const vocabularyHash = teamVocabularyHash(vocabulary);
  if (table.vocabulary_hash !== vocabularyHash || table.engine_data_hash !== vocabulary.engine_data_hash || table.format_id !== vocabulary.format_id) bad("the role-set table was not exported for this vocabulary");
  if (fill.source.kind !== "generator-matched") bad("the fill prior is not a generator-matched prior");
  if (fill.vocabulary_hash !== vocabularyHash || fill.engine_data_hash !== vocabulary.engine_data_hash || fill.format_id !== vocabulary.format_id) bad("the fill prior was not derived for this vocabulary");
  const roster = new Map(vocabulary.roster.map((r) => [r.id, r]));
  const fillFormes = new Map(fill.species.flatMap((s) => s.formes.map((f) => [f.id, f] as const)));

  const groups = new Map<string, { id: string; sets: readonly RoleSet[] }[]>();
  for (const entry of table.species) {
    const shown = shows_as.get(entry.id) ?? entry.id;
    if (!roster.has(shown)) bad(`table entry ${entry.id} shows as ${shown}, which is not in the roster`);
    if (shows_as.has(shown)) bad(`table entry ${entry.id} shows as ${shown}, which itself shows as another species`);
    groups.set(shown, [...(groups.get(shown) ?? []), { id: entry.id, sets: entry.sets }]);
  }
  const species = [...groups.keys()].sort(cmp).map((id) => {
    const entries = (groups.get(id) as { id: string; sets: readonly RoleSet[] }[]).sort((a, b) => cmp(a.id, b.id));
    // The roles a base forme's ability list is mixed over: its own, and those of the stone holders that show as it.
    const holders = entries.filter((e) => shows_as.has(e.id)).flatMap((e) => e.sets);
    const formes = entries.map((e) => {
      const holder = shows_as.has(e.id);
      const pool = [...new Set((roster.get(e.id) as TeamVocabulary["roster"][number]).abilities)].sort(cmp);
      const abilities = holder ? { denominator: pool.length, entries: pool.map((a) => ({ id: a, weight: 1 })), residual: 0, unmapped: 0 } : abilityMixture((e.id === id ? [...e.sets, ...holders] : e.sets).map((s) => s.abilities));
      return forme(e.id, e.sets, abilities, fillFormes.get(e.id) ?? bad(`the fill prior has no forme ${e.id}`));
    });
    return { id, denominator: formes.reduce((t, f) => t + f.weight, 0), formes: formes as [Forme, ...Forme[]] };
  });
  const all = species.flatMap((s) => s.formes);
  const prior: UsagePrior = {
    schema_version: 1,
    format_id: vocabulary.format_id,
    engine_data_hash: vocabulary.engine_data_hash,
    vocabulary_hash: vocabularyHash,
    source: { kind: "role-sets", role_set_table_hash: roleSetTableHash(table), fill_prior_hash: usagePriorHash(fill) },
    derivation: {
      script_version: ROLE_SETS_PRIOR_SCRIPT,
      k_items: Math.max(1, ...all.map((f) => f.items.entries.length)),
      k_moves: Math.max(1, ...all.map((f) => f.moves.entries.length)),
      k_spreads: Math.max(1, fill.derivation.k_spreads),
    },
    species: species as UsagePrior["species"],
  };
  return readUsagePrior(prior, vocabulary);
}
