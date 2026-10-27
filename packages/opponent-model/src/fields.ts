// The ability, item and spread marginals of one forme with the residual expanded (ADR-025 section
// 3, review finding N2): a list's residual mass is spread equally over every option the vocabulary
// allows for that field and the list does not name, so the support is everything the vocabulary
// allows and a revealed fact outside the truncated list never empties the belief. Unmapped mass is
// dropped (the weights are relative within the field). Integers only: a list's weights are scaled
// by the number of residual options n, so a named weight w becomes w * n and each residual option
// has weight `residual` (the share residual / (denominator * n) of the record, times n).

export interface Option<T> {
  readonly id: T;
  readonly weight: bigint;
}

/** Relative integer weights over a field's options (positive weights only, sorted). */
export interface Dist<T> {
  readonly total: bigint;
  readonly options: readonly Option<T>[];
}

interface Named {
  readonly id: string;
  readonly weight: number;
}

const byId = <T extends string | null>(a: Option<T>, b: Option<T>): number => (a.id === b.id ? 0 : a.id === null ? -1 : b.id === null ? 1 : a.id < b.id ? -1 : 1);

/** Build a distribution from named weights and a residual spread over `targets`. */
function expand<T extends string | null>(named: readonly Option<T>[], residual: number, targets: readonly T[]): Dist<T> {
  const scale = BigInt(Math.max(targets.length, 1));
  const options = [
    ...named.map((o) => ({ id: o.id, weight: o.weight * scale })),
    ...(residual > 0 ? targets.map((id) => ({ id, weight: BigInt(residual) })) : []),
  ].filter((o) => o.weight > 0n);
  options.sort(byId);
  return { total: options.reduce((s, o) => s + o.weight, 0n), options };
}

/** An ability list: named abilities, the residual over the rest of `pool` (the vocabulary's list for the forme). */
export function abilityDist(entries: readonly Named[], residual: number, pool: readonly string[]): Dist<string> {
  const named = new Set(entries.map((e) => e.id));
  return expand(entries.map((e) => ({ id: e.id, weight: BigInt(e.weight) })), residual, [...new Set(pool)].filter((id) => !named.has(id)));
}

/** The abilities of a forme that has no usage entry: uniform over the vocabulary's list. */
export function uniformDist(pool: readonly string[]): Dist<string> {
  const options = [...new Set(pool)].sort().map((id) => ({ id, weight: 1n }));
  return { total: BigInt(options.length), options };
}

/**
 * An item list. `none` (holds no item) is a named option of its own. The residual goes to every item of
 * the format that the list does not name and that is not in `leftOut` (stones that would make the Pokemon
 * another forme), so `nothing` never takes residual mass.
 */
export function itemDist(entries: readonly Named[], none: number, residual: number, items: readonly string[], leftOut: ReadonlySet<string>): Dist<string | null> {
  const named = new Set(entries.map((e) => e.id));
  const targets = [...new Set(items)].filter((id) => !named.has(id) && !leftOut.has(id));
  return expand<string | null>([...entries.map((e) => ({ id: e.id as string | null, weight: BigInt(e.weight) })), { id: null, weight: BigInt(none) }], residual, targets);
}

/** A single option with all the weight: a Mega forme's stone, or a fact the evidence fixes. */
export const only = <T>(id: T): Dist<T> => ({ total: 1n, options: [{ id, weight: 1n }] });

/** The weight of one option. */
export const weightOf = <T>(dist: Dist<T>, id: T): bigint => dist.options.find((o) => o.id === id)?.weight ?? 0n;

export interface SpreadInput {
  readonly nature: string;
  readonly stat_points: Readonly<Record<"hp" | "atk" | "def" | "spa" | "spd" | "spe", number>>;
  readonly weight: number;
}

/** A spread's key: its nature and the six Stat Points in stat order. */
export const spreadKey = (s: Pick<SpreadInput, "nature" | "stat_points">): string => {
  const p = s.stat_points;
  return `${s.nature}:${p.hp}-${p.atk}-${p.def}-${p.spa}-${p.spd}-${p.spe}`;
};

const STAT_ORDER = ["hp", "atk", "def", "spa", "spd", "spe"] as const;

/** The nature and Stat Points a spread key spells, or null for a key that is not one (the "other" entry). */
export function parseSpreadKey(key: string): { nature: string; stat_points: Record<(typeof STAT_ORDER)[number], number> } | null {
  const m = /^([a-z0-9]+):(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)$/.exec(key);
  if (m === null) return null;
  const points = STAT_ORDER.map((_, k) => Number(m[k + 2]));
  return { nature: m[1] as string, stat_points: Object.fromEntries(STAT_ORDER.map((stat, k) => [stat, points[k]])) as Record<(typeof STAT_ORDER)[number], number> };
}

/** The key of every spread the list does not name, taken as one entry (no evidence touches spreads in this slice). */
export const OTHER_SPREAD = "other";

export function spreadDist(entries: readonly SpreadInput[], residual: number): Dist<string> {
  const options = [...entries.map((s) => ({ id: spreadKey(s), weight: BigInt(s.weight) })), ...(residual > 0 ? [{ id: OTHER_SPREAD, weight: BigInt(residual) }] : [])];
  options.sort(byId);
  return { total: options.reduce((s, o) => s + o.weight, 0n), options };
}
