// Hand-made test data for the belief filter: a six-species vocabulary and a prior small enough to
// work out on paper. Species `alpha` (items, abilities and moves with a residual), `gamma` (items),
// `beta` (a Mega forme `betamega` with the stone `betite`), and three inert species that always
// hold no item and carry their whole move pool.

import { interpretChannel } from "@areneon/public-state";
import type { PublicDex, PublicFlag, PublicSide } from "@areneon/public-state";
import type { BeliefConfig } from "@areneon/schema/types/v1/belief-config.ts";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import type { BeliefInput, BeliefVocabulary } from "../src/index.ts";

type Forme = UsagePrior["species"][number]["formes"][number];
type Rules = BeliefVocabulary["rules"];

export const HEX = "0".repeat(64);
export const MOVES = ["ma", "mb", "mc", "md"] as const;

export function vocabulary(rules: Partial<Rules> = {}): BeliefVocabulary {
  const four = (p: string): [string, ...string[]] => [`${p}1`, `${p}2`, `${p}3`, `${p}4`];
  return {
    format_id: "testformat",
    engine_data_hash: HEX,
    rules: { adjust_level: 50, min_team_size: 6, max_team_size: 6, picked_team_size: 4, item_clause: 1, ...rules },
    items: ["betite", "ix", "iy"],
    roster: [
      { id: "alpha", abilities: ["a1", "a2"], moves: [...MOVES] },
      { id: "beta", abilities: ["b1", "b2"], moves: four("bm") },
      { id: "betamega", abilities: ["bmega"], moves: four("bm"), required_item: "betite" },
      { id: "gamma", abilities: ["g1"], moves: four("gm") },
      { id: "delta", abilities: ["d1"], moves: four("dm") },
      { id: "epsilon", abilities: ["ill"], moves: four("em") },
      { id: "zeta", abilities: ["z1"], moves: four("zm") },
    ],
  };
}

export const config: BeliefConfig = { schema_version: 1, identity_hiding_abilities: [], max_legality_redraws: 32, max_disguise_assignments: 64 };

/** The config of a game in which `epsilon` can hide its identity. */
export const hidingConfig: BeliefConfig = { ...config, identity_hiding_abilities: ["ill"] };

const spreads = (d: number): Forme["spreads"] => ({
  denominator: d,
  entries: [{ nature: "hardy", stat_points: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, weight: d }],
  residual: 0,
  unmapped: 0,
});

/** A forme that always holds no item and carries every move of its pool of four. */
export function inert(id: string, ability: string, movePrefix: string, weight = 1): Forme {
  return {
    id,
    weight,
    abilities: { denominator: 1, entries: [{ id: ability, weight: 1 }], residual: 0, unmapped: 0 },
    items: { denominator: 1, entries: [], no_item: 1, residual: 0, unmapped: 0 },
    spreads: spreads(1),
    moves: { denominator: 1, slots: 4, entries: [1, 2, 3, 4].map((k) => ({ id: `${movePrefix}${k}`, inclusion: 1 })), empty_slots: 0, residual: 0, unmapped: 0 },
  };
}

export const alpha = (change: (f: Forme) => void = () => undefined): Forme => {
  const f: Forme = {
    id: "alpha",
    weight: 1,
    abilities: { denominator: 4, entries: [{ id: "a1", weight: 3 }], residual: 1, unmapped: 0 },
    items: { denominator: 10, entries: [{ id: "ix", weight: 5 }, { id: "iy", weight: 2 }], no_item: 3, residual: 0, unmapped: 0 },
    spreads: spreads(10),
    // 10 sets: ma always, mb and mc half the time, md (outside the list) half the time through the residual, 1.5 empty slots on average.
    moves: { denominator: 10, slots: 4, entries: [{ id: "ma", inclusion: 10 }, { id: "mb", inclusion: 5 }, { id: "mc", inclusion: 5 }], empty_slots: 15, residual: 5, unmapped: 0 },
  };
  change(f);
  return f;
};

export const gamma = (change: (f: Forme) => void = () => undefined): Forme => {
  const f: Forme = {
    ...inert("gamma", "g1", "gm"),
    items: { denominator: 4, entries: [{ id: "ix", weight: 1 }, { id: "iy", weight: 2 }], no_item: 1, residual: 0, unmapped: 0 },
    moves: { denominator: 4, slots: 4, entries: [1, 2, 3, 4].map((k) => ({ id: `gm${k}`, inclusion: 4 })), empty_slots: 0, residual: 0, unmapped: 0 },
  };
  change(f);
  return f;
};

/** The base forme of `beta` and its Mega forme: stone-holding share 1/4, base abilities b1:3 b2:1, Mega ability bmega. */
export const betaFormes = (): [Forme, Forme] => [
  { ...inert("beta", "b1", "bm", 3), abilities: { denominator: 4, entries: [{ id: "b1", weight: 3 }, { id: "b2", weight: 1 }], residual: 0, unmapped: 0 } },
  inert("betamega", "bmega", "bm", 1),
];

export function prior(change: (p: UsagePrior) => void = () => undefined): UsagePrior {
  const species = (id: string, formes: Forme[]) => ({ id, denominator: formes.reduce((s, f) => s + f.weight, 0), formes: formes as [Forme, ...Forme[]] });
  const p: UsagePrior = {
    schema_version: 1,
    format_id: "testformat",
    engine_data_hash: HEX,
    vocabulary_hash: HEX,
    source: { kind: "generator-matched", generator_config_hash: HEX, generator_commit: "a".repeat(40) },
    derivation: { script_version: "test", k_items: 16, k_moves: 24, k_spreads: 32 },
    species: [
      species("alpha", [alpha()]),
      species("beta", betaFormes()),
      species("delta", [inert("delta", "d1", "dm")]),
      species("epsilon", [inert("epsilon", "ill", "em")]),
      species("gamma", [gamma()]),
      species("zeta", [inert("zeta", "z1", "zm")]),
    ],
  };
  change(p);
  return p;
}

export const NAMES: Record<string, string> = { alpha: "Alpha", beta: "Beta", gamma: "Gamma", delta: "Delta", epsilon: "Epsilon", zeta: "Zeta" };
export const PREVIEW = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"] as const;

/** Lines of the player p1's own channel: the opponent's six preview Pokemon, then `more`. */
export function channel(more: readonly string[] = [], order: readonly string[] = PREVIEW, hidden: readonly string[] = []): string[] {
  return ["|gametype|doubles", "|player|p1|Alice|", "|player|p2|Bob|", "|clearpoke", ...order.map((s) => `|poke|p2|${NAMES[s]}${hidden.includes(s) ? "-*" : ""}, L50, M|`), ...more];
}

export interface Seen {
  readonly opponent: PublicSide;
  readonly flags: PublicFlag[];
}

/** What the shared reader makes of the lines, as the filter's inputs. */
export function seen(more: readonly string[] = [], dex?: PublicDex, hidden: readonly string[] = []): Seen {
  const run = interpretChannel("p1", channel(more, PREVIEW, hidden), dex);
  const last = run.frames[run.frames.length - 1];
  if (last === undefined) throw new Error("empty channel");
  return { opponent: last.state.opponent, flags: run.flags };
}

export function input(more: readonly string[] = [], change: Partial<BeliefInput> = {}): BeliefInput {
  const s = seen(more);
  return { prior: prior(), vocabulary: vocabulary(), config, opponent: s.opponent, flags: s.flags, ...change };
}

/** A BigInt-safe JSON text of a belief, for equality checks. */
export const text = (value: unknown): string => JSON.stringify(value, (_, v: unknown) => (typeof v === "bigint" ? `${v}n` : v));
