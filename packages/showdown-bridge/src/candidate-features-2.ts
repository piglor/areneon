// The candidate feature builder, tier A, second feature set `probe-damage/2` (ADR-022 erratum E-22.3, #1172).
// It holds `probe-damage/1`'s `rows` byte for byte (the same code builds them) and four more arrays, all from the
// ADR-016 probe on ADR-018's v1 assumed opponent sets and the shared reader:
//
//   threats         what each opposing active's REVEALED damaging move would do to each own Pokemon (an own active, an
//                   own active Mega Evolved this turn, or a benched Pokemon a `switch N` candidate names);
//   speeds          who acts first between each such own Pokemon and each opposing active, as the engine orders their
//                   action speed (Speed stat, stages, Tailwind, Trick Room, paralysis, a held Scarf, a Mega's new Speed);
//   own_priorities  the engine's priority key of each own move part of the candidates (status moves included);
//   foe_priorities  the same for each revealed move of each opposing active.
//
// The foe's side of `threats`, `speeds` and `foe_priorities` is the ASSUMED set (revealed facts as revealed, the rest from
// the v1 record: Hardy, 11 Stat Points in every stat, no item, the first ability in the vocabulary's order): relations and
// values on assumed opponent sets, never facts about the real opponent. The own side is exact (own request and team).
// A benched Pokemon's rows are advisory next-turn quantities: the switch acts first and the foe chooses its target.
// The probe resolves every action against a fixed foe target, so a priority event that depends on the chosen target is not
// modelled (as in ADR-018). A row the probe cannot answer is `unsupported` with its closed reason (plus `unqueueable` and
// `fractional_priority` for the speed-order rows); nothing is ever filled with a guess.
//
// Pure and deterministic: the arrays are the same bytes with or without the memo, in any candidate order, and for any
// batching of the priority queries (a batch the probe does not answer is asked again one action at a time).
// Excluded by construction, as in `candidate-features.ts`: the input is a `PlayerChannel`, the player's own team and the
// candidate strings; no opponent team, hidden verdict, `BattleRecord` or seed is a parameter of anything here.
import type { PublicState } from "@areneon/public-state";
import { type CandidateFeatures, createFeatureMemo, damageRows, damageValues, type DamageValues, type FeatureConfigs, type FeatureMemo, type FeatureRow, featureSetHash, featureSetIdentity, lastRequest, openRequest, refusing } from "./candidate-features.ts";
import type { Candidate } from "./choices.ts";
import { parseJointChoice, type SlotPart } from "./damage-aware-choice.ts";
import type { SpeedOrderAnswer } from "./damage-aware-score.ts";
import { array, DamageAwareError, type DamageView, type Fighter, type Foe, foeTargetKind, object, type OwnSlot, type SpeedLimit, text } from "./damage-aware-view.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import type { MechanicsQuery } from "./mechanics-query.ts";
import type { MechanicsResult } from "./mechanics-result.ts";
import type { PlayerChannel } from "./observation.ts";
import { loadRulesSnapshot } from "./rules-load.ts";

/** The feature set this module builds. A change of what any array holds is a new id. */
export const FEATURE_SET_2 = "probe-damage/2";

type ProbeReason = Extract<MechanicsResult, { kind: "unsupported" }>["reason"];
/** Why a speed or priority row is `unsupported`: the probe's closed reasons, plus its two speed-order limits. */
export type OrderReason = ProbeReason | SpeedLimit;
export type OrderValue<V> = ({ kind: "computed" } & V) | { kind: "unsupported"; reason: OrderReason };

/** An opposing active's revealed damaging move into one own Pokemon (the foe's assumed set as attacker, the own Pokemon at its exact current HP). */
export type ThreatRow = { own: number; mega: boolean; foe: number; move: string } & DamageValues;
/** Exactly one row per (own position, own Mega flag, foe location). */
export type SpeedRow = { own: number; mega: boolean; foe: number } & OrderValue<{ relation: "own_first" | "tie" | "foe_first" }>;
export type OwnPriorityRow = { slot: number; move: number; mega: boolean } & OrderValue<{ priority: number }>;
export type FoePriorityRow = { foe: number; move: string } & OrderValue<{ priority: number }>;

export interface CandidateFeatures2 extends Omit<CandidateFeatures, "feature_set"> {
  feature_set: typeof FEATURE_SET_2;
  /** Own positions are indices into the request's `side.pokemon`; foe locations are the engine's choice locations. */
  threats: ThreatRow[];
  speeds: SpeedRow[];
  own_priorities: OwnPriorityRow[];
  foe_priorities: FoePriorityRow[];
}

/** One own Pokemon the arrays are about: an active or a benched switch-in. */
interface Entry {
  position: number;
  fighter: Fighter;
  /** The request slot, or null for a benched Pokemon. */
  slot: number | null;
}

type SpeedOrderQuery = Extract<MechanicsQuery["query"], { kind: "speed_order" }>;
type Sides = SpeedOrderQuery["sides"];
type OrderAction = SpeedOrderQuery["actions"][number];
type Asked = { kind: "answer"; answer: SpeedOrderAnswer } | { kind: "unsupported"; reason: OrderReason };

const compare = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
const fainted = (member: Record<string, unknown>): boolean => / fnt$/.test(text(member.condition, "condition"));
/** The index of the tie group that holds `action`. */
const group = (answer: SpeedOrderAnswer, action: number): number => answer.groups.findIndex((members) => members.includes(action));

/**
 * The features of one decision under `probe-damage/2`. `memo` is this game's: with or without it the arrays are the same
 * bytes. Team Preview and wait requests have no rows in any array; a forced switch has `threats` and `speeds` for the Pokemon
 * its `switch N` candidates name. Throws `FeatureBuildError` when the request cannot be featured.
 */
export function candidateFeatures2(
  engine: Engine,
  configs: FeatureConfigs,
  channel: PlayerChannel,
  team: readonly Readonly<PokemonSet>[],
  candidates: readonly Pick<Candidate, "choice" | "pending">[],
  memo: FeatureMemo = createFeatureMemo(),
): CandidateFeatures2 {
  const snapshot = configs.snapshot ?? loadRulesSnapshot(engine.commit);
  const identity = featureSetIdentity(engine, { ...configs, snapshot }, FEATURE_SET_2);
  const out = (parts: Partial<Omit<CandidateFeatures2, "feature_set" | "feature_set_hash">> = {}): CandidateFeatures2 => ({
    feature_set: FEATURE_SET_2,
    feature_set_hash: featureSetHash(identity),
    rows: [],
    threats: [],
    speeds: [],
    own_priorities: [],
    foe_priorities: [],
    ...parts,
  });

  const body = lastRequest(channel);
  const moveRequest = body.teamPreview === undefined && body.active !== undefined;
  const forcedSwitch = body.teamPreview === undefined && body.active === undefined && body.forceSwitch !== undefined;
  if (!moveRequest && !forcedSwitch) return out();
  const { view, state } = openRequest(engine, configs, snapshot, channel, memo);

  return refusing(() => {
    const side = object(body.side, "side");
    const pokemon = array(side.pokemon, "side.pokemon").map((member) => object(member, "a side Pokemon"));
    const slots = moveRequest ? array(body.active, "active").length : array(body.forceSwitch, "forceSwitch").length;
    const parsed = candidates.map(({ choice }) => {
      const parts = parseJointChoice(choice);
      if (parts.length !== slots) throw new DamageAwareError("a choice has a different number of parts than the request has active Pokemon");
      return parts;
    });

    let foes: Foe[] | null = null;
    const foesNow = (): Foe[] => (foes ??= view.foesOf(state, (mon) => view.assumedFor(mon, state)));
    const own: OwnSlot[] = moveRequest ? view.ownSlots(body, state, team) : [];
    const rows: FeatureRow[] = moveRequest ? damageRows(view, state, body, own, foesNow, candidates) : [];

    // The own Pokemon the arrays are about: the living actives of a move request, and every benched Pokemon a `switch N` candidate names.
    const entries: Entry[] = [];
    if (moveRequest) own.forEach((slot, index) => void (fainted(pokemon[index] as Record<string, unknown>) || entries.push({ position: index, fighter: slot.fighter, slot: index })));
    const benched = new Set<number>();
    for (const parts of parsed) for (const part of parts) if (part.kind === "switch") benched.add(part.to - 1);
    for (const position of [...benched].sort((a, b) => a - b)) {
      const member = pokemon[position];
      if (member === undefined) throw new DamageAwareError("a switch candidate names a Pokemon the request does not list");
      if (!fainted(member)) entries.push({ position, fighter: view.ownBench(body, state, team, position).fighter, slot: null });
    }
    entries.sort((a, b) => a.position - b.position);
    // An own Mega variant exists for an active slot where some candidate Mega Evolves.
    const megaSlots = new Set<number>();
    for (const parts of parsed) parts.forEach((part, slot) => void (part.kind === "move" && part.mega && megaSlots.add(slot)));
    const variants = (entry: Entry): boolean[] => (entry.slot !== null && megaSlots.has(entry.slot) ? [false, true] : [false]);
    const asMega = (entry: Entry, mega: boolean): Fighter => ({ ...entry.fighter, mega: entry.fighter.mega || mega });

    const living = foesNow();
    return {
      feature_set: FEATURE_SET_2,
      feature_set_hash: featureSetHash(identity),
      rows,
      threats: threatRows(view, state, entries, variants, asMega, living),
      speeds: speedRows(view, state, entries, variants, living),
      ...priorityRows(view, state, body, parsed, entries, living),
    } satisfies CandidateFeatures2;
  });
}

// ---------------------------------------------------------------- threats

/** One damage query per (own Pokemon and variant, foe, revealed damaging move that reaches an opposing Pokemon). */
function threatRows(view: DamageView, state: PublicState, entries: readonly Entry[], variants: (entry: Entry) => boolean[], asMega: (entry: Entry, mega: boolean) => Fighter, foes: readonly Foe[]): ThreatRow[] {
  const out: ThreatRow[] = [];
  for (const entry of entries) {
    for (const mega of variants(entry)) {
      for (const foe of foes) {
        for (const move of foe.assumed.revealed_moves) {
          const data = view.dex.moves.get(move);
          if (!data.exists || data.category === "Status" || foeTargetKind(data.target) === "none") continue;
          out.push({ own: entry.position, mega, foe: foe.loc, move, ...damageValues(view.damageResult(foe.fighter, asMega(entry, mega), move, state, false)) });
        }
      }
    }
  }
  return out.sort((a, b) => a.own - b.own || Number(a.mega) - Number(b.mega) || a.foe - b.foe || compare(a.move, b.move));
}

// ---------------------------------------------------------------- speed orders

/** One `speed_order` query: `own` and `foes` declared, the given actions asked, the field declared with the own side first. */
function ask(view: DamageView, state: PublicState, own: readonly Fighter[], foes: readonly Fighter[], actions: OrderAction[]): Asked {
  const sides = [own.map((f) => view.combatant(f)), foes.map((f) => view.combatant(f))] as unknown as Sides;
  const result = view.queryLimited({ kind: "speed_order", sides, actions, state: view.field(state, true) } as SpeedOrderQuery);
  if ("limit" in result) return { kind: "unsupported", reason: result.limit };
  if (result.kind === "unsupported") return { kind: "unsupported", reason: result.reason };
  if (result.kind !== "computed" || result.answer.kind !== "speed_order") throw new DamageAwareError("the probe gave no speed order to a speed-order query");
  return { kind: "answer", answer: result.answer as SpeedOrderAnswer };
}

/**
 * One `speeds` row per (own position, own Mega flag, foe location). Every combatant is declared with a switch action, so the
 * engine orders them by its action speed alone. At most two own positions share a query (a Mega variant is a query of its own,
 * its Pokemon declared Mega Evolved); a query the probe does not answer is asked again for each own Pokemon alone.
 */
function speedRows(view: DamageView, state: PublicState, entries: readonly Entry[], variants: (entry: Entry) => boolean[], foes: readonly Foe[]): SpeedRow[] {
  if (foes.length === 0) return [];
  const foeFighters = foes.map((foe) => foe.fighter);
  const switches = (ownCount: number): OrderAction[] => [
    ...Array.from({ length: ownCount }, (_, position) => ({ side: 0, position, move: null, mega: false })),
    ...foeFighters.map((_, position) => ({ side: 1, position, move: null, mega: false })),
  ];
  const rowsOf = (owns: { entry: Entry; mega: boolean }[], asked: Asked): SpeedRow[] =>
    owns.flatMap(({ entry, mega }, index) =>
      foes.map((foe, other): SpeedRow => {
        const key = { own: entry.position, mega, foe: foe.loc };
        if (asked.kind === "unsupported") return { ...key, kind: "unsupported", reason: asked.reason };
        const mine = group(asked.answer, index);
        const theirs = group(asked.answer, owns.length + other);
        if (mine < 0 || theirs < 0) throw new DamageAwareError("a speed order names every action in some group");
        return { ...key, kind: "computed", relation: mine < theirs ? "own_first" : mine === theirs ? "tie" : "foe_first" };
      }),
    );
  const askGroup = (owns: { entry: Entry; mega: boolean }[]): SpeedRow[] => {
    const fighters = (list: typeof owns): Fighter[] => list.map(({ entry, mega }) => ({ ...entry.fighter, mega: entry.fighter.mega || mega }));
    const asked = ask(view, state, fighters(owns), foeFighters, switches(owns.length));
    if (asked.kind === "answer" || owns.length === 1) return rowsOf(owns, asked);
    return owns.flatMap((single) => rowsOf([single], ask(view, state, fighters([single]), foeFighters, switches(1))));
  };
  const out: SpeedRow[] = [];
  for (let at = 0; at < entries.length; at += 2) out.push(...askGroup(entries.slice(at, at + 2).map((entry) => ({ entry, mega: false }))));
  for (const entry of entries) if (variants(entry).includes(true)) out.push(...askGroup([{ entry, mega: true }]));
  return out.sort((a, b) => a.own - b.own || Number(a.mega) - Number(b.mega) || a.foe - b.foe);
}

// ---------------------------------------------------------------- priorities

interface Pending {
  action: OrderAction;
  /** Where the answer goes. */
  set: (value: OrderValue<{ priority: number }>) => void;
}

/**
 * `own_priorities` and `foe_priorities`: the engine's priority key of each move asked, from batched `speed_order` queries
 * (round k asks the k-th move of every combatant, one action per combatant). A batch the probe does not answer is asked again
 * one action at a time, so only the offending row is `unsupported` and no row depends on how the moves were batched.
 */
function priorityRows(
  view: DamageView,
  state: PublicState,
  body: Record<string, unknown>,
  parsed: readonly SlotPart[][],
  entries: readonly Entry[],
  foes: readonly Foe[],
): { own_priorities: OwnPriorityRow[]; foe_priorities: FoePriorityRow[] } {
  const none = { own_priorities: [], foe_priorities: [] };
  const active = Array.isArray(body.active) ? (body.active as unknown[]) : null;
  if (active === null) return none; // a forced switch asks no move
  // Every distinct own move part of the candidates, by slot and Mega flag.
  const parts = new Map<string, { slot: number; move: number; mega: boolean; id: string }>();
  for (const list of parsed) {
    list.forEach((part, slot) => {
      if (part.kind !== "move") return;
      const key = `${slot}/${part.index}/${part.mega}`;
      if (parts.has(key)) return;
      const moves = array(object(active[slot], "an active entry").moves, "moves");
      parts.set(key, { slot, move: part.index, mega: part.mega, id: text(object(moves[part.index - 1], "a move of the request").id, "a move id") });
    });
  }
  const ownRows: OwnPriorityRow[] = [];
  const foeRows: FoePriorityRow[] = [];
  const actives = entries.filter((entry) => entry.slot !== null);
  if (actives.length === 0) return none;
  const ownFighters = actives.map((entry) => entry.fighter);
  const foeFighters = foes.length > 0 ? foes.map((foe) => foe.fighter) : [ownFighters[0] as Fighter]; // the order never reads the foe side's identity; a side must hold a Pokemon

  /** Ask `pending` (at most one action per combatant), batched, then one by one if the batch is not answered. */
  const resolve = (own: readonly Fighter[], pending: Pending[]): void => {
    if (pending.length === 0) return;
    const asked = ask(view, state, own, foeFighters, pending.map((p) => p.action));
    const apply = (list: Pending[], result: Asked): void =>
      list.forEach((p, index) => (result.kind === "unsupported" ? p.set({ kind: "unsupported", reason: result.reason }) : p.set({ kind: "computed", priority: (result.answer.actions[index] as { priority: number }).priority })));
    if (asked.kind === "answer" || pending.length === 1) return apply(pending, asked);
    for (const single of pending) apply([single], ask(view, state, own, foeFighters, [single.action]));
  };

  // Rounds of the non-Mega variant: the k-th move of every own active and every foe.
  const ownQueues = actives.map((entry) => [...parts.values()].filter((part) => part.slot === entry.slot && !part.mega).sort((a, b) => a.move - b.move));
  const foeQueues = foes.map((foe) => [...new Set(foe.assumed.revealed_moves)].sort(compare));
  const rounds = Math.max(0, ...ownQueues.map((q) => q.length), ...foeQueues.map((q) => q.length));
  for (let k = 0; k < rounds; k++) {
    const pending: Pending[] = [];
    ownQueues.forEach((queue, index) => {
      const part = queue[k];
      if (part === undefined) return;
      pending.push({ action: { side: 0, position: index, move: part.id, mega: false }, set: (value) => void ownRows.push({ slot: part.slot, move: part.move, mega: false, ...value }) });
    });
    foeQueues.forEach((queue, index) => {
      const move = queue[k];
      if (move === undefined) return;
      pending.push({ action: { side: 1, position: index, move, mega: false }, set: (value) => void foeRows.push({ foe: (foes[index] as Foe).loc, move, ...value }) });
    });
    resolve(ownFighters, pending);
  }
  // The Mega variant of a slot: its Pokemon alone, each move asked with the Mega Evolution this turn.
  for (const entry of actives) {
    const queue = [...parts.values()].filter((part) => part.slot === entry.slot && part.mega).sort((a, b) => a.move - b.move);
    for (const part of queue) {
      resolve([entry.fighter], [{ action: { side: 0, position: 0, move: part.id, mega: true }, set: (value) => void ownRows.push({ slot: part.slot, move: part.move, mega: true, ...value }) }]);
    }
  }
  return {
    own_priorities: ownRows.sort((a, b) => a.slot - b.slot || a.move - b.move || Number(a.mega) - Number(b.mega)),
    foe_priorities: foeRows.sort((a, b) => a.foe - b.foe || compare(a.move, b.move)),
  };
}
