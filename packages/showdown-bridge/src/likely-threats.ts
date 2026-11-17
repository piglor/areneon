// The likely-move threats of the feature set `probe-outcome-likely/1` (ADR-022 erratum E-22.7.2 to E-22.7.4).
//
// What a foe that has shown nothing could hit the player's Pokemon with: the moves its forme is likely to hold (a committed frequency table,
// not the opponent belief), ranked by EXPECTED THREAT, the few kept per foe priced against every own Pokemon. Three parts:
//
//  1. Likelihood (`inclusionsOf`): a move's share in the table, revealed moves removed and the rest renormalised over the slots left,
//     in exact integers: `min(10^6, floor(10^6 x mass x (slots - r) / (total - R)))`, r the revealed count and R their mass. Under the
//     generator-matched table that is exactly (slots - r) / (n - r) per pool move, the exact belief's marginal; nothing else is inferred
//     (no item, ability, spread or brought-set evidence), except that a revealed Choice item on a foe that has used a move since its
//     switch-in leaves it with no likely rows.
//  2. Ranking (`rank(m) = inclusion x max over own actives of value(m, a)`): the value is read from the probe's damage rolls at a BASE key
//     (the two sets and the move, no field, stages or status: so a pairing's rolls are asked once per game) against the active's CURRENT
//     hit points, exactly as the probe derives a value from rolls. The static pre-screen (`rank_candidates_max`) keeps the m most promising
//     candidates by engine data before any query.
//  3. Pricing: the kept k are priced at a FULL key (the base key plus both Pokemon's stat stages and status, the declared field and the
//     screens): the damage rolls come from one probe query per key and are kept in the game's `LikelyCache`; the values at a decision are
//     derived from the cached rolls and the defender's CURRENT hit points. A Pokemon's hit points are part of a key only when the move, or
//     its ability or item, defines an event handler of the record's `hp_key_handlers`. Nothing here ever changes a byte: with a warm, a
//     cold or no cache the rows are the same (the equality test is the proof).
//
// The values are on ADR-018's assumed foe set plus the move, never facts about the real opponent. Likely-threat queries declare no history
// (a move whose power reads it is priced at its undeclared value, a declared limit); the foe-priority kind of a likely move is NONE.
import { toId, type PublicMon, type WorldFacts } from "@areneon/public-state";
import { type ArrayContext } from "./candidate-features-2.ts";
import type { BeliefRefusal } from "./candidate-features-3.ts";
import { effectsOf, worldDataFor, worldFactsOf, rawDexOf } from "./candidate-features-outcome.ts";
import { type DamageValues, damageValuesOf, type FeatureConfigs, type FeatureMemo, emptyBeliefCounts, share } from "./candidate-features.ts";
import type { PlayerChannel } from "./observation.ts";
import { assumedWithMove } from "./damage-aware-assumed-set.ts";
import { rollsValue } from "./damage-aware-score.ts";
import { DamageAwareError, type DamageView, type Fighter, type Foe, foeTargetKind, ProbeRefusal } from "./damage-aware-view.ts";
import type { Engine } from "./engine.ts";
import { type LikelyConfig, type LikelyScreen, type LikelySetup, type MoveList, readsHitPoints } from "./likely-threats-config.ts";
import type { MechanicsQuery } from "./mechanics-query.ts";
import type { PlainReason } from "./mechanics-result.ts";

const MILLION = 1_000_000;
/** Order of two distinct ids (a duplicate is never compared). */
const compare = (a: string, b: string): number => (a < b ? -1 : 1);

/** What a likely threat row holds: the damage values of a `/3` row and the table's inclusion of the move. */
export type LikelyRow = { own: number; mega: boolean; foe: number; move: string; inclusion_ppm: number } & DamageValues;

/** The probe's damage rolls of one key, or the closed reason it could not answer. */
export type CachedDamage = { kind: "rolls"; no_crit: readonly number[]; hp_max: number; ko_blocked: boolean } | { kind: "unsupported"; reason: PlainReason };

/** Integer counts of a game's likely threats (a measurement, never part of the rows). */
export interface LikelyCounts {
  /** Requests that built likely rows (a refused request counts in the memo's belief counts instead). */
  decisions: number;
  /** Foe-decisions that reached the cut: candidates after the filters, and the foes that needed a ranking. */
  candidates: number;
  ranked_foes: number;
  /** Candidates dropped by the static pre-screen. */
  pre_screened: number;
  /** Foes left with no likely rows by the revealed-Choice rule. */
  choice_locked: number;
  rows: number;
  /** Roll lookups by key level: asked of the cache and answered by it. */
  base_lookups: number;
  base_hits: number;
  full_lookups: number;
  full_hits: number;
  /** Placement lookups (a foe's assumed set with a move) and the ones the cache answered. */
  placement_lookups: number;
  placement_hits: number;
}
export const emptyLikelyCounts = (): LikelyCounts => ({ decisions: 0, candidates: 0, ranked_foes: 0, pre_screened: 0, choice_locked: 0, rows: 0, base_lookups: 0, base_hits: 0, full_lookups: 0, full_hits: 0, placement_lookups: 0, placement_hits: 0 });

/** A game's cache (in the game's `FeatureMemo`): damage rolls by key, placements by (assumed set, move), and whether an effect reads hit points. */
export interface LikelyCache {
  rolls: Map<string, CachedDamage>;
  placed: Map<string, { paste: string } | null>;
  hpReads: Map<string, boolean>;
  counts: LikelyCounts;
}
export const createLikelyCache = (): LikelyCache => ({ rolls: new Map(), placed: new Map(), hpReads: new Map(), counts: emptyLikelyCounts() });

// ---------------------------------------------------------------- likelihood

/**
 * The inclusion (ppm) of each move of `list` that is not revealed, with `revealedCount` moves already shown: the share of the list renormalised
 * over the slots left. Sorted by id; a move with a zero inclusion is left out; no rows when no slot or no mass is left.
 */
export function inclusionsOf(list: MoveList, revealed: ReadonlySet<string>, revealedCount: number): { id: string; inclusion_ppm: number }[] {
  const slotsLeft = list.slots - revealedCount;
  let seen = 0;
  for (const id of revealed) seen += list.masses.get(id) ?? 0;
  const massLeft = list.total - seen;
  if (slotsLeft <= 0 || massLeft <= 0) return [];
  const out: { id: string; inclusion_ppm: number }[] = [];
  for (const [id, mass] of list.masses) {
    if (revealed.has(id)) continue;
    // Exact in binary64: the numerator is at most 10^6 x mass x slots, and `loadMoveTable` refuses a table with a mass x slots above
    // `MAX_MASS_TIMES_SLOTS` (9 x 10^9), so it stays below 9 x 10^15 < 2^53; a quotient of two exact integers is then correctly rounded
    // and its floor cannot cross an integer (an exact multiple is exactly representable).
    const inclusion = Math.min(MILLION, Math.floor((MILLION * mass * slotsLeft) / massLeft));
    if (inclusion > 0) out.push({ id, inclusion_ppm: inclusion });
  }
  return out.sort((a, b) => compare(a.id, b.id));
}

/** The table list of a shown Pokemon: its shown forme's, else its base species'; undefined when the table has neither. */
export function listOf(table: ReadonlyMap<string, MoveList>, mon: PublicMon): MoveList | undefined {
  return (mon.species === null ? undefined : table.get(toId(mon.species))) ?? table.get(toId(mon.base_species));
}

// ---------------------------------------------------------------- keys and rolls

type DamageQuery = Extract<MechanicsQuery["query"], { kind: "damage" }>;
type DamageState = DamageQuery["state"];

const ZERO_BOOSTS: Fighter["boosts"] = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const NEUTRAL_SIDE = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
const NEUTRAL_FIELD: DamageState = { weather: null, terrain: null, trick_room: false, sides: [NEUTRAL_SIDE, NEUTRAL_SIDE] };

/** The engine data the cache keys and the pre-screen read (the pinned checkout's own tables). */
interface RawDex {
  moves: { get(id: string): Record<string, unknown> };
  species: { get(name: string): { types: string[] } };
  types: { get(name: string): { damageTaken: Record<string, number> } };
  items: { get(name: string): Record<string, unknown> };
}

/** Everything one decision's likely rows need besides the table and the record. */
interface Run {
  engine: Engine;
  configs: FeatureConfigs;
  setup: LikelySetup;
  view: DamageView;
  cache: LikelyCache;
  raw: ReturnType<typeof rawDexOf>;
  dex: RawDex;
  koWeight: number;
}

const importSets = (engine: Engine) => (paste: string): { species: string; ability?: string; item?: string }[] => engine.sim.Teams.import(paste) as { species: string; ability?: string; item?: string }[];

/** Whether the damage a (attacker, move, defender) triple deals can depend on a hit-point value (a Mega's own ability and an unknown name count as yes: nothing is proven). */
function hpKeyed(run: Run, attacker: Fighter, defender: Fighter, move: string): boolean {
  const key = JSON.stringify([attacker.paste, attacker.mega, defender.paste, defender.mega, move]);
  let found = run.cache.hpReads.get(key);
  if (found === undefined) {
    const handlers = run.setup.config.hp_key_handlers;
    const read = importSets(run.engine);
    const a = effectsOf(read, run.raw, attacker.paste);
    const d = effectsOf(read, run.raw, defender.paste);
    found =
      attacker.mega || defender.mega || a === null || d === null || readsHitPoints(run.dex.moves.get(move), handlers) || [a.ability, a.item, d.ability, d.item].some((data) => readsHitPoints(data, handlers));
    run.cache.hpReads.set(key, found);
  }
  return found;
}

/** One damage query without declared history; a refusal or an `unsupported` answer is a cached reason, never a thrown error. */
function askRolls(run: Run, attacker: Fighter, defender: Fighter, move: string, state: DamageState): CachedDamage {
  let result: ReturnType<DamageView["query"]>;
  try {
    result = run.view.query({ kind: "damage", attacker: run.view.combatant(attacker), defender: run.view.combatant(defender), move, state });
  } catch (error) {
    if (!(error instanceof ProbeRefusal)) throw error;
    run.view.noteRefused();
    return { kind: "unsupported", reason: "query_kind" };
  }
  if (result?.kind === "unsupported") return { kind: "unsupported", reason: run.view.reasonOf(result.reason) };
  if (result?.kind !== "computed" || result.answer.kind !== "damage") throw new DamageAwareError("the probe gave no damage answer to a damage query");
  return { kind: "rolls", no_crit: [...result.answer.no_crit], hp_max: result.answer.hp_max, ko_blocked: result.answer.ko.kind !== "counts" };
}

/** The rolls of a key, from the cache or one probe query (`level` only says which counters to move). */
function rollsOf(run: Run, level: "base" | "full", key: string, ask: () => CachedDamage): CachedDamage {
  const counts = run.cache.counts;
  counts[level === "base" ? "base_lookups" : "full_lookups"] += 1;
  const found = run.cache.rolls.get(key);
  if (found !== undefined) {
    counts[level === "base" ? "base_hits" : "full_hits"] += 1;
    return found;
  }
  const made = ask();
  run.cache.rolls.set(key, made);
  return made;
}

/** The test seam's row: one uncached query at the real state, valued as `probe-damage/3` values its rows. */
function directValues(run: Run, attacker: Fighter, defender: Fighter, move: string, state: Parameters<DamageView["field"]>[0]): DamageValues {
  return damageValuesOf(run.view, () => run.view.query({ kind: "damage", attacker: run.view.combatant(attacker), defender: run.view.combatant(defender), move, state: run.view.field(state, false) }));
}

/** The values of a damage row from cached rolls and the defender's current hit points, exactly as the probe's answer gives them. */
export function valuesAt(cached: CachedDamage, hp: number | null): DamageValues {
  if (cached.kind === "unsupported") return cached;
  const current = hp ?? cached.hp_max;
  const rolls = cached.no_crit;
  return {
    kind: "computed",
    mean_ppm: rollsValue(rolls, current, 0),
    max_ppm: share(Math.min(Math.max(...rolls), current), current),
    min_ppm: share(Math.min(Math.min(...rolls), current), current),
    ohko_rolls: cached.ko_blocked ? null : rolls.filter((roll) => roll >= current).length,
  };
}

/** The value a rank reads: the mean plus the KO weight over the rolls that reach the hit points (a computed row only). */
const valueOf = (values: DamageValues, koWeight: number): number => (values.kind === "computed" ? values.mean_ppm + Math.floor((koWeight * (values.ohko_rolls ?? 0)) / 16) : 0);

const baseKey = (attacker: Fighter, defender: Fighter, move: string): string => JSON.stringify([attacker.paste, attacker.mega, defender.paste, defender.mega, move]);

function baseRolls(run: Run, attacker: Fighter, defender: Fighter, move: string): CachedDamage {
  const a: Fighter = { paste: attacker.paste, hp: null, status: null, boosts: ZERO_BOOSTS, mega: attacker.mega };
  const d: Fighter = { paste: defender.paste, hp: null, status: null, boosts: ZERO_BOOSTS, mega: defender.mega };
  return rollsOf(run, "base", baseKey(attacker, defender, move), () => askRolls(run, a, d, move, NEUTRAL_FIELD));
}

function fullRolls(run: Run, attacker: Fighter, defender: Fighter, move: string, state: Parameters<DamageView["field"]>[0]): CachedDamage {
  const field = run.view.field(state, false);
  const drop = run.setup.drop_key;
  const reads = drop !== "hp" && hpKeyed(run, attacker, defender, move);
  const a: Fighter = { ...attacker, hp: reads ? attacker.hp : null };
  const d: Fighter = { ...defender, hp: reads ? defender.hp : null };
  const key = JSON.stringify([baseKey(attacker, defender, move), drop === "stages" ? null : [a.boosts, d.boosts], a.status, d.status, field, reads ? [a.hp, d.hp] : null]);
  return rollsOf(run, "full", key, () => askRolls(run, a, d, move, field));
}

// ---------------------------------------------------------------- the static pre-screen

/** The static score of a candidate against one defender: inclusion x base power x type chart x same-type x spread, in exact integers. */
function staticScore(run: Run, config: LikelyConfig, foe: Foe, move: string, inclusion: number, defender: Fighter): bigint {
  const screen = config.screen as LikelyScreen; // a record with a pre-screen always has its factors (the reader requires both)
  const data = run.dex.moves.get(move) as { basePower: number; type: string; target: string };
  const types = (paste: string): string[] => run.dex.species.get((importSets(run.engine)(paste)[0] as { species: string }).species).types;
  const power = data.basePower > 0 ? data.basePower : screen.variable_power;
  let score = BigInt(inclusion) * BigInt(power);
  for (const type of types(defender.paste)) {
    const code = run.dex.types.get(type).damageTaken[data.type] as number;
    score = (score * BigInt(screen.effectiveness_ppm[code] as number)) / BigInt(MILLION);
  }
  if (types(foe.fighter.paste).includes(data.type)) score = (score * BigInt(screen.same_type_ppm)) / BigInt(MILLION);
  if (foeTargetKind(data.target) === "every") score = (score * BigInt(screen.spread_ppm)) / BigInt(MILLION);
  return score;
}

// ---------------------------------------------------------------- the rows

interface Kept {
  move: string;
  inclusion_ppm: number;
  paste: string;
}

/** The foe with its set holding `move` (the placement the validator accepts), cached per game. */
function placement(run: Run, foe: Foe, move: string): { paste: string } | null {
  const counts = run.cache.counts;
  counts.placement_lookups += 1;
  const key = `${foe.fighter.paste}|${move}`;
  if (run.cache.placed.has(key)) {
    counts.placement_hits += 1;
    return run.cache.placed.get(key) as { paste: string } | null;
  }
  const placed = assumedWithMove({ engine: run.engine, format_id: run.configs.config.format_id }, foe.assumed, move);
  const kept = placed === null ? null : { paste: placed.paste };
  run.cache.placed.set(key, kept);
  return kept;
}

/** Whether a foe holds a Choice item it has been seen to use a move under since it came in: then it can only repeat a revealed move. */
export function choiceLocked(items: { get(name: string): Record<string, unknown> }, foe: Pick<Foe, "mon" | "fighter">, facts: () => WorldFacts | null): boolean {
  const item = foe.mon.item;
  if (item === null || foe.mon.item_lost || items.get(item).isChoice !== true) return false;
  const known = facts();
  const letter = foe.fighter.at?.letter;
  if (known === null || letter === null || letter === undefined) return foe.mon.moves_revealed.length > 0;
  return (known.opponent.active[letter]?.last_move ?? null) !== null;
}

/** The likely rows of one move or forced-switch request, or the refusal code that left them empty. */
export function likelyThreatRows(engine: Engine, configs: FeatureConfigs, setup: LikelySetup, memo: FeatureMemo, channel: PlayerChannel, context: ArrayContext): { belief_threats: LikelyRow[]; belief_threats_refused?: BeliefRefusal } {
  const { view, state, entries, variants, asMega, foes, threats, flags } = context;
  const counts = (memo.belief ??= emptyBeliefCounts());
  counts.decisions += 1;
  const refuse = (code: BeliefRefusal): { belief_threats: LikelyRow[]; belief_threats_refused: BeliefRefusal } => {
    counts.fallbacks[code] = (counts.fallbacks[code] ?? 0) + 1;
    return { belief_threats: [], belief_threats_refused: code };
  };
  if (flags.length > 0) return refuse("reader_flagged");
  const real = foes.filter((foe) => foe.mon.phantom !== true);
  const lists = new Map<Foe, MoveList>();
  for (const foe of real) {
    const list = listOf(setup.table, foe.mon);
    if (list === undefined) return refuse("species_not_in_prior");
    lists.set(foe, list);
  }
  const cache = (memo.likely ??= createLikelyCache());
  cache.counts.decisions += 1;
  const dex = rawDexOf(engine, configs) as unknown as RawDex;
  const run: Run = { engine, configs, setup, view, cache, raw: rawDexOf(engine, configs), dex, koWeight: configs.config.scoring.ko_weight_ppm };
  const { config } = setup;
  let facts: WorldFacts | null | undefined;
  const lazyFacts = (): WorldFacts | null => (facts === undefined ? (facts = worldFactsOf(channel, worldDataFor(engine, configs))) : facts);

  // The own Pokemon a threat is ranked against: the living actives of a move request, the benched switch-ins of a forced switch.
  const defenders = entries.some((entry) => entry.slot !== null) ? entries.filter((entry) => entry.slot !== null) : entries;
  const kept = new Map<Foe, Kept[]>();
  for (const foe of foes) kept.set(foe, []);
  for (const foe of real) {
    if (choiceLocked(run.dex.items, foe, lazyFacts)) {
      cache.counts.choice_locked += 1;
      continue;
    }
    const shown = new Set([...foe.mon.moves_revealed.map((id) => toId(id)), ...foe.assumed.revealed_moves, ...threats.filter((t) => t.foe === foe.loc).map((t) => t.move)]);
    const candidates: Kept[] = [];
    for (const { id, inclusion_ppm } of inclusionsOf(lists.get(foe) as MoveList, shown, new Set(foe.mon.moves_revealed.map((move) => toId(move))).size)) {
      const data = view.dex.moves.get(id);
      if (!data.exists || data.category === "Status" || foeTargetKind(data.target) === "none") continue;
      const placed = placement(run, foe, id);
      if (placed !== null) candidates.push({ move: id, inclusion_ppm, paste: placed.paste });
    }
    cache.counts.candidates += candidates.length;
    let pool = candidates;
    if (config.rank_candidates_max !== undefined && pool.length > config.rank_candidates_max && defenders.length > 0) {
      const scored = pool.map((candidate) => ({ candidate, score: defenders.reduce((best, entry) => { const next = staticScore(run, config, foe, candidate.move, candidate.inclusion_ppm, entry.fighter); return next > best ? next : best; }, 0n) }));
      scored.sort((a, b) => (a.score === b.score ? compare(a.candidate.move, b.candidate.move) : a.score > b.score ? -1 : 1));
      pool = scored.slice(0, config.rank_candidates_max).map((entry) => entry.candidate);
      cache.counts.pre_screened += candidates.length - pool.length;
    }
    if (pool.length > config.per_foe) {
      cache.counts.ranked_foes += 1;
      const ranked = pool.map((candidate) => {
        const attacker: Fighter = { ...foe.fighter, paste: candidate.paste };
        const best = defenders.reduce((top, entry) => Math.max(top, valueOf(valuesAt(baseRolls(run, attacker, entry.fighter, candidate.move), entry.fighter.hp), run.koWeight)), 0);
        return { candidate, rank: candidate.inclusion_ppm * best };
      });
      ranked.sort((a, b) => b.rank - a.rank || compare(a.candidate.move, b.candidate.move));
      pool = ranked.slice(0, config.per_foe).map((entry) => entry.candidate);
    }
    kept.set(foe, pool);
  }

  const out: LikelyRow[] = [];
  for (const entry of entries) {
    for (const mega of variants(entry)) {
      const defender = asMega(entry, mega);
      for (const foe of foes) {
        for (const { move, inclusion_ppm, paste } of kept.get(foe) as Kept[]) {
          const attacker: Fighter = { ...foe.fighter, paste };
          const values = setup.direct === true ? directValues(run, attacker, defender, move, state) : valuesAt(fullRolls(run, attacker, defender, move, state), defender.hp);
          out.push({ own: entry.position, mega, foe: foe.loc, move, inclusion_ppm, ...values });
        }
      }
    }
  }
  cache.counts.rows += out.length;
  return { belief_threats: out.sort((a, b) => a.own - b.own || Number(a.mega) - Number(b.mega) || a.foe - b.foe || compare(a.move, b.move)) };
}

