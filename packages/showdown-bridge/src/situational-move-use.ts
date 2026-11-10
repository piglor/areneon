// The situational move-use diagnostic (#1283, for the pre-league gate #1274): does a policy protect and switch IN THE RIGHT
// SITUATIONS, not merely often. Evaluation code only: it reads finished games and writes integer counts.
//
// LABEL: situational move-use diagnostic on assumed opponent sets: descriptive, not a strength measure.
//
// Input: the script reads a progress generation only through `situational-input.ts` (manifest, progress record and game records
// verified); `tallyGame` takes such a record. The classification is per slot, with the own action at priority 0, whatever move
// the seat then chooses.
//
// Every situation is computed from ONE seat's own view at the decision, never from the omniscient log: the seat's own channel
// up to the request (taken from a re-simulation of the stored game, as the feature bench does), its own team, the decision's
// candidates, and the `probe-damage/2` arrays (threats, speed relations and priorities on ADR-018's assumed opponent sets) plus
// the world facts' `protect_streak`. The stalling moves are the engine's own list (`stall_moves` of the world facts' data,
// read from the pinned source); no move, threat or order rule is written here.
//
// Per own living active at a move request (position = slot, `side.pokemon[slot]`):
//   Threatened      some REVEALED foe damaging move reaches its HP with at least one no-crit roll (`ohko_rolls > 0`, or `max_ppm >= 1,000,000` when the KO counts are null) and that
//                   foe acts first: the foe move's priority is above 0, or equal to 0 and the speed relation is `foe_first`
//                   (the own action is taken at default priority 0; a tie is not "first").
//   Safe            no revealed foe move reaches its HP at all.
//   Exposed slower  reachable, but every reaching foe acts after it (neither of the two above).
//   Bad matchup     Threatened and a `switch` candidate of that slot names a bench Pokemon no revealed foe move reaches.
//   Repeat          the slot's Pokemon used a stalling move successfully last turn (`protect_streak >= 1`).
// An unsupported or null probe row never threatens (and is counted); nothing is guessed.
import { type OwnChannel, ownChannel, type WorldFactsData, worldFactsAtRequests } from "@areneon/public-state";
import type { CandidateFeatures2 } from "./candidate-features-2.ts";
import { parseJointChoice, type SlotPart } from "./damage-aware-choice.ts";
import { classifySlot, reaches as reachesRows, SITUATIONS, type Situation } from "./situation.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { createFeatureService, type FeatureService, type FeatureServiceSpec } from "./feature-service.ts";
import { ResimulationInputError, resimulate } from "./resimulate.ts";
import { BattleAbortedError, type Player } from "./runner.ts";

export const SITUATIONAL_FORMAT = "situational-move-use/1";
export const SITUATIONAL_LABEL = "situational move-use diagnostic on assumed opponent sets: descriptive, not a strength measure";

export { SITUATIONS, type Situation };
/** The switch classes: a Threatened slot splits by whether a bench Pokemon offered by a `switch` candidate is safe. */
export const SWITCH_CLASSES = ["bad_matchup", "threatened_no_safe_bench", "safe", "exposed_slower"] as const;

/** Integer counts by flat dotted key; keys are written sorted. */
export type Counts = Record<string, number>;
const bump = (counts: Counts, key: string, by = 1): void => void (counts[key] = (counts[key] ?? 0) + by);

/** What one move-request decision of one seat is classified from. */
export interface DecisionInput {
  /** The request body (the JSON after `|request|`). */
  body: Record<string, unknown>;
  candidates: readonly { choice: string; pending: boolean }[];
  chosen: string;
  features: CandidateFeatures2;
  /** Per own slot, the `protect_streak` of the facts; null when the facts were unavailable. */
  streaks: readonly (number | null)[];
  stall: ReadonlySet<string>;
}

const asRecord = (value: unknown): Record<string, unknown> => (typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {});
const asList = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

/** Add one decision's slot classifications and choices to `counts`. */
export function tallyDecision(counts: Counts, input: DecisionInput): void {
  const { body, features, stall } = input;
  const pokemon = asList(asRecord(body.side).pokemon).map(asRecord);
  const active = asList(body.active).map(asRecord);
  const parsed = input.candidates.map((c) => parseJointChoice(c.choice));
  const chosen = parseJointChoice(input.chosen);
  if (chosen.length !== active.length || parsed.some((p) => p.length !== active.length)) throw new Error("a choice has a different number of parts than the request has active Pokemon");
  bump(counts, "decisions");

  // The reading itself (a row reaches the HP when some no-crit roll does, who acts first) is `situation.ts`, shared with `damage-aware@v4`.
  const unknown = (kind: "foe_priority" | "speed_relation"): void => bump(counts, kind === "foe_priority" ? "unknown_foe_priority" : "unknown_speed_relation");
  const reaches = (position: number): ReturnType<typeof reachesRows> => reachesRows(features, position);
  for (const t of features.threats) if (t.kind === "computed" && t.ohko_rolls === null) bump(counts, "ko_fallback_rows");
  const moveId = (slot: number, part: SlotPart): string | null => (part.kind === "move" ? ((asList(asRecord(active[slot]).moves)[part.index - 1] as { id?: string } | undefined)?.id ?? null) : null);

  for (let slot = 0; slot < active.length; slot++) {
    if (/ fnt$/.test(String(asRecord(pokemon[slot]).condition))) continue;
    const { situation, first } = classifySlot(features, slot, unknown);
    if (situation === "threatened" && first.every((r) => r.fallback)) bump(counts, "threatened_by_max_fallback");
    bump(counts, `slots.${situation}`);

    const protectOffered = parsed.some((p) => stall.has(moveId(slot, p[slot] as SlotPart) ?? ""));
    const protectChosen = stall.has(moveId(slot, chosen[slot] as SlotPart) ?? "");
    const streak = input.streaks[slot] ?? null;
    if (protectOffered) {
      if (streak === null) bump(counts, "protect.no_facts.offered");
      else {
        const key = `protect.${situation}.${streak >= 1 ? "repeat" : "fresh"}`;
        bump(counts, `${key}.offered`);
        if (protectChosen) bump(counts, `${key}.chosen`);
      }
    }

    const benches = new Set<number>();
    for (const p of parsed) {
      const part = p[slot] as SlotPart;
      if (part.kind === "switch") benches.add(part.to - 1);
    }
    if (benches.size > 0) {
      const safeBench = [...benches].some((b) => reaches(b).length === 0);
      const cls = situation === "threatened" ? (safeBench ? "bad_matchup" : "threatened_no_safe_bench") : situation;
      bump(counts, `switch.${cls}.offered`);
      if ((chosen[slot] as SlotPart).kind === "switch") bump(counts, `switch.${cls}.chosen`);
    }
  }
}

/** The counts with their keys sorted (the bytes of a result never depend on the order things happened). */
export const sortedCounts = (counts: Counts): Counts => Object.fromEntries(Object.entries(counts).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));

/** A stored progress game record: the fields this reads. Its omniscient log is not read. */
export interface StoredGame {
  data_class: string;
  match_id: string;
  game: number;
  pair?: number;
  seed: string;
  policies: Record<Player, string>;
  teams: Record<Player, { team: PokemonSet[] }>;
  decisions: { player: Player; request_seq: number; chosen: string; candidates: { choice: string; pending: boolean }[] }[];
}

export interface SeatTally {
  policy: string;
  counts: Counts;
}
export interface GameTally {
  match_id: string;
  game: number;
  pair: number | null;
  seats: Record<Player, SeatTally>;
}

export interface SituationalSetup {
  engine: Engine;
  format_id: string;
  max_turns: number;
  facts_data: WorldFactsData;
  /** The feature service (`probe-damage/2`); one per process, it keeps a probe memo per game. */
  service: FeatureService;
}

export function situationalSetup(engine: Engine, formatId: string, maxTurns: number, factsData: WorldFactsData, spec?: Partial<FeatureServiceSpec>): SituationalSetup {
  const full: FeatureServiceSpec = { format_id: formatId, data_class: "synthetic", feature_set: "probe-damage/2", ...spec };
  return { engine, format_id: formatId, max_turns: maxTurns, facts_data: factsData, service: createFeatureService(full, engine) };
}

const REQUEST = "|request|";

/** Tally one stored game: re-simulate it from seed, teams and recorded choices, and classify every move decision of each seat from that seat's own channel. */
export async function tallyGame(setup: SituationalSetup, record: StoredGame): Promise<GameTally | null> {
  if (record.data_class !== "synthetic") throw new Error("the diagnostic reads synthetic progress games only");
  let channels: Record<Player, string[]>;
  try {
    channels = (
      await resimulate(setup.engine, {
        format_id: setup.format_id,
        seed: record.seed,
        max_turns: setup.max_turns,
        teams: { p1: record.teams.p1.team, p2: record.teams.p2.team },
        choices: record.decisions.map((d) => ({ player: d.player, request_seq: d.request_seq, choice: d.chosen })),
      })
    ).channels;
  } catch (error) {
    // Only the re-simulation's own refusals mean "this game cannot be replayed"; a defect is not hidden as a count.
    if (error instanceof ResimulationInputError || error instanceof BattleAbortedError) return null;
    throw error;
  }
  const stall = new Set(setup.facts_data.stall_moves);
  const seats = {} as Record<Player, SeatTally>;
  for (const player of ["p1", "p2"] as const) {
    const counts: Counts = {};
    const channel = channels[player];
    const own: OwnChannel = ownChannel(player, channel);
    const facts = worldFactsAtRequests({ channel: own, data: setup.facts_data });
    const requestLines = channel.flatMap((line, index) => (line.startsWith(REQUEST) ? [index] : []));
    const mine = record.decisions.filter((d) => d.player === player).sort((a, b) => a.request_seq - b.request_seq);
    mine.forEach((d, order) => {
      const end = requestLines[d.request_seq - 1] as number; // `resimulate` refused any choice for a request the battle did not ask
      const line = channel[end] as string;
      const body = JSON.parse(line.slice(REQUEST.length)) as Record<string, unknown>;
      if (body.active === undefined || body.teamPreview !== undefined) {
        bump(counts, "not_move_request");
        return;
      }
      const result = setup.service.run({ game: `${record.match_id}:${record.game}:${player}`, player, lines: channel.slice(0, end + 1), team: record.teams[player].team, candidates: d.candidates, ...(order === mine.length - 1 ? { final: true } : {}) });
      if (result.outcome.kind === "error") {
        bump(counts, `feature_error.${result.outcome.code}`);
        return;
      }
      const f = facts.find((x) => x.line === end);
      const letters = asList(body.active).map((_, slot) => String.fromCharCode(97 + slot));
      const streaks = letters.map((letter) => f?.facts?.own.active[letter]?.protect_streak ?? null);
      tallyDecision(counts, { body, candidates: d.candidates, chosen: d.chosen, features: result.outcome.features as CandidateFeatures2, streaks, stall });
    });
    seats[player] = { policy: record.policies[player], counts: sortedCounts(counts) };
  }
  return { match_id: record.match_id, game: record.game, pair: record.pair ?? null, seats };
}
