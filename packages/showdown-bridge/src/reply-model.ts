// The probe max-damage reply (ADR-026 section 2, finding B1, rules N-L; slice D): the opponent's reply in a world.
// In a scenario world (`world-builder.ts`) the opponent knows the sets of that world, so its reply is the legal joint
// choice that deals the most damage by the engine's own numbers: ADR-016 `damage` queries on the world's declared
// sets and state, valued by the pure `damageValue` it shares with the ADR-018 baseline under the reviewed weights of
// `config/baselines/damage-aware.v1.json`.
//
// This is a new code path beside the damage-aware actor, not the actor and not its `assume` seam: it reads no PublicState, no
// reader and no channel, only the world (a battle the builder made, ours, so the engine's verdicts on it reveal nothing).
// The world's opponent request is the engine's own for the world as written; legality is `enumerateCandidates` and a
// `CloneJudge` on a clone of it. The world passed in is never changed.
//
// Rules (N-L):
//  - a spread move is the sum of its single-target values over the foes it hits (the foes only, as ADR-018's v1; a random-target
//    move is the mean over the foes);
//  - an `unsupported` query scores 0, and so does a query the engine refuses to state (counted as `probe_refusals`);
//  - the reply never switches and never uses a status move: a slot part is eligible only when it is `pass` or a damaging
//    move that hits foes; a slot that has no eligible part in any legal candidate (a locked move, a Pokemon with only status
//    moves) is held to its first legal part in canonical order (counted as `forced_slots`);
//  - the candidate with the highest summed value wins, the first of equal values in canonical order;
//  - when that value is 0 (no damaging move, or every query unsupported) the reply is the `actor-rule` choice of the world's
//    legal set, the first legal joint candidate in canonical order, counted as `all_zero`.
//
// Limits: the attacker is its declared set (a temporary forme a world shows is not stated to the probe), its volatiles are not
// stated (the world builder refuses them), and the reply is one choice (the uniform draws and the weights are slice R).

import { actorRule } from "./actor-rule.ts";
import { type Candidate, enumerateCandidates } from "./choices.ts";
import { CloneJudge, legalChoiceSet } from "./choice-judge.ts";
import { type SlotPart, parseJointChoice } from "./damage-aware-choice.ts";
import type { DamageAwareConfig } from "./damage-aware-config.ts";
import { DAMAGE_VALUE_FORMAT, damageValue } from "./damage-aware-score.ts";
import { type Fighter, ProbeRefusal, type ProbeMemo, damageView, foeTargetKind } from "./damage-aware-view.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import type { MechanicsConfig } from "./mechanics-config.ts";
import { type MechanicsContext, type ProbeOutcome, probeMechanics } from "./mechanics-probe.ts";
import type { ParsedMechanicsQuery } from "./mechanics-query.ts";
import type { RulesSnapshot } from "./rules.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import { decodeRequest } from "./request.ts";
import { exportPaste } from "./teams.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";
import type { ScenarioWorld } from "./world-builder.ts";

/** The version of this reply model as the search identity names it (ADR-026 section 5). Bump it when the rules above, or what a world's state is read as, change. */
export const REPLY_MODEL_FORMAT = 1;
export { DAMAGE_VALUE_FORMAT };

/** The world cannot be replied in (its opponent has no move request, or a part is of a shape this model does not know). A defect of the caller's world, not a refusal of it. */
export class ReplyModelError extends Error {
  override name = "ReplyModelError";
}

export type ReplyKind = "max_damage" | "all_zero";

export interface Reply {
  /** The opponent's joint choice, in canonical engine syntax. */
  choice: string;
  /** `max_damage`: the best candidate by value; `all_zero`: every value was 0 and the `actor-rule` choice was played. */
  kind: ReplyKind;
  /** The winning candidate's value in ppm (0 for `all_zero`). */
  value: number;
}

export interface ReplyCounts {
  /** Replies computed. */
  replies: number;
  max_damage: number;
  /** Replies where every candidate was worth 0: the first legal joint candidate was played. */
  all_zero: number;
  /** Slots with no eligible part, held to their first legal part. */
  forced_slots: number;
  /** Legal non-pending joint candidates considered (before the eligibility rule). */
  candidates: number;
  /** Probe `damage` queries asked (a repeated query counts each time it is asked). */
  queries: number;
  /** Probe runs actually made (queries not answered from the world's memo). */
  probe_runs: number;
  /** Queries the probe answered `unsupported` (they score 0). */
  unsupported_queries: number;
  /** Queries whose declared state the engine refused (they score 0). */
  probe_refusals: number;
}

export interface ReplyModelOptions {
  engine: Engine;
  vocabulary: TeamVocabulary;
  /** The reviewed weights (`config/baselines/damage-aware.v1.json`); `damage_aware_config_hash` names them in the search identity. */
  config: DamageAwareConfig;
  mechanics: MechanicsConfig;
  /** The rules snapshot; read from `rules/` once when absent. */
  snapshot?: RulesSnapshot;
  /** The class the probe queries carry (they are never stored). */
  data_class: "synthetic" | "private";
  /** Test seam: the probe (default `probeMechanics`). */
  probe?: (context: MechanicsContext, query: ParsedMechanicsQuery) => ProbeOutcome;
}

export interface ReplyModel {
  /**
   * The opponent's probe max-damage reply in `world`. `memo` is the world's memo of probe answers by query hash: a pure
   * function of the query, so sharing one across calls on the same world changes no answer; a fresh one is made when absent.
   */
  reply(world: ScenarioWorld, memo?: ProbeMemo): Reply;
  /** The counts so far (a copy). */
  counts(): ReplyCounts;
}

/** A fresh per-world memo. */
export const newWorldMemo = (): ProbeMemo => new Map();

// ---- the engine objects this model reads, narrowed here ----
interface RMon {
  set: PokemonSet;
  hp: number;
  maxhp: number;
  status: string;
  fainted: boolean;
  boosts: Record<string, number>;
  species: { isMega?: boolean };
}
interface RSide {
  active: (RMon | null)[];
  sideConditions: Record<string, unknown>;
}
interface RBattle {
  sides: RSide[];
  field: { weather: string; terrain: string; pseudoWeather: Record<string, unknown> };
}
interface MoveData {
  id: string;
  target?: string;
}

const emptyCounts = (): ReplyCounts => ({ replies: 0, max_damage: 0, all_zero: 0, forced_slots: 0, candidates: 0, queries: 0, probe_runs: 0, unsupported_queries: 0, probe_refusals: 0 });

export function createReplyModel(options: ReplyModelOptions): ReplyModel {
  const { engine, vocabulary, config, mechanics, data_class } = options;
  const snapshot = options.snapshot ?? loadRulesSnapshot(engine.commit);
  const probe = options.probe ?? probeMechanics;
  const counts = emptyCounts();

  function reply(world: ScenarioWorld, memo: ProbeMemo = newWorldMemo()): Reply {
    const probeCounts = { queries: 0, probe_runs: 0, unsupported_queries: 0 };
    const view = damageView({ engine, vocabulary, config, v1: config, mechanics, snapshot, data_class, probe, counts: probeCounts, memo });
    const seat = world.battle.sides.findIndex((side) => side.id === world.seat);
    if (seat < 0) throw new ReplyModelError("the world has no side for its seat");
    const foeIndex = 1 - seat;
    const foeId = world.battle.sides[foeIndex]?.id as string;

    // A clone rebuilds every side's request from the state as written (the world's own requests are those of the start); the world is not changed.
    const clone = engine.sim.Battle.fromJSON(world.battle.toJSON());
    const request = (clone.sides[foeIndex] as { activeRequest: object | null }).activeRequest;
    if (request === null) throw new ReplyModelError("the world's opponent has no request");
    const decoded = decodeRequest(JSON.stringify(request));
    if (decoded.phase !== "move") throw new ReplyModelError("the world's opponent request is not a move request");
    const numbered = { ...decoded, request_seq: 1 };
    const legal = legalChoiceSet({
      request: numbered,
      showdown_commit: engine.commit,
      judge: new CloneJudge(engine, clone, foeId),
    });
    const { candidates } = enumerateCandidates(numbered);
    const open = candidates.filter((candidate: Candidate) => !candidate.pending);
    counts.candidates += open.length;

    const battle = world.battle as unknown as RBattle;
    const attackers = (battle.sides[foeIndex] as RSide).active;
    const defenders = (battle.sides[seat] as RSide).active;
    const body = decoded.request as { active?: { moves: MoveData[] }[] };
    const slots = body.active as { moves: MoveData[] }[];
    const dex = view.dex;

    // ---- the field as the world holds it, the attacker's side first ----
    const sideOf = (side: RSide) => ({
      aurora_veil: "auroraveil" in side.sideConditions,
      light_screen: "lightscreen" in side.sideConditions,
      reflect: "reflect" in side.sideConditions,
      tailwind: "tailwind" in side.sideConditions,
    });
    const state = {
      weather: battle.field.weather === "" ? null : battle.field.weather,
      terrain: battle.field.terrain === "" ? null : battle.field.terrain,
      trick_room: "trickroom" in battle.field.pseudoWeather,
      sides: [sideOf(battle.sides[foeIndex] as RSide), sideOf(battle.sides[seat] as RSide)] as [ReturnType<typeof sideOf>, ReturnType<typeof sideOf>],
    };

    const stage = (mon: RMon, id: string): number => mon.boosts[id] as number; // the engine keeps all seven stages
    const fighterOf = (mon: RMon): Fighter => ({
      paste: exportPaste(engine, [mon.set]),
      hp: mon.hp === mon.maxhp ? null : mon.hp,
      status: mon.status === "" ? null : mon.status,
      boosts: { atk: stage(mon, "atk"), def: stage(mon, "def"), spa: stage(mon, "spa"), spd: stage(mon, "spd"), spe: stage(mon, "spe") },
      mega: mon.species.isMega === true,
    });
    const fighters = new Map<RMon, Fighter>();
    const fighter = (mon: RMon): Fighter => {
      let found = fighters.get(mon);
      if (found === undefined) fighters.set(mon, (found = fighterOf(mon)));
      return found;
    };

    /** One move into one defender in the world's state: its value, 0 for what the probe cannot answer. */
    const against = (attacker: RMon, defender: RMon, move: string, mega: boolean): number => {
      const from = fighter(attacker);
      try {
        const result = view.query({
          kind: "damage",
          attacker: view.combatant({ ...from, mega: from.mega || mega }),
          defender: view.combatant(fighter(defender)),
          move,
          state,
        });
        if (result?.kind !== "computed" || result.answer.kind !== "damage") return 0;
        return damageValue(result.answer, config.scoring);
      } catch (error) {
        if (!(error instanceof ProbeRefusal)) throw error;
        counts.probe_refusals += 1;
        return 0;
      }
    };

    /** The foes (the player's living actives) a part hits: one by location (`loc` 1 is the last active slot), or every one. */
    const hit = (aim: "single" | "random" | "every", target: number | null): RMon[] => {
      if (aim === "single") {
        const found = defenders[defenders.length - (target as number)];
        return found != null && !found.fainted && found.hp > 0 ? [found] : [];
      }
      return defenders.filter((mon): mon is RMon => mon !== null && !mon.fainted && mon.hp > 0);
    };

    const parts = new Map<string, { eligible: boolean; value: number }>();
    const partOf = (slot: number, part: SlotPart): { eligible: boolean; value: number } => {
      if (part.kind === "pass") return { eligible: true, value: 0 };
      if (part.kind === "switch") return { eligible: false, value: 0 };
      const key = `${slot}/${part.index}/${part.target}/${part.mega}`;
      const known = parts.get(key);
      if (known !== undefined) return known;
      const attacker = attackers[slot] as RMon; // a candidate that moves names a living active and one of its moves
      const data = (slots[slot] as { moves: MoveData[] }).moves[part.index - 1] as MoveData;
      const kind = dex.moves.get(data.id);
      let found = { eligible: false, value: 0 };
      if (kind.exists && kind.category !== "Status") {
        // Eligible: a damaging move aimed at foes (one at an ally or self, and a lock with no target type, are not).
        const aim = foeTargetKind(data.target);
        if (aim === "random" || aim === "every" || (aim === "single" && part.target !== null && part.target > 0)) {
          const targets = hit(aim, part.target);
          const sum = targets.reduce((total, defender) => total + against(attacker, defender, data.id, part.mega), 0);
          found = { eligible: true, value: aim === "random" ? Math.floor(sum / Math.max(1, targets.length)) : sum };
        }
      }
      parts.set(key, found);
      return found;
    };

    const parsed = open.map((candidate) => ({ choice: candidate.choice, slots: parseJointChoice(candidate.choice) }));
    // A slot with no eligible part in any candidate is held to its first legal part in canonical order.
    const forced = new Map<number, string>();
    for (let slot = 0; slot < 2; slot += 1) {
      const texts = parsed.map((entry) => entry.choice.split(", ")[slot]).filter((text): text is string => text !== undefined);
      if (texts.length > 0 && !parsed.some((entry) => partOf(slot, entry.slots[slot] as SlotPart).eligible)) {
        forced.set(slot, texts[0] as string);
        counts.forced_slots += 1;
      }
    }

    let best: { choice: string; value: number } | null = null;
    for (const entry of parsed) {
      let value = 0;
      let allowed = true;
      for (const [slot, part] of entry.slots.entries()) {
        const held = forced.get(slot);
        if (held !== undefined) {
          if (entry.choice.split(", ")[slot] !== held) allowed = false;
          continue;
        }
        const found = partOf(slot, part);
        if (!found.eligible) allowed = false;
        value += found.value;
      }
      if (allowed && (best === null || value > best.value)) best = { choice: entry.choice, value };
    }

    counts.replies += 1;
    counts.queries += probeCounts.queries;
    counts.probe_runs += probeCounts.probe_runs;
    counts.unsupported_queries += probeCounts.unsupported_queries;
    if (best === null || best.value === 0) {
      counts.all_zero += 1;
      return { choice: actorRule(legal), kind: "all_zero", value: 0 };
    }
    counts.max_damage += 1;
    return { choice: best.choice, kind: "max_damage", value: best.value };
  }

  return { reply, counts: () => ({ ...counts }) };
}
