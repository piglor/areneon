// The search's opponent worlds (ADR-026 sections 2 to 4, slice B; ADR-025 section 4): ADR-025's one sampler,
// asked by index, mapped onto the world builder's input and redrawn until a world passes V1 to V5.
//
// There is no sampler here. The search's `OpponentTeamSampler` is ADR-025's `sample(belief, stream)` over the
// per-index stream of `sampleWorlds`: attempt r of world i (0 <= r < R) is sampler index `i * R + r`, whose
// stream the model derives under `opponent-world-index/1`. This module only joins that draw, the mapping of an
// `OpponentWorld` onto `WorldInput.opponent` and the builder's checks, and counts what happens. The bridge does
// not import the opponent model (dependency direction, ADR-025 section 1; the importer image copies the
// bridge's workspace dependencies): the model arrives through the structural `WorldSamplePort`, which the
// composition root in `packages/team-dataset` implements over a belief and `createSampler`.
//
// Two redraw bounds, counted apart: the model's `max_legality_redraws` (inside one draw, until every set passes
// `validateSet`; `legality_redraws` here) and this module's R (the sampler indices tried for one world until it
// passes V1 to V5; `redraws`, `attempts`). A world that no attempt makes valid is refused and counted by the
// check and class of its last refusal; a decision is searched only when at least `min_valid` of its `worlds`
// worlds are valid (W_min, ADR-026 N-I), else the pair's own choice is played and the decision is counted.
//
// The sampler path sees the player's view only: the world input without its opponent (`WorldRequest`), the
// belief's input (the reader's opponent half and flags) and the 32-byte search stream the harness derived. It
// has no field for a battle seed, the live battle, a `BattleRecord`, an `OmniscientLog`, hidden verdicts or the
// opponent's team (a type test).

import type { PublicFlag, PublicSide } from "@areneon/public-state";
import { jcs } from "./canonical-json.ts";
import { BELIEF_DEFECTS, refusalCode, type WorldLike } from "./damage-aware-worlds.ts";
import type { WorldSetLike } from "./damage-aware-assumed-set.ts";
import { createTrappingCheck, type TrappingCheckPort } from "./trapping-check.ts";
import type { DamageAwareConfig } from "./damage-aware-config.ts";
import type { Engine, PokemonSet, StatTable } from "./engine.ts";
import { defaultIvs } from "./engine.ts";
import { sha256Hex } from "./rules.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";
import type { ScenarioWorld, WorldBuilder, WorldInput, WorldOpponentMember, WorldRefusal } from "./world-builder.ts";

/** The purpose string of the per-decision search stream (ADR-026 section 4). */
export const SEARCH_PURPOSE = "search/1";

/**
 * How a shown opponent percentage becomes hit points in a searched world. The Champions display is
 * `floor(100 * hp / max)`; `ceil` inverts it for every maximum of 100 or more (`ceil(p * max / 100)` is at least
 * `p * max / 100` and less than that plus 1, so its display is `p`), while damage-aware v1's `nearest` does not
 * for every maximum (check V3 would refuse those worlds). The v1 record is unchanged; the search says `ceil` here
 * and V3 stays the independent check (a test shows the inversion over every maximum the format allows).
 */
export const SEARCH_HP_ROUNDING: DamageAwareConfig["hp_rounding"] = "ceil";

/** The prior arms of the search (ADR-026 section 3). */
export const PRIOR_KINDS = ["generator-matched", "role-sets"] as const;
export type PriorKind = (typeof PRIOR_KINDS)[number];

/** The label every result of the primary arm carries. */
export const GENERATOR_MATCHED_LABEL = "belief matched to the synthetic generator: an upper bound for this evaluation";
/** The label every result of the secondary arm carries. */
export const ROLE_SETS_LABEL = "belief from the role-set table, mismatched to uniform teams: moves outside the roles have probability zero";

/** The evidence classes of a decision, first match wins (the coverage report's rows). */
export const EVIDENCE_CLASSES = ["maybe_trapped", "volatile", "timed_condition", "unseen_brought", "rest"] as const;
export type EvidenceClass = (typeof EVIDENCE_CLASSES)[number];

/** What the search asks the opponent model for one decision, after the belief is computed. */
export interface WorldSource {
  /** Sampler index `index` on `stream`: ADR-025's `sample(belief, worldStream(stream, index))`. */
  draw(stream: Uint8Array, index: number): { readonly world: WorldLike; readonly redraws: number };
}

export type WorldSourceResult = { readonly kind: "source"; readonly source: WorldSource } | { readonly kind: "refused"; readonly code: string };

/**
 * What the search needs of the opponent model: the belief of the reader's opponent half, then draws by index.
 * A belief refusal is a `refused` result with the model's code; a defect of the run's setup is thrown.
 */
export interface WorldSamplePort {
  /** `belief_identity_hash` (ADR-025 section 5). */
  readonly identity_hash: string;
  readonly prior_hash: string;
  readonly belief_config_hash: string;
  readonly prior_format_id: string;
  readonly prior_kind: PriorKind;
  /** The label every result of this arm carries. */
  readonly prior_label: string;
  open(input: { readonly opponent: PublicSide; readonly flags: readonly PublicFlag[]; readonly trapping?: TrappingCheckPort | null }): WorldSourceResult;
}

/** The world input without the opponent: the player's own view. The sampler fills the opponent in. */
export type WorldRequest = Omit<WorldInput, "opponent">;

/** The 32-byte per-decision search stream: SHA-256 of the JCS of `{seed, game, player, policy_id, request_seq, purpose: "search/1"}`, derived by the harness. */
export function searchStream(parts: { readonly seed: string; readonly game: number; readonly player: string; readonly policy_id: string; readonly request_seq: number }): Uint8Array {
  const hex = sha256Hex(jcs({ game: parts.game, player: parts.player, policy_id: parts.policy_id, purpose: SEARCH_PURPOSE, request_seq: parts.request_seq, seed: parts.seed }));
  return Uint8Array.from(hex.match(/../g) as string[], (byte) => parseInt(byte, 16));
}

/** The class of a decision's evidence: a trapping flag of the own request, a volatile in play, a timed condition, a brought opponent Pokemon not yet seen, else the rest. */
export function evidenceClass(input: WorldRequest): EvidenceClass {
  const { request, state } = input;
  if ((request.active ?? []).some((slot) => slot.trapped === true || slot.maybeTrapped === true)) return "maybe_trapped";
  const inPlay = [...state.own.roster, ...state.opponent.roster].filter((mon) => mon.active_slot !== null);
  if (inPlay.some((mon) => mon.volatiles.length > 0)) return "volatile";
  const sides = [state.own.conditions, state.opponent.conditions];
  if (state.weather !== null || state.terrain !== null || state.field_conditions.length > 0 || sides.some((side) => Object.keys(side).length > 0)) return "timed_condition";
  const teamsize = state.opponent.teamsize ?? 0;
  if (state.opponent.roster.filter((mon) => mon.brought === true).length < teamsize) return "unseen_brought";
  return "rest";
}

/** A sampled world does not fit the team vocabulary or the engine's names: a defect of the setup, not a refusal. */
export class WorldMappingError extends Error {
  override name = "WorldMappingError";
}

/**
 * Map an `OpponentWorld` onto the builder's opponent: each brought slot is a member naming its roster entry (the
 * preview position) and a whole set in the engine's names. A Mega slot is the preview species holding its stone
 * (the form Team Preview shows); an ability the preview species cannot hold (an evolved Mega's) becomes the
 * species' first listed one, as the builder runs the Evolution itself. The builder's V1 judges the result again.
 */
export function mapWorld(context: { readonly engine: Engine; readonly vocabulary: TeamVocabulary }, formatId: string, world: WorldLike): WorldOpponentMember[] {
  const { engine, vocabulary } = context;
  const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(formatId).mod);
  const natures = new Map(vocabulary.natures.map((nature) => [nature.id, nature.name]));
  const members: WorldOpponentMember[] = [];
  for (const slot of world.slots) {
    if (!slot.brought) continue;
    const set: WorldSetLike | null = slot.set;
    if (set === null) throw new WorldMappingError("a brought slot has no set");
    const entry = vocabulary.roster.find((r) => r.id === set.species);
    const nature = natures.get(set.nature);
    if (entry === undefined || nature === undefined) throw new WorldMappingError("a sampled set names a species or nature outside the team vocabulary");
    const mega = vocabulary.roster.find((r) => r.id === set.forme)?.required_item !== undefined && set.forme !== set.species;
    const ability = entry.abilities.includes(set.ability) ? set.ability : (entry.abilities[0] as string);
    const built: PokemonSet = {
      name: "",
      species: dex.species.get(mega ? set.species : set.forme).name,
      item: set.item === null ? "" : dex.items.get(set.item).name,
      ability: dex.abilities.get(ability).name,
      moves: set.moves.map((move) => dex.moves.get(move).name),
      nature,
      gender: set.gender ?? "",
      evs: { ...set.stat_points } as StatTable,
      ivs: defaultIvs(),
      level: set.level,
    };
    members.push({ roster_index: slot.index, set: built });
  }
  return members;
}

export interface WorldSamplerSettings {
  /** N: worlds asked for per decision. */
  readonly worlds: number;
  /** R: sampler indices tried per world. */
  readonly attempts: number;
  /** W_min: valid worlds a decision needs to be searched. */
  readonly min_valid: number;
}

export interface WorldSamplerContext {
  readonly engine: Engine;
  readonly vocabulary: TeamVocabulary;
  readonly format_id: string;
  readonly port: WorldSamplePort;
  readonly builder: WorldBuilder;
  readonly settings: WorldSamplerSettings;
  /**
   * Condition the belief on the trapping flags of the player's own request (ticket #1127): the scenario format id
   * the engine check starts its battles in (`checkScenario`). Absent: no trapping evidence, as before.
   */
  readonly trapping?: { readonly scenario_id: string };
}

/** One valid world of a decision and the sampler index that produced it. */
export interface ValidWorld {
  /** The world's number i, 0 <= i < N. */
  readonly world: number;
  /** The attempt r that passed, 0 <= r < R. */
  readonly attempt: number;
  /** The sampler index `i * R + r`. */
  readonly index: number;
  readonly built: ScenarioWorld;
}

export interface RefusedWorld {
  readonly world: number;
  /** Attempts made for this world (1 to R). */
  readonly attempts: number;
  /** The last refusal, or null when the last attempt's draw itself failed (`draw_code`). */
  readonly refusal: WorldRefusal | null;
  readonly draw_code: string | null;
}

export type DecisionWorlds =
  | { readonly kind: "refused"; readonly evidence: EvidenceClass; readonly code: string; readonly prior_label: string }
  | {
      readonly kind: "worlds";
      readonly evidence: EvidenceClass;
      readonly prior_label: string;
      /** At least `min_valid` worlds are valid: the decision is searched. */
      readonly searched: boolean;
      readonly valid: readonly ValidWorld[];
      readonly refused: readonly RefusedWorld[];
    };

export interface WorldSamplerCounts {
  readonly prior_kind: PriorKind;
  readonly prior_label: string;
  readonly identity_hash: string;
  readonly settings: WorldSamplerSettings;
  decisions: number;
  /** Decisions with at least W_min valid worlds. */
  searched: number;
  /** Decisions whose belief was computed and which had fewer valid worlds than W_min. */
  below_min_valid: number;
  /** Decisions the belief refused (no world could be drawn), by the model's code. */
  belief_refusals: Record<string, number>;
  /** Worlds asked for and valid worlds. */
  /** Decisions whose own request gave the belief trapping evidence (an informative own active), and those among them that showed `maybeTrapped`. */
  trapping_checks: number;
  trapping_seen: number;
  worlds_asked: number;
  worlds_valid: number;
  /** Worlds no attempt made valid. */
  worlds_refused: number;
  /** Builder calls made. */
  attempts: number;
  /** Sampler indices drawn after the first attempt of a world (the search's redraws). */
  redraws: number;
  /** ADR-025's redraws inside the draws (until every set passes `validateSet`), summed. */
  legality_redraws: number;
  /** Attempts whose draw itself was refused by the model (its legality bound), by code. */
  draw_refusals: Record<string, number>;
  /** Attempts the builder refused and the search redrew, by `check:class`. */
  redrawn: Record<string, number>;
  /** Worlds refused, by the last refusal's code and by `check:class` (`-` is none). */
  refusals: Record<string, number>;
  refusal_check_class: Record<string, number>;
  /** Decisions by valid-world count (index 0 to N). */
  valid_histogram: number[];
  /** Coverage by evidence class. */
  by_evidence: Record<EvidenceClass, { decisions: number; searched: number; worlds_valid: number; belief_refused: number }>;
}

export interface WorldSampler {
  /** Draw and build the worlds of one decision on the harness's search stream. */
  sample(request: WorldRequest, stream: Uint8Array): DecisionWorlds;
  counts(): WorldSamplerCounts;
}

const bump = (table: Record<string, number>, key: string): void => {
  table[key] = (table[key] ?? 0) + 1;
};

export function createWorldSampler(context: WorldSamplerContext): WorldSampler {
  const { engine, vocabulary, format_id, port, builder, settings } = context;
  for (const [name, value, least] of [["worlds", settings.worlds, 1], ["attempts", settings.attempts, 1], ["min_valid", settings.min_valid, 0]] as const) {
    if (!Number.isSafeInteger(value) || value < least) throw new RangeError(`${name} is an integer of at least ${least}`);
  }
  if (settings.min_valid > settings.worlds) throw new RangeError("min_valid is at most worlds");
  const counts: WorldSamplerCounts = {
    prior_kind: port.prior_kind,
    prior_label: port.prior_label,
    identity_hash: port.identity_hash,
    settings: { ...settings },
    decisions: 0,
    searched: 0,
    below_min_valid: 0,
    belief_refusals: {},
    trapping_checks: 0,
    trapping_seen: 0,
    worlds_asked: 0,
    worlds_valid: 0,
    worlds_refused: 0,
    attempts: 0,
    redraws: 0,
    legality_redraws: 0,
    draw_refusals: {},
    redrawn: {},
    refusals: {},
    refusal_check_class: {},
    valid_histogram: Array.from({ length: settings.worlds + 1 }, () => 0),
    by_evidence: Object.fromEntries(EVIDENCE_CLASSES.map((name) => [name, { decisions: 0, searched: 0, worlds_valid: 0, belief_refused: 0 }])) as WorldSamplerCounts["by_evidence"],
  };
  const keyOf = (refusal: WorldRefusal): string => `${refusal.check ?? "-"}:${refusal.state_class ?? "-"}`;

  return {
    sample(request, stream) {
      if (!(stream instanceof Uint8Array) || stream.length !== 32) throw new RangeError("the search stream is 32 bytes");
      counts.decisions += 1;
      const evidence = evidenceClass(request);
      const row = counts.by_evidence[evidence];
      row.decisions += 1;
      const check = context.trapping === undefined ? null : createTrappingCheck({ engine, vocabulary, scenario_id: context.trapping.scenario_id }, { request: request.request, team: request.team, state: request.state });
      if (check !== null) {
        counts.trapping_checks += 1;
        if (check.observed.some((level) => level === 1)) counts.trapping_seen += 1;
      }
      const opened = port.open({ opponent: request.state.opponent, flags: request.flags, trapping: check });
      if (opened.kind === "refused") {
        if (BELIEF_DEFECTS.has(opened.code)) throw new WorldMappingError(`the belief refused the setup: ${opened.code}`);
        bump(counts.belief_refusals, opened.code);
        row.belief_refused += 1;
        return { kind: "refused", evidence, code: opened.code, prior_label: port.prior_label };
      }
      const valid: ValidWorld[] = [];
      const refused: RefusedWorld[] = [];
      const used = new Set<number>();
      for (let i = 0; i < settings.worlds; i += 1) {
        counts.worlds_asked += 1;
        let last: { refusal: WorldRefusal | null; draw_code: string | null } = { refusal: null, draw_code: null };
        let made = 0;
        let found: ValidWorld | null = null;
        for (let r = 0; r < settings.attempts; r += 1) {
          const index = i * settings.attempts + r;
          if (used.has(index)) throw new WorldMappingError("a sampler index would be used twice");
          used.add(index);
          made += 1;
          counts.attempts += 1;
          if (r > 0) counts.redraws += 1;
          let drawn: ReturnType<WorldSource["draw"]>;
          try {
            drawn = opened.source.draw(stream, index);
          } catch (error) {
            const code = refusalCode(error);
            if (code === null || BELIEF_DEFECTS.has(code)) throw error;
            bump(counts.draw_refusals, code);
            last = { refusal: null, draw_code: code };
            continue;
          }
          counts.legality_redraws += drawn.redraws;
          const result = builder.build({ ...request, opponent: { members: mapWorld({ engine, vocabulary }, format_id, drawn.world) } });
          if (result.kind === "world") {
            found = { world: i, attempt: r, index, built: result.world };
            break;
          }
          last = { refusal: result.refusal, draw_code: null };
          if (!result.refusal.redraw) break;
          if (r + 1 < settings.attempts) bump(counts.redrawn, keyOf(result.refusal));
        }
        if (found !== null) {
          valid.push(found);
          counts.worlds_valid += 1;
          row.worlds_valid += 1;
        } else {
          counts.worlds_refused += 1;
          bump(counts.refusals, last.refusal?.code ?? `draw:${last.draw_code}`);
          if (last.refusal !== null) bump(counts.refusal_check_class, keyOf(last.refusal));
          refused.push({ world: i, attempts: made, ...last });
        }
      }
      const searched = valid.length >= settings.min_valid;
      counts.valid_histogram[valid.length] = (counts.valid_histogram[valid.length] ?? 0) + 1;
      if (searched) {
        counts.searched += 1;
        row.searched += 1;
      } else counts.below_min_valid += 1;
      return { kind: "worlds", evidence, prior_label: port.prior_label, searched, valid, refused };
    },
    counts: () => structuredClone(counts),
  };
}
