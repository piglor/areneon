import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { CloneJudge, type Verdict } from "./choice-judge.ts";
import { enumerateCandidates } from "./choices.ts";
import type { Engine, PokemonSet, SimBattle, SimSide } from "./engine.ts";
import { resolveFormat, type VersionManifest } from "./formats.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "./golden.ts";
import { type DecodedRequest, decodeRequest, type RequestPhase } from "./request.ts";
import type { FormatRulesRecord } from "./rules.ts";
import type { Player } from "./runner.ts";
import { validateTeam } from "./teams.ts";

/** The golden choice corpus (ADR-006 rule 7; CC0, see fixtures/LICENSE). */
export const CHOICES_DIR = join(GOLDEN_DIR, "choices");

type Active = { [key: string]: unknown; moves: { target?: string }[] };
const actives = (r: DecodedRequest): Active[] =>
  r.phase === "move" ? (r.request.active as Active[]) : [];
const team = (r: DecodedRequest): { condition: string }[] =>
  (r.request.side as { pokemon: { condition: string }[] }).pokemon;
const fainted = (mon: { condition: string }): boolean => mon.condition.endsWith(" fnt");
const bench = (r: DecodedRequest): number => team(r).filter((m, i) => i >= 2 && !fainted(m)).length;
const forced = (r: DecodedRequest): number =>
  r.phase === "forceSwitch" ? (r.request.forceSwitch as boolean[]).filter(Boolean).length : 0;

/**
 * The situations the corpus must show (ADR-006 rule 7), each a test on the
 * case's request. Every kind needs at least one case; what the engine
 * cannot produce for this format is a gap record instead (`rules/gaps/`).
 */
export const CASE_KINDS = {
  preview: (r: DecodedRequest) => r.phase === "preview",
  /** A plain move request; together the cases cover every target type. */
  targets: (r: DecodedRequest) => r.phase === "move",
  mega: (r) => actives(r).filter((a) => a.canMegaEvo === true).length === 1,
  "mega-both": (r) => actives(r).filter((a) => a.canMegaEvo === true).length === 2,
  "force-switch-one": (r) => forced(r) === 1 && bench(r) >= 1,
  "force-switch-two": (r) => forced(r) === 2 && bench(r) >= 2,
  "force-switch-short-bench": (r) => forced(r) === 2 && bench(r) === 1,
  /** Both slots may switch and the bench has two: the same-slot conflict. */
  "double-switch": (r) =>
    actives(r).length === 2 && actives(r).every((a) => a.trapped !== true) && bench(r) >= 2,
  trapped: (r) => actives(r).some((a) => a.trapped === true),
  "maybe-trapped": (r) => actives(r).some((a) => a.maybeTrapped === true),
  "maybe-disabled": (r) =>
    actives(r).some((a) => a.maybeDisabled === true && a.maybeLocked === true),
  "fainted-no-replacement": (r) => r.phase === "move" && bench(r) === 0 && team(r).slice(0, 2).some(fainted),
  /** Locked into a move the request sends without a target (e.g. Outrage). */
  "locked-move": (r) => actives(r).some((a) => a.moves.length === 1 && a.moves[0]?.target === undefined),
} satisfies Record<string, (request: DecodedRequest) => boolean>;

export type CaseKind = keyof typeof CASE_KINDS;

/** What a case is: a seeded battle replayed to one player's request. */
export interface ChoiceCaseSpec {
  id: string;
  kind: CaseKind;
  purpose: string;
  seed: string;
  teams: Record<Player, PokemonSet[]>;
  player: Player;
  /** Every choice both sides made before the request, in order. */
  log: { player: Player; choice: string }[];
}

export interface CorpusCandidate {
  choice: string;
  pending: boolean;
  /**
   * The engine's verdict, judged on a clone. For a pending candidate it is
   * hidden state: an evaluation artifact, never a policy input.
   */
  legal: boolean;
  error: string;
}

/** One committed corpus file. */
export interface ChoiceCase extends ChoiceCaseSpec {
  provenance: string;
  manifest: VersionManifest;
  format_rules: FormatRulesRecord;
  phase: RequestPhase;
  request_hash: string;
  /** In `enumerateCandidates` order, so a changed order is a diff. */
  candidates: CorpusCandidate[];
}

export class ChoiceCaseError extends Error {
  override name = "ChoiceCaseError";
}

export interface ReplayedCase {
  battle: SimBattle;
  request: DecodedRequest;
}

/** Replay a case's log on the pinned engine and decode the player's request. */
export function replayCase(engine: Engine, spec: ChoiceCaseSpec): ReplayedCase {
  const format = resolveFormat(engine, GOLDEN_FORMAT);
  const battle = new engine.sim.Battle({ formatid: format.manifest.format_id, seed: spec.seed });
  for (const player of ["p1", "p2"] as const) {
    const verdict = validateTeam(engine, format, spec.teams[player]);
    if (!verdict.valid) throw new ChoiceCaseError(`${spec.id}: ${player} team: ${verdict.problems.join("; ")}`);
    battle.setPlayer(player, { name: player, team: engine.sim.Teams.pack(spec.teams[player]) });
  }
  for (const { player, choice } of spec.log) {
    if (!battle.choose(player, choice)) {
      throw new ChoiceCaseError(`${spec.id}: the engine refused logged ${player} choice ${choice}`);
    }
  }
  const side = battle.sides.find((s) => s.id === spec.player) as SimSide;
  if (battle.ended || side.activeRequest === null || side.isChoiceDone()) {
    throw new ChoiceCaseError(`${spec.id}: ${spec.player} has no choice to make`);
  }
  return { battle, request: decodeRequest(JSON.stringify(side.activeRequest)) };
}

/** Replay a case and record its candidates with the engine's verdicts. */
export function captureCase(engine: Engine, spec: ChoiceCaseSpec): ChoiceCase {
  const { battle, request } = replayCase(engine, spec);
  if (!CASE_KINDS[spec.kind](request)) {
    throw new ChoiceCaseError(`${spec.id}: the request is not a ${spec.kind} case`);
  }
  const { candidates } = enumerateCandidates(request);
  const verdicts = new CloneJudge(engine, battle, spec.player).judge(
    request.request_hash,
    candidates.map((c) => c.choice),
  );
  const format = resolveFormat(engine, GOLDEN_FORMAT);
  return {
    ...spec,
    provenance: "synthetic: Areneon choice corpus, seeded search over synthetic CC0 teams",
    manifest: format.manifest,
    format_rules: format.rules,
    phase: request.phase,
    request_hash: request.request_hash,
    candidates: candidates.map((c, i) => {
      const { legal, error } = verdicts[i] as Verdict;
      return { ...c, legal, error };
    }),
  };
}

/**
 * Every syntactically plausible choice string for one side, knowing only
 * the phase and team size (ADR-006 option C). No request knowledge, so it
 * is an independent oracle for the enumerator.
 */
export function choiceUniverse(phase: RequestPhase, teamSize: number): string[] {
  if (phase === "preview") {
    const out: string[] = [];
    const walk = (prefix: number[]): void => {
      if (prefix.length > 0) out.push(`team ${prefix.join("")}`);
      if (prefix.length === teamSize) return;
      for (let i = 1; i <= teamSize; i++) if (!prefix.includes(i)) walk([...prefix, i]);
    };
    walk([]);
    return out;
  }
  const slot = ["pass"];
  for (let m = 1; m <= 4; m++) {
    for (const target of ["", " 1", " 2", " -1", " -2"]) {
      for (const event of ["", " mega", " megax", " megay", " terastallize"]) {
        slot.push(`move ${m}${target}${event}`);
      }
    }
  }
  for (let s = 1; s <= teamSize; s++) slot.push(`switch ${s}`);
  return slot.flatMap((a) => slot.map((b) => `${a}, ${b}`));
}

/**
 * Engine-accepted actions missing from `offered` (default: the enumerator's
 * candidates). Every string of the universe is judged, and each accepted
 * action, normalised by the engine's own parse, must be among the actions
 * of `offered`. Returns one universe spelling per missed action.
 */
export function missedActions(
  engine: Engine,
  replayed: ReplayedCase,
  player: Player,
  offered: string[] = enumerateCandidates(replayed.request).candidates.map((c) => c.choice),
): string[] {
  const { battle, request } = replayed;
  const judge = new CloneJudge(engine, battle, player);
  const actions = (choices: string[]): Map<string, string> => {
    const out = new Map<string, string>();
    for (const v of judge.judge(request.request_hash, choices)) {
      if (v.action !== undefined && !out.has(v.action)) out.set(v.action, v.choice);
    }
    return out;
  };
  const enumerated = actions(offered);
  const universe = actions(choiceUniverse(request.phase, team(request).length));
  return [...universe].filter(([action]) => !enumerated.has(action)).map(([, choice]) => choice);
}

/** The committed corpus, by file name. */
export function loadCorpus(dir: string = CHOICES_DIR): Record<string, ChoiceCase> {
  const out: Record<string, ChoiceCase> = {};
  for (const file of readdirSync(dir).filter((f) => f.endsWith(".json")).sort()) {
    out[file] = JSON.parse(readFileSync(join(dir, file), "utf8")) as ChoiceCase;
  }
  return out;
}

/** The spec part of a committed case, as the capture script and CI replay it. */
export function caseSpec(c: ChoiceCase): ChoiceCaseSpec {
  const { id, kind, purpose, seed, teams, player, log } = c;
  return { id, kind, purpose, seed, teams, player, log };
}

