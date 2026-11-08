/* eslint-disable @typescript-eslint/no-explicit-any */
// S0 oracle rollouts (ADR-026 section 8 S0 measure 3, #1147; #1308). TEST TOOLING: this file, like the fork oracle it builds on
// (`showdown-bridge/test/fidelity/oracle.ts`), lives under test/ and is named by `fidelity-isolation.test.ts`: the fork of a live
// battle and the real opponent policy exist only here. Nothing in it reaches a policy input or any shipped code path.
//
// An oracle value of a candidate is the mean score of R rollouts from the test-only fork: the fork (`Battle.fromJSON` of the live
// battle at the decision) plays the candidate for the seat, the opponent's REAL policy answers from its own request and channel,
// and the pair (the seat's continuation policy) and the opponent play on to the end of the game or a turn cap. Rollout r reseeds
// the fork's generator with a seed derived from (battle, seat, turn, r) and nothing else, so every candidate of a decision meets the
// same draws (common random numbers) and a rerun gives the same values.

import {
  type Actor,
  CloneJudge,
  decodeRequest,
  type Engine,
  hpKoDifferential,
  legalChoiceSet,
  outcomeScore,
  type PokemonSet,
  type RequestRecord,
} from "@areneon/showdown-bridge";
import { jcs } from "../../../showdown-bridge/src/canonical-json.ts";
import { sha256Hex } from "../../../showdown-bridge/src/rules.ts";
import { activeRequest, playChoices, type TurnBattle } from "../../../showdown-bridge/src/search-turn.ts";
import { type Decision, revive } from "../../../showdown-bridge/test/fidelity/oracle.ts";

type Player = "p1" | "p2";
const other = (p: Player): Player => (p === "p1" ? "p2" : "p1");

/** The policies a rollout plays with: a fresh actor per rollout for the seat (the pair) and for the opponent (its real policy). */
export interface RolloutPolicies {
  own(rollout: number): Actor;
  foe(rollout: number): Actor;
}

export interface RolloutSpec {
  engine: Engine;
  decision: Decision;
  /** The opponent's own channel up to and including its move request of the decision's turn (`foeChannelAt`). */
  foe_channel: readonly string[];
  teams: Record<Player, readonly PokemonSet[]>;
  policies: RolloutPolicies;
  /** Turns played after the first before a rollout is cut and scored by the hit-point differential (counted as `capped`). */
  max_turns: number;
}

export interface RolloutResult {
  /** The seat's score in ppm (win 1,000,000, tie 500,000, loss 0; a capped rollout the hit-point differential), or null when the engine refused a choice. */
  score_ppm: number | null;
  turns: number;
  capped: boolean;
}

/** The opponent's channel through its first move request after `|turn|<turn>`: what its actor was given at that decision. */
export function foeChannelAt(channel: readonly string[], turn: number): string[] {
  const at = channel.lastIndexOf(`|turn|${turn}`);
  if (at < 0) throw new Error(`the opponent's channel has no turn ${turn}`);
  const request = channel.findIndex((line, i) => i > at && line.startsWith("|request|"));
  if (request < 0) throw new Error(`the opponent's channel has no request after turn ${turn}`);
  return channel.slice(0, request + 1);
}

/** The rollout's engine seed: `sodium,` + 32 hex of SHA-256 over the JCS of the battle, seat, turn and rollout (no candidate: common random numbers). */
export const rolloutSeed = (battle_id: string, player: Player, turn: number, rollout: number): string => `sodium,${sha256Hex(jcs({ battle: battle_id, player, purpose: "s0-oracle-rollout/1", rollout, turn })).slice(0, 32)}`;

const requestCount = (channel: readonly string[]): number => channel.filter((line) => line.startsWith("|request|")).length;

/** The actor request for a side's request text: decoded, numbered by the channel, with the engine-judged legal set; null for a wait. */
function requestFor(engine: Engine, battle: TurnBattle, player: Player, text: string, seq: number): RequestRecord | null {
  const decoded = decodeRequest(text);
  if (decoded.phase === "wait") return null;
  const request: RequestRecord = { ...decoded, request_seq: seq };
  request.legal = legalChoiceSet({ request, showdown_commit: engine.commit, judge: new CloneJudge(engine, battle as never, player) });
  return request;
}

/** Play one rollout of `candidate` (the seat's choice at the decision) with the policies of `spec`. */
export async function oracleRollout(spec: RolloutSpec, candidate: string, rollout: number): Promise<RolloutResult> {
  const { engine, decision: d, policies } = spec;
  const seat = d.player;
  const foe = other(seat);
  const battle = revive(engine, d.fork) as TurnBattle;
  (battle as any).prng = new engine.sim.PRNG(rolloutSeed(d.battle_id, seat, d.turn, rollout));
  const own = policies.own(rollout);
  const foeActor = policies.foe(rollout);
  const channels: Record<Player, string[]> = { [seat]: [...d.channel], [foe]: [...spec.foe_channel] } as Record<Player, string[]>;
  const ask = async (side: Player, text: string | null): Promise<string | null> => {
    if (text === null) return null;
    const request = requestFor(engine, battle, side, text, requestCount(channels[side]));
    if (request === null) return null;
    return (side === seat ? own : foeActor)(request, { player: side, channel: [...channels[side]], team: spec.teams[side] });
  };
  const foeText = JSON.stringify(activeRequest(battle, foe));
  let turn = playChoices(battle, seat, { own: candidate, foe: await ask(foe, foeText) }, true);
  for (let played = 1; ; played += 1) {
    if (turn.refused !== null) return { score_ppm: null, turns: played, capped: false };
    channels[seat].push(...turn.lines);
    channels[foe].push(...(turn.foe as { lines: string[] }).lines);
    if (turn.ended !== null) return { score_ppm: outcomeScore(turn.ended), turns: played, capped: false };
    if (played > spec.max_turns) return { score_ppm: hpKoDifferential(battle, seat), turns: played, capped: true };
    const ownChoice = await ask(seat, turn.request);
    const foeChoice = await ask(foe, (turn.foe as { request: string | null }).request);
    if (ownChoice === null && foeChoice === null) return { score_ppm: null, turns: played, capped: false };
    turn = playChoices(battle, seat, { own: ownChoice, foe: foeChoice }, true);
  }
}

export interface OracleValues {
  /** `scores[c][r]`: candidate c's score in rollout r. */
  scores: number[][];
  capped: number;
}

/**
 * The oracle values of `candidates`: R rollouts each. A refused rollout (the engine took a choice back) would leave a hole, so a
 * candidate with one is an error here: the tool does not average over what it could not play.
 */
export async function oracleValues(spec: RolloutSpec, candidates: readonly string[], rollouts: number): Promise<OracleValues> {
  const scores: number[][] = [];
  let capped = 0;
  for (const candidate of candidates) {
    const row: number[] = [];
    for (let r = 0; r < rollouts; r += 1) {
      const result = await oracleRollout(spec, candidate, r);
      if (result.score_ppm === null) throw new Error("the engine refused a choice in an oracle rollout");
      if (result.capped) capped += 1;
      row.push(result.score_ppm);
    }
    scores.push(row);
  }
  return { scores, capped };
}
