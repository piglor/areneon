// The hypothetical Team Preview battle (ADR-029 sections 3 and 7, findings N5 and N6). A Team Preview solve asks what the
// opponent would be shown, and what our side would see at its first move request, for each pair of team orders. The answer
// is the engine's own: one in-process battle of the TARGET format with our six as brought and the six of a sampled world,
// started once to Team Preview, serialised once, and cloned per (our order, opponent order).
//
// Everything built from it is hypothetical. The channels and observations are opaque brands (`HypotheticalChannel`,
// `HypotheticalObservation`): they hold none of a `PlayerChannel`'s or an `Observation`'s fields, so no Episode writer,
// `recordedChannel`, training loader, `buildObservation` caller or real `PolicyQuery` can take one (type tests), and the
// text behind them is reachable only through this module. The worker is reached through `evaluateHypothetical`, which
// sends the existing `evaluate` query (the wire contract is unchanged: an evaluate query carries an observation and the
// request's candidates, and the worker cannot tell the origin; the guarantee is on the sending side).
//
// What a solve may read is `PreviewSolveInput` and nothing else: no running battle, `BattleRecord`, `OmniscientLog`, hidden
// verdicts, battle seed or opponent team. The world's sets come in as a separate argument of the start (the prior's draw,
// ours), never from the game.
//
// Reuse, not a new fork: the clones are made by `cloneWorld` and played by `playChoices` (`search-turn.ts`, a reviewed
// `Battle.fromJSON` site) on a battle this module built. Engine seed: `SHA-256(JCS({purpose: "preview-solve-engine/1",
// stream, world}))`, as ADR-029 section 3 writes it (`world` is the world's index, as in ADR-026's `search-engine/1`).

import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { jcs } from "./canonical-json.ts";
import { enumerateCandidates } from "./choices.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { resolveFormat, type VersionManifest } from "./formats.ts";
import { observe, playerChannel } from "./player-channel.ts";
import type { Observation } from "./observation.ts";
import type { EvaluateQueryInput, Evaluation } from "./policy-protocol.ts";
import type { EvaluationClient, PolicyClient } from "./policy-worker.ts";
import { type DecodedRequest, decodeRequest } from "./request.ts";
import type { RequestPhase } from "./request.ts";
import { sha256Hex } from "./rules.ts";
import type { RulesSnapshot } from "./rules.ts";
import { InvalidTeamError, type Player } from "./runner.ts";
import { cloneWorld, type Outcome, playChoices, playerView, type TurnBattle } from "./search-turn.ts";
import type { DataClass } from "./team-class.ts";
import { validateTeam } from "./teams.ts";

/** Bumped when the hypothetical battle is started, cut or observed differently for the same input (part of the solver's identity, ADR-029 section 4). */
export const HYPOTHETICAL_BATTLE_FORMAT = 1;

/** The engine seed's purpose (ADR-029 section 3). */
export const HYPOTHETICAL_ENGINE_PURPOSE = "preview-solve-engine/1";

const PLAYERS: readonly Player[] = ["p1", "p2"];
const REQUEST = "|request|";
const hex = (bytes: Uint8Array): string => Buffer.from(bytes).toString("hex");

/** The battle's engine seed (`sodium,<32 hex>`): the same for every pair of orders of a world. */
export const hypotheticalEngineSeed = (stream: Uint8Array, world: number): string =>
  `sodium,${sha256Hex(jcs({ purpose: HYPOTHETICAL_ENGINE_PURPOSE, stream: hex(stream), world })).slice(0, 32)}`;

/** The hypothetical battle cannot be made or read: a team the validator refuses is an `InvalidTeamError`; the rest are this. */
export class HypotheticalError extends Error {
  override name = "HypotheticalError";
}

declare const hypothetical: unique symbol;

/**
 * One seat's channel in a hypothetical battle. Opaque: it has no `lines`, no `schema_version` and no `PlayerChannel` brand,
 * so neither `playerChannel`, `recordedChannel`, `continueChannel`, `buildObservation` nor an Episode writer accepts it.
 */
export interface HypotheticalChannel {
  readonly [hypothetical]: "channel";
  readonly player: Player;
}

/**
 * An observation built from a hypothetical channel. Opaque: it is not an `Observation`, so it cannot be a `PolicyQuery`'s,
 * a `TrainingExample`'s or an Episode's. `evaluateHypothetical` is the only way to a worker.
 */
export interface HypotheticalObservation {
  readonly [hypothetical]: "observation";
  readonly player: Player;
  readonly kind: Observation["kind"];
}

/**
 * A request of a hypothetical battle, summarised: its seat, phase and identity hash. Opaque: it is not a `DecodedRequest`
 * (no `side`, `legal` or body), so `enumerateCandidates`, the runner's actors and the loaders cannot take it.
 */
export interface HypotheticalRequest {
  readonly [hypothetical]: "request";
  readonly player: Player;
  readonly phase: RequestPhase;
  readonly request_hash: string;
}

/**
 * One engine-distinct joint choice of a hypothetical request (a `team ...` order at Team Preview). Opaque: it has no
 * `choice` field, so it is not a `Candidate`; `order` is the choice string to hand back to `play`.
 */
export interface HypotheticalCandidate {
  readonly [hypothetical]: "candidate";
  readonly order: string;
  readonly pending: boolean;
}

interface ChannelText {
  lines: readonly string[];
  data_class: DataClass;
}
interface ObservationText {
  observation: Observation;
  candidates: { choice: string; pending: boolean }[];
}
const channelTexts = new WeakMap<object, ChannelText>();
const observationTexts = new WeakMap<object, ObservationText>();

function brandChannel(player: Player, lines: readonly string[], data_class: DataClass): HypotheticalChannel {
  const channel = Object.freeze({ player }) as unknown as HypotheticalChannel;
  channelTexts.set(channel, { lines: Object.freeze([...lines]), data_class });
  return channel;
}

function textOf(channel: HypotheticalChannel): ChannelText {
  const text = channelTexts.get(channel);
  if (text === undefined) throw new HypotheticalError("not a hypothetical channel made by this module");
  return text;
}

function decodedRequest(channel: HypotheticalChannel): DecodedRequest {
  const { lines } = textOf(channel);
  const last = lines.findLast((line) => line.startsWith(REQUEST));
  if (last === undefined) throw new HypotheticalError("the hypothetical channel holds no request");
  return decodeRequest(last.slice(REQUEST.length));
}

const brandRequest = ({ player, phase, request_hash }: DecodedRequest): HypotheticalRequest =>
  Object.freeze({ player, phase, request_hash }) as unknown as HypotheticalRequest;

/** The request the channel ends in (its last `|request|` line), as a hypothetical request. */
export const hypotheticalRequest = (channel: HypotheticalChannel): HypotheticalRequest => brandRequest(decodedRequest(channel));

/** The candidates (engine-distinct joint choices) of the request a hypothetical channel ends in, in canonical order. */
export const hypotheticalCandidates = (channel: HypotheticalChannel): HypotheticalCandidate[] =>
  enumerateCandidates(decodedRequest(channel)).candidates.map(({ choice, pending }) => Object.freeze({ order: choice, pending }) as unknown as HypotheticalCandidate);

/** The team spec an observation is checked against: the engine sets as brought (one cast, here). */
const teamSpec = (data_class: DataClass, team: readonly Readonly<PokemonSet>[]): PrivateTeamSpec =>
  ({ schema_version: 1, data_class, team: [...team] }) as unknown as PrivateTeamSpec;

function deepFreeze<T extends object>(value: T): T {
  for (const inner of Object.values(value)) if (typeof inner === "object" && inner !== null) deepFreeze(inner);
  return Object.freeze(value);
}

/**
 * What one Team Preview solve reads (ADR-029 section 7): the target format, our seat, our own team as brought, our own channel
 * up to the Team Preview request and the harness stream. No field for the running battle, a `BattleRecord`, an `OmniscientLog`,
 * hidden verdicts, the battle seed or the opponent's team (a type test pins the key set).
 */
export interface PreviewSolveInput {
  format_id: string;
  seat: Player;
  team: readonly Readonly<PokemonSet>[];
  channel: readonly string[];
  /** The harness's per-decision stream (32 bytes); it reaches only `hypotheticalEngineSeed`. */
  stream: Uint8Array;
}

/** What starts one hypothetical battle: the solve's own inputs, one world's six sets and the world's index. */
export interface HypotheticalStartInput {
  engine: Engine;
  format_id: string;
  /** Our seat in the real game; the hypothetical battle gives the same seat to our six. */
  seat: Player;
  /** Our six as brought. */
  own_team: readonly Readonly<PokemonSet>[];
  /** The world's six sets (`sampleTeam`'s draw, mapped to engine sets by the caller), in preview order. */
  world_team: readonly Readonly<PokemonSet>[];
  stream: Uint8Array;
  /** The world's index in the decision (the engine seed's `world`). */
  world: number;
  /** The class of the own team; the hypothetical channels carry it. */
  data_class: DataClass;
  /** The rules snapshot to resolve the format against; default reads and validates `rules/`. */
  snapshot?: RulesSnapshot;
}

/** The result of playing one pair of orders to our first request. */
export type HypotheticalTurn =
  | { kind: "request"; channel: HypotheticalChannel; request: HypotheticalRequest }
  | { kind: "ended"; outcome: Outcome }
  | { kind: "refused"; reason: string };

export interface HypotheticalPreview {
  readonly manifest: VersionManifest;
  readonly seat: Player;
  readonly foe: Player;
  /** Our seat's channel to its Team Preview request: what our real channel holds at the same point. */
  readonly own_channel: HypotheticalChannel;
  /** The opponent seat's own channel to its Team Preview request, as the format shows a player. */
  readonly foe_channel: HypotheticalChannel;
  /** The opponent's Team Preview request. */
  readonly foe_request: HypotheticalRequest;
  /** Our order and the opponent's, each a `team ...` choice, run on a clone of the started battle to our seat's next request. */
  play(own_choice: string, foe_choice: string): HypotheticalTurn;
  /** An observation of a hypothetical channel of this battle, with the team of the seat it belongs to. */
  observation(channel: HypotheticalChannel): Promise<HypotheticalObservation>;
}

/** Start the battle once to Team Preview. Both teams are validated first, on copies, as `runBattle` does. */
export function startHypothetical(input: HypotheticalStartInput): HypotheticalPreview {
  const { engine, format_id, seat, own_team, world_team, stream, world, data_class, snapshot } = input;
  if (!PLAYERS.includes(seat)) throw new HypotheticalError("seat must be p1 or p2");
  const foe: Player = seat === "p1" ? "p2" : "p1";
  const resolved = resolveFormat(engine, format_id, snapshot);
  const teams: Record<Player, PokemonSet[]> = { [seat]: structuredClone(own_team) as PokemonSet[], [foe]: structuredClone(world_team) as PokemonSet[] } as Record<Player, PokemonSet[]>;
  for (const player of PLAYERS) {
    const verdict = validateTeam(engine, resolved, teams[player]);
    if (!verdict.valid) throw new InvalidTeamError(player, verdict.problems);
  }

  const seed = hypotheticalEngineSeed(stream, world);
  const lines: Record<Player, string[]> = { p1: [], p2: [] };
  const text = (type: string, data: string | string[]): void => {
    if (type === "update") {
      const joined = (data as string[]).join("\n");
      lines.p1.push(...playerView(joined, 1));
      lines.p2.push(...playerView(joined, 2));
    } else if (type === "sideupdate") {
      const raw = data as string;
      const at = raw.indexOf("\n");
      lines[raw.slice(0, at) as Player].push(raw.slice(at + 1));
    }
  };
  const battle = new engine.sim.Battle({ formatid: format_id, seed, send: text }) as TurnBattle;
  for (const player of PLAYERS) battle.setPlayer(player, { name: player, team: engine.sim.Teams.pack(teams[player]) });
  battle.sendUpdates();

  for (const player of PLAYERS) {
    const request = lines[player].findLast((line) => line.startsWith(REQUEST));
    if (request === undefined || (JSON.parse(request.slice(REQUEST.length)) as { teamPreview?: true }).teamPreview !== true) {
      throw new HypotheticalError(`${player} was not asked for a Team Preview order: the format has no Team Preview`);
    }
  }
  const serialised = JSON.stringify(battle.toJSON());
  const ownLines = lines[seat];
  const own_channel = brandChannel(seat, ownLines, data_class);
  const foe_channel = brandChannel(foe, lines[foe], data_class);
  const teamOf = (player: Player): readonly Readonly<PokemonSet>[] => (player === seat ? own_team : world_team);

  return {
    manifest: resolved.manifest,
    seat,
    foe,
    own_channel,
    foe_channel,
    foe_request: brandRequest(decodedRequest(foe_channel)),
    play(own_choice, foe_choice) {
      const turn = playChoices(cloneWorld(engine, serialised, seed), seat, { own: own_choice, foe: foe_choice });
      if (turn.refused !== null) return { kind: "refused", reason: turn.refused };
      if (turn.ended !== null) return { kind: "ended", outcome: turn.ended };
      if (turn.request === null) return { kind: "refused", reason: "the engine sent the seat no request" };
      const channel = brandChannel(seat, [...ownLines, ...turn.lines], data_class);
      return { kind: "request", channel, request: brandRequest(decodeRequest(turn.request)) };
    },
    async observation(channel) {
      const { lines: channelLines, data_class: channelClass } = textOf(channel);
      const real = playerChannel({ schema_version: 1, data_class: channelClass, player: channel.player, lines: channelLines });
      const built = await observe({ channel: real, team: teamSpec(channelClass, teamOf(channel.player)), manifest: resolved.manifest });
      // A copy, frozen: the query the worker client validates and sends is the one this module made, whatever a caller does after.
      const observation = deepFreeze(structuredClone(built));
      const candidates = enumerateCandidates(decodedRequest(channel)).candidates.map(({ choice, pending }) => ({ choice, pending }));
      const branded = Object.freeze({ player: channel.player, kind: observation.kind }) as unknown as HypotheticalObservation;
      observationTexts.set(branded, { observation, candidates });
      return branded;
    },
  };
}

/**
 * The dedicated evaluate entry for hypothetical observations (ADR-029 section 3): the policy's prior over the request's
 * candidates and its value, from the existing `evaluate` query. The wire contract is unchanged; `HypotheticalObservation`
 * is accepted by this function alone. Returns the candidates in the order of `prior_ppm`.
 */
export async function evaluateHypothetical(
  client: EvaluationClient & Pick<PolicyClient, "featureSet">,
  battle: string,
  observation: HypotheticalObservation,
  policy_id: string,
): Promise<{ evaluation: Evaluation; candidates: HypotheticalCandidate[] }> {
  const held = observationTexts.get(observation);
  if (held === undefined) throw new HypotheticalError("not a hypothetical observation made by this module");
  // A pair that reads candidate features would need their rows at every hypothetical request (ADR-029 section 3): refused, as in ADR-026.
  if (client.featureSet !== undefined && client.featureSet !== null) throw new HypotheticalError("features_pair: a pair that reads candidate features is not supported by the Team Preview solve");
  const evaluation = await client.evaluate(battle, {
    schema_version: 1,
    identity: held.observation.identity,
    observation: held.observation,
    candidates: held.candidates,
    sampling: { kind: "evaluate" },
    policy_id,
  } as EvaluateQueryInput);
  return { evaluation, candidates: held.candidates.map(({ choice, pending }) => Object.freeze({ order: choice, pending }) as unknown as HypotheticalCandidate) };
}
