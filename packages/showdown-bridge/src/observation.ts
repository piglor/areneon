// The ObservationBuilder (ADR-008 §5): the only constructor of a player's
// observation. It is engine-free and imports no Node built-in, so the advisor
// extension can bundle it: its inputs are contracts and a branded
// PlayerChannel, it hashes through an injected Sha256 port, and it never
// receives the Battle, the omniscient log or hidden verdicts. It refuses a
// team that is not the player's own (checked against their own request) and
// any field the closed observation schema does not name. Full schema
// validation needs Ajv: `observe` (player-channel.ts) adds it on Node.

import previewSchema from "@areneon/schema/contracts/v1/preview-observation.schema.json" with { type: "json" };
import type { BattleObservation } from "@areneon/schema/types/v1/battle-observation.ts";
import type { DecisionContext } from "@areneon/schema/types/v1/decision-context.ts";
import type { PlayerChannelLog } from "@areneon/schema/types/v1/player-channel-log.ts";
import type { PreviewObservation } from "@areneon/schema/types/v1/preview-observation.ts";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import type { VersionManifest } from "@areneon/schema/types/v1/version-manifest.ts";
import { canonicalJson, type Json } from "./canonical-json.ts";

export type { BattleObservation, PreviewObservation };
export type Observation = PreviewObservation | BattleObservation;

/** SHA-256 of `bytes` as lowercase hex: Node's `createHash` or WebCrypto. */
export type Sha256 = (bytes: Uint8Array) => Promise<string>;

/**
 * One player's own channel, checked by an adapter (`playerChannel`): it
 * matches PlayerChannelLog and every `|request|` in it is that player's and
 * of the engine's shape. Only an adapter makes one; the omniscient log can
 * never be passed as one.
 */
export type PlayerChannel = PlayerChannelLog & { readonly __brand: "PlayerChannel" };

export class ObservationError extends Error {
  override name = "ObservationError";
}

export interface ObservationInput {
  channel: PlayerChannel;
  /** The player's own team. */
  team: PrivateTeamSpec;
  manifest: VersionManifest;
  sha256: Sha256;
  decision_context?: DecisionContext;
}

const REQUEST = "|request|";

// Every object schema in an observation names its required keys.
type Schema = { $ref?: string; properties?: Record<string, Schema>; required: string[]; items?: Schema };
// Both observation schemas share one shape and differ only in `kind`'s const.
const SHAPE = previewSchema as unknown as Schema & { $defs: Record<string, Schema> };

/**
 * Closure only, without Ajv: every object holds no key its schema does not
 * name and every required one, all the way down. A field smuggled in through
 * the team, manifest or decision context fails here.
 */
function assertClosed(value: unknown, schema: Schema, at: string): void {
  const resolved = schema.$ref ? (SHAPE.$defs[schema.$ref.slice("#/$defs/".length)] as Schema) : schema;
  const { properties, items } = resolved;
  if (properties) {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      throw new ObservationError(`${at} must be an object`);
    }
    const object = value as Record<string, unknown>;
    for (const key of Object.keys(object)) {
      const sub = properties[key];
      if (!sub) throw new ObservationError(`${at} holds ${key}, which the observation schema does not name`);
      assertClosed(object[key], sub, `${at}/${key}`);
    }
    for (const key of resolved.required) {
      if (!(key in object)) throw new ObservationError(`${at} lacks ${key}`);
    }
  } else if (items) {
    if (!Array.isArray(value)) throw new ObservationError(`${at} must be an array`);
    value.forEach((item, i) => assertClosed(item, items, `${at}/${i}`));
  }
}

/**
 * The team must be this player's: every Pokémon in their own request is one
 * of its sets, by the name the engine shows (nickname, else species). An
 * opponent's team, or another of the player's teams, fails.
 */
function assertOwnTeam(team: PrivateTeamSpec, request: { [key: string]: Json }): void {
  const names = team.team.map((set) => set.name || set.species);
  const side = request.side as { pokemon: { ident: string }[] };
  for (const { ident } of side.pokemon) {
    const name = ident.slice(ident.indexOf(": ") + 2);
    const at = names.indexOf(name);
    if (at < 0) throw new ObservationError(`the team is not this player's: it has no ${name}`);
    names.splice(at, 1);
  }
}

/**
 * The observation for the last `|request|` in the channel: the decision that
 * request asks for. Its identity is computed exactly as `decodeRequest`
 * computes it: `request_seq` counts every request so far, wait requests
 * included, and `request_hash` is the SHA-256 of the canonical request
 * without `rqid`.
 */
export async function buildObservation({
  channel,
  team,
  manifest,
  sha256,
  decision_context,
}: ObservationInput): Promise<Observation> {
  const at = channel.lines.findLastIndex((line) => line.startsWith(REQUEST));
  if (at < 0) throw new ObservationError("the channel holds no request to observe");
  const request = JSON.parse((channel.lines[at] as string).slice(REQUEST.length)) as { [key: string]: Json };
  if (request.wait === true) throw new ObservationError("the last request is a wait request: there is no decision");
  assertOwnTeam(team, request);
  const { rqid, ...hashed } = request;
  const identity = {
    request_seq: channel.lines.slice(0, at + 1).filter((line) => line.startsWith(REQUEST)).length,
    request_hash: await sha256(new TextEncoder().encode(canonicalJson(hashed))),
    ...(typeof rqid === "number" ? { rqid } : {}),
  };
  const common: Omit<BattleObservation, "kind"> = {
    schema_version: 1,
    data_class:
      team.data_class === "private" || channel.data_class === "private" ? "private" : "synthetic",
    player: channel.player,
    manifest,
    identity,
    // Non-empty: it ends with the request line found above.
    log: channel.lines.slice(0, at + 1) as [string, ...string[]],
    team,
  };
  if (decision_context !== undefined) common.decision_context = decision_context;
  const observation: Observation =
    request.teamPreview === true ? { ...common, kind: "preview" } : { ...common, kind: "battle" };
  assertClosed(observation, SHAPE, "observation");
  return observation;
}
