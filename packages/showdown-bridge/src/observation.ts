// The ObservationBuilder (ADR-008 §5): the only constructor of a player's
// observation. It is engine-free and imports no Node built-in, so it runs
// wherever WebCrypto does: its inputs are contracts and a branded
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
import type { Json } from "./canonical-json.ts";
import { hashRequest, type Sha256 } from "./request-bytes.ts";

export type { BattleObservation, PreviewObservation };
export type Observation = PreviewObservation | BattleObservation;

// The Sha256 port is defined beside the identity bytes it hashes.
export type { Sha256 };

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
type Schema = {
  $ref?: string;
  type?: string;
  properties?: Record<string, Schema>;
  required: string[];
  items?: Schema;
};
const LEAF: Readonly<Record<string, (value: unknown) => boolean>> = {
  string: (v) => typeof v === "string",
  integer: (v) => Number.isSafeInteger(v),
};
// Both observation schemas share one shape and differ only in `kind`'s const.
const SHAPE = previewSchema as unknown as Schema & { $defs: Record<string, Schema> };

/**
 * Closure only, without Ajv: every object holds no key its schema does not
 * name and every required one, all the way down. A field smuggled in through
 * the team, manifest or decision context fails here.
 */
function assertClosed(value: unknown, schema: Schema, at: string): void {
  const resolved = schema.$ref ? (SHAPE.$defs[schema.$ref.slice("#/$defs/".length)] as Schema) : schema;
  const { properties, items, type } = resolved;
  const leaf = type === undefined ? undefined : LEAF[type];
  if (leaf && !leaf(value)) throw new ObservationError(`${at} must be a ${type}`);
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

/** Showdown's `toID`: lowercase letters and digits only. */
const toID = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

interface RequestMon {
  ident: string;
  details: string;
  item: string;
  baseAbility: string;
  moves: string[];
}

/**
 * The team must be this player's own, set for set. It is checked against the
 * player's Team Preview request, the first in the channel: no Mega, Transform,
 * Trick or Knock Off has changed a forme, move or item yet. Each Pokémon there
 * must match one set by species, item, ability and its moves in any order
 * (and by nickname when the set has one). A mirror team with other items or
 * moves fails, as does a channel that does not start at Team Preview.
 * Residual risk (#743): a mirror differing only in EVs, IVs, nature or level
 * passes, since telling them apart needs the engine's stat calculation.
 */
function assertOwnTeam(team: PrivateTeamSpec, lines: readonly string[]): void {
  const first = lines.find((line) => line.startsWith(REQUEST));
  const preview = JSON.parse((first as string).slice(REQUEST.length)) as { teamPreview?: true; side: { pokemon: RequestMon[] } };
  if (preview.teamPreview !== true) {
    throw new ObservationError("the channel does not start at Team Preview, so the team cannot be checked (a channel after a rejoin is joined to the earlier one with continueChannel first)");
  }
  const key = (species: string, item: string, ability: string, moves: readonly string[]): string =>
    [toID(species), toID(item), toID(ability), ...moves.map(toID).sort()].join("|");
  const sets = team.team.map((set) => ({ name: set.name, key: key(set.species, set.item, set.ability, set.moves) }));
  for (const mon of preview.side.pokemon) {
    const name = mon.ident.slice(mon.ident.indexOf(": ") + 2);
    const wanted = key(mon.details.split(",")[0] as string, mon.item, mon.baseAbility, mon.moves);
    const at = sets.findIndex((set) => set.key === wanted && (set.name === "" || set.name === name));
    if (at < 0) throw new ObservationError(`the team is not this player's: no set matches ${name}`);
    sets.splice(at, 1);
  }
}

/**
 * The observation for the last `|request|` in the channel: the decision that
 * request asks for. Its identity is computed exactly as `decodeRequest`
 * computes it (`request-bytes.ts` is the one definition of the hashed
 * bytes): `request_seq` counts every request so far, wait requests included,
 * and `request_hash` is the SHA-256 of the canonical request without `rqid`.
 */
export async function buildObservation({
  channel,
  team,
  manifest,
  sha256,
  decision_context,
}: ObservationInput): Promise<Observation> {
  // Copies, checked as copied: no getter, toJSON or later mutation of an
  // input can change what was checked.
  try {
    [team, manifest, decision_context] = structuredClone([team, manifest, decision_context]);
  } catch (error) {
    throw new ObservationError(`an input is not plain data: ${(error as Error).message}`);
  }
  const at = channel.lines.findLastIndex((line) => line.startsWith(REQUEST));
  if (at < 0) throw new ObservationError("the channel holds no request to observe");
  const request = JSON.parse((channel.lines[at] as string).slice(REQUEST.length)) as { [key: string]: Json };
  if (request.wait === true) throw new ObservationError("the last request is a wait request: there is no decision");
  const { rqid } = request;
  const identity = {
    request_seq: channel.lines.slice(0, at + 1).filter((line) => line.startsWith(REQUEST)).length,
    request_hash: await hashRequest(request, sha256),
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
  // Closure first: the team check reads the sets' fields.
  assertClosed(observation, SHAPE, "observation");
  assertOwnTeam(observation.team, channel.lines);
  return observation;
}
