import { ajvFor } from "@areneon/schema/ajv";
import engineRequestSchema from "@areneon/schema/contracts/v1/engine-request.schema.json" with { type: "json" };
import type { Json } from "./canonical-json.ts";
import { nodeSha256Hex } from "./request-identity.ts";
import { requestIdentityBytes } from "./request-bytes.ts";

export type JsonObject = { [key: string]: Json };

/** The decision a `|request|` asks for. `wait` asks for nothing. */
export type RequestPhase = "preview" | "move" | "forceSwitch" | "wait";

export interface DecodedRequest {
  phase: RequestPhase;
  player: string;
  /** Present only on the Showdown server path (ADR-008 §4). */
  rqid?: number;
  request_hash: string;
  /** Preview only: Pokémon brought and how many must be chosen. */
  preview?: { teamSize: number; maxChosenTeamSize: number };
  request: JsonObject;
}

/** A request validated and decoded, not yet hashed (`decodeRequest` adds `request_hash`). */
export type ParsedRequest = Omit<DecodedRequest, "request_hash"> & {
  /** The bytes `request_hash` is the SHA-256 of (`requestIdentityBytes`). */
  identity_bytes: Uint8Array;
};

export class RequestDecodeError extends Error {
  override name = "RequestDecodeError";
}

// The engine's own wire shape, closed (ADR-008 §1b): a key the pinned engine
// does not send, or one no supported format uses, fails here, before hashing.
const validEngineRequest = ajvFor("engine-mirror").compile(engineRequestSchema);

// One marker per engine request variant (`sim/side.ts` ChoiceRequest). The
// schema is one closed object, so it cannot refuse two markers at once.
const PHASE_MARKERS: readonly [RequestPhase, (request: JsonObject) => boolean][] = [
  ["preview", (r) => r.teamPreview === true],
  ["wait", (r) => r.wait === true],
  ["forceSwitch", (r) => Array.isArray(r.forceSwitch)],
  ["move", (r) => Array.isArray(r.active)],
];

function phaseOf(request: JsonObject): RequestPhase {
  const phases = PHASE_MARKERS.filter(([, marks]) => marks(request)).map(([phase]) => phase);
  if (phases.length === 0) throw new RequestDecodeError("request has no recognisable phase");
  if (phases.length > 1) throw new RequestDecodeError(`request marks more than one phase: ${phases.join(", ")}`);
  return phases[0] as RequestPhase;
}

/**
 * Parse the JSON body of one `|request|` line. Pure: it validates the body
 * against the engine-mirror `EngineRequest` schema (ADR-008 §4), decodes the
 * phase and returns the bytes a request identity hashes, and it hashes
 * nothing, so it runs wherever a `Sha256` port does.
 */
export function parseRequest(body: string): ParsedRequest {
  const parsed = JSON.parse(body) as Json;
  if (!validEngineRequest(parsed)) {
    const why = (validEngineRequest.errors ?? [])
      .map((e) => {
        const key = (e.params as { additionalProperty?: string }).additionalProperty;
        return `${e.instancePath || "/"} ${e.message}${key === undefined ? "" : ` (${key})`}`;
      })
      .join("; ");
    throw new RequestDecodeError(`request does not match the engine's request shape: ${why}`);
  }
  const request = parsed as JsonObject;
  const side = request.side as { id: string; pokemon: Json[] };
  const result: ParsedRequest = {
    phase: phaseOf(request),
    player: side.id,
    identity_bytes: requestIdentityBytes(request),
    request,
  };
  if (typeof request.rqid === "number") result.rqid = request.rqid;
  if (result.phase === "preview") {
    const max = request.maxChosenTeamSize;
    if (typeof max !== "number") {
      throw new RequestDecodeError("preview request has no maxChosenTeamSize");
    }
    result.preview = { teamSize: side.pokemon.length, maxChosenTeamSize: max };
  }
  return result;
}

/** `parseRequest`, then `request_hash` on Node (the A1 and A2 Node path). */
export function decodeRequest(body: string): DecodedRequest {
  const { identity_bytes, ...parsed } = parseRequest(body);
  return { ...parsed, request_hash: nodeSha256Hex(identity_bytes) };
}
