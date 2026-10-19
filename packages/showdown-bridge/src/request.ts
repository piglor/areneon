import { ajvFor } from "@areneon/schema/ajv";
import engineRequestSchema from "@areneon/schema/contracts/v1/engine-request.schema.json" with { type: "json" };
import { type Json, requestHash } from "./request-identity.ts";

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
 * Decode the JSON body of one `|request|` line. The body is validated against
 * the engine-mirror `EngineRequest` schema before `request_hash` is computed
 * (ADR-008 §4), so only a request of the pinned engine's shape is hashed.
 */
export function decodeRequest(body: string): DecodedRequest {
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
  const decoded: DecodedRequest = {
    phase: phaseOf(request),
    player: side.id,
    request_hash: requestHash(request),
    request,
  };
  if (typeof request.rqid === "number") decoded.rqid = request.rqid;
  if (decoded.phase === "preview") {
    const max = request.maxChosenTeamSize;
    if (typeof max !== "number") {
      throw new RequestDecodeError("preview request has no maxChosenTeamSize");
    }
    decoded.preview = { teamSize: side.pokemon.length, maxChosenTeamSize: max };
  }
  return decoded;
}
