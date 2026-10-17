import { type Json, requestHash } from "./request-identity.ts";

export type JsonObject = { [key: string]: Json };

/** The decision a `|request|` asks for. `wait` asks for nothing. */
export type RequestPhase = "preview" | "move" | "forceSwitch" | "wait";

export interface DecodedRequest {
  phase: RequestPhase;
  player: string;
  /** Present only on the Showdown server path (ADR-004 r2). */
  rqid?: number;
  request_hash: string;
  /** Preview only: Pokémon brought and how many must be chosen. */
  preview?: { teamSize: number; maxChosenTeamSize: number };
  request: JsonObject;
}

export class RequestDecodeError extends Error {
  override name = "RequestDecodeError";
}

function isObject(value: Json | undefined): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function phaseOf(request: JsonObject): RequestPhase {
  if (request.teamPreview === true) return "preview";
  if (request.wait === true) return "wait";
  if (Array.isArray(request.forceSwitch)) return "forceSwitch";
  if (Array.isArray(request.active)) return "move";
  throw new RequestDecodeError("request has no recognisable phase");
}

/** Decode the JSON body of one `|request|` line. */
export function decodeRequest(body: string): DecodedRequest {
  const request = JSON.parse(body) as Json;
  if (!isObject(request)) {
    throw new RequestDecodeError("request is not a JSON object");
  }
  const side = request.side;
  if (!isObject(side) || typeof side.id !== "string" || !Array.isArray(side.pokemon)) {
    throw new RequestDecodeError("request has no side");
  }
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
