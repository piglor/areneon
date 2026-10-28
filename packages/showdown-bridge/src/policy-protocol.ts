// The policy worker's wire protocol (ADR-010 §2, ADR-008 §3): the contracts it
// speaks, their validators, the reader rules the profile cannot say, and the
// framing. One JCS JSON object per line, UTF-8, `\n`-terminated. The worker
// (`ml/areneon_ml/worker`) implements the same rules in Python.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import battleSchema from "@areneon/schema/contracts/v1/battle-observation.schema.json" with { type: "json" };
import handshakeSchema from "@areneon/schema/contracts/v1/policy-handshake.schema.json" with { type: "json" };
import querySchema from "@areneon/schema/contracts/v1/policy-query.schema.json" with { type: "json" };
import replySchema from "@areneon/schema/contracts/v1/policy-reply.schema.json" with { type: "json" };
import previewSchema from "@areneon/schema/contracts/v1/preview-observation.schema.json" with { type: "json" };
import recommendationSchema from "@areneon/schema/contracts/v1/recommendation.schema.json" with { type: "json" };
import type { ContractVersion, PolicyHandshake } from "@areneon/schema/types/v1/policy-handshake.ts";
import type { PolicyQuery } from "@areneon/schema/types/v1/policy-query.ts";
import type { PolicyReply } from "@areneon/schema/types/v1/policy-reply.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { featureRowsProblem } from "./feature-rows.ts";

export type { PolicyHandshake, PolicyQuery, PolicyReply };

/** A query as the actor builds it; the client assigns the `query_id`. */
export type PolicyQueryInput = Omit<PolicyQuery, "query_id">;

/** The message does not match its contract, or breaks one of its reader rules. */
export class PolicyProtocolError extends Error {
  override name = "PolicyProtocolError";
}

const contract = ajvFor("contract");
const validQuery = contract.compile(querySchema);
const validReply = contract.compile(replySchema);
const validHandshake = contract.compile(handshakeSchema);

// The contracts both sides must agree on (the handshake schema's name enum).
const SCHEMAS = [battleSchema, querySchema, replySchema, previewSchema, recommendationSchema];

/**
 * The contracts this side speaks, sorted by name: each with its
 * `schema_version` and the SHA-256 of its schema file's JCS bytes.
 */
export function contractVersions(): ContractVersion[] {
  return SCHEMAS.map((schema) => ({
    name: schema.title as ContractVersion["name"],
    schema_version: (schema.properties.schema_version as { const: number }).const,
    schema_sha256: createHash("sha256").update(jcs(schema as unknown as Json)).digest("hex"),
  })).sort((a, b) => (a.name < b.name ? -1 : 1));
}

const describe = (errors: unknown): string => JSON.stringify(errors);

/** `value` as a PolicyQuery, or a `PolicyProtocolError`: the schema, then the reader rules. */
export function readPolicyQuery(value: unknown): PolicyQuery {
  if (!validQuery(value)) throw new PolicyProtocolError(`not a PolicyQuery: ${describe(validQuery.errors)}`);
  const query = value as unknown as PolicyQuery;
  if (jcs(query.identity as unknown as Json) !== jcs(query.observation.identity as unknown as Json)) {
    throw new PolicyProtocolError("the query's identity differs from its observation's identity");
  }
  const seen = new Set<string>();
  for (const { choice } of query.candidates) {
    if (seen.has(choice)) throw new PolicyProtocolError(`the candidate "${choice}" is listed twice`);
    seen.add(choice);
  }
  if (query.sampling.kind === "sample" && query.sampling.temperature_ppm === FULL_TEMPERATURE_PPM) {
    throw new PolicyProtocolError("temperature_ppm 1000000 is the default and must be omitted");
  }
  if (query.candidate_features !== undefined) {
    const problem = featureRowsProblem(query.candidate_features);
    if (problem !== null) throw new PolicyProtocolError(problem);
  }
  return query;
}

/** Temperature 1 in millionths: the default, which a query never writes (ADR-023 §1). */
export const FULL_TEMPERATURE_PPM = 1_000_000;

/** How a sampled policy is played beyond temperature 1 (ADR-023 §1); both parts optional. */
export interface PlaySettings {
  /** 1 to 999,999; absent is temperature 1. */
  temperature_ppm?: number;
  truncation?: { kind: "threshold"; min_ppm: number };
}

/**
 * The `policy:` id of a frozen pair played in `mode` (ADR-010 §2): SHA-256 of
 * `JCS({preview, battle, mode, temperature, precision})`. Play settings enter the preimage only
 * when set (`temperature_ppm` replaces `temperature`; `truncation` is added), so every earlier id
 * is unchanged. A pair that reads a candidate feature set (ADR-022 section 6) adds `features: <feature_set_hash>`;
 * `features` is absent for every other pair, whose id is unchanged. The Python
 * `areneon_ml.loop.records.policy_id` computes the same value.
 */
export function policyId(pair: { preview: string; battle: string }, mode: "sample" | "argmax", play: PlaySettings = {}, features?: string): string {
  const body: Record<string, Json> = { preview: pair.preview, battle: pair.battle, mode, temperature: 1, precision: "float32" };
  if (features !== undefined) {
    if (!/^[0-9a-f]{64}$/.test(features)) throw new PolicyProtocolError("features is a feature_set_hash: 64 lowercase hex digits");
    body.features = features;
  }
  if (play.temperature_ppm !== undefined || play.truncation !== undefined) {
    if (mode !== "sample") throw new PolicyProtocolError("a temperature or a truncation applies to sample mode only");
    if (play.temperature_ppm !== undefined) {
      if (!Number.isInteger(play.temperature_ppm) || play.temperature_ppm < 1 || play.temperature_ppm >= FULL_TEMPERATURE_PPM) {
        throw new PolicyProtocolError("temperature_ppm must be 1 to 999999; the default is omitted");
      }
      delete body.temperature;
      body.temperature_ppm = play.temperature_ppm;
    }
    if (play.truncation !== undefined) {
      const { kind, min_ppm } = play.truncation;
      if (kind !== "threshold" || !Number.isInteger(min_ppm) || min_ppm < 1 || min_ppm > 999_999) {
        throw new PolicyProtocolError("an unknown truncation, or min_ppm outside 1 to 999999");
      }
      body.truncation = { kind, min_ppm };
    }
  }
  return `policy:${createHash("sha256").update(jcs(body)).digest("hex")}`;
}

/** `value` as a PolicyReply, or a `PolicyProtocolError`. */
export function readPolicyReply(value: unknown): PolicyReply {
  if (!validReply(value)) throw new PolicyProtocolError(`not a PolicyReply: ${describe(validReply.errors)}`);
  return value as unknown as PolicyReply;
}

/** `value` as a PolicyHandshake message, or a `PolicyProtocolError`. */
export function readHandshake(value: unknown): PolicyHandshake {
  if (!validHandshake(value)) throw new PolicyProtocolError(`not a PolicyHandshake: ${describe(validHandshake.errors)}`);
  const handshake = value as unknown as PolicyHandshake;
  const { message } = handshake;
  const names = message.contracts.map((c) => c.name);
  if (names.some((name, i) => i > 0 && (names[i - 1] as string) >= name)) {
    throw new PolicyProtocolError("the handshake's contracts must be sorted by name, each once");
  }
  if (message.kind === "ready") {
    const roles = message.checkpoints.map((c) => c.role);
    if (roles.some((role, i) => i > 0 && (roles[i - 1] as string) >= role)) {
      throw new PolicyProtocolError("the handshake's checkpoints must be sorted by role, each once");
    }
  }
  return handshake;
}

/**
 * One frame: the JCS bytes of `value` and `\n`. Refused if the line (without
 * its newline) is longer than `maxLineBytes`, the limit both sides apply.
 */
export function encodeFrame(value: unknown, maxLineBytes: number): Buffer {
  const line = Buffer.from(jcs(value as Json), "utf8");
  if (line.length > maxLineBytes) {
    throw new PolicyProtocolError(`a frame of ${line.length} bytes exceeds the maximum line length ${maxLineBytes}`);
  }
  return Buffer.concat([line, Buffer.from("\n")]);
}
