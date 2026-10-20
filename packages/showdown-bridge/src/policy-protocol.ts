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
  return query;
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
