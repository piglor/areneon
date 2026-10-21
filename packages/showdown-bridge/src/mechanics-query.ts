// The MechanicsQuery (ADR-016 section 2) and its privacy boundary (section 5).
// `readMechanicsQuery` is the only constructor of a `ParsedMechanicsQuery`:
// the probe takes nothing else, so a Battle, a BattleRecord, an omniscient log,
// hidden verdicts or a PlayerChannel is a compile error, and at run time the
// closed schema refuses such a value (a type test and a schema test show both).

import { ajvFor } from "@areneon/schema/ajv";
import querySchema from "@areneon/schema/contracts/v1/mechanics-query.schema.json" with { type: "json" };
import type { MechanicsQuery } from "@areneon/schema/types/v1/mechanics-query.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { sha256Hex } from "./rules.ts";

export type { MechanicsQuery };

/** A query that passed the schema; only `readMechanicsQuery` makes one. */
export type ParsedMechanicsQuery = MechanicsQuery & { readonly __brand: "ParsedMechanicsQuery" };

/** A document that is not a MechanicsQuery: it has no `query_hash`, so it gets no result. */
export class MechanicsQueryError extends Error {
  override name = "MechanicsQueryError";
}

const validQuery = ajvFor("contract").compile(querySchema);

/** Read a query. The schema's own messages name fields, never the values read. */
export function readMechanicsQuery(value: unknown): ParsedMechanicsQuery {
  if (!validQuery(value)) {
    const errors = validQuery.errors as { instancePath: string; message: string }[];
    const where = errors.map((error) => `${error.instancePath || "/"} ${error.message}`);
    throw new MechanicsQueryError(`invalid MechanicsQuery: ${where.join("; ")}`);
  }
  return structuredClone(value) as unknown as ParsedMechanicsQuery;
}

/** SHA-256 of the query's JCS bytes: its identity in a result. */
export const mechanicsQueryHash = (query: MechanicsQuery): string => sha256Hex(jcs(query as unknown as Json));

/** The sets a query declares, in the order a result's positions count them. */
export function declaredSets(query: MechanicsQuery): string[] {
  const kind = query.query;
  if (kind.kind === "stats") return [kind.set];
  if (kind.kind === "damage") return [kind.attacker.set, kind.defender.set];
  return kind.sides.flatMap((side) => side.map((combatant) => combatant.set));
}
