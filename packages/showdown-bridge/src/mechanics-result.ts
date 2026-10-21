// The MechanicsResult (ADR-016 section 2): the three closed outcomes of the
// probe and their reader. A result carries no free text from the input; its
// identity is the query hash, the version manifest, the mechanics config hash
// and the adapter format.

import { ajvFor } from "@areneon/schema/ajv";
import resultSchema from "@areneon/schema/contracts/v1/mechanics-result.schema.json" with { type: "json" };
import type { MechanicsResult } from "@areneon/schema/types/v1/mechanics-result.ts";
import type { VersionManifest } from "./formats.ts";

export type { MechanicsResult };

/**
 * Bumped when the probe changes how it reads a value from the engine, as
 * `ENCODER_FORMAT` is for the encoder: the same pin and config do not give
 * the same result across a change of reading.
 */
export const MECHANICS_ADAPTER_FORMAT = 1;

export class MechanicsResultError extends Error {
  override name = "MechanicsResultError";
}

/** The fields every outcome carries. */
export interface ResultIdentity {
  data_class: "synthetic" | "private";
  query_hash: string;
  manifest: VersionManifest;
  mechanics_config_hash: string;
  gap_ids: string[];
}

export type Outcome =
  | { kind: "computed"; answer: Extract<MechanicsResult, { kind: "computed" }>["answer"] }
  | { kind: "unsupported"; reason: Extract<MechanicsResult, { kind: "unsupported" }>["reason"]; position: number | null }
  | { kind: "refused"; code: "invalid_set"; position: number };

/** Put an outcome and its identity together. */
export const buildResult = (identity: ResultIdentity, outcome: Outcome): MechanicsResult =>
  ({ schema_version: 1, adapter_format: MECHANICS_ADAPTER_FORMAT, ...identity, ...outcome }) as MechanicsResult;

const validResult = ajvFor("contract").compile(resultSchema);

/** Read a result: the schema, then the sorted-and-unique `gap_ids` the profile cannot say. */
export function readMechanicsResult(value: unknown): MechanicsResult {
  if (!validResult(value)) throw new MechanicsResultError(`invalid MechanicsResult: ${JSON.stringify(validResult.errors)}`);
  const result = structuredClone(value) as unknown as MechanicsResult;
  if (result.gap_ids.some((id, index) => index > 0 && id <= (result.gap_ids[index - 1] as string))) {
    throw new MechanicsResultError("gap_ids must be sorted and unique");
  }
  return result;
}
