// The reader of a `BattleTrace` (ADR-028 section 5 F0): the schema, then what the profile cannot say. A trace is an evaluation
// artifact holding both sides' hidden information; the harness reads it as the oracle's answers, nothing else does.
import { ajvFor } from "@areneon/schema/ajv";
import schema from "@areneon/schema/contracts/v1/battle-trace.schema.json" with { type: "json" };
import type { BattleTrace, Step } from "@areneon/schema/types/v1/battle-trace.ts";
import type { Json } from "./canonical.ts";
import { sha256Jcs } from "./hash.ts";
import { SEATS } from "./battle-backend.ts";

export type { BattleTrace, Step };

export class TraceError extends Error {
  override name = "TraceError";
}

const validate = ajvFor("contract").compile(schema);

/** The step that ends the battle: the first whose omniscient lines hold `|win|` or `|tie`. */
export function endStep(trace: Pick<BattleTrace, "steps">): number {
  return trace.steps.findIndex((step) => step.omniscient.some((line) => line.startsWith("|win|") || line === "|tie"));
}

/**
 * Read a trace: the schema, then the reader rules. Step 0 (the start) has no action and every later step at least one; each
 * seat's requests are numbered 1, 2, ... across the steps; each team hashes to its `team_hashes` entry (SHA-256 of the JCS of
 * the set array, the runner's `teamHash`); exactly one step ends the battle, the last.
 */
export function readBattleTrace(value: unknown): BattleTrace {
  if (!validate(value)) throw new TraceError(`invalid BattleTrace: ${JSON.stringify(validate.errors)}`);
  const trace = value as unknown as BattleTrace;
  trace.steps.forEach((step, index) => {
    if ((index === 0) !== (step.actions.length === 0)) throw new TraceError("only the start step has no action");
  });
  for (const seat of SEATS) {
    const seqs = trace.steps.flatMap((step) => step.requests[seat].map((r) => r.request_seq));
    if (seqs.some((seq, i) => seq !== i + 1)) throw new TraceError("a seat's requests are numbered 1, 2, ... in order");
    if (sha256Jcs(trace.teams[seat] as unknown as Json) !== trace.team_hashes[seat]) throw new TraceError("a team does not hash to its team_hashes entry");
  }
  if (endStep(trace) !== trace.steps.length - 1) throw new TraceError("the last step, and only it, ends the battle");
  return trace;
}
