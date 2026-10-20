// The sealed team set's two records (ADR-013 section 2, evaluation protocol
// L3 and L5): the generation manifest, which holds hashes and counts only, and
// the reveal on retirement, which alone holds the key and the salt. The
// generator and the store that produce them are in `packages/team-dataset`;
// the readers live here, with the contracts, so any language-neutral consumer
// reads them the same way. Engine-free and Node-only, like `evaluation-plan.ts`.

import { ajvFor } from "@areneon/schema/ajv";
import retiredSchema from "@areneon/schema/contracts/v1/retired-sealed-manifest.schema.json" with { type: "json" };
import manifestSchema from "@areneon/schema/contracts/v1/sealed-team-set-manifest.schema.json" with { type: "json" };
import type { RetiredSealedManifest } from "@areneon/schema/types/v1/retired-sealed-manifest.ts";
import type { SealedTeamSetManifest } from "@areneon/schema/types/v1/sealed-team-set-manifest.ts";
import { domainHash, EvaluationConfigError, strictlyAscending } from "./evaluation-config.ts";

export type { RetiredSealedManifest, SealedTeamSetManifest };

const validManifest = ajvFor("contract").compile(manifestSchema);
const validRetired = ajvFor("contract").compile(retiredSchema);

/** The domain hash of a manifest: how the register's retirement and a reveal name it. */
export const sealedManifestHash = (manifest: SealedTeamSetManifest): string => domainHash("sealed-team-set-manifest", manifest);
/** The domain hash of a reveal: the `reveal_hash` of the register's `set_retirement` entry. */
export const retiredManifestHash = (manifest: RetiredSealedManifest): string => domainHash("retired-sealed-manifest", manifest);

/** The label of a stratum in a SealedSetCommitment: `s` and the stratum number in five digits, so the labels sort as the numbers do. */
export const stratumLabel = (stratum: number): string => `s${String(stratum).padStart(5, "0")}`;

/** Read a SealedTeamSetManifest: the schema, then the strata sorted and unique and their quotas summing to team_count. */
export function readSealedTeamSetManifest(value: unknown): SealedTeamSetManifest {
  if (!validManifest(value)) throw new EvaluationConfigError(`invalid SealedTeamSetManifest: ${JSON.stringify(validManifest.errors)}`);
  const manifest = structuredClone(value) as unknown as SealedTeamSetManifest;
  if (!strictlyAscending(manifest.strata.map((s) => s.stratum))) throw new EvaluationConfigError("the strata must be sorted by stratum and unique");
  const total = manifest.strata.reduce((sum, s) => sum + s.quota, 0);
  if (total !== manifest.team_count) throw new EvaluationConfigError(`the quotas sum to ${total} but team_count is ${manifest.team_count}`);
  return manifest;
}

/** Read a RetiredSealedManifest: the schema, then a key that is not the salt (one secret reused for both would weaken L3). */
export function readRetiredSealedManifest(value: unknown): RetiredSealedManifest {
  if (!validRetired(value)) throw new EvaluationConfigError(`invalid RetiredSealedManifest: ${JSON.stringify(validRetired.errors)}`);
  const manifest = structuredClone(value) as unknown as RetiredSealedManifest;
  if (manifest.reveal.key === manifest.reveal.salt) throw new EvaluationConfigError("the key and the salt must differ");
  return manifest;
}
