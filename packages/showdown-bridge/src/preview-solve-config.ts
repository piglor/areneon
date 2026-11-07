// The Team Preview solver's reviewed configuration and its identity (ADR-029 section 4, ticket #1259).
//
// `PreviewSolveConfig` is reviewed data (`config/preview/<format>/preview-solve-1.v<N>.json`), never code; the schema checks the shape
// and `readPreviewSolveConfig` adds what the profile cannot say. `previewSolveConfigHash` is the `preview_solve_config_hash` of the
// solver identity and `previewSolveIdentityHash` the hash a solver policy id carries as `preview_solve: <hash>` (`policyId` in
// `policy-protocol.ts`), independent of ADR-026's `search`. The identity names every input that changes what the solver plays:
// the config, the belief identity, the shared reader, the version manifest and the format of each module whose behaviour it
// depends on (the matrix solver, the payoff table's leaf, the hypothetical battle). Nothing here runs a solve.

import { readFileSync } from "node:fs";
import { ajvFor } from "@areneon/schema/ajv";
import previewSolveConfigSchema from "@areneon/schema/contracts/v1/preview-solve-config.schema.json" with { type: "json" };
import type { PreviewSolveConfig } from "@areneon/schema/types/v1/preview-solve-config.ts";
import { type Json, jcs } from "./canonical-json.ts";
import type { VersionManifest } from "./formats.ts";
import { sha256Hex } from "./rules.ts";

export type { PreviewSolveConfig };

/** The label of a policy whose Team Preview is chosen by the solver (ADR-029 section 4). */
export const PREVIEW_SOLVE_POLICY_LABEL = "learned pair with a decision-time Team Preview solve (evaluation only)";

/** The matrix solvers that iterate (and so state `iterations`); `exact-support` is declared by the ADR and not built. */
export const ITERATIVE_SOLVERS: readonly string[] = ["regret-matching", "regret-matching-plus"];

export class PreviewSolveConfigError extends Error {
  override name = "PreviewSolveConfigError";
}

const validConfig = ajvFor("contract").compile(previewSolveConfigSchema);

const PREVIEW_CONFIG_DIR = new URL("../../../config/preview/", import.meta.url);

/** Read a config: the schema, then the rules it cannot say (`min_valid <= n`; `iterations` exactly for the iterative solvers). */
export function readPreviewSolveConfig(value: unknown): PreviewSolveConfig {
  if (!validConfig(value)) throw new PreviewSolveConfigError(`invalid PreviewSolveConfig: ${JSON.stringify(validConfig.errors)}`);
  const config = structuredClone(value) as unknown as PreviewSolveConfig;
  if (config.worlds.min_valid > config.worlds.n) throw new PreviewSolveConfigError("min_valid exceeds the worlds per decision");
  if (ITERATIVE_SOLVERS.includes(config.solver.kind) !== (config.solver.iterations !== undefined)) {
    throw new PreviewSolveConfigError("iterations is present exactly for the iterative solvers");
  }
  return config;
}

/** The hash the solver identity names the config by: SHA-256 of the config's JCS bytes. */
export const previewSolveConfigHash = (config: PreviewSolveConfig): string => sha256Hex(jcs(config as unknown as Json));

/** Read `config/preview/<format>/<name>.json` (default `preview-solve-1.v1`) for a format. */
export function loadPreviewSolveConfig(formatId: string, name = "preview-solve-1.v1", dir: URL = PREVIEW_CONFIG_DIR): PreviewSolveConfig {
  const config = readPreviewSolveConfig(JSON.parse(readFileSync(new URL(`${formatId}/${name}.json`, dir), "utf8")) as unknown);
  if (config.format_id !== formatId) throw new PreviewSolveConfigError(`the config is for ${config.format_id}, not ${formatId}`);
  return config;
}

/** Everything that changes what the solver plays, besides the config's own values (which `preview_solve_config_hash` covers). */
export interface PreviewSolveIdentityInputs {
  /** ADR-025's `belief_identity_hash` of the belief in use; must equal the config's. */
  belief_identity_hash: string;
  reader_sha256: string;
  version_manifest: VersionManifest;
  /** The matrix solver's format string (`SOLVER_FORMAT`, `preview-solver/1`). */
  solver_format: string;
  /** The payoff table's `PAYOFF_LEAF_FORMAT`. */
  leaf_format: number;
  /** The hypothetical battle's `HYPOTHETICAL_BATTLE_FORMAT`. */
  hypothetical_battle_format: number;
}

const HEX64 = /^[0-9a-f]{64}$/;

/** The identity preimage: the inputs plus `preview_solve_config_hash`, checked. */
export function previewSolveIdentity(config: PreviewSolveConfig, inputs: PreviewSolveIdentityInputs): Record<string, Json> {
  for (const field of ["belief_identity_hash", "reader_sha256"] as const) if (!HEX64.test(inputs[field])) throw new PreviewSolveConfigError(`${field} is a SHA-256: 64 lowercase hex digits`);
  if (inputs.belief_identity_hash !== config.belief_identity_hash) throw new PreviewSolveConfigError("the belief identity is not the one the config names");
  for (const field of ["leaf_format", "hypothetical_battle_format"] as const) {
    if (!Number.isInteger(inputs[field]) || inputs[field] < 1) throw new PreviewSolveConfigError(`${field} is a positive integer`);
  }
  if (typeof inputs.solver_format !== "string" || inputs.solver_format === "") throw new PreviewSolveConfigError("solver_format is a non-empty string");
  return {
    preview_solve_config_hash: previewSolveConfigHash(config),
    belief_identity_hash: inputs.belief_identity_hash,
    reader_sha256: inputs.reader_sha256,
    version_manifest: inputs.version_manifest as unknown as Json,
    solver_format: inputs.solver_format,
    leaf_format: inputs.leaf_format,
    hypothetical_battle_format: inputs.hypothetical_battle_format,
  };
}

/** SHA-256 of the identity's JCS bytes: the `preview_solve` member of a solver policy id. */
export const previewSolveIdentityHash = (config: PreviewSolveConfig, inputs: PreviewSolveIdentityInputs): string => sha256Hex(jcs(previewSolveIdentity(config, inputs) as Json));
