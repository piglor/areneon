// What the sealed-set tests share: a small public dataset made with the real generator (its manifest fixes the
// strata, its teams are the known teams), complete evaluation parameters, and a deterministic stand-in for the
// OS CSPRNG. The keys here are throwaway test values; no real key or sealed team exists in this repository.

import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { readEvaluationConfig, recordHash, teamVocabularyHash, type TeamGeneratorConfig, type TeamRecord } from "@areneon/showdown-bridge";
import { KnownHashes, referencePoolHash } from "../src/dedupe.ts";
import { generate } from "../src/generate.ts";
import { type Pilot, runPilot } from "../src/pilot.ts";
import { ASSIGNMENT_KEY, drawerFor, engine, format, KEY, light, small, table, vocabulary } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));

/** The committed config with the schema's pilot length, and a pilot that is a real 800-draw pilot scaled by 50. */
export const cfg: TeamGeneratorConfig = { ...small(), pilot_draws: 40_000, pilot_min_hits: 150 };
const identity = { config_hash: recordHash(cfg), label_set_hash: recordHash(light), vocabulary_hash: teamVocabularyHash(vocabulary) };
const trial = runPilot(drawerFor(cfg, KEY, light), identity, light, 800);
export const pilot: Pilot = { ...trial, draws: 40_000, rejected: trial.rejected * 50, cell_hits: trial.cell_hits.map((h) => h * 50) };

const scratch = mkdtempSync(join(tmpdir(), "sealed-support-"));
const publicRun = generate({
  engine,
  format,
  table,
  vocabulary,
  labelSet: light,
  config: cfg,
  generationKey: KEY,
  assignmentKey: ASSIGNMENT_KEY,
  datasetId: "public-1",
  planId: "public-plan-1",
  sizes: { dev: 2, validation: 2, training: 5 },
  generatorCommit: "a".repeat(40),
  reference: { hash: referencePoolHash([]), known: new KnownHashes([]) },
  pilot,
  logDir: (mkdirSync(join(scratch, "log")), join(scratch, "log")),
});
rmSync(scratch, { recursive: true, force: true });

export const publicManifest = publicRun.manifest;
/** Every public team: the three splits. */
export const publicTeams: TeamRecord[] = Object.values(publicRun.records).flat() as TeamRecord[];

const complete = (change: Record<string, unknown> = {}) => {
  const read = (file: string): Record<string, unknown> => JSON.parse(readFileSync(join(ROOT, "config/evaluation", file), "utf8")) as Record<string, unknown>;
  return { ...read("evaluation-parameters.v1.json"), n_teams_floor: 8, n_teams: 24, n_seeds: 1, d_team: { differing_fields: 2, basis: "measured" }, ...change };
};
const rule = JSON.parse(readFileSync(join(ROOT, "config/evaluation/training-gate-rule.v1.json"), "utf8")) as Record<string, unknown>;
export const completeParameters = complete;
export const evaluationRule = rule;
export const evaluation = readEvaluationConfig(complete(), rule);

/** A deterministic 256-bit source: hash of a counter and a label. Never a real key. */
export function counterSecret(label: string): () => string {
  let n = 0;
  return () => createHash("sha256").update(`${label}:${n++}`).digest("hex");
}

/** A fresh private store directory under the system temp dir. */
export function freshStore(): { dir: string; remove: () => void } {
  const dir = mkdtempSync(join(tmpdir(), "sealed-store-"));
  return { dir, remove: () => rmSync(dir, { recursive: true, force: true }) };
}
