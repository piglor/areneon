import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readTeamDatasetManifest, readTeamGeneratorConfig, recordHash, teamVocabularyHash, type TeamGeneratorConfig } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import type { Engine } from "@areneon/showdown-bridge";
import { notices, runCli, type CliIo, trialDraws } from "../src/command.ts";
import { type Pilot, runPilot } from "../src/pilot.ts";
import { ASSIGNMENT_KEY, drawerFor, engine, FORMAT, KEY, light, small, table, VOCABULARY_FILE, vocabulary } from "./support.ts";

const scratch = mkdtempSync(join(tmpdir(), "command-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

// A config that satisfies the schema (40,000 pilot draws), with a recorded pilot that is a real
// 800-draw pilot scaled by 50: the rates are real, the run stays small.
const cfg: TeamGeneratorConfig = { ...small(), pilot_draws: 40_000, pilot_min_hits: 150 };
const identity = { config_hash: recordHash(cfg), label_set_hash: recordHash(light), vocabulary_hash: teamVocabularyHash(vocabulary) };
const trial = runPilot(drawerFor(cfg, KEY, light), identity, light, 800);
const scaled: Pilot = { ...trial, draws: 40_000, rejected: trial.rejected * 50, cell_hits: trial.cell_hits.map((h) => h * 50) };

const files = {
  config: join(scratch, "config.json"),
  labels: join(scratch, "labels.json"),
  pilot: join(scratch, "pilot.json"),
};
writeFileSync(files.config, JSON.stringify(cfg));
writeFileSync(files.labels, JSON.stringify(light));
writeFileSync(files.pilot, JSON.stringify(scaled));

function cli(args: string[], extra: Partial<CliIo> = {}): { code: number; out: string[]; err: string[] } {
  const out: string[] = [];
  const err: string[] = [];
  const code = runCli(args, { out: (l) => out.push(l), err: (l) => err.push(l), engine, ...extra });
  return { code, out, err };
}

const generate = (out: string, more: string[] = []): string[] => [
  "generate",
  "--config", files.config,
  "--labels", files.labels,
  "--format", FORMAT,
  "--out", out,
  "--dataset-id", "cli-1",
  "--generation-key", KEY,
  "--assignment-key", ASSIGNMENT_KEY,
  "--dev", "2",
  "--validation", "2",
  "--training", "5",
  "--vocabulary", VOCABULARY_FILE,
  "--pilot", files.pilot,
  "--generator-commit", "b".repeat(40),
  ...more,
];

describe("the generate command", () => {
  const a = join(scratch, "a");
  const b = join(scratch, "b");

  it("writes the records, manifest, coverage, pilot and rejection log, and prints a summary", { timeout: 30_000 }, () => {
    const run = cli(generate(a));
    expect(run.code).toBe(0);
    const summary = JSON.parse(run.out[0] as string) as { dataset_id: string; manifest_hash: string; teams: Record<string, number>; files: string[] };
    expect(summary.dataset_id).toBe("cli-1");
    expect(readdirSync(a).sort()).toEqual([...summary.files].sort());
    const manifest = JSON.parse(readFileSync(join(a, "manifest.json"), "utf8"));
    expect(recordHash(manifest)).toBe(summary.manifest_hash);
    expect(readTeamDatasetManifest(manifest, { config: readTeamGeneratorConfig(cfg, table), labelSet: light }).generator.generator_commit).toBe("b".repeat(40));
    expect(run.err.at(-1)).toMatch(/draws \(0 of them the pilot\) in .* s: \d+ draws\/s/);
    expect(summary.teams.dev).toBe(manifest.splits[0].teams);
  });

  it("is deterministic: the same inputs give the same bytes", { timeout: 30_000 }, () => {
    cli(generate(b));
    for (const file of readdirSync(a)) expect(readFileSync(join(b, file)), file).toEqual(readFileSync(join(a, file)));
  });

  it("dedupes against earlier records, from memory or from a sorted file on disk", () => {
    const reference = ["--reference", join(a, "records.dev.jsonl"), "--reference", join(a, "records.training.jsonl")];
    const memory = join(scratch, "c");
    const disk = join(scratch, "d");
    expect(cli(generate(memory, reference)).code).toBe(0);
    expect(cli(generate(disk, [...reference, "--reference-on-disk"])).code).toBe(0);
    for (const file of readdirSync(memory)) expect(readFileSync(join(disk, file)), file).toEqual(readFileSync(join(memory, file)));
    const earlier = new Set(
      ["records.dev.jsonl", "records.training.jsonl"].flatMap((f) => readFileSync(join(a, f), "utf8").split("\n").filter(Boolean).map((l) => (JSON.parse(l) as { canonical_team_hash: string }).canonical_team_hash)),
    );
    for (const f of readdirSync(memory).filter((n) => n.startsWith("records."))) {
      for (const line of readFileSync(join(memory, f), "utf8").split("\n").filter(Boolean)) expect(earlier.has((JSON.parse(line) as { canonical_team_hash: string }).canonical_team_hash)).toBe(false);
    }
    const manifest = JSON.parse(readFileSync(join(memory, "manifest.json"), "utf8")) as { rejection_log: { duplicates: number }; reference_pool_hash: string };
    expect(manifest.rejection_log.duplicates).toBeGreaterThan(0);
    expect(manifest.reference_pool_hash).not.toBe(JSON.parse(readFileSync(join(a, "manifest.json"), "utf8")).reference_pool_hash);
  }, 60_000);

  it("reports progress every N draws", () => {
    const run = cli(generate(join(scratch, "p"), ["--progress-every", "200"]));
    expect(run.code).toBe(0);
    expect(run.err.filter((l) => /^\d+ draws, \d+ bins open$/.test(l)).length).toBeGreaterThan(0);
  });

  it("uses the plan id when given, and the git commit when none is given", () => {
    const out = join(scratch, "e");
    const args = generate(out, ["--plan-id", "plan-x"]);
    args.splice(args.indexOf("--generator-commit"), 2);
    expect(cli(args).code).toBe(0);
    const manifest = JSON.parse(readFileSync(join(out, "manifest.json"), "utf8")) as { plan_id: string; generator: { generator_commit: string } };
    expect(manifest.plan_id).toBe("plan-x");
    expect(manifest.generator.generator_commit).toMatch(/^[0-9a-f]{40}$/);
  });
});

describe("the pilot command", () => {
  it("is as long as the config says, unless --draws asks for a trial", () => {
    expect(trialDraws(undefined, { pilot_draws: 40_000 })).toBe(40_000);
    expect(trialDraws("25", { pilot_draws: 40_000 })).toBe(25);
    expect(() => trialDraws("-1", { pilot_draws: 40_000 })).toThrow(/--draws must be a whole number/);
  });

  it("runs a trial pilot with the vocabulary exported from the pinned engine, and generate refuses a pilot of the wrong length", () => {
    const path = join(scratch, "trial.json");
    // no --vocabulary: the command exports it from the engine, as the golden file says
    const run = cli(["pilot", "--config", files.config, "--labels", files.labels, "--format", FORMAT, "--out", path, "--generation-key", KEY, "--draws", "40"]);
    expect(run.code).toBe(0);
    const pilot = JSON.parse(readFileSync(path, "utf8")) as Pilot;
    expect(pilot).toMatchObject({ draws: 40, vocabulary_hash: identity.vocabulary_hash, config_hash: identity.config_hash });
    expect(JSON.parse(run.out[0] as string)).toMatchObject({ pilot: path, draws: 40 });
    const refused = cli(generate(join(scratch, "f"), ["--pilot", path]));
    expect(refused.code).toBe(1);
    expect(refused.err.join()).toMatch(/the pilot has 40 draws, the config says 40000/);
  }, 60_000);
});

describe("the command line refuses", () => {
  const ok = (): string[] => generate(join(scratch, "g"));
  const without = (name: string): string[] => {
    const args = ok();
    args.splice(args.indexOf(name), 2);
    return args;
  };

  it.each([
    ["no command", [], /usage/],
    ["an unknown command", ["frobnicate"], /usage/],
    ["an unknown option", ["generate", "--nope"], /nope/],
    ["a missing --config", () => without("--config"), /--config is required/],
    ["a missing --out", () => without("--out"), /--out is required/],
    ["a missing --dataset-id", () => without("--dataset-id"), /--dataset-id is required/],
    ["a bad generation key", () => [...ok(), "--generation-key", "abc"], /generation-key/],
    ["a bad commit", () => [...ok(), "--generator-commit", "HEAD"], /40-digit/],
    ["a non-number size", () => [...ok(), "--dev", "ten"], /--dev must be a whole number/],
    ["a missing file", () => [...ok(), "--config", join(scratch, "nope.json")], /ENOENT/],
    ["a config the schema refuses", () => { writeFileSync(join(scratch, "bad.json"), JSON.stringify({ ...cfg, iv: 99 })); return [...ok(), "--config", join(scratch, "bad.json")]; }, /invalid team generator config/],
    ["a pilot for another config", () => { writeFileSync(join(scratch, "other.json"), JSON.stringify({ ...scaled, config_hash: "0".repeat(64) })); return [...ok(), "--pilot", join(scratch, "other.json")]; }, /another config/],
  ] as const)("%s (exit 2)", (_name, args, message) => {
    const run = cli(typeof args === "function" ? args() : [...args]);
    expect(run.code).toBe(2);
    expect(run.err.join("\n")).toMatch(message);
  });

  it("a plan it cannot make (exit 1)", () => {
    const run = cli([...ok(), "--dev", "0", "--validation", "0", "--training", "0"]);
    expect(run.code).toBe(1);
    expect(run.err.join()).toMatch(/PlanError: no split has a team count/);
  });

  it("data beyond the in-memory limits (exit 1, with a message, not a stack)", () => {
    const failing = (error: Error): Engine => ({ commit: "x", sim: { get Dex(): never { throw error; } } }) as never;
    const set = cli(generate(join(scratch, "i")), { engine: failing(new RangeError("Set maximum size exceeded")) });
    expect(set.code).toBe(1);
    expect(set.err.join()).toMatch(/Set maximum size exceeded/);
    const long = cli(generate(join(scratch, "i")), { engine: failing(Object.assign(new Error("Cannot create a string longer than 0x1fffffe8 characters"), { code: "ERR_STRING_TOO_LONG" })) });
    expect(long.code).toBe(1);
  });

  it("an unexpected failure is not hidden", () => {
    mkdirSync(join(scratch, "h"), { recursive: true });
    expect(() => cli(generate(join(scratch, "h")), { engine: { commit: "x", sim: {} } as never })).toThrow();
  });
});

describe("what a finished run tells the operator", () => {
  const base = { provisioning: { exceeds_d_max: false, planned_draws: 900 }, replan: 0, splits: [{ split: "dev", requested: 10, teams: 10 }] };
  it("says nothing when all went as planned", () => {
    expect(notices(base as never, 1000)).toEqual([]);
  });
  it("warns about a quantile beyond d_max, a re-plan, and a split above its request", () => {
    const lines = notices({ provisioning: { exceeds_d_max: true, planned_draws: 5000 }, replan: 2, splits: [{ split: "dev", requested: 20, teams: 55 }] } as never, 1000);
    expect(lines).toHaveLength(3);
    expect(lines[0]).toMatch(/5000-draw fill quantile is beyond d_max 1000/);
    expect(lines[1]).toMatch(/re-planned 2 time/);
    expect(lines[2]).toBe("dev: 20 requested, 55 delivered (proportional quotas with a floor of n_stratum_min per stratum)");
  });
});
