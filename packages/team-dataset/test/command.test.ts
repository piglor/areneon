import { cpSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readTeamDatasetManifest, readTeamGeneratorConfig, recordHash, teamVocabularyHash, type TeamGeneratorConfig } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import type { Engine } from "@areneon/showdown-bridge";
import { DatasetReadError, readDataset } from "../src/dataset-reader.ts";
import { referencePoolHash } from "../src/dedupe.ts";
import { fileURLToPath } from "node:url";
import { notices, runCli, type CliIo, trialDraws } from "../src/command.ts";
import { Drawer } from "../src/draw.ts";
import { Proposal } from "../src/proposal.ts";
import { roleSetsSource } from "../src/role-set-proposal.ts";
import { type Pilot, runPilot } from "../src/pilot.ts";
import { ASSIGNMENT_KEY, drawerFor, engine, FORMAT, format, KEY, light, small, table, VOCABULARY_FILE, vocabulary } from "./support.ts";

const scratch = mkdtempSync(join(tmpdir(), "command-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

// A config that satisfies the schema (40,000 pilot draws), with a recorded pilot that is a real
// 800-draw pilot scaled by 50: the rates are real, the run stays small.
const cfg: TeamGeneratorConfig = { ...small(), pilot_draws: 40_000, pilot_min_hits: 150 };
const identity = { config_hash: recordHash(cfg), label_set_hash: recordHash(light), vocabulary_hash: teamVocabularyHash(vocabulary) };
const trial = runPilot(drawerFor(cfg, KEY, light), identity, light, 800);
const scaled: Pilot = { ...trial, draws: 40_000, rejected: trial.rejected * 50, cell_hits: trial.cell_hits.map((h) => h * 50) };

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));
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

  // The readers of the evaluation paths pass no reference hashes: the dataset's own list is what they check (#1170).
  describe("a dataset generated with --reference", () => {
    const context = { engine, format, table, config: readTeamGeneratorConfig(cfg, table), labelSet: light, vocabulary };
    const refused = (dir: string, extra: Partial<typeof context & { referenceHashes: string[] }> = {}): string => {
      try {
        readDataset(dir, { ...context, ...extra });
      } catch (error) {
        expect(error).toBeInstanceOf(DatasetReadError);
        return (error as Error).message;
      }
      throw new Error("the dataset was accepted");
    };
    const listed = (dir: string): string[] => readFileSync(join(dir, "reference-hashes.txt"), "utf8").split("\n").filter(Boolean);
    const copyOf = (name: string): string => {
      const dir = join(scratch, name);
      cpSync(join(scratch, "c"), dir, { recursive: true });
      return dir;
    };

    it("is read from its own reference list, which the first dataset does not have", () => {
      expect(readdirSync(join(scratch, "c"))).toContain("reference-hashes.txt");
      expect(readdirSync(join(scratch, "a"))).not.toContain("reference-hashes.txt");
      expect(readDataset(join(scratch, "c"), context).manifest.reference_pool_hash).toBe(referencePoolHash(listed(join(scratch, "c"))));
      expect(readDataset(join(scratch, "a"), context).manifest.reference_pool_hash).toBe(referencePoolHash([]));
      // a caller's own list still decides, and a wrong one is refused
      expect(readDataset(join(scratch, "c"), { ...context, referenceHashes: listed(join(scratch, "c")) }).manifest.dataset_id).toBe("cli-1");
      expect(refused(join(scratch, "c"), { referenceHashes: [] })).toMatch(/reference_pool_hash/);
    });

    it("refuses a doctored, shortened, unsorted or malformed reference list, and a missing one", () => {
      const hashes = listed(join(scratch, "c"));
      const put = (name: string, lines: string[] | null): string => {
        const dir = copyOf(name);
        if (lines === null) rmSync(join(dir, "reference-hashes.txt"));
        else writeFileSync(join(dir, "reference-hashes.txt"), lines.map((l) => `${l}\n`).join(""));
        return dir;
      };
      expect(refused(put("r-short", hashes.slice(1)))).toMatch(/reference_pool_hash/);
      expect(refused(put("r-extra", [...hashes, "f".repeat(64)]))).toMatch(/reference_pool_hash/);
      expect(refused(put("r-missing", null))).toMatch(/reference_pool_hash/);
      expect(refused(put("r-order", [hashes[1] as string, hashes[0] as string, ...hashes.slice(2)]))).toMatch(/sorted and unique/);
      expect(refused(put("r-dup", [hashes[0] as string, ...hashes]))).toMatch(/sorted and unique/);
      expect(refused(put("r-bad", ["xyz", ...hashes]))).toMatch(/line 1 is not a hash/);
    });

    it("refuses a team that repeats a reference team, even when the manifest names that list", () => {
      const dir = join(scratch, "r-repeat");
      cpSync(join(scratch, "a"), dir, { recursive: true });
      const own = (JSON.parse(readFileSync(join(dir, "records.dev.jsonl"), "utf8").split("\n")[0] as string) as { canonical_team_hash: string }).canonical_team_hash;
      writeFileSync(join(dir, "reference-hashes.txt"), `${own}\n`);
      const manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8")) as { reference_pool_hash: string };
      manifest.reference_pool_hash = referencePoolHash([own]);
      writeFileSync(join(dir, "manifest.json"), JSON.stringify(manifest, null, 2));
      expect(refused(dir)).toMatch(/repeats a reference team/);
    });
  });

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
  }, 180_000);
});

describe("a mixture config with a role_sets component", () => {
  const tableFile = join(ROOT, "fixtures", "showdown", FORMAT, "role-sets.json");
  const mixture = join(scratch, "mixture.json");
  writeFileSync(mixture, readFileSync(join(ROOT, "config", "teams", FORMAT, "generator.mixture-r.json"), "utf8"));
  const pilot = (out: string, more: string[]): ReturnType<typeof cli> => cli(["pilot", "--config", mixture, "--labels", files.labels, "--format", FORMAT, "--out", out, "--generation-key", KEY, "--draws", "20", "--vocabulary", VOCABULARY_FILE, ...more]);

  it("reads the golden table by default and the one --role-sets names", () => {
    expect(pilot(join(scratch, "m1.json"), []).code).toBe(0);
    expect(pilot(join(scratch, "m2.json"), ["--role-sets", tableFile]).code).toBe(0);
  }, 120_000);

  it("refuses a table that is not the vocabulary's (exit 2) or not the one the config names (exit 1)", () => {
    const doctored = join(scratch, "table-doctored.json");
    writeFileSync(doctored, JSON.stringify({ ...JSON.parse(readFileSync(tableFile, "utf8")), vocabulary_hash: "0".repeat(64) }));
    expect(pilot(join(scratch, "m3.json"), ["--role-sets", doctored]).code).toBe(2);
    const other = JSON.parse(readFileSync(mixture, "utf8")) as { proposal: { components: { role_set_table_hash?: string }[] } };
    (other.proposal.components[0] as { role_set_table_hash: string }).role_set_table_hash = "0".repeat(64);
    const wrong = join(scratch, "mixture-wrong.json");
    writeFileSync(wrong, JSON.stringify(other));
    const run = cli(["pilot", "--config", wrong, "--labels", files.labels, "--format", FORMAT, "--out", join(scratch, "m4.json"), "--generation-key", KEY, "--draws", "20", "--vocabulary", VOCABULARY_FILE]);
    expect(run.code).toBe(1);
    expect(run.err.join()).toMatch(/hash mismatch/);
  }, 120_000);
});

describe("a mixture dataset from the command line (ADR-020 gates and the evaluation-only option)", () => {
  const tableFile = join(ROOT, "fixtures", "showdown", FORMAT, "role-sets.json");
  const sourceTable = roleSetsSource(engine, vocabulary, table, JSON.parse(readFileSync(tableFile, "utf8")));
  const loose = { p2_min_accept_permille: 1, p3_min_bits: 1, p4_min_strata_permille: 1 };
  const mixtureOf = (proposal: object): TeamGeneratorConfig => ({ ...cfg, config_id: "cli-mixture", proposal } as TeamGeneratorConfig);
  const scale = (pilot: Pilot): Pilot => ({
    ...pilot,
    draws: 40_000,
    rejected: pilot.rejected * 50,
    cell_hits: pilot.cell_hits.map((h) => h * 50),
    ...(pilot.components === undefined ? {} : { components: pilot.components.map((c) => ({ ...c, draws: c.draws * 50, rejected: c.rejected * 50, cell_hits: c.cell_hits.map((h) => h * 50) })) }),
  });
  const pilotFor = (config: TeamGeneratorConfig): Pilot => {
    const ident = { config_hash: recordHash(config), label_set_hash: recordHash(light), vocabulary_hash: teamVocabularyHash(vocabulary) };
    return scale(runPilot(new Drawer({ engine, format, proposal: new Proposal(vocabulary, table, config, config.proposal === undefined ? undefined : sourceTable), labelSet: light, generationKey: KEY }), ident, light, 800));
  };
  const write = (name: string, value: unknown): string => {
    const path = join(scratch, name);
    writeFileSync(path, JSON.stringify(value));
    return path;
  };
  const mixed = (gates?: object): TeamGeneratorConfig =>
    mixtureOf({ kind: "mixture", components: [{ kind: "role_sets", role_set_table_hash: sourceTable.hash, share_permille: 700 }, { kind: "uniform", share_permille: 300 }], ...(gates === undefined ? {} : { gates }) });
  const plainConfig = (config: TeamGeneratorConfig): TeamGeneratorConfig => {
    const copy = structuredClone(config);
    delete copy.proposal;
    return copy;
  };
  const args = (out: string, config: string, pilot: string, more: string[]): string[] =>
    generate(out).map((a, k, all) => (all[k - 1] === "--config" ? config : all[k - 1] === "--pilot" ? pilot : a)).concat(more);
  const stdout = (run: ReturnType<typeof cli>) => JSON.parse(run.out.join("")) as { gates?: { publishable: boolean; failed: string[] }; files: string[] };

  it("a publishing run with no gates configured writes the files and exits 1", () => {
    const config = mixed();
    const out = join(scratch, "pub-1");
    const run = cli(args(out, write("mix-nogates.json", config), write("mix-nogates-pilot.json", pilotFor(config)), ["--publish"]));
    expect(run.code).toBe(1);
    expect(run.err.join("\n")).toMatch(/not publishable: gates:not_configured/);
    expect(stdout(run).gates?.publishable).toBe(false);
    expect(stdout(run).gates?.failed).toContain("gates:not_configured");
    expect(readdirSync(out)).toContain("coverage.json");
    // The same run is not an error without --publish.
    const plain = cli(args(join(scratch, "pub-1b"), write("mix-nogates.json", config), join(scratch, "mix-nogates-pilot.json"), []));
    expect(plain.code).toBe(0);
    expect(plain.err.join("\n")).toMatch(/pilot gates not met or not evaluated/);
  }, 300_000);

  it("a publishing run whose gates are met exits 0 and the report carries the counts", () => {
    const config = mixed(loose);
    const out = join(scratch, "pub-2");
    const run = cli(args(out, write("mix-gates.json", config), write("mix-gates-pilot.json", pilotFor(config)), ["--publish", "--d-team", "2", "--uniform-pilot", write("mix-uniform-pilot.json", pilotFor(plainConfig(config)))]));
    expect(run.err.join("\n")).not.toMatch(/not publishable/);
    expect(run.code).toBe(0);
    const coverage = JSON.parse(readFileSync(join(out, "coverage.json"), "utf8")) as { proposal: { publishable: boolean; p4: { reference_strata: number } }; near_duplicates: { d_team: number }; population: string };
    expect(coverage.proposal.publishable).toBe(true);
    expect(coverage.proposal.p4.reference_strata).toBeGreaterThan(0);
    expect(coverage.near_duplicates.d_team).toBe(2);
    expect(coverage.population).toMatch(/^proposal mixture cli-mixture/);
  }, 300_000);

  it("--evaluation-only drops teams near a --reference dataset and marks the manifest", () => {
    const config = mixtureOf({ kind: "mixture", components: [{ kind: "role_sets", role_set_table_hash: sourceTable.hash, share_permille: 1000 }] });
    const out = join(scratch, "eval-1");
    const configFile = write("pure-r.json", config);
    const pilotFile = write("pure-r-pilot.json", pilotFor(config));
    // The reference is the same stream's own dev split: its teams are identical to the first candidates of the run.
    const first = cli(args(join(scratch, "eval-0"), configFile, pilotFile, []));
    expect(first.code).toBe(0);
    // ... with one nature changed and a made-up canonical hash, so L1 does not see them and only the distance can.
    const nudgedLines = readFileSync(join(scratch, "eval-0", "records.dev.jsonl"), "utf8")
      .split("\n")
      .filter((l) => l !== "")
      .map((l) => {
        const record = JSON.parse(l) as { paste: string; canonical_team_hash: string };
        const paste = record.paste.replace(/(\w+) Nature/, (_m, nature: string) => `${nature === "Quirky" ? "Hardy" : "Quirky"} Nature`);
        return JSON.stringify({ ...record, paste, canonical_team_hash: "0".repeat(64) });
      });
    const reference = write("reference.jsonl", "");
    writeFileSync(reference, `${nudgedLines.join("\n")}\n`);
    const run = cli(args(out, configFile, pilotFile, ["--evaluation-only", "--d-team", "1", "--reference", reference]));
    expect(run.err.join("\n")).not.toMatch(/Error/);
    expect(run.code).toBe(0);
    const manifest = JSON.parse(readFileSync(join(out, "manifest.json"), "utf8")) as { evaluation_only: { d_team: number; near_references: number } };
    expect(manifest.evaluation_only.d_team).toBe(1);
    expect(manifest.evaluation_only.near_references).toBeGreaterThan(0);
    const log = readFileSync(join(out, "rejections.jsonl"), "utf8").split("\n").filter((l) => l.includes('"near_reference"'));
    expect(log).toHaveLength(manifest.evaluation_only.near_references);
    const again = cli(args(join(scratch, "eval-2"), configFile, pilotFile, ["--evaluation-only", "--d-team", "1"]));
    expect(again.code).toBe(2);
    expect(again.err.join()).toMatch(/--evaluation-only needs --d-team and at least one --reference/);
  }, 300_000);

  it("--publish on a config with no proposal prints a notice and keeps the exit code", () => {
    const run = cli(generate(join(scratch, "pub-uniform"), ["--publish"]));
    expect(run.code).toBe(0);
    expect(run.err.join("\n")).toMatch(/--publish: the config has no proposal/);
  }, 300_000);

  it("the pilot command records the uniform reference with --without-proposal", () => {
    const config = mixed(loose);
    const out = join(scratch, "ref-pilot.json");
    const run = cli(["pilot", "--config", write("mix-ref.json", config), "--labels", files.labels, "--format", FORMAT, "--out", out, "--generation-key", KEY, "--draws", "20", "--vocabulary", VOCABULARY_FILE, "--without-proposal"]);
    expect(run.code).toBe(0);
    const pilot = JSON.parse(readFileSync(out, "utf8")) as Pilot;
    expect(pilot.config_hash).toBe(recordHash(plainConfig(config)));
    expect(pilot.components).toBeUndefined();
    const withProposal = join(scratch, "mixed-pilot.json");
    expect(cli(["pilot", "--config", join(scratch, "mix-ref.json"), "--labels", files.labels, "--format", FORMAT, "--out", withProposal, "--generation-key", KEY, "--draws", "20", "--vocabulary", VOCABULARY_FILE]).code).toBe(0);
    expect((JSON.parse(readFileSync(withProposal, "utf8")) as Pilot).components?.map((c) => c.kind)).toEqual(["role_sets", "uniform"]);
  }, 300_000);
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
