import { chmodSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { appendEntry } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { runSealedCli, type SealedCliIo } from "../src/sealed-command.ts";
import { cfg, completeParameters, counterSecret, evaluation, evaluationRule, freshStore, publicManifest, publicTeams } from "./sealed-support.ts";
import { engine, FORMAT, light, VOCABULARY_FILE } from "./support.ts";

const scratch = mkdtempSync(join(tmpdir(), "sealed-cli-"));
const stores: { remove: () => void }[] = [];
afterAll(() => {
  rmSync(scratch, { recursive: true, force: true });
  stores.forEach((s) => s.remove());
});

const files = {
  config: join(scratch, "config.json"),
  labels: join(scratch, "labels.json"),
  manifest: join(scratch, "public-manifest.json"),
  parameters: join(scratch, "parameters.json"),
  rule: join(scratch, "rule.json"),
  known: join(scratch, "known.jsonl"),
};
writeFileSync(files.config, JSON.stringify(cfg));
writeFileSync(files.labels, JSON.stringify(light));
writeFileSync(files.manifest, JSON.stringify(publicManifest));
writeFileSync(files.parameters, JSON.stringify(completeParameters()));
writeFileSync(files.rule, JSON.stringify(evaluationRule));
writeFileSync(files.known, publicTeams.map((t) => `${JSON.stringify(t)}\n`).join(""));

function cli(args: string[], extra: Partial<SealedCliIo> = {}): { code: number; out: string[]; err: string[] } {
  const out: string[] = [];
  const err: string[] = [];
  const code = runSealedCli(args, { out: (l) => out.push(l), err: (l) => err.push(l), engine, ...extra });
  return { code, out, err };
}

const store = (): string => {
  const s = freshStore();
  stores.push(s);
  return s.dir;
};

const generateArgs = (storeDir: string, more: string[] = []): string[] => [
  "generate", "--store", storeDir, "--set-id", "set-c", "--generation", "1", "--public-manifest", files.manifest,
  "--parameters", files.parameters, "--rule", files.rule, "--config", files.config, "--labels", files.labels, "--format", FORMAT,
  "--vocabulary", VOCABULARY_FILE, "--known", files.known, "--generator-commit", "b".repeat(40), ...more,
];

describe("the sealed command line", () => {
  const storeDir = store();
  const secret = counterSecret("cli");
  const issued: string[] = [];
  const run = cli(generateArgs(storeDir, ["--progress-every", "20"]), { secret: () => (issued.push(secret()), issued.at(-1) as string) });
  const setDir = join(storeDir, "sealed", "set-c");

  it("generates a set and prints hashes and counts only", () => {
    expect(run.code).toBe(0);
    const summary = JSON.parse(run.out[0] as string) as Record<string, unknown>;
    expect(summary).toMatchObject({ set_id: "set-c", team_count: 24, discarded: 0 });
    expect(Object.keys(summary).sort()).toEqual(["commitment_hash", "discarded", "key_commitment", "set_commitment", "set_id", "set_manifest_hash", "team_count"]);
    expect(run.err.some((l) => /^attempt 1: \d+ draws, \d+ bins open$/.test(l))).toBe(true);
  });

  it("never prints the key, the salt or a team (canary), and the repository-side scratch holds none of them", () => {
    const [key, salt] = issued as [string, string];
    const printed = [...run.out, ...run.err].join("\n");
    expect(printed.includes(key) || printed.includes(salt)).toBe(false);
    const teams = readFileSync(join(setDir, "teams.jsonl"), "utf8");
    for (const line of teams.split("\n").filter(Boolean)) expect(printed.includes((JSON.parse(line) as { paste: string }).paste)).toBe(false);
    expect(readFileSync(join(setDir, "key"), "utf8").trim()).toBe(key);
    for (const file of readdirSync(scratch)) {
      const text = readFileSync(join(scratch, file), "utf8");
      expect(text.includes(key) || text.includes(salt), file).toBe(false);
    }
  });

  it("prints the reveal hash, and retires and verifies once the register has retired the set", () => {
    const hashRun = cli(["reveal-hash", "--store", storeDir, "--set-id", "set-c"]);
    expect(hashRun.code).toBe(0);
    const { reveal_hash: hash } = JSON.parse(hashRun.out[0] as string) as { reveal_hash: string };
    const commitment = JSON.parse(readFileSync(join(setDir, "commitment.json"), "utf8"));
    const at = { recorded_at: "2026-11-01T10:00:00Z", operator: "owner" };
    let text = appendEntry("", { kind: "set_commitment", commitment, ...at } as never, [evaluation]).text;
    text = appendEntry(text, { kind: "incident", set_id: "set-c", cause: "sealed_artifact_read", description: "A test incident.", affected_candidates: [], ...at } as never, [evaluation]).text;
    text = appendEntry(text, { kind: "set_retirement", set_id: "set-c", reason: "contaminated", reveal_hash: hash, ...at } as never, [evaluation]).text;
    const registerFile = join(scratch, "register.jsonl");
    writeFileSync(registerFile, text);
    const out = join(scratch, "bundle");
    const retired = cli(["retire", "--store", storeDir, "--set-id", "set-c", "--out", out, "--register", registerFile, "--parameters", files.parameters, "--rule", files.rule]);
    expect(retired.code).toBe(0);
    expect(JSON.parse(retired.out[0] as string)).toMatchObject({ set_id: "set-c", reveal_hash: hash });
    const verified = cli(["verify", "--dir", out, "--parameters", files.parameters, "--rule", files.rule, "--config", files.config, "--labels", files.labels, "--format", FORMAT, "--vocabulary", VOCABULARY_FILE, "--known", files.known]);
    expect(verified.code).toBe(0);
    expect(JSON.parse(verified.out[0] as string)).toMatchObject({ verified: true, set_id: "set-c", teams: 24 });
    // a tampered reveal exits 1 and the message does not quote the file
    const bad = join(scratch, "bundle-bad");
    mkdirSync(bad);
    for (const f of readdirSync(out)) writeFileSync(join(bad, f), readFileSync(join(out, f), "utf8"));
    writeFileSync(join(bad, "teams.jsonl"), `${readFileSync(join(out, "teams.jsonl"), "utf8")}\n`);
    const refused = cli(["verify", "--dir", bad, "--parameters", files.parameters, "--rule", files.rule, "--config", files.config, "--labels", files.labels, "--format", FORMAT, "--vocabulary", VOCABULARY_FILE, "--known", files.known]);
    expect(refused.code).toBe(1);
    expect(refused.err.join()).toMatch(/SealedVerifyError: teams.jsonl is not the file the reveal names/);
  });
});

describe("the sealed command line refuses", () => {
  it("a store that group or other can read, with exit 1, before any secret is made", () => {
    const s = store();
    chmodSync(s, 0o755);
    let drawn = 0;
    const run = cli(generateArgs(s), { secret: () => (drawn++, "1".repeat(64)) });
    expect(run.code).toBe(1);
    expect(run.err.join()).toMatch(/SealedStoreError: .*permission bits/);
    expect(drawn).toBe(0);
    expect(readdirSync(s)).toEqual([]);
  });

  it("provisional parameters, with exit 1", () => {
    const provisional = join(scratch, "provisional.json");
    writeFileSync(provisional, JSON.stringify({ ...completeParameters(), n_teams: null }));
    const args = generateArgs(store());
    args[args.indexOf("--parameters") + 1] = provisional;
    const run = cli(args);
    expect(run.code).toBe(1);
    expect(run.err.join()).toMatch(/provisional/);
  });

  it.each([
    [["nonsense"], /usage/],
    [["generate", "extra"], /usage/],
    [["generate"], /--parameters and --rule are required/],
    [["reveal-hash", "--set-id", "a"], /--store is required/],
    [["generate", "--bogus"], /Unknown option/],
  ])("%j with exit 2", (args, message) => {
    const run = cli(args as string[]);
    expect(run.code).toBe(2);
    expect(run.err.join()).toMatch(message);
  });

  it("bad numbers, a bad commit, two parameter files, a known file without pastes, with exit 2", () => {
    const withArgs = (more: string[]) => cli(generateArgs(store(), more));
    expect(withArgs(["--max-attempts", "x"]).err.join()).toMatch(/--max-attempts must be a whole number/);
    expect(withArgs(["--generator-commit", "abc"]).err.join()).toMatch(/40-digit/);
    expect(withArgs(["--parameters", files.parameters, "--rule", files.rule]).err.join()).toMatch(/one --parameters and one --rule/);
    const noPaste = join(scratch, "nopaste.jsonl");
    writeFileSync(noPaste, '{"x":1}\n');
    expect(withArgs(["--known", noPaste]).err.join()).toMatch(/line 1 has no paste/);
    const args = generateArgs(store());
    args.splice(args.indexOf("--generation"), 2);
    expect(cli(args).err.join()).toMatch(/--generation is required/);
    const two = cli(["verify", "--dir", scratch, "--parameters", files.parameters, "--parameters", files.parameters, "--rule", files.rule, "--rule", files.rule, "--config", files.config, "--labels", files.labels, "--format", FORMAT]);
    expect(two.code).toBe(2);
    expect(two.err.join()).toMatch(/one --parameters and one --rule/);
  });

  it("the pinned engine is loaded when none is injected", () => {
    const out: string[] = [];
    const err: string[] = [];
    const code = runSealedCli(["verify", "--dir", join(scratch, "missing"), "--parameters", files.parameters, "--rule", files.rule, "--config", files.config, "--labels", files.labels, "--format", FORMAT, "--vocabulary", VOCABULARY_FILE], { out: (l) => out.push(l), err: (l) => err.push(l) });
    expect(code).toBe(2);
    expect(err.join()).toMatch(/ENOENT/);
  });
});
