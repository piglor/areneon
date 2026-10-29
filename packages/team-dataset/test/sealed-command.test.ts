import { chmodSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { appendEntry } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it, vi } from "vitest";
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

const writeHead = (dir: string, head: { seq: number; entry_hash: string }): void => writeFileSync(join(dir, `${String(head.seq).padStart(10, "0")}.json`), JSON.stringify(head));

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
    expect(Object.keys(summary).sort()).toEqual(["attempts", "commitment_hash", "discarded", "key_commitment", "set_commitment", "set_id", "set_manifest_hash", "team_count"]);
    expect(run.err.some((l) => /^attempt 1: \d+ draws, \d+ bins open$/.test(l))).toBe(true);
  });

  it("never prints the key, the salt or a team (canary), and the repository-side scratch holds none of them", () => {
    const [key, salt] = issued as [string, string];
    const printed = [...run.out, ...run.err].join("\n");
    expect(printed.includes(key) || printed.includes(salt)).toBe(false);
    const teams = readFileSync(join(setDir, "teams.jsonl"), "utf8");
    for (const line of teams.split("\n").filter(Boolean)) expect(printed.includes((JSON.parse(line) as { paste: string }).paste)).toBe(false);
    expect(readFileSync(join(setDir, "key"), "utf8").trim()).toBe(key);
    for (const file of readdirSync(scratch, { withFileTypes: true }).filter((d) => d.isFile()).map((d) => d.name)) {
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
    const last = appendEntry(text, { kind: "set_retirement", set_id: "set-c", reason: "contaminated", reveal_hash: hash, ...at } as never, [evaluation]);
    text = last.text;
    const registerFile = join(scratch, "register.jsonl");
    writeFileSync(registerFile, text);
    const headsDir = join(scratch, "heads");
    mkdirSync(headsDir);
    writeHead(headsDir, last.head);
    const out = join(scratch, "bundle");
    const retired = cli(["retire", "--store", storeDir, "--set-id", "set-c", "--out", out, "--register", registerFile, "--heads", headsDir, "--parameters", files.parameters, "--rule", files.rule]);
    expect(retired.code).toBe(0);
    expect(JSON.parse(retired.out[0] as string)).toMatchObject({ set_id: "set-c", reveal_hash: hash });
    const verified = cli(["verify", "--dir", out, "--public-manifest", files.manifest, "--register", registerFile, "--heads", headsDir, "--parameters", files.parameters, "--rule", files.rule, "--config", files.config, "--labels", files.labels, "--format", FORMAT, "--vocabulary", VOCABULARY_FILE, "--known", files.known]);
    expect(verified.code).toBe(0);
    expect(JSON.parse(verified.out[0] as string)).toMatchObject({ verified: true, set_id: "set-c", teams: 24 });
    // a tampered reveal exits 1 and the message does not quote the file
    const bad = join(scratch, "bundle-bad");
    mkdirSync(bad);
    for (const f of readdirSync(out)) writeFileSync(join(bad, f), readFileSync(join(out, f), "utf8"));
    writeFileSync(join(bad, "teams.jsonl"), `${readFileSync(join(out, "teams.jsonl"), "utf8")}\n`);
    const refused = cli(["verify", "--dir", bad, "--public-manifest", files.manifest, "--register", registerFile, "--heads", headsDir, "--parameters", files.parameters, "--rule", files.rule, "--config", files.config, "--labels", files.labels, "--format", FORMAT, "--vocabulary", VOCABULARY_FILE, "--known", files.known]);
    expect(refused.code).toBe(1);
    expect(refused.err.join()).toMatch(/SealedVerifyError: teams.jsonl is not the file the reveal names/);
  });
});

describe("the sealed command line writes nowhere but its io", () => {
  it("prints nothing to the console or the process streams, whatever it handles", () => {
    const writes: string[] = [];
    const spies = [
      vi.spyOn(process.stdout, "write").mockImplementation((c: unknown) => (writes.push(String(c)), true)),
      vi.spyOn(process.stderr, "write").mockImplementation((c: unknown) => (writes.push(String(c)), true)),
      ...(["log", "error", "warn", "info", "debug"] as const).map((n) => vi.spyOn(console, n).mockImplementation((...a: unknown[]) => void writes.push(a.map(String).join(" ")))),
    ];
    const seen: string[] = [];
    const next = counterSecret("quiet");
    const run = cli(generateArgs(store()), { secret: () => (seen.push(next()), seen.at(-1) as string) });
    spies.forEach((x) => x.mockRestore());
    expect(run.code).toBe(0);
    expect(writes).toEqual([]);
    expect([...run.out, ...run.err].join("\n")).not.toContain(seen[0] as string);
    expect(JSON.parse(run.out[0] as string)).toMatchObject({ attempts: 1 });
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

describe("retire and verify enforce the published chain heads", () => {
  const storeDir = store();
  cli(generateArgs(storeDir), { secret: counterSecret("heads") });
  const setDir = join(storeDir, "sealed", "set-c");
  const hash = (JSON.parse(cli(["reveal-hash", "--store", storeDir, "--set-id", "set-c"]).out[0] as string) as { reveal_hash: string }).reveal_hash;
  const commitment = JSON.parse(readFileSync(join(setDir, "commitment.json"), "utf8"));
  const at = { recorded_at: "2026-11-01T10:00:00Z", operator: "owner" };
  const first = appendEntry("", { kind: "set_commitment", commitment, ...at } as never, [evaluation]);
  const second = appendEntry(first.text, { kind: "incident", set_id: "set-c", cause: "sealed_artifact_read", description: "A test incident.", affected_candidates: [], ...at } as never, [evaluation]);
  const third = appendEntry(second.text, { kind: "set_retirement", set_id: "set-c", reason: "contaminated", reveal_hash: hash, ...at } as never, [evaluation]);
  const registerFile = join(scratch, "heads-register.jsonl");
  writeFileSync(registerFile, third.text);
  const bundle = join(scratch, "heads-bundle");
  const retireArgs = (out: string, heads: string[]): string[] => ["retire", "--store", storeDir, "--set-id", "set-c", "--out", out, "--register", registerFile, ...heads, "--parameters", files.parameters, "--rule", files.rule];
  const headsIn = (...heads: { seq: number; entry_hash: string }[]): string => {
    const dir = mkdtempSync(join(scratch, "h-"));
    heads.forEach((h) => writeHead(dir, h));
    return dir;
  };
  const retired = cli(retireArgs(bundle, ["--heads", headsIn(first.head, third.head)]));
  const verifyArgs = (heads: string[]): string[] => ["verify", "--dir", bundle, "--public-manifest", files.manifest, "--register", registerFile, ...heads, "--parameters", files.parameters, "--rule", files.rule, "--config", files.config, "--labels", files.labels, "--format", FORMAT, "--vocabulary", VOCABULARY_FILE, "--known", files.known];
  it("passes with the heads of the register", () => {
    expect(retired.code).toBe(0);
    expect(cli(verifyArgs(["--heads", headsIn(first.head, third.head)])).code).toBe(0);
  });

  it("fails closed with exit 2 when no head is published: an absent or empty directory, or a ref with none", () => {
    const empty = mkdtempSync(join(scratch, "h-empty-"));
    for (const dir of [join(scratch, "no-such-heads-dir"), empty]) {
      const run = cli(verifyArgs(["--heads", dir]));
      expect(run.code).toBe(2);
      expect(run.err.join()).toMatch(/EvaluationConfigError: no published chain head was found/);
    }
    // a real ref (HEAD) that holds no head files at that path
    const ref = cli(verifyArgs(["--heads", "no/such/heads/path", "--heads-ref", "HEAD"]));
    expect(ref.code).toBe(2);
    expect(ref.err.join()).toMatch(/no published chain head was found/);
  });

  it("a malformed head file is a fixed-message config error (exit 2) that does not quote the text", () => {
    const dir = mkdtempSync(join(scratch, "h-bad-"));
    writeFileSync(join(dir, "0000000001.json"), '{"seq": MARKER-884-HEAD oops');
    const run = cli(verifyArgs(["--heads", dir]));
    expect(run.code).toBe(2);
    expect(run.err.join()).toMatch(/a published head file is not valid JSON/);
    expect(run.err.join()).not.toContain("MARKER-884-HEAD");
  });

  it("refuses without --heads (exit 2) and --heads-ref without --heads", () => {
    const missing = cli(verifyArgs([]));
    expect(missing.code).toBe(2);
    expect(missing.err.join()).toMatch(/--heads is required/);
    expect(cli(verifyArgs(["--heads-ref", "origin/main"])).code).toBe(2);
  });

  it("refuses a rollback (a head beyond the register) and a fork (another hash at a position), exit 1", () => {
    const rollback = cli(verifyArgs(["--heads", headsIn({ ...third.head, seq: 4 })]));
    expect(rollback.code).toBe(1);
    expect(rollback.err.join()).toMatch(/RegisterError: rollback/);
    const fork = cli(verifyArgs(["--heads", headsIn({ ...second.head, entry_hash: "f".repeat(64) })]));
    expect(fork.code).toBe(1);
    expect(fork.err.join()).toMatch(/RegisterError: fork or rewrite/);
  });

  describe("retire checks the heads before it writes anything", () => {
    const refusedOut = (name: string): string => join(scratch, name);
    const noWrite = (out: string): void => expect(existsSync(out)).toBe(false);

    it("passes with the heads of the register and writes the bundle", () => {
      expect(retired.code).toBe(0);
      expect(existsSync(join(bundle, "retired-manifest.json"))).toBe(true);
    });

    it("refuses a rollback (exit 1) and a fork (exit 1) and writes no file", () => {
      const a = refusedOut("retire-rollback");
      const rollback = cli(retireArgs(a, ["--heads", headsIn({ ...third.head, seq: 4 })]));
      expect(rollback.code).toBe(1);
      expect(rollback.err.join()).toMatch(/RegisterError: rollback/);
      noWrite(a);
      const b = refusedOut("retire-fork");
      const fork = cli(retireArgs(b, ["--heads", headsIn({ ...second.head, entry_hash: "f".repeat(64) })]));
      expect(fork.code).toBe(1);
      expect(fork.err.join()).toMatch(/RegisterError: fork or rewrite/);
      noWrite(b);
    });

    it("fails closed (exit 2) on absent or empty heads, a malformed head file, a missing --heads and an unknown ref; no file is written", () => {
      const empty = mkdtempSync(join(scratch, "h-empty-r-"));
      const bad = mkdtempSync(join(scratch, "h-bad-r-"));
      writeFileSync(join(bad, "0000000001.json"), '{"seq": MARKER-1242-HEAD oops');
      const cases: [string, string[], RegExp][] = [
        ["retire-absent", ["--heads", join(scratch, "no-such-heads-dir")], /no published chain head was found/],
        ["retire-empty", ["--heads", empty], /no published chain head was found/],
        ["retire-bad", ["--heads", bad], /a published head file is not valid JSON/],
        ["retire-missing", [], /--heads is required/],
        ["retire-ref", ["--heads", "evaluation/register-heads", "--heads-ref", "no-such-ref-1242"], /git ls-tree no-such-ref-1242 failed/],
      ];
      for (const [name, heads, message] of cases) {
        const out = refusedOut(name);
        const run = cli(retireArgs(out, heads));
        expect(run.code).toBe(2);
        expect(run.err.join()).toMatch(message);
        expect(run.err.join()).not.toContain("MARKER-1242-HEAD");
        noWrite(out);
      }
    });
  });

  it("reads the heads from a git ref when --heads-ref is given (an unknown ref is refused)", () => {
    const run = cli(verifyArgs(["--heads", "evaluation/register-heads", "--heads-ref", "no-such-ref-884"]));
    expect(run.code).toBe(2);
    expect(run.err.join()).toMatch(/git ls-tree no-such-ref-884 failed/);
  });
});

describe("no JSON read of the command quotes its input", () => {
  const MARKER = "MARKER-884-SECRET";
  const bad = (name: string, text: string): string => {
    const path = join(scratch, name);
    writeFileSync(path, text);
    return path;
  };

  it("a malformed config, parameters or labels file is a usage error that names the file, not the text", () => {
    for (const option of ["config", "labels", "public-manifest", "parameters", "rule"] as const) {
      const args = generateArgs(store());
      args[args.indexOf(`--${option}`) + 1] = bad(`bad-${option}.json`, `{"key": "${MARKER}" oops`);
      const run = cli(args);
      expect(run.code, option).toBe(2);
      expect(run.err.join(), option).toMatch(/is not valid JSON/);
      expect([...run.out, ...run.err].join("\n"), option).not.toContain(MARKER);
    }
  });

  it("a malformed --known line is a usage error that names the line, not the text", () => {
    const run = cli(generateArgs(store(), ["--known", bad("bad-known.jsonl", `{"paste": "${MARKER}" oops\n`)]));
    expect(run.code).toBe(2);
    expect(run.err.join()).toMatch(/line 1 is not valid JSON/);
    expect(run.err.join()).not.toContain(MARKER);
  });

  it("a literal null known line is a missing paste, not a crash", () => {
    const run = cli(generateArgs(store(), ["--known", bad("null-known.jsonl", "null\n")]));
    expect(run.code).toBe(2);
    expect(run.err.join()).toMatch(/line 1 has no paste/);
  });
});

describe("retire and verify print no key, salt, paste or team content", () => {
  const storeDir = store();
  const secrets: string[] = [];
  const next = counterSecret("canary");
  cli(generateArgs(storeDir), { secret: () => (secrets.push(next()), secrets.at(-1) as string) });
  const setDir = join(storeDir, "sealed", "set-c");
  const pastes = readFileSync(join(setDir, "teams.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => (JSON.parse(l) as { paste: string }).paste);
  const hash = (JSON.parse(cli(["reveal-hash", "--store", storeDir, "--set-id", "set-c"]).out[0] as string) as { reveal_hash: string }).reveal_hash;
  const commitment = JSON.parse(readFileSync(join(setDir, "commitment.json"), "utf8"));
  const at = { recorded_at: "2026-11-01T10:00:00Z", operator: "owner" };
  const committed = appendEntry("", { kind: "set_commitment", commitment, ...at } as never, [evaluation]);
  const incident = appendEntry(committed.text, { kind: "incident", set_id: "set-c", cause: "sealed_artifact_read", description: "A test incident.", affected_candidates: [], ...at } as never, [evaluation]);
  const withRetirement = appendEntry(incident.text, { kind: "set_retirement", set_id: "set-c", reason: "contaminated", reveal_hash: hash, ...at } as never, [evaluation]);
  const registerFile = join(scratch, "canary-register.jsonl");
  writeFileSync(registerFile, withRetirement.text);
  const headsDir = join(scratch, "canary-heads");
  mkdirSync(headsDir);
  writeHead(headsDir, withRetirement.head);
  const out = join(scratch, "canary-bundle");
  const retire = cli(["retire", "--store", storeDir, "--set-id", "set-c", "--out", out, "--register", registerFile, "--heads", headsDir, "--parameters", files.parameters, "--rule", files.rule]);
  const verifyArgs = (dir: string): string[] => ["verify", "--dir", dir, "--public-manifest", files.manifest, "--register", registerFile, "--heads", headsDir, "--parameters", files.parameters, "--rule", files.rule, "--config", files.config, "--labels", files.labels, "--format", FORMAT, "--vocabulary", VOCABULARY_FILE, "--known", files.known];
  const verify = cli(verifyArgs(out));
  const tampered = join(scratch, "canary-bundle-tampered");
  mkdirSync(tampered);
  for (const f of readdirSync(out)) writeFileSync(join(tampered, f), readFileSync(join(out, f), "utf8"));
  writeFileSync(join(tampered, "teams.jsonl"), `${readFileSync(join(out, "teams.jsonl"), "utf8")}\n`);
  const refused = cli(verifyArgs(tampered));
  const leaks = (text: string): boolean => secrets.some((x) => text.includes(x)) || pastes.some((p) => text.includes(p) || text.includes(p.split("\n")[0] as string));
  const streams = (run: { out: string[]; err: string[] }): string => [...run.out, ...run.err].join("\n");

  it("neither command, in success or refusal, prints the key, the salt or a team", () => {
    expect(retire.code).toBe(0);
    expect(verify.code).toBe(0);
    expect(refused.code).toBe(1);
    for (const run of [retire, verify, refused]) expect(leaks(streams(run))).toBe(false);
  });

  it("negative control: the scan does flag output that carries a key or a paste", () => {
    expect(leaks(`${streams(verify)} ${secrets[0]}`)).toBe(true);
    expect(leaks(`${streams(verify)} ${pastes[0]}`)).toBe(true);
    // and the bundle on disk does hold them, so the control is not vacuous
    expect(leaks(readFileSync(join(out, "teams.jsonl"), "utf8"))).toBe(true);
    expect(leaks(readFileSync(join(out, "retired-manifest.json"), "utf8"))).toBe(true);
  });
});
