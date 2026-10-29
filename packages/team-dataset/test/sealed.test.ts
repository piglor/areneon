/* eslint-disable @typescript-eslint/no-non-null-assertion -- test fixtures: the arrays are known to be filled */
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { inspect } from "node:util";
import { canonicalTeamHash, checkCommitment, exportPaste, importPaste, jcs, keyCommitment, readEvaluationConfig, readSealedSetCommitment, readSealedTeamSetManifest, recordHash, type TeamGeneratorConfig, setCommitment, stratumLabel, syntheticTeamHashes, teamHash } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it, vi } from "vitest";
import {
  fillSealed,
  generateSealedSet,
  loadSealedTeamHashes,
  NearIndex,
  parseSealedTeams,
  readStoredSet,
  referenceTeams,
  SEALED_FILES,
  SealedError,
  type SealedGenerateInput,
  SealedShortfallError,
  SealedTeamHashes,
  sealedBudget,
  sealedPlan,
  sealedQuotas,
  teamDistance,
} from "../src/sealed.ts";
import { Drawer } from "../src/draw.ts";
import { Proposal } from "../src/proposal.ts";
import { cfg, completeParameters, counterSecret, evaluation, evaluationRule, freshStore, publicManifest, publicTeams } from "./sealed-support.ts";
import { engine, format, light, table, vocabulary } from "./support.ts";

const stores: { remove: () => void }[] = [];
afterAll(() => stores.forEach((s) => s.remove()));
const store = (): string => {
  const s = freshStore();
  stores.push(s);
  return s.dir;
};

function input(over: Partial<SealedGenerateInput> = {}): SealedGenerateInput {
  return {
    engine,
    format,
    table,
    vocabulary,
    labelSet: light,
    config: cfg,
    evaluation,
    publicManifest,
    known: publicTeams,
    storeDir: store(),
    setId: "set-a",
    generation: 1,
    planId: "plan-a",
    generatorCommit: "b".repeat(40),
    maxAttempts: 3,
    secret: counterSecret("a"),
    ...over,
  };
}

const setsOf = (paste: string) => importPaste(engine, paste);
const set0 = setsOf(publicTeams[0]!.paste)[0]!;

describe("teamDistance (L2)", () => {
  const team = setsOf(publicTeams[0]!.paste);

  it("is 0 for a team and for any reordering of its sets", () => {
    expect(teamDistance(team, team)).toBe(0);
    expect(teamDistance(team, [...team].reverse())).toBe(0);
  });

  it("counts one differing field per field: nature, item, ability, moves, Stat Points, species", () => {
    const change = (edit: (s: (typeof team)[number]) => void) => {
      const copy = structuredClone(team);
      edit(copy[0]!);
      return teamDistance(team, copy);
    };
    expect(change((s) => (s.nature = s.nature === "Hardy" ? "Serious" : "Hardy"))).toBe(1);
    expect(change((s) => (s.item = `${s.item}x`))).toBe(1);
    expect(change((s) => (s.ability = `${s.ability}x`))).toBe(1);
    expect(change((s) => (s.moves = [...s.moves].reverse()))).toBe(0);
    expect(change((s) => (s.moves = [`${s.moves[0]}x`, ...s.moves.slice(1)]))).toBe(1);
    expect(change((s) => (s.evs = { ...s.evs, hp: s.evs.hp + 1 }))).toBe(1);
    expect(change((s) => (s.species = `${s.species}x`))).toBe(1);
    expect(change((s) => ((s.nature = s.nature === "Hardy" ? "Serious" : "Hardy"), (s.item = `${s.item}x`)))).toBe(2);
    // level, IVs, gender and nickname are not fields
    expect(change((s) => ((s.name = "Nick"), (s.gender = "F"), (s.level = 1)))).toBe(0);
  });

  it("pairs the sets as cheaply as possible, and counts an unpaired set as every field", () => {
    expect(teamDistance(team.slice(0, 3), team)).toBe(6 * (team.length - 3));
    expect(teamDistance([], team.slice(0, 1))).toBe(6);
    expect(() => teamDistance(Array.from({ length: 13 }, () => set0), [])).toThrow(/too large to match/);
  });
});

describe("NearIndex", () => {
  const pool = publicTeams.map((t) => setsOf(t.paste));
  const mutate = (team: typeof pool[number], fields: number) => {
    const copy = structuredClone(team);
    if (fields >= 1) copy[0]!.nature = copy[0]!.nature === "Hardy" ? "Serious" : "Hardy";
    if (fields >= 2) copy[1]!.item = `${copy[1]!.item}x`;
    if (fields >= 3) copy[2]!.ability = `${copy[2]!.ability}x`;
    return copy;
  };

  it("agrees with the brute-force distance for every bound", () => {
    const candidates = [pool[0]!, mutate(pool[0]!, 1), mutate(pool[0]!, 2), mutate(pool[1]!, 3), pool[2]!.slice(0, 2), setsOf(publicTeams[3]!.paste)];
    for (const bound of [0, 1, 2, 3, 4, 5]) {
      const index = new NearIndex(pool.slice(0, 3), bound);
      expect(index.size).toBe(3);
      for (const candidate of candidates) {
        const brute = pool.slice(0, 3).some((known) => teamDistance(candidate, known) <= bound);
        expect(index.near(candidate), `bound ${bound}`).toBe(brute);
      }
    }
  });

  it("uses d_team inclusively and compares every pair when the bound reaches the team size", () => {
    expect(new NearIndex([pool[0]!], 2).near(mutate(pool[0]!, 2))).toBe(true);
    expect(new NearIndex([pool[0]!], 2).near(mutate(pool[0]!, 3))).toBe(false);
    expect(new NearIndex([pool[0]!], 100).near(pool[1]!)).toBe(true);
  });

  it("refuses a bound that is not a non-negative integer", () => {
    expect(() => new NearIndex([], -1)).toThrow(/non-negative integer/);
    expect(() => new NearIndex([], 1.5)).toThrow(/non-negative integer/);
  });
});

describe("the sealed plan", () => {
  it("shares n_teams over the strata by pilot hits, each at least the floor, exactly", () => {
    expect(sealedQuotas([850, 1650, 200, 26300, 7800], 24, 2)).toEqual([2, 2, 2, 14, 4]);
    expect(sealedQuotas([1, 1, 1], 10, 2)).toEqual([4, 3, 3]);
    expect(sealedQuotas([5], 7, 2)).toEqual([7]);
    for (const total of [10, 11, 37, 100]) expect(sealedQuotas([3, 9, 1, 40], total, 2).reduce((a, b) => a + b, 0)).toBe(total);
    // ties go to the lower stratum
    expect(sealedQuotas([1, 1], 3, 1)).toEqual([2, 1]);
  });

  it("refuses too few teams for the floors, and a stratum with no pilot hit", () => {
    expect(() => sealedQuotas([5, 5, 5], 5, 2)).toThrow(/need 6 teams, n_teams is 5/);
    expect(() => sealedQuotas([5, 0], 8, 2)).toThrow(/never fills/);
  });

  it("over-provisions the public draw budget by the margin", () => {
    expect(sealedBudget(6000, 200)).toBe(7200);
    expect(sealedBudget(1001, 1)).toBe(1003);
    expect(sealedBudget(1000, 0)).toBe(1000);
  });

  it("is built on the public strata with their merge, and refuses a stratum the budget cannot fill", () => {
    const plan = sealedPlan({ publicManifest, teamCount: 24, nStratumMin: 2, marginPermille: 200, fillQuantilePermille: 990 });
    expect(plan.strata.map((s) => s.stratum)).toEqual(publicManifest.splits[0]!.cells.map((c) => c.stratum));
    expect(plan.strata.reduce((n, s) => n + s.quota, 0)).toBe(24);
    expect(plan.dMax).toBe(7200);
    const merged = publicManifest.merged_cells?.[0];
    expect(plan.stratumOfCell(merged?.cells[1] ?? 99)).toBe(merged?.stratum ?? 99);
    expect(plan.stratumOfCell(3)).toBe(merged === undefined ? 3 : plan.stratumOfCell(3));
    expect(() => sealedPlan({ publicManifest: { ...publicManifest, d_max: 100 }, teamCount: 24, nStratumMin: 2, marginPermille: 0, fillQuantilePermille: 990 })).toThrow(/over the sealed budget 100/);
    expect(() => sealedPlan({ publicManifest: { ...publicManifest, splits: [] as never }, teamCount: 24, nStratumMin: 2, marginPermille: 0, fillQuantilePermille: 990 })).toThrow(/no split/);
  });
});

describe("generating a sealed set in the store", () => {
  const run = generateSealedSet(input());
  const dir = run.dir;
  const files = readdirSync(dir).sort();
  const key = readFileSync(join(dir, SEALED_FILES.key), "utf8").trim();
  const salt = readFileSync(join(dir, SEALED_FILES.salt), "utf8").trim();

  it("commits a set of n_teams teams on the public strata, under L3, and holds the files in the store", () => {
    expect(files).toEqual(["commitment.json", "discards.jsonl", "key", "manifest.json", "rejections.jsonl", "salt", "teams.jsonl"]);
    const stored = readStoredSet(dirOfStore(dir), "set-a");
    expect(stored.teams).toHaveLength(24);
    expect(run.manifest.team_count).toBe(24);
    expect(run.commitment.team_count).toBe(24);
    checkCommitment(run.commitment, evaluation);
    expect(setCommitment(salt, stored.teams as never)).toBe(run.commitment.set_commitment);
    expect(keyCommitment(key)).toBe(run.commitment.key_commitment);
    expect(run.commitment.stratum_counts).toEqual(run.manifest.strata.map((s) => ({ stratum: stratumLabel(s.stratum), count: s.quota })));
    expect(readSealedTeamSetManifest(JSON.parse(readFileSync(join(dir, SEALED_FILES.manifest), "utf8")))).toEqual(run.manifest);
    expect(readSealedSetCommitment(JSON.parse(readFileSync(join(dir, SEALED_FILES.commitment), "utf8")))).toEqual(run.commitment);
    expect(run.manifest.discards).toEqual({ count: 0, log_sha256: expect.stringMatching(/^[0-9a-f]{64}$/) });
    expect(run.manifest.sealed_margin_permille).toBe(cfg.sealed_margin_permille);
    expect(run.discarded).toBe(0);
  });

  it("holds teams the engine accepts, each in its stratum, in stratum then draw order, none known", () => {
    const stored = readStoredSet(dirOfStore(dir), "set-a");
    const known = new Set(publicTeams.map((t) => t.canonical_team_hash));
    const seen = new Set<string>();
    let last = [-1, -1];
    for (const team of stored.teams) {
      const hash = canonicalTeamHash(setsOf(team.paste));
      expect(known.has(hash)).toBe(false);
      expect(seen.has(hash)).toBe(false);
      seen.add(hash);
      expect(team.stratum > last[0]! || (team.stratum === last[0] && team.k > last[1]!)).toBe(true);
      last = [team.stratum, team.k];
    }
    const counts = new Map<number, number>();
    for (const t of stored.teams) counts.set(t.stratum, (counts.get(t.stratum) ?? 0) + 1);
    for (const s of run.manifest.strata) expect(counts.get(s.stratum)).toBe(s.quota);
  });

  it("keeps the store private: directories 0700, files 0600", () => {
    expect(statSync(dir).mode & 0o7777).toBe(0o700);
    expect(statSync(join(dir, "..")).mode & 0o7777).toBe(0o700);
    for (const file of files) expect(statSync(join(dir, file)).mode & 0o7777, file).toBe(0o600);
  });

  it("is deterministic given the key and the salt: another store gets the same bytes", () => {
    const again = generateSealedSet(input());
    for (const file of files) expect(readFileSync(join(again.dir, file)), file).toEqual(readFileSync(join(dir, file)));
  });

  it("uses the OS CSPRNG by default: a key and a salt of 256 bits, different every run", () => {
    const a = generateSealedSet(input({ secret: undefined, setId: "set-r1" }));
    const b = generateSealedSet(input({ secret: undefined, setId: "set-r2" }));
    const k = (d: string) => readFileSync(join(d, SEALED_FILES.key), "utf8").trim();
    expect(k(a.dir)).toMatch(/^[0-9a-f]{64}$/);
    expect(k(a.dir)).not.toBe(k(b.dir));
    expect(a.commitment.key_commitment).not.toBe(b.commitment.key_commitment);
  });

  it("never lets the key, the salt or a team appear in a result, a progress event, or any output", () => {
    const writes: string[] = [];
    const spies = [
      vi.spyOn(process.stdout, "write").mockImplementation((chunk: unknown) => (writes.push(String(chunk)), true)),
      vi.spyOn(process.stderr, "write").mockImplementation((chunk: unknown) => (writes.push(String(chunk)), true)),
      ...(["log", "error", "warn", "info", "debug"] as const).map((name) => vi.spyOn(console, name).mockImplementation((...args: unknown[]) => void writes.push(args.map(String).join(" ")))),
    ];
    const events: unknown[] = [];
    const secretOf = counterSecret("canary");
    const made: string[] = [];
    const second = generateSealedSet(input({ secret: () => (made.push(secretOf()), made.at(-1) as string), setId: "set-c", progressEvery: 5, onProgress: (e) => events.push(e) }));
    spies.forEach((s) => s.mockRestore());
    const stored = readStoredSet(dirOfStore(second.dir), "set-c");
    const secrets = [stored.key, stored.salt, ...stored.teams.map((t) => t.paste)];
    const visible = [JSON.stringify(second), JSON.stringify(events), writes.join(""), JSON.stringify(second.manifest), JSON.stringify(second.commitment)].join("\n");
    for (const secret of secrets) expect(visible.includes(secret)).toBe(false);
    expect(events.length).toBeGreaterThan(0);
    expect(Object.keys(events[0] as object).sort()).toEqual(["attempt", "draws", "open"]);
    // the files other than key and salt do not contain them either
    for (const file of readdirSync(second.dir).filter((f) => f !== "key" && f !== "salt")) {
      const text = readFileSync(join(second.dir, file), "utf8");
      expect(text.includes(stored.key) || text.includes(stored.salt), file).toBe(false);
    }
    expect(made.length).toBe(2);
    expect(writes.join("")).toBe("");
  });

  it("warns, with counts only, of strata whose fill quantile is beyond the budget", () => {
    const tightPlan = sealedPlan({ publicManifest: { ...publicManifest, d_max: 334 }, teamCount: 24, nStratumMin: 2, marginPermille: 200, fillQuantilePermille: 990 });
    expect(tightPlan.warnings.length).toBeGreaterThan(0);
    expect(tightPlan.warnings[0]).toMatch(/^stratum \d+: the 990\/1000 fill quantile is \d+ draws, beyond the sealed budget 401/);
    expect(sealedPlan({ publicManifest, teamCount: 24, nStratumMin: 2, marginPermille: 200, fillQuantilePermille: 500 }).warnings).toEqual([]);
    expect(generateSealedSet(input({ publicManifest: { ...publicManifest, d_max: 334 }, setId: "set-w", maxAttempts: 40 })).warnings.length).toBeGreaterThan(0);
  });

  it("refuses a bad set id, generation or attempt count", () => {
    expect(() => generateSealedSet(input({ setId: "Set A" }))).toThrow(/set id must be/);
    // a config with a role_sets component is refused (the sealed generator takes no role-set table)
    const mixture: TeamGeneratorConfig = { ...cfg, proposal: { kind: "mixture", components: [{ kind: "role_sets", role_set_table_hash: "a".repeat(64), share_permille: 500 }, { kind: "uniform", share_permille: 500 }] } };
    const manifest = { ...publicManifest, generator: { ...publicManifest.generator, config_hash: recordHash(mixture) } };
    expect(() => generateSealedSet(input({ config: mixture, publicManifest: manifest, setId: "set-mix" }))).toThrow(/no role-set table/);
    expect(() => generateSealedSet(input({ generation: 0 }))).toThrow(/generation must be/);
    expect(() => generateSealedSet(input({ maxAttempts: 0 }))).toThrow(/max attempts/);
  });

  it("refuses a set that exists, and a store that is not private", () => {
    const s = store();
    generateSealedSet(input({ storeDir: s }));
    expect(() => generateSealedSet(input({ storeDir: s }))).toThrow(/cannot be created: EEXIST/);
    expect(() => generateSealedSet(input({ storeDir: "relative" }))).toThrow(/absolute/);
  });

  it("refuses provisional parameters before drawing anything", () => {
    const provisional = readEvaluationConfig({ ...completeParameters(), n_teams: null }, evaluationRule);
    const s = store();
    expect(() => generateSealedSet(input({ storeDir: s, evaluation: provisional }))).toThrow(/provisional/);
    expect(readdirSync(s)).toEqual([]);
  });

  it("refuses a public dataset made by another generator", () => {
    const g = publicManifest.generator;
    const other = (change: Partial<typeof g>) => ({ ...publicManifest, generator: { ...g, ...change } });
    expect(() => generateSealedSet(input({ publicManifest: other({ config_hash: "0".repeat(64) }) }))).toThrow(/another generator config/);
    expect(() => generateSealedSet(input({ publicManifest: other({ label_set_hash: "0".repeat(64) }) }))).toThrow(/another label set/);
    expect(() => generateSealedSet(input({ publicManifest: other({ vocabulary_hash: "0".repeat(64) }) }))).toThrow(/another vocabulary/);
    expect(() => generateSealedSet(input({ publicManifest: other({ manifest: { ...g.manifest, ruleset_hash: "0".repeat(64) } }) }))).toThrow(/another version manifest/);
  });

  it("refuses a public plan with a cell outside the label set, an unreadable known team, and a weak key", () => {
    const wide = { ...publicManifest, splits: publicManifest.splits.map((s) => ({ ...s, cells: s.cells.map((c, i) => (i === 0 ? { ...c, stratum: 9999 } : c)) })) } as unknown as typeof publicManifest;
    expect(() => generateSealedSet(input({ publicManifest: wide }))).toThrow(/outside the label set/);
    expect(() => generateSealedSet(input({ known: [{ paste: "not a team" }] }))).toThrow(/known team 1/);
    expect(() => generateSealedSet(input({ known: [{} as never] }))).toThrow(/known team 1 has no paste/);
    expect(() => generateSealedSet(input({ secret: () => "abc" }))).toThrow(/64 lowercase hex/);
    expect(() => generateSealedSet(input({ secret: (() => { const s = counterSecret("same"); const first = s(); let n = 0; return () => (n++ < 2 ? first : s()); })() }))).toThrow(/key and the salt must differ/);
  });
});

/** The store directory a set directory is in: `<store>/sealed/<id>`. */
const dirOfStore = (setDir: string): string => join(setDir, "..", "..");

describe("L1 and L2 against everything public", () => {
  const first = generateSealedSet(input({ setId: "set-l0", known: [] }));
  const teams = readStoredSet(dirOfStore(first.dir), "set-l0").teams;
  const logOf = (dir: string) => readFileSync(join(dir, SEALED_FILES.rejections), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as { k: number; reason: string });

  it("skips a candidate equal to a known team, and logs it as a duplicate", () => {
    const run = generateSealedSet(input({ setId: "set-l1", known: [{ paste: teams[0]!.paste }] }));
    const got = readStoredSet(dirOfStore(run.dir), "set-l1").teams;
    expect(got.map((t) => t.paste)).not.toContain(teams[0]!.paste);
    expect(logOf(run.dir).find((l) => l.k === teams[0]!.k)?.reason).toBe("duplicate");
    expect(run.manifest.rejection_log.duplicates).toBeGreaterThanOrEqual(1);
    expect(run.manifest.reference_pool_hash).not.toBe(first.manifest.reference_pool_hash);
  });

  it("skips a candidate within d_team of a known team, though its hash is new, and logs it as near", () => {
    const sets = setsOf(teams[0]!.paste);
    sets[0]!.nature = sets[0]!.nature === "Hardy" ? "Serious" : "Hardy";
    sets[1]!.item = `${sets[1]!.item}x`;
    const neighbour = exportPaste(engine, sets);
    expect(canonicalTeamHash(setsOf(neighbour))).not.toBe(canonicalTeamHash(setsOf(teams[0]!.paste)));
    const run = generateSealedSet(input({ setId: "set-l2", known: [{ paste: neighbour }] }));
    const got = readStoredSet(dirOfStore(run.dir), "set-l2").teams;
    expect(got.map((t) => t.paste)).not.toContain(teams[0]!.paste);
    expect(logOf(run.dir).find((l) => l.k === teams[0]!.k)?.reason).toBe("near_duplicate");
    expect(run.manifest.rejection_log.near_duplicates).toBeGreaterThanOrEqual(1);
    // three differing fields are out of reach of d_team 2: the team is kept
    sets[2]!.ability = `${sets[2]!.ability}x`;
    const far = generateSealedSet(input({ setId: "set-l3", known: [{ paste: exportPaste(engine, sets) }] }));
    expect(readStoredSet(dirOfStore(far.dir), "set-l3").teams.map((t) => t.paste)).toContain(teams[0]!.paste);
  });

  it("checks every public split, dataset and the training pool alike: a known team of any of them is skipped", () => {
    const run = generateSealedSet(input({ setId: "set-l4", known: [...publicTeams, { paste: teams[0]!.paste }] }));
    for (const t of readStoredSet(dirOfStore(run.dir), "set-l4").teams) {
      expect(publicTeams.some((p) => p.canonical_team_hash === canonicalTeamHash(setsOf(t.paste)))).toBe(false);
    }
  });
});

describe("a shortfall discards the whole generation", () => {
  // d_max 334 and the config's 20 percent margin give a budget of 401 draws, about the expected fill time of the scarcest
  // stratum (2 teams at 200 hits in 40,000): roughly two generations in five fall short.
  const tight = { ...publicManifest, d_max: 334 };
  const flat = cfg;

  it("logs the discard (hashes and counts), draws a new key and never merges", () => {
    let found = false;
    for (const label of ["t1", "t2", "t3", "t4", "t5", "t6"]) {
      const s = store();
      const run = generateSealedSet(input({ storeDir: s, publicManifest: tight, config: flat, maxAttempts: 40, secret: counterSecret(label), setId: "set-d" }));
      if (run.discarded === 0) continue;
      found = true;
      const lines = readFileSync(join(run.dir, SEALED_FILES.discards), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as { attempt: number; draws: number; key_commitment: string; short: number[] });
      expect(lines).toHaveLength(run.discarded);
      expect(lines.map((l) => l.attempt)).toEqual(lines.map((_, i) => i + 1));
      for (const l of lines) {
        expect(l.draws).toBe(401);
        expect(l.short.length).toBeGreaterThan(0);
        expect(l.key_commitment).toMatch(/^[0-9a-f]{64}$/);
        expect(Object.keys(l).sort()).toEqual(["attempt", "draws", "key_commitment", "short"]);
      }
      expect(run.manifest.discards.count).toBe(run.discarded);
      // the strata are the public ones: no merge happened
      expect(run.manifest.strata.map((x) => x.stratum)).toEqual(publicManifest.splits[0]!.cells.map((c) => c.stratum));
      // the committed key is the last one drawn, not a discarded one
      expect(lines.map((l) => l.key_commitment)).not.toContain(run.commitment.key_commitment);
      expect(readdirSync(run.dir)).not.toContain("rejections.attempt");
      break;
    }
    expect(found).toBe(true);
  });

  it("commits nothing when every attempt is discarded, and says where the log is", () => {
    let thrown: unknown;
    for (const label of ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]) {
      const s = store();
      try {
        generateSealedSet(input({ storeDir: s, publicManifest: tight, config: flat, maxAttempts: 1, secret: counterSecret(label), setId: "set-e" }));
      } catch (error) {
        thrown = error;
        const dir = join(s, "sealed", "set-e");
        expect(readdirSync(dir)).toEqual(["discards.jsonl"]);
        expect(readFileSync(join(dir, "discards.jsonl"), "utf8").split("\n").filter(Boolean)).toHaveLength(1);
        break;
      }
    }
    expect(thrown).toBeInstanceOf(SealedShortfallError);
    expect((thrown as Error).message).toMatch(/every one of 1 generations left a bin short; nothing was committed/);
    expect(thrown).toBeInstanceOf(SealedError);
  });

  it("fillSealed reports the short strata when the budget runs out", () => {
    const s = store();
    const plan = { ...sealedPlan({ publicManifest, teamCount: 24, nStratumMin: 2, marginPermille: 0, fillQuantilePermille: 990 }), dMax: 5 };
    const reference = referenceTeams(engine, [], 2);
    const logPath = join(s, "log");
    writeFileSync(logPath, "", { mode: 0o600 });
    const drawer = new Drawer({ engine, format, proposal: new Proposal(vocabulary, table, cfg), labelSet: light, generationKey: "11".repeat(32) });
    const result = fillSealed({ engine, drawer, stream: { config_hash: "0".repeat(64), plan_id: "p" }, plan, known: reference.known, near: reference.near, logPath });
    expect(result.filled).toBe(false);
    expect(result.draws).toBe(5);
    if (!result.filled) expect(result.short.length).toBeGreaterThan(0);
      });
});

describe("SealedTeamHashes (stamping inside the store)", () => {
  const run = generateSealedSet(input({ setId: "set-h" }));
  const root = dirOfStore(run.dir);
  const stored = readStoredSet(root, "set-h");

  it("holds the sealed teams' hashes and unions them with the public ones for stamping", () => {
    const hashes = loadSealedTeamHashes(engine, root, "set-h");
    expect(hashes.size).toBe(24);
    const publicHashes = syntheticTeamHashes(publicTeams.map((t) => t.team_hash));
    const union = hashes.withPublic(publicHashes);
    expect(union.size).toBe(publicHashes.size + 24);
    for (const t of stored.teams) expect(union.has(teamHash(setsOf(t.paste)))).toBe(true);
    for (const t of publicTeams) expect(union.has(t.team_hash)).toBe(true);
    expect(publicHashes.has(teamHash(setsOf(stored.teams[0]!.paste)))).toBe(false);
    expect(hashes.hasCanonical(canonicalTeamHash(setsOf(stored.teams[0]!.paste)))).toBe(true);
    expect(hashes.hasCanonical("0".repeat(64))).toBe(false);
  });

  it("cannot be serialised or printed", () => {
    const hashes = loadSealedTeamHashes(engine, root, "set-h");
    expect(() => JSON.stringify(hashes)).toThrow(/never be serialised/);
    expect(() => JSON.stringify({ nested: hashes })).toThrow(SealedError);
    expect(String(hashes)).toBe("SealedTeamHashes(24)");
    expect(inspect(hashes)).toBe("SealedTeamHashes(24)");
  });

  it("is built only after the L3 commitment is recomputed: a changed team, salt or key is refused", () => {
    const edit = (file: string, change: (text: string) => string) => {
      const path = join(run.dir, file);
      const old = readFileSync(path, "utf8");
      writeFileSync(path, change(old));
      try {
        expect(() => loadSealedTeamHashes(engine, root, "set-h")).toThrow(SealedError);
      } finally {
        writeFileSync(path, old);
      }
    };
    edit(SEALED_FILES.salt, (t) => `${"0".repeat(64)}\n${t.slice(65)}`);
    edit(SEALED_FILES.key, (t) => `${"0".repeat(64)}\n${t.slice(65)}`);
    edit(SEALED_FILES.teams, (t) => t.replace(/"k":\d+/, '"k":999999'));
    expect(loadSealedTeamHashes(engine, root, "set-h").size).toBe(24);
  });

  it("refuses a set whose records name another set, and a loose file", () => {
    const manifestPath = join(run.dir, SEALED_FILES.manifest);
    const old = readFileSync(manifestPath, "utf8");
    writeFileSync(manifestPath, old.replace('"set_id": "set-h"', '"set_id": "set-x"'));
    expect(() => readStoredSet(root, "set-h")).toThrow(/not set set-h/);
    writeFileSync(manifestPath, "{ not json");
    expect(() => readStoredSet(root, "set-h")).toThrow(/manifest.json is not valid JSON/);
    writeFileSync(manifestPath, old);
    expect(readStoredSet(root, "set-h").teams).toHaveLength(24);
  });

  it("parses teams.jsonl strictly", () => {
    const line = jcs({ k: 1, paste: "p", stratum: 2 });
    expect(parseSealedTeams(`${line}\n`)).toEqual([{ k: 1, paste: "p", stratum: 2 }]);
    expect(parseSealedTeams("")).toEqual([]);
    expect(() => parseSealedTeams(line)).toThrow(/newline/);
    expect(() => parseSealedTeams('{"stratum":2,"paste":"p","k":1}\n')).toThrow(/canonical/);
    expect(() => parseSealedTeams(`${jcs({ k: 1, paste: "p" })}\n`)).toThrow(/not a sealed team/);
    expect(() => parseSealedTeams(`${jcs({ k: 1.5, paste: "p", stratum: 1 })}\n`)).toThrow(/not a sealed team/);
  });
});

describe("the reason of a skipped candidate comes from the check that skipped it", () => {
  it("logs a, near, a as duplicate-free, near_duplicate, duplicate (a repeated binned team is not near)", () => {
    const [a, b, c] = publicTeams.slice(0, 3).map((t) => t.paste) as [string, string, string];
    const neighbour = importPaste(engine, b);
    neighbour[0]!.nature = neighbour[0]!.nature === "Hardy" ? "Serious" : "Hardy";
    const reference = referenceTeams(engine, [{ paste: exportPaste(engine, neighbour) }], 2);
    const order = [a, b, a, c];
    const drawer = {
      draw: (_stream: unknown, k: number) => ({ status: "accepted", k, canonical_team_hash: canonicalTeamHash(importPaste(engine, order[k]!)), team_hash: "", paste: order[k]!, labels: [], cell: 0 }),
    } as unknown as Drawer;
    const plan = { teamCount: 3, strata: [{ stratum: 0, pilot_hits: 1, quota: 3 }], mergedCells: [], stratumOfCell: () => 0, dMax: 4, pilotDraws: 1, warnings: [] };
    const dir = store();
    const logPath = join(dir, "log");
    writeFileSync(logPath, "", { mode: 0o600 });
    const result = fillSealed({ engine, drawer, stream: { config_hash: "0".repeat(64), plan_id: "p" }, plan, known: reference.known, near: reference.near, logPath });
    const reasons = readFileSync(logPath, "utf8").split("\n").filter(Boolean).map((l) => (JSON.parse(l) as { k: number; reason: string }));
    expect(reasons).toEqual([
      { k: 1, reason: "near_duplicate", canonical_team_hash: expect.any(String), stratum: 0 },
      { k: 2, reason: "duplicate", canonical_team_hash: expect.any(String), stratum: 0 },
    ]);
    expect(result.draws).toBe(4);
  });
});

describe("the test seams for the key source", () => {
  it("are never used by production code: the script passes none, and only the command forwards the injected one", () => {
    const root = new URL("../", import.meta.url);
    const script = readFileSync(new URL("scripts/sealed.ts", root), "utf8");
    expect(script).not.toMatch(/secret/);
    const sources = readdirSync(new URL("src/", root)).filter((f) => f.startsWith("sealed")).map((f) => readFileSync(new URL(`src/${f}`, root), "utf8"));
    const passes = sources.join("\n").match(/\bsecret: [a-z]+\.[a-z]+/g) ?? [];
    expect(passes).toHaveLength(1);
    expect(sources.join("\n")).toContain("secret: io.secret");
  });
});

describe("SealedTeamHashes accepts only a set read from the store", () => {
  it("refuses a hand-built set", () => {
    const run = generateSealedSet(input({ setId: "set-b" }));
    const real = readStoredSet(dirOfStore(run.dir), "set-b");
    expect(() => SealedTeamHashes.fromVerified(engine, { ...real })).toThrow(/did not return is not trusted/);
    expect(SealedTeamHashes.fromVerified(engine, real).size).toBe(24);
  });

  it("a set read from the store cannot be mutated to change what it trusts", () => {
    const run = generateSealedSet(input({ setId: "set-f" }));
    const real = readStoredSet(dirOfStore(run.dir), "set-f");
    const paste = real.teams[0]?.paste;
    expect(Object.isFrozen(real)).toBe(true);
    expect(Object.isFrozen(real.teams)).toBe(true);
    expect(real.teams.every((t) => Object.isFrozen(t))).toBe(true);
    // sloppy-mode writes would be silent no-ops; these modules are strict, so each mutation throws
    expect(() => (real.teams as unknown as unknown[]).push({ k: 99, paste: "x", stratum: 0 })).toThrow(TypeError);
    expect(() => ((real.teams[0] as { paste: string }).paste = "x")).toThrow(TypeError);
    expect(() => ((real as { teams: unknown }).teams = [])).toThrow(TypeError);
    expect(real.teams[0]?.paste).toBe(paste);
    expect(SealedTeamHashes.fromVerified(engine, real).size).toBe(24);
  });
});
