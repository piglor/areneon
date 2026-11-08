// The private pool of the loop (ADR-015 erratum E1 item 3), on stand-in teams only:
// the canary world's synthetic teams with marker nicknames, in a stand-in snapshot.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  canonicalJson,
  canonicalTeamHash,
  classOfSets,
  exportPaste,
  holdoutFlag,
  importPaste,
  interimPoolSource,
  planGeneration,
  poolShape,
  type PrivateTeamRecord,
  readTrainingRunConfig,
  teamVocabularyHash,
  stampTeam,
  teamHash,
} from "@areneon/showdown-bridge";
import { labelTeam, loadFixtureInputs } from "@areneon/team-dataset";
import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { expectNoLeak, findLeaks, makeCanaryWorld, observe, worldIsPrivate } from "../../showdown-bridge/test/support/private-canary.ts";
import { writeLocalCopy } from "../src/local-copy.ts";
import { loadPrivatePool, PrivatePoolRefusal, withPrivatePool } from "../src/private-pool.ts";
import { writePrivateSnapshot } from "../src/snapshot-writer.ts";
import { engine, FORMAT_ID, HOLDOUT_KEY, hex32, input, labelSet, ROOT, standIn, workDir } from "./snapshot-support.ts";

vi.setConfig({ testTimeout: 90_000 });

const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const config = readTrainingRunConfig(tiny);
const vocabularyHash = teamVocabularyHash(loadFixtureInputs(engine, FORMAT_ID).vocabulary);
const world = makeCanaryWorld(engine, 4, { legal: true });

function recordOfPaste(paste: string, id: number, status: "active" | "retired" = "active"): PrivateTeamRecord {
  const sets = importPaste(engine, paste);
  return {
    schema_version: 1,
    data_class: "private",
    team_id: hex32(id),
    revision: 1,
    supersedes_record_hash: null,
    team_hash: teamHash(sets),
    canonical_team_hash: canonicalTeamHash(sets),
    paste: exportPaste(engine, sets),
    status,
    tags: [],
    holdout: holdoutFlag(HOLDOUT_KEY, hex32(id)),
    labels: labelTeam(labelSet, sets).map((l) => ({ label_id: l.label_id, value: l.value })),
  } as PrivateTeamRecord;
}

async function localCopy(records: PrivateTeamRecord[]): Promise<{ dir: string; hash: string }> {
  const w = await writePrivateSnapshot(input({ records, vocabularyHash }, standIn()));
  const dir = join(dirname(workDir()), "snapshots");
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  writeLocalCopy(dir, w.snapshotHash, new TextEncoder().encode(canonicalJson(w.manifest as never)), w.shards);
  return { dir, hash: w.snapshotHash };
}

const refusal = async (run: () => unknown): Promise<string> => {
  try {
    await run();
  } catch (error) {
    if (error instanceof PrivatePoolRefusal) return error.message;
    throw error;
  }
  return "no error";
};

describe("loadPrivatePool", () => {
  it("returns the active teams of the pinned snapshot, normalised, sorted by canonical hash, with their hold-out flags", async () => {
    expect(worldIsPrivate(world)).toBe(true);
    const records = [...world.teams.map((team, k) => recordOfPaste(team.paste, k + 1, k === 3 ? "retired" : "active"))];
    const copy = await localCopy(records);
    const pool = await loadPrivatePool({ engine, format_id: FORMAT_ID, snapshotsDir: copy.dir, snapshot_hash: copy.hash, holdout: "include" });
    expect(pool.snapshot_hash).toBe(copy.hash);
    expect(pool.teams).toHaveLength(3); // the retired record is not in the pool
    // The team's nickname marker names the record it came from.
    const source = pool.teams.map((sets) => records.find((r) => r.paste.includes((sets[0] as { name: string }).name)) as PrivateTeamRecord);
    const canonical = source.map((r) => r.canonical_team_hash);
    expect([...canonical].sort()).toEqual(canonical);
    expect(pool.hashes).toEqual(pool.teams.map((sets) => teamHash(sets)));
    // The flag follows the record it came from.
    pool.holdout.forEach((flag, k) => expect(flag).toBe((source[k] as PrivateTeamRecord).holdout));
  });

  it("with holdout exclude leaves the hold-out teams out, with include keeps them", async () => {
    const records = world.teams.map((team, k) => recordOfPaste(team.paste, k + 1));
    const held = records.filter((r) => r.holdout).length;
    expect(held).toBeGreaterThan(0);
    expect(held).toBeLessThan(records.length);
    const copy = await localCopy(records);
    const base = { engine, format_id: FORMAT_ID, snapshotsDir: copy.dir, snapshot_hash: copy.hash };
    const excluded = await loadPrivatePool({ ...base, holdout: "exclude" });
    expect(excluded.teams).toHaveLength(records.length - held);
    expect(excluded.holdout.every((flag) => !flag)).toBe(true);
    expect((await loadPrivatePool({ ...base, holdout: "include" })).teams).toHaveLength(records.length);
  });

  it("leaves out a team holding an excluded ability, and refuses when nothing is left", async () => {
    const copy = await localCopy(world.teams.slice(0, 2).map((team, k) => recordOfPaste(team.paste, k + 1)));
    const ability = (world.teams[0]?.sets[0] as { ability: string }).ability;
    const some = await loadPrivatePool({ engine, format_id: FORMAT_ID, snapshotsDir: copy.dir, snapshot_hash: copy.hash, holdout: "include", excluded_abilities: [ability] });
    expect(some.teams.length).toBeLessThan(2);
    const every = [...new Set(world.teams.slice(0, 2).flatMap((team) => team.sets.map((set) => set.ability)))];
    expect(await refusal(() => loadPrivatePool({ engine, format_id: FORMAT_ID, snapshotsDir: copy.dir, snapshot_hash: copy.hash, holdout: "include", excluded_abilities: every }))).toBe("private-pool: source_empty");
  });

  it("refuses with a fixed code: a copy that is not there, a bad hash, a directory outside private-data and a forged manifest", async () => {
    const copy = await localCopy(world.teams.slice(0, 2).map((team, k) => recordOfPaste(team.paste, k + 1)));
    const ask = (over: Partial<Parameters<typeof loadPrivatePool>[0]>) => loadPrivatePool({ engine, format_id: FORMAT_ID, snapshotsDir: copy.dir, snapshot_hash: copy.hash, holdout: "include", ...over });
    expect(await refusal(() => ask({ snapshot_hash: "cd".repeat(32) }))).toBe("private-pool: source_unreadable");
    for (const bad of ["../escape", "AB".repeat(32), ""]) expect(await refusal(() => ask({ snapshot_hash: bad }))).toBe("private-pool: source_unreadable");
    expect(await refusal(() => ask({ snapshotsDir: world.outside }))).toBe("private-pool: source_unreadable");
    const forged = join(copy.dir, "ef".repeat(32));
    mkdirSync(forged, { mode: 0o700 });
    writeFileSync(join(forged, "manifest.json"), readFileSync(join(copy.dir, copy.hash, "manifest.json")));
    expect(await refusal(() => ask({ snapshot_hash: "ef".repeat(32) }))).toBe("private-pool: snapshot_unverified");
  });
});

describe("an illegal team", () => {
  it("is refused with its position, naming nothing of the team", async () => {
    const long = makeCanaryWorld(engine, 2); // nicknames over 18 characters: the validator refuses them
    try {
      const copy = await localCopy(long.teams.map((team, k) => recordOfPaste(team.paste, k + 1)));
      const observation = await observe(long, () => loadPrivatePool({ engine, format_id: FORMAT_ID, snapshotsDir: copy.dir, snapshot_hash: copy.hash, holdout: "include" }));
      expect(observation.thrown).toContain("private-pool: team_invalid 1");
      expectNoLeak(long, observation);
    } finally {
      long.cleanup();
    }
  });
});

describe("withPrivatePool", () => {
  const synthetic = interimPoolSource(engine, { format_id: FORMAT_ID, generation: { excluded_abilities: [] } });

  it("adds the private teams as a second component and never lets one into known_synthetic", async () => {
    const copy = await localCopy(world.teams.slice(0, 3).map((team, k) => recordOfPaste(team.paste, k + 1)));
    const priv = await loadPrivatePool({ engine, format_id: FORMAT_ID, snapshotsDir: copy.dir, snapshot_hash: copy.hash, holdout: "include" });
    const source = withPrivatePool(() => synthetic, priv, 300);
    const pool = source(engine, { format_id: FORMAT_ID, generation: { excluded_abilities: [] } });
    expect(pool.teams).toHaveLength(synthetic.teams.length + 3);
    expect(pool.teams.slice(0, synthetic.teams.length)).toEqual(synthetic.teams);
    expect(pool.private).toEqual({ count: 3, share_permille: 300, holdout: priv.holdout });
    expect(pool.known_synthetic).toBe(synthetic.known_synthetic);
    for (const sets of priv.teams) {
      expect(pool.known_synthetic.has(teamHash(sets))).toBe(false);
      // stampTeam makes every private team private, whatever the synthetic set holds.
      expect(stampTeam(sets as never, pool.known_synthetic).data_class).toBe("private");
      expect(classOfSets(sets, pool.known_synthetic)).toBe("private");
    }
    // The synthetic teams stay synthetic.
    expect(stampTeam(synthetic.teams[0] as never, pool.known_synthetic).data_class).toBe("synthetic");
    expect(poolShape(pool)).toEqual({ size: pool.teams.length, private: { count: 3, share_permille: 300 } });
  });

  it("plans from both pools: a public pool plans as before, a mixed pool is a pure function of config, generation and both pools", async () => {
    const copy = await localCopy(world.teams.slice(0, 3).map((team, k) => recordOfPaste(team.paste, k + 1)));
    const priv = await loadPrivatePool({ engine, format_id: FORMAT_ID, snapshotsDir: copy.dir, snapshot_hash: copy.hash, holdout: "include" });
    const pool = withPrivatePool(() => synthetic, priv, 500)(engine, { format_id: FORMAT_ID, generation: { excluded_abilities: [] } });
    expect(planGeneration(config, 1, poolShape(synthetic))).toEqual(planGeneration(config, 1, synthetic.teams.length));
    const plan = planGeneration(config, 1, poolShape(pool));
    expect(planGeneration(config, 1, poolShape(pool))).toEqual(plan);
    const privateSeats = plan.flatMap((entry) => [entry.teams.p1, entry.teams.p2]).filter((i) => i >= synthetic.teams.length);
    expect(privateSeats.length).toBeGreaterThan(0);
    for (const i of privateSeats) expect(i).toBeLessThan(pool.teams.length);
  });

  it("refuses: a bad share, a pool that is already mixed and a private team that is also synthetic", async () => {
    const copy = await localCopy(world.teams.slice(0, 2).map((team, k) => recordOfPaste(team.paste, k + 1)));
    const priv = await loadPrivatePool({ engine, format_id: FORMAT_ID, snapshotsDir: copy.dir, snapshot_hash: copy.hash, holdout: "include" });
    for (const share of [0, 1001, 0.5, Number.NaN]) expect(await refusal(() => withPrivatePool(() => synthetic, priv, share))).toBe("private-pool: share_invalid");
    const request = { format_id: FORMAT_ID, generation: { excluded_abilities: [] } };
    const mixed = withPrivatePool(() => synthetic, priv, 100);
    expect(await refusal(() => withPrivatePool(mixed, priv, 100)(engine, request))).toBe("private-pool: pool_has_private_component");
    const overlap = { ...priv, hashes: [synthetic.hashes[0] as string, ...priv.hashes.slice(1)] };
    expect(await refusal(() => withPrivatePool(() => synthetic, overlap, 100)(engine, request))).toBe("private-pool: synthetic_overlap 1");
  });

  it("is a PoolSource that the interim source composes with", async () => {
    const copy = await localCopy(world.teams.slice(0, 2).map((team, k) => recordOfPaste(team.paste, k + 1)));
    const priv = await loadPrivatePool({ engine, format_id: FORMAT_ID, snapshotsDir: copy.dir, snapshot_hash: copy.hash, holdout: "include" });
    const pool = withPrivatePool(interimPoolSource, priv, 200)(engine, { format_id: FORMAT_ID, generation: { excluded_abilities: [] } });
    expect(pool.private?.count).toBe(2);
  });
});

describe("canary over pool construction and plan writing", () => {
  it("leaves no team content in the output, the plan file or any thrown refusal", async () => {
    const records = world.teams.map((team, k) => recordOfPaste(team.paste, k + 1));
    const copy = await localCopy(records);
    const observation = await observe(
      world,
      async () => {
        const priv = await loadPrivatePool({ engine, format_id: FORMAT_ID, snapshotsDir: copy.dir, snapshot_hash: copy.hash, holdout: "include" });
        const pool = withPrivatePool(interimPoolSource, priv, 400)(engine, { format_id: FORMAT_ID, generation: { excluded_abilities: [] } });
        const plan = planGeneration(config, 1, poolShape(pool));
        const text = canonicalJson(plan as never);
        writeFileSync(join(world.root, "plan.json"), text, { mode: 0o600 });
        // Refusals from a bad copy are scanned as well.
        let refusalText = "";
        try {
          await loadPrivatePool({ engine, format_id: FORMAT_ID, snapshotsDir: copy.dir, snapshot_hash: "ab".repeat(32), holdout: "exclude" });
        } catch (error) {
          refusalText = String(error);
        }
        return { surfaces: { plan: text, refusal: refusalText } };
      },
      [],
    );
    expect(observation.thrown).toBeNull();
    expect(observation.surfaces.plan).toContain('"teams"');
    expectNoLeak(world, observation);
  });

  it("flags a plan that carries a private team hash (negative control)", async () => {
    const hash = teamHash(world.teams[0]?.sets as never);
    const observation = await observe(world, async () => ({ surfaces: { plan: JSON.stringify({ hashes: [hash] }) } }));
    expect(findLeaks(world, observation).some((leak) => leak.kind === "team_hash")).toBe(true);
  });
});
