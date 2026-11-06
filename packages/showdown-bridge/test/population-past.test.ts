import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { checkLeagueRunRecord, loadEngine, readLeagueRunRecordV2 } from "../src/index.ts";
import { trainingTeamPool, type TrainingPool } from "../src/league-generation.ts";
import { type PastCell, type PastPoolFile, type PastSnapshot, PAST_POOL_FORMAT } from "../src/past-pool.ts";
import { type PolicyClient } from "../src/index.ts";
import { runPopulationCli } from "../src/population-cli.ts";
import { memberDir, playPopulationGeneration, type PopulationMember, type SnapshotSession, writePopulationGeneration } from "../src/population-generation.ts";
import { planPopulation } from "../src/population-plan.ts";
import { runConfigHash } from "../src/training-run-config.ts";
import { populationConfig } from "./population-support.ts";

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const scratch = mkdtempSync(join(tmpdir(), "population-past-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

function actorClient(): PolicyClient & { beginGame: () => Promise<void>; asked: number } {
  const client = {
    asked: 0,
    beginGame: () => Promise.resolve(),
    ask(_battle, query) {
      client.asked++;
      const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
      const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
      return Promise.resolve({ schema_version: 1 as const, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" as const });
    },
  } satisfies PolicyClient & { beginGame: () => Promise<void>; asked: number };
  return client;
}

// M = 2, 4 seats a member split 2 ladder / 1 self-play / 1 past, one baseline pair; n_parallel is the sample's.
const config = populationConfig(2, 4, [500_000, 250_000, 250_000]);
const POLICIES = [`policy:${"a".repeat(64)}`, `policy:${"b".repeat(64)}`];
const snapshot = (member: string, digit: string): PastSnapshot => ({ id: `${member}@g0`, member, generation: 0, policy_id: `policy:${digit.repeat(64)}`, preview: digit.repeat(64), battle: digit.repeat(64) });
const SNAPSHOTS = [snapshot("m0", "c"), snapshot("m1", "d")];
const CELLS: PastCell[] = [];

let pool: TrainingPool;
let pastFile: PastPoolFile;
beforeAll(async () => {
  pool = trainingTeamPool(engine, config.generation.excluded_abilities);
  pastFile = { schema_version: 1, format: PAST_POOL_FORMAT, run_id: config.seeds.run_key, run_config_hash: await runConfigHash(config), generation: 0, snapshots: SNAPSHOTS, cells: CELLS };
});

const membersOf = (): (PopulationMember & { client: ReturnType<typeof actorClient> })[] =>
  POLICIES.map((policy_id, k) => ({ member: `m${k}`, policy_id, checkpoints: { preview: String(k).repeat(64), battle: String(k + 2).repeat(64) }, client: actorClient() }));

/** A pool whose sessions are in-process clients, recording when each was opened and closed. */
const sessions = (): { events: string[]; clients: Map<string, ReturnType<typeof actorClient>>; open: (s: PastSnapshot) => Promise<SnapshotSession> } => {
  const events: string[] = [];
  const clients = new Map<string, ReturnType<typeof actorClient>>();
  return {
    events,
    clients,
    open: (entry) => {
      events.push(`open ${entry.id}`);
      const client = actorClient();
      clients.set(entry.id, client);
      return Promise.resolve({
        client,
        close: () => {
          events.push(`close ${entry.id}`);
          return Promise.resolve();
        },
      });
    },
  };
};

const treeBytes = (dir: string, prefix = ""): Record<string, string> => {
  const out: Record<string, string> = {};
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) Object.assign(out, treeBytes(path, `${prefix}${entry.name}/`));
    else out[`${prefix}${entry.name}`] = readFileSync(path, "utf8");
  }
  return out;
};

describe("a tiny population generation with a past pool (M = 2, the stand-in workers)", () => {
  const dirs = [join(scratch, "a"), join(scratch, "b")];
  let members: ReturnType<typeof membersOf>;
  let result: Awaited<ReturnType<typeof playPopulationGeneration>>;
  let first: ReturnType<typeof sessions>;
  beforeAll(async () => {
    members = membersOf();
    first = sessions();
    result = await playPopulationGeneration({ engine, config, generation: 1, members, pool, out_dir: dirs[0] as string, past: { pool: pastFile, open: first.open } });
    writePopulationGeneration(dirs[0] as string, result);
    const again = sessions();
    const rerun = await playPopulationGeneration({ engine, config, generation: 1, members: membersOf(), pool, out_dir: dirs[1] as string, past: { pool: pastFile, open: again.open } });
    writePopulationGeneration(dirs[1] as string, rerun);
  }, 180_000);

  it("plays the plan with the drawn snapshots, each past game on the snapshot's own session", () => {
    const plan = planPopulation(config, 1, pool.teams.length, { pool: pastFile, current: { m0: POLICIES[0] as string, m1: POLICIES[1] as string } });
    expect(result.plan).toEqual(plan);
    const past = result.games.filter((g) => g.kind === "past");
    expect(past.length).toBe(4); // one pair, two games, for each member
    expect(past.every((g) => g.outcome === "played" && /^m[01]@g0$/.test(g.opponent))).toBe(true);
    const used = new Set(past.map((g) => g.opponent));
    expect([...first.clients.keys()].sort()).toEqual([...used].sort());
    for (const [id, client] of first.clients) expect(client.asked, id).toBeGreaterThan(0);
  });

  it("opens a snapshot's session when a wave first needs it and closes every one before the generation ends", () => {
    const opened = first.events.filter((e) => e.startsWith("open "));
    const closed = first.events.filter((e) => e.startsWith("close "));
    expect(opened.length).toBe(closed.length);
    for (const e of opened) expect(first.events.indexOf(e.replace("open", "close"))).toBeGreaterThan(first.events.indexOf(e));
  });

  it("writes a record that passes its reader: the snapshot is an opponent of its policy id, in the member's entry only", async () => {
    const record = readLeagueRunRecordV2(JSON.parse(readFileSync(join(dirs[0] as string, "league-run.json"), "utf8")));
    await checkLeagueRunRecord(record, config);
    expect(record.games).toBe(result.games.length);
    for (const entry of record.members) {
      const met = entry.opponents.filter((o) => o.member === undefined && SNAPSHOTS.some((s) => s.policy_id === o.opponent_policy_id));
      expect(met.reduce((sum, o) => sum + o.games, 0)).toBe(2);
      expect(entry.episodes).toBe(entry.played + entry.opponents.filter((o) => o.member === entry.member).reduce((sum, o) => sum + o.played, 0));
    }
  });

  it("keeps a snapshot seat's Episodes on disk under the member it was scheduled for, and out of its stream: only the pair's own policy trains", () => {
    let kept = 0;
    for (const entry of result.members) {
      const dir = memberDir(dirs[0] as string, entry.member);
      const onDisk = readdirSync(join(dir, "episodes")).map((f) => JSON.parse(readFileSync(join(dir, "episodes", f), "utf8")));
      const snapshotSeats = onDisk.filter((e) => SNAPSHOTS.some((s) => s.policy_id === e.provenance.behaviour_policy_id));
      kept += snapshotSeats.length;
      expect(entry.set.behaviour_policy_id).toBe(POLICIES[Number(entry.member.slice(1))]);
      expect(entry.episode_hashes.length).toBe(onDisk.length);
      expect(entry.set.episode_hashes.length).toBe(onDisk.length - snapshotSeats.length - onDisk.filter((e) => e.provenance.behaviour_policy_id === "random-legal@tiny").length);
    }
    expect(kept).toBe(4); // two past pairs of two games, one snapshot seat each
  });

  it("is byte-identical on a rerun, every file of the tree", () => {
    const [a, b] = dirs.map((d) => treeBytes(d));
    expect(a).toEqual(b);
  });
});

describe("the CLI port with a past pool", () => {
  const input = (change: object = {}) => ({
    engine, config, generation: 1, pool, out: join(scratch, "cli"), run_dir: scratch, worker_python: "unused", engine_data: "unused", engine_data_hash: "e".repeat(64), encoder_version: "enc1:x",
    members: POLICIES.map((policy_id, k) => ({ member: `m${k}`, preview: String(k).repeat(64), battle: String(k + 2).repeat(64), policy_id })),
    make_client: () => Object.assign(actorClient(), { start: () => Promise.resolve(), close: () => Promise.resolve() }),
    ...change,
  });

  it("starts a session for each snapshot the plan meets, and closes it", async () => {
    const started: string[] = [];
    const closed: string[] = [];
    const lines: string[] = [];
    const make = (entry: PastSnapshot) => Object.assign(actorClient(), { start: () => (started.push(entry.id), Promise.resolve()), close: () => (closed.push(entry.id), Promise.resolve()) });
    expect(await runPopulationCli(input({ past: pastFile, make_snapshot_client: make }), (line) => lines.push(line))).toBe(0);
    expect(started.length).toBeGreaterThan(0);
    expect([...started].sort()).toEqual([...closed].sort());
    expect(JSON.parse(lines[0] as string).generation).toBe(1);
  }, 120_000);

  it("refuses a past-pool share without its pool before it creates any client", async () => {
    let made = 0;
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const make = () => (made++, Object.assign(actorClient(), { start: () => Promise.resolve(), close: () => Promise.resolve() }));
    expect(await runPopulationCli(input({ make_client: make }), () => undefined)).toBe(1);
    expect(error.mock.calls.flat().join(" ")).toMatch(/needs the committed past pool/);
    error.mockRestore();
    expect(made).toBe(0);
  });
});
