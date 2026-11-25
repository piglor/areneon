// Shared by the battle-state projection tests (#1464): the Python child (`python -m areneon_ml.projection serve`), the frames it
// reads, and the corpus gate, which runs as several shards (one test file each) so a CI machine's workers share the replay.
import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { committedBattles } from "../../src/battle-trace-corpus.ts";
import { type BattleStateSpec, loadBattleStateSpec } from "../../src/battle-state-spec.ts";
import { type Engine, loadEngine } from "../../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../../src/golden.ts";
import { type ProjectionBattle, projectionBattle, type ProjectionPoint, SPEC_V2_PATH } from "../../src/projection-points.ts";

export const ROOT = fileURLToPath(new URL("../../../../", import.meta.url));
export const PYTHON = join(ROOT, ".venv/bin/python");
// Replaying every committed corpus and encoding each decision three ways takes several minutes on one thread.
export const CORPUS_TIMEOUT_MS = 3_600_000;

/** What the child answers (the fields this test reads). */
export interface Reply {
  id: string;
  status: "equal" | "different" | "refused" | "ok" | "error";
  code?: string;
  differences?: unknown;
  canary?: { engine: boolean; tensors: boolean };
  view_sha256?: string;
}

/** One line out, one line back: the projection child (`python -m areneon_ml.projection serve`). */
export class Child {
  readonly process: ChildProcessWithoutNullStreams;
  readonly #waiting: { resolve: (line: string) => void; reject: (error: Error) => void }[] = [];
  /** `args` replaces the server's command line (a test seam: a child that dies). */
  constructor(args: string[] = ["-m", "areneon_ml.projection", "serve", "--engine-data", join(GOLDEN_DIR, "engine-data-2.json"), "--spec", SPEC_V2_PATH]) {
    this.process = spawn(PYTHON, args, { cwd: ROOT, stdio: ["pipe", "pipe", "pipe"] });
    this.process.stderr.on("data", (chunk) => process.stderr.write(chunk));
    createInterface({ input: this.process.stdout }).on("line", (line) => this.#waiting.shift()?.resolve(line));
    // A dead child answers nothing more: fail every pending and later ask at once (not at the test's timeout).
    const failAll = (why: string) => {
      this.#dead = why;
      for (const waiting of this.#waiting.splice(0)) waiting.reject(new Error(why));
    };
    this.process.on("exit", (code, signal) => failAll(`the projection child exited (code ${code}, signal ${signal})`));
    this.process.on("error", (error) => failAll(`the projection child failed: ${error.message}`));
  }
  #dead: string | null = null;
  ask(frame: unknown): Promise<Reply> {
    return new Promise((resolve, reject) => {
      if (this.#dead !== null) return reject(new Error(this.#dead));
      this.#waiting.push({
        resolve: (line) => {
          const reply = JSON.parse(line) as Reply & { code?: string; detail?: string };
          if (reply.status === "error") reject(new Error(`the projection child could not answer ${reply.id}: ${reply.code}: ${reply.detail}`));
          else resolve(reply);
        },
        reject,
      });
      this.process.stdin.write(`${JSON.stringify(frame)}\n`, (error) => {
        if (error) reject(error);
      });
    });
  }
  close(): void {
    this.process.stdin.end();
  }
}

export const pointFrame = (battle: ProjectionBattle, point: ProjectionPoint): unknown => ({
  format_id: GOLDEN_FORMAT,
  state: point.state,
  channels: { p1: battle.channels.p1.slice(0, point.cut.p1), p2: battle.channels.p2.slice(0, point.cut.p2) },
  teams: battle.teams,
});

export const skipProjection = !existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1";

/**
 * What the committed corpora hold at this pin, per shard of four (battle k belongs to shard k mod 4): battles, requests that ask for a
 * decision, Team Preview requests. Exact, so a request that disappears (a recorder returning no mask, a battle that stops early) fails.
 */
const EXPECTED_SHARDS: readonly { battles: number; decisions: number; preview: number }[] = [
  { battles: 63, decisions: 2253, preview: 126 },
  { battles: 62, decisions: 1992, preview: 124 },
  { battles: 62, decisions: 2098, preview: 124 },
  { battles: 62, decisions: 2230, preview: 124 },
];

/**
 * The requests the projection refuses, by battle id and code, and why: every one is in a battle that holds an Illusion user (checked
 * from the battle's teams). `phantom`: an Illusion twin, a Pokemon shown in two positions (a phantom roster entry the spec's roster
 * dimension has no row for). `reveal_overflow`: a disguise that never breaks, whose user's moves the reducer attributes to the Pokemon
 * it is shown as, giving six revealed moves where the spec holds four slots and one extra (search-fidelity b63 and b86). Follow-up:
 * #1504 (route Illusion teams to Showdown, ADR-030 non-admitted entity rule). Any other refusal fails.
 */
const EXPECTED_REFUSALS: Record<string, Record<string, number>> = {
  "search-fidelity/b160": { phantom: 6 },
  "search-fidelity/b43": { phantom: 6 },
  "search-fidelity/b50": { phantom: 4 },
  "search-fidelity/b63": { reveal_overflow: 14 },
  "search-fidelity/b86": { phantom: 8, reveal_overflow: 3 },
};

const holdsIllusionUser = (teams: Record<string, { ability?: string }[]>): boolean => Object.values(teams).some((team) => team.some((set) => (set.ability ?? "").toLowerCase() === "illusion"));

/** The corpus gate over the battles of one shard: battle k belongs to shard k mod `shards` (4: the pinned counts above). */
export function projectionCorpusShard(shard: number, shards: number): void {
  if (shards !== EXPECTED_SHARDS.length) throw new Error(`the corpus gate pins its counts for ${EXPECTED_SHARDS.length} shards`);
  describe.skipIf(skipProjection)(`the battle-state projection against the encoder (record v2), corpus shard ${shard + 1} of ${shards}`, () => {
    let engine: Engine;
    let record: BattleStateSpec;
    let child: Child;
    beforeAll(() => {
      engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
      record = loadBattleStateSpec(engine, GOLDEN_FORMAT, new URL(`file://${SPEC_V2_PATH}`));
      child = new Child();
    });
    afterAll(() => child?.close());

    it(
      "is bit-equal to Encoder.encode at every request of both seats of every committed corpus, and the canary holds at each",
      async () => {
        const counts = { battles: 0, decisions: 0, equal: 0, preview: 0, canary: 0, canaryEngine: 0 };
        const refused: Record<string, Record<string, number>> = {};
        const withoutIllusionUser: string[] = [];
        const ids = new Set<string>();
        const different: string[] = [];
        const leaks: string[] = [];
        let k = -1;
        for (const { spec } of committedBattles()) {
          k += 1;
          if (k % shards !== shard) continue;
          const battle = await projectionBattle(engine, spec, record);
          counts.battles += 1;
          ids.add(battle.battle_id);
          for (const point of battle.points) {
            for (const decision of point.decisions) {
              counts.decisions += 1;
              if (decision.query.observation.kind === "preview") counts.preview += 1;
              const reply = await child.ask({ kind: "check", id: `${battle.battle_id}/${point.step}/${decision.seat}`, point: pointFrame(battle, point), seat: decision.seat, query: decision.query });
              if (reply.status === "refused") {
                const code = reply.code ?? "none";
                const own = (refused[battle.battle_id] ??= {});
                own[code] = (own[code] ?? 0) + 1;
                if (!holdsIllusionUser(battle.teams as never) && !withoutIllusionUser.includes(battle.battle_id)) withoutIllusionUser.push(battle.battle_id);
                continue;
              }
              if (reply.status === "equal") counts.equal += 1;
              else if (different.length < 20) different.push(`${reply.id}: ${JSON.stringify(reply.differences)}`);
              counts.canary += 1;
              if (reply.canary?.engine === true && reply.canary.tensors) counts.canaryEngine += 1;
              else if (leaks.length < 20) leaks.push(`${reply.id}: ${JSON.stringify(reply.canary)}`);
            }
          }
        }
        console.log(JSON.stringify({ shard, ...counts, refused }));
        expect(different).toEqual([]);
        expect(leaks).toEqual([]);
        const want = EXPECTED_SHARDS[shard] as { battles: number; decisions: number; preview: number };
        expect({ battles: counts.battles, decisions: counts.decisions, preview: counts.preview }).toEqual(want);
        // The refusals are exactly the pinned ones of this shard's battles, each in a battle that holds an Illusion user.
        const shardBattles = Object.fromEntries(Object.entries(EXPECTED_REFUSALS).filter(([id]) => ids.has(id)));
        expect(refused).toEqual(shardBattles);
        expect(withoutIllusionUser).toEqual([]);
        const refusedCount = Object.values(refused).reduce((n, codes) => n + Object.values(codes).reduce((a, b) => a + b, 0), 0);
        expect(counts.equal + refusedCount).toBe(counts.decisions);
        expect(counts.canary).toBe(counts.equal);
        expect(counts.canaryEngine).toBe(counts.equal);
      },
      CORPUS_TIMEOUT_MS,
    );
  });
}
