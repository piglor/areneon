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
  status: "equal" | "different" | "refused" | "ok";
  code?: string;
  differences?: unknown;
  canary?: { engine: boolean; tensors: boolean };
  view_sha256?: string;
}

/** One line out, one line back: the projection child (`python -m areneon_ml.projection serve`). */
export class Child {
  readonly process: ChildProcessWithoutNullStreams;
  readonly #waiting: ((line: string) => void)[] = [];
  constructor() {
    const args = ["-m", "areneon_ml.projection", "serve", "--engine-data", join(GOLDEN_DIR, "engine-data-2.json"), "--spec", SPEC_V2_PATH];
    this.process = spawn(PYTHON, args, { cwd: ROOT, stdio: ["pipe", "pipe", "pipe"] });
    this.process.stderr.on("data", (chunk) => process.stderr.write(chunk));
    createInterface({ input: this.process.stdout }).on("line", (line) => this.#waiting.shift()?.(line));
  }
  ask(frame: unknown): Promise<Reply> {
    return new Promise((resolve) => {
      this.#waiting.push((line) => resolve(JSON.parse(line) as Reply));
      this.process.stdin.write(`${JSON.stringify(frame)}\n`);
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

/** The corpus gate over the battles of one shard: battle k belongs to shard k mod `shards`. */
export function projectionCorpusShard(shard: number, shards: number): void {
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
      const counts = { battles: 0, decisions: 0, equal: 0, preview: 0, battle: 0, refused: {} as Record<string, number>, canary: 0, canaryEngine: 0 };
      const different: string[] = [];
      const leaks: string[] = [];
      let k = -1;
      for (const { spec } of committedBattles()) {
        k += 1;
        if (k % shards !== shard) continue;
        const battle = await projectionBattle(engine, spec, record);
        counts.battles += 1;
        for (const point of battle.points) {
          for (const decision of point.decisions) {
            counts.decisions += 1;
            counts[decision.query.observation.kind === "preview" ? "preview" : "battle"] += 1;
            const reply = await child.ask({ kind: "check", id: `${battle.battle_id}/${point.step}/${decision.seat}`, point: pointFrame(battle, point), seat: decision.seat, query: decision.query });
            if (reply.status === "refused") {
              const code = reply.code ?? "none";
              counts.refused[code] = (counts.refused[code] ?? 0) + 1;
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
      console.log(JSON.stringify(counts));
      expect(different).toEqual([]);
      expect(leaks).toEqual([]);
      expect(counts.battles).toBeGreaterThan(Math.floor(240 / shards) - 1);
      expect(counts.decisions).toBeGreaterThan(Math.floor(7000 / shards) - 400);
      expect(counts.preview).toBeGreaterThan(Math.floor(400 / shards) - 40);
      // Only the reader's Illusion limits may be refused, each counted: a twin (a phantom roster entry the spec's roster dimension has
      // no row for) and a disguise that never breaks, whose user's moves the reducer attributes to the Pokemon it is shown as (a
      // list of six revealed moves, which the spec's four slots and one extra cannot hold: search-fidelity b63 and b86 at this pin).
      expect(Object.keys(counts.refused).filter((code) => code !== "phantom" && code !== "reveal_overflow")).toEqual([]);
      expect(counts.refused.reveal_overflow ?? 0).toBeLessThanOrEqual(17);
      expect(counts.equal + Object.values(counts.refused).reduce((a, b) => a + b, 0)).toBe(counts.decisions);
    },
    CORPUS_TIMEOUT_MS,
  );

  });
}
