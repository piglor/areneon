/* eslint-disable @typescript-eslint/no-explicit-any */
// The `projection` comparison kind end to end (ADR-030 G4, #1464): the reference views come from the Python projection (`view`
// frames of its server), an echo backend answers with the views it was given, and `checkProjection` compares them; a doctored
// view is flagged at the right seat and field. The battle-level harness side is in packages/differential/test.
import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { checkProjection, type SeatView } from "@areneon/differential/battle";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const PYTHON = join(ROOT, ".venv/bin/python");
const FORMAT_DIR = join(ROOT, "fixtures/showdown/gen9championsvgc2026regmc");
const SPEC = join(ROOT, "config/engine-state/gen9championsvgc2026regmc/battle-state-spec.v2.json");
const skip = !existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1";

describe.skipIf(skip)("checkProjection against the Python projection", () => {
  let child: ChildProcessWithoutNullStreams;
  const waiting: ((line: string) => void)[] = [];
  const ask = (frame: unknown): Promise<any> =>
    new Promise((resolve) => {
      waiting.push((line) => resolve(JSON.parse(line)));
      child.stdin.write(`${JSON.stringify(frame)}\n`);
    });
  let reference: Record<"p1" | "p2", SeatView>;

  beforeAll(async () => {
    child = spawn(PYTHON, ["-m", "areneon_ml.projection", "serve", "--engine-data", join(FORMAT_DIR, "engine-data-2.json"), "--spec", SPEC], { cwd: ROOT, stdio: ["pipe", "pipe", "pipe"] });
    createInterface({ input: child.stdout }).on("line", (line) => waiting.shift()?.(line));
    const points = JSON.parse(readFileSync(join(FORMAT_DIR, "projection/points.json"), "utf8"));
    const battle = points.battles[0];
    const point = battle.points.find((p: any) => p.decisions.length > 0);
    const frame = { format_id: "gen9championsvgc2026regmc", state: point.state, channels: { p1: battle.channels.p1.slice(0, point.cut.p1), p2: battle.channels.p2.slice(0, point.cut.p2) }, teams: battle.teams };
    const views = await Promise.all(["p1", "p2"].map(async (seat) => (await ask({ kind: "view", id: `${battle.battle_id}/${point.step}/${seat}`, point: frame, seat })).view));
    reference = { p1: views[0], p2: views[1] };
  });
  afterAll(() => child?.stdin.end());

  it("finds no difference between an echo backend's views and the reference views", () => {
    const echo = JSON.parse(JSON.stringify(reference));
    expect(Object.keys(reference.p1.fields).length).toBeGreaterThan(50);
    expect(checkProjection(echo, reference, 1)).toBeNull();
  });

  it("flags a doctored view at the seat and field that differ, and one that leaks a field the seat may not read", () => {
    const doctored = JSON.parse(JSON.stringify(reference));
    const key = Object.keys(doctored.p2.fields).find((k) => k.startsWith("seat_roster.hp_shown@own")) as string;
    doctored.p2.fields[key][0] += 1;
    expect(checkProjection(doctored, reference, 4)).toEqual({ kind: "projection", step: 4, seat: "p2", where: `projected field ${key} differs` });
    const leak = JSON.parse(JSON.stringify(reference));
    leak.p1.fields["roster.hp@opponent"] = [1];
    expect(checkProjection(leak, reference, 4)).toEqual({ kind: "projection", step: 4, seat: "p1", where: "projected field roster.hp@opponent is present on one side only" });
  });
});
