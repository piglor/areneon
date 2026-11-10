import { spawnSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { type HistoryFactsData, historyFactsAtRequests, ownHistoryChannel } from "../src/index.ts";
import { FORMAT_DIR, ROOT, type Fixture, fixtures, readJson } from "./support.ts";

// ADR-022 erratum E-22.5.2, slice H1: the TS module and the Python reducer's `history` hold the same facts at every
// request of every committed own channel (the Python reducer is the one the encoder will read; the TS module is the
// one the probe will declare from). Both read the committed data files the exporter wrote, so the test needs no
// engine. A Pokemon the TS module marks `unknown` (an Illusion's `|replace|`) is compared as the TS says it.

const python = process.env.ARENEON_PYTHON ?? join(ROOT, ".venv", "bin", "python");
const here = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = join(ROOT, "ml", "areneon_ml", "reducer", "data");

export function committedHistoryData(): HistoryFactsData {
  const engine = readJson(join(FORMAT_DIR, "engine-data.json"));
  const file = readJson(join(DATA_DIR, `history-facts.${engine.format_id}.json`));
  const stall = readJson(join(DATA_DIR, `stall-moves.${engine.format_id}.json`)).stall_moves as string[];
  const copy = readJson(join(DATA_DIR, `volatile-copy.${engine.format_id}.json`));
  return {
    stall_moves: stall,
    ally_side_moves: engine.moves.filter((m: { id: string; target?: string }) => m.target === "allySide" && stall.includes(m.id)).map((m: { id: string }) => m.id),
    stall_counter: file.stall_counter,
    action_sources: file.action_sources,
    external_action_sources: file.external_action_sources,
    item_use_sources: file.item_use_sources,
    eat_tags: file.eat_tags,
    eaten_berries: file.eaten_berries,
    last_item_restorers: file.last_item_restorers,
    last_item_takers: file.last_item_takers,
    direct_end_items: file.direct_end_items,
    null_cant_reasons: file.null_cant_reasons,
    protect_breaker_moves: file.protect_breaker_moves,
    slot_conditions: file.slot_conditions,
    volatile_not_copied: copy.not_copied,
    shed_tail_copies: copy.shed_tail_copies,
  };
}

/** The committed channels this test reads: every channel the shared reader's parity reads, and the targeted corpora. */
function channels(): Fixture[] {
  const out = fixtures();
  const battles = (dir: string, files: string[]) => {
    for (const f of files) {
      const doc = readJson(join(FORMAT_DIR, dir, f));
      for (const p of ["p1", "p2"]) out.push({ id: `${dir}/${f.replace(".json", "")}/${p}`, player: p, lines: doc.channels[p].lines, team: doc.teams[p] });
    }
  };
  battles("transform", ["battle.json"]);
  battles("revival", ["battle.json"]);
  battles("side-guard", ["battle.json"]);
  battles("search-fidelity", readdirSync(join(FORMAT_DIR, "search-fidelity")).filter((f) => /^b\d+\.json$/.test(f)).sort());
  return out;
}

describe.skipIf(!existsSync(python))("the TS history facts against the Python reducer's, at every request of the committed channels", () => {
  const all = channels();
  const data = committedHistoryData();
  const reference = (() => {
    const jobs = all.map((f) => ({ id: f.id, player: f.player, lines: f.lines, team: f.team }));
    const run = spawnSync(python, [join(here, "history_points.py")], {
      input: JSON.stringify({ engine_data: join(FORMAT_DIR, "engine-data.json"), jobs }),
      maxBuffer: 1 << 30,
      encoding: "utf8",
      env: { ...process.env, PYTHONPATH: [join(ROOT, "ml"), join(ROOT, "packages", "schema", "python"), process.env.PYTHONPATH ?? ""].join(":") },
    });
    if (run.status !== 0) throw new Error(run.stderr);
    return JSON.parse(run.stdout) as Record<string, { points: { line: number; facts: any }[]; refused: string | null }>;
  })();

  let compared = 0;
  let refusedByTs = 0;
  it("holds equal facts at every request", () => {
    for (const fixture of all) {
      const ref = reference[fixture.id] as { points: { line: number; facts: any }[]; refused: string | null };
      // the reducer reads every committed channel to its end, but for the one known Illusion twin
      const ts = historyFactsAtRequests({ channel: ownHistoryChannel(fixture.player, fixture.lines), data });
      const at = new Map(ref.points.map((p) => [p.line, p.facts]));
      for (const point of ts) {
        const python = at.get(point.line);
        if (python === undefined) continue; // the reducer stopped before this request
        if (point.facts === null) {
          refusedByTs += 1;
          expect(python.refused, `${fixture.id} line ${point.line}: both refuse`).toBeDefined();
          continue;
        }
        compared += 1;
        expect(python, `${fixture.id} line ${point.line}`).toEqual(JSON.parse(JSON.stringify(point.facts)));
      }
    }
    process.stderr.write(`history parity: ${all.length} channels, ${compared} compared, ${refusedByTs} refused by TS\n`);
    expect(compared).toBeGreaterThan(5000);
    expect(refusedByTs).toBeLessThan(compared / 100);
  }, 600_000);
});
