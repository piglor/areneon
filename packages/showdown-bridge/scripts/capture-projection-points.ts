// Re-capture the battle-state projection points (fixtures/showdown/<format>/projection/points.json) that the Python packer and
// projection tests read (tests/ml/projection, ADR-030 G4, #1464):
//   node packages/showdown-bridge/scripts/capture-projection-points.ts
// The first steps of the golden battle and of the Baton Pass and Illusion battles, played again with the G2 recorder: at each
// step the engine state trimmed to what record v2 of the battle-state spec reads, each seat's channel length, and each decision's
// seat, kind and candidates (the Python side rebuilds the observation from the channel and the team, which is all the encoder
// reads of it). Synthetic teams (CC0); the state holds both sides' hidden information: an evaluation artifact. Re-run after a pin
// bump, a recorder change or a new spec record, and review the diff. The full corpus runs in battle-state-projection.python.test.ts.
// SHOWDOWN_DIR points at another built checkout (a worktree without its own).
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { committedBattles } from "../src/battle-trace-corpus.ts";
import { battleStateSpecHash, loadBattleStateSpec } from "../src/battle-state-spec.ts";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { projectionBattle, SPEC_V2_PATH } from "../src/projection-points.ts";

/** Battles and how many steps of each are kept. */
const KEEP: Record<string, number> = { golden: 12, "baton-pass": 8, illusion: 10 };

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const record = loadBattleStateSpec(engine, GOLDEN_FORMAT, new URL(`file://${SPEC_V2_PATH}`));
const battles = [];
for (const { spec } of committedBattles(["golden", "singles"])) {
  const keep = KEEP[spec.battle_id.split("/")[0] as string];
  if (keep === undefined) continue;
  const battle = await projectionBattle(engine, spec, record);
  const points = battle.points.slice(0, keep).filter((p) => p.decisions.length > 0);
  const last = points.at(-1);
  if (last === undefined) throw new Error(`${spec.battle_id}: no decision in the first ${keep} steps`);
  battles.push({
    battle_id: battle.battle_id,
    teams: battle.teams,
    channels: { p1: battle.channels.p1.slice(0, last.cut.p1), p2: battle.channels.p2.slice(0, last.cut.p2) },
    points: points.map((p) => ({
      step: p.step,
      cut: p.cut,
      state: p.state,
      decisions: p.decisions.map((d) => ({ seat: d.seat, kind: d.query.observation.kind, candidates: d.query.candidates })),
    })),
  });
}
const out = join(GOLDEN_DIR, "projection");
mkdirSync(out, { recursive: true });
const document = { data_class: "synthetic", showdown_commit: engine.commit, spec_hash: battleStateSpecHash(record), battles };
writeFileSync(join(out, "points.json"), `${JSON.stringify(document)}\n`);
console.log(`${battles.length} battles, ${battles.reduce((n, b) => n + b.points.reduce((m, p) => m + p.decisions.length, 0), 0)} decisions`);
