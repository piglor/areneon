// SPIKE (#677): throwaway. Shared evidence writer for the bot-vs-bot and
// human-vs-bot runs. Everything written here is player-perspective.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine, resolveFormat } from "../../../packages/showdown-bridge/src/index.ts";
import { type BattleResult, type DecisionRecord, FORMAT_ID } from "./bot.ts";
import { DEFAULT_PORT, HOST } from "./server.ts";

export function writeEvidence(
  outDir: string,
  prefix: string,
  result: BattleResult,
  decisions: DecisionRecord[],
  playerView: string[],
  wallSeconds: number,
): object {
  mkdirSync(outDir, { recursive: true });
  const timed = decisions.filter((d) => d.timer_turn_left_s !== undefined);
  const ms = decisions.map((d) => d.decision_ms).sort((x, y) => x - y);
  const summary = {
    spike: "#677",
    manifest: resolveFormat(loadEngine(), FORMAT_ID).manifest,
    server: `pinned Showdown server, ${HOST}:${DEFAULT_PORT}, --no-security, forcetimer`,
    battle: result.battle,
    winner: result.winner,
    turns: result.turns,
    wall_seconds: Number(wallSeconds.toFixed(2)),
    timer_on: result.timerOn,
    ots_rejected_by: result.otsRejectedBy,
    bot_decisions: decisions.length,
    decisions_with_timer_state: timed.length,
    fallbacks: decisions.filter((d) => d.fallback).length,
    decision_ms: { min: ms[0], p50: ms[Math.floor(ms.length / 2)], max: ms[ms.length - 1] },
    vgc_timer: { preview_s: 90, bank_s: 420, max_per_turn_s: 55 },
    min_timer_turn_left_s: Math.min(...timed.map((d) => d.timer_turn_left_s as number)),
    min_timer_bank_left_s: Math.min(...timed.map((d) => d.timer_bank_left_s as number)),
  };
  writeFileSync(
    join(outDir, `${prefix}.decisions.jsonl`),
    decisions.map((d) => JSON.stringify(d)).join("\n") + "\n",
  );
  writeFileSync(join(outDir, `${prefix}.summary.json`), JSON.stringify(summary, null, 2) + "\n");
  if (playerView.length > 0) {
    writeFileSync(join(outDir, `${prefix}.player-view.log`), playerView.join("\n") + "\n");
  }
  return summary;
}
