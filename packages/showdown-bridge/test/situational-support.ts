// A small committed-shape progress root built from real battles, for the situational move-use tests.
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { PokemonSet } from "../src/engine.ts";
import { firstLegal, GOLDEN_FORMAT } from "../src/golden.ts";
import { progressGameStreams } from "../src/progress-seed.ts";
import { omniscientLogHash, type ProgressGameRecord } from "../src/progress-records.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import { type Actor, type BattleRecord, type Player, runBattle } from "../src/runner.ts";
import { teamHash } from "../src/team-class.ts";
import { corpus, engine } from "./damage-aware-support.ts";

const sample = (name: string): Record<string, unknown> => JSON.parse(readFileSync(new URL(`../../../fixtures/contracts/v1/${name}.json`, import.meta.url), "utf8"));

export const MAX_TURNS = 4;
/** The battle seed `progress-seed/1` gives pair 0 of the sample run, which the game record reader requires. */
export const PAIR_SEED = progressGameStreams("tiny-run", 0).seed;

/** A seeded battle: p1 plays random-legal, p2 `p2Actor` (first-legal by default), over corpus battle `index`'s seed and teams. */
export async function play(index: number, change: (team: PokemonSet[]) => PokemonSet[] = (team) => team, p2Actor: Actor = firstLegal): Promise<{ record: BattleRecord; teams: Record<Player, PokemonSet[]> }> {
  const battle = corpus[index] as unknown as { teams: Record<Player, PokemonSet[]>; actor_seeds: Record<Player, number> };
  const teams = { p1: battle.teams.p1, p2: change(structuredClone(battle.teams.p2)) };
  const record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: PAIR_SEED, teams, actors: { p1: randomLegalActor(battle.actor_seeds.p1), p2: p2Actor }, maxTurns: MAX_TURNS });
  return { record, teams };
}

/** A valid stored game of `record`, as the progress evaluation would store it (a shape template from the committed samples). */
export function storedGame(record: BattleRecord, teams: Record<Player, PokemonSet[]>, game: number): ProgressGameRecord {
  const template = sample("progress-game-record/valid/played-pair");
  const lines = [...record.omniscient_log];
  return {
    ...template,
    game,
    pair: game >> 1,
    seed: PAIR_SEED,
    teams: { p1: { team: teams.p1, team_hash: teamHash(teams.p1) }, p2: { team: teams.p2, team_hash: teamHash(teams.p2) } },
    decisions: record.choices.map((c) => {
      const legal = record.requests[c.player][c.request_seq - 1]?.legal;
      return { player: c.player, request_seq: c.request_seq, candidates: [...(legal?.choices ?? []).map((choice) => ({ choice, pending: false })), ...(legal?.pending ?? []).map((choice) => ({ choice, pending: true }))], chosen: c.choice };
    }),
    outcome: { kind: "played", winner: record.winner ?? "tie", turns: record.turns },
    omniscient_log: lines,
    omniscient_log_sha256: omniscientLogHash(lines),
  } as unknown as ProgressGameRecord;
}

const sha = (bytes: Buffer): string => createHash("sha256").update(bytes).digest("hex");

/** Write `<root>/g000001` holding `games` and the committed baseline-match record, with its manifest; returns the root. */
export function writeRoot(games: ProgressGameRecord[], evaluation: string = "progress-evaluation-record-2/valid/baseline-match", edit: (record: Record<string, unknown>) => void = () => {}): string {
  const root = mkdtempSync(join(tmpdir(), "situational-"));
  const dir = join(root, "g000001");
  mkdirSync(join(dir, "games"), { recursive: true });
  const files: Record<string, string> = {};
  const put = (name: string, value: unknown): void => {
    const bytes = Buffer.from(JSON.stringify(value));
    writeFileSync(join(dir, name), bytes);
    files[name] = sha(bytes);
  };
  const record = sample(evaluation) as { plan: { max_turns: number } };
  record.plan.max_turns = MAX_TURNS; // the battles of these tests end at their turn cap
  edit(record);
  put("progress.json", record);
  for (const g of games) put(`games/${g.match_id}-${g.game}.json`, g);
  writeFileSync(join(dir, "manifest.json"), JSON.stringify({ generation: 1, run_id: "tiny-run", schema_version: 1, files }));
  return root;
}
