// Run-weighted counts for the mechanics inventory (ADR-030 groundwork G5, #1465):
// how often the entities of the inventory are exercised in the Episodes of a
// training run, read from each Episode's OWN player channel (the lines a player
// sees). Counts only: the report is printed and journaled, never committed (run
// data stays local, the owner's rule), and the reader refuses any Episode that is
// not of the synthetic data class and any path under `private-data/`.
//
// An event is: a `|move|` line without a `[from]` tag (a move chosen, not one
// another effect called); an `ability:` or `item:` name on any line (a tag
// `[from] ability: X` or an `-activate|P|ability: X`); and the id a `-start`,
// `-sidestart`, `-fieldstart`, `-weather` or `-status` line names, which is a
// condition. A turn is the stretch of lines after a `|turn|` line. Each game
// appears once per seat (two Episodes), so shares are over Episodes and the turns
// of both views; the counts are per view, not per game.

import { readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { MechanicsInventory, Kind } from "./mechanics-inventory.ts";
import { toId } from "./mechanics-inventory.ts";
import { buildIndex, type Curve, curve, entityKey, type InventoryIndex } from "./mechanics-inventory-weights.ts";

export class RunReadError extends Error {
  override name = "RunReadError";
}

const NAMED = /^(?:\[from\]\s*)?(ability|item):\s*(.+)$/;
const cmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/** The inventory entities a protocol line exercises, as `[kind, id]`. */
export function lineEvents(line: string): [Kind, string][] {
  if (!line.startsWith("|")) return [];
  const parts = line.split("|").slice(1);
  const type = parts[0] as string;
  const events: [Kind, string][] = [];
  for (const part of parts.slice(1)) {
    const named = NAMED.exec(part.trim());
    if (named !== null) events.push([named[1] as Kind, toId(named[2] as string)]);
  }
  const condition = (text: string | undefined): void => void (text !== undefined && events.push(["condition", toId(text.replace(/^(?:move|ability|item):\s*/, ""))]));
  if (type === "move" && !parts.slice(3).some((part) => part.startsWith("[from]"))) events.push(["move", toId(parts[2] ?? "")]);
  else if (type === "-start" || type === "-sidestart" || type === "-status") condition(parts[2]);
  else if (type === "-fieldstart") condition(parts[1]);
  else if (type === "-weather" && parts[1] !== "none") condition(parts[1]);
  return events;
}

export interface RunEpisode {
  data_class: string;
  channel: string[];
}

export interface RunReport {
  episodes: number;
  /** `|turn|` lines over all Episodes. */
  turns: number;
  /** Events naming an id the inventory does not have. */
  unmatched_events: number;
  /** Damage-relevant entities: uses, turns with a use and Episodes with a use. */
  entities: { key: string; uses: number; turns: number; episodes: number }[];
  coverage: { episodes: Curve; turns: Curve; handler_names: Curve };
  /** Damage-relevant moves, abilities and items exercised in the same turn. */
  pairs: { turns: number; pairs_seen: number; pairs_ge_5_turns: number; pairs_ge_20_turns: number };
}

/** Count the exercised entities of `episodes` (synthetic only) against the inventory. */
export function runReport(episodes: Iterable<RunEpisode>, inv: MechanicsInventory, index: InventoryIndex = buildIndex(inv)): RunReport {
  const uses = new Map<string, number>();
  const turnCount = new Map<string, number>();
  const episodeCount = new Map<string, number>();
  const pairs = new Map<string, number>();
  const episodeUnits: string[][] = [];
  const turnUnits: string[][] = [];
  let total = 0;
  let turns = 0;
  let unmatched = 0;
  const bump = (map: Map<string, number>, key: string): void => void map.set(key, (map.get(key) ?? 0) + 1);

  for (const episode of episodes) {
    if (episode.data_class !== "synthetic") throw new RunReadError("an Episode is not of the synthetic data class");
    total++;
    const seen = new Set<string>();
    let turn = new Set<string>();
    const close = (): void => {
      const unit = [...turn].filter((key) => index.damage.has(key)).sort(cmp);
      turnUnits.push(unit);
      for (const key of unit) bump(turnCount, key);
      const plain = unit.filter((key) => !key.startsWith("condition:"));
      for (const [i, a] of plain.entries()) for (const b of plain.slice(i + 1)) bump(pairs, `${a} ${b}`);
      turn = new Set();
    };
    for (const line of episode.channel) {
      if (line.startsWith("|turn|")) {
        close();
        turns++;
        continue;
      }
      for (const [kind, id] of lineEvents(line)) {
        const key = entityKey(kind, id);
        if (!index.keys[kind].has(key)) unmatched++;
        else {
          turn.add(key);
          seen.add(key);
          bump(uses, key);
        }
      }
    }
    close();
    const unit = [...seen].filter((key) => index.damage.has(key)).sort(cmp);
    episodeUnits.push(unit);
    for (const key of unit) bump(episodeCount, key);
  }

  const damage = [...index.damage.keys()].sort(cmp);
  const ge = (n: number): number => [...pairs.values()].filter((count) => count >= n).length;
  return {
    episodes: total,
    turns,
    unmatched_events: unmatched,
    entities: damage.map((key) => ({ key, uses: uses.get(key) ?? 0, turns: turnCount.get(key) ?? 0, episodes: episodeCount.get(key) ?? 0 })),
    coverage: {
      episodes: curve(episodeUnits),
      turns: curve(turnUnits),
      handler_names: curve(turnUnits.map((unit) => unit.flatMap((key) => index.damage.get(key) as string[]))),
    },
    pairs: { turns: turnUnits.length, pairs_seen: pairs.size, pairs_ge_5_turns: ge(5), pairs_ge_20_turns: ge(20) },
  };
}

const pad = (generation: number): string => `g${String(generation).padStart(6, "0")}`;

/**
 * The Episodes of a run's committed generations, read one at a time: each generation's own `episodes`
 * directory and the `episodes` of every league member under `members/`. The run directory is only read.
 */
export function* readRunEpisodes(runDir: string, generations: number[]): Generator<RunEpisode> {
  if (resolve(runDir).split("/").includes("private-data")) throw new RunReadError("a run under private-data is not read");
  for (const generation of generations) {
    const base = join(runDir, "generations", pad(generation));
    const members = (() => {
      try {
        return readdirSync(join(base, "members")).sort(cmp);
      } catch {
        return [];
      }
    })();
    for (const dir of [join(base, "episodes"), ...members.map((m) => join(base, "members", m, "episodes"))]) {
      let files: string[];
      try {
        files = readdirSync(dir).filter((f) => f.endsWith(".json")).sort(cmp);
      } catch {
        continue;
      }
      for (const file of files) {
        const episode = JSON.parse(readFileSync(join(dir, file), "utf8")) as Record<string, unknown>;
        if (typeof episode.data_class !== "string" || !Array.isArray(episode.channel) || episode.channel.some((l) => typeof l !== "string")) throw new RunReadError(`${pad(generation)}: an Episode file has no data class or channel`);
        yield { data_class: episode.data_class, channel: episode.channel as string[] };
      }
    }
  }
}
