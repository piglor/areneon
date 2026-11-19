// The teams the inventory weights read (#1465): the committed fixtures-v1 dev
// split, or the dev and training splits of a generated dataset directory. The
// validation split is never opened (a file list, not a filter: only these names
// are read), and a path under `private-data/` is refused.

import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { type Engine } from "./engine.ts";
import { GOLDEN_FORMAT, TEAMS_DIR } from "./golden.ts";
import { importPaste } from "./teams.ts";
import { type TeamMember } from "./mechanics-inventory-weights.ts";

export class InventoryTeamsError extends Error {
  override name = "InventoryTeamsError";
}

/** The only split files read: the committed layout's dev split and a generated dataset's dev and training splits. */
export const TEAM_FILES = ["teams-dev.jsonl", "records.dev.jsonl", "records.training.jsonl"] as const;

/** The teams of the split files that exist in `dir`, with the SHA-256 of their bytes joined in file order. */
export function readDatasetTeams(engine: Engine, dir: string): { teams: TeamMember[][]; sha256: string; files: string[] } {
  if (resolve(dir).split("/").includes("private-data")) throw new InventoryTeamsError("a dataset under private-data is not read");
  const files = TEAM_FILES.filter((name) => existsSync(join(dir, name)));
  if (files.length === 0) throw new InventoryTeamsError("no dev or training split file in the dataset directory");
  const hash = createHash("sha256");
  const teams: TeamMember[][] = [];
  for (const name of files) {
    const text = readFileSync(join(dir, name), "utf8");
    hash.update(text);
    for (const line of text.split("\n").filter((l) => l !== "")) {
      const record = JSON.parse(line) as { paste?: unknown };
      if (typeof record.paste !== "string") throw new InventoryTeamsError(`${name}: a record has no paste`);
      teams.push(importPaste(engine, record.paste).map((set) => ({ item: set.item, ability: set.ability, moves: set.moves })));
    }
  }
  return { teams, sha256: hash.digest("hex"), files: [...files] };
}

/** The committed fixtures-v1 dataset directory (its dev split is the default weight source). */
export const FIXTURES_V1_DIR = join(TEAMS_DIR, "datasets", "fixtures-v1");

export const USAGE_DIR = fileURLToPath(new URL(`../../../config/teams/${GOLDEN_FORMAT}/usage`, import.meta.url));

/** The latest committed usage weight table of the format (the last `<month>-<cutoff>.json` by name). */
export function latestUsage(dir: string = USAGE_DIR): { file: string; value: unknown } {
  const file = readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d+\.json$/.test(f)).sort().at(-1);
  if (file === undefined) throw new InventoryTeamsError("no usage weight table is committed");
  return { file, value: JSON.parse(readFileSync(join(dir, file), "utf8")) };
}
