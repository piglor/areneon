import { readFileSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
export const FORMAT = "gen9championsvgc2026regmc";
export const FORMAT_DIR = join(ROOT, "fixtures", "showdown", FORMAT);
export const INTERPRETER = join(ROOT, "packages", "public-state", "src", "interpreter.cjs");

export interface Frame {
  index: number;
  line: string;
  kind: string;
  caption: string | null;
  flagged: string | null;
  request: number;
  state: any;
}
export interface Interpreted {
  frames: Frame[];
  flags: { index: number; line: string; reason: string }[];
  requests: any[];
}
export interface Api {
  toId(text: string): string;
  parseLine(raw: string): { raw: string; kind: string; args: string[]; tags: Record<string, string> };
  createInterpreter(options: { player: string; dex?: unknown; omniscient?: boolean }): {
    feed(raw: string): { kind: string; caption: string | null; flagged: string | null };
    snapshot(): any;
    requestCount(): number;
    requests(): any[];
  };
  interpretChannel(player: string, lines: string[], dex?: unknown, omniscient?: boolean): Interpreted;
}

/**
 * The interpreter, `require`d from its own file so coverage sees it: it is a classic script that
 * also exports itself when a `module` exists (the page has none). The page-shaped load, one script
 * in a fresh context, is `loadPageScript`.
 */
export function loadInterpreter(): Api {
  return require(INTERPRETER) as Api;
}

export function readJson(path: string): any {
  return JSON.parse(readFileSync(path, "utf8"));
}

/** The species table the page embeds, for the whole engine-data export. */
export function fullDex(): {
  species: Record<string, { name: string; base: string; types: string[] }>;
  terrains: string[];
  volatile_copy: { not_copied: string[]; shed_tail_copies: string[] };
} {
  const data = readJson(join(FORMAT_DIR, "engine-data.json"));
  const species: Record<string, { name: string; base: string; types: string[] }> = {};
  for (const row of data.species) {
    species[row.id] = { name: row.name, base: row.base_species, types: row.types };
  }
  const copy = readJson(join(ROOT, "ml", "areneon_ml", "reducer", "data", `volatile-copy.${data.format_id}.json`));
  return {
    species,
    terrains: data.terrains.map((t: { id: string }) => t.id),
    volatile_copy: { not_copied: copy.not_copied, shed_tail_copies: copy.shed_tail_copies },
  };
}

export interface Fixture {
  id: string;
  player: string;
  lines: string[];
  team: unknown[];
}

/** Every committed own channel the replay is tested on: golden, random, unavailable, illusion, baton-pass, fling. */
export function fixtures(): Fixture[] {
  const out: Fixture[] = [];
  const team = readJson(join(FORMAT_DIR, "synthetic-team.json"));
  for (const p of ["p1", "p2"]) {
    const doc = readJson(join(FORMAT_DIR, "channels", `${p}.json`));
    out.push({ id: `golden/${p}`, player: p, lines: doc.channel.lines, team });
  }
  const battles = (dir: string, files: string[]) => {
    for (const f of files) {
      const doc = readJson(join(FORMAT_DIR, dir, f));
      for (const p of ["p1", "p2"]) {
        out.push({ id: `${dir}/${f.replace(".json", "")}/${p}`, player: p, lines: doc.channels[p].lines, team: doc.teams[p] });
      }
    }
  };
  const list = (dir: string) => readdirSync(join(FORMAT_DIR, dir)).filter((f) => /^b\d+\.json$/.test(f)).sort();
  battles("random", list("random"));
  battles("unavailable", list("unavailable"));
  battles("illusion", ["battle.json"]);
  battles("baton-pass", ["battle.json"]);
  battles("baton-pass-volatiles", ["battle.json"]);
  battles("fling", ["battle.json"]);
  return out;
}

