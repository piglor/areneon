import { readFileSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
export const FORMAT = "gen9championsvgc2026regmc";
export const FORMAT_DIR = join(ROOT, "fixtures", "showdown", FORMAT);
export const WEB = join(ROOT, "packages", "replay-view", "web");

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
  createInterpreter(options: { player: string; dex?: unknown }): {
    feed(raw: string): { kind: string; caption: string | null; flagged: string | null };
    snapshot(): any;
    requestCount(): number;
    requests(): any[];
  };
  interpretChannel(player: string, lines: string[], dex?: unknown): Interpreted;
}

/** The interpreter, loaded as the page loads it: the same bytes, a classic script, no module. */
export function loadInterpreter(): Api {
  const source = readFileSync(join(WEB, "interpreter.js"), "utf8");
  return vm.runInNewContext(`${source}\n;ArReplay`, {}) as Api;
}

export function readJson(path: string): any {
  return JSON.parse(readFileSync(path, "utf8"));
}

/** The species table the page embeds, for the whole engine-data export. */
export function fullDex(): { species: Record<string, { name: string; base: string; types: string[] }>; terrains: string[] } {
  const data = readJson(join(FORMAT_DIR, "engine-data.json"));
  const species: Record<string, { name: string; base: string; types: string[] }> = {};
  for (const row of data.species) {
    species[row.id] = { name: row.name, base: row.base_species, types: row.types };
  }
  return { species, terrains: data.terrains.map((t: { id: string }) => t.id) };
}

export interface Fixture {
  id: string;
  player: string;
  lines: string[];
  team: unknown[];
}

/** Every committed own channel the replay is tested on: golden, random, unavailable, illusion. */
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
  return out;
}

export interface ViewApi {
  init(doc: any, root: any, data: any, options?: any): any;
  boot(doc: any, win: any): any;
  make(doc: any, tag: string, attrs: Record<string, string | number> | null, kids?: unknown[], svg?: boolean): any;
  align(master: string[], other: string[]): number[];
  parseChoice(choice: string): { kind: string; index: number; target: string; extra: string; raw: string }[];
}

/** The interpreter and the view, loaded in one context the way the page loads them (one script). */
export function loadView(globals: Record<string, unknown> = {}): { replay: Api; view: ViewApi } {
  const source = ["interpreter.js", "view.js"].map((f) => readFileSync(join(WEB, f), "utf8")).join("\n");
  const context = vm.createContext({ ...globals });
  vm.runInContext(source, context);
  return vm.runInContext("({ replay: ArReplay, view: ArView })", context) as { replay: Api; view: ViewApi };
}
