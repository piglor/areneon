// Owner-team feature inventory (#775): what a set of private teams uses
// (species, items, abilities, moves, Mega Evolution), judged by the pinned
// engine, and which of those features a gap record names. Nothing here
// decides legality or carries a rule: the engine judges the team, and the
// gap flags are read from `rules/gaps/`. Output goes only under a
// `private-data/` directory; stdout carries counts and paths, and an error
// names a file, never what is in it.

import {
  chmodSync,
  lstatSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve, sep } from "node:path";
import { type Engine, type PokemonSet } from "./engine.ts";
import { resolveFormat } from "./formats.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import type { RulesSnapshot } from "./rules.ts";
import { importPaste, TeamImportError, validateTeam } from "./teams.ts";

/** The only directory name teams are read from and reports written under. */
export const PRIVATE_DATA = "private-data";

export class InventoryError extends Error {
  override name = "InventoryError";
}

export type FeatureKind = "species" | "item" | "ability" | "move" | "mechanic";

export interface Feature {
  kind: FeatureKind;
  name: string;
}

export interface QuarantinedFeature extends Feature {
  gaps: { gap_id: string; status: string }[];
}

export interface PokemonInventory {
  species: string;
  /** The species without its forme (Rotom-Wash: Rotom); gap prose usually names this. */
  base_species: string;
  item: string;
  ability: string;
  moves: string[];
  /** The Mega forme this set's item gives its species, if any. */
  mega: string | null;
}

export type TeamStatus = "legal" | "illegal" | "import_failed";

export interface TeamInventory {
  /** Base name of the file; never a path. */
  file: string;
  status: TeamStatus;
  /** The pinned validator's messages, verbatim (private report only). */
  problems: string[];
  /** Why the paste was refused, when `status` is `import_failed` (private report only). */
  import_error: string | null;
  pokemon: PokemonInventory[];
  quarantined: QuarantinedFeature[];
}

export interface FormatWideGap {
  gap_id: string;
  status: string;
  area: string;
}

export const ADVISORY_NOTE =
  "Per-team flags are whole-word name matches against gap record prose and evidence symbols. " +
  "They are not a complete list of unverified features: a team is never clean because nothing matched, " +
  "so read format_wide_gaps too.";

export interface InventoryReport {
  report_version: 1;
  data_class: "private";
  flags_are_advisory: true;
  flags_note: string;
  /** Every gap record in the rules snapshot: they all describe the pinned engine for the format. */
  format_wide_gaps: FormatWideGap[];
  showdown_commit: string;
  format_id: string;
  rules_snapshot_hash: string;
  teams: TeamInventory[];
}

export interface InventoryCounts {
  teams: number;
  legal: number;
  illegal: number;
  import_failed: number;
  quarantined_features: number;
}

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const escapeRegExp = (text: string): string => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Which gap records name `feature`. A gap names a feature when its area or
 * description holds the feature's name as whole words (case-insensitive), or
 * one of its evidence symbols is the same identifier. Read from the records
 * in `snapshot`, so a new gap changes the flags without a code change.
 */
export function gapsNaming(
  snapshot: RulesSnapshot,
  feature: Feature,
): { gap_id: string; status: string }[] {
  const name = feature.name.trim();
  if (name === "") return [];
  const words = new RegExp(`(^|[^a-z0-9])${escapeRegExp(name.toLowerCase())}([^a-z0-9]|$)`);
  return Object.entries(snapshot.gaps)
    .filter(([, gap]) => {
      if (words.test(`${gap.area} ${gap.description}`.toLowerCase())) return true;
      return gap.evidence.some((item) => item.kind === "engine" && item.symbols.some((s) => toId(s) === toId(name)));
    })
    .map(([gap_id, gap]) => ({ gap_id, status: gap.status }))
    .sort((a, b) => a.gap_id.localeCompare(b.gap_id));
}

/**
 * The Mega forme `set`'s item gives its species in the format's mod, if any.
 * A pasted Mega forme (`X-Mega`) is checked through its base species too.
 */
function megaForme(engine: Engine, mod: string, set: PokemonSet): string | null {
  const dex = engine.sim.Dex.mod(mod);
  const item = dex.items.get(set.item);
  const species = dex.species.get(set.species);
  if (!item.exists || !species.exists || !item.megaStone) return null;
  return item.megaStone[species.name] ?? item.megaStone[species.baseSpecies] ?? null;
}

function baseSpecies(engine: Engine, mod: string, name: string): string {
  const species = engine.sim.Dex.mod(mod).species.get(name);
  return species.exists ? species.baseSpecies : name;
}

const MEGA_MECHANIC = "Mega Evolution";

function pokemonInventory(engine: Engine, mod: string, set: PokemonSet): PokemonInventory {
  return {
    species: set.species,
    base_species: baseSpecies(engine, mod, set.species),
    item: set.item,
    ability: set.ability,
    moves: [...set.moves],
    mega: megaForme(engine, mod, set),
  };
}

function featuresOf(pokemon: PokemonInventory[]): Feature[] {
  const seen = new Map<string, Feature>();
  const add = (kind: FeatureKind, name: string): void => {
    if (name !== "") seen.set(`${kind}:${toId(name)}`, { kind, name });
  };
  for (const p of pokemon) {
    add("species", p.species);
    add("species", p.base_species);
    add("item", p.item);
    add("ability", p.ability);
    for (const move of p.moves) add("move", move);
    if (p.mega !== null) add("mechanic", MEGA_MECHANIC);
  }
  return [...seen.values()].sort((a, b) => (`${a.kind}:${a.name}` < `${b.kind}:${b.name}` ? -1 : 1));
}

/** Inventory one paste. A refused paste is a result (`import_failed`), not a throw. */
export function inventoryTeam(
  engine: Engine,
  snapshot: RulesSnapshot,
  formatId: string,
  file: string,
  paste: string,
): TeamInventory {
  const format = resolveFormat(engine, formatId, snapshot);
  let team: PokemonSet[];
  try {
    team = importPaste(engine, paste);
  } catch (error) {
    if (!(error instanceof TeamImportError)) throw error;
    return { file, status: "import_failed", problems: [], import_error: error.message, pokemon: [], quarantined: [] };
  }
  const verdict = validateTeam(engine, format, team);
  const mod = engine.sim.Dex.formats.get(formatId).mod;
  const pokemon = team.map((set) => pokemonInventory(engine, mod, set));
  const quarantined = featuresOf(pokemon).flatMap((feature) => {
    const gaps = gapsNaming(snapshot, feature);
    return gaps.length > 0 ? [{ ...feature, gaps }] : [];
  });
  return {
    file,
    status: verdict.valid ? "legal" : "illegal",
    problems: verdict.problems,
    import_error: null,
    pokemon,
    quarantined,
  };
}

/**
 * The `private-data` directory `path` lies in or is: the longest prefix of
 * its real path ending in a segment named `private-data`. A symlink is
 * followed first, so a link cannot lead out. Throws if there is none.
 */
export function privateDataRoot(path: string): string {
  const real = realpathOfDeepest(resolve(path));
  const parts = real.split(sep);
  const at = parts.lastIndexOf(PRIVATE_DATA);
  if (at < 1) throw new InventoryError(`refusing a path outside a ${PRIVATE_DATA}/ directory`);
  return parts.slice(0, at + 1).join(sep);
}

/** The real path of `path`, resolving the deepest part that exists. */
function realpathOfDeepest(path: string): string {
  let existing = path;
  const rest: string[] = [];
  for (;;) {
    try {
      return join(realpathSync(existing), ...rest.reverse());
    } catch {
      rest.push(basename(existing));
      existing = dirname(existing); // the filesystem root always exists, so this ends
    }
  }
}

/** Write `text` to `path`, only if `path` is inside a `private-data/` directory. Mode 0600, dir 0700. */
export function writePrivate(path: string, text: string): void {
  const target = resolve(path);
  const root = privateDataRoot(dirname(target));
  const real = realpathOfDeepest(target);
  if (!real.startsWith(root + sep)) {
    throw new InventoryError(`refusing to write outside ${PRIVATE_DATA}/`);
  }
  mkdirSync(dirname(real), { recursive: true, mode: 0o700 });
  chmodSync(dirname(real), 0o700);
  // Replace, never reuse: an old file with looser modes must not hold the new text.
  rmSync(real, { force: true });
  writeFileSync(real, text, { mode: 0o600, flag: "wx" });
}

/** The `*.txt` files directly in `teamsDir`, sorted; each must really lie inside `private-data/`. */
export function teamFiles(teamsDir: string): string[] {
  const root = privateDataRoot(teamsDir);
  let names: string[];
  try {
    names = readdirSync(teamsDir).filter((name) => name.endsWith(".txt")).sort();
  } catch {
    throw new InventoryError("the teams directory cannot be read");
  }
  const files = names.map((name) => join(teamsDir, name));
  for (const [i, file] of files.entries()) {
    const real = realpathSync(file);
    if (!real.startsWith(root + sep) || !lstatSync(real).isFile()) {
      throw new InventoryError(`refusing ${names[i]}: not a regular file inside ${PRIVATE_DATA}/`);
    }
  }
  if (files.length === 0) throw new InventoryError("no .txt team files in the teams directory");
  return files;
}

export function countReport(report: InventoryReport): InventoryCounts {
  const by = (status: TeamStatus): number => report.teams.filter((t) => t.status === status).length;
  return {
    teams: report.teams.length,
    legal: by("legal"),
    illegal: by("illegal"),
    import_failed: by("import_failed"),
    quarantined_features: report.teams.reduce((n, t) => n + t.quarantined.length, 0),
  };
}

export interface InventoryRun {
  engine: Engine;
  teamsDir: string;
  /** Defaults to the only format in the rules snapshot. */
  formatId?: string;
  snapshot?: RulesSnapshot;
}

/** Build the report for every `*.txt` in `teamsDir`. Writes nothing. */
export function buildReport(run: InventoryRun): InventoryReport {
  const snapshot = run.snapshot ?? loadRulesSnapshot(run.engine.commit);
  const formats = Object.keys(snapshot.formats);
  const formatId = run.formatId ?? (formats.length === 1 ? formats[0] : undefined);
  if (formatId === undefined) throw new InventoryError("several formats are supported: pass --format");
  const format = resolveFormat(run.engine, formatId, snapshot);
  const teams = teamFiles(run.teamsDir).map((file) => {
    try {
      return inventoryTeam(run.engine, snapshot, formatId, basename(file), readFileSync(file, "utf8"));
    } catch (error) {
      // Only the file and the error class: an engine message can quote the team.
      throw new InventoryError(`unexpected ${error instanceof Error ? error.name : "error"} on ${basename(file)}`);
    }
  });
  return {
    report_version: 1,
    data_class: "private",
    flags_are_advisory: true,
    flags_note: ADVISORY_NOTE,
    format_wide_gaps: Object.entries(snapshot.gaps)
      .map(([gap_id, gap]) => ({ gap_id, status: gap.status, area: gap.area }))
      .sort((a, b) => a.gap_id.localeCompare(b.gap_id)),
    showdown_commit: run.engine.commit,
    format_id: formatId,
    rules_snapshot_hash: format.rules_snapshot_hash,
    teams,
  };
}

/** Where the report goes: `reports/team-inventory.json` beside the teams' `private-data` root. */
export function reportPath(teamsDir: string): string {
  return join(privateDataRoot(teamsDir), "reports", "team-inventory.json");
}

/** Build the report and write it. Returns its path and counts, never its content. */
export function runInventory(run: InventoryRun): { path: string; counts: InventoryCounts } {
  const report = buildReport(run);
  const path = reportPath(run.teamsDir);
  writePrivate(path, `${JSON.stringify(report, null, 2)}\n`);
  return { path, counts: countReport(report) };
}

export const INVENTORY_USAGE =
  "usage: team-inventory.ts [--teams-dir DIR] [--format ID]\n" +
  "  DIR defaults to $ARENEON_TEAMS_DIR, else <repo>/private-data/teams";

/** Parse the command line; `defaultDir` is used when neither option nor env names one. */
export function parseInventoryArgs(
  argv: string[],
  env: Record<string, string | undefined>,
  defaultDir: string,
): { teamsDir: string; formatId?: string } {
  let teamsDir = env["ARENEON_TEAMS_DIR"] || defaultDir;
  let formatId: string | undefined;
  for (let i = 0; i < argv.length; i += 2) {
    const flag = argv[i];
    const value = argv[i + 1];
    if (value === undefined || value.startsWith("--")) throw new InventoryError(INVENTORY_USAGE);
    if (flag === "--teams-dir") teamsDir = value;
    else if (flag === "--format") formatId = value;
    else throw new InventoryError(INVENTORY_USAGE);
  }
  return formatId === undefined ? { teamsDir } : { teamsDir, formatId };
}
