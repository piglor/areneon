// Owner-team feature inventory (#775): what a set of private teams uses
// (species, items, abilities, moves, Mega Evolution), judged by the pinned
// engine, and which of those features a gap record names. Nothing here
// decides legality or carries a rule: the engine judges the team, and the
// gap flags are read from `rules/gaps/`. Teams come through a
// `PrivateTeamSource` (ADR-014 section 11). Nothing is written to a file:
// stdout carries counts only, the detail goes to the terminal device, and an
// error names a team's label, never what is in it.

import { type Engine, type PokemonSet } from "./engine.ts";
import { resolveFormat } from "./formats.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import type { RulesSnapshot } from "./rules.ts";
import { type PrivateTeamSource, InventoryError } from "./private-team-source.ts";
import { importPaste, TeamImportError, validateTeam } from "./teams.ts";

export { InventoryError, PRIVATE_DATA } from "./private-team-source.ts";

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
  source: PrivateTeamSource;
  /** Defaults to the only format in the rules snapshot. */
  formatId?: string;
  snapshot?: RulesSnapshot;
}

/** Build the report for every team the source yields. Writes nothing. */
export async function buildReport(run: InventoryRun): Promise<InventoryReport> {
  const snapshot = run.snapshot ?? loadRulesSnapshot(run.engine.commit);
  const formats = Object.keys(snapshot.formats);
  const formatId = run.formatId ?? (formats.length === 1 ? formats[0] : undefined);
  if (formatId === undefined) throw new InventoryError("several formats are supported: pass --format");
  const format = resolveFormat(run.engine, formatId, snapshot);
  const teams: TeamInventory[] = [];
  for await (const entry of run.source.entries()) {
    try {
      teams.push(inventoryTeam(run.engine, snapshot, formatId, entry.name, entry.paste));
    } catch (error) {
      // Only the label and the error class: an engine message can quote the team.
      throw new InventoryError(`unexpected ${error instanceof Error ? error.name : "error"} on ${entry.name}`);
    }
  }
  if (teams.length === 0) throw new InventoryError("the source holds no teams");
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

/** How many teams carry a quarantined feature that each gap record names. Gap ids are public rules data. */
export function gapMatchCounts(report: InventoryReport): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const team of report.teams) {
    const ids = new Set(team.quarantined.flatMap((f) => f.gaps.map((g) => g.gap_id)));
    for (const id of [...ids].sort()) counts[id] = (counts[id] ?? 0) + 1;
  }
  return counts;
}

/** A handle on the terminal device the private detail goes to. */
export interface DetailSink {
  write(text: string): void;
  close(): void;
}

/**
 * Where the output may go. `refusal` says why this is not an interactive
 * terminal session (or null); `openDetail` opens the terminal device
 * (`/dev/tty`) and throws if there is none. Detail never goes to stdout,
 * stderr or a file.
 */
export interface InventoryTerminal {
  refusal(): string | null;
  openDetail(): DetailSink;
}

/**
 * Why a run must be refused, or null (ADR-014 X6). Accident prevention, not a
 * boundary: an agent that sets up a pty and clears its environment gets past it.
 */
export function terminalRefusal(
  env: Record<string, string | undefined>,
  isTerminal: { stdin: boolean; stdout: boolean; stderr: boolean },
): string | null {
  if (env["CLAUDECODE"] || env["AI_AGENT"]) {
    return "refusing: an agent session is running this; use your own terminal";
  }
  if (!isTerminal.stdin || !isTerminal.stdout || !isTerminal.stderr) {
    return "refusing: stdin, stdout and stderr must all be a terminal; run it in an interactive terminal";
  }
  return null;
}

export interface InventorySummary {
  counts: InventoryCounts;
  /** Gap id to the number of teams with a feature it names. */
  gap_matches: Record<string, number>;
}

/**
 * Build the report, send its detail to the terminal and return counts only.
 * Refuses, and opens the terminal device, before reading any team. Writes no file.
 */
export async function runInventory(run: InventoryRun, terminal: InventoryTerminal): Promise<InventorySummary> {
  const refusal = terminal.refusal();
  if (refusal !== null) throw new InventoryError(refusal);
  const sink = terminal.openDetail();
  try {
    const report = await buildReport(run);
    sink.write(`${JSON.stringify(report, null, 2)}\n`);
    return { counts: countReport(report), gap_matches: gapMatchCounts(report) };
  } finally {
    sink.close();
  }
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
