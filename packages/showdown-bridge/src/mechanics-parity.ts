// The mechanics parity report (ADR-016 section 4, slice 5): docs/mechanics-parity.md,
// generated from the fixtures under fixtures/mechanics/<format>/, the calc
// divergence list (when it exists) and the gap ledger, the way
// docs/showdown-champions-gaps.md is. Nothing here is written by hand: a test
// compares the committed report with a fresh render, so an edit to a fixture,
// the config or a gap record shows up as a diff of the report.
//
// The divergence list is `calc-divergences.json` beside the fixture folders
// (written by the @smogon/calc differential, #935):
//   { "data_class": "synthetic", "divergences": [{ "case": "damage/<id>", "reason": "<one line>" }] }
// A missing file is rendered as "not built yet"; a file of any other shape is refused.

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { type MechanicsConfig, mechanicsConfigHash } from "./mechanics-config.ts";
import { MECHANICS_ADAPTER_FORMAT } from "./mechanics-result.ts";
import { MEGA_GAP_ID } from "./mechanics-speed.ts";
import { type RulesSnapshot, sha256Hex } from "./rules.ts";

export const PARITY_DOC = fileURLToPath(new URL("../../../docs/mechanics-parity.md", import.meta.url));
export const MECHANICS_FIXTURES_DIR = fileURLToPath(new URL("../../../fixtures/mechanics", import.meta.url));

export class ParityError extends Error {
  override name = "ParityError";
}

export interface FixtureCase {
  id: string;
  description: string;
}
export interface FixtureFile {
  /** Path under fixtures/mechanics/<format>/. */
  path: string;
  sha256: string;
  /** What the cases are, for the heading. */
  label: string;
  cases: FixtureCase[];
}
export interface Divergence {
  case: string;
  reason: string;
}
export interface ParityInputs {
  config: MechanicsConfig;
  snapshot: RulesSnapshot;
  fixtures: FixtureFile[];
  /** null when the divergence file does not exist. */
  divergences: Divergence[] | null;
}

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);

/** The id/description pairs of a list of fixture cases; every entry needs a string id and description. */
function casesOf(path: string, list: unknown): FixtureCase[] {
  if (!Array.isArray(list)) throw new ParityError(`${path}: expected a list of cases`);
  return list.map((entry) => {
    if (!isRecord(entry) || typeof entry.id !== "string" || typeof entry.description !== "string") throw new ParityError(`${path}: every case needs an id and a description`);
    return { id: entry.id, description: entry.description };
  });
}

/** The fixture files of the report, in a fixed order, each read for its cases and hashed. */
function readFixtures(dir: string): FixtureFile[] {
  const file = (path: string): { text: string; json: Record<string, unknown> } => {
    const text = readFileSync(join(dir, path), "utf8");
    const json = JSON.parse(text) as unknown;
    if (!isRecord(json)) throw new ParityError(`${path}: expected an object`);
    return { text, json };
  };
  const speed = file("speed/cases.json");
  const damage = file("damage/cases.json");
  const ko = file("ko/cases.json");
  const enumeration = ko.json.enumeration;
  if (!isRecord(enumeration) || typeof enumeration.id !== "string" || typeof enumeration.description !== "string") throw new ParityError("ko/cases.json: the enumeration needs an id and a description");
  return [
    { path: "speed/cases.json", sha256: sha256Hex(speed.text), label: "Speed order (hand-worked from the stat formula and the engine's modifier rules)", cases: casesOf("speed/cases.json", speed.json.cases) },
    { path: "damage/cases.json", sha256: sha256Hex(damage.text), label: "Damage rolls (hand-worked from the modifyDamage steps; each also checked against HP lost in seeded real battles)", cases: casesOf("damage/cases.json", damage.json.cases) },
    {
      path: "ko/cases.json",
      sha256: sha256Hex(ko.text),
      label: "KO counts (counted outside the probe from roll vectors; includes the boundary where the lowest roll equals HP and an explicit 256-pair enumeration)",
      cases: [
        ...casesOf("ko/cases.json", ko.json.vector_cases),
        { id: enumeration.id, description: enumeration.description },
        ...casesOf("ko/cases.json", ko.json.probe_cases),
      ],
    },
  ];
}

/** The calc divergence list, or null when the differential has not written one. */
function readDivergences(dir: string): Divergence[] | null {
  const path = join(dir, "calc-divergences.json");
  if (!existsSync(path)) return null;
  const json = JSON.parse(readFileSync(path, "utf8")) as unknown;
  if (!isRecord(json) || json.data_class !== "synthetic" || !Array.isArray(json.divergences)) throw new ParityError("calc-divergences.json: expected { data_class: synthetic, divergences: [...] }");
  return json.divergences.map((entry) => {
    if (!isRecord(entry) || typeof entry.case !== "string" || typeof entry.reason !== "string") throw new ParityError("calc-divergences.json: every divergence needs a case and a reason");
    return { case: entry.case, reason: entry.reason };
  });
}

/** Read the fixtures and the divergence list of `formatId`. */
export function loadParityInputs(config: MechanicsConfig, snapshot: RulesSnapshot, fixturesDir: string = MECHANICS_FIXTURES_DIR): ParityInputs {
  const dir = join(fixturesDir, config.format_id);
  return { config, snapshot, fixtures: readFixtures(dir), divergences: readDivergences(dir) };
}

/** One line of Markdown table text: backslashes first, then pipes, are escaped, so no pipe or line break can end a cell. */
const cell = (text: string): string =>
  text.replace(/\\/g, "\\\\").replace(/\|/g, "\\|").replace(/\s+/g, " ").trim();

/** Gap rows the results of the config depend on, with the query kinds that list each. */
function dependencies(config: MechanicsConfig): Map<string, string[]> {
  const out = new Map<string, string[]>();
  const add = (gap: string, kind: string): void => {
    const kinds = out.get(gap) ?? [];
    if (!kinds.includes(kind)) kinds.push(kind);
    out.set(gap, kinds);
  };
  for (const entry of config.scope.query_kinds) for (const gap of entry.gap_ids) add(gap, entry.kind);
  // a speed order or a damage result with a Mega Evolution names the Mega row itself (the probe's constant)
  if (config.scope.query_kinds.some((entry) => entry.kind === "speed_order")) add(MEGA_GAP_ID, "speed_order");
  return out;
}

/** Render the report. Pure: the same inputs give the same bytes. */
export function renderMechanicsParity({ config, snapshot, fixtures, divergences }: ParityInputs): string {
  const deps = [...dependencies(config)].sort(([a], [b]) => (a < b ? -1 : 1));
  const gapRows = deps.map(([id, kinds]) => {
    const gap = snapshot.gaps[id];
    if (gap === undefined) throw new ParityError(`gap ${id} is not in the gap ledger`);
    return { id, kinds: kinds.sort(), gap };
  });
  const unverified = gapRows.filter(({ gap }) => gap.status === "unverified");
  const lines: string[] = [
    "# Mechanics parity report",
    "",
    "<!-- Generated by `node packages/showdown-bridge/scripts/mechanics-parity.ts` from fixtures/mechanics/, the calc divergence list and rules/gaps/. Edit those, not this file. -->",
    "",
    `Pin: smogon/pokemon-showdown \`${snapshot.showdown_commit}\`. Format: \`${config.format_id}\`. Mechanics config hash: \`${mechanicsConfigHash(config)}\`. Adapter format: ${MECHANICS_ADAPTER_FORMAT}. Tracking: Redmine #917 (ADR-016), #936.`,
    "",
    "Areneon computes no stat, Speed or damage value: the pinned engine does, through the mechanics probe, and the engine is not independent of itself. This report lists what the probe's results were compared with, what they depend on, and what is not compared yet. Areneon makes **no claim of parity with the Pokémon Champions game**.",
    "",
    "## Comparison with the game",
    "",
    "The Wave 2 gate item \"comparison with the game and independent tools\" is met only for hand-worked values until the following are delivered. Damage, Speed order and KO results stay unverified against the game.",
    "",
    "| Source | Status |",
    "|---|---|",
    `| Hand-worked values (fixtures below) | ${fixtures.reduce((sum, file) => sum + file.cases.length, 0)} cases in ${fixtures.length} files, worked outside the probe |`,
    divergences === null
      ? "| `@smogon/calc` differential | differential not built yet (#935); no divergence list exists, so no independent-tool comparison is claimed |"
      : `| \`@smogon/calc\` differential | ${divergences.length} listed ${divergences.length === 1 ? "divergence" : "divergences"} (below) |`,
    "| In-game captures | not available: the game comparison waits for the owner's captures (#937, Wave 9); no gap row is `verified` by this work |",
    "",
    "## Gap rows the results depend on",
    "",
    `Every result lists the gap ids it depends on. ${unverified.length} of ${gapRows.length} rows below are unverified in-game.`,
    "",
    "| Gap | Status | Used by | What Showdown does at the pin |",
    "|---|---|---|---|",
    ...gapRows.map(({ id, kinds, gap }) => `| \`${id}\` | ${gap.status} | ${kinds.map((kind) => `\`${kind}\``).join(", ")} | ${cell(gap.description)} |`),
    "",
    "## Hand-worked fixtures",
    "",
    "Synthetic sets (CC0) under `fixtures/mechanics/" + config.format_id + "/`. The SHA-256 is of the committed file.",
    ...fixtures.flatMap((file) => [
      "",
      `### \`${file.path}\``,
      "",
      `${file.label}. SHA-256 \`${file.sha256}\`, ${file.cases.length} cases.`,
      "",
      "| Case | What it checks |",
      "|---|---|",
      ...file.cases.map((entry) => `| \`${entry.id}\` | ${cell(entry.description)} |`),
    ]),
    "",
    "## KO counts and blockers",
    "",
    "KO counts are integers, never a probability: `ohko_no_crit` and `ohko_crit` count the rolls (of 16) at or above the defender's current HP, `thko_no_crit` the ordered pairs of two independent no-crit rolls (of 256) whose sum is at least that HP, before the HP clamp and before damage-time effects. A defender holding a blocker gets `unsupported` counts (`ko_state_change`); the roll vectors are still returned. Residual damage, accuracy and multi-hit moves are out of scope.",
    "",
    `- Blocking items (${config.ko_blockers.items.length}): ${config.ko_blockers.items.map((id) => `\`${id}\``).join(", ") || "none"}`,
    `- Blocking abilities (${config.ko_blockers.abilities.length}): ${config.ko_blockers.abilities.map((id) => `\`${id}\``).join(", ") || "none"}`,
    "",
    "## Calc divergences",
    "",
    ...(divergences === null
      ? ["differential not built yet (#935). The divergence list is `fixtures/mechanics/" + config.format_id + "/calc-divergences.json`; this section is regenerated when it exists."]
      : divergences.length === 0
        ? ["The differential lists no divergence: every fixture case agrees with `@smogon/calc`."]
        : ["| Case | Reason |", "|---|---|", ...divergences.map((entry) => `| \`${entry.case}\` | ${cell(entry.reason)} |`)]),
    "",
  ];
  return lines.join("\n");
}
