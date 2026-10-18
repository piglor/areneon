import { fileURLToPath } from "node:url";
import type { Engine } from "./engine.ts";
import type { GapEvidence, GapRecord, RulesSnapshot } from "./rules.ts";

/** The gap ledger as Markdown, generated from `rules/gaps/` (ADR-007). */
export const GAPS_DOC = fileURLToPath(new URL("../../../docs/showdown-champions-gaps.md", import.meta.url));

/** How the pinned engine names a format; the doc header shows it. */
export interface FormatLabel {
  format_id: string;
  name: string;
  mod: string;
}

type Gap = Omit<GapRecord, "schema_version">;

const cell = (text: string): string => text.replace(/\|/g, "\\|");

function evidence(item: GapEvidence): string {
  if (item.kind === "fixture") return `fixture \`${item.path}\``;
  const symbols = item.symbols.map((symbol) => `\`${symbol}\``).join(", ");
  return symbols ? `\`${item.path}\` ${symbols}` : `\`${item.path}\``;
}

function table(gaps: Gap[]): string[] {
  return [
    "| Area | What Showdown does at the pin | Where |",
    "|---|---|---|",
    ...gaps.map((gap) => `| ${cell(gap.area)} | ${cell(gap.description)} | ${gap.evidence.map(evidence).join(", ")} |`),
  ];
}

/**
 * Render the gap ledger. Rows are ordered by `gap_id`. A gap with no
 * evidence has not been exercised and is listed, not tabulated.
 */
export function renderGapsDoc(snapshot: RulesSnapshot, formats: FormatLabel[]): string {
  const gaps: Gap[] = Object.entries(snapshot.gaps)
    .map(([gap_id, gap]) => ({ gap_id, ...gap }))
    .sort((a, b) => (a.gap_id < b.gap_id ? -1 : 1));
  const tabulated = (status: GapRecord["status"]): Gap[] =>
    gaps.filter((gap) => gap.status === status && (status !== "unverified" || gap.evidence.length > 0));
  const sections: [string, string[]][] = [
    ["Verified in-game", table(tabulated("verified"))],
    ["Implemented by the pinned engine, unverified in-game", table(tabulated("unverified"))],
    ["Not supported by Areneon", table(tabulated("unsupported"))],
    [
      "Unknown or not yet exercised",
      gaps.filter((gap) => gap.status === "unverified" && gap.evidence.length === 0).map((gap) => `- ${gap.description}`),
    ],
  ];
  const formatList = formats
    .map(({ format_id, name, mod }) => `\`${format_id}\` (\`${name}\`, mod \`${mod}\`)`)
    .join(", ");
  return [
    "# Showdown Champions gap list",
    "",
    "<!-- Generated from rules/gaps/ by `node packages/showdown-bridge/scripts/gaps-doc.ts`. Edit the records, not this file. -->",
    "",
    `Pin: smogon/pokemon-showdown \`${snapshot.showdown_commit}\`. Format: ${formatList}. Tracking: Redmine #670, ADR-003, ADR-007.`,
    "",
    "Areneon makes **no claim of parity with the Pokémon Champions game**. This list records what the pinned engine does for Champions so that nothing is assumed silently. Every row stays *unverified* until an in-game golden fixture confirms it. Results that depend on an unverified row are labelled with it.",
    "",
    "Engine paths are relative to `vendor/pokemon-showdown`. Review the records in `rules/gaps/` in every pin-bump PR.",
    ...sections
      .filter(([heading, lines]) => lines.length > (heading.startsWith("Unknown") ? 0 : 2))
      .flatMap(([heading, lines]) => ["", `## ${heading}`, "", ...lines]),
    "",
  ].join("\n");
}

/** Each snapshot format as the pinned engine names it. */
export function formatLabels(engine: Engine, snapshot: RulesSnapshot): FormatLabel[] {
  return Object.keys(snapshot.formats)
    .sort()
    .map((format_id) => {
      const { name, mod } = engine.sim.Dex.formats.get(format_id);
      return { format_id, name, mod };
    });
}
