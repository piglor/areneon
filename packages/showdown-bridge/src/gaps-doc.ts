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

function evidence(item: GapEvidence): string {
  if (item.kind === "fixture") return `fixture \`${item.path}\``;
  const symbols = item.symbols.map((symbol) => `\`${symbol}\``).join(", ");
  return symbols ? `\`${item.path}\` ${symbols}` : `\`${item.path}\``;
}

// The gap schema keeps area and description to one line without `|`, and
// paths and symbols to plain characters, so nothing here needs escaping.
const row = (gap: Gap): string => `| ${gap.area} | ${gap.description} | ${gap.evidence.map(evidence).join(", ")} |`;

const TABLE_HEADER = ["| Area | What Showdown does at the pin | Where |", "|---|---|---|"];

/**
 * Render the gap ledger. Rows are ordered by `gap_id`. A gap with no
 * evidence has not been exercised and is listed, not tabulated.
 */
export function renderGapsDoc(snapshot: RulesSnapshot, formats: FormatLabel[]): string {
  const gaps: Gap[] = Object.entries(snapshot.gaps)
    .map(([gap_id, gap]) => ({ gap_id, ...gap }))
    .sort((a, b) => (a.gap_id < b.gap_id ? -1 : 1));
  const exercised = (status: GapRecord["status"]): Gap[] =>
    gaps.filter((gap) => gap.status === status && (status !== "unverified" || gap.evidence.length > 0));
  const sections: { heading: string; header: string[]; body: string[] }[] = [
    { heading: "Verified in-game", header: TABLE_HEADER, body: exercised("verified").map(row) },
    {
      heading: "Implemented by the pinned engine, unverified in-game",
      header: TABLE_HEADER,
      body: exercised("unverified").map(row),
    },
    { heading: "Not supported by Areneon", header: TABLE_HEADER, body: exercised("unsupported").map(row) },
    {
      heading: "Unknown or not yet exercised",
      header: [],
      body: gaps.filter((gap) => gap.status === "unverified" && gap.evidence.length === 0).map((gap) => `- ${gap.description}`),
    },
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
    "Areneon makes **no claim of parity with the Pokémon Champions game**. This list records what the pinned engine does for Champions so that nothing is assumed silently. A row is *unverified* until an in-game golden fixture confirms it. Results that depend on an unverified row are labelled with it.",
    "",
    "Engine paths are relative to `vendor/pokemon-showdown`. Review the records in `rules/gaps/` in every pin-bump PR.",
    ...sections
      .filter(({ body }) => body.length > 0)
      .flatMap(({ heading, header, body }) => ["", `## ${heading}`, "", ...header, ...body]),
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
