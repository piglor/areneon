import { createHash } from "node:crypto";
import type { Engine } from "./engine.ts";
import { canonicalJson } from "./request-identity.ts";

/** What Areneon requires of a format before it will run it. */
export interface FormatSpec {
  regulation: string;
  gameType: "doubles";
  /** Active Pokémon per side, i.e. leads chosen at Team Preview. */
  activeSlots: number;
}

/**
 * Formats Areneon supports, keyed by Showdown format ID. CTS needs Bo1 with
 * Open Team Sheets declined (ADR-003), so the Bo3 variant, which forces open
 * sheets, is deliberately absent. A new regulation is a new entry here plus
 * a pin bump.
 */
export const SUPPORTED_FORMATS: Readonly<Record<string, FormatSpec>> = {
  gen9championsvgc2026regmc: { regulation: "M-C", gameType: "doubles", activeSlots: 2 },
};

/** Reproducibility key carried by every fixture, episode and benchmark. */
export interface VersionManifest {
  showdown_commit: string;
  format_id: string;
  regulation: string;
  ruleset_hash: string;
}

export interface ResolvedFormat {
  formatId: string;
  spec: FormatSpec;
  manifest: VersionManifest;
}

export class UnsupportedFormatError extends Error {
  override name = "UnsupportedFormatError";
}

/**
 * Resolve `formatId` against the pinned engine. Anything not in the registry
 * (singles, Bo3, other mods) is rejected, as is a registry entry the engine
 * no longer defines the way Areneon expects.
 */
export function resolveFormat(
  engine: Engine,
  formatId: string,
  registry: Readonly<Record<string, FormatSpec>> = SUPPORTED_FORMATS,
): ResolvedFormat {
  const spec = registry[formatId];
  if (!spec) {
    throw new UnsupportedFormatError(`${formatId} is not a supported Areneon format`);
  }
  const format = engine.sim.Dex.formats.get(formatId);
  if (!format.exists || format.gameType !== spec.gameType) {
    throw new UnsupportedFormatError(
      `${formatId} is not a ${spec.gameType} format at Showdown ${engine.commit}`,
    );
  }
  const rules = engine.sim.Dex.formats.getRuleTable(format);
  if (rules.has("forceopenteamsheets")) {
    throw new UnsupportedFormatError(`${formatId} forces Open Team Sheets; CTS needs Bo1`);
  }
  // Everything that changes how the format plays. Mod data (learnsets,
  // mechanics) is covered by showdown_commit; `mod` says which data applies.
  const ruleset = {
    mod: format.mod,
    gameType: format.gameType,
    rules: [...rules.keys()].sort(),
    values: Object.fromEntries(rules.valueRules),
    teamSize: [rules.minTeamSize, rules.maxTeamSize, rules.pickedTeamSize],
    adjustLevel: rules.adjustLevel,
    evLimit: rules.evLimit,
  };
  return {
    formatId,
    spec,
    manifest: {
      showdown_commit: engine.commit,
      format_id: formatId,
      regulation: spec.regulation,
      ruleset_hash: createHash("sha256").update(canonicalJson(ruleset)).digest("hex"),
    },
  };
}
