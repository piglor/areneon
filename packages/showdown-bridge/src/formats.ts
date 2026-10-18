import type { Engine } from "./engine.ts";
import { canonicalJson } from "./canonical-json.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import {
  formatRulesHash,
  formatRulesRecord,
  type FormatRulesRecord,
  rulesSnapshotHash,
  type RulesSnapshot,
  sha256Hex,
} from "./rules.ts";

/** Reproducibility key carried by every fixture, episode and benchmark. */
export interface VersionManifest {
  showdown_commit: string;
  format_id: string;
  regulation: string;
  ruleset_hash: string;
  /** Replay identity of Areneon's own rules for this format (ADR-007). */
  format_rules_hash: string;
}

export interface ResolvedFormat {
  formatId: string;
  /** Embedded in every artifact so replay needs neither a database nor GitHub. */
  rules: FormatRulesRecord;
  manifest: VersionManifest;
  /** Provenance only: which published snapshot the rules came from. */
  rules_snapshot_hash: string;
}

export class UnsupportedFormatError extends Error {
  override name = "UnsupportedFormatError";
}

/**
 * Resolve `formatId` against the rules snapshot and the pinned engine.
 * Which formats Areneon runs, and which engine rules disqualify one, are
 * data (ADR-007); whether the format exists and how it plays is the engine's
 * answer. Anything absent from the snapshot (singles, Bo3, other mods) is
 * rejected, as is a format the engine defines differently or with a
 * forbidden rule.
 *
 * Without `snapshot` it reads and validates `rules/` on every call. That is
 * fine for scripts and tests; anything resolving repeatedly loads the
 * snapshot once and passes it (live runtimes get theirs from #704).
 */
export function resolveFormat(
  engine: Engine,
  formatId: string,
  snapshot: RulesSnapshot = loadRulesSnapshot(engine.commit),
): ResolvedFormat {
  if (snapshot.showdown_commit !== engine.commit) {
    throw new UnsupportedFormatError(
      `rules snapshot was built for Showdown ${snapshot.showdown_commit}, not ${engine.commit}`,
    );
  }
  const spec = snapshot.formats[formatId];
  if (!spec) {
    throw new UnsupportedFormatError(`${formatId} is not a supported Areneon format`);
  }
  const format = engine.sim.Dex.formats.get(formatId);
  if (!format.exists || format.gameType !== spec.game_type) {
    throw new UnsupportedFormatError(
      `${formatId} is not a ${spec.game_type} format at Showdown ${engine.commit}`,
    );
  }
  const rules = engine.sim.Dex.formats.getRuleTable(format);
  for (const { rule, reason } of spec.forbidden_rules) {
    if (rules.has(rule)) throw new UnsupportedFormatError(`${formatId} has ${rule}: ${reason}`);
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
  const record = formatRulesRecord(snapshot, formatId);
  return {
    formatId,
    rules: record,
    rules_snapshot_hash: rulesSnapshotHash(snapshot),
    manifest: {
      showdown_commit: engine.commit,
      format_id: formatId,
      regulation: record.regulation,
      ruleset_hash: sha256Hex(canonicalJson(ruleset)),
      format_rules_hash: formatRulesHash(record),
    },
  };
}
