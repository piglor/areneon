// Where a team's data class comes from (ADR-008 §5, #817). A class is proven
// by source, never by a caller's label: a team is `synthetic` only if the
// JCS hash of its sets is a member of a given set of known synthetic hashes
// (today the committed fixtures' teams; later the synthetic-team manifest).
// Every other team, a `private-data/` one included, is `private`.

import { createHash } from "node:crypto";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { jcs, type Json } from "./canonical-json.ts";

export type DataClass = PrivateTeamSpec["data_class"];

/** Hashes of the teams known to be synthetic (CC0). */
export type SyntheticTeamHashes = ReadonlySet<string>;

/**
 * The known set built from verified lists of positional team hashes (the
 * `team_hash` of each record of a verified dataset, ADR-013 section 4). It
 * is the only way to turn hashes into a `SyntheticTeamHashes` besides the
 * golden fixtures: a hash that is not a SHA-256 digest is refused, so a typo
 * can never make an unrelated string a member.
 */
export function syntheticTeamHashes(...lists: Iterable<string>[]): SyntheticTeamHashes {
  const known = new Set<string>();
  for (const list of lists) {
    for (const hash of list) {
      if (!/^[0-9a-f]{64}$/.test(hash)) throw new TypeError(`${JSON.stringify(hash)} is not a team hash`);
      known.add(hash);
    }
  }
  return known;
}

/**
 * SHA-256 of the JCS bytes of a team's set ARRAY, the same on every path.
 * It excludes the spec's `data_class`, so one set of sets has one hash
 * whatever it is labelled.
 */
export function teamHash(sets: readonly unknown[]): string {
  return createHash("sha256")
    .update(jcs(sets as Json))
    .digest("hex");
}

/** The higher of two classes: `private` if either is. */
export const higherClass = (a: DataClass, b: DataClass): DataClass =>
  a === "private" || b === "private" ? "private" : "synthetic";

/**
 * A class proven by source: `synthetic` only for a known synthetic team,
 * `private` for everything else.
 */
export const classOfSets = (sets: readonly unknown[], known: SyntheticTeamHashes): DataClass =>
  known.has(teamHash(sets)) ? "synthetic" : "private";

/**
 * A team spec whose class was proven by source. The brand is a type only
 * (like `BattleClass`): only `stampTeam` returns one, so a hand-built
 * `PrivateTeamSpec` does not compile where a team enters an Episode
 * (`episodeFromRecord`). A cast defeats it; `episodeFromRecord` and
 * `readEpisode` still check the label at run time.
 */
export type StampedTeam = PrivateTeamSpec & { readonly __stampedTeam: true };

/** The one way to build a PrivateTeamSpec whose label is not the caller's say-so. */
export const stampTeam = (team: PrivateTeamSpec["team"], known: SyntheticTeamHashes): StampedTeam =>
  ({
    schema_version: 1,
    data_class: classOfSets(team, known),
    team,
  }) as StampedTeam;

/**
 * A battle's class, from the hashes of the teams it ran with: `synthetic`
 * only if both are known synthetic teams. It brands the result, so a channel
 * label must come from here (`recordedChannel`).
 */
export type BattleClass = DataClass & { readonly __battleClass: true };

export function battleClass(team_hashes: Record<"p1" | "p2", string>, known: SyntheticTeamHashes): BattleClass {
  return higherClass(
    known.has(team_hashes.p1) ? "synthetic" : "private",
    known.has(team_hashes.p2) ? "synthetic" : "private",
  ) as BattleClass;
}
