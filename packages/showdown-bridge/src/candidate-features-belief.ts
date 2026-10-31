// The candidate feature builder on sampled belief worlds: feature set `probe-damage-belief/1` (ADR-025 section 6 item 2,
// slice 5; ADR-022 erratum E-22.2). It has `probe-damage/1`'s row keys and wire shape; the difference is the opponent: each
// row is the MEAN over N worlds the opponent model samples from the player's own view, instead of one fixed assumed set.
//
//   - The N worlds come from the belief port on the stream SHA-256(JCS({request_hash, request_seq, player, purpose:
//     "probe-features-worlds/1"})), all three of the player's own request, so the rows stay a pure function of the player's view
//     (no seed, opponent team or omniscient log is a parameter of anything here; the same type rule as `candidateFeatures`).
//   - Each world gives the opposing actives' assumed sets exactly as `damage-aware@v2` builds them (`buildWorldAssumedSet`: public
//     facts win over the world, a revealed move is the only threat); a world that contradicts a public fact is dropped and counted.
//   - Row values are integer means over the used worlds (floor of the sum over the count), the same for `mean_ppm`, `max_ppm`,
//     `min_ppm` and `ohko_rolls`. A row that is `unsupported` in ANY used world is `unsupported` with the reason of the first
//     such world (the worlds are in sample order): a partial mean would be a guess. The probe memo by query hash is shared across
//     the worlds and the game.
//   - A belief refusal (any code but the run-setup defects of `BELIEF_DEFECTS`, which propagate) or no consistent world falls back
//     to ADR-018's v1 assumed sets for that request: the rows are then `probe-damage/1`'s bytes under this set's hash. The fallback
//     is counted by cause in `memo.belief.fallbacks` (the service reports it per task).
//   - `feature_set_hash` adds the belief identity, the world count and `BELIEF_FEATURES_FORMAT` to `probe-damage/1`'s preimage;
//     the other sets' hashes are untouched.
import { AssumedSetError, buildWorldAssumedSet, WorldMismatchError } from "./damage-aware-assumed-set.ts";
import type { Candidate } from "./choices.ts";
import { jcs } from "./canonical-json.ts";
import { type BeliefFeatureCounts, type CandidateFeatures, createFeatureMemo, damageRows, emptyBeliefCounts, FEATURE_SET_BELIEF, type FeatureConfigs, type FeatureMemo, type FeatureRow, featureSetHash, featureSetIdentity, lastRequest, openRequest, refusing } from "./candidate-features.ts";
import { DamageAwareError, type Foe } from "./damage-aware-view.ts";
import { BELIEF_DEFECTS, refusalCode, type WorldLike } from "./damage-aware-worlds.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import type { PlayerChannel } from "./observation.ts";
import { decodeRequest } from "./request.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import { sha256Hex } from "./rules.ts";

/** The purpose string of the feature builder's world stream (ADR-025 section 4, a purpose of its own so it never shares a draw with the baseline's). */
export const FEATURE_WORLDS_PURPOSE = "probe-features-worlds/1";

export interface CandidateFeaturesBelief extends Omit<CandidateFeatures, "feature_set"> {
  feature_set: typeof FEATURE_SET_BELIEF;
}

/** The 32-byte stream of one request: every input is the player's own request. */
export function featureWorldsStream(request: { readonly request_hash: string; readonly request_seq: number; readonly player: string }): Uint8Array {
  const hex = sha256Hex(jcs({ request_hash: request.request_hash, request_seq: request.request_seq, player: request.player, purpose: FEATURE_WORLDS_PURPOSE }));
  return Uint8Array.from(hex.match(/../g) as string[], (byte) => parseInt(byte, 16));
}

const REQUEST = "|request|";
const rowKey = (row: FeatureRow): string => `${row.slot}/${row.move}/${row.target}/${row.mega}`;
const mean = (values: readonly number[]): number => Math.floor(values.reduce((sum, value) => sum + value, 0) / values.length);

/** The rows of every used world, one list per world in canonical order, as one list of the integer means (unsupported in any world: unsupported). */
export function meanRows(perWorld: readonly (readonly FeatureRow[])[]): FeatureRow[] {
  const byKey = new Map<string, FeatureRow[]>();
  for (const rows of perWorld) for (const row of rows) byKey.set(rowKey(row), [...(byKey.get(rowKey(row)) ?? []), row]);
  const out: FeatureRow[] = [];
  for (const rows of byKey.values()) {
    const first = rows[0] as FeatureRow;
    const key = { slot: first.slot, move: first.move, target: first.target, mega: first.mega };
    const refused = rows.find((row) => row.kind === "unsupported");
    if (refused !== undefined && refused.kind === "unsupported") {
      out.push({ ...key, kind: "unsupported", reason: refused.reason });
      continue;
    }
    const computed = rows.filter((row): row is Extract<FeatureRow, { kind: "computed" }> => row.kind === "computed");
    const ohko = computed.map((row) => row.ohko_rolls);
    out.push({
      ...key,
      kind: "computed",
      mean_ppm: mean(computed.map((row) => row.mean_ppm)),
      max_ppm: mean(computed.map((row) => row.max_ppm)),
      min_ppm: mean(computed.map((row) => row.min_ppm)),
      ohko_rolls: ohko.some((value) => value === null) ? null : mean(ohko as number[]),
    });
  }
  return out.sort((a, b) => a.slot - b.slot || a.move - b.move || a.target - b.target || Number(a.mega) - Number(b.mega));
}

/**
 * The features of one decision under `probe-damage-belief/1`. `memo` is this game's: with or without it the rows are the same
 * bytes. Throws `FeatureBuildError` when the request cannot be featured (as `candidateFeatures`).
 */
export function candidateFeaturesBelief(
  engine: Engine,
  configs: FeatureConfigs,
  channel: PlayerChannel,
  team: readonly Readonly<PokemonSet>[],
  candidates: readonly Pick<Candidate, "choice" | "pending">[],
  memo: FeatureMemo = createFeatureMemo(),
): CandidateFeaturesBelief {
  const belief = configs.belief;
  if (belief === undefined) throw new DamageAwareError("probe-damage-belief/1 needs the belief port");
  const { port, worlds: n } = belief;
  if (!Number.isInteger(n) || n < 1) throw new DamageAwareError("the world count is a positive integer");
  if (port.prior_format_id !== configs.config.format_id) throw new DamageAwareError("the usage prior is for another format than the record's: it is never used silently");
  const snapshot = configs.snapshot ?? loadRulesSnapshot(engine.commit);
  const identity = featureSetIdentity(engine, { ...configs, snapshot }, FEATURE_SET_BELIEF);
  const out = (rows: FeatureRow[]): CandidateFeaturesBelief => ({ feature_set: FEATURE_SET_BELIEF, feature_set_hash: featureSetHash(identity), rows });

  const body = lastRequest(channel);
  if (body.teamPreview !== undefined || body.active === undefined) return out([]);
  const { view, state, allFlags } = openRequest(engine, configs, snapshot, channel, memo);
  const counts: BeliefFeatureCounts = (memo.belief ??= emptyBeliefCounts());
  const own = refusing(() => view.ownSlots(body, state, team));
  const fallback = (cause: string): FeatureRow[] => {
    counts.fallbacks[cause] = (counts.fallbacks[cause] ?? 0) + 1;
    return refusing(() => {
      let foes: Foe[] | null = null;
      return damageRows(view, state, body, own, () => (foes ??= view.foesOf(state, (mon) => view.assumedFor(mon, state))), candidates);
    });
  };

  // The request's own identity: what the stream is made of (`request_seq` counts every request of the channel so far, as an observation's does).
  const lines = channel.lines;
  const request_hash = decodeRequest((lines[lines.length - 1] as string).slice(REQUEST.length)).request_hash;
  const request_seq = lines.filter((line) => line.startsWith(REQUEST)).length;

  counts.decisions += 1;
  let sampled;
  try {
    sampled = port.sample({ opponent: state.opponent, flags: allFlags, stream: featureWorldsStream({ request_hash, request_seq, player: channel.player }), n });
  } catch (error) {
    const code = refusalCode(error);
    if (code === null || BELIEF_DEFECTS.has(code)) throw error;
    return out(fallback(code));
  }
  counts.worlds_sampled += sampled.worlds.length;
  counts.redraws += sampled.redraws;

  const perWorld: FeatureRow[][] = [];
  for (const { world } of sampled.worlds) {
    const rows = refusing(() => {
      let foes: Foe[];
      try {
        foes = view.foesOf(state, (mon, position) => {
          const set = worldAssumed(world, mon, position);
          if (mon.status !== null && mon.ability === null && view.statusBlocked(set, mon, state)) throw new WorldMismatchError("the Pokemon's ability in this world blocks the status it is shown with");
          return set;
        });
      } catch (error) {
        // A world that contradicts a public fact, or whose set the validator refuses once the public facts are applied, is dropped like `damage-aware@v2` drops it.
        if (!(error instanceof WorldMismatchError || error instanceof AssumedSetError)) throw error;
        return null;
      }
      return damageRows(view, state, body, own, () => foes, candidates);
    });
    if (rows === null) counts.contradicted_worlds += 1;
    else perWorld.push(rows);
  }
  if (perWorld.length === 0) return out(fallback("no_consistent_world"));
  return out(meanRows(perWorld));

  function worldAssumed(world: WorldLike, mon: Parameters<typeof buildWorldAssumedSet>[1], position: number) {
    const slot = world.slots[position];
    if (slot === undefined || !slot.brought || slot.set === null) throw new WorldMismatchError("the world does not bring the Pokemon on the field");
    return buildWorldAssumedSet({ engine, vocabulary: configs.vocabulary, format_id: configs.config.format_id }, mon, slot.set);
  }
}
