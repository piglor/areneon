// Episodes and recommendations (ADR-008 §3-§6), engine-free and with no Node
// built-in, like the ObservationBuilder: an Episode's content hash goes
// through the injected Sha256 port, and a recommendation is checked against
// the current request before anyone acts on it.

import type { Episode } from "@areneon/schema/types/v1/episode.ts";
import type { Recommendation } from "@areneon/schema/types/v1/recommendation.ts";
import type { RequestIdentity } from "@areneon/schema/types/v1/request-identity.ts";
import { jcs, type Json } from "./canonical-json.ts";
import type { Sha256 } from "./observation.ts";

export type { Episode, Recommendation };

/**
 * An Episode's content hash: SHA-256 of its JCS bytes, as stored, in the
 * version it was written with (ADR-008 §3). An upcasted view never takes it.
 */
export function episodeHash(episode: Episode, sha256: Sha256): Promise<string> {
  return sha256(new TextEncoder().encode(jcs(episode as unknown as Json)));
}

/** The rights that go with a data class (ADR-008 §5). */
export const rightsFor = (dataClass: Episode["data_class"]): Episode["rights"] =>
  dataClass === "synthetic" ? "cc0-synthetic" : "owner-private";

/**
 * Stale invalidation (ADR-008 §4): a recommendation answers the
 * current request only if `request_seq` and `request_hash` both match, and
 * `rqid` too when both carry one. Anything else is dropped.
 */
export function answersRequest(recommendation: Recommendation, current: RequestIdentity): boolean {
  const { identity } = recommendation;
  return (
    identity.request_seq === current.request_seq &&
    identity.request_hash === current.request_hash &&
    (identity.rqid === undefined || current.rqid === undefined || identity.rqid === current.rqid)
  );
}
