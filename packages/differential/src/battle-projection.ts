// The `projection` comparison kind of the battle level (ADR-030 groundwork G4, #1464; ADR-028 section 5 direct tensor path). A
// backend that keeps battle state as the battle-state spec's tensors is checked, at a step, by its player-view projection for each
// seat: the views must equal the reference projection of the pinned engine's state at that step field for field. A view is what
// the projection lets one seat read: field ids (with `@own` / `@opponent` for a field of a side), each with its values, the
// unknown marker as null; a field the seat may not read is absent. The comparison names the first differing seat and field with
// fixed words: a field id of the spec, never a value. Engine-free; the reference views come from the reference packer and the
// projection (`ml/areneon_ml/projection`).
import type { Seat } from "./battle-backend.ts";
import { SEATS } from "./battle-backend.ts";
import type { BattleDifference } from "./battle-compare.ts";
import { CanonicalError, type Json, jcs } from "./canonical.ts";

/** One seat's view: the record it was projected with, by hash, and the fields it may read. */
export interface SeatView {
  seat: Seat;
  spec_hash: string;
  fields: Record<string, Json>;
}

/** The reference views of a trace's step, or null when the oracle has none for it (the step is then not projected). */
export type ProjectionOracle = (trace: number, step: number) => Record<Seat, SeatView> | null;

export class ProjectionError extends Error {
  override name = "ProjectionError";
}

/** A field id of the spec, with the side it belongs to for a field of a side. */
const FIELD_KEY = /^(active|battle|field|roster|seat_active|seat_field|seat_roster|seat_side|side)\.[a-z0-9_]+(@(own|opponent))?$/;

const isView = (value: unknown): value is SeatView =>
  typeof value === "object" && value !== null && typeof (value as SeatView).seat === "string" && typeof (value as SeatView).spec_hash === "string" && typeof (value as SeatView).fields === "object" && (value as SeatView).fields !== null;

/** The JCS bytes of a value, or null when it is no integer JSON value (a float compares unequal to everything). */
const bytes = (value: unknown): string | null => {
  try {
    return jcs(value as Json);
  } catch (error) {
    if (error instanceof CanonicalError) return null;
    throw error;
  }
};

/**
 * Compare a backend's views of a step with the reference views, seat by seat (p1 first), field by field in sorted id order. The
 * reference views must be integer JSON (a ProjectionError otherwise: the oracle's defect, not the backend's).
 */
export function checkProjection(got: Record<Seat, unknown>, want: Record<Seat, SeatView>, step: number): BattleDifference | null {
  const at = (seat: Seat, where: string): BattleDifference => ({ kind: "projection", step, seat, where });
  for (const seat of SEATS) {
    const expected = want[seat];
    for (const [key, value] of Object.entries(expected.fields)) {
      if (bytes(value) === null) throw new ProjectionError(`the reference view of ${seat} holds a value that is no integer JSON at ${key}`);
    }
    const view = got[seat];
    if (!isView(view)) return at(seat, "the projection is not a seat view");
    if (view.seat !== seat || view.spec_hash !== expected.spec_hash) return at(seat, "the projection is of another seat or another spec");
    const keys = [...new Set([...Object.keys(view.fields), ...Object.keys(expected.fields)])].sort();
    for (const key of keys) {
      if (!FIELD_KEY.test(key)) return at(seat, "the projection names a field outside the spec");
      if (!(key in view.fields) || !(key in expected.fields)) return at(seat, `projected field ${key} is present on one side only`);
      if (bytes(view.fields[key]) !== bytes(expected.fields[key])) return at(seat, `projected field ${key} differs`);
    }
  }
  return null;
}
