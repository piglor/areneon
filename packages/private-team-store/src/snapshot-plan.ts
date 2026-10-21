// The stratum rule of a private snapshot: a record's stratum is its base cell
// in the label set (ADR-013 section 3), with no merged strata (those belong to
// a generated dataset's manifest, and an owner's teams come from no dataset).

import { cellOf, type RowPlan } from "@areneon/team-dataset";
import type { TeamLabelSet } from "@areneon/showdown-bridge";

export function labelPlan(labelSet: TeamLabelSet): RowPlan {
  return { labelSet, stratumOf: (labels) => cellOf(labelSet, labels) };
}
