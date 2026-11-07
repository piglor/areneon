// The composition root of the Team Preview solver's ports (ADR-029 section 3, ticket #1259): the belief's full-team draw
// (`createOpponentTeams`, ADR-025 erratum E-25.1) and the integer matrix solver of the opponent model, joined structurally for
// the bridge's `previewSolveActor`, which imports neither (the bridge does not depend on the opponent model).

import { SOLVER_FORMAT, solvePreview } from "@areneon/opponent-model";
import type { PreviewSolveSetup } from "@areneon/showdown-bridge";
import { createOpponentTeams, type OpponentTeamPort } from "./opponent-teams.ts";
import type { SearchWorldsContext } from "./search-worlds.ts";

/** The matrix solver as the bridge's port: `preview-solver/1` and `solve` unchanged. */
export const previewSolverPort: PreviewSolveSetup["solver"] = {
  format: SOLVER_FORMAT,
  solve: (matrix, iterations, kind) => solvePreview(matrix, iterations, kind),
};

/** The ports a solve is built from for one prior arm. `teams` also exposes its counts. */
export function createPreviewSolvePorts(context: SearchWorldsContext): { teams: OpponentTeamPort; solver: PreviewSolveSetup["solver"] } {
  return { teams: createOpponentTeams(context), solver: previewSolverPort };
}
