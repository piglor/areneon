// The composition root of the Team Preview solver candidate of the progress evaluation (ADR-029 section 8, #1260): the committed
// `PreviewSolveConfig`, the opponent model's full-team draw (generator-matched arm) and its matrix solver, joined for the committed
// format (the same restriction the search candidate and the damage-aware baseline have: their vocabulary is the golden one).

import { readerSha256 } from "../../public-state/src/index.ts";
import { type Engine, loadPreviewSolveConfig, loadRulesSnapshot, type PreviewSolveConfig, type PreviewSolveSetup, previewSolveSetup, resolveFormat } from "@areneon/showdown-bridge";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { createPreviewSolvePorts } from "./preview-solver-port.ts";

export function progressPreviewSolveSetup(engine: Engine, formatId: string, config?: PreviewSolveConfig): PreviewSolveSetup {
  if (formatId !== GOLDEN_FORMAT) throw new Error(`the Team Preview solver has a vocabulary for ${GOLDEN_FORMAT} only, not ${formatId}`);
  const vocabulary = goldenTeamVocabulary();
  const snapshot = loadRulesSnapshot(engine.commit);
  const manifest = resolveFormat(engine, formatId, snapshot).manifest;
  const ports = createPreviewSolvePorts({ engine, vocabulary, reader_sha256: readerSha256(), manifest, arm: "generator-matched" });
  return previewSolveSetup({ engine, format_id: formatId, vocabulary, config: config ?? loadPreviewSolveConfig(formatId), teams: ports.teams, solver: ports.solver, data_class: "synthetic" });
}
