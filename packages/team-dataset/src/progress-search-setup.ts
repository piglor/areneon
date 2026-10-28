// The composition root of the search candidate of the progress evaluation (ADR-026 slice P): the committed `rerank-1` config, the
// reviewed damage-aware, mechanics and world-writer records, and the opponent model's generator-matched worlds, joined for the
// committed format (the same restriction the damage-aware baseline has: its vocabulary is the golden one).

import { readerSha256 } from "../../public-state/src/index.ts";
import { type Engine, goldenWorldFactsData, loadRulesSnapshot, resolveFormat, type SearchSetup, searchSetup } from "@areneon/showdown-bridge";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { createSearchWorlds } from "./search-worlds.ts";

export function progressSearchSetup(engine: Engine, formatId: string): SearchSetup {
  if (formatId !== GOLDEN_FORMAT) throw new Error(`the search has a vocabulary for ${GOLDEN_FORMAT} only, not ${formatId}`);
  const vocabulary = goldenTeamVocabulary();
  const snapshot = loadRulesSnapshot(engine.commit);
  const manifest = resolveFormat(engine, formatId, snapshot).manifest;
  const port = createSearchWorlds({ engine, vocabulary, reader_sha256: readerSha256(), manifest, arm: "generator-matched" });
  return searchSetup({ engine, format_id: formatId, vocabulary, port, facts_data: goldenWorldFactsData(), data_class: "synthetic", snapshot });
}
