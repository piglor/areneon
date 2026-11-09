// The play-time opponent sheet sidecar (ADR-027, #1315), the sheet's twin of the candidate rows sidecar (#1271).
//
// The block a seat attaches to a query is a pure function of the Episode's own channel prefix and team, so the stream
// writer would recompute it with the belief a second time. The game process instead leaves the blocks its seat built in
// `<episode hash>.sheet.json`; the stream writer uses a file only when it names the Episode and the stream's sheet source
// hash and holds one block per decision at the same request numbers, and recomputes otherwise. A missing, stale or damaged
// file costs time and never a byte. The files are not part of a generation, an Episode or any contract.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import type { OpponentSheet } from "@areneon/schema/types/v1/opponent-sheet.ts";
import type { TrainingExample } from "@areneon/schema/types/v1/training-example.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { decisionObservations } from "./episode-io.ts";
import { opponentSheetProblem } from "./opponent-sheet.ts";

type TrainingBlock = NonNullable<TrainingExample["opponent_sheet"]>;

const sheetFile = (dir: string, hash: string): string => join(dir, `${hash}.sheet.json`);

/**
 * Write the blocks the seat attached while the game was played as the Episode's sheet file. `played` maps a request number
 * of the seat's channel to its block. Nothing is written unless every decision of the Episode has a block of `sourceHash`
 * (a gap leaves the stream writer to recompute). Returns whether a file was written.
 */
export async function writePlaySheets(
  sourceHash: string,
  episode: Parameters<typeof decisionObservations>[0],
  hash: string,
  played: ReadonlyMap<number, OpponentSheet>,
  dir: string,
): Promise<boolean> {
  const blocks: OpponentSheet[] = [];
  const request_seqs: number[] = [];
  for (const observation of await decisionObservations(episode)) {
    const found = played.get(observation.identity.request_seq);
    if (found === undefined || found.source.hash !== sourceHash) return false;
    blocks.push(found);
    request_seqs.push(observation.identity.request_seq);
  }
  mkdirSync(dir, { recursive: true });
  writeFileSync(sheetFile(dir, hash), jcs({ episode_hash: hash, source_hash: sourceHash, request_seqs, blocks } as unknown as Json));
  return true;
}

/**
 * The reader for `TrainingInput.sheet.precomputed`: the blocks an Episode's sidecar holds, or undefined when there is no
 * file, it is not for this Episode and source, its request numbers are not the Episode's decisions, a block is invalid or
 * of an oracle source, or it does not parse (the blocks are then recomputed).
 */
export function readSheetBlocks(sourceHash: string, dir: string): (hash: string, request_seqs: readonly number[]) => TrainingBlock[] | undefined {
  return (hash, request_seqs) => {
    try {
      const found = JSON.parse(readFileSync(sheetFile(dir, hash), "utf8")) as { episode_hash?: unknown; source_hash?: unknown; request_seqs?: unknown; blocks?: unknown };
      if (found.episode_hash !== hash || found.source_hash !== sourceHash || !Array.isArray(found.blocks) || !Array.isArray(found.request_seqs)) return undefined;
      if (found.request_seqs.length !== request_seqs.length || found.request_seqs.some((seq, i) => seq !== request_seqs[i])) return undefined;
      const blocks = found.blocks as TrainingBlock[];
      if (blocks.length !== request_seqs.length) return undefined;
      return blocks.every((block) => block.source.hash === sourceHash && (block.source.kind as string) !== "oracle-diagnostic" && opponentSheetProblem(block as OpponentSheet) === null) ? blocks : undefined;
    } catch {
      return undefined;
    }
  };
}
