import { readFileSync } from "node:fs";
import { join } from "node:path";
import { legalChoiceSet, loadEngine, type LegalChoiceSet } from "@areneon/showdown-bridge";
import type { SequencedRequest } from "../src/index.ts";
import { type BattleFrames, FRAMES_DIR, type ServerLayerFrames } from "../testing/server-frames.ts";

/** The committed frame fixtures (fixtures/bot-client/frames). */
export const battle = JSON.parse(readFileSync(join(FRAMES_DIR, "battle.json"), "utf8")) as BattleFrames;
const layer = JSON.parse(readFileSync(join(FRAMES_DIR, "server-layer.json"), "utf8")) as ServerLayerFrames;

/** One hand-written server-layer frame by name. */
export function serverFrame(name: string): string {
  const entry = layer.frames[name];
  if (entry === undefined) throw new Error(`no server-layer frame ${name}`);
  return entry.frame;
}

/** The index of the first battle frame whose lines include one starting with `prefix`. */
export function frameIndex(prefix: string): number {
  const index = battle.frames.findIndex((frame) => frame.split("\n").some((line) => line.startsWith(prefix)));
  if (index < 0) throw new Error(`no battle frame with ${prefix}`);
  return index;
}

/** The choice p1 sent right after reading frame `index`. */
export function sentAfter(index: number): string {
  const entry = battle.sent.find((s) => s.after_frame === index + 1);
  if (entry === undefined) throw new Error(`p1 sent nothing after frame ${index}`);
  return entry.choice;
}

const commit = loadEngine().commit;

/** The A2 legal choice set of a request (judge: null, ADR-006 rule 6). */
export function legalFor(request: SequencedRequest): LegalChoiceSet {
  return legalChoiceSet({ request, showdown_commit: commit, judge: null });
}
