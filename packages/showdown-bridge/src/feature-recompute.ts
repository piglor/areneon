// The candidate feature rows of the training stream, recomputed by the game processes (ADR-022 section 4, #1185).
//
// The stream's rows are a pure function of an Episode (its channel prefix at each decision, its own team and the
// candidates), so any process may compute them. Done serially by the process that writes the stream, they were a
// tail that did not shrink with the number of game processes. Here each game process recomputes the rows of the
// Episodes it wrote, as soon as its games are done, into files of a scratch directory; the stream writer uses a
// file only when it names the Episode and the feature set it is asked for and holds one entry per decision, and
// recomputes the rows itself otherwise. The files are never part of a generation, an Episode or any contract, and
// the stream's bytes are the same with or without them (the stream writer still reads and checks every Episode).
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import type { FeatureConfigs } from "./candidate-features.ts";
import { type Json, jcs } from "./canonical-json.ts";
import type { Engine } from "./engine.ts";
import { episodeHash } from "./episode.ts";
import { type AnyCandidateFeatures, configuredSetHash } from "./feature-build.ts";
import { decisionObservations, readEpisode } from "./episode-io.ts";
import { nodeSha256 } from "./player-channel.ts";
import type { SyntheticTeamHashes } from "./team-class.ts";
import { episodeFeatureRows } from "./training-loader.ts";

/** The file a process leaves the rows of one Episode in. */
const rowsFile = (dir: string, hash: string): string => join(dir, `${hash}.rows.json`);

/** One Episode stored on disk, as a game process wrote it. */
export interface RowsEpisode {
  hash: string;
  path: string;
}

/**
 * Recompute the rows of each stored Episode and leave them in `dir` (created if needed), one file per Episode.
 * An Episode that does not read back as the one its name addresses, or whose rows cannot be built, gets no file:
 * the stream writer recomputes it and gives the loader's own refusal. `stopped` is checked before each Episode.
 */
export async function writeFeatureRows(
  { engine, configs, known_synthetic }: { engine: Engine; configs: FeatureConfigs; known_synthetic: SyntheticTeamHashes },
  episodes: readonly RowsEpisode[],
  dir: string,
  stopped: () => boolean = () => false,
): Promise<number> {
  mkdirSync(dir, { recursive: true });
  const setHash = configuredSetHash(engine, configs);
  let written = 0;
  for (const { hash, path } of episodes) {
    if (stopped()) break;
    // The play-time sidecar of this Episode is already there (`writePlayRows`): nothing to build.
    if (existsSync(rowsFile(dir, hash))) continue;
    try {
      const episode = await readEpisode(JSON.parse(readFileSync(path, "utf8")), { known_synthetic });
      if ((await episodeHash(episode, nodeSha256)) !== hash) continue;
      const rows = await episodeFeatureRows(episode, hash, { engine, configs });
      writeFileSync(rowsFile(dir, hash), jcs({ episode_hash: hash, feature_set_hash: setHash, rows } as unknown as Json));
      written++;
    } catch {
      // No file: the stream writer recomputes this Episode and reports what is wrong with it, if anything.
    }
  }
  return written;
}

/**
 * The play-time sidecar (#1271): the rows the policy seat attached to its queries while the game was played, written as
 * the Episode's rows file. `played` maps a request number of the seat's channel to the rows built for it. The file is
 * written only when every decision of the Episode has rows, in the Episode's decision order; otherwise nothing is
 * written and the post-game pass (`writeFeatureRows`) or the stream writer builds them, so a game with a gap costs time
 * and never a byte. The rows are the ones the same builder makes from the same channel prefix, team and candidates, which
 * the stream writer would recompute (a test holds the file equal to the recomputation).
 */
export async function writePlayRows(
  { engine, configs }: { engine: Engine; configs: FeatureConfigs },
  episode: Parameters<typeof decisionObservations>[0],
  hash: string,
  played: ReadonlyMap<number, AnyCandidateFeatures>,
  dir: string,
): Promise<boolean> {
  const setHash = configuredSetHash(engine, configs);
  const rows: AnyCandidateFeatures[] = [];
  for (const observation of await decisionObservations(episode)) {
    const found = played.get(observation.identity.request_seq);
    if (found === undefined || found.feature_set_hash !== setHash) return false;
    rows.push(found);
  }
  mkdirSync(dir, { recursive: true });
  writeFileSync(rowsFile(dir, hash), jcs({ episode_hash: hash, feature_set_hash: setHash, rows } as unknown as Json));
  return true;
}

/**
 * The reader for `TrainingInput.features.precomputed`: the rows `writeFeatureRows` left for an Episode, or undefined
 * when there is no file, it is not for this Episode and this feature set, or it does not parse (then the rows are
 * recomputed, so a stale or damaged file can cost time but never change a byte).
 */
export function readFeatureRows(engine: Engine, configs: FeatureConfigs, dir: string): (hash: string) => AnyCandidateFeatures[] | undefined {
  const setHash = configuredSetHash(engine, configs);
  return (hash) => {
    try {
      const found = JSON.parse(readFileSync(rowsFile(dir, hash), "utf8")) as { episode_hash?: unknown; feature_set_hash?: unknown; rows?: unknown };
      if (found.episode_hash !== hash || found.feature_set_hash !== setHash || !Array.isArray(found.rows)) return undefined;
      const rows = found.rows as AnyCandidateFeatures[];
      return rows.every((row) => row.feature_set_hash === setHash) ? rows : undefined;
    } catch {
      return undefined;
    }
  };
}
