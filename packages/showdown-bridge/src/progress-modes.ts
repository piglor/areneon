// The play modes of the progress evaluation (ADR-023 section 1, #1044): `argmax` and `sample` (temperature 1, no truncation) are
// always played; an EXTRA mode is a sample at a declared temperature and/or with a probability threshold. One closed grammar names
// it on the command line and one fixed rule names its matches, so no mode string or match id is free text.
//
//   mode  := "argmax" | "sample" | "sample" [":t" ppm] [":th" ppm]       (ppm: 1 to 999999, no leading zero)
//   match := <opponent>-sample [-t<ppm>] [-th<ppm>]
//
// The explicit default temperature (1,000,000) is not a mode: omit the part (the policy id never writes it).

import type { PlaySettings } from "./policy-protocol.ts";

export class ProgressModeError extends Error {
  override name = "ProgressModeError";
}

const MODE = /^sample(?::t([1-9]\d{0,5}))?(?::th([1-9]\d{0,5}))?$/;
const MAX_PPM = 999_999;

/** A sample mode's declared settings; both absent is the default `sample`. */
export type ExtraPlay = PlaySettings;

/** The play settings a mode string names, or null for `argmax` and plain `sample` (the modes always played). */
export function parseExtraMode(text: string): ExtraPlay | null {
  if (text === "argmax" || text === "sample") return null;
  const found = MODE.exec(text);
  if (found === null) throw new ProgressModeError(`unknown play mode ${JSON.stringify(text.slice(0, 40))}: argmax, sample, sample:t<ppm>, sample:th<ppm> or sample:t<ppm>:th<ppm> (ppm 1 to ${MAX_PPM})`);
  const temperature = found[1] === undefined ? undefined : Number(found[1]);
  const threshold = found[2] === undefined ? undefined : Number(found[2]);
  if ((temperature !== undefined && temperature > MAX_PPM) || (threshold !== undefined && threshold > MAX_PPM)) throw new ProgressModeError(`a play mode's ppm is 1 to ${MAX_PPM}`);
  return { ...(temperature === undefined ? {} : { temperature_ppm: temperature }), ...(threshold === undefined ? {} : { truncation: { kind: "threshold" as const, min_ppm: threshold } }) };
}

/** The `--modes` list: the extra modes it names, each once, in the order given (argmax and plain sample are no-ops). */
export function parseModes(texts: readonly string[]): ExtraPlay[] {
  const seen = new Set<string>();
  const extras: ExtraPlay[] = [];
  for (const text of texts) {
    if (seen.has(text)) throw new ProgressModeError(`play mode ${JSON.stringify(text.slice(0, 40))} is listed twice`);
    seen.add(text);
    const play = parseExtraMode(text);
    if (play !== null) extras.push(play);
  }
  return extras;
}

/** The mode string of settings (the inverse of `parseExtraMode`): `sample:t500000:th30000`. */
export const modeName = (play: ExtraPlay): string => `sample${play.temperature_ppm === undefined ? "" : `:t${play.temperature_ppm}`}${play.truncation === undefined ? "" : `:th${play.truncation.min_ppm}`}`;

/** The match-id suffix of settings: `-t500000-th30000`, empty for the default sample. */
export const matchSuffix = (play: ExtraPlay): string => `${play.temperature_ppm === undefined ? "" : `-t${play.temperature_ppm}`}${play.truncation === undefined ? "" : `-th${play.truncation.min_ppm}`}`;
