// What the Team Preview best-response evaluation and the Team Preview solver (`preview-solve.ts`) share: reading the seat's
// own Team Preview channel, and starting a hypothetical battle (`hypothetical-preview.ts`) for each full-team draw of the belief
// (ADR-029 section 3). It reads only the seat's own view: the channel to its Team Preview request, its own team, and the draws of
// the belief port, which are drawn from the reader's opponent half and nothing else.

import { interpretChannel } from "@areneon/public-state";
import type { PublicFlag, PublicSide } from "@areneon/public-state";
import type { WorldSetLike } from "./damage-aware-assumed-set.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { resolveFormat } from "./formats.ts";
import { type HypotheticalPreview, startHypothetical } from "./hypothetical-preview.ts";
import type { Player } from "./runner.ts";
import { validateTeam } from "./teams.ts";
import type { DataClass } from "./team-class.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";
import { mapWorld } from "./world-sampler.ts";

/** The structural port of the full-team opponent draw (the composition root joins the opponent model; the bridge does not import it). */
export type ResponderTeamDraw = { readonly kind: "team"; readonly team: { readonly sets: readonly WorldSetLike[] }; readonly redraws: number } | { readonly kind: "refused"; readonly code: string };
export type ResponderTeamSource = { readonly kind: "refused"; readonly code: string } | { readonly kind: "source"; readonly source: { draw(stream: Uint8Array, index: number): ResponderTeamDraw } };

const REQUEST_PREFIX = "|request|";

/** What a seat's own Team Preview channel gives the belief: the reader's opponent half, the flags and the request body. */
export function readPreviewChannel(seat: Player, channel: readonly string[]): { opponent: PublicSide; flags: readonly PublicFlag[]; request: string } {
  const run = interpretChannel(seat, channel);
  const frame = run.frames.at(-1);
  const request = channel.findLast((line) => line.startsWith(REQUEST_PREFIX));
  if (frame === undefined || request === undefined) throw new Error("the Team Preview channel holds no line or no request");
  return { opponent: frame.state.opponent, flags: run.flags, request: request.slice(REQUEST_PREFIX.length) };
}

/**
 * The one error that says a sampled team cannot be started as a hypothetical battle (counted): the validator refuses it as a whole.
 * Anything else is a defect of the run and propagates, a `HypotheticalError` for our own request or team that does not match,
 * or for a format with no Team Preview, included.
 */
const UNSTARTABLE = new Set(["InvalidTeamError"]);

export interface PreviewWorlds {
  readonly worlds: { index: number; preview: HypotheticalPreview }[];
  /** Draws the belief refused (coded). */
  readonly refused: number;
  /** The same refusals by the belief's code. */
  readonly refusal_codes: Readonly<Record<string, number>>;
  /** Draws that mapped to a team the validator refuses as a whole. */
  readonly unstarted: number;
}

export interface StartWorldsInput {
  engine: Engine;
  vocabulary: TeamVocabulary;
  format_id: string;
  source: { draw(stream: Uint8Array, index: number): ResponderTeamDraw };
  count: number;
  stream: Uint8Array;
  seat: Player;
  own_team: readonly Readonly<PokemonSet>[];
  channel_class: DataClass;
  /** The body of our real Team Preview request. */
  own_request: string;
}

/** Draw `count` full teams by index and start a hypothetical battle for each that maps to a startable team. */
export function startPreviewWorlds(input: StartWorldsInput): PreviewWorlds {
  const { engine, vocabulary, format_id, source, count, stream, seat, own_team, channel_class, own_request } = input;
  const format = resolveFormat(engine, format_id);
  const worlds: { index: number; preview: HypotheticalPreview }[] = [];
  let refused = 0;
  const refusal_codes: Record<string, number> = {};
  let unstarted = 0;
  for (let index = 0; index < count; index += 1) {
    const drawn = source.draw(stream, index);
    if (drawn.kind === "refused") {
      refused += 1;
      refusal_codes[drawn.code] = (refusal_codes[drawn.code] ?? 0) + 1;
      continue;
    }
    const world = { slots: drawn.team.sets.map((set, slot) => ({ index: slot, brought: true, set })) };
    try {
      // The sets a battle plays are the validator's normalised copy (a set with no nickname is named by its species, as in the pool's teams
      // a real game plays): the worker's reader checks the engine's request against those names. A team the validator refuses stays refused.
      const sets = mapWorld({ engine, vocabulary }, format_id, world).map((member) => member.set);
      // Called for its in-place normalisation (nicknames), not its verdict: startHypothetical validates the team again and refuses it.
      validateTeam(engine, format, sets);
      worlds.push({ index, preview: startHypothetical({ engine, format_id, seat, own_team, world_team: sets, stream, world: index, data_class: channel_class, own_request }) });
    } catch (error) {
      if (!UNSTARTABLE.has((error as Error).name)) throw error;
      unstarted += 1;
    }
  }
  return { worlds, refused, refusal_codes, unstarted };
}
