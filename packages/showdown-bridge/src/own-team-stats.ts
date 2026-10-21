// The engine-dependent half of the own-team check (ADR-008 §5, #893).
//
// `buildObservation` (engine-free) matches the player's team against the Team
// Preview request by species, item, ability, moves and nickname. A mirror team
// that differs only in Stat Points / EVs, IVs, nature or level passes that, and
// its spread would enter the observation. Telling them apart needs the pinned
// engine's own stat calculation, so the check lives here, in the Node adapter
// layer: the engine builds a battle from the team the observation will record,
// and the stats and level of the request it sends that player must equal those
// of the player's own Team Preview request, Pokémon for Pokémon. No formula is
// written in this module.
//
// Hidden Power: the request names it `hiddenpower<type>`, so a bare "Hidden
// Power" set would be refused by `buildObservation`'s move match. The pinned
// engine's validator does not allow the move in any supported format (it
// "does not exist in Gen 9" there), so no legal team reaches this case and the
// refusal stays, fail closed; `own-team-stats.test.ts` pins that fact so a pin
// bump that allows the move fails there and the case is decided then.

import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { type Engine, loadEngine } from "./engine.ts";
import { ObservationError, type PlayerChannel } from "./observation.ts";

/**
 * The team does not give the stats of the player's own Team Preview request.
 * The message is fixed apart from the position (an index into the team): it
 * never echoes a set, a stat or a request.
 */
export class OwnTeamStatsError extends ObservationError {
  override name = "OwnTeamStatsError";
  readonly position: number | undefined;
  constructor(message: string, position?: number) {
    super(message);
    this.position = position;
  }
}

interface RequestMon {
  details: string;
  condition: string;
  stats: { [stat: string]: number };
}

interface PreviewRequest {
  teamPreview?: true;
  side: { pokemon: RequestMon[] };
}

const REQUEST = "|request|";
const SEED = "sodium,00000000000000000000000000000000";

let pinned: Engine | undefined;
/** The pinned engine, loaded once: `loadEngine` checks the checkout with git on every call. */
export function pinnedEngine(): Engine {
  pinned ??= loadEngine();
  return pinned;
}

/** What a request shows of one Pokémon's numbers: maximum HP, level and the five stats. */
function numbers(mon: RequestMon): string {
  const level = /, L(\d+)/.exec(mon.details)?.[1] ?? "100";
  const maxhp = (mon.condition.split(" ")[0] as string).split("/")[1];
  const stats = Object.keys(mon.stats).sort().map((stat) => `${stat}=${mon.stats[stat]}`);
  return [`L${level}`, `hp=${maxhp}`, ...stats].join(",");
}

/**
 * Check that the engine, given `team` in `formatid`, sends `channel.player` a
 * Team Preview request with the stats of the channel's own first request.
 * Throws OwnTeamStatsError (never echoing the team) on any difference, on a
 * channel that does not start at Team Preview, on a team of another size and
 * on a team the engine cannot start a battle with.
 */
export function assertOwnTeamStats(
  engine: Engine,
  formatid: string,
  team: PrivateTeamSpec,
  channel: PlayerChannel,
): void {
  const first = channel.lines.find((line) => line.startsWith(REQUEST));
  const shown = first === undefined ? undefined : (JSON.parse(first.slice(REQUEST.length)) as PreviewRequest);
  if (shown?.teamPreview !== true) {
    throw new OwnTeamStatsError("the channel does not start at Team Preview, so the team's stats cannot be checked");
  }
  let computed: PreviewRequest | undefined;
  try {
    const packed = engine.sim.Teams.pack(structuredClone(team.team));
    const battle = new engine.sim.Battle({ formatid, seed: SEED });
    // The player's side is checked; the other gets the same team, which the
    // battle needs to ask for Team Preview and which nothing here reads.
    battle.setPlayer("p1", { name: "p1", team: packed });
    battle.setPlayer("p2", { name: "p2", team: packed });
    computed = battle.sides.find((side) => side.id === channel.player)?.activeRequest as PreviewRequest | undefined;
  } catch {
    throw new OwnTeamStatsError("the engine cannot start a battle with the team, so its stats cannot be checked");
  }
  if (computed?.teamPreview !== true) {
    throw new OwnTeamStatsError("the engine sent no Team Preview request for the team, so its stats cannot be checked");
  }
  if (computed.side.pokemon.length !== shown.side.pokemon.length) {
    throw new OwnTeamStatsError("the team and the player's Team Preview request differ in size");
  }
  shown.side.pokemon.forEach((mon, position) => {
    if (numbers(mon) !== numbers(computed.side.pokemon[position] as RequestMon)) {
      throw new OwnTeamStatsError(
        `the stats of the team's Pokémon at position ${position} are not those of the player's Team Preview request`,
        position,
      );
    }
  });
}
