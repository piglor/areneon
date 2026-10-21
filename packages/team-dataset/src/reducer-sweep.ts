// The reducer sweep's play side (#919): seeded battles on teams from a verified
// generated dataset, played by actor mixes that include the sampler the
// generation loop's random-initialised pair is (uniform over ALL candidates,
// pending ones included). Each battle is written as one JSON line holding both
// players' own channels, their teams and, per decision, the channel length at
// the request and the candidates `enumerateCandidates` returned, so the Python
// side (`scripts/reducer_sweep.py`) can rebuild every query the policy worker
// would receive and feed it to the reducer and the encoder. Nothing here is a
// game rule: the engine judges every choice.
import {
  type Actor,
  type ActorContext,
  actorRule,
  type Engine,
  enumerateCandidates,
  type GenerationTeamPool,
  leagueGameStreams,
  type PokemonSet,
  type Player,
  pendingFirstActor,
  randomLegal,
  randomLegalActor,
  runBattle,
} from "@areneon/showdown-bridge";

export const SWEEP_MIXES = ["random-legal", "pending-first", "rule-vs-random", "uniform-all", "uniform-vs-random"] as const;
export type SweepMix = (typeof SWEEP_MIXES)[number];

export interface SweepDecision {
  /** The player's channel length at this request (the request line is the last one kept). */
  n: number;
  kind: "preview" | "battle";
  candidates: { choice: string; pending: boolean }[];
}

export interface SweepBattle {
  index: number;
  mix: SweepMix;
  seed: string;
  teams: Record<Player, PokemonSet[]>;
  actors: Record<Player, string>;
  turns: number;
  winner: Player | null;
  channels: Record<Player, string[]>;
  decisions: Record<Player, SweepDecision[]>;
  /** Set when the battle itself failed (runner abort, enumerator throw): kind and message, no more. */
  failure?: { kind: string; message: string };
}

/** The sampler of the loop's frozen pair, without a network: uniform over every candidate, pending included. */
export function uniformAllActor(seed: number): Actor {
  let state = seed;
  return (request) => {
    const all = enumerateCandidates(request).candidates.map((candidate) => candidate.choice);
    const pick = randomLegal({ choices: all, pending: [] }, state);
    state = pick.state;
    return pick.choice;
  };
}

/** The actor rule over the candidates a policy would be offered: the first non-pending one, else the first pending one. */
const ruleActor: Actor = (request) => {
  const candidates = enumerateCandidates(request).candidates;
  return actorRule({
    choices: candidates.filter((candidate) => !candidate.pending).map((candidate) => candidate.choice),
    pending: candidates.filter((candidate) => candidate.pending).map((candidate) => candidate.choice),
  });
};

function actorsOf(mix: SweepMix, seeds: Record<Player, number>): Record<Player, { name: string; actor: Actor }> {
  const named = (name: string, actor: Actor) => ({ name, actor });
  switch (mix) {
    case "random-legal":
      return { p1: named("random-legal", randomLegalActor(seeds.p1)), p2: named("random-legal", randomLegalActor(seeds.p2)) };
    case "pending-first":
      return { p1: named("pending-first", pendingFirstActor(seeds.p1)), p2: named("pending-first", pendingFirstActor(seeds.p2)) };
    case "rule-vs-random":
      return { p1: named("actor-rule", ruleActor), p2: named("random-legal", randomLegalActor(seeds.p2)) };
    case "uniform-all":
      return { p1: named("uniform-all", uniformAllActor(seeds.p1)), p2: named("uniform-all", uniformAllActor(seeds.p2)) };
    case "uniform-vs-random":
      return { p1: named("uniform-all", uniformAllActor(seeds.p1)), p2: named("random-legal", randomLegalActor(seeds.p2)) };
  }
}

/** Play battle `index` of a mix; the seed, the teams and the actor seeds follow from (run key, mix, index) alone. */
export async function playSweepBattle(
  engine: Engine,
  formatId: string,
  pool: GenerationTeamPool,
  options: { run_key: string; mix: SweepMix; index: number; max_turns: number },
): Promise<SweepBattle> {
  const { run_key, mix, index, max_turns } = options;
  const streams = leagueGameStreams(`${run_key}/${mix}`, 1, index);
  const teams: Record<Player, PokemonSet[]> = {
    p1: pool.teams[streams.team_draws.p1 % pool.teams.length] as PokemonSet[],
    p2: pool.teams[streams.team_draws.p2 % pool.teams.length] as PokemonSet[],
  };
  const actorSeeds = { p1: streams.baseline_seed, p2: (streams.baseline_seed ^ 0x9e3779b9) >>> 0 };
  const chosen = actorsOf(mix, actorSeeds);
  const decisions: Record<Player, SweepDecision[]> = { p1: [], p2: [] };
  const recording =
    (player: Player, inner: Actor): Actor =>
    async (request, context) => {
      const candidates = enumerateCandidates(request).candidates.map(({ choice, pending }) => ({ choice, pending }));
      decisions[player].push({ n: (context as ActorContext).channel.length, kind: request.phase === "preview" ? "preview" : "battle", candidates });
      return inner(request, context);
    };
  const base = {
    index,
    mix,
    seed: streams.seed,
    teams,
    actors: { p1: chosen.p1.name, p2: chosen.p2.name },
    decisions,
  };
  try {
    const record = await runBattle({
      engine,
      formatId,
      seed: streams.seed,
      teams,
      actors: { p1: recording("p1", chosen.p1.actor), p2: recording("p2", chosen.p2.actor) },
      maxTurns: max_turns,
    });
    return { ...base, turns: record.turns, winner: record.winner, channels: record.channels };
  } catch (error) {
    // The battle did not finish: the failure is the finding; the channels are not kept.
    return {
      ...base,
      turns: 0,
      winner: null,
      channels: { p1: [], p2: [] },
      failure: { kind: (error as Error).name, message: (error as Error).message.slice(0, 400) },
    };
  }
}

export interface SweepArgs {
  config: string;
  out: string;
  mix: SweepMix;
  battles: number;
  start: number;
  run_key: string;
}

export class SweepUsageError extends Error {
  override name = "SweepUsageError";
}

const FLAGS = ["--config", "--out", "--mix", "--battles", "--start", "--run-key"];

/** Strict `--flag value` parsing: an unknown flag, a missing or repeated value, a bad mix or a count that is not a whole number (battles at least 1) is refused, so a sweep cannot play nothing and pass. */
export function parseSweepArgs(argv: readonly string[]): SweepArgs {
  const values = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 2) {
    const flag = argv[i] as string;
    if (!FLAGS.includes(flag)) throw new SweepUsageError(`unknown argument ${flag}`);
    const value = argv[i + 1];
    if (value === undefined || value === "" || value.startsWith("--")) throw new SweepUsageError(`${flag} needs a value`);
    if (values.has(flag)) throw new SweepUsageError(`${flag} is given twice`);
    values.set(flag, value);
  }
  const need = (flag: string): string => {
    const value = values.get(flag);
    if (value === undefined) throw new SweepUsageError(`${flag} is required`);
    return value;
  };
  const whole = (flag: string, text: string, least: number): number => {
    if (!/^\d+$/.test(text) || Number(text) < least || !Number.isSafeInteger(Number(text))) {
      throw new SweepUsageError(`${flag} is a whole number of at least ${least}`);
    }
    return Number(text);
  };
  const mix = need("--mix");
  if (!(SWEEP_MIXES as readonly string[]).includes(mix)) throw new SweepUsageError(`--mix is one of ${SWEEP_MIXES.join(", ")}`);
  return {
    config: need("--config"),
    out: need("--out"),
    mix: mix as SweepMix,
    battles: whole("--battles", need("--battles"), 1),
    start: whole("--start", values.get("--start") ?? "0", 0),
    run_key: values.get("--run-key") ?? "sweep-1",
  };
}
