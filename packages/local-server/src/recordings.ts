import type { DropPlan } from "./harness.ts";

/** The format of the fixtures under fixtures/a2/. */
export const FIXTURE_FORMAT = "gen9championsvgc2026regmc";

export interface Recording {
  /** The fixture's file name under fixtures/a2/<format>/. */
  file: string;
  /** The bots' user names; fixed, so only the server's seed and avatars vary between recordings. */
  names: { sparring: string; challenger: string };
  /** Each bot's team: a file under fixtures/teams/<format>/. */
  teams: { sparring: string; challenger: string };
  /** Cut one bot's connection mid-battle so it reconnects (schema 3 frames and channels hold two connections for that seat). */
  drop?: DropPlan;
}

/**
 * Both pairings of the two synthetic teams, one battle each, then the same
 * with a real reconnect: one bot's connection is cut right after it sent its
 * choice for a request (the server re-sends the request with `|sentchoice|`),
 * and another's when a request arrives, before the bot answers (re-sent
 * without it). Which request that is depends on the battle, which the server
 * draws; `recordHarnessBattle` refuses a recording whose re-sent request is a
 * `wait` or whose `|sentchoice|` is not the planned one.
 */
export const RECORDINGS: readonly Recording[] = [
  {
    file: "bot-vs-bot-1.json",
    names: { sparring: "Areneon-bot-a2f1", challenger: "Areneon-ch-a2f1" },
    teams: { sparring: "legal-synthetic.txt", challenger: "legal-opponent.txt" },
  },
  {
    file: "bot-vs-bot-2.json",
    names: { sparring: "Areneon-bot-a2f2", challenger: "Areneon-ch-a2f2" },
    teams: { sparring: "legal-opponent.txt", challenger: "legal-synthetic.txt" },
  },
  {
    file: "bot-vs-bot-3.json",
    names: { sparring: "Areneon-bot-a2f3", challenger: "Areneon-ch-a2f3" },
    teams: { sparring: "legal-synthetic.txt", challenger: "legal-opponent.txt" },
    drop: { who: "challenger", request: 4, when: "after_choice" },
  },
  {
    file: "bot-vs-bot-4.json",
    names: { sparring: "Areneon-bot-a2f4", challenger: "Areneon-ch-a2f4" },
    teams: { sparring: "legal-opponent.txt", challenger: "legal-synthetic.txt" },
    drop: { who: "sparring", request: 3, when: "before_choice" },
  },
];

export function recordingProvenance(commit: string, drop?: DropPlan): string {
  const base =
    "synthetic: two Areneon bot clients (runA2Bot, shared actor rule) play one Bo1 with open team sheets declined " +
    `on the pinned Showdown server (${commit}) on 127.0.0.1, launched with the harness config (logchallenges on); ` +
    "teams are fixtures/teams synthetic CC0 pastes; the input log is the server's own, unmodified";
  if (drop === undefined) return base;
  const point = drop.when === "after_choice" ? "right after it sent its choice for" : "when it received";
  return (
    `${base}; the ${drop.who} bot's connection is closed by the recorder ${point} its request ${drop.request}, ` +
    "and the bot reconnects through its normal path (the server rejoins it and re-sends the request)"
  );
}
