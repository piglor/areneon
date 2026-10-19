/** The format of the fixtures under fixtures/a2/. */
export const FIXTURE_FORMAT = "gen9championsvgc2026regmc";

export interface Recording {
  /** The fixture's file name under fixtures/a2/<format>/. */
  file: string;
  /** The bots' user names; fixed, so only the server's seed and avatars vary between recordings. */
  names: { sparring: string; challenger: string };
  /** Each bot's team: a file under fixtures/teams/<format>/. */
  teams: { sparring: string; challenger: string };
}

/** Both pairings of the two synthetic teams, one battle each. */
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
];

export function recordingProvenance(commit: string): string {
  return (
    "synthetic: two Areneon bot clients (runA2Bot, shared actor rule) play one Bo1 with open team sheets declined " +
    `on the pinned Showdown server (${commit}) on 127.0.0.1, launched with the harness config (logchallenges on); ` +
    "teams are fixtures/teams synthetic CC0 pastes; the input log is the server's own, unmodified"
  );
}
