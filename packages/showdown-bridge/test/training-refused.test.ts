// ADR-010 §1: a refused pending choice is a transition like any other in the
// player's own Episode. The learner gives it no reward of its own: the game
// score arrives only at the end (ml/areneon_ml/learner/ppo.py terminal_rewards).
import { describe, expect, it } from "vitest";
import { type ChoiceCase, loadCorpus } from "../src/choice-corpus.ts";
import { firstLegal, GOLDEN_FORMAT } from "../src/golden.ts";
import { type Actor, type BattleRecord, episodeFromRecord, loadEngine, loadTrainingSet, runBattle, stampTeam } from "../src/index.ts";
import { teamHash } from "../src/team-class.ts";

const engine = loadEngine();
const POLICY = `policy:${"d".repeat(64)}`;

describe("a pending switch refused at submission, in the training stream", () => {
  const c = loadCorpus()["maybe-trapped.json"] as ChoiceCase;
  const known = new Set([teamHash(c.teams.p1), teamHash(c.teams.p2)]);
  const scripted = (player: "p1" | "p2", after: Actor): Actor => {
    const queue = c.log.filter((e) => e.player === player).map((e) => e.choice);
    return (request) => queue.shift() ?? after(request);
  };
  const play = async (): Promise<BattleRecord> => {
    let tried = false;
    const p1 = scripted("p1", (request) => {
      const pending = request.legal?.pending ?? [];
      if (tried || pending.length === 0) return firstLegal(request);
      tried = true;
      return pending[0] as string;
    });
    return runBattle({ engine, formatId: GOLDEN_FORMAT, seed: c.seed, teams: c.teams, actors: { p1, p2: scripted("p2", firstLegal) } });
  };

  it("keeps the refused attempt and the new request that follows as two examples of one Episode, both with the game's outcome", async () => {
    const record = await play();
    const refused = record.choices.find((entry) => entry.refused !== undefined) as BattleRecord["choices"][number];
    const episode = await episodeFromRecord(record, "p1", {
      team: stampTeam(c.teams.p1 as never, known),
      opponent_team: stampTeam(c.teams.p2 as never, known),
      known_synthetic: known,
      behaviour_policy_id: POLICY,
      collection: { kind: "league", run_id: "refused", seed: record.seed, opponent_policy_id: POLICY },
    });
    const { examples, set } = await loadTrainingSet({ episodes: [episode], evaluation_manifests: [], known_synthetic: known, behaviour_policy_id: POLICY });
    const seqs = examples.map((example) => example.observation.identity.request_seq);
    expect(seqs).toContain(refused.request_seq);
    expect(seqs).toContain(refused.request_seq + 1);
    const bySeq = (seq: number) => examples.find((example) => example.observation.identity.request_seq === seq) as (typeof examples)[number];
    const attempt = bySeq(refused.request_seq);
    const retry = bySeq(refused.request_seq + 1);
    expect(attempt.choice).toBe(refused.choice);
    expect(attempt.candidates?.find((candidate) => candidate.choice === attempt.choice)?.pending).toBe(true);
    expect(retry.choice).not.toBe(refused.choice);
    const outcome = record.winner === null ? "tie" : record.winner === "p1" ? "win" : "loss";
    expect([attempt.outcome, retry.outcome]).toEqual([outcome, outcome]);
    expect(set.learner_ready).toBe(true);
  });
});
