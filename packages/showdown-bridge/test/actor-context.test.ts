import { describe, expect, expectTypeOf, it } from "vitest";
import type { PokemonSet } from "../src/engine.ts";
import { guardActor } from "../src/evaluation-harness.ts";
import { firstLegal, GOLDEN_FORMAT, GOLDEN_SEED, syntheticTeam } from "../src/golden.ts";
import { type Actor, type ActorContext, loadEngine, type Player, runBattle, teamHash } from "../src/index.ts";

const engine = loadEngine();

const frozen = (value: unknown): boolean => typeof value !== "object" || value === null || (Object.isFrozen(value) && Object.values(value).every(frozen));

describe("ActorContext.team (ADR-018 section 2)", () => {
  const p1 = syntheticTeam();
  const p2 = syntheticTeam().reverse();

  async function play(wrap: (actor: Actor) => Actor = (actor) => actor) {
    const seen: Record<Player, ActorContext[]> = { p1: [], p2: [] };
    const recorder = (player: Player): Actor =>
      wrap((request, context) => {
        seen[player].push(context as ActorContext);
        return firstLegal(request, context);
      });
    const record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, teams: { p1, p2 }, actors: { p1: recorder("p1"), p2: recorder("p2") }, maxTurns: 3 });
    return { seen, record };
  }

  it("is each seat's own set array as brought, frozen, and never the other seat's", async () => {
    const before = { p1: teamHash(p1), p2: teamHash(p2) };
    const { seen, record } = await play();
    for (const player of ["p1", "p2"] as const) {
      expect(seen[player].length).toBeGreaterThan(1);
      for (const context of seen[player]) {
        expect(context.player).toBe(player);
        expect(context.team).toEqual(player === "p1" ? p1 : p2);
        expect(Object.keys(context).sort()).toEqual(["channel", "player", "team"]);
        expect(frozen(context.team)).toBe(true);
        expect(teamHash(context.team as PokemonSet[])).toBe(record.team_hashes[player]);
      }
    }
    expect(seen.p1[0]?.team).not.toEqual(seen.p2[0]?.team);
    // a copy: the caller's sets are neither frozen nor changed
    expect(seen.p1[0]?.team).not.toBe(p1);
    expect(Object.isFrozen(p1)).toBe(false);
    expect({ p1: teamHash(p1), p2: teamHash(p2) }).toEqual(before);
  });

  it("cannot be changed by an actor", async () => {
    const { seen } = await play();
    const team = seen.p1[0]?.team as PokemonSet[];
    expect(() => team.push(team[0] as PokemonSet)).toThrow(TypeError);
    expect(() => {
      (team[0] as PokemonSet).name = "changed";
    }).toThrow(TypeError);
    expect(() => {
      (team[0] as PokemonSet).evs.hp = 0;
    }).toThrow(TypeError);
  });

  it("reaches an actor under guardActor unchanged", async () => {
    const { seen } = await play((actor) => guardActor(actor, { role: "scripted", latency_budget: { preview_ms: 5_000, battle_ms: 5_000 } }));
    expect(seen.p1[0]?.team).toEqual(p1);
  });

  it("holds nothing of the opponent's team, the omniscient log, the hidden verdicts or the seed (a type test)", () => {
    expectTypeOf<ActorContext>().toEqualTypeOf<{ player: Player; channel: readonly string[]; team?: readonly Readonly<PokemonSet>[] }>();
    expectTypeOf<keyof ActorContext>().toEqualTypeOf<"player" | "channel" | "team">();
    const neverRun = (): void => {
      // @ts-expect-error: the opponent's team is not a part of the context
      const a: ActorContext = { player: "p1", channel: [], opponent_team: p2 };
      // @ts-expect-error: nor the omniscient log
      const b: ActorContext = { player: "p1", channel: [], omniscient_log: [] };
      // @ts-expect-error: nor the hidden verdicts
      const c: ActorContext = { player: "p1", channel: [], hidden_verdicts: [] };
      // @ts-expect-error: nor the battle seed
      const d: ActorContext = { player: "p1", channel: [], seed: GOLDEN_SEED };
      void [a, b, c, d];
    };
    expect(neverRun).toBeTypeOf("function");
  });
});
