import { beforeAll, describe, expect, it } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { MechanicsProbeError } from "../src/mechanics-probe.ts";
import type { ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import type { PublicMon } from "@areneon/public-state";
import { capture, type Captured, corpus, newActor, stateOf, stubProbe } from "./damage-aware-support.ts";

// public-main-004 stopped at game 604 with `baseline_failure: the engine does not take that status`: a Pokemon
// the player SAW paralysed was declared against an assumed ability (Limber) that blocks paralysis, so the probe
// refused the declared state and the refusal escaped the actor. The assumption contradicted a public fact; and
// whatever else the probe may refuse, it must never end a game.

let captured: Captured[] = [];
beforeAll(async () => {
  captured = [...(await capture(corpus[0] as never)), ...(await capture(corpus[1] as never)), ...(await capture(corpus[3] as never))];
}, 120_000);

const STATUS = "the engine does not take that status";
const abilityOf = (paste: string): string => (/Ability: (.+?)\s*\n/.exec(paste) as RegExpExecArray)[1] as string;

/** A move request in which an opponent Pokemon is on the field, with a public status line appended to the player's own channel. */
function statused(): { c: Captured; foe: PublicMon } {
  for (const c of captured) {
    if (c.request.phase !== "move" || (c.request.legal?.choices.length ?? 0) < 2) continue;
    const state = stateOf(c);
    for (const letter of Object.keys(state.opponent.slots).sort()) {
      const mon = state.opponent.roster[state.opponent.slots[letter] as number] as PublicMon;
      if (mon.fainted || mon.ability !== null || mon.hp_num === null) continue;
      const foeSide = c.context.player === "p1" ? "p2" : "p1";
      const channel = [...c.context.channel, `|-status|${foeSide}${letter}: ${mon.name}|par`];
      return { c: { ...c, context: { ...c.context, channel } }, foe: mon };
    }
  }
  throw new Error("no request with an opponent on the field");
}

describe("an assumed ability that contradicts a public status", () => {
  it("is ruled out: the next ability in the record's order is assumed, with no refusal and no fallback", () => {
    const { c } = statused();
    const seen: string[] = [];
    let banned: string | null = null;
    const inner = stubProbe({ values: { tackle: 1 } });
    const probe = (context: Parameters<typeof inner>[0], query: ParsedMechanicsQuery) => {
      if (query.query.kind === "damage") {
        for (const side of [query.query.attacker, query.query.defender]) {
          if (side.status === null) continue;
          const ability = abilityOf(side.set);
          seen.push(ability);
          banned ??= ability; // the first ability assumed is the one the stand-in engine will not let hold the status
          if (ability === banned) throw new MechanicsProbeError(STATUS);
        }
      }
      return inner(context, query);
    };
    const actor = newActor({ probe });
    const choice = actor(c.request, c.context) as string;
    expect(c.request.legal?.choices).toContain(choice);
    const counts = actor.report().counts;
    expect(seen.length).toBeGreaterThan(1);
    expect(new Set(seen).size).toBeGreaterThan(1); // a second ability was tried
    expect(counts.probe_refusals).toBe(0);
    expect(counts.assumed_set_fallbacks).toBe(0);
    expect(counts.scored_decisions).toBe(1);
  });

  it("is answered by the actor-rule candidate when every ability blocks the status", () => {
    const { c } = statused();
    const probe = (_context: unknown, query: ParsedMechanicsQuery) => {
      if (query.query.kind === "damage" && (query.query.attacker.status !== null || query.query.defender.status !== null)) throw new MechanicsProbeError(STATUS);
      return stubProbe({ values: { tackle: 1 } })(_context as never, query);
    };
    const actor = newActor({ probe: probe as never });
    // all the species' abilities are refused: either no assumed set is left (assumed_set_fallbacks) or the refusal is counted; the request is answered, never thrown
    expect(actor(c.request, c.context)).toBe(actorRule(c.request.legal as never));
    const counts = actor.report().counts;
    expect(counts.probe_refusals + counts.assumed_set_fallbacks).toBe(1);
  });

  it("is not looked for when the ability was revealed or the status is another refusal", () => {
    const { c } = statused();
    const probe = (context: unknown, query: ParsedMechanicsQuery) => {
      if (query.query.kind === "damage" && (query.query.attacker.status !== null || query.query.defender.status !== null)) throw new MechanicsProbeError("hp is above the Pokemon's maximum HP");
      return stubProbe({ values: { tackle: 1 } })(context as never, query);
    };
    const actor = newActor({ probe: probe as never });
    expect(actor(c.request, c.context)).toBe(actorRule(c.request.legal as never));
    expect(actor.report().counts.probe_refusals).toBe(1);
  });
});

describe("a probe refusal", () => {
  it("is counted once per request, answered by the actor-rule candidate, and does not leave the actor", () => {
    const moves = captured.filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2).slice(0, 4);
    const actor = newActor({ probe: stubProbe({ throwOn: { damage: new MechanicsProbeError("any declared state the engine refuses") } }) });
    for (const c of moves) expect(actor(c.request, c.context)).toBe(actorRule(c.request.legal as never));
    const counts = actor.report().counts;
    expect(counts.probe_refusals).toBeGreaterThan(0);
    expect(counts.probe_refusals).toBeLessThanOrEqual(counts.queries);
    expect(counts.scored_decisions).toBe(0);
  });

  it("counts every probe run it makes, refused or not", () => {
    const c = captured.find((x) => x.request.phase === "move" && (x.request.legal?.choices.length ?? 0) >= 2) as Captured;
    let runs = 0;
    const refusing = (): never => {
      runs++;
      throw new MechanicsProbeError("refused");
    };
    const actor = newActor({ probe: refusing as never });
    actor(c.request, c.context);
    const first = runs;
    expect(first).toBeGreaterThan(0);
    expect(actor.report().counts.probe_runs).toBe(first);
  });
});
