// The candidate count bound N_max (ADR-010 section 3): `TrainingExample.candidates`
// carries `maxItems` N_max, derived from the enumerator and the pinned engine's
// data, not from the corpus. For every supported format this test reads M (moves
// per set), k (picked team size), n (team size) and the event flags the engine can
// set from the pinned engine, builds the worst-case request of that shape,
// requires the enumerator to return exactly the formula's count after its joint
// filters, and requires that to be within the cap written into the schema. A pin
// bump that raises the bound fails here before training.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import trainingExampleSchema from "@areneon/schema/contracts/v1/training-example.schema.json" with { type: "json" };
import { describe, expect, it } from "vitest";
import { GOLDEN_DIR, type RequestFixture, syntheticTeam } from "../src/golden.ts";
import { ACTIVE_SLOTS } from "../src/choices.ts";
import { decodeRequest, enumerateCandidates, type JsonObject, loadEngine, loadRulesSnapshot } from "../src/index.ts";

/** Doubles: a move names at most 3 target locations (`targetsFor`): both foes and the ally, or the ally and self. */
const TARGETS = 3;
/** The Mega flags share one kind (one Mega per side per choice); Terastallization is another. */
type Kind = "mega" | "tera";

interface Probe {
  dex: { items: { all(): { megaStone?: unknown }[] } };
  actions: { canMegaEvoX?: unknown; canMegaEvoY?: unknown; canTerastallize(pokemon: object): unknown };
  sides: { pokemon: object[] }[];
}

interface Bound {
  format: string;
  /** Team Preview: ordered picks of k from n. */
  preview: { n: number; k: number; count: number };
  /** One battle slot: moves x targets x (1 + event flags) + bench switches. */
  battle: { moves: number; bench: number; flags: Record<string, Kind>; slot: number; upper: number; exact: number };
}

// Edits reach deep into engine JSON; typing every path would bury what each builder changes.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Loose = { [key: string]: any };

const permutations = (n: number, k: number): number => (k === 0 ? 1 : n * permutations(n - 1, k - 1));
const powerOfTwoAbove = (n: number): number => 2 ** Math.ceil(Math.log2(n));
const golden = (file: string): JsonObject => structuredClone((JSON.parse(readFileSync(join(GOLDEN_DIR, file), "utf8")) as RequestFixture).request);

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);

/** What the pinned engine says about a supported format. */
function boundOf(formatId: string): Bound {
  const format = engine.sim.Dex.formats.get(formatId);
  const rules = engine.sim.Dex.formats.getRuleTable(format) as ReturnType<typeof engine.sim.Dex.formats.getRuleTable> & { maxMoveCount: number };
  const k = rules.pickedTeamSize ?? rules.maxTeamSize;
  const n = rules.maxTeamSize;
  const moves = rules.maxMoveCount;
  const bench = k - ACTIVE_SLOTS;

  const battle = new engine.sim.Battle({ formatid: formatId, seed: "sodium,0000000000000001" }) as unknown as Probe & ReturnType<typeof engine.sim.Battle.prototype.toJSON>;
  const team = engine.sim.Teams.pack(syntheticTeam());
  (battle as unknown as { setPlayer(slot: string, o: object): void }).setPlayer("p1", { name: "p1", team });
  (battle as unknown as { setPlayer(slot: string, o: object): void }).setPlayer("p2", { name: "p2", team });
  const probe = battle as unknown as Probe;
  const flags: Record<string, Kind> = {};
  if (probe.dex.items.all().some((item) => item.megaStone !== undefined)) flags.canMegaEvo = "mega";
  if (typeof probe.actions.canMegaEvoX === "function") flags.canMegaEvoX = "mega";
  if (typeof probe.actions.canMegaEvoY === "function") flags.canMegaEvoY = "mega";
  const canTera = probe.sides.some((side) => side.pokemon.some((pokemon) => Boolean(probe.actions.canTerastallize(pokemon))));
  if (canTera) flags.canTerastallize = "tera";

  const kinds = Object.values(flags);
  const slot = moves * TARGETS * (1 + kinds.length) + bench;
  // The joint filter drops a pair that switches in the same bench slot and a pair with two events of one kind.
  const sameKind = (kind: Kind): number => (moves * TARGETS * kinds.filter((k2) => k2 === kind).length) ** 2;
  const exact = slot ** 2 - bench - sameKind("mega") - sameKind("tera");
  return { format: formatId, preview: { n, k, count: permutations(n, k) }, battle: { moves, bench, flags, slot, upper: slot ** 2, exact } };
}

/** The worst-case Team Preview request of the format: n Pokemon, pick k. */
function previewRequest({ n, k }: Bound["preview"]): JsonObject {
  const request = golden("preview.request.json") as Loose;
  request.maxChosenTeamSize = k;
  request.side.pokemon = Array.from({ length: n }, (_, i) => structuredClone(request.side.pokemon[i % request.side.pokemon.length]));
  return request;
}

/** The worst-case battle request: every move targets one of 3 locations, every event flag is set on both actives, every bench slot is alive. */
function battleRequest({ moves, bench, flags }: Bound["battle"]): JsonObject {
  const request = golden("turn.request.json") as Loose;
  const move = { move: "Tackle", id: "tackle", pp: 8, maxpp: 8, target: "normal", disabled: false };
  request.active = request.active.map(() => ({
    moves: Array.from({ length: moves }, (_, i) => ({ ...move, id: `tackle${i}` })),
    ...Object.fromEntries(Object.keys(flags).map((flag) => [flag, flag === "canTerastallize" ? "Normal" : true])),
  }));
  const mon = request.side.pokemon[0];
  request.side.pokemon = Array.from({ length: ACTIVE_SLOTS + bench }, (_, i) => ({ ...structuredClone(mon), active: i < ACTIVE_SLOTS }));
  return request;
}

const capacity = trainingExampleSchema.properties.candidates.maxItems;
const bounds = Object.keys(snapshot.formats).map(boundOf);

describe("candidate count bound N_max", () => {
  it("covers at least one supported format", () => {
    expect(bounds.length).toBeGreaterThan(0);
  });

  it.each(bounds)("the enumerator returns exactly the formula's count for $format", (bound) => {
    const preview = enumerateCandidates(decodeRequest(JSON.stringify(previewRequest(bound.preview)))).candidates;
    expect(preview).toHaveLength(bound.preview.count);
    const battle = enumerateCandidates(decodeRequest(JSON.stringify(battleRequest(bound.battle)))).candidates;
    expect(new Set(battle.map((c) => c.choice)).size).toBe(battle.length);
    expect(battle).toHaveLength(bound.battle.exact);
    expect(battle.length).toBeLessThanOrEqual(bound.battle.upper);
  });

  it.each(bounds)("N_max holds the worst case of $format", (bound) => {
    expect(bound.preview.count).toBeLessThanOrEqual(capacity);
    expect(bound.battle.upper).toBeLessThanOrEqual(capacity);
  });

  it("N_max is the larger bound over the supported formats, rounded up to a power of two", () => {
    const worst = Math.max(...bounds.map((b) => Math.max(b.preview.count, b.battle.upper)));
    expect(capacity).toBe(powerOfTwoAbove(worst));
  });

  it("reads the Reg M-C values the ADR derived from the engine", () => {
    const bound = bounds.find((b) => b.format === "gen9championsvgc2026regmc") as Bound;
    expect(bound.preview).toEqual({ n: 6, k: 4, count: 360 });
    expect(bound.battle).toMatchObject({ moves: 4, bench: 2, flags: { canMegaEvo: "mega" }, slot: 26, upper: 676 });
  });
});
