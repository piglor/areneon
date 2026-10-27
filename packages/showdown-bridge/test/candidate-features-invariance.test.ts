import { describe, expect, it } from "vitest";
import type { PublicMon } from "@areneon/public-state";
import { probeMechanics } from "../src/mechanics-probe.ts";
import { type CandidateFeatures, candidateFeatures, type FeatureConfigs } from "../src/candidate-features.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { type AssumedSet } from "../src/damage-aware-assumed-set.ts";
import type { PokemonSet } from "../src/engine.ts";
import { firstLegal, GOLDEN_FORMAT } from "../src/golden.ts";
import { type BattleRecord, type HiddenVerdicts, type OmniscientLog, type PlayerChannel } from "../src/index.ts";
import { playerChannel } from "../src/player-channel.ts";
import type { RandomBattleFile } from "../src/random-corpus.ts";
import { type Actor, runBattle } from "../src/runner.ts";
import { exportPaste } from "../src/teams.ts";
import { config, corpus, engine, mechanics, snapshot, stateOf, toId, vocabulary } from "./damage-aware-support.ts";
import { mutated } from "./damage-aware-invariance-support.ts";

// ADR-022 section 3 (privacy test) and R1, ticket #1029: the tier A rows and `feature_set_hash` are a function of
// the player's own channel, own team and candidates, so two battles whose player channels agree up to a request
// but whose opponents hold different unrevealed fields give byte-identical features there. The REAL probe is used
// (a stand-in would not see a stat, an item or an ability). p1 and p2 both play `firstLegal`, which reads nothing
// of its own team beyond the first legal choice, so the pair differs only in what p2 holds.

const MAX_TURNS = 10;
const configs = (change: Partial<FeatureConfigs> = {}): FeatureConfigs => ({ vocabulary, config, mechanics, snapshot, data_class: "synthetic", ...change });

interface Point {
  seq: number;
  channel: string[];
  features: (change?: Partial<FeatureConfigs>) => CandidateFeatures;
  state: ReturnType<typeof stateOf>;
}

/** p1's move requests of one battle, each with a closure that builds the features the way the policy side would. */
async function points(battle: RandomBattleFile, p2: PokemonSet[]): Promise<Point[]> {
  const out: Point[] = [];
  const p1: Actor = (request, context) => {
    if (request.phase === "move" && request.legal !== undefined && context !== undefined && context.team !== undefined) {
      const legal = request.legal;
      const candidates = [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) }));
      const lines = [...context.channel];
      const team = context.team;
      out.push({
        seq: request.request_seq,
        channel: lines,
        features: (change = {}) => candidateFeatures(engine, configs(change), playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines }), team as never, candidates),
        state: stateOf({ context: { ...context, channel: lines } } as never),
      });
    }
    return firstLegal(request, context as never);
  };
  await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: battle.seed, teams: { p1: battle.teams.p1, p2 }, actors: { p1, p2: firstLegal }, maxTurns: MAX_TURNS });
  return out;
}

/** p2's team with every set's ability changed to another the species' roster entry lists (kept only while the team stays legal). */
function otherAbilities(team: PokemonSet[]): PokemonSet[] {
  const out = team.map((set) => structuredClone(set));
  out.forEach((set) => {
    const entry = vocabulary.roster.find((e) => e.id === toId(set.species) || e.id === toId(set.species.split("-")[0] ?? ""));
    const other = entry?.abilities.find((id) => id !== toId(set.ability));
    if (other === undefined) return;
    const before = set.ability;
    set.ability = (engine.sim.Dex.mod("champions").abilities.get(other) as { name: string }).name;
    if ((engine.sim.TeamValidator.get(GOLDEN_FORMAT).validateTeam(structuredClone(out)) ?? []).length > 0) set.ability = before;
  });
  return out;
}

const bytes = (f: CandidateFeatures): string => jcs(f as unknown as Json);
const byId = (id: string): RandomBattleFile => corpus.find((b) => b.id === id) as RandomBattleFile;

/** Request points of a pair whose p1 channels are equal (the prefix up to and including the request). */
function equalPoints(a: Point[], b: Point[]): { a: Point; b: Point }[] {
  const out: { a: Point; b: Point }[] = [];
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const [left, right] = [a[i] as Point, b[i] as Point];
    if (left.seq !== right.seq || jcs(left.channel as unknown as Json) !== jcs(right.channel as unknown as Json)) break;
    out.push({ a: left, b: right });
  }
  return out;
}

// Pairs 1 and 2: nature, Stat Points, rotated items and moves 2 to 4 differ (abilities kept, so the ability the
// opponent reveals is the same in both). Pairs 3 and 4: those plus every ability changed to the species' other one. The rest change only unrevealed moves, so the channels stay equal for longer.
/** Only the moves 2 to 4 differ (nature, Stat Points and items are as brought), so the channels stay equal for longer and across a reveal of an item or ability. */
const movesOnly = (team: PokemonSet[]): PokemonSet[] => mutated(team).map((set, index) => ({ ...set, nature: (team[index] as PokemonSet).nature, evs: (team[index] as PokemonSet).evs, item: (team[index] as PokemonSet).item }));

// What p2 holds unrevealed differs in each pair: `mutated` (nature, Stat Points, rotated items, moves 2 to 4), `movesOnly`, and
// either with every ability changed to the species' other one where it has two. The `reveal` pairs keep abilities and items as
// brought, so a reveal of an ability (Pressure, Intimidate) or an item (Shuca Berry) is the same in both battles and the
// channels stay equal after it.
const PAIRS: { id: string; variant: (team: PokemonSet[]) => PokemonSet[] }[] = [
  { id: "b06", variant: mutated },
  { id: "b12", variant: mutated },
  { id: "b01", variant: (team) => otherAbilities(mutated(team)) },
  { id: "b02", variant: (team) => otherAbilities(mutated(team)) },
  { id: "b00", variant: (team) => otherAbilities(movesOnly(team)) },
  { id: "b03", variant: (team) => otherAbilities(movesOnly(team)) },
  { id: "b04", variant: movesOnly },
  { id: "b05", variant: movesOnly },
  { id: "b14", variant: movesOnly }, // reveal: Kingambit's Pressure at request 2, not its vocabulary-first ability
  { id: "b27", variant: movesOnly }, // reveal: Krookodile's Intimidate at request 2, not its vocabulary-first ability
  { id: "b20", variant: movesOnly }, // reveal: Gourgeist's Shuca Berry at request 2
];

/** What the real opponent holds: the negative control's assumed set, as a baseline that was handed the real team would build it. */
const cheating = (real: PokemonSet[]) => (mon: PublicMon): AssumedSet => {
  const set = structuredClone(real.find((s) => [mon.species, mon.base_species].includes(toId(s.species)))) as PokemonSet;
  return { set, paste: exportPaste(engine, [structuredClone(set)]), revealed_moves: set.moves.map(toId), filler_moves: [], unplaceable_moves: [], shown_hp_percent: mon.fainted ? 0 : mon.hp_num, mega: mon.mega };
};

describe("hidden-information invariance of the tier A features (ADR-022 section 3, R1)", () => {
  // The real probe on 20+ request points of eleven seeded battle pairs, each built twice: tens of seconds, minutes on a loaded machine.
  it("gives byte-identical rows and feature_set_hash at every request with an equal player channel", async () => {
    let total = 0;
    let withRows = 0;
    let pairsUsed = 0;
    let afterReveal = 0;
    for (const { id, variant } of PAIRS) {
      const battle = byId(id);
      const other = variant(battle.teams.p2);
      expect(JSON.stringify(other), `${id} differs`).not.toBe(JSON.stringify(battle.teams.p2));
      const equal = equalPoints(await points(battle, battle.teams.p2), await points(battle, other));
      if (equal.length > 0) pairsUsed += 1;
      for (const { a, b } of equal) {
        const [left, right] = [a.features(), b.features()];
        expect(bytes(right), `${id} request ${a.seq}`).toBe(bytes(left));
        expect(right.feature_set_hash).toBe(left.feature_set_hash);
        total += 1;
        if (left.rows.length > 0) withRows += 1;
        if (Object.values(a.state.opponent.roster).some((mon) => mon.ability !== null || mon.item !== null)) afterReveal += 1;
      }
    }
    expect(pairsUsed).toBeGreaterThanOrEqual(2);
    expect(total).toBeGreaterThanOrEqual(20);
    // not vacuous: the rows are really there
    expect(withRows).toBeGreaterThanOrEqual(15);
    // and some of the equal points come after an ability or item reveal
    expect(afterReveal).toBeGreaterThanOrEqual(3);
  }, 300_000);

  it("declares a revealed ability or item to the probe as revealed, not the vocabulary default", async () => {
    let abilities = 0;
    let notDefault = 0;
    let items = 0;
    for (const id of ["b14", "b27", "b20"]) {
      const battle = byId(id);
      for (const point of await points(battle, movesOnly(battle.teams.p2))) {
        const declared: string[] = [];
        const probe: NonNullable<FeatureConfigs["probe"]> = (context, query) => {
          if (query.query.kind === "damage") declared.push(query.query.defender.set);
          return probeMechanics(context, query);
        };
        point.features({ probe });
        for (const mon of Object.values(point.state.opponent.roster)) {
          const mine = declared.filter((paste) => toId((paste.split("\n")[0] as string).split("@")[0] as string).includes(toId(mon.base_species)));
          // b20 is the item case only: the reader marks Gardevoir's Telepathy as received (Trace), so it is no opponent evidence of its own
          if (mon.ability !== null && id !== "b20") {
            for (const paste of mine) expect(toId(/^Ability: (.+)$/m.exec(paste)?.[1] ?? ""), `${id} request ${point.seq}`).toBe(mon.ability);
            abilities += mine.length;
            const first = vocabulary.roster.find((r) => r.id === mon.base_species)?.abilities[0];
            if (mine.length > 0 && first !== mon.ability) notDefault += 1;
          }
          if (mon.item !== null) {
            for (const paste of mine) expect(toId((paste.split("\n")[0] as string).split("@")[1] ?? ""), `${id} request ${point.seq}`).toBe(mon.item);
            items += mine.length;
          }
        }
      }
    }
    expect(abilities).toBeGreaterThanOrEqual(2);
    expect(notDefault).toBeGreaterThanOrEqual(1);
    expect(items).toBeGreaterThanOrEqual(1);
  }, 300_000);

  // Negative control: the invariance comparison, pair for pair, with a builder handed the opponent's REAL team must differ.
  it("has teeth: a builder handed the real opponent team gives different rows at such a request", async () => {
    let differing = 0;
    let compared = 0;
    for (const { id, variant } of PAIRS) {
      const battle = byId(id);
      const other = variant(battle.teams.p2);
      for (const pair of equalPoints(await points(battle, battle.teams.p2), await points(battle, other))) {
        compared += 1;
        const [left, right] = [pair.a.features({ assume: cheating(battle.teams.p2) }), pair.b.features({ assume: cheating(other) })];
        if (bytes(left) !== bytes(right)) differing += 1;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(20);
    expect(differing).toBeGreaterThan(0);
  }, 300_000);
});

describe("canary (ADR-022 section 3)", () => {
  it("shows no never-public field of the opponent in the JCS bytes of any row set", async () => {
    const markers = (team: PokemonSet[], channel: string): string[] =>
      team.flatMap((set) => [set.item, set.nature, ...set.moves.slice(1)]).filter((name) => name !== "" && !channel.toLowerCase().includes(name.toLowerCase())).map((name) => name.toLowerCase());
    let scanned = 0;
    for (const id of ["b00", "b06", "b12"]) {
      const battle = byId(id);
      for (const team of [battle.teams.p2, mutated(battle.teams.p2)]) {
        for (const point of await points(battle, team)) {
          const text = bytes(point.features()).toLowerCase();
          const hidden = markers(team, point.channel.join("\n"));
          expect(hidden.length).toBeGreaterThan(0);
          for (const marker of hidden) expect(text.includes(marker), `${id} request ${point.seq}`).toBe(false);
          scanned += 1;
        }
      }
    }
    expect(scanned).toBeGreaterThanOrEqual(20);
  }, 300_000);

  it("is a scan that sees a marker when one is there (negative control)", () => {
    const text = jcs({ rows: [], marker: "Choice Scarf" } as unknown as Json).toLowerCase();
    expect(text.includes("choice scarf")).toBe(true);
  });
});

describe("the type rule (ADR-022 section 3)", () => {
  it("takes no opponent team, omniscient log, hidden verdicts, battle record or seed", () => {
    // Compile-time checks: `pnpm typecheck` fails if any of these compiles.
    const neverRun = (log: OmniscientLog, verdicts: HiddenVerdicts[], record: BattleRecord, seed: string, raw: { player: "p1"; lines: string[] }, channel: PlayerChannel): void => {
      const team = [] as never;
      // @ts-expect-error: the omniscient log is not a PlayerChannel
      void candidateFeatures(engine, configs(), log, team, []);
      // @ts-expect-error: nor hidden verdicts
      void candidateFeatures(engine, configs(), verdicts, team, []);
      // @ts-expect-error: nor a battle record
      void candidateFeatures(engine, configs(), record, team, []);
      // @ts-expect-error: nor a seed
      void candidateFeatures(engine, configs(), seed, team, []);
      // @ts-expect-error: nor an unchecked log
      void candidateFeatures(engine, configs(), raw, team, []);
      // @ts-expect-error: the configs hold no opponent team
      void candidateFeatures(engine, { ...configs(), opponent_team: [] }, channel, team, []);
      // @ts-expect-error: the team argument is the player's own team, not a record
      void candidateFeatures(engine, configs(), channel, record, []);
    };
    expect(neverRun).toBeTypeOf("function");
  });
});
