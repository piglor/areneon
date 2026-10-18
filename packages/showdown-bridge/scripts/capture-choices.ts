// Re-capture the golden choice corpus (ADR-006 rule 7) after a Showdown pin
// bump or an enumerator change:
//   node packages/showdown-bridge/scripts/capture-choices.ts
// Each case below is found by a deterministic seeded search: both sides
// play engine-accepted candidates (never pending ones) until the case's
// player gets a request of the case's kind. The found log is committed;
// CI replays it, never the search. Review the diff: a changed case is a
// behaviour change of the engine or the enumerator.
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { CloneJudge, legalChoiceSet } from "../src/choice-judge.ts";
import {
  CASE_KINDS,
  type CaseKind,
  captureCase,
  CHOICES_DIR,
  type ChoiceCaseSpec,
  missedActions,
  replayCase,
} from "../src/choice-corpus.ts";
import { loadEngine, type PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT, syntheticTeam } from "../src/golden.ts";
import { decodeRequest } from "../src/request.ts";
import type { Player } from "../src/runner.ts";

const engine = loadEngine();

const set = (species: string, item: string, ability: string, moves: string[]): PokemonSet => ({
  name: "",
  species,
  item,
  ability,
  moves,
  nature: "Serious",
  gender: "",
  evs: { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: 4 },
  ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
  level: 50,
});

/** Its six Pokémon's moves cover every target type in the Reg M-C move pool. */
const TARGETS: PokemonSet[] = [
  set("Clefable", "Leftovers", "Magic Guard", ["Moonblast", "Helping Hand", "Life Dew", "Protect"]),
  set("Garchomp", "Life Orb", "Rough Skin", ["Earthquake", "Outrage", "Stealth Rock", "Rock Slide"]),
  set("Medicham", "Focus Sash", "Pure Power", ["Acupressure", "Counter", "Coaching", "Zen Headbutt"]),
  set("Chimecho", "Sitrus Berry", "Levitate", ["Heal Bell", "Trick Room", "Psychic", "Protect"]),
  set("Talonflame", "Sharp Beak", "Gale Wings", ["Brave Bird", "Tailwind", "Flare Blitz", "Protect"]),
  set("Arcanine", "Charcoal", "Intimidate", ["Howl", "Flamethrower", "Extreme Speed", "Protect"]),
];

const withItems = (items: Record<string, string>): PokemonSet[] =>
  syntheticTeam().map((s) => ({ ...s, item: items[s.species] ?? s.item }));

/** Golden team; its Gengar holds Gengarite and knows Imprison and Mean Look. */
const TRAPPER: PokemonSet[] = syntheticTeam().map((s) =>
  s.species === "Gengar"
    ? { ...s, item: "Gengarite", moves: ["Imprison", "Mean Look", "Shadow Ball", "Protect"] }
    : s,
);
const GOLDEN = syntheticTeam();

interface Search {
  id: string;
  kind: CaseKind;
  purpose: string;
  teams: Record<Player, PokemonSet[]>;
  player: Player;
  /** Team Preview choice per side; otherwise searched like any request. */
  preview?: Partial<Record<Player, string>>;
  /** Steer a side: pick among the accepted candidates matching this. */
  prefer?: Partial<Record<Player, RegExp>>;
}

const SEARCHES: Search[] = [
  { id: "preview", kind: "preview", purpose: "Team Preview: every ordered pick of 4 from 6", teams: { p1: GOLDEN, p2: GOLDEN }, player: "p1" },
  { id: "targets-clefable-garchomp", kind: "targets", purpose: "target types normal, adjacentAlly, allies, self, allAdjacent, randomNormal, foeSide, allAdjacentFoes", teams: { p1: TARGETS, p2: GOLDEN }, player: "p1", preview: { p1: "team 1234" } },
  { id: "targets-medicham-chimecho", kind: "targets", purpose: "target types adjacentAllyOrSelf, scripted, adjacentAlly, allyTeam, all", teams: { p1: TARGETS, p2: GOLDEN }, player: "p1", preview: { p1: "team 3456" } },
  { id: "targets-talonflame-arcanine", kind: "targets", purpose: "target types any, allySide, allies", teams: { p1: TARGETS, p2: GOLDEN }, player: "p1", preview: { p1: "team 5612" } },
  { id: "mega", kind: "mega", purpose: "one Mega-capable active", teams: { p1: withItems({ Charizard: "Charizardite Y" }), p2: GOLDEN }, player: "p1", preview: { p1: "team 1234" } },
  { id: "mega-both", kind: "mega-both", purpose: "two Mega-capable actives: Mega on either slot, never both", teams: { p1: withItems({ Venusaur: "Venusaurite", Charizard: "Charizardite Y" }), p2: GOLDEN }, player: "p1", preview: { p1: "team 1234" } },
  { id: "double-switch", kind: "double-switch", purpose: "both slots may switch to the same two bench Pokémon", teams: { p1: GOLDEN, p2: GOLDEN }, player: "p1" },
  { id: "force-switch-one", kind: "force-switch-one", purpose: "one fainted slot replaced, the other passes", teams: { p1: GOLDEN, p2: GOLDEN }, player: "p1" },
  { id: "force-switch-two", kind: "force-switch-two", purpose: "both slots fainted, two bench Pokémon: each switches in a different one", teams: { p1: GOLDEN, p2: GOLDEN }, player: "p1", prefer: { p2: /^move 4, move 4$|move 4$/ } },
  { id: "force-switch-short-bench", kind: "force-switch-short-bench", purpose: "both slots fainted, one bench Pokémon: one switches, one passes", teams: { p1: GOLDEN, p2: GOLDEN }, player: "p1", prefer: { p2: /move 4$/ } },
  { id: "trapped", kind: "trapped", purpose: "a visibly trapped active (Mean Look): no switch offered for it", teams: { p1: GOLDEN, p2: TRAPPER }, player: "p1", preview: { p2: "team 6123" }, prefer: { p2: /^move 2 [12],/ } },
  { id: "maybe-trapped", kind: "maybe-trapped", purpose: "foe Mega Gengar may have Shadow Tag: the last active's switches are pending", teams: { p1: GOLDEN, p2: TRAPPER }, player: "p1", preview: { p2: "team 6123" }, prefer: { p2: /^move 3 [12] mega,/ } },
  { id: "maybe-disabled", kind: "maybe-disabled", purpose: "foe used Imprison: the last active's moves are pending (maybeDisabled, maybeLocked)", teams: { p1: GOLDEN, p2: TRAPPER }, player: "p1", preview: { p2: "team 6123" }, prefer: { p2: /^move 1,/ } },
  { id: "fainted-no-replacement", kind: "fainted-no-replacement", purpose: "one active fainted with no bench left: that slot passes", teams: { p1: GOLDEN, p2: GOLDEN }, player: "p1" },
  { id: "locked-move", kind: "locked-move", purpose: "locked into Outrage: one move, no target, no switch", teams: { p1: TARGETS, p2: GOLDEN }, player: "p1", preview: { p1: "team 1234" }, prefer: { p1: /, move 2$/ } },
];

function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function search(s: Search, battles = 200, maxTurns = 60): ChoiceCaseSpec {
  for (let b = 0; b < battles; b++) {
    const seed = `sodium,${b.toString(16).padStart(32, "0")}`;
    const spec: ChoiceCaseSpec = { id: s.id, kind: s.kind, purpose: s.purpose, seed, teams: s.teams, player: s.player, log: [] };
    // An empty log starts the battle; replayCase also validates both teams.
    const { battle } = replayCase(engine, spec);
    const next = random(b + 1);
    while (!battle.ended && battle.turn < maxTurns) {
      let progressed = false;
      for (const side of battle.sides) {
        if (battle.ended || !side.requestState || side.isChoiceDone()) continue;
        const request = decodeRequest(JSON.stringify(side.activeRequest));
        if (request.phase === "wait") continue;
        const player = side.id as Player;
        if (player === s.player && CASE_KINDS[s.kind](request)) return spec;
        let choice = request.phase === "preview" ? s.preview?.[player] : undefined;
        if (choice === undefined) {
          const { choices } = legalChoiceSet({
            request: { ...request, request_seq: 0 },
            showdown_commit: engine.commit,
            judge: new CloneJudge(engine, battle, player),
          });
          const preferred = choices.filter((c) => s.prefer?.[player]?.test(c) === true);
          const pool = preferred.length > 0 ? preferred : choices;
          choice = pool.length > 0 ? (pool[Math.floor(next() * pool.length)] as string) : "default";
        }
        if (!battle.choose(player, choice)) throw new Error(`${s.id}: engine refused ${choice}`);
        spec.log.push({ player, choice });
        progressed = true;
      }
      if (!progressed) break;
    }
  }
  throw new Error(`${s.id}: no ${s.kind} request in ${battles} battles`);
}

rmSync(CHOICES_DIR, { recursive: true, force: true });
mkdirSync(CHOICES_DIR, { recursive: true });
for (const s of SEARCHES) {
  const spec = search(s);
  const replayed = replayCase(engine, spec);
  const missed = missedActions(engine, replayed, spec.player);
  if (missed.length > 0) throw new Error(`${s.id}: enumerator missed ${missed.join("; ")}`);
  const file = `${s.id}.json`;
  writeFileSync(join(CHOICES_DIR, file), `${JSON.stringify(captureCase(engine, spec), null, 2)}\n`);
  console.log(`wrote ${file} (${spec.seed}, ${spec.log.length} choices, ${GOLDEN_FORMAT})`);
}
