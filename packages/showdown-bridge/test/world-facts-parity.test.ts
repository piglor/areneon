import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { interpretChannel, ownChannel, toId, worldFactsAtRequests, type WorldFacts, type WorldFactsData } from "@areneon/public-state";
import { GOLDEN_DIR, goldenBattle } from "../src/golden.ts";
import { type Engine, loadEngine } from "../src/index.ts";
import { loadRandomCorpus, pendingFirstActor, playRandomBattle, type RandomBattleFile, UNAVAILABLE_DIR } from "../src/random-corpus.ts";
import { DEFAULT_SHOWDOWN_DIR } from "../src/engine.ts";

// ADR-026 slice L, Done-when: every committed channel, both seats, every request: each world fact equals a
// value recomputed here from the OMNISCIENT log's turn structure (the test oracle), or the module refuses
// with a typed error, counted. The oracle works backwards from a cut of the omniscient log, by Pokemon name,
// with no state machine shared with the module; the shared reader is the cross-check where the two overlap.

let engine: Engine;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
});

const FORMAT_DIR = GOLDEN_DIR;
const engineData = JSON.parse(readFileSync(join(FORMAT_DIR, "engine-data.json"), "utf8")) as { terrains: { id: string }[]; moves: { id: string }[] };

/** The moves whose engine definition adds the `stall` volatile, read from the pinned source, restricted to the format's moves. */
function stallMoves(): string[] {
  const source = readFileSync(join(process.env.SHOWDOWN_DIR || DEFAULT_SHOWDOWN_DIR, "data", "moves.ts"), "utf8");
  const ids = new Set(engineData.moves.map((m) => m.id));
  const out: string[] = [];
  for (const block of source.split(/\n\t(?=[a-z0-9]+: \{)/)) {
    const id = /^([a-z0-9]+): \{/.exec(block)?.[1];
    if (id !== undefined && ids.has(id) && /addVolatile\('stall'\)/.test(block)) out.push(id);
  }
  return out.sort();
}

/** The side conditions Court Change swaps, read from the pinned move's own list. */
function swapConditions(): string[] {
  const source = readFileSync(join(process.env.SHOWDOWN_DIR || DEFAULT_SHOWDOWN_DIR, "data", "moves.ts"), "utf8");
  const block = /\n\tcourtchange: \{[\s\S]*?const sideConditions = \[([\s\S]*?)\];/.exec(source)?.[1] ?? "";
  return [...block.matchAll(/'([a-z0-9]+)'/g)].map((m) => m[1] as string).sort();
}

let data: WorldFactsData;
beforeAll(() => {
  data = { stall_moves: stallMoves(), terrains: engineData.terrains.map((t) => t.id), swap_side_conditions: swapConditions() };
  expect(data.swap_side_conditions).toEqual(expect.arrayContaining(["reflect", "stealthrock", "spikes"]));
});

describe("the data the test passes", () => {
  it("holds the Protect family, Endure and the guards, from the engine's own source", () => {
    expect(data.stall_moves).toEqual(expect.arrayContaining(["protect", "detect", "endure", "kingsshield", "spikyshield", "quickguard", "wideguard"]));
    expect(data.stall_moves).not.toContain("tackle");
  });
});

// ---- the oracle ----

const IGNORED = new Set(["uhtml", "uhtmlchange", "request", "error"]);
const kind = (line: string): string => line.slice(1).split("|")[0] as string;
const fields = (line: string): string[] => line.slice(1).split("|").slice(1);
const tag = (line: string, key: string): string | undefined => fields(line).find((f) => f.startsWith(`[${key}]`))?.slice(key.length + 2).trim();
const who = (ident: string): { side: string; letter: string | null; name: string } => {
  const m = /^(p[12])([a-c])?: (.+)$/.exec(ident) as RegExpExecArray;
  return { side: m[1] as string, letter: m[2] ?? null, name: m[3] as string };
};
const turnAt = (omni: readonly string[], index: number): number => omni.slice(0, index).filter((l) => kind(l) === "turn").length === 0 ? 0 : Number(fields(omni.slice(0, index).filter((l) => kind(l) === "turn").at(-1) as string)[0]);
const isOwnChoice = (line: string): boolean => !/^(move|ability|item):/.test(tag(line, "from") ?? "");

function oracle(omni: readonly string[], cut: number, viewer: "p1" | "p2"): WorldFacts {
  const lines = omni.slice(0, cut);
  const turn = turnAt(omni, cut);
  const sideFacts = (side: "p1" | "p2") => {
    // who stands in each position: the last switch, drag, swap or faint that names it
    const slots = new Map<string, string | null>();
    lines.forEach((line) => {
      const f = fields(line);
      if (kind(line) === "switch" || kind(line) === "drag") {
        const w = who(f[0] as string);
        if (w.side === side) slots.set(w.letter as string, w.name);
      } else if (kind(line) === "faint") {
        const w = who(f[0] as string);
        if (w.side === side) slots.set(w.letter as string, null);
      } else if (kind(line) === "swap") {
        const w = who(f[0] as string);
        if (w.side === side) {
          const to = String.fromCharCode(97 + Number(f[1]));
          const a = slots.get(w.letter as string) ?? null;
          slots.set(w.letter as string, slots.get(to) ?? null);
          slots.set(to, a);
        }
      }
    });
    const named = (line: string, index: number, name: string): boolean => {
      const w = who(fields(line)[index] as string);
      return w.side === side && w.name === name;
    };
    const active: WorldFacts["own"]["active"] = {};
    for (const [letter, name] of [...slots].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
      if (name === null) {
        active[letter] = null;
        continue;
      }
      let since = -1;
      lines.forEach((l, i) => {
        if ((kind(l) === "switch" || kind(l) === "drag") && named(l, 0, name)) since = i;
      });
      const after = lines.slice(since + 1);
      const active_turns = after.filter((l) => kind(l) === "turn").length;
      let last_move: string | null = null;
      after.forEach((l) => {
        if (kind(l) === "move" && named(l, 0, name) && isOwnChoice(l)) last_move = toId(fields(l)[1] as string);
      });
      // successes of stall moves in turn order, a failure forgetting all before it
      let run: number[] = [];
      after.forEach((l, i) => {
        if (kind(l) !== "move" || !named(l, 0, name) || !isOwnChoice(l) || !data.stall_moves.includes(toId(fields(l)[1] as string))) return;
        const next = after[i + 1] as string;
        const t = turnAt(omni, since + 1 + i);
        if (kind(next) === "-singleturn" && named(next, 0, name)) run = run.length > 0 && run.at(-1) === t - 1 ? [...run, t] : [t];
        else run = [];
      });
      const protect_streak = run.length > 0 && (run.at(-1) as number) >= turn - 1 ? run.length : 0;
      // Toxic: the latest status of the Pokemon, if nothing cured it since, and the ticks after the later of it and the switch-in
      let statusAt = -1;
      let status = "";
      lines.forEach((l, i) => {
        if (kind(l) === "-status" && named(l, 0, name)) {
          statusAt = i;
          status = fields(l)[1] as string;
        }
        const cured = (kind(l) === "-curestatus" && named(l, 0, name)) || (kind(l) === "faint" && named(l, 0, name)) || (kind(l) === "-cureteam" && who(fields(l)[0] as string).side === side);
        if (cured && i > statusAt) statusAt = -1;
      });
      let toxic_ticks: number | null = null;
      if (statusAt >= 0 && status === "tox") {
        toxic_ticks = Math.min(15, lines.slice(Math.max(statusAt, since) + 1).filter((l) => {
          if (kind(l) === "-damage") return named(l, 0, name) && tag(l, "from") === "psn";
          if (kind(l) === "-heal") return named(l, 0, name) && tag(l, "from") === "ability: Poison Heal";
          return false;
        }).length);
      }
      active[letter] = { name, active_turns, last_move, protect_streak, toxic_ticks };
    }
    const faints = lines.map((l, i) => ({ l, i })).filter(({ l }) => kind(l) === "faint" && who(fields(l)[0] as string).side === side);
    const fainted_in = (t: number): boolean => faints.some(({ i }) => turnAt(omni, i) === t);
    // conditions: the events of each id in order
    const conditions: WorldFacts["own"]["conditions"] = {};
    const events: [string, string, number][] = [];
    lines.forEach((l, i) => {
      if (kind(l) === "-sidestart" || kind(l) === "-sideend") events.push([kind(l), `${who(fields(l)[0] as string).side}|${toId((fields(l)[1] as string).replace(/^move: /, ""))}`, i]);
      if (kind(l) === "-swapsideconditions") events.push(["swap", "", i]);
    });
    const mine = new Map<string, { started_turn: number; layers: number }>();
    const theirs = new Map<string, { started_turn: number; layers: number }>();
    const other = side === "p1" ? "p2" : "p1";
    for (const [k, key, i] of events) {
      if (k === "swap") {
        for (const id of data.swap_side_conditions) {
          const a = mine.get(id);
          const b = theirs.get(id);
          mine.delete(id);
          theirs.delete(id);
          if (b !== undefined) mine.set(id, b);
          if (a !== undefined) theirs.set(id, a);
        }
        continue;
      }
      const [s, id] = key.split("|") as [string, string];
      const target = s === side ? mine : theirs;
      void other;
      if (k === "-sideend") target.delete(id);
      else target.set(id, { started_turn: target.get(id)?.started_turn ?? turnAt(omni, i), layers: (target.get(id)?.layers ?? 0) + 1 });
    }
    for (const [id, c] of [...mine].sort((a, b) => (a[0] < b[0] ? -1 : 1))) conditions[id] = c;
    return {
      total_fainted: faints.length,
      fainted_this_turn: fainted_in(turn),
      fainted_last_turn: fainted_in(turn - 1),
      active,
      conditions,
    };
  };
  // field: the latest event of each id
  let weather: WorldFacts["weather"] = null;
  let terrain: WorldFacts["terrain"] = null;
  const pseudo = new Map<string, { started_turn: number }>();
  lines.forEach((l, i) => {
    if (kind(l) === "-weather" && !l.includes("|[upkeep]")) {
      const id = toId(fields(l)[0] as string);
      weather = id === "none" ? null : { id, started_turn: turnAt(omni, i) };
    } else if (kind(l) === "-fieldstart" || kind(l) === "-fieldend") {
      const id = toId((fields(l)[0] as string).replace(/^move: /, ""));
      const start = kind(l) === "-fieldstart";
      if (data.terrains.includes(id)) terrain = start ? { id, started_turn: turnAt(omni, i) } : terrain?.id === id ? null : terrain;
      else if (start) pseudo.set(id, { started_turn: turnAt(omni, i) });
      else pseudo.delete(id);
    }
  });
  const opposing = viewer === "p1" ? "p2" : "p1";
  return {
    world_facts_format: "world-facts/1",
    turn,
    weather,
    terrain,
    field_conditions: Object.fromEntries([...pseudo].sort((a, b) => (a[0] < b[0] ? -1 : 1))),
    own: sideFacts(viewer),
    opponent: sideFacts(opposing),
  };
}

/** The omniscient cut that matches the channel prefix before `request_line`: same count of lines both sides print. */
function cutFor(channel: readonly string[], omni: readonly string[], requestLine: number): number {
  const shown = channel.slice(0, requestLine).filter((l) => !IGNORED.has(kind(l))).length;
  let seen = 0;
  for (let i = 0; i < omni.length; i++) {
    if (seen === shown) return i;
    if (!IGNORED.has(kind(omni[i] as string))) seen += 1;
  }
  return omni.length;
}

interface Played {
  id: string;
  channels: { p1: readonly string[]; p2: readonly string[] };

  omni: readonly string[];
  teams: boolean;
}

const tally = new Map<string, number>();
const exercised = new Map<string, number>();
const hit = (what: string, yes: boolean): void => {
  if (yes) exercised.set(what, (exercised.get(what) ?? 0) + 1);
};
let requestsChecked = 0;
let factsChecked = 0;

function sweep(played: Played, using: WorldFactsData = data): void {
  for (const player of ["p1", "p2"] as const) {
    const channel = played.channels[player];
    // the two logs print the same lines, but for the ones only one of them has
    const a = channel.filter((l) => !IGNORED.has(kind(l))).map(kind);
    const b = played.omni.filter((l) => !IGNORED.has(kind(l))).map(kind);
    expect(a, `${played.id} ${player}: line kinds`).toEqual(b);
    const reader = interpretChannel(player, channel, { terrains: [...data.terrains] });

    for (const r of worldFactsAtRequests({ channel: ownChannel(player, channel), data: using })) {
      requestsChecked += 1;
      if (r.facts === null) {
        tally.set(r.refusal as string, (tally.get(r.refusal as string) ?? 0) + 1);
        continue;
      }
      factsChecked += 1;
      for (const side of [r.facts.own, r.facts.opponent]) {
        for (const a of Object.values(side.active)) {
          hit("active_turns >= 3", (a?.active_turns ?? 0) >= 3);
          hit("last_move", a?.last_move != null);
          hit("protect_streak >= 1", (a?.protect_streak ?? 0) >= 1);
          hit("protect_streak >= 2", (a?.protect_streak ?? 0) >= 2);
          hit("toxic_ticks >= 1", (a?.toxic_ticks ?? 0) >= 1);
        }
        hit("fainted_last_turn", side.fainted_last_turn);
        hit("fainted_this_turn", side.fainted_this_turn);
        hit("total_fainted >= 2", side.total_fainted >= 2);
        hit("side condition", Object.keys(side.conditions).length > 0);
        hit("layers >= 2", Object.values(side.conditions).some((c) => c.layers >= 2));
        hit("position emptied", Object.values(side.active).some((a) => a === null));
      }
      hit("weather", r.facts.weather !== null);
      hit("terrain", r.facts.terrain !== null);
      hit("field condition", Object.keys(r.facts.field_conditions).length > 0);
      expect(r.facts, `${played.id} ${player} request at line ${r.line}`).toEqual(oracle(played.omni, cutFor(channel, played.omni, r.line), player));
      // the shared reader's overlap: the turn, weather, terrain, field and side conditions (layers)
      const state = reader.frames[r.line]?.state;
      if (state !== undefined) {
        expect(r.facts.turn).toBe(state.turn);
        expect(r.facts.weather?.id ?? null).toBe(state.weather === null ? null : toId(state.weather));
        expect(r.facts.terrain?.id ?? null).toBe(state.terrain);
        expect(Object.keys(r.facts.field_conditions)).toEqual([...state.field_conditions].sort());
        for (const side of ["own", "opponent"] as const) {
          const layers = Object.fromEntries(Object.entries(r.facts[side].conditions).map(([id, c]) => [id, c.layers]));
          expect(layers).toEqual(state[side].conditions);
        }
      }
    }
  }
}

const COMMITTED = (dir: string): RandomBattleFile[] => loadRandomCorpus(join(FORMAT_DIR, dir));

describe("world facts against the omniscient log, on every committed channel", () => {
  it("random-play corpus", async () => {
    for (const b of loadRandomCorpus()) {
      const { id, seed, actor_seeds, teams } = b;
      const record = await playRandomBattle(engine, { id, seed, actor_seeds, teams });
      expect(record.channels).toEqual({ p1: b.channels.p1.lines, p2: b.channels.p2.lines });
      sweep({ id: `random/${id}`, channels: record.channels, omni: record.omniscient_log, teams: true });
    }
  }, 180_000);

  it("unavailable corpus (pending-first play)", async () => {
    for (const b of COMMITTED("unavailable")) {
      const { id, seed, actor_seeds, teams } = b;
      const record = await playRandomBattle(engine, { id, seed, actor_seeds, teams }, pendingFirstActor);
      expect(record.channels).toEqual({ p1: b.channels.p1.lines, p2: b.channels.p2.lines });
      sweep({ id: `unavailable/${id}`, channels: record.channels, omni: record.omniscient_log, teams: true });
    }
    expect(UNAVAILABLE_DIR).toContain("unavailable");
  }, 60_000);

  it("the Illusion, Baton Pass and Fling battles", async () => {
    for (const dir of ["illusion", "baton-pass", "fling"]) {
      const b = JSON.parse(readFileSync(join(FORMAT_DIR, dir, "battle.json"), "utf8")) as RandomBattleFile;
      const { id, seed, actor_seeds, teams } = b;
      const record = await playRandomBattle(engine, { id, seed, actor_seeds, teams });
      expect(record.channels, dir).toEqual({ p1: b.channels.p1.lines, p2: b.channels.p2.lines });
      sweep({ id: `${dir}/${id}`, channels: record.channels, omni: record.omniscient_log, teams: true });
    }
  }, 60_000);

  it("the golden battle", async () => {
    const record = await goldenBattle(engine);
    sweep({ id: "golden", channels: record.channels, omni: record.omniscient_log, teams: true });
  }, 60_000);

  it("compared many facts, and the only refusals are the Illusion replace", () => {
    expect(factsChecked).toBeGreaterThan(500);
    console.log(`world-facts parity: ${factsChecked} of ${requestsChecked} requests compared; refusals ${JSON.stringify([...tally])}; exercised ${JSON.stringify([...exercised])}`);
    expect([...tally.keys()].filter((code) => code !== "illusion_replace")).toEqual([]);
    // the corpus reaches the states the facts exist for (hazard stacking is hand-written in the public-state tests)
    for (const what of ["active_turns >= 3", "last_move", "protect_streak >= 2", "toxic_ticks >= 1", "fainted_last_turn", "fainted_this_turn", "side condition", "weather", "terrain", "field condition", "position emptied"]) {
      expect(exercised.get(what) ?? 0, what).toBeGreaterThan(0);
    }
  });

  it("has teeth: a wrong stall-move list makes the sweep fail against the oracle", async () => {
    let failed = false;
    for (const b of loadRandomCorpus()) {
      const { id, seed, actor_seeds, teams } = b;
      const record = await playRandomBattle(engine, { id, seed, actor_seeds, teams });
      try {
        sweep({ id, channels: record.channels, omni: record.omniscient_log, teams: true }, { ...data, stall_moves: [] });
      } catch {
        failed = true;
        break;
      }
    }
    expect(failed).toBe(true);
  }, 120_000);
});
