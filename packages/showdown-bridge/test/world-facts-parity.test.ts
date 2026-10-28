import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { interpretChannel, ownChannel, toId, worldFactsAtRequests, type WorldFacts, type WorldFactsData } from "@areneon/public-state";
import { GOLDEN_DIR, goldenBattle } from "../src/golden.ts";
import { type Engine, loadEngine } from "../src/index.ts";
import { loadRandomCorpus, pendingFirstActor, playRandomBattle, type RandomBattleFile, UNAVAILABLE_DIR } from "../src/random-corpus.ts";
import { readWorldFactsData } from "./world-facts-data.ts";

// ADR-026 slice L, Done-when: every committed channel, both seats, every request: each world fact equals a
// value recomputed here from the OMNISCIENT log's turn structure (the test oracle), or the module refuses
// with a typed error, counted. The oracle works backwards from a cut of the omniscient log, by Pokemon name,
// with no state machine shared with the module; the shared reader is the cross-check where the two overlap.

let engine: Engine;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
});

const FORMAT_DIR = GOLDEN_DIR;

let data: WorldFactsData;
beforeAll(() => {
  data = readWorldFactsData();
  expect(data.toxic_stage_max, "the engine's Toxic stage cap").toBe(15);
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
/** A move line the engine ran as a move action of its own though a tag names an effect: Round, Instruct, Dancer. */
const isRunAction = (line: string): boolean => data.action_sources.map(toId).includes(toId(tag(line, "from") ?? ""));

function oracle(omni: readonly string[], cut: number, viewer: "p1" | "p2"): WorldFacts {
  const lines = omni.slice(0, cut);
  const turn = turnAt(omni, cut);
  /** Whether the residual of the turn at line `i` had run (or no turn had begun): nothing begun there has been counted down. */
  const afterResidual = (i: number): boolean => {
    let j = i - 1;
    while (j >= 0 && kind(omni[j] as string) !== "turn" && kind(omni[j] as string) !== "upkeep") j--;
    return j < 0 || kind(omni[j] as string) === "upkeep";
  };
  /** Whether the Pokemon had run a move action (as `move_actions` counts them) in the turn of line `i`, before it. */
  const actedThisTurn = (i: number, name: string): boolean => {
    let j = i - 1;
    while (j >= 0 && kind(omni[j] as string) !== "turn") {
      const l = omni[j] as string;
      const mine = fields(l)[0] !== undefined && /^p[12][a-c]?: /.test(fields(l)[0] as string) && who(fields(l)[0] as string).name === name;
      if (mine && ((kind(l) === "move" && (isOwnChoice(l) || isRunAction(l))) || kind(l) === "cant" || (kind(l) === "-damage" && tag(l, "from") === "confusion"))) return true;
      j--;
    }
    return false;
  };
  /** Who began a condition at line `i`: the `[of]` Pokemon of the line, else the actor of the latest `move` line before it (any tag). */
  const sourceAt = (i: number): { side: "own" | "opponent"; name: string } | null => {
    const seen = (ident: string) => {
      const w = who(ident);
      return { side: w.side === viewer ? ("own" as const) : ("opponent" as const), name: w.name };
    };
    const of = tag(omni[i] as string, "of");
    if (of !== undefined && /^p[12][a-c]?: /.test(of)) return seen(of);
    for (let j = i - 1; j >= 0; j--) if (kind(omni[j] as string) === "move") return seen(fields(omni[j] as string)[0] as string);
    return null;
  };
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
      const move_actions = after.filter((l) => (kind(l) === "move" && named(l, 0, name) && (isOwnChoice(l) || isRunAction(l))) || (kind(l) === "cant" && named(l, 0, name)) || (kind(l) === "-damage" && tag(l, "from") === "confusion" && named(l, 0, name))).length;
      let last_move: string | null = null;
      after.forEach((l) => {
        if (kind(l) === "move" && named(l, 0, name) && (isOwnChoice(l) || isRunAction(l))) last_move = toId(fields(l)[1] as string);
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
        toxic_ticks = Math.min(data.toxic_stage_max, lines.slice(Math.max(statusAt, since) + 1).filter((l) => {
          if (kind(l) === "-damage") return named(l, 0, name) && tag(l, "from") === "psn";
          if (kind(l) === "-heal") return named(l, 0, name) && tag(l, "from") === "ability: Poison Heal";
          return false;
        }).length);
      }
      // volatiles: dated at their first -start since the switch-in, forgotten at -end
      const volatiles: Record<string, { started_turn: number; after_residual: boolean; moved_before: boolean }> = {};
      after.forEach((l, i) => {
        if ((kind(l) !== "-start" && kind(l) !== "-end") || !named(l, 0, name) || !/^p[12][a-c]: /.test(fields(l)[0] as string)) return;
        const id = toId((fields(l)[1] as string).replace(/^[a-z ]+: /, ""));
        if (kind(l) === "-end") Reflect.deleteProperty(volatiles, id);
        else if (volatiles[id] === undefined) volatiles[id] = { started_turn: turnAt(omni, since + 1 + i), after_residual: afterResidual(since + 1 + i), moved_before: actedThisTurn(since + 1 + i, name) };
      });
      active[letter] = { name, active_turns, move_actions, last_move, protect_streak, toxic_ticks, volatiles: Object.fromEntries(Object.entries(volatiles).sort((a, b) => (a[0] < b[0] ? -1 : 1))) };
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
    type Cond = { started_turn: number; after_residual: boolean; source: { side: "own" | "opponent"; name: string } | null; layers: number };
    const mine = new Map<string, Cond>();
    const theirs = new Map<string, Cond>();
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
      else target.set(id, { started_turn: target.get(id)?.started_turn ?? turnAt(omni, i), after_residual: target.get(id)?.after_residual ?? afterResidual(i), source: target.get(id) === undefined ? sourceAt(i) : (target.get(id) as Cond).source, layers: (target.get(id)?.layers ?? 0) + 1 });
    }
    for (const [id, c] of [...mine].sort((a, b) => (a[0] < b[0] ? -1 : 1))) conditions[id] = c;
    return {
      total_fainted: faints.length,
      fainted_this_turn: fainted_in(turn),
      fainted_last_turn: fainted_in(turn - 1),
      active,
      conditions,
      move_uses: moveUses(side),
    };
  };
  const moveUses = (side: "p1" | "p2"): WorldFacts["own"]["move_uses"] => {
    const uses: Record<string, Record<string, number>> = {};
    for (const l of lines) {
      if (kind(l) !== "move" || !isOwnChoice(l)) continue;
      const w = who(fields(l)[0] as string);
      if (w.side !== side) continue;
      const move = toId(fields(l)[1] as string);
      const mine = (uses[w.name] ??= {});
      mine[move] = (mine[move] ?? 0) + 1;
    }
    return Object.fromEntries(Object.entries(uses).sort((a, b) => (a[0] < b[0] ? -1 : 1)).map(([n, m]) => [n, Object.fromEntries(Object.entries(m).sort((a, b) => (a[0] < b[0] ? -1 : 1)))]));
  };
  // field: the latest event of each id
  let weather: WorldFacts["weather"] = null;
  let terrain: WorldFacts["terrain"] = null;
  const pseudo = new Map<string, { started_turn: number; after_residual: boolean; source: { side: "own" | "opponent"; name: string } | null }>();
  lines.forEach((l, i) => {
    if (kind(l) === "-weather" && !l.includes("|[upkeep]")) {
      const id = toId(fields(l)[0] as string);
      weather = id === "none" ? null : { id, started_turn: turnAt(omni, i), after_residual: afterResidual(i), source: sourceAt(i) };
    } else if (kind(l) === "-fieldstart" || kind(l) === "-fieldend") {
      const id = toId((fields(l)[0] as string).replace(/^move: /, ""));
      const start = kind(l) === "-fieldstart";
      if (data.terrains.includes(id)) terrain = start ? { id, started_turn: turnAt(omni, i), after_residual: afterResidual(i), source: sourceAt(i) } : terrain?.id === id ? null : terrain;
      else if (start) pseudo.set(id, { started_turn: turnAt(omni, i), after_residual: afterResidual(i), source: sourceAt(i) });
      else pseudo.delete(id);
    }
  });
  const opposing = viewer === "p1" ? "p2" : "p1";
  return {
    world_facts_format: "world-facts/2",
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
          hit("move_actions >= 3", (a?.move_actions ?? 0) >= 3);
          hit("protect_streak >= 1", (a?.protect_streak ?? 0) >= 1);
          hit("protect_streak >= 2", (a?.protect_streak ?? 0) >= 2);
          hit("toxic_ticks >= 1", (a?.toxic_ticks ?? 0) >= 1);
          hit("volatile", Object.keys(a?.volatiles ?? {}).length > 0);
        }
        hit("fainted_last_turn", side.fainted_last_turn);
        hit("fainted_this_turn", side.fainted_this_turn);
        hit("total_fainted >= 2", side.total_fainted >= 2);
        hit("side condition", Object.keys(side.conditions).length > 0);
        hit("layers >= 2", Object.values(side.conditions).some((c) => c.layers >= 2));
        hit("position emptied", Object.values(side.active).some((a) => a === null));
        hit("move used 3 times", Object.values(side.move_uses).some((moves) => Object.values(moves).some((n) => n >= 3)));
        hit("condition source", Object.values(side.conditions).some((c) => c.source !== null));
      }
      hit("weather", r.facts.weather !== null);
      hit("weather source", r.facts.weather?.source != null);
      hit("weather after residual", r.facts.weather?.after_residual === true && r.facts.weather.started_turn > 0);
      hit("field source", Object.values(r.facts.field_conditions).some((c) => c.source !== null));
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
    for (const what of ["active_turns >= 3", "last_move", "protect_streak >= 2", "toxic_ticks >= 1", "fainted_last_turn", "fainted_this_turn", "side condition", "weather", "terrain", "field condition", "position emptied", "volatile", "move_actions >= 3", "move used 3 times", "condition source", "weather source"]) {
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
