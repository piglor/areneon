// The engine-sent team sheet as hard evidence in the belief (ADR-025 erratum E1, #1155): in an open-sheet format the opponent
// half of the reader's state carries `sheet`, the team as built. The item, ability, exact move set and nature of every slot
// become constraints; the Stat Points and the brought four stay open. The belief of a closed channel (no `sheet` key) keeps its
// bytes: no `sheet` key appears in it.

import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import type { PublicSide, PublicSheetMon } from "@areneon/public-state";
import { toId } from "@areneon/public-state";
import { BELIEF_FORMAT, BeliefError, type BeliefErrorCode, type OpponentBelief, type SetLegality, type SpreadSampler, SHEET_DISAGREES, SHEET_FIELDS, belief, createSampler, worldsDigest } from "../src/index.ts";
import { natureSpreads } from "../src/belief.ts";
import { parseSpreadKey } from "../src/fields.ts";
import { ratio } from "../src/ratio.ts";
import { type Game, type Point, beliefAt, points, type Seat } from "./corpus.ts";
import { ROOT, realVocabulary, uniformPrior } from "./real.ts";
import { hidingConfig, input, prior, seen, text, vocabulary } from "./support.ts";

const SW = (letter: "a" | "b", name: string, species: string) => `|switch|p2${letter}: ${name}|${species}, L50, M|100/100`;
const p = (num: bigint, den = 1n) => ratio(num, den);
const slot = (b: OpponentBelief, species: string) => b.slots.find((s) => s.species === species)!;
const caught = (run: () => unknown): BeliefError => {
  try {
    run();
  } catch (e) {
    if (e instanceof BeliefError) return e;
    throw e;
  }
  throw new Error("expected a BeliefError");
};
const code = (run: () => unknown): BeliefErrorCode => caught(run).code;

// The sheet of the default team, as the engine prints it: the species name as NAME, no nickname, Stat Points, IVs or shiny flag.
const BUILT: Record<string, [item: string, ability: string, moves: string, nature: string]> = {
  alpha: ["Ix", "A1", "Ma,Mb,Mc,Md", "Hardy"],
  beta: ["", "B1", "Bm1,Bm2,Bm3,Bm4", "Hardy"],
  gamma: ["Iy", "G1", "Gm1,Gm2,Gm3,Gm4", "Hardy"],
  delta: ["", "D1", "Dm1,Dm2,Dm3,Dm4", "Hardy"],
  epsilon: ["", "Ill", "Em1,Em2,Em3,Em4", "Hardy"],
  zeta: ["", "Z1", "Zm1,Zm2,Zm3,Zm4", "Hardy"],
};
const NAME: Record<string, string> = { alpha: "Alpha", beta: "Beta", gamma: "Gamma", delta: "Delta", epsilon: "Epsilon", zeta: "Zeta" };
const ORDER = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"];

/** The `|showteam|` line of the opponent's sheet; `change` edits a Pokemon's [item, ability, moves, nature]. */
function showteam(change: Record<string, (string | undefined)[]> = {}): string {
  const packed = ORDER.map((id) => {
    const [item, ability, moves, nature] = BUILT[id]!.map((v, k) => change[id]?.[k] ?? v);
    return `${NAME[id]}||${item}|${ability}|${moves}|${nature}||M|||50|`;
  }).join("]");
  return `|showteam|p2|${packed}`;
}

/** The belief over the opponent's lines after the sheet line. */
const run = (more: readonly string[] = [], change: Record<string, (string | undefined)[]> = {}, extra: Parameters<typeof input>[1] = {}) => belief(input([showteam(change), ...more], extra));

describe("the sheet is read into the reader's state, and into the belief", () => {
  it("the reader attaches the side's sheet, and a closed channel has none", () => {
    expect(seen([showteam()]).opponent.sheet).toHaveLength(6);
    expect(seen().opponent.sheet).toBeUndefined();
    expect(Object.keys(seen([showteam()]).opponent.sheet![0]!).sort()).toEqual([...SHEET_FIELDS].sort());
  });

  it("a belief without a sheet has no `sheet` key anywhere, and the format string is the one before this change", () => {
    const b = belief(input());
    expect(b.format).toBe(BELIEF_FORMAT);
    expect(b.slots.some((s) => "sheet" in s)).toBe(false);
    expect(b.assignments.some((a) => a.slots.some((s) => "sheet" in s))).toBe(false);
    expect(text(b)).not.toContain('"sheet"');
  });

  it("item, ability, the exact move set and the nature are fixed; the sheet is on the slot", () => {
    const b = run();
    const a = slot(b, "alpha");
    expect(a.items).toEqual([{ id: "ix", p: p(1n) }]);
    expect(a.abilities).toEqual([{ id: "a1", p: p(1n) }]);
    expect(a.moves.map((m) => m.id)).toEqual(["ma", "mb", "mc", "md"]);
    expect(a.moves.every((m) => m.p.num === m.p.den)).toBe(true);
    expect(a.spreads).toEqual([{ id: "hardy:0-0-0-0-0-0", p: p(1n) }]);
    expect(a.sheet).toEqual({ item: "ix", ability: "a1", moves: ["ma", "mb", "mc", "md"], nature: "hardy", gender: "M", level: 50 });
    expect(a.constraints).toEqual(expect.arrayContaining(["sheet:item:ix", "sheet:ability:a1", "sheet:moves:4", "sheet:nature:hardy"]));
    expect(slot(b, "gamma").items).toEqual([{ id: "iy", p: p(1n) }]);
    expect(slot(b, "delta").items).toEqual([{ id: null, p: p(1n) }]);
    expect(slot(b, "delta").constraints).toContain("sheet:item:none");
    // the brought four stays open
    for (const s of b.slots) expect(s.p_brought).toEqual(p(2n, 3n));
    expect(b.brought).toEqual({ picked: 4, settled: [], subsets: 15n });
  });

  it("a shorter move list is exact too: the others have probability zero", () => {
    const b = run([], { alpha: [undefined, undefined, "Ma,Mc"] });
    expect(slot(b, "alpha").moves).toEqual([{ id: "ma", p: p(1n) }, { id: "mc", p: p(1n) }]);
    expect(b.assignments[0]!.slots[0]!.formes[0]!.moves.outcomes.map((o) => o.size)).toEqual([2]);
  });

  it("the Item Clause applies unchanged: two slots built with one item leave no world", () => {
    expect(code(() => run([], { gamma: ["Ix"] }))).toBe("evidence_has_no_support");
    expect(run([], { gamma: [""] }).assignments[0]!.normaliser).toBeGreaterThan(0n);
  });

  it("the sheet changes only the slots it constrains differently: a belief with and without it agree on what the sheet fixes", () => {
    const withSheet = run();
    const without = belief(input());
    // the sheet tightens: every marginal of the sheet-less belief that has the sheet's value includes it
    for (const id of ORDER) {
      const s = slot(withSheet, id);
      const w = slot(without, id);
      expect(w.items.map((m) => m.id)).toContain(s.items[0]!.id);
      expect(w.abilities.map((m) => m.id)).toContain(s.abilities[0]!.id);
    }
  });
});

describe("the forme: a Mega stone on the sheet", () => {
  const stone = { beta: ["Betite", "B2"] as (string | undefined)[] };

  it("the stone holder is the Mega forme before the Evolution, with the ability it was built with", () => {
    const b = run([], stone);
    const s = slot(b, "beta");
    expect(s.formes.map((f) => f.id)).toEqual(["betamega"]);
    expect(s.items).toEqual([{ id: "betite", p: p(1n) }]);
    expect(s.abilities).toEqual([{ id: "b2", p: p(1n) }]);
  });

  it("without the stone it is the base forme, with the ability it was built with", () => {
    const s = slot(run(), "beta");
    expect(s.formes.map((f) => f.id)).toEqual(["beta"]);
    expect(s.abilities).toEqual([{ id: "b1", p: p(1n) }]);
  });

  it("after the Evolution the Mega forme's own ability applies, and the stone must be the sheet's", () => {
    const evolve = [SW("a", "Bet", "Beta"), "|detailschange|p2a: Bet|Beta-Mega, L50, M", "|-mega|p2a: Bet|Beta|Betite"];
    const b = run(evolve, stone);
    expect(slot(b, "beta").abilities).toEqual([{ id: "bmega", p: p(1n) }]);
    expect(slot(b, "beta").settled).toBe(true);
    // a stone other than the built one disagrees
    const wrong = caught(() => run(evolve, { beta: ["Ix", "B1"] }));
    expect(wrong).toMatchObject({ code: "evidence_has_no_support", detail: SHEET_DISAGREES });
  });

  it("a built ability the forme's list lacks leaves no forme", () => {
    expect(code(() => run([], { beta: ["Betite", "Bmega"] }))).toBe("evidence_has_no_support");
  });

  it("a sheet item that is a stone of another species is a plain item of this one when the vocabulary allows it", () => {
    const wide = prior((pr) => {
      pr.species[0]!.formes[0]!.items = { denominator: 10, entries: [{ id: "ix", weight: 5 }], no_item: 3, residual: 2, unmapped: 0 };
    });
    const b = run([], { alpha: ["Betite"] }, { prior: wide });
    expect(slot(b, "alpha").items).toEqual([{ id: "betite", p: p(1n) }]);
    expect(slot(b, "alpha").formes.map((f) => f.id)).toEqual(["alpha"]);
  });
});

describe("the battle must agree with the sheet, and live changes never alter it", () => {
  const out = [SW("a", "Alp", "Alpha")];
  const disagreement = (lines: string[], change: Record<string, (string | undefined)[]> = {}) => {
    const e = caught(() => run([...out, ...lines], change));
    expect(e.code).toBe("evidence_has_no_support");
    expect(e.detail).toBe(SHEET_DISAGREES);
    expect(e.slot).toBe(0);
  };

  it("a revealed item, ability, move, gender or level that the sheet does not hold is a typed refusal with the sheet detail", () => {
    disagreement(["|-item|p2a: Alp|Iy"]);
    disagreement(["|-ability|p2a: Alp|A2"]);
    disagreement(["|move|p2a: Alp|Md"], { alpha: [undefined, undefined, "Ma,Mb,Mc"] });
  });

  it("the revealed gender and level are checked against the sheet", () => {
    const base = seen([showteam(), SW("a", "Alp", "Alpha")]).opponent;
    const edit = (change: (m: PublicSide["roster"][number]) => void): PublicSide => {
      const copy = structuredClone(base);
      change(copy.roster[0]!);
      return copy;
    };
    const fed = (opponent: PublicSide) => () => belief({ ...input(), opponent });
    expect(belief({ ...input(), opponent: base }).slots[0]!.settled).toBe(true);
    expect(caught(fed(edit((m) => (m.gender = "F")))).detail).toBe(SHEET_DISAGREES);
    expect(caught(fed(edit((m) => (m.level = 51)))).detail).toBe(SHEET_DISAGREES);
  });

  it("a consumed or knocked-off item, a received item or ability and a copied move are live state: no disagreement, same belief", () => {
    const quiet = text(run(out));
    expect(text(run([...out, "|-item|p2a: Alp|Ix", "|-enditem|p2a: Alp|Ix"])) === quiet).toBe(true); // eaten: no longer a reveal
    expect(() => run([...out, "|-item|p2a: Alp|Ix"])).not.toThrow(); // revealed Ix is the sheet's
    expect(() => run([...out, "|-item|p2a: Alp|Iy|[from] move: Trick"])).not.toThrow(); // received, not its own
    expect(text(run([...out, "|-item|p2a: Alp|Iy|[from] move: Trick"])) === quiet).toBe(true);
    expect(text(run([...out, "|-ability|p2a: Alp|A2|[from] ability: Trace"])) === quiet).toBe(true);
    expect(text(run([...out, "|-damage|p2a: Alp|40/100"])) === quiet).toBe(true);
    expect(text(run([...out, "|-status|p2a: Alp|par", "|-boost|p2a: Alp|atk|1"])) === quiet).toBe(true);
    expect(slot(run([...out, "|-enditem|p2a: Alp|Ix"]), "alpha").items).toEqual([{ id: "ix", p: p(1n) }]); // Knock Off does not change what it was built with
  });

  it("a move no pool holds (Struggle) is no disagreement", () => {
    expect(() => run([...out, "|move|p2a: Alp|Struggle"])).not.toThrow();
  });

  it("revealed moves that the sheet holds stay in the notes", () => {
    expect(slot(run([...out, "|move|p2a: Alp|Ma"]), "alpha").constraints).toContain("move:ma");
  });
});

describe("the sheet's natures and the other-spread entry", () => {
  const dist = { total: 10n, options: [{ id: "hardy:0-0-0-0-0-0", weight: 6n }, { id: "other", weight: 4n }] };

  it("keeps the named spreads of the nature and the 'other' entry, with the probability of the nature", () => {
    expect(natureSpreads(dist, "hardy", 1)).toEqual({ dist, share: p(1n) });
    const jolly = natureSpreads(dist, "jolly", 2);
    expect(jolly.dist).toEqual({ total: 4n, options: [{ id: "other", weight: 4n }] });
    expect(jolly.share).toEqual(p(1n, 5n)); // 'other' holds one of two natures: 4 / (10 * 2)
    expect(natureSpreads(dist, "hardy", 2).share).toEqual(p(4n, 5n)); // (6 * 2 + 4) / 20
    expect(natureSpreads({ total: 6n, options: [dist.options[0]!] }, "jolly", 2)).toEqual({ dist: { total: 0n, options: [] }, share: p(0n) });
  });

  it("a sheet nature the prior's named spreads lack and no residual covers leaves no forme", () => {
    expect(code(() => run([], { alpha: [undefined, undefined, undefined, "Jolly"] }))).toBe("evidence_has_no_support");
  });

  it("with a residual the nature is kept and the 'other' entry carries it", () => {
    const loose = prior((pr) => {
      pr.species[0]!.formes[0]!.spreads = { denominator: 10, entries: [{ nature: "hardy", stat_points: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, weight: 6 }], residual: 4, unmapped: 0 };
    });
    const b = run([], { alpha: [undefined, undefined, undefined, "Jolly"] }, { prior: loose, vocabulary: { ...vocabulary(), natures: [{ id: "hardy", name: "Hardy" }, { id: "jolly", name: "Jolly" }] as never } });
    expect(slot(b, "alpha").spreads).toEqual([{ id: "other", p: p(1n) }]);
    expect(slot(b, "alpha").sheet?.nature).toBe("jolly");
  });

  it("a nature, item or species the format or the preview does not have is refused", () => {
    expect(code(() => run([], { alpha: [undefined, undefined, undefined, "Nonesuch"] }, { vocabulary: { ...vocabulary(), natures: [{ id: "hardy", name: "Hardy" }] as never } }))).toBe("evidence_has_no_support");
    expect(code(() => run([], { alpha: ["Nonesuch"] }))).toBe("evidence_has_no_support");
    const edited = structuredClone(seen([showteam()]).opponent);
    edited.sheet![0]!.species = "zeta";
    expect(caught(() => belief({ ...input(), opponent: edited })).detail).toBe(SHEET_DISAGREES);
    edited.sheet = edited.sheet!.slice(1);
    expect(caught(() => belief({ ...input(), opponent: edited })).detail).toBe(SHEET_DISAGREES);
  });
});

describe("the disguise mixture with a sheet", () => {
  // epsilon may build with a hiding ability or not; the sheet says which
  const wide = () =>
    prior((pr) => {
      const f = pr.species.find((s) => s.id === "epsilon")!.formes[0]!;
      f.abilities = { denominator: 2, entries: [{ id: "ill", weight: 1 }], residual: 1, unmapped: 0 };
    });
  const vocab = () => {
    const v = vocabulary();
    return { ...v, roster: v.roster.map((r) => (r.id === "epsilon" ? { ...r, abilities: ["e2", "ill"] as [string, ...string[]] } : r)) };
  };
  const shown = [SW("a", "Alp", "Alpha")];

  it("builds as Illusion: the position may be this member; builds without it: it cannot be", () => {
    const hiding = belief(input([showteam({ epsilon: [undefined, "Ill"] }), ...shown], { config: hidingConfig, prior: wide(), vocabulary: vocab() }));
    const plain = belief(input([showteam({ epsilon: [undefined, "E2"] }), ...shown], { config: hidingConfig, prior: wide(), vocabulary: vocab() }));
    expect(hiding.assignments.length).toBeGreaterThan(1);
    expect(hiding.disguise.hiding).toEqual([4]);
    expect(plain.assignments).toHaveLength(1);
    expect(plain.disguise.hiding).toEqual([]);
  });

  it("a revealed fact that the assigned slot's sheet cannot hold drops the assignment", () => {
    const b = belief(input([showteam({ epsilon: [undefined, "Ill"] }), ...shown, "|move|p2a: Alp|Mb"], { config: hidingConfig, prior: wide(), vocabulary: vocab() }));
    expect(b.assignments.every((a) => a.moved.length === 0)).toBe(true);
  });
});

describe("the sampler draws the sheet", () => {
  const accept: SetLegality = { legal: () => true };
  const stub: SpreadSampler = { other: () => ({ nature: "adamant", stat_points: { hp: 1, atk: 2, def: 3, spa: 4, spd: 5, spe: 6 } }) };
  const sampler = createSampler({ legality: accept, spreads: stub });
  const STREAM = Uint8Array.from({ length: 32 }, (_, i) => i * 5 + 3);
  const loose = prior((pr) => {
    pr.species[0]!.formes[0]!.spreads = { denominator: 4, entries: [], residual: 4, unmapped: 0 };
  });

  it("every drawn world's item, ability, moves and nature equal the sheet, and its Stat Points come from the port", () => {
    const b = run([], { alpha: [undefined, undefined, undefined, "Jolly"] }, { prior: loose });
    const { worlds } = sampler.sampleWorlds(b, STREAM, 12);
    let seenAlpha = 0;
    for (const w of worlds) {
      for (const s of w.world.slots) {
        if (s.set === null) continue;
        const built = BUILT[s.set.species]!;
        expect(s.set.item).toBe(built[0] === "" ? null : toId(built[0]));
        expect(s.set.ability).toBe(toId(built[1]));
        expect(s.set.moves).toEqual(built[2].split(",").map(toId).sort());
        expect(s.set.nature).toBe(s.set.species === "alpha" ? "jolly" : "hardy");
        expect(s.set.gender).toBe("M");
        if (s.set.species === "alpha") {
          seenAlpha += 1;
          expect(s.set.stat_points).toEqual({ hp: 1, atk: 2, def: 3, spa: 4, spd: 5, spe: 6 });
        }
      }
    }
    expect(seenAlpha).toBeGreaterThan(0);
    // whole teams too
    const team = sampler.sampleTeam(b, STREAM).team;
    expect(team.sets.map((s) => s.nature)).toEqual(["jolly", "hardy", "hardy", "hardy", "hardy", "hardy"]);
  });

  it("is deterministic, and a sheet-less belief draws exactly as before", () => {
    const b = run();
    expect(worldsDigest(sampler.sampleWorlds(b, STREAM, 3))).toBe(worldsDigest(sampler.sampleWorlds(b, STREAM, 3)));
    const plain = belief(input());
    expect(sampler.sampleWorlds(plain, STREAM, 2).worlds[0]!.world.slots.some((s) => s.set?.gender !== null && s.set !== null)).toBe(true); // the preview shows M
  });

  it("a sheet gender and level fill what the channel does not show", () => {
    const b = run();
    const edited = structuredClone(b);
    (edited.slots[0] as { shown: { level: number | null; gender: string | null } }).shown = { ...edited.slots[0]!.shown, level: null, gender: null };
    for (const a of edited.assignments) (a.slots[0] as { shown: unknown }).shown = edited.slots[0]!.shown;
    const world = sampler.sampleWorlds(edited, STREAM, 4).worlds.flatMap((w) => w.world.slots).find((s) => s.index === 0 && s.set !== null)!;
    expect(world.set).toMatchObject({ level: 50, gender: "M" });
  });
});

describe("the committed open-regime games", () => {
  const dir = join(ROOT, "fixtures", "showdown", "gen9championsvgc2026regmcbo3", "open-regime");
  const files = readdirSync(dir).filter((f) => /^o\d+\.json$/.test(f)).sort();
  const games: Game[] = files.map((f) => {
    const file = JSON.parse(readFileSync(join(dir, f), "utf8")) as { teams: Record<Seat, Game["teams"][Seat]>; channels: Record<Seat, { lines: string[] }> };
    return { id: f.slice(0, -5), teams: file.teams, channels: { p1: file.channels.p1.lines, p2: file.channels.p2.lines } };
  });
  const all = games.flatMap((g) => points(g));
  const withTruth = all.filter((pt): pt is Point & { truth: NonNullable<Point["truth"]> } => pt.truth !== null && pt.flags.length === 0);

  it("hold a sheet for each opponent at every request, in both seats", () => {
    expect(games.length).toBeGreaterThanOrEqual(8);
    expect(withTruth.length).toBeGreaterThan(100);
    for (const pt of all) expect(pt.state.opponent.sheet, `${pt.game} ${pt.seat}`).toHaveLength(6);
  });

  it("soundness: the true team satisfies every constraint at every request, both seats, and the sheet is the team as built", () => {
    const priors = new Map<string, ReturnType<typeof uniformPrior>>();
    let judged = 0;
    for (const pt of withTruth) {
      const ids = pt.state.opponent.roster.map((m) => m.base_species);
      const key = ids.join(",");
      if (!priors.has(key)) priors.set(key, uniformPrior(ids));
      const b = beliefAt(pt, priors.get(key)!);
      pt.truth.team.forEach((set, i) => {
        const s = b.slots[i]!;
        const sheet = s.sheet!;
        expect(sheet.item, `${pt.game} ${pt.seat} ${pt.request} slot ${i}`).toBe(set.item === "" ? null : toId(set.item));
        expect(sheet.ability).toBe(toId(set.ability));
        expect(sheet.moves).toEqual(set.moves.map(toId).sort());
        expect(sheet.nature).toBe(toId(set.nature));
        // the belief holds exactly the sheet's value
        expect(s.items).toEqual([{ id: sheet.item, p: p(1n) }]);
        expect(s.moves.map((m) => m.id)).toEqual(sheet.moves);
        if (!pt.state.opponent.roster[i]!.mega) expect(s.abilities.map((m) => m.id)).toEqual([sheet.ability]);
        judged += 1;
      });
    }
    expect(judged).toBeGreaterThan(600);
  });

  it("sampled worlds equal the sheet in item, ability, moves and nature, at every request", () => {
    const accept: SetLegality = { legal: () => true };
    const spreads: SpreadSampler = { other: (_s, _f, below) => ({ nature: realVocabulary.natures[Number(below(BigInt(realVocabulary.natures.length)))]!.id, stat_points: { hp: Number(below(33n)), atk: 0, def: 0, spa: 0, spd: 0, spe: Number(below(2n)) } }) };
    const sampler = createSampler({ legality: accept, spreads });
    const stream = Uint8Array.from({ length: 32 }, (_, i) => i + 9);
    let checked = 0;
    for (const pt of withTruth.filter((_, k) => k % 7 === 0)) {
      const b = beliefAt(pt, uniformPrior(pt.state.opponent.roster.map((m) => m.base_species)));
      for (const w of sampler.sampleWorlds(b, stream, 2).worlds) {
        for (const s of w.world.slots) {
          if (s.set === null) continue;
          const sheet = b.slots[s.index]!.sheet!;
          expect(s.set.item).toBe(sheet.item);
          expect(s.set.moves).toEqual(sheet.moves);
          expect(s.set.nature).toBe(sheet.nature);
          if (!pt.state.opponent.roster[s.index]!.mega) expect(s.set.ability).toBe(sheet.ability);
          checked += 1;
        }
      }
    }
    expect(checked).toBeGreaterThan(50);
  });
});

describe("the allowlist of the sheet", () => {
  it("the belief reads the sheet's fields and nothing else of it", () => {
    const source = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "src", "belief.ts"), "utf8");
    const read = new Set([...source.matchAll(/\bsheet\??\.([a-z_]+)\b/g)].map((m) => m[1]));
    for (const field of read) if (field !== "length" && field !== "moves") expect(SHEET_FIELDS as readonly string[]).toContain(field);
    expect([...read].filter((f) => f !== "length").every((f) => f !== undefined && (SHEET_FIELDS as readonly string[]).includes(f))).toBe(true);
    const sheet: PublicSheetMon = seen([showteam()]).opponent.sheet![0]!;
    expect(Object.keys(sheet).sort()).toEqual([...SHEET_FIELDS].sort());
    expect(parseSpreadKey("hardy:0-0-0-0-0-0")?.nature).toBe("hardy");
  });
});
