// Isolated hits and the reasons a hit is not isolated (ticket #1132, finding N9): every reviewed modifier of the
// committed config makes a hit non-isolated, on hand-written channels.

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { interpretChannel } from "@areneon/public-state";
import { describe, expect, it } from "vitest";
import { readBeliefConfig } from "../src/belief-config.ts";
import { type HitContext, extractHits } from "../src/damage-hits.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import type { TeamVocabulary } from "../src/team-vocabulary.ts";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");

const vocabulary = JSON.parse(readFileSync(join(ROOT, "fixtures", "showdown", GOLDEN_FORMAT, "team-vocabulary.json"), "utf8")) as TeamVocabulary;
const config = readBeliefConfig(JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", GOLDEN_FORMAT, "belief.damage-v1.json"), "utf8")), vocabulary);
const rules = (config.damage_evidence as NonNullable<typeof config.damage_evidence>).non_isolating;

/** The golden battle as the second player saw it: the first player's Charizard uses Flamethrower on the second player's Blastoise in turn 3. */
const golden = (JSON.parse(readFileSync(join(ROOT, "fixtures", "showdown", GOLDEN_FORMAT, "channels", "p2.json"), "utf8")) as { channel: { lines: string[] } }).channel.lines;
const at = golden.findIndex((l) => l === "|move|p1b: Charizard|Flamethrower|p2a: Blastoise");
const HEAD = golden.slice(0, at);
const HIT = golden.slice(at, at + 3);
const ATT = "p1b: Charizard";
const DEF = "p2a: Blastoise";
const DMG = "|-damage|p2a: Blastoise|124/158";
const use = (who: string, move: string, target: string, ...tags: string[]): string => ["", "move", who, move, target, ...tags].join("|");
const cap = (id: string): string => id.charAt(0).toUpperCase() + id.slice(1);

const stub = (over: Partial<HitContext> = {}): HitContext => ({
  rules,
  targetOf: () => "normal",
  abilitiesOf: () => [],
  mayHide: () => false,
  declarable: () => true,
  ...over,
});
/** The hits of the lines added after the golden battle's first turns. */
const hits = (lines: readonly string[], ctx: HitContext = stub()) => extractHits(interpretChannel("p2", [...HEAD, ...lines]).frames, ctx).filter((h) => h.frame >= HEAD.length);
const reasonOf = (lines: readonly string[], ctx?: HitContext) => hits(lines, ctx).at(-1)?.reason;

describe("an isolated hit", () => {
  it("is one move into one foe with one damage line, and carries the shown hit points and the crit line", () => {
    const [hit] = hits(HIT);
    expect(hit).toMatchObject({ move: "flamethrower", crit: false, reason: null, before: { num: 158, den: 158 }, after: { num: 124, den: 158 } });
    expect(hit?.attacker).toMatchObject({ own: false, letter: "b" });
    expect(hit?.defender).toMatchObject({ own: true, letter: "a" });
    expect(hits([HIT[0] as string, `|-crit|${DEF}`, HIT[1] as string, DMG])[0]).toMatchObject({ crit: true, reason: null });
  });

  it("a move with no damage line is no hit", () => {
    expect(hits([HIT[0] as string, `|-miss|${ATT}|${DEF}`])).toEqual([]);
  });
});

describe("a hit is not isolated when", () => {
  it.each([
    ["the move line carries a tag ([spread])", [use(ATT, "Surf", DEF, "[spread] p2a,p2b"), DMG], "move_tags"],
    ["the move line names no target", [use(ATT, "Flamethrower", ""), DMG], "no_target"],
    ["it hits a Pokemon of its own side", [use("p2b: Charizard", "Earthquake", DEF), "|-damage|p2a: Blastoise|100/158"], "same_side"],
    ["it has two damage lines (several targets or hits)", [...HIT, "|-damage|p2b: Charizard|120/157"], "damage_count"],
    ["the damage is from an effect ([from])", [HIT[0] as string, `${DMG}|[from] item: Rocky Helmet`], "damage_tag"],
    ["another effect shows in it (-activate)", [...HIT, `|-activate|${DEF}|ability: Static`], "extra_line"],
    ["it carries two crit lines", [HIT[0] as string, `|-crit|${DEF}`, `|-crit|${DEF}`, DMG], "extra_line"],
    ["a volatile is on a combatant", [`|-start|${DEF}|confusion`, ...HIT], "volatile"],
    ["an opponent's item came from another Pokemon", [`|-item|${ATT}|Leftovers|[from] move: Trick|[of] ${DEF}`, ...HIT], "changed_mon"],
  ])("%s", (_, lines, reason) => {
    expect(reasonOf(lines as string[])).toBe(reason);
  });
});

describe("every reviewed modifier of the committed config makes a hit non-isolated", () => {
  it.each(rules.moves)("the move %s used earlier this turn, or the hit itself", (move) => {
    expect(reasonOf([use("p2b: Charizard", cap(move), DEF), ...HIT])).toBe("listed_move");
    expect(reasonOf([use(ATT, cap(move), DEF), DMG])).toBe("listed_move");
    // a turn later it no longer is, for a move whose effect lasts a turn
    expect(reasonOf([use("p2b: Charizard", cap(move), DEF), "|turn|9", ...HIT])).toBeNull();
  });

  it.each(rules.lasting_moves)("the lasting move %s used earlier in the channel, in any turn", (move) => {
    expect(reasonOf([use("p2b: Charizard", cap(move), DEF), "|turn|9", ...HIT])).toBe("lasting_move");
  });

  it.each(rules.abilities)("the ability %s held by a Pokemon in play besides the two combatants", (ability) => {
    expect(reasonOf(HIT, stub({ abilitiesOf: (mon) => (mon.name === "Charizard" && mon.own ? [ability] : []) }))).toBe("listed_ability");
    // the same ability on a combatant is a candidate's field, not a bystander's
    expect(reasonOf(HIT, stub({ abilitiesOf: (mon) => (mon.name === "Blastoise" || (mon.name === "Charizard" && !mon.own) ? [ability] : []) }))).toBeNull();
  });

  it.each(rules.targets)("a move whose target type is %s", (target) => {
    expect(reasonOf(HIT, stub({ targetOf: () => target }))).toBe("listed_target");
    expect(reasonOf(HIT, stub({ targetOf: () => null }))).toBeNull();
  });

  it("an unrevealed identity-hiding member of the preview, and a field the probe cannot state", () => {
    expect(reasonOf(HIT, stub({ mayHide: (mon) => mon.base_species === "gengar" }))).toBe("possible_disguise");
    expect(reasonOf(HIT, stub({ declarable: () => false }))).toBe("undeclared_field");
  });
});
