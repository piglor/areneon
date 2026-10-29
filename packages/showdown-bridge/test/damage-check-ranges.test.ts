// A candidate whose maximum hit points cannot show the observed percentage cannot be the one (ticket #1132): the display scan finds no
// value, which no committed roster entry can make happen, so the scan is replaced here.
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";

vi.mock("../src/hp-display.ts", async (original) => ({ ...(await original<typeof import("../src/hp-display.ts")>()), hpRange: () => null }));

const { readBeliefConfig } = await import("../src/belief-config.ts");
const { loadDamageAwareConfig } = await import("../src/damage-aware-config.ts");
const { createDamageCheck } = await import("../src/damage-check.ts");
const { GOLDEN_FORMAT } = await import("../src/golden.ts");
const { loadEngine } = await import("../src/index.ts");
const { loadMechanicsConfig } = await import("../src/mechanics-config.ts");

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const FIXTURES = join(ROOT, "fixtures", "showdown", GOLDEN_FORMAT);
const vocabulary = JSON.parse(readFileSync(join(FIXTURES, "team-vocabulary.json"), "utf8"));
const belief = readBeliefConfig(JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", GOLDEN_FORMAT, "belief.damage-v1.json"), "utf8")), vocabulary);
const lines = (JSON.parse(readFileSync(join(FIXTURES, "channels", "p2.json"), "utf8")) as { channel: { lines: string[] } }).channel.lines;
const team = JSON.parse(readFileSync(join(FIXTURES, "synthetic-team.json"), "utf8"));

describe("a percentage no hit-point value of the candidate shows", () => {
  it("rules the candidate out, as attacker and as defender", () => {
    const port = createDamageCheck({ engine: loadEngine(), vocabulary, config: loadDamageAwareConfig("v1", vocabulary), mechanics: loadMechanicsConfig(GOLDEN_FORMAT), belief, data_class: "synthetic" }, { player: "p2", lines, team }) as NonNullable<ReturnType<typeof createDamageCheck>>;
    const candidate = { species: "charizard", forme: "charizard", mega: false, ability: "blaze", item: "charcoal", nature: "serious", stat_points: { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: 4 } };
    for (const role of ["attacker", "defender"] as const) expect(port.check(port.events.find((e) => e.role === role) as (typeof port.events)[number], candidate)).toBe(false);
  });
});
