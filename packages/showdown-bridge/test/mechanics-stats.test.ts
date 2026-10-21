import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import { beforeAll, describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import type { Engine, PokemonSet, StatTable } from "../src/engine.ts";
import { GOLDEN_FORMAT, syntheticTeam, TEAMS_DIR } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig, mechanicsConfigHash } from "../src/mechanics-config.ts";
import { type MechanicsContext, probeMechanics } from "../src/mechanics-probe.ts";
import { readMechanicsQuery, type ParsedMechanicsQuery } from "../src/mechanics-query.ts";
import { readMechanicsResult } from "../src/mechanics-result.ts";
import { exportPaste, importPaste } from "../src/teams.ts";

// ADR-016 section 3, the required test: for every fixture set the probe's six
// stats equal the stats of the target format's own Team Preview request (the
// champions-stats.test.ts seam), so the scenario format reads what the target
// format shows. The five #680 reference cases are also pinned to the values the
// game displays.

const ZERO: StatTable = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const IVS: StatTable = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };

let engine: Engine;
let config: MechanicsConfig;
let context: MechanicsContext;
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
  context = { engine, config };
});

const statsQuery = (paste: string, data_class: "synthetic" | "private" = "synthetic"): ParsedMechanicsQuery =>
  readMechanicsQuery({ schema_version: 1, data_class, format_id: GOLDEN_FORMAT, query: { kind: "stats", set: paste } });

/** The probe's six stats of one set. */
function probeStats(set: PokemonSet): StatTable {
  const { result } = probeMechanics(context, statsQuery(exportPaste(engine, [set])));
  if (result.kind !== "computed" || result.answer.kind !== "stats") throw new Error(`not computed: ${JSON.stringify(result)}`);
  return result.answer.stats;
}

/** The target format's Team Preview request stats for p1's team, set by set. */
function previewStats(sets: PokemonSet[]): StatTable[] {
  const battle = new engine.sim.Battle({ formatid: GOLDEN_FORMAT, seed: "sodium,00000000000000000000000000000000" });
  battle.setPlayer("p1", { name: "p1", team: engine.sim.Teams.pack(sets) });
  battle.setPlayer("p2", { name: "p2", team: engine.sim.Teams.pack(sets) });
  const request = battle.sides.find((side) => side.id === "p1")?.activeRequest as EngineRequest;
  return request.side.pokemon.map((pokemon) => ({ hp: Number(pokemon.condition.split("/")[1]), ...pokemon.stats }));
}

const jsonl = (file: string): { paste: string }[] => readFileSync(file, "utf8").trim().split("\n").map((line) => JSON.parse(line) as { paste: string });
const DATASET = join(TEAMS_DIR, "datasets", "fixtures-v1");

describe("the probe's stats equal the target format's Team Preview request stats", () => {
  const teams: [string, PokemonSet[]][] = [
    ["the golden synthetic team", syntheticTeam()],
    ["legal-synthetic.txt", importPaste(loadEngine(), readFileSync(join(TEAMS_DIR, "legal-synthetic.txt"), "utf8"))],
    ["legal-opponent.txt", importPaste(loadEngine(), readFileSync(join(TEAMS_DIR, "legal-opponent.txt"), "utf8"))],
    ...["teams-dev.jsonl", "teams-validation.jsonl"].flatMap((file) =>
      jsonl(join(DATASET, file)).map((record, index): [string, PokemonSet[]] => [`${file} #${index}`, importPaste(loadEngine(), record.paste)]),
    ),
  ];

  it("holds for every set of every committed fixture team", () => {
    expect(teams.length).toBeGreaterThan(200);
    for (const [name, sets] of teams) {
      const shown = previewStats(sets);
      sets.forEach((set, index) => expect(probeStats(set), `${name} set ${index}`).toEqual(shown[index]));
    }
  });
});

describe("the five reference cases of the research ticket, against the values the game displays", () => {
  // species, ability, nature, Stat Points, displayed stats (docs/research/stat-formula-and-turn-order-evidence.md)
  const CASES: [string, string, string, Partial<StatTable>, StatTable][] = [
    ["Empoleon", "Torrent", "Modest", { hp: 32, spa: 32, spe: 2 }, { hp: 191, atk: 95, def: 108, spa: 179, spd: 121, spe: 82 }],
    ["Whimsicott", "Prankster", "Bold", { hp: 32, def: 32, spe: 2 }, { hp: 167, atk: 78, def: 150, spa: 97, spd: 95, spe: 138 }],
    ["Pikachu", "Static", "Jolly", { atk: 32, def: 2, spe: 32 }, { hp: 110, atk: 107, def: 62, spa: 63, spd: 70, spe: 156 }],
    ["Gardevoir", "Synchronize", "Timid", { spa: 32, spd: 2, spe: 32 }, { hp: 143, atk: 76, def: 85, spa: 177, spd: 137, spe: 145 }],
    ["Arcanine", "Intimidate", "Relaxed", { atk: 2, def: 32, spa: 32 }, { hp: 165, atk: 132, def: 145, spa: 152, spd: 100, spe: 103 }],
  ];
  it.each(CASES)("%s", (species, ability, nature, sp, displayed) => {
    const set: PokemonSet = { name: species, species, item: "", ability, moves: ["protect"], nature, gender: "", evs: { ...ZERO, ...sp }, ivs: IVS, level: 50 };
    expect(probeStats(set)).toEqual(displayed);
    expect(previewStats([set])[0]).toEqual(displayed);
  });
});

describe("golden results are byte-identical on a rerun", () => {
  const paste = (): string => exportPaste(engine, [syntheticTeam()[0] as PokemonSet]);
  it("gives the same JCS bytes twice and the committed corpus sample", () => {
    const run = (): string => jcs(probeMechanics(context, statsQuery(paste())).result as unknown as Json);
    expect(run()).toBe(run());
    const sample = JSON.parse(readFileSync(new URL("../../../fixtures/contracts/v1/mechanics-result/valid/computed-stats.json", import.meta.url), "utf8")) as unknown;
    expect(readMechanicsResult(sample).mechanics_config_hash).toBe(mechanicsConfigHash(config));
  });

  it("names the config, the manifest, the query and the data class of the query", () => {
    const query = statsQuery(paste(), "private");
    const { result } = probeMechanics(context, query);
    expect(result).toMatchObject({ kind: "computed", data_class: "private", mechanics_config_hash: mechanicsConfigHash(config), gap_ids: ["level", "stat-points"] });
    expect(result.manifest.format_id).toBe(GOLDEN_FORMAT);
    expect(result.query_hash).toMatch(/^[0-9a-f]{64}$/);
  });
});
