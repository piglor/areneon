/* eslint-disable @typescript-eslint/no-explicit-any */
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import type { Engine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig } from "../src/mechanics-config.ts";
import { loadParityInputs, MECHANICS_FIXTURES_DIR, PARITY_DOC, ParityError, renderMechanicsParity } from "../src/mechanics-parity.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";
import type { RulesSnapshot } from "../src/rules.ts";

// ADR-016 slice 5: the parity report is generated and drift-checked.

let engine: Engine;
let config: MechanicsConfig;
let snapshot: RulesSnapshot;
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
  snapshot = loadRulesSnapshot(engine.commit);
});
const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

/** A scratch copy of the fixtures with `edit` applied to the format's folder. */
function withFixtures(edit: (dir: string) => void): string {
  const root = mkdtempSync(join(tmpdir(), "areneon-parity-test-"));
  dirs.push(root);
  cpSync(MECHANICS_FIXTURES_DIR, root, { recursive: true });
  edit(join(root, GOLDEN_FORMAT));
  return root;
}
const render = (fixturesDir?: string, gaps: RulesSnapshot = snapshot): string => renderMechanicsParity(loadParityInputs(config, gaps, fixturesDir));

describe("the committed report", () => {
  it("equals a fresh render (regenerate with scripts/mechanics-parity.ts)", () => {
    expect(readFileSync(PARITY_DOC, "utf8")).toBe(render());
  });

  it("is deterministic", () => {
    expect(render()).toBe(render());
  });

  it("lists every gap row a result depends on, each with its status", () => {
    const doc = render();
    const used = new Set(config.scope.query_kinds.flatMap((entry) => entry.gap_ids));
    used.add("mega-evolution");
    for (const id of used) expect(doc, id).toContain(`| \`${id}\` | ${(snapshot.gaps[id] as { status: string }).status} |`);
    const unverified = [...used].filter((id) => (snapshot.gaps[id] as { status: string }).status === "unverified");
    expect(unverified.length).toBeGreaterThan(0);
    expect(doc).toContain(`${unverified.length} of ${used.size} rows below are unverified in-game`);
  });

  it("says plainly that the game comparison waits for the captures, and reports the committed calc differential", () => {
    const doc = render();
    expect(doc).toContain("met only for hand-worked values");
    expect(doc).toContain("1 listed divergence ");
    expect(doc).toContain("`damage:always-crit:no_crit`");
    expect(doc).not.toContain("differential not built yet");
    expect(doc).toContain("#937, Wave 9");
    expect(doc).toContain("no claim of parity");
  });

  it("lists the fixture cases with the file hashes and the config's blockers", () => {
    const doc = render();
    for (const id of ["boundary-lowest-roll-equals-hp", "explicit-256-pair-enumeration", "probe-hp-equals-lowest-pair-sum", "single-target", "exact-tie"]) expect(doc).toContain(`| \`${id}\` |`);
    for (const id of [...config.ko_blockers.items, ...config.ko_blockers.abilities]) expect(doc).toContain(`\`${id}\``);
    expect(doc).toMatch(/SHA-256 `[0-9a-f]{64}`, 20 cases/);
  });
});

describe("drift", () => {
  it("a changed fixture changes the report", () => {
    const changed = withFixtures((dir) => {
      const path = join(dir, "ko", "cases.json");
      writeFileSync(path, readFileSync(path, "utf8").replace("Synthetic and CC0.", "Synthetic and CC0. "));
    });
    expect(render(changed)).not.toBe(render());
  });

  it("a changed gap record changes the report", () => {
    const gaps = structuredClone(snapshot);
    (gaps.gaps.paralysis as { status: string }).status = "verified";
    const doc = render(undefined, gaps);
    expect(doc).not.toBe(render());
    expect(doc).toContain("5 of 6 rows below are unverified in-game");
  });

  it("a gap id missing from the ledger is refused", () => {
    const gaps = structuredClone(snapshot);
    delete gaps.gaps["trick-room"];
    expect(() => render(undefined, gaps)).toThrow(ParityError);
  });
});

describe("the calc divergence list", () => {
  const write = (body: unknown): string => withFixtures((dir) => writeFileSync(join(dir, "calc-divergences.json"), JSON.stringify(body)));

  it("says the differential is not built when the divergence file is absent", () => {
    const doc = render(withFixtures((dir) => rmSync(join(dir, "calc-divergences.json"))));
    expect(doc).toContain("differential not built yet (#935)");
  });

  it("is rendered when the differential has written one", () => {
    const doc = render(write({ data_class: "synthetic", divergences: [{ case: "damage/reflect", reason: "the calc | patch table lags" }] }));
    expect(doc).toContain("| `damage/reflect` | the calc \\| patch table lags |");
    expect(doc).not.toContain("differential not built yet");
    expect(doc).toContain("1 listed divergence ");
  });

  it("escapes a backslash before a pipe, so a reason cannot end its cell", () => {
    const doc = render(write({ data_class: "synthetic", divergences: [{ case: "damage/reflect", reason: "ends with \\| here" }] }));
    expect(doc).toContain("| `damage/reflect` | ends with \\\\\\| here |");
  });

  it("an empty list says every case agrees", () => {
    const doc = render(write({ data_class: "synthetic", divergences: [] }));
    expect(doc).toContain("every fixture case agrees");
    expect(doc).toContain("0 listed divergences");
  });

  it("a file of another shape is refused", () => {
    for (const bad of [[], { data_class: "private", divergences: [] }, { data_class: "synthetic", divergences: [{ case: "x" }] }, { data_class: "synthetic", divergences: [1] }]) {
      expect(() => render(write(bad))).toThrow(ParityError);
    }
  });
});

describe("fixture readers", () => {
  it("refuses a fixture file that is not a list of cases with ids and descriptions", () => {
    const rewrite = (file: string, change: (json: any) => void): string =>
      withFixtures((dir) => {
        const path = join(dir, file);
        const json = JSON.parse(readFileSync(path, "utf8"));
        change(json);
        writeFileSync(path, JSON.stringify(json));
      });
    expect(() => render(rewrite("speed/cases.json", (json) => (json.cases = {})))).toThrow("expected a list of cases");
    expect(() => render(rewrite("damage/cases.json", (json) => delete json.cases[0].description))).toThrow("every case needs an id and a description");
    expect(() => render(rewrite("ko/cases.json", (json) => delete json.enumeration.id))).toThrow("the enumeration needs an id");
    expect(() => render(rewrite("ko/cases.json", (json) => (json.vector_cases = [null])))).toThrow(ParityError);
    const notObject = withFixtures((dir) => writeFileSync(join(dir, "speed", "cases.json"), "[]"));
    expect(() => render(notObject)).toThrow("expected an object");
  });
});
