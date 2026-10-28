// The golden digest test (ADR-025 section 5, review finding N-G): a committed fixture fixes the SHA-256
// of the canonical JSON of a belief and of `sampleWorlds(belief, stream, 8)` for fixed inputs. It fails on
// any change of how a value is computed (the filter, the draw, the canonical encoding) unless
// `BELIEF_FORMAT` is bumped and the fixture is rewritten with it:
//
//   ARENEON_UPDATE_GOLDEN=1 pnpm exec vitest run packages/opponent-model/test/golden.test.ts
//
// and the diff reviewed. The inputs are the hand-made vocabulary and prior of `support.ts` (CC0,
// synthetic; no team, no real usage data).

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { BELIEF_FORMAT, beliefDigest, belief, createSampler, worldsDigest } from "../src/index.ts";
import { hidingConfig, input } from "./support.ts";

// A stand-in for the bridge's engine check: Alpha with ability a1 would show maybeTrapped to the own active.
const alphaTrap = (observed: 0 | 1) => ({ observed: [observed] as const, shows: (c: { species: string; ability: string }) => [c.species === "alpha" && c.ability === "a1" ? 1 : 0] as const });

const FILE = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "fixtures", "opponent-model", "golden.json");
const STREAM = Uint8Array.from({ length: 32 }, (_, i) => 255 - i * 5);
const SW = (letter: "a" | "b", name: string, species: string) => `|switch|p2${letter}: ${name}|${species}, L50, M|100/100`;

const cases = {
  "preview-only": () => input(),
  "revealed-facts": () => input([SW("a", "Alp", "Alpha"), "|move|p2a: Alp|Mb", "|-item|p2a: Alp|Ix", SW("b", "Gam", "Gamma")]),
  "mega-evolved": () => input([SW("a", "Bet", "Beta"), "|detailschange|p2a: Bet|Beta-Mega, L50, M", "|-mega|p2a: Bet|Beta|Betite"]),
  "disguise-mixture": () => input([SW("a", "Alp", "Alpha")], { config: hidingConfig }),
  "trapping-seen": () => input([SW("a", "Alp", "Alpha")], { trapping: alphaTrap(1) }),
  "trapping-unseen": () => input([SW("a", "Alp", "Alpha")], { trapping: alphaTrap(0) }),
} as const;

interface Golden {
  readonly description: string;
  readonly belief_format: string;
  readonly stream: string;
  readonly worlds: number;
  readonly cases: Record<string, { readonly belief_sha256: string; readonly worlds_sha256: string }>;
}

const sampler = createSampler({ legality: { legal: () => true } });
const compute = (): Golden["cases"] =>
  Object.fromEntries(
    Object.entries(cases).map(([name, make]) => {
      const b = belief(make());
      return [name, { belief_sha256: beliefDigest(b), worlds_sha256: worldsDigest(sampler.sampleWorlds(b, STREAM, 8)) }];
    }),
  );

describe("the golden digests", () => {
  if (process.env["ARENEON_UPDATE_GOLDEN"] === "1") {
    it("rewrites the fixture", () => {
      const golden: Golden = {
        description: "SHA-256 of the canonical JSON (bigints as decimal strings) of belief() and of sampleWorlds(belief, stream, 8) on the synthetic inputs of packages/opponent-model/test/support.ts (CC0, no team data). Rewrite with ARENEON_UPDATE_GOLDEN=1 only together with a BELIEF_FORMAT bump.",
        belief_format: BELIEF_FORMAT,
        stream: Buffer.from(STREAM).toString("hex"),
        worlds: 8,
        cases: compute(),
      };
      writeFileSync(FILE, `${JSON.stringify(golden, null, 2)}\n`);
    });
  }

  const read = (): Golden => JSON.parse(readFileSync(FILE, "utf8")) as Golden;

  it("pins BELIEF_FORMAT: a change of how a value is computed needs a new format and a rewritten fixture", () => {
    const golden = read();
    expect(golden.belief_format, "bump BELIEF_FORMAT and rewrite the fixture together").toBe(BELIEF_FORMAT);
    expect(golden.stream).toBe(Buffer.from(STREAM).toString("hex"));
    expect(golden.worlds).toBe(8);
  });

  for (const name of Object.keys(cases)) {
    it(`the belief and the eight worlds of "${name}" hash to the committed digests`, () => {
      const got = compute()[name];
      expect(got, "a digest differs: bump BELIEF_FORMAT, then rewrite the fixture").toEqual(read().cases[name]);
    });
  }

  it("a changed value changes the digest (negative control)", () => {
    const b = belief(input());
    const tampered = { ...b, brought: { ...b.brought, subsets: b.brought.subsets + 1n } };
    expect(beliefDigest(tampered)).not.toBe(beliefDigest(b));
    const other = sampler.sampleWorlds(b, Uint8Array.from(STREAM, (v) => v ^ 1), 8);
    expect(worldsDigest(other)).not.toBe(worldsDigest(sampler.sampleWorlds(b, STREAM, 8)));
  });
});
