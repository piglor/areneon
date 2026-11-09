import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { GOLDEN_DIR, type RequestFixture } from "../src/golden.ts";
import { decodeRequest } from "../src/index.ts";

const made: Record<string, unknown>[] = [];
vi.mock("../src/search-actor.ts", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../src/search-actor.ts")>()),
  searchActor: vi.fn((options: Record<string, unknown>) => {
    made.push(options);
    return Object.assign(async () => "default", { report: () => ({}) });
  }),
}));
vi.mock("../src/policy-actor.ts", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../src/policy-actor.ts")>()),
  policyActor: vi.fn(() => async () => "inner"),
}));

const { controlSeat, searchSeat } = await import("../src/search-progress.ts");

const actorOptions = { battle: "b", player: "p1", team: [], manifest: {}, channel_class: "synthetic", seed: "s", game: 1 } as never;
const client = {} as never;
const setup = {} as never;
const play = { temperature_ppm: 500_000 } as never;
const features = { configs: {} } as never;
const pair = { preview: "a", battle: "b" };

describe("searchSeat", () => {
  it("passes only what it was given", () => {
    made.length = 0;
    const plain = searchSeat({ setup, client, policy_id: "id", identity_hash: "h", mode: "argmax", onActor: () => undefined });
    expect("play" in plain).toBe(false);
    plain.make?.(actorOptions);
    expect("play" in (made[0] as object)).toBe(false);
    expect("pair" in (made[0] as object)).toBe(false);
    expect("features" in (made[0] as object)).toBe(false);
  });

  it("passes play, pair and features through to the seat and the actor", () => {
    made.length = 0;
    const seen: unknown[] = [];
    const seat = searchSeat({ setup, client, policy_id: "id", identity_hash: "h", mode: "sample", play, pair, features, onActor: (a) => seen.push(a) });
    expect(seat.play).toBe(play);
    seat.make?.(actorOptions);
    expect(made[0]).toMatchObject({ play, pair, features });
    expect(seen).toHaveLength(1);
  });
});

describe("controlSeat", () => {
  it("omits play and features when absent, includes them when given", () => {
    expect("play" in controlSeat({ client, policy_id: "id", mode: "argmax", onTally: () => undefined })).toBe(false);
    expect("features" in controlSeat({ client, policy_id: "id", mode: "argmax", onTally: () => undefined })).toBe(false);
    const seat = controlSeat({ client, policy_id: "id", mode: "sample", play, features, onTally: () => undefined });
    expect(seat.play).toBe(play);
    expect(seat.features).toBe(features);
  });

  it("counts decisions and structural fallbacks", async () => {
    let tally: { counts(): { decisions: number; fallbacks: Record<string, number> } } | undefined;
    const seat = controlSeat({ client, policy_id: "id", mode: "argmax", onTally: (t) => (tally = t as never) });
    const actor = seat.make?.(actorOptions) as (r: unknown, c?: unknown) => Promise<string>;
    expect(await actor({ phase: "preview" })).toBe("inner");
    expect(await actor({ phase: "forceSwitch" })).toBe("inner");
    const counts = tally?.counts();
    expect(counts?.decisions).toBe(2);
    expect(counts?.fallbacks.team_preview).toBe(1);
    expect(counts?.fallbacks.forced_switch).toBe(1);
  });

  it("counts a move request with fewer than two non-pending candidates as structural, and one with more as none", async () => {
    let tally: { counts(): { fallbacks: Record<string, number> } } | undefined;
    const seat = controlSeat({ client, policy_id: "id", mode: "argmax", onTally: (t) => (tally = t as never) });
    const actor = seat.make?.(actorOptions) as (r: unknown, c?: unknown) => Promise<string>;
    const fixture = JSON.parse(readFileSync(join(GOLDEN_DIR, "turn.request.json"), "utf8")) as RequestFixture;
    const request = decodeRequest(JSON.stringify(fixture.request));
    await actor(request);
    const lone = structuredClone(fixture.request) as never as { active: { moves: unknown[] }[]; side: { pokemon: { condition: string }[] } };
    lone.active = lone.active.map((a) => ({ ...a, moves: a.moves.slice(0, 1) }));
    for (const p of lone.side.pokemon.slice(1)) p.condition = "0 fnt";
    await actor(decodeRequest(JSON.stringify(lone)));
    const fallbacks = tally?.counts().fallbacks;
    expect(fallbacks?.too_few_candidates).toBeLessThanOrEqual(1);
    expect(fallbacks?.team_preview).toBe(0);
  });
});
