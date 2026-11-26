import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import type { Engine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";
import { loadMechanicsConfig, type MechanicsConfig, withMoveOutcome } from "../src/mechanics-config.ts";
import { type ParsedMechanicsQuery, readMechanicsQuery } from "../src/mechanics-query.ts";
import { NATIVE_PROBE_FORMAT, NativeAbstain, recover, nativeAbstained, nativeCase, nativeProbeBackend, type NativeProbeBackend, NativeProbeBackendError } from "../src/native-probe-backend.ts";
import { NativeProbeError, NativeProbeProcess } from "../src/native-probe-process.ts";
import { NativeProtocolError } from "../src/native-probe-wire.ts";

const FAKE = fileURLToPath(new URL("./support/fake-native-probe.mjs", import.meta.url));
const FORMAT = GOLDEN_FORMAT;
const damageFixture = JSON.parse(readFileSync(new URL(`../../../fixtures/mechanics/${FORMAT}/damage/cases.json`, import.meta.url), "utf8")) as { cases: { query: { query: { kind: string; attacker: { set: string }; defender: { set: string } } } }[] };

let engine: Engine;
let config: MechanicsConfig;
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(FORMAT);
});

const backends: NativeProbeBackend[] = [];
const processes: NativeProbeProcess[] = [];
afterEach(() => {
  delete process.env.FAKE_MODE;
  for (const b of backends.splice(0)) b.close();
  for (const p of processes.splice(0)) p.close();
});
const stand = (mode: string, extra: { timeout_ms?: number; start_ms?: number } = { start_ms: 500 }): NativeProbeBackend => {
  process.env.FAKE_MODE = mode;
  const backend = nativeProbeBackend(engine, { config, command: { bin: FAKE, dex: "unused" }, backend_commit: "a".repeat(40), ...extra });
  backends.push(backend);
  return backend;
};
const query = (kind: "stats" | "damage"): ParsedMechanicsQuery => {
  const base = damageFixture.cases[0]?.query as { query: { kind: string; attacker: { set: string } } };
  return readMechanicsQuery(kind === "damage" ? base : { ...base, query: { kind: "stats", set: base.query.attacker.set } });
};
const invalid = (): ParsedMechanicsQuery => readMechanicsQuery({ ...(damageFixture.cases[0]?.query as object), query: { ...(damageFixture.cases[0]?.query.query as object), attacker: { ...damageFixture.cases[0]?.query.query.attacker, set: "Nopemon  \n- Tackle\n" } } });

describe("the process transport", () => {
  it("answers a frame and carries a frame larger than a pipe's buffer", () => {
    process.env.FAKE_MODE = "huge";
    const p = new NativeProbeProcess({ bin: FAKE, dex: "unused" });
    processes.push(p);
    const out = p.request(Buffer.from(JSON.stringify({ op: "batch", queries: [{ kind: "stats" }, { kind: "stats" }] })));
    expect(out.length).toBeGreaterThan(130_000);
    expect(p.alive).toBe(true);
    p.close();
    p.close();
    expect(p.alive).toBe(false);
    expect(() => p.request(Buffer.from("{}"))).toThrow(NativeProbeError);
  });

  it("ends with an error when the process dies, stays silent or never starts", async () => {
    process.env.FAKE_MODE = "die";
    const dying = new NativeProbeProcess({ bin: FAKE, dex: "unused" });
    processes.push(dying);
    expect(() => dying.request(Buffer.from(JSON.stringify({ op: "batch", queries: [] })))).toThrow(NativeProbeError);
    expect(dying.alive).toBe(false);
    process.env.FAKE_MODE = "silent";
    const silent = new NativeProbeProcess({ bin: FAKE, dex: "unused" }, { timeout_ms: 300 });
    processes.push(silent);
    expect(() => silent.request(Buffer.from(JSON.stringify({ op: "batch", queries: [] })))).toThrow("did not answer in time");
    // A process that never reads: a request larger than the pipe's buffer cannot be written.
    process.env.FAKE_MODE = "deaf";
    const deaf = new NativeProbeProcess({ bin: FAKE, dex: "unused" }, { timeout_ms: 300 });
    processes.push(deaf);
    expect(() => deaf.request(Buffer.alloc(400_000, 32))).toThrow("did not read its request in time");
    process.env.FAKE_MODE = "closed-input";
    const closed = new NativeProbeProcess({ bin: FAKE, dex: "unused" }, { timeout_ms: 300 });
    processes.push(closed);
    expect(() => closed.request(Buffer.alloc(400_000, 32))).toThrow("pipe broke");
    process.env.FAKE_MODE = "no-start";
    expect(() => new NativeProbeProcess({ bin: FAKE, dex: "unused" }, { start_ms: 400 })).toThrow(NativeProbeError);
    expect(() => new NativeProbeProcess({ bin: "/nonexistent/ninjask-probe", dex: "unused" }, { start_ms: 300 })).toThrow();
    // The spawn failure arrives as an event once the event loop runs.
    await new Promise((resolve) => setTimeout(resolve, 100));
  });
});

describe("the backend behind the probe's interface", () => {
  it("carries the identity the ledger names and refuses a backend commit that is not 40 hex digits", () => {
    const b = stand("plain");
    expect(b).toMatchObject({ id: "native-probe", backend_format: NATIVE_PROBE_FORMAT, backend_commit: "a".repeat(40) });
    expect(() => nativeProbeBackend(engine, { config, command: { bin: FAKE, dex: "unused" }, backend_commit: "abc" })).toThrow(NativeProbeBackendError);
  });

  it("answers in the probe's envelope and abstains, by entity, on what the native layer leaves to the probe", () => {
    const b = stand("plain");
    const [stats, damage] = b.run([query("stats"), query("damage")]);
    expect(stats).toMatchObject({ ok: true, result: { kind: "computed", answer: { kind: "stats", stats: { hp: 1, spe: 6 } }, adapter_format: 2, data_class: "synthetic" } });
    expect(nativeAbstained(damage as never)).toBe(true);
    expect((damage as { error: NativeAbstain }).error).toMatchObject({ entity: "move:fake" });
    expect(b.counts).toMatchObject({ answered: 1, abstained: 1 });
    expect(b.counts.by_entity.get("move:fake")).toBe(1);
  });

  it("takes an unsupported outcome as the probe's own result and a refused set from the envelope", () => {
    const b = stand("unsupported");
    const [damage, refused] = b.run([query("damage"), invalid()]);
    expect(damage).toMatchObject({ ok: true, result: { kind: "unsupported", reason: "query_kind", position: null } });
    expect(refused).toMatchObject({ ok: true, result: { kind: "refused", code: "invalid_set", position: 0 } });
    expect(b.counts.unsupported).toBe(1);
  });

  it("abstains on a history block and on a set the native dex does not know", () => {
    const q = structuredClone(query("damage")) as ParsedMechanicsQuery;
    (q.query as { history?: unknown }).history = { turn: 1 };
    expect(nativeAbstained(stand("plain").run([q])[0] as never)).toBe(true);
    const refusing = stand("refuse-sets");
    const out = refusing.run([query("stats")])[0] as { error: NativeAbstain };
    expect(out.error.entity).toBe("state:set");
  });

  it("abstains with the process entity when the process answers wrongly, and starts it again", () => {
    for (const mode of ["short", "garbage", "error"]) {
      const b = stand(mode);
      const [out] = b.run([query("damage")]);
      expect((out as { error: NativeAbstain }).error.entity).toBe("native:process");
      // The next batch starts a new process (the same bad one): it abstains again, and the restart is counted.
      b.run([query("damage")]);
      expect(b.counts.restarts).toBeGreaterThanOrEqual(1);
    }
  });

  it("abstains on a kind it never sends, and reports what the envelope throws as the probe does", () => {
    const base = query("damage") as unknown as { query: { attacker: { set: string } } };
    const set = base.query.attacker.set;
    const side = { set, boosts: { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, status: null, hp: null, mega: false };
    const state = { weather: null, terrain: null, trick_room: false, sides: [{ aurora_veil: false, light_screen: false, reflect: false, tailwind: false }, { aurora_veil: false, light_screen: false, reflect: false, tailwind: false }] };
    const outcome = { schema_version: 1, data_class: "synthetic", format_id: FORMAT, query: { kind: "move_outcome", sides: [[side], [side]], actions: [], state } } as unknown as ParsedMechanicsQuery;
    process.env.FAKE_MODE = "plain";
    const wide = nativeProbeBackend(engine, { config: withMoveOutcome(config), command: { bin: FAKE, dex: "unused" }, backend_commit: "a".repeat(40) });
    backends.push(wide);
    expect((wide.run([outcome])[0] as { error: NativeAbstain }).error.entity).toBe("kind:move_outcome");
    const other = nativeProbeBackend(engine, { config: { ...config, format_id: "gen9other" }, command: { bin: FAKE, dex: "unused" }, backend_commit: "a".repeat(40) });
    backends.push(other);
    const thrown = other.run([query("stats")])[0] as { ok: boolean; error: Error };
    expect(thrown.ok).toBe(false);
    expect(thrown.error.message).toContain("another format");
  });

  it("sends a set once, and takes the rules snapshot it is given", () => {
    process.env.FAKE_MODE = "plain";
    const b = nativeProbeBackend(engine, { config, snapshot: loadRulesSnapshot(engine.commit), command: { bin: FAKE, dex: "unused" }, backend_commit: "a".repeat(40) });
    backends.push(b);
    b.run([query("stats")]);
    const frames = b.counts.frames;
    b.run([query("stats")]);
    // The second batch registers no set: one frame (the batch), not two.
    expect(b.counts.frames).toBe(frames + 1);
  });

  it("routes a failure of the transport and propagates any other error", () => {
    expect(() => recover(new NativeProbeError("x"))).not.toThrow();
    expect(() => recover(new NativeProtocolError("x"))).not.toThrow();
    expect(() => recover(new TypeError("a defect"))).toThrow(TypeError);
  });

  it("recovers from a dead process on the next batch", () => {
    const b = stand("die");
    expect(nativeAbstained(b.run([query("damage")])[0] as never)).toBe(true);
    process.env.FAKE_MODE = "plain";
    expect(b.run([query("stats")])[0]).toMatchObject({ ok: true });
    expect(b.counts.restarts).toBe(1);
  });

  it("stops restarting a process that keeps dying", () => {
    const b = stand("no-start");
    for (let i = 0; i < 8; i++) expect(nativeAbstained(b.run([query("stats")])[0] as never)).toBe(true);
    expect(b.counts.abstained).toBe(8);
  });

  it("gives the native layer's form of a query for a test of the fork, and none for a query it never sends", () => {
    const context = { engine, config };
    const wire = nativeCase(context, query("damage"));
    expect(wire).not.toBeNull();
    expect(wire?.sets).toHaveLength(2);
    expect(wire?.query).toMatchObject({ kind: "damage", attacker: { set: 0 }, defender: { set: 1 } });
    expect(nativeCase(context, invalid())).toBeNull();
    const side = { set: (query("damage").query as { attacker: { set: string } }).attacker.set, boosts: { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, status: null, hp: null, mega: false };
    const off = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
    const outcome = { schema_version: 1, data_class: "synthetic", format_id: FORMAT, query: { kind: "move_outcome", sides: [[side], [side]], actions: [], state: { weather: null, terrain: null, trick_room: false, sides: [off, off] } } } as unknown as ParsedMechanicsQuery;
    expect(nativeCase({ engine, config: withMoveOutcome(config) }, outcome)).toBeNull();
    const withHistory = structuredClone(query("damage")) as ParsedMechanicsQuery;
    (withHistory.query as { history?: unknown }).history = { turn: 1 };
    expect(nativeCase(context, withHistory)).toBeNull();
  });
});
