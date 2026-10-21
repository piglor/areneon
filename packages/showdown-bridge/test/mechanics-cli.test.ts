import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeAll, describe, expect, it, vi } from "vitest";
import type { Engine } from "../src/engine.ts";
import { GOLDEN_FORMAT, TEAMS_DIR } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { MECHANICS_CONFIG_DIR } from "../src/mechanics-config.ts";
import { nodeMechanicsIo, runMechanicsCli } from "../src/mechanics-cli.ts";
import { readMechanicsResult } from "../src/mechanics-result.ts";
import { exportPaste, importPaste } from "../src/teams.ts";

let engine: Engine;
let own: string;
beforeAll(() => {
  engine = loadEngine();
  own = exportPaste(engine, [importPaste(engine, readFileSync(join(TEAMS_DIR, "legal-synthetic.txt"), "utf8"))[0] as never]);
});

interface Run {
  code: number;
  out: string;
  err: string;
}
function run(argv: string[], files: Record<string, string>): Run {
  let out = "";
  let err = "";
  const code = runMechanicsCli(
    argv,
    {
      out: (text) => void (out += text),
      err: (text) => void (err += text),
      read: (path) => {
        const text = files[path];
        if (text === undefined) throw new Error(`ENOENT ${path}`);
        return text;
      },
    },
    () => engine,
  );
  return { code, out, err };
}

const query = (set: string) => JSON.stringify({ schema_version: 1, data_class: "synthetic", format_id: GOLDEN_FORMAT, query: { kind: "stats", set } });

describe("scripts/mechanics.ts", () => {
  it("prints the result as one canonical JSON line and exits 0", () => {
    const { code, out, err } = run(["q.json"], { "q.json": query(own) });
    expect({ code, err }).toEqual({ code: 0, err: "" });
    expect(out.endsWith("\n")).toBe(true);
    expect(out.trim().split("\n")).toHaveLength(1);
    expect(readMechanicsResult(JSON.parse(out) as unknown)).toMatchObject({ kind: "computed", answer: { kind: "stats" } });
    // The same query gives the same bytes.
    expect(run(["q.json"], { "q.json": query(own) }).out).toBe(out);
  });

  it("prints a refusal as a result and the validator's words on stderr", () => {
    const bad = exportPaste(engine, [importPaste(engine, readFileSync(join(TEAMS_DIR, "illegal-stat-points-over-32.txt"), "utf8"))[0] as never]);
    const { code, out, err } = run(["/dev/stdin"], { "/dev/stdin": query(bad) });
    expect(code).toBe(0);
    expect(JSON.parse(out)).toMatchObject({ kind: "refused", code: "invalid_set", position: 0 });
    expect(err).toContain("Stat Points");
    expect(out).not.toContain("Stat Points");
  });

  it("takes a config file with --config", () => {
    const dir = mkdtempSync(join(tmpdir(), "mechanics-cli-"));
    const path = join(dir, "config.json");
    const text = readFileSync(join(MECHANICS_CONFIG_DIR, `${GOLDEN_FORMAT}.json`), "utf8");
    writeFileSync(path, text);
    expect(run(["q.json", "--config", path], { "q.json": query(own), [path]: text }).code).toBe(0);
  });

  it.each([
    ["no query file", [], () => ({}), "usage:"],
    ["two query files", ["a", "b"], () => ({}), "usage:"],
    ["a file that is not there", ["missing.json"], () => ({}), "ENOENT"],
    ["text that is not JSON", ["q.json"], () => ({ "q.json": "{" }), "mechanics:"],
    ["a document that is not a MechanicsQuery", ["q.json"], () => ({ "q.json": JSON.stringify({ kind: "stats" }) }), "invalid MechanicsQuery"],
    ["a format with no mechanics config", ["q.json"], () => ({ "q.json": query(own).replace(GOLDEN_FORMAT, "gen9ou"), "config/mechanics/gen9ou.json": "{}" }), "gen9ou"],
    ["a config that is not a MechanicsConfig", ["q.json", "--config", "c.json"], () => ({ "q.json": query(own), "c.json": "{}" }), "invalid MechanicsConfig"],
  ] as [string, string[], () => Record<string, string>, string][])("exits 2 with no result for %s", (_name, argv, files, message) => {
    const { code, out, err } = run(argv, files());
    expect(code).toBe(2);
    expect(out).toBe("");
    expect(err).toContain(message);
  });

  it("reads files and writes to the process through nodeMechanicsIo", () => {
    const dir = mkdtempSync(join(tmpdir(), "mechanics-io-"));
    writeFileSync(join(dir, "x.txt"), "hello");
    const io = nodeMechanicsIo();
    expect(io.read(join(dir, "x.txt"))).toBe("hello");
    const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    const err = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    io.out("result\n");
    io.err("words\n");
    expect(out).toHaveBeenCalledWith("result\n");
    expect(err).toHaveBeenCalledWith("words\n");
    vi.restoreAllMocks();
  });
});
