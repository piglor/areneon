// ADR-027 slice 4: the handshake states an opponent sheet source, and the client sends a block only to a worker that stated one.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { PolicyHandshakeError, PolicyWorkerClient } from "../src/policy-worker.ts";
import type { PolicyQueryInput } from "../src/policy-protocol.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { CHECKPOINTS, ENCODER_VERSION, ENGINE_DATA_HASH, MAX_LINE_BYTES } from "./policy-fake.ts";

const FAKE = fileURLToPath(new URL("./fake-policy-worker.ts", import.meta.url));
const sample = (file: string): PolicyQueryInput => {
  const { query_id: _id, ...input } = JSON.parse(readFileSync(`${SAMPLES_DIR}/v1/${file}.json`, "utf8")) as PolicyQueryInput & { query_id: number };
  return input;
};
const withBlock = sample("policy-query/valid/battle-opponent-sheet");
const plain = sample("policy-query/valid/battle");
const ref = { feature_set: "opponent-sheet/1", feature_set_hash: withBlock.opponent_sheet?.source.hash as string };

describe("the handshake states an opponent sheet source", () => {
  const opened: PolicyWorkerClient[] = [];
  afterEach(async () => {
    await Promise.all(opened.splice(0).map((c) => c.close()));
  });
  const client = (stated: unknown, supply: typeof ref | undefined): PolicyWorkerClient => {
    const made = new PolicyWorkerClient({
      command: process.execPath,
      args: [FAKE, "ok"],
      env: stated === undefined ? {} : { FAKE_OPPONENT_SHEET: JSON.stringify(stated) },
      max_line_bytes: MAX_LINE_BYTES,
      n_parallel: 1,
      handshake_timeout_ms: 10_000,
      expect: { encoder_version: ENCODER_VERSION, engine_data_hash: ENGINE_DATA_HASH, checkpoints: CHECKPOINTS },
      ...(supply === undefined ? {} : { sheet: supply }),
    });
    opened.push(made);
    return made;
  };

  it("a worker that states none needs none, whether or not this side can supply one", async () => {
    for (const supply of [undefined, ref]) {
      const worker = client(undefined, supply);
      await worker.start();
      expect(worker.sheetSource).toBeNull();
      await expect(worker.ask("g", plain)).resolves.toMatchObject({ source_policy_id: plain.policy_id });
    }
  });

  it("a stated source that this side supplies is accepted, and a restart states it again", async () => {
    const worker = client(ref, ref);
    await worker.start();
    expect(worker.sheetSource).toEqual(ref);
    await worker.restart();
    expect(worker.sheetSource).toEqual(ref);
  });

  it.each([
    ["another hash", { ...ref, feature_set_hash: "b".repeat(64) }, ref],
    ["another set name", { ...ref, feature_set: "opponent-sheet/2" }, ref],
    ["a source while this side supplies none", ref, undefined],
  ])("refuses %s before any battle", async (_name, stated, supply) => {
    await expect(client(stated, supply).start()).rejects.toThrow(PolicyHandshakeError);
    await expect(client(stated, supply).start()).rejects.toThrow(/cannot supply/);
  });

  it("sends opponent_sheet to a worker that states a source only, and only of that source", async () => {
    const worker = client(ref, ref);
    await worker.start();
    await expect(worker.ask("g", withBlock)).resolves.toBeDefined();
    await expect(worker.ask("g", plain)).rejects.toThrow(/carries no opponent_sheet/);
    const other = { ...withBlock, opponent_sheet: { ...withBlock.opponent_sheet, source: { ...withBlock.opponent_sheet?.source, hash: "c".repeat(64) } } } as PolicyQueryInput;
    await expect(worker.ask("g", other)).rejects.toThrow(/another source/);
    const none = client(undefined, undefined);
    await none.start();
    await expect(none.ask("g", withBlock)).rejects.toThrow(/stated no sheet source/);
  });
});
