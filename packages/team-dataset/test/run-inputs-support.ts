/* eslint-disable @typescript-eslint/no-explicit-any */
// What the run-input tests share: a run config on the committed fixtures-v1
// dataset, a way to publish (possibly doctored) inputs to an in-memory store,
// and a fake R2 behind `fetch` for the credential canary.

import { createHash } from "node:crypto";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { InMemoryObjectStore } from "@areneon/object-store";
import { jcs, readTrainingRunConfig, type RunInputManifest, runConfigHash } from "@areneon/showdown-bridge";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASET_DIR, fileKey, manifestKey, RUN_CONFIG_FILE, runInputLayout } from "../src/run-inputs.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));
export const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
export const RUN_KEY = "run-inputs-1078";
export const COMMIT = "1234567890abcdef1234567890abcdef12345678";
export const sha256 = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");
export const scratchDir = (): string => mkdtempSync(join(tmpdir(), "run-inputs-"));
export const removeDir = (dir: string): void => rmSync(dir, { recursive: true, force: true });

export const DATASET_HASH = verifiedDatasetHash(engine, SOURCE, FORMAT);
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, any>;

/** A run config (raw JSON) that names `path` as its dataset. */
export function configFor(path: string, change: (raw: Record<string, any>) => void = () => undefined): Record<string, any> {
  const raw = structuredClone(tiny);
  raw.seeds.run_key = RUN_KEY;
  raw.generation.excluded_abilities = [];
  raw.team_dataset = { path, dataset_hash: DATASET_HASH };
  change(raw);
  return raw;
}
export const configText = (raw: Record<string, any>): string => `${JSON.stringify(raw, null, 2)}\n`;

/** The dataset's files as bytes, by path under the dataset directory. */
export function datasetFiles(dir = SOURCE): Map<string, Uint8Array> {
  return new Map(readdirSync(dir).sort().map((name) => [name, readFileSync(join(dir, name))]));
}

export interface Doctor {
  config?: (raw: Record<string, any>) => void;
  /** Change a file's bytes before its manifest entry is computed. */
  files?: (files: Map<string, Uint8Array>) => void;
  /** Change the manifest after its entries were computed. */
  manifest?: (manifest: Record<string, any>) => void;
  /** Replace the stored manifest bytes. */
  manifestBytes?: (canonical: Uint8Array) => Uint8Array;
  /** Leave one file out of the bucket (by path). */
  leaveOut?: string;
}

export interface Published {
  readonly store: InMemoryObjectStore;
  readonly manifest: RunInputManifest;
  readonly manifestBytes: Uint8Array;
  readonly manifestHash: string;
}

/** Put a run's inputs in an in-memory store by hand, so a test can doctor any part. */
export async function publish(doctor: Doctor = {}, runKey = RUN_KEY): Promise<Published> {
  const raw = configFor(join(SOURCE), doctor.config);
  const configBytes = new TextEncoder().encode(configText(raw));
  const files = new Map<string, Uint8Array>([[RUN_CONFIG_FILE, configBytes]]);
  for (const [name, bytes] of datasetFiles()) files.set(`${DATASET_DIR}/${name}`, bytes);
  doctor.files?.(files);
  const manifest: Record<string, any> = {
    schema_version: 1,
    data_class: "synthetic",
    run_key: runKey,
    commit: COMMIT,
    run_config_hash: await runConfigHash(readTrainingRunConfig(raw)),
    dataset_hash: DATASET_HASH,
    config_file: RUN_CONFIG_FILE,
    dataset_dir: DATASET_DIR,
    files: [...files.keys()].sort().map((path) => ({ path, size: (files.get(path) as Uint8Array).byteLength, sha256: sha256(files.get(path) as Uint8Array) })),
  };
  doctor.manifest?.(manifest);
  const store = new InMemoryObjectStore(runInputLayout(runKey));
  for (const [path, bytes] of files) if (path !== doctor.leaveOut) store.seed(fileKey(runKey, path), bytes);
  const canonical = new TextEncoder().encode(jcs(manifest as never));
  const manifestBytes = doctor.manifestBytes?.(canonical) ?? canonical;
  store.seed(manifestKey(runKey), manifestBytes);
  return { store, manifest: manifest as RunInputManifest, manifestBytes, manifestHash: sha256(manifestBytes) };
}

/** A fake R2 behind `fetch`: path-style objects in a map, conditional writes, one-page listings. */
export class FakeR2 {
  readonly objects = new Map<string, Uint8Array>();
  readonly seen: string[] = [];
  /** Answer a request yourself (a status and a body that may echo markers). */
  override: ((request: Request) => Response | undefined) | undefined;

  readonly bucket: string;

  constructor(bucket: string) {
    this.bucket = bucket;
  }

  readonly fetch = async (request: Request): Promise<Response> => {
    this.seen.push(`${request.method} ${request.url}`);
    const forced = this.override?.(request);
    if (forced !== undefined) return forced;
    const url = new URL(request.url);
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts[0] !== this.bucket) return new Response("<Error>NoSuchBucket</Error>", { status: 404 });
    const key = parts.slice(1).join("/");
    if (key === "") {
      const prefix = url.searchParams.get("prefix") ?? "";
      const contents = [...this.objects.keys()].filter((k) => k.startsWith(prefix)).sort().map((k) => `<Contents><Key>${k}</Key></Contents>`).join("");
      return new Response(`<?xml version="1.0"?><ListBucketResult><IsTruncated>false</IsTruncated>${contents}</ListBucketResult>`, { status: 200 });
    }
    if (request.method === "PUT") {
      if (request.headers.get("if-none-match") === "*" && this.objects.has(key)) return new Response(null, { status: 412 });
      this.objects.set(key, new Uint8Array(await request.arrayBuffer()));
      return new Response(null, { status: 200 });
    }
    const found = this.objects.get(key);
    if (found === undefined) return new Response("<Error>NoSuchKey</Error>", { status: 404 });
    return new Response(request.method === "HEAD" ? null : (found as unknown as BodyInit), { status: 200, headers: { "content-length": String(found.byteLength) } });
  };
}
