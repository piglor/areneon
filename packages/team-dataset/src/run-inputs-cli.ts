// The command line of the run inputs (ADR-024 section 3.5), logic kept here so
// a test can drive it with a fake store; `scripts/run-inputs.ts` only wires the
// process.
//
//   run-inputs.ts publish --run-config FILE --dataset-dir DIR --commit SHA --bucket NAME
//       owner: build the manifest, put the files and then the manifest in the inputs bucket
//   run-inputs.ts fetch --run-key KEY --out DIR --bucket NAME (--manifest-hash HEX | --no-pin) [--repo DIR]
//       rented machine: pull, check every size and hash, then check the dataset hash and the commit
//
// The manifest hash (printed by `publish`, given to the machine out of band) is
// the ONLY authenticity anchor: everything else the fetch checks is a hash the
// manifest itself lists, so whoever can write the bucket can write a manifest
// that agrees with doctored files. `fetch` therefore needs `--manifest-hash`,
// or `--no-pin` to say out loud that no anchor is used. The checked-out
// repository must have no uncommitted changes to tracked files. The run
// config's `team_dataset.path` is part of its hash, so the owner writes into it,
// before `publish`, the path the machine will use (for example a relative
// `inputs/dataset` with `--out inputs`, run from the repository root); `fetch`
// refuses a config whose path is not the fetched dataset directory. One input
// file may be at most 256 MiB (the object store's ceiling).
//
// The credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL,
// AWS_DEFAULT_REGION and, for a temporary credential, AWS_SESSION_TOKEN) come
// from the environment only; a command line that carries one is refused before
// the environment is read. `publish` needs a credential that may write
// `runs/<run_key>/inputs/`; `fetch` needs one that may only read it.
//
// Exit codes: 0 done (one JSON line on stdout), 1 a refusal (`run-inputs:
// <code>`, a fixed message, on stderr), 2 a usage or credential error.

import { parseArgs } from "node:util";
import { CredentialError, type ObjectLayout, type ObjectStore, ObjectStoreError, R2Settings } from "@areneon/object-store";
import type { Engine } from "@areneon/showdown-bridge";
import { buildRunInputs, fetchRunInputs, RunInputError, runInputLayout, uploadRunInputs } from "./run-inputs.ts";

export interface RunInputsDeps {
  readonly env: Readonly<Record<string, string | undefined>>;
  readonly engine: () => Engine;
  /** The store behind the settings: the process wires the R2 client, tests give a fake (no CI test opens a socket). */
  readonly openStore: (settings: R2Settings, layout: ObjectLayout) => ObjectStore;
  /** The HEAD commit of the repository at `dir` and whether its tracked files have uncommitted changes (the process asks git). */
  readonly repoState: (dir: string) => { readonly commit: string; readonly dirty: boolean };
  readonly out: (line: string) => void;
  readonly err: (line: string) => void;
}

class UsageError extends Error {}

const OPTIONS = {
  "run-config": { type: "string" },
  "dataset-dir": { type: "string" },
  commit: { type: "string" },
  bucket: { type: "string" },
  "run-key": { type: "string" },
  out: { type: "string" },
  "manifest-hash": { type: "string" },
  "no-pin": { type: "boolean" },
  repo: { type: "string" },
} as const;

function need(value: string | undefined, name: string): string {
  if (value === undefined || value === "") throw new UsageError(`--${name} is required`);
  return value;
}

/** Run the command; returns the exit code. */
export async function runInputsCli(argv: readonly string[], deps: RunInputsDeps): Promise<number> {
  try {
    const [command, ...rest] = argv;
    if (command !== "publish" && command !== "fetch") throw new UsageError("usage: run-inputs publish|fetch ...");
    let values;
    try {
      values = parseArgs({ args: [...rest], options: OPTIONS, strict: true, allowPositionals: false }).values;
    } catch {
      // An option that names a credential is reported as that, not as an unknown option.
      try {
        R2Settings.fromEnvironment({}, argv, "argv-probe");
      } catch (probe) {
        if (probe instanceof CredentialError && probe.code === "credentials_on_argv") throw probe;
      }
      throw new UsageError("unknown or malformed option");
    }
    const bucket = need(values.bucket, "bucket");
    const settings = R2Settings.fromEnvironment(deps.env, argv, bucket);
    if (command === "publish") {
      const inputs = await buildRunInputs({ engine: deps.engine(), configFile: need(values["run-config"], "run-config"), datasetDir: need(values["dataset-dir"], "dataset-dir"), commit: need(values.commit, "commit") });
      const hash = await uploadRunInputs(deps.openStore(settings, runInputLayout(inputs.manifest.run_key)), inputs);
      deps.out(JSON.stringify({ manifest_hash: hash, run_key: inputs.manifest.run_key, files: inputs.manifest.files.length }));
      return 0;
    }
    const runKey = need(values["run-key"], "run-key");
    const pin = values["manifest-hash"];
    if ((pin === undefined) === (values["no-pin"] !== true)) throw new UsageError("fetch needs exactly one of --manifest-hash HEX or --no-pin");
    const repo = values.repo ?? ".";
    const state = deps.repoState(repo);
    if (state.dirty) throw new RunInputError("tree_dirty");
    const fetched = await fetchRunInputs({
      store: deps.openStore(settings, runInputLayout(runKey)),
      engine: deps.engine(),
      runKey,
      outDir: need(values.out, "out"),
      headCommit: () => state.commit,
      ...(pin === undefined ? {} : { expectedManifestHash: pin }),
    });
    deps.out(JSON.stringify({ manifest_hash: fetched.manifestHash, run_config_hash: fetched.runConfigHash, dataset_hash: fetched.datasetHash, files: fetched.files, dataset_dir: fetched.datasetDir }));
    return 0;
  } catch (error) {
    if (error instanceof RunInputError) deps.err(`run-inputs: ${error.code}: ${error.message}`);
    else if (error instanceof ObjectStoreError) deps.err(`run-inputs: store: ${error.code}: ${error.message}`);
    else if (error instanceof CredentialError) {
      deps.err(`run-inputs: ${error.code}: ${error.message}`);
      return 2;
    } else if (error instanceof UsageError) {
      deps.err(`run-inputs: ${error.message}`);
      return 2;
    } else deps.err("run-inputs: failed");
    return 1;
  }
}
