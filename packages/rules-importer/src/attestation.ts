import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

/**
 * The importer's trust check on a downloaded rules artifact (ADR-007): it
 * imports a snapshot or channel file only once the file is attested as
 * published by the rules-publish workflow on main.
 */
export interface AttestationVerifier {
  /** Resolves only when `file` is attested as required; else throws AttestationError. */
  verify(file: string): Promise<void>;
}

export class AttestationError extends Error {
  override name = "AttestationError";
}

/** Who must have signed the attestation. */
export interface AttestationPolicy {
  /** `owner/name` of the repository the workflow ran in. */
  readonly repo: string;
  /** `owner/name/path` of the workflow that signed, as `gh --signer-workflow` takes it. */
  readonly signerWorkflow: string;
  /** The git ref the signing run checked out. */
  readonly sourceRef: string;
}

/** The rules-publish workflow on main of piglor/areneon. */
export const RULES_PUBLISH_POLICY: AttestationPolicy = {
  repo: "piglor/areneon",
  signerWorkflow: "piglor/areneon/.github/workflows/rules-publish.yml",
  sourceRef: "refs/heads/main",
};

/** The OIDC issuer of every GitHub Actions signing certificate. */
const ACTIONS_ISSUER = "https://token.actions.githubusercontent.com";
const STATEMENT_TYPE = "https://in-toto.io/Statement/v1";
const PROVENANCE_TYPE = "https://slsa.dev/provenance/v1";

export interface CommandResult {
  /** The exit code, or null when a signal ended the process. */
  readonly code: number | null;
  readonly stdout: string;
  readonly stderr: string;
}

/** Runs one command without a shell. Rejects only when it cannot be started. */
export type CommandRunner = (command: string, args: readonly string[]) => Promise<CommandResult>;

export const spawnRunner: CommandRunner = (command, args) =>
  new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
    child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
    child.on("error", reject);
    child.on("close", (code) =>
      resolve({
        code,
        stdout: Buffer.concat(stdout).toString("utf8"),
        stderr: Buffer.concat(stderr).toString("utf8"),
      }),
    );
  });

export interface GhAttestationVerifierOptions {
  readonly run?: CommandRunner;
  /** The gh executable; default `gh` on PATH. */
  readonly gh?: string;
  readonly policy?: AttestationPolicy;
}

/**
 * Verifies with `gh attestation verify`, then re-checks gh's JSON itself:
 * a zero exit code alone is never trusted. Every returned result must name
 * the policy's repository, signer workflow and source ref and the file's
 * SHA-256 as a subject. The attested commit is deliberately not compared
 * with an artifact's `source_commit`: a later run (bootstrap, or catch-up
 * after a cancelled run) may release a channel file, and its attestation
 * names the commit that run checked out.
 */
export class GhAttestationVerifier implements AttestationVerifier {
  readonly #run: CommandRunner;
  readonly #gh: string;
  readonly #policy: AttestationPolicy;

  constructor(options: GhAttestationVerifierOptions = {}) {
    this.#run = options.run ?? spawnRunner;
    this.#gh = options.gh ?? "gh";
    this.#policy = options.policy ?? RULES_PUBLISH_POLICY;
  }

  async verify(file: string): Promise<void> {
    const digest = await sha256File(file);
    const { repo, signerWorkflow, sourceRef } = this.#policy;
    const args = [
      "attestation",
      "verify",
      "--repo",
      repo,
      "--signer-workflow",
      signerWorkflow,
      "--source-ref",
      sourceRef,
      "--format",
      "json",
      "--",
      file,
    ];
    let result: CommandResult;
    try {
      result = await this.#run(this.#gh, args);
    } catch (cause) {
      throw new AttestationError(`${file}: could not run ${this.#gh}: ${String(cause)}`, { cause });
    }
    if (result.code !== 0) {
      throw new AttestationError(
        `${file}: gh attestation verify exited ${String(result.code)}: ${result.stderr.trim()}`,
      );
    }
    checkResults(file, digest, parseJson(file, result.stdout), this.#policy);
  }
}

async function sha256File(file: string): Promise<string> {
  try {
    return createHash("sha256")
      .update(await readFile(file))
      .digest("hex");
  } catch (cause) {
    throw new AttestationError(`${file}: cannot read: ${String(cause)}`, { cause });
  }
}

function parseJson(file: string, stdout: string): unknown {
  try {
    return JSON.parse(stdout);
  } catch (cause) {
    throw new AttestationError(`${file}: gh printed malformed JSON: ${String(cause)}`, { cause });
  }
}

function checkResults(file: string, digest: string, results: unknown, policy: AttestationPolicy): void {
  if (!Array.isArray(results) || results.length === 0) {
    throw new AttestationError(`${file}: gh returned no verified attestation`);
  }
  const repoUri = `https://github.com/${policy.repo}`;
  const signerUri = `https://github.com/${policy.signerWorkflow}@${policy.sourceRef}`;
  const expected: readonly (readonly [string, string])[] = [
    ["verificationResult.signature.certificate.issuer", ACTIONS_ISSUER],
    ["verificationResult.signature.certificate.sourceRepositoryURI", repoUri],
    ["verificationResult.signature.certificate.githubWorkflowRepository", policy.repo],
    ["verificationResult.signature.certificate.subjectAlternativeName", signerUri],
    ["verificationResult.signature.certificate.buildSignerURI", signerUri],
    ["verificationResult.signature.certificate.sourceRepositoryRef", policy.sourceRef],
    ["verificationResult.signature.certificate.githubWorkflowRef", policy.sourceRef],
    ["verificationResult.statement._type", STATEMENT_TYPE],
    ["verificationResult.statement.predicateType", PROVENANCE_TYPE],
  ];
  results.forEach((result: unknown, i) => {
    for (const [path, want] of expected) {
      const got = field(result, path);
      if (got !== want) {
        throw new AttestationError(`${file}: result ${i}: ${path} is ${JSON.stringify(got)}, expected ${want}`);
      }
    }
    const subjects = field(result, "verificationResult.statement.subject");
    if (!Array.isArray(subjects) || !subjects.some((s: unknown) => field(s, "digest.sha256") === digest)) {
      throw new AttestationError(`${file}: result ${i}: no statement subject has sha256 ${digest}`);
    }
  });
}

/** The value at a dotted path, or undefined where an object is missing. */
function field(value: unknown, path: string): unknown {
  let current = value;
  for (const key of path.split(".")) {
    if (typeof current !== "object" || current === null) return undefined;
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}
