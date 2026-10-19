import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  AttestationError,
  type CommandResult,
  type CommandRunner,
  FakeAttestationVerifier,
  GhAttestationVerifier,
  RULES_PUBLISH_POLICY,
  spawnRunner,
} from "../src/index.ts";

const fixture = (name: string): string =>
  fileURLToPath(new URL(`../../../fixtures/rules-releases/${name}`, import.meta.url));

// The two release assets of the first rules-publish run, and the real
// `gh attestation verify --format json` output for them (one attestation
// names both files, so gh printed the same JSON for each), trimmed to the
// fields the verifier reads plus the attested commit.
const SNAPSHOT = fixture("rules-snapshot-7c618d78d6f569e875d7e5be4863152599dc8f4207f6e10cfd5f12301a853fe8.json");
const CHANNEL = fixture("rules-channel-stable-b226c3bfe889f8cf775933b0a964a44ef6d69913.json");
const GH_JSON = await readFile(fixture("gh-verify-rules-publish.json"), "utf8");
// Real gh stderr (exit 1) for a file no attestation names.
const GH_UNSIGNED = await readFile(fixture("gh-verify-unsigned.stderr"), "utf8");

const CERT = "verificationResult.signature.certificate";
const SIGNER = "https://github.com/piglor/areneon/.github/workflows/rules-publish.yml@refs/heads/main";

interface Call {
  command: string;
  args: readonly string[];
}

function runner(result: Partial<CommandResult>): { run: CommandRunner; calls: Call[] } {
  const calls: Call[] = [];
  const run: CommandRunner = async (command, args) => {
    calls.push({ command, args });
    return { code: 0, stdout: "", stderr: "", ...result };
  };
  return { run, calls };
}

/** The recorded gh JSON with one dotted path set to `value` in result `index`. */
function edited(path: string, value: unknown, index = 0): string {
  const results = JSON.parse(GH_JSON) as Record<string, unknown>[];
  const keys = path.split(".");
  const last = keys.pop() as string;
  let node = results[index] as Record<string, unknown>;
  for (const key of keys) node = node[key] as Record<string, unknown>;
  node[last] = value;
  return JSON.stringify(results);
}

async function refusal(stdout: string, file = CHANNEL): Promise<string> {
  const verifier = new GhAttestationVerifier({ run: runner({ stdout }).run });
  const error = await verifier.verify(file).then(
    () => undefined,
    (e: unknown) => e,
  );
  expect(error).toBeInstanceOf(AttestationError);
  return (error as AttestationError).message;
}

describe("GhAttestationVerifier", () => {
  it("accepts the real gh output for both published release assets", async () => {
    for (const file of [SNAPSHOT, CHANNEL]) {
      const { run, calls } = runner({ stdout: GH_JSON });
      await expect(new GhAttestationVerifier({ run }).verify(file)).resolves.toBeUndefined();
      expect(calls).toEqual([
        {
          command: "gh",
          args: [
            "attestation",
            "verify",
            "--repo",
            "piglor/areneon",
            "--signer-workflow",
            "piglor/areneon/.github/workflows/rules-publish.yml",
            "--source-ref",
            "refs/heads/main",
            "--format",
            "json",
            "--",
            file,
          ],
        },
      ]);
    }
  });

  it("does not require the attested commit to be the channel's source_commit", async () => {
    const channel = JSON.parse(await readFile(CHANNEL, "utf8")) as { source_commit: string };
    const [result] = JSON.parse(GH_JSON) as {
      verificationResult: { signature: { certificate: Record<string, string> } };
    }[];
    const attested = result?.verificationResult.signature.certificate.sourceRepositoryDigest;
    expect(attested).toBe("35f9c5f04aecbe1dfc38e55d2599b1013f1b43b7");
    expect(attested).not.toBe(channel.source_commit);
    const { run } = runner({ stdout: GH_JSON });
    await expect(new GhAttestationVerifier({ run }).verify(CHANNEL)).resolves.toBeUndefined();
  });

  it.each([
    ["sourceRepositoryURI", "https://github.com/evil/areneon"],
    ["githubWorkflowRepository", "evil/areneon"],
  ])("refuses a wrong repository in %s", async (key, value) => {
    expect(await refusal(edited(`${CERT}.${key}`, value))).toContain(`${CERT}.${key} is "${value}"`);
  });

  it.each([
    ["buildSignerURI", "https://github.com/piglor/areneon/.github/workflows/ci.yml@refs/heads/main"],
    ["subjectAlternativeName", "https://github.com/piglor/areneon/.github/workflows/ci.yml@refs/heads/main"],
    ["buildSignerURI", `${SIGNER}-evil`],
    ["buildSignerURI", "https://github.com/piglor/areneon/.github/workflows/rules-publish.yml@refs/heads/dev"],
  ])("refuses a wrong signer workflow in %s = %s", async (key, value) => {
    expect(await refusal(edited(`${CERT}.${key}`, value))).toContain(`${CERT}.${key}`);
  });

  it.each([
    ["sourceRepositoryRef", "refs/heads/dev"],
    ["githubWorkflowRef", "refs/pull/1/merge"],
  ])("refuses a wrong source ref in %s", async (key, value) => {
    expect(await refusal(edited(`${CERT}.${key}`, value))).toContain(`${CERT}.${key} is "${value}"`);
  });

  it("refuses an attestation signed on a self-hosted runner", async () => {
    expect(await refusal(edited(`${CERT}.runnerEnvironment`, "self-hosted"))).toContain(
      `${CERT}.runnerEnvironment is "self-hosted", expected github-hosted`,
    );
  });

  it("refuses a certificate from another issuer", async () => {
    expect(await refusal(edited(`${CERT}.issuer`, "https://accounts.example.com"))).toContain(`${CERT}.issuer`);
  });

  it("refuses a statement that is not SLSA provenance in in-toto v1", async () => {
    expect(await refusal(edited("verificationResult.statement.predicateType", "https://example.com/x"))).toContain(
      "predicateType",
    );
    expect(await refusal(edited("verificationResult.statement._type", "https://in-toto.io/Statement/v0.1"))).toContain(
      "_type",
    );
  });

  it("refuses a result with a missing or non-object field", async () => {
    expect(await refusal(edited("verificationResult.signature", null))).toContain(`${CERT}.issuer is undefined`);
    expect(await refusal(JSON.stringify([7]))).toContain("result 0");
  });

  it("refuses when any one of several results is wrong", async () => {
    const good = JSON.parse(GH_JSON) as unknown[];
    const bad = JSON.parse(edited(`${CERT}.sourceRepositoryRef`, "refs/heads/dev")) as unknown[];
    expect(await refusal(JSON.stringify([...good, ...bad]))).toContain("result 1");
  });

  it("refuses a file whose digest no statement subject names", async () => {
    const dir = await mkdtemp(join(tmpdir(), "rules-importer-"));
    const tampered = join(dir, "rules-channel-stable-b226c3bfe889f8cf775933b0a964a44ef6d69913.json");
    await writeFile(tampered, `${await readFile(CHANNEL, "utf8")} `);
    expect(await refusal(GH_JSON, tampered)).toContain("no statement subject has sha256");
    expect(await refusal(edited("verificationResult.statement.subject", "x"))).toContain("no statement subject");
    expect(await refusal(edited("verificationResult.statement.subject", [{ name: "x" }]))).toContain(
      "no statement subject",
    );
  });

  it("refuses an unsigned file when gh exits non-zero, quoting its stderr", async () => {
    const { run } = runner({ code: 1, stderr: GH_UNSIGNED });
    const verify = new GhAttestationVerifier({ run }).verify(CHANNEL);
    await expect(verify).rejects.toThrow(AttestationError);
    await expect(verify).rejects.toThrow(/exited 1: Error: HTTP 404: Not Found/);
  });

  it("refuses a non-zero exit even when gh printed valid JSON", async () => {
    const { run } = runner({ code: 1, stdout: GH_JSON });
    await expect(new GhAttestationVerifier({ run }).verify(CHANNEL)).rejects.toThrow(AttestationError);
    const signalled = runner({ code: null, stdout: GH_JSON });
    await expect(new GhAttestationVerifier({ run: signalled.run }).verify(CHANNEL)).rejects.toThrow(/exited null/);
  });

  it("refuses malformed JSON, a non-list and an empty result list", async () => {
    expect(await refusal('[{"verificationResult":')).toContain("malformed JSON");
    expect(await refusal("{}")).toContain("no verified attestation");
    expect(await refusal("[]")).toContain("no verified attestation");
  });

  it("refuses when gh cannot be started", async () => {
    const run: CommandRunner = () => Promise.reject(new Error("spawn gh ENOENT"));
    await expect(new GhAttestationVerifier({ run }).verify(CHANNEL)).rejects.toThrow(
      /could not run gh: Error: spawn gh ENOENT/,
    );
  });

  it("refuses a file it cannot read, without calling gh", async () => {
    const { run, calls } = runner({ stdout: GH_JSON });
    await expect(new GhAttestationVerifier({ run }).verify(fixture("missing.json"))).rejects.toThrow(/cannot read/);
    expect(calls).toEqual([]);
  });

  it("checks against a given policy", async () => {
    const { run, calls } = runner({ stdout: GH_JSON });
    const policy = { ...RULES_PUBLISH_POLICY, sourceRef: "refs/heads/release" };
    await expect(new GhAttestationVerifier({ run, policy, gh: "/usr/bin/gh" }).verify(CHANNEL)).rejects.toThrow(
      /subjectAlternativeName is ".*@refs\/heads\/main", expected .*@refs\/heads\/release$/,
    );
    expect(calls[0]?.command).toBe("/usr/bin/gh");
    expect(calls[0]?.args).toContain("refs/heads/release");
  });

  it("runs gh through the spawn runner by default", async () => {
    // node stands in for gh: it cannot load the script `attestation`, exits 1.
    const verify = new GhAttestationVerifier({ gh: process.execPath }).verify(CHANNEL);
    await expect(verify).rejects.toThrow(/exited 1: .*attestation/s);
  });
});

describe("spawnRunner", () => {
  it("returns the exit code, stdout and stderr", async () => {
    const script = "process.stdout.write('out'); process.stderr.write('err'); process.exitCode = 3";
    await expect(spawnRunner(process.execPath, ["-e", script])).resolves.toEqual({
      code: 3,
      stdout: "out",
      stderr: "err",
    });
  });

  it("rejects when the command cannot be started", async () => {
    await expect(spawnRunner(join(tmpdir(), "no-such-gh-binary"), [])).rejects.toThrow(/ENOENT/);
  });
});

describe("FakeAttestationVerifier", () => {
  it("accepts only the given paths and records each call", async () => {
    const fake = new FakeAttestationVerifier([SNAPSHOT]);
    await expect(fake.verify(SNAPSHOT)).resolves.toBeUndefined();
    await expect(fake.verify(CHANNEL)).rejects.toThrow(AttestationError);
    expect(fake.calls).toEqual([SNAPSHOT, CHANNEL]);
  });
});
