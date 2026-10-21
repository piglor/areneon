import { inspect } from "node:util";
import { describe, expect, it } from "vitest";
import { CredentialError, R2Settings } from "../src/r2-settings.ts";
import { ACCOUNT, BUCKET, ENDPOINT, ENV, KEY_ID, SECRET, SECRETS } from "./r2-support.ts";

const code = (run: () => unknown): string => {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(CredentialError);
    return (error as CredentialError).code;
  }
  return "no error";
};
const from = (env: Record<string, string | undefined>, argv: string[] = [], bucket = BUCKET): R2Settings => R2Settings.fromEnvironment(env, argv, bucket);

describe("R2 settings come from the environment only", () => {
  it("reads the four variables and a bucket name", () => {
    expect(from(ENV)).toBeInstanceOf(R2Settings);
    expect(from(ENV, ["--parent", "ab".repeat(32), "team.txt", "--first"])).toBeInstanceOf(R2Settings);
  });

  it.each(Object.keys(ENV))("refuses when %s is missing or empty", (name) => {
    expect(code(() => from({ ...ENV, [name]: undefined }))).toBe("credentials_missing");
    expect(code(() => from({ ...ENV, [name]: "" }))).toBe("credentials_missing");
  });

  it("refuses values of a form the store does not accept", () => {
    expect(code(() => from({ ...ENV, AWS_ACCESS_KEY_ID: "short" }))).toBe("credentials_invalid");
    expect(code(() => from({ ...ENV, AWS_ACCESS_KEY_ID: `${KEY_ID} ` }))).toBe("credentials_invalid");
    expect(code(() => from({ ...ENV, AWS_SECRET_ACCESS_KEY: "has space in it 123456" }))).toBe("credentials_invalid");
    expect(code(() => from({ ...ENV, AWS_DEFAULT_REGION: "us-east-1" }))).toBe("credentials_invalid");
    expect(code(() => from(ENV, [], "Bad Bucket"))).toBe("credentials_invalid");
    expect(code(() => from(ENV, [], ""))).toBe("credentials_invalid");
    for (const endpoint of [
      "not a url",
      `http://${ACCOUNT}.r2.cloudflarestorage.com`,
      `https://user:pw@${ACCOUNT}.r2.cloudflarestorage.com`,
      `https://${ACCOUNT}.r2.cloudflarestorage.com:8443`,
      `${ENDPOINT}/path`,
      `${ENDPOINT}?x=1`,
      `${ENDPOINT}#x`,
      "https://example.com",
      "https://evil.example/.r2.cloudflarestorage.com",
      `https://${ACCOUNT}.r2.cloudflarestorage.com.evil.example`,
    ]) {
      expect(code(() => from({ ...ENV, AWS_ENDPOINT_URL: endpoint })), endpoint).toBe("credentials_invalid");
    }
    expect(from({ ...ENV, AWS_ENDPOINT_URL: `https://${ACCOUNT}.eu.r2.cloudflarestorage.com/` })).toBeInstanceOf(R2Settings);
  });

  it("refuses a command line that names a credential setting or carries a value, before it reads the environment", () => {
    const argvs = [
      ["--access-key-id", "x"],
      ["--access-key=x"],
      ["--secret-access-key", "x"],
      ["--secret", "x"],
      ["-token", "x"],
      ["--endpoint", "x"],
      ["--endpoint-url=x"],
      ["--account-id", "x"],
      ["--region", "auto"],
      ["--password=x"],
      ["--credentials", "x"],
      ["AWS_ACCESS_KEY_ID=abc"],
      ["aws_secret_access_key=abc"],
      ["--env", "AWS_DEFAULT_REGION=auto"],
      ["x", KEY_ID],
      [`--thing=${SECRET}`],
      [`prefix-${ACCOUNT}-suffix`],
      [ENDPOINT],
    ];
    for (const argv of argvs) {
      expect(code(() => from(ENV, argv)), argv.join(" ")).toBe("credentials_on_argv");
      expect(code(() => from({}, argv)), `${argv.join(" ")} (empty env)`).toBe(argv.some((a) => /AWS_|^--?(access|secret|token|credential|endpoint|account|region|password)/i.test(a)) ? "credentials_on_argv" : "credentials_missing");
    }
  });

  it("checks the account id only when it is long enough to be one", () => {
    const short = { ...ENV, AWS_ENDPOINT_URL: "https://acct.r2.cloudflarestorage.com" };
    expect(from(short, ["acct.txt"])).toBeInstanceOf(R2Settings);
    expect(code(() => from(short, ["https://acct.r2.cloudflarestorage.com"]))).toBe("credentials_on_argv");
  });

  it("does not mistake an ordinary word for a credential flag", () => {
    expect(from(ENV, ["laws_of_teams.txt", "--parent", "--retire", "ab".repeat(16), "tokens.txt".replace("tokens", "t0kens")])).toBeInstanceOf(R2Settings);
  });

  it("prints as a fixed word to every inspector and has fixed messages", () => {
    const s = from(ENV);
    for (const text of [String(s), JSON.stringify(s), inspect(s, { depth: 9, showHidden: true }), JSON.stringify({ s }), `${s}`]) {
      expect(text).toContain("R2Settings");
      for (const secret of SECRETS) expect(text).not.toContain(secret);
    }
    const error = (() => {
      try {
        from({ ...ENV, AWS_SECRET_ACCESS_KEY: `${SECRET} bad` });
      } catch (e) {
        return e as CredentialError;
      }
      throw new Error("expected a refusal");
    })();
    for (const secret of SECRETS) expect(error.message + (error.stack ?? "") + inspect(error, { showHidden: true })).not.toContain(secret);
  });
});
