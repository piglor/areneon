// The R2 credentials and endpoint of the private store (ADR-015 section 1,
// docs/runbooks/private-store.md section 6). They come from the environment of
// the one command that needs them and from nowhere else: AWS_ACCESS_KEY_ID,
// AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL and AWS_DEFAULT_REGION. No file, no
// option, no cache, no profile chain; and a command line that carries any of
// them (a flag naming one, or one of their values) is refused before the
// environment is looked at, because an argument shows in `ps` and in history.
//
// `R2Settings` is the only holder of the values. Its fields are private
// (`#`), it prints as a fixed word whatever inspects it (String, JSON,
// util.inspect), and it signs requests itself, so the key, the secret and the
// endpoint (which contains the account id) never leave it except inside the
// signing library. `CredentialError` has fixed messages: no value, and no
// variable's content, is ever in one.

import { createHash } from "node:crypto";
import { inspect } from "node:util";
import { AwsClient } from "aws4fetch";

export const CREDENTIAL_REFUSALS = ["credentials_on_argv", "credentials_missing", "credentials_invalid"] as const;
export type CredentialRefusal = (typeof CREDENTIAL_REFUSALS)[number];

const MESSAGES: Readonly<Record<CredentialRefusal, string>> = {
  credentials_on_argv: "the R2 credentials, endpoint or region were given on the command line; they are read from the environment only",
  credentials_missing: "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL and AWS_DEFAULT_REGION must all be set in the environment of this command",
  credentials_invalid: "an R2 credential, the endpoint or the region has a form the private store does not accept",
};

export class CredentialError extends Error {
  override name = "CredentialError";
  readonly code: CredentialRefusal;

  constructor(code: CredentialRefusal) {
    super(`R2 settings refused: ${MESSAGES[code]}`);
    this.code = code;
  }
}

const VARIABLES = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "AWS_DEFAULT_REGION"] as const;
/** An argument that names a credential setting: a variable, or a flag that looks like one. */
const ARGV_NAMES = /(?:^|[^A-Za-z0-9])AWS_[A-Z_]+|^--?(?:access[-_]?key|secret|token|credential|endpoint|account|region|password)/i;
const KEY_ID = /^[A-Za-z0-9]{16,128}$/;
const SECRET = /^[A-Za-z0-9/+=_-]{16,256}$/;
const ENDPOINT_HOST = /^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9-]+)*\.r2\.cloudflarestorage\.com$/;
const BUCKET = /^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/;
const EMPTY_SHA256 = createHash("sha256").update("").digest("hex");

export const sha256Hex = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");

export interface SignedRequestInit {
  readonly method: "GET" | "HEAD" | "PUT";
  /** Object key, or null for the bucket itself (a listing). */
  readonly key: string | null;
  readonly query?: Readonly<Record<string, string>>;
  readonly body?: Uint8Array;
  readonly headers?: Readonly<Record<string, string>>;
  readonly signal?: AbortSignal;
}

export class R2Settings {
  readonly #client: AwsClient;
  readonly #origin: string;
  readonly #bucket: string;

  private constructor(client: AwsClient, origin: string, bucket: string) {
    this.#client = client;
    this.#origin = origin;
    this.#bucket = bucket;
  }

  /**
   * Read the settings. `argv` is the command line to check (the program's
   * arguments after the script name); `bucket` is a name, not a credential.
   * Throws `CredentialError` with a fixed message.
   */
  static fromEnvironment(env: Readonly<Record<string, string | undefined>>, argv: readonly string[], bucket: string): R2Settings {
    if (argv.some((arg) => ARGV_NAMES.test(arg))) throw new CredentialError("credentials_on_argv");
    const values = VARIABLES.map((name) => env[name]);
    if (values.some((v) => v === undefined || v === "")) throw new CredentialError("credentials_missing");
    const [id, secret, endpoint, region] = values as [string, string, string, string];
    // A value passed as a bare argument (no flag) is as visible as a flag.
    // The account id is the endpoint's first label.
    const account = /^https?:\/\/([^./:?#]+)\./.exec(endpoint)?.[1] ?? "";
    if (argv.some((arg) => [id, secret, endpoint, ...(account.length >= 8 ? [account] : [])].some((v) => arg.includes(v)))) throw new CredentialError("credentials_on_argv");
    if (!KEY_ID.test(id) || !SECRET.test(secret) || region !== "auto" || !BUCKET.test(bucket)) throw new CredentialError("credentials_invalid");
    let url: URL;
    try {
      url = new URL(endpoint);
    } catch {
      throw new CredentialError("credentials_invalid");
    }
    const bare = url.protocol === "https:" && url.username === "" && url.password === "" && url.port === "" && (url.pathname === "/" || url.pathname === "") && url.search === "" && url.hash === "";
    if (!bare || !ENDPOINT_HOST.test(url.hostname)) throw new CredentialError("credentials_invalid");
    return new R2Settings(new AwsClient({ accessKeyId: id, secretAccessKey: secret, service: "s3", region, retries: 0 }), url.origin, bucket);
  }

  /** A signed request: SigV4 headers, `x-amz-content-sha256` of the real body, path-style URL, no redirects. */
  sign(init: SignedRequestInit): Promise<Request> {
    const url = new URL(`${this.#origin}/${this.#bucket}${init.key === null ? "" : `/${init.key}`}`);
    for (const [name, value] of Object.entries(init.query ?? {})) url.searchParams.set(name, value);
    const headers = new Headers(init.headers ?? {});
    headers.set("x-amz-content-sha256", init.body === undefined ? EMPTY_SHA256 : sha256Hex(init.body));
    if (init.body !== undefined) headers.set("content-type", "application/octet-stream");
    return this.#client.sign(url.toString(), {
      method: init.method,
      headers,
      ...(init.body === undefined ? {} : { body: init.body as BodyInit }),
      ...(init.signal === undefined ? {} : { signal: init.signal }),
      redirect: "error",
    });
  }

  toString(): string {
    return "[R2Settings]";
  }

  toJSON(): string {
    return "[R2Settings]";
  }

  [inspect.custom](): string {
    return "[R2Settings]";
  }
}
