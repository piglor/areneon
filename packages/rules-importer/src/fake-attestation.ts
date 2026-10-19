import { AttestationError, type AttestationVerifier } from "./attestation.ts";

/**
 * An AttestationVerifier for tests of the modules that use one: it accepts
 * exactly the paths it was given, returns the bytes given for each, and
 * records every call, without gh or the file system.
 */
export class FakeAttestationVerifier implements AttestationVerifier {
  readonly calls: string[] = [];
  readonly #attested: ReadonlyMap<string, Uint8Array>;

  constructor(attested: Iterable<readonly [string, Uint8Array]>) {
    this.#attested = new Map(attested);
  }

  async verify(file: string): Promise<Uint8Array> {
    this.calls.push(file);
    const bytes = this.#attested.get(file);
    if (bytes === undefined) {
      throw new AttestationError(`${file}: not attested (fake verifier)`);
    }
    return bytes;
  }
}
