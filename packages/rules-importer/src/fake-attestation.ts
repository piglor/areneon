import { AttestationError, type AttestationVerifier } from "./attestation.ts";

/**
 * An AttestationVerifier for tests of the modules that use one: it accepts
 * exactly the paths it was given and records every call, without gh.
 */
export class FakeAttestationVerifier implements AttestationVerifier {
  readonly calls: string[] = [];
  readonly #attested: ReadonlySet<string>;

  constructor(attested: Iterable<string>) {
    this.#attested = new Set(attested);
  }

  async verify(file: string): Promise<void> {
    this.calls.push(file);
    if (!this.#attested.has(file)) {
      throw new AttestationError(`${file}: not attested (fake verifier)`);
    }
  }
}
