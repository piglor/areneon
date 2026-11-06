// The shared object store client (ADR-024 section 3.6): the `ObjectStore`
// port, the R2 client (path-style SigV4, environment-only credentials, optional
// session token), the in-memory store for tests, and the key-layout policy each
// user injects. It admits no key its user did not declare.
export {
  assertLayout,
  checkCreateKey,
  checkKey,
  checkPrefix,
  MAX_OBJECT_BYTES,
  OBJECT_STORE_REFUSALS,
  type ObjectHead,
  type ObjectLayout,
  type ObjectStore,
  ObjectStoreError,
  type ObjectStoreRefusal,
} from "./object-store.ts";
export { InMemoryObjectStore } from "./memory-store.ts";
export { type FetchLike, R2ObjectStore, type R2ClientOptions } from "./r2-client.ts";
export { CREDENTIAL_REFUSALS, CredentialError, type CredentialRefusal, R2Settings, type SignedRequestInit } from "./r2-settings.ts";
