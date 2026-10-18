// The Sha256 port on WebCrypto (ADR-008 §5): what the advisor extension
// passes to the ObservationBuilder. Only a type import, so it bundles as is; Node 24
// has the same `crypto.subtle`, so CI checks it against `nodeSha256`.
import type { Sha256 } from "./observation.ts";

export const webSha256: Sha256 = async (bytes) => {
  // A copy, so the digest reads an ArrayBuffer-backed view (BufferSource).
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new Uint8Array(bytes)));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
};
