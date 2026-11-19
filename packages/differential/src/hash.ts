// SHA-256 over JCS bytes: every hash the harness names (the fresh seed, the shadow key, admission_hash).
import { createHash } from "node:crypto";
import { type Json, jcs } from "./canonical.ts";

/** Hex SHA-256 of the JCS bytes of `value`. */
export const sha256Jcs = (value: Json): string => createHash("sha256").update(jcs(value)).digest("hex");

/** The first 8 bytes, big endian, of the SHA-256 of the JCS bytes of `value` (the bottom-k key of #1195). */
export const key64 = (value: Json): bigint => createHash("sha256").update(jcs(value)).digest().readBigUInt64BE(0);
