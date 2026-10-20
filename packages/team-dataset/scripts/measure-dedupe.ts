// Measure the memory of the dedupe index (ADR-013 section 2, "Not verified":
// the prefix-set estimate). For each size it builds the index over that many
// random 256-bit hashes (the sorted 64-bit prefix array and the exact stage
// in memory) and prints the prefix array's size, the array-buffer memory the
// whole index holds (prefixes plus the exact stage's 32 bytes per hash), the
// build time, the cost of a lookup of a known and of an unknown hash, and how
// often the exact stage was asked:
//   node --expose-gc packages/team-dataset/scripts/measure-dedupe.ts 1000000 5000000
import { randomBytes } from "node:crypto";
import { KnownHashes, sortedUnique } from "../src/dedupe.ts";

const gc = (globalThis as { gc?: () => void }).gc;
for (const size of process.argv.slice(2).map(Number)) {
  const sorted = sortedUnique(Array.from({ length: size }, () => randomBytes(32).toString("hex")));
  gc?.();
  const before = process.memoryUsage().arrayBuffers;
  const built = process.hrtime.bigint();
  const index = new KnownHashes(sorted);
  const buildMs = Number(process.hrtime.bigint() - built) / 1e6;
  gc?.();
  const held = process.memoryUsage().arrayBuffers - before;
  const probes = 100_000;
  const unknown = Array.from({ length: probes }, () => randomBytes(32).toString("hex"));
  const timeIt = (run: () => void): number => {
    const start = process.hrtime.bigint();
    run();
    return Number(process.hrtime.bigint() - start) / 1e3 / probes;
  };
  const knownUs = timeIt(() => sorted.slice(0, probes).forEach((h) => index.has(h)));
  const checksAfterKnown = index.exactChecks;
  const unknownUs = timeIt(() => unknown.forEach((h) => index.has(h)));
  console.log(
    JSON.stringify({
      hashes: size,
      prefix_array_mb: index.prefixBytes / 1e6,
      bytes_per_hash: index.prefixBytes / size,
      index_array_buffers_mb: Math.round(held / 1e6),
      build_ms: Math.round(buildMs),
      known_lookup_us: Number(knownUs.toFixed(2)),
      unknown_lookup_us: Number(unknownUs.toFixed(2)),
      exact_checks_for_unknown: index.exactChecks - checksAfterKnown,
    }),
  );
}
