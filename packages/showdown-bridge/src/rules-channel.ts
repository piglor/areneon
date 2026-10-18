import type { ChannelFile } from "./rules.ts";

export interface ChannelCheck {
  /** The channel file at the commit under test. */
  head: ChannelFile;
  /** The same file at the base commit, or `undefined` if it did not exist. */
  base: ChannelFile | undefined;
  /**
   * Snapshots the channel may name, mapped to the Showdown commit each was
   * built for: the build of the commit under test plus every earlier `main`
   * commit that changed the rules, each rebuilt with its own builder.
   */
  allowed: ReadonlyMap<string, string>;
  /** The engine commit pinned at the commit under test. */
  pin: string;
}

/**
 * ADR-007 channel rules, checked offline in CI. Returns the problems found;
 * empty means the channel file is acceptable. A channel may only name a
 * snapshot built for the current pin, and every change raises `seq` by
 * exactly 1 so the importer can order changes without trusting arrival time.
 */
export function checkChannel({ head, base, allowed, pin }: ChannelCheck): string[] {
  const problems: string[] = [];
  if (base === undefined) {
    if (head.seq !== 1) problems.push(`a new channel starts at seq 1, not ${head.seq}`);
  } else if (head.hash === base.hash) {
    if (head.seq !== base.seq) problems.push(`seq changed from ${base.seq} without a new hash`);
  } else if (head.seq !== base.seq + 1) {
    problems.push(`seq must go from ${base.seq} to ${base.seq + 1}, not ${head.seq}`);
  }
  const builtFor = allowed.get(head.hash);
  if (builtFor === undefined) {
    problems.push(`${head.hash} is not a snapshot built from this commit or an earlier main commit`);
  } else if (builtFor !== pin) {
    problems.push(`${head.hash} was built for Showdown ${builtFor}, not the pinned ${pin}`);
  }
  return problems;
}
