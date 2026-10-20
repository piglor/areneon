// Where the owner-team inventory (#775, ADR-014 section 11) gets its teams.
// The port is the narrowest thing a consumer needs: a name and a paste per
// team, one at a time. It assumes no file system, no network and no crypto,
// so an in-memory or a decrypting source can stand behind it. Only
// `PasteDirectorySource` is here: it reads `private-data/teams/*.txt` and
// keeps the `private-data` path check. An error names a file, never its content.

import { lstatSync, readdirSync, readFileSync, realpathSync } from "node:fs";
import { basename, dirname, join, resolve, sep } from "node:path";

/** The only directory name teams are read from. */
export const PRIVATE_DATA = "private-data";

export class InventoryError extends Error {
  override name = "InventoryError";
}

export interface PrivateTeamEntry {
  /** A label for messages: a base name, never a path and never team content. */
  name: string;
  /** The team's PokePaste text. Private: it must never reach stdout, a log or an error. */
  paste: string;
}

/** A source of private teams. A consumer reads each entry once, in the order given. */
export interface PrivateTeamSource {
  entries(): AsyncIterable<PrivateTeamEntry>;
}

/**
 * The `private-data` directory `path` lies in or is: the longest prefix of
 * its real path ending in a segment named `private-data`. A symlink is
 * followed first, so a link cannot lead out. Throws if there is none.
 */
export function privateDataRoot(path: string): string {
  const real = realpathOfDeepest(resolve(path));
  const parts = real.split(sep);
  const at = parts.lastIndexOf(PRIVATE_DATA);
  if (at < 1) throw new InventoryError(`refusing a path outside a ${PRIVATE_DATA}/ directory`);
  return parts.slice(0, at + 1).join(sep);
}

/** The real path of `path`, resolving the deepest part that exists. */
function realpathOfDeepest(path: string): string {
  let existing = path;
  const rest: string[] = [];
  for (;;) {
    try {
      return join(realpathSync(existing), ...rest.reverse());
    } catch {
      rest.push(basename(existing));
      existing = dirname(existing); // the filesystem root always exists, so this ends
    }
  }
}

/** The `*.txt` files directly in `teamsDir`, sorted; each must really lie inside `private-data/`. */
export function teamFiles(teamsDir: string): string[] {
  const root = privateDataRoot(teamsDir);
  let names: string[];
  try {
    names = readdirSync(teamsDir).filter((name) => name.endsWith(".txt")).sort();
  } catch {
    throw new InventoryError("the teams directory cannot be read");
  }
  const files = names.map((name) => join(teamsDir, name));
  for (const [i, file] of files.entries()) {
    const real = realpathSync(file);
    if (!real.startsWith(root + sep) || !lstatSync(real).isFile()) {
      throw new InventoryError(`refusing ${names[i]}: not a regular file inside ${PRIVATE_DATA}/`);
    }
  }
  if (files.length === 0) throw new InventoryError("no .txt team files in the teams directory");
  return files;
}

/** Today's reader: the `*.txt` files of one directory inside `private-data/`. */
export class PasteDirectorySource implements PrivateTeamSource {
  readonly #teamsDir: string;

  constructor(teamsDir: string) {
    this.#teamsDir = teamsDir;
  }

  async *entries(): AsyncGenerator<PrivateTeamEntry> {
    for (const file of teamFiles(this.#teamsDir)) {
      let paste: string;
      try {
        paste = readFileSync(file, "utf8");
      } catch {
        throw new InventoryError(`${basename(file)} cannot be read`);
      }
      yield { name: basename(file), paste };
    }
  }
}
