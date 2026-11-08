// The link from a further game process to the process that started it (#980, #1192).
//
// The starter holds the child's stdin pipe open. Two things travel over it: its closing (the starter died:
// the child ends, and the policy workers the child started have pipes to it and end with it), and, once the
// starter has loaded the team pool and written it for the children, one line `pool <sha256>` naming the
// pool file's bytes. The digest is not a secret; it is how the child knows the file is the one its starter
// vouched for and not another file left in a directory.

/** The part of a readable stream this module uses (`process.stdin`, or a stand-in in a test). */
export interface ParentPipe {
  setEncoding?: (encoding: "utf8") => unknown;
  on: (event: "data", listener: (chunk: string | Buffer) => void) => unknown;
  once: (event: "end", listener: () => void) => unknown;
  resume: () => unknown;
  unref?: () => void;
}

/** The line the starter sends once the pool file is written. */
export const poolLine = (digest: string): string => `pool ${digest}\n`;

/**
 * End this process with code 1 when the starter's pipe closes, and resolve with the pool file's digest when
 * the starter sends it. Register it before anything slow (the pool, the workers) so it is heard during them.
 * The promise never settles when the starter dies: the process ends instead.
 */
export function exitWhenParentDies(stdin: ParentPipe = process.stdin as unknown as ParentPipe, exit: (code: number) => void = process.exit.bind(process)): Promise<string> {
  const digest = new Promise<string>((resolve) => {
    let seen = "";
    stdin.on("data", (chunk) => {
      seen += chunk.toString();
      const line = /^pool ([0-9a-f]{64})\n/m.exec(seen);
      if (line !== null) resolve(line[1] as string);
    });
  });
  stdin.once("end", () => exit(1));
  stdin.resume();
  stdin.unref?.(); // an open pipe must not keep the finished process alive
  return digest;
}

/** What the starter does when the pool file is written: tell every further game process its digest. */
export const vouchTo =
  (children: readonly { tell: (line: string) => void }[]) =>
  (digest: string): void => {
    for (const child of children) child.tell(poolLine(digest));
  };
