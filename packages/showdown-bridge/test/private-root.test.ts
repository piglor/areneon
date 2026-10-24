import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import {
  assertPrivateRoot,
  onWindowsMount,
  placeRefusal,
  PRIVATE_ROOT_REFUSALS,
  PrivateRootError,
  privateRootFile,
  type PrivateRootCode,
  writePrivateFile,
} from "../src/private-root.ts";

const SHARED = JSON.parse(readFileSync(fileURLToPath(new URL("../../../fixtures/private-guard/refusals.json", import.meta.url)), "utf8")) as {
  private_root: Record<string, string>;
};
const scratch = mkdtempSync(join(tmpdir(), "private-root-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

let n = 0;
/** A fresh `<scratch>/case-N/private-data/out` with mode 0700 and its parents. */
function root(mode = 0o700): string {
  const dir = join(scratch, `case-${n++}`, "private-data", "out");
  mkdirSync(dir, { recursive: true });
  chmodSync(dir, mode);
  return dir;
}
const codeOf = (run: () => unknown): string => {
  try {
    run();
  } catch (error) {
    if (error instanceof PrivateRootError) {
      // The words are fixed and name no path.
      expect(error.message).toBe(PRIVATE_ROOT_REFUSALS[error.code]);
      expect(error.message).not.toContain(scratch);
      return error.code;
    }
    throw error;
  }
  return "ok";
};
const me = process.getuid?.() as number;

describe("shared words", () => {
  it("equal the fixed words the Python twin is pinned to", () => {
    expect(PRIVATE_ROOT_REFUSALS).toEqual(SHARED.private_root);
  });
});

describe("assertPrivateRoot", () => {
  it("accepts a 0700 real directory under private-data and returns its real path", () => {
    const dir = root();
    expect(assertPrivateRoot(dir)).toBe(dir);
  });

  it("refuses a relative path", () => {
    expect(codeOf(() => assertPrivateRoot("private-data/out"))).toBe("not_absolute");
  });

  it("refuses a missing path and a file", () => {
    expect(codeOf(() => assertPrivateRoot(join(root(), "nope")))).toBe("unreadable");
    const file = join(root(), "f");
    writeFileSync(file, "x");
    expect(codeOf(() => assertPrivateRoot(file))).toBe("not_a_directory");
  });

  it("refuses a symbolic link, and a path under one", () => {
    const dir = root();
    const link = join(scratch, `link-${n++}`);
    symlinkSync(dir, link);
    expect(codeOf(() => assertPrivateRoot(link))).toBe("symbolic_link");
    expect(codeOf(() => assertPrivateRoot(join(link, "..", "out")))).not.toBe("ok");
    const viaLink = join(scratch, `via-${n++}`);
    symlinkSync(join(dir, ".."), viaLink);
    mkdirSync(join(dir, "inner"), { mode: 0o700 });
    expect(codeOf(() => assertPrivateRoot(join(viaLink, "out", "inner")))).toBe("symbolic_link");
  });

  it("refuses another owner (the probe seam)", () => {
    expect(codeOf(() => assertPrivateRoot(root(), { uid: me + 1 }))).toBe("wrong_owner");
  });

  it("refuses a group- or world-accessible mode", () => {
    for (const mode of [0o755, 0o750, 0o705, 0o770]) expect(codeOf(() => assertPrivateRoot(root(mode)))).toBe("group_or_other_access");
  });

  it("refuses a group- or world-writable parent, but takes a sticky one", () => {
    const dir = root();
    const parent = join(dir, "..");
    chmodSync(parent, 0o777);
    expect(codeOf(() => assertPrivateRoot(dir))).toBe("unsafe_parent");
    chmodSync(parent, 0o1777);
    expect(codeOf(() => assertPrivateRoot(dir))).toBe("ok");
    chmodSync(parent, 0o700);
  });

  it("refuses a parent owned by someone else who is not root (the probe seam), and takes root", () => {
    const dir = root();
    const parent = join(dir, "..");
    expect(codeOf(() => assertPrivateRoot(dir, { parentUid: (p) => (p === parent ? me + 1 : me) }))).toBe("unsafe_parent");
    expect(codeOf(() => assertPrivateRoot(dir, { parentUid: () => 0 }))).toBe("ok");
  });

  it("refuses a directory with no private-data segment", () => {
    const dir = join(scratch, `plain-${n++}`);
    mkdirSync(dir, { mode: 0o700 });
    expect(codeOf(() => assertPrivateRoot(dir))).toBe("no_private_data_segment");
  });
});

describe("placeRefusal", () => {
  it("refuses a Windows drive mount and a path with no private-data segment", () => {
    expect(onWindowsMount("/mnt/c/x/private-data/out")).toBe(true);
    expect(onWindowsMount("/mnt/cc/x")).toBe(false);
    expect(placeRefusal("/mnt/c/private-data/out")).toBe("windows_mount");
    expect(placeRefusal("/mnt/d")).toBe("windows_mount");
    expect(placeRefusal("/home/u/out")).toBe("no_private_data_segment");
    expect(placeRefusal("/home/u/private-data-x/out")).toBe("no_private_data_segment");
    expect(placeRefusal("/home/u/private-data/out")).toBeNull();
  });
});

describe("files in the root", () => {
  it("writes a new 0600 file and never overwrites", () => {
    const dir = root();
    const path = writePrivateFile(dir, "result.json", "{}");
    expect(path).toBe(join(dir, "result.json"));
    expect(statSync(path).mode & 0o777).toBe(0o600);
    expect(readFileSync(path, "utf8")).toBe("{}");
    expect(codeOf(() => writePrivateFile(dir, "result.json", "x"))).toBe("exists");
    expect(readFileSync(path, "utf8")).toBe("{}");
    expect(readFileSync(writePrivateFile(dir, "b.bin", new Uint8Array([1, 2])))).toEqual(Buffer.from([1, 2]));
  });

  it("accepts only a plain file name", () => {
    const dir = root();
    for (const name of ["", ".", "..", "a/b", "../x", "a\\b", "a\0b", "/abs"]) {
      expect(codeOf(() => privateRootFile(dir, name))).toBe("bad_file_name");
    }
    expect(existsSync(join(dir, "..", "x"))).toBe(false);
  });

  it("re-checks the root on every call", () => {
    const dir = root();
    chmodSync(dir, 0o755);
    expect(codeOf(() => privateRootFile(dir, "a"))).toBe("group_or_other_access");
    expect(codeOf(() => writePrivateFile(dir, "a", "x"))).toBe("group_or_other_access");
  });

  it("reports a write that fails for another reason as unreadable", () => {
    const dir = root();
    chmodSync(dir, 0o500); // no write permission (a non-root user cannot create the file)
    const code = codeOf(() => writePrivateFile(dir, "a", "x"));
    chmodSync(dir, 0o700);
    expect(me === 0 ? ["ok", "unreadable"] : ["unreadable"]).toContain(code);
  });
});

it("every code has fixed words", () => {
  const codes = Object.keys(PRIVATE_ROOT_REFUSALS) as PrivateRootCode[];
  for (const code of codes) expect(new PrivateRootError(code).message).toBe(PRIVATE_ROOT_REFUSALS[code]);
});
