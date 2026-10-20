import { chmodSync, existsSync, mkdirSync, mkdtempSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, afterEach, describe, expect, it, vi } from "vitest";
import { assertPrivateStore, ownerOk, createPrivateDir, createSecretFile, ensurePrivateDir, parseSecretJson, readSecretFile, SealedStoreError, writeSecretFile } from "../src/sealed-store.ts";

const root = mkdtempSync(join(tmpdir(), "sealed-store-test-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));
afterEach(() => vi.restoreAllMocks());

let n = 0;
const dir = (mode = 0o700): string => {
  const path = join(root, `d${n++}`);
  mkdirSync(path);
  chmodSync(path, mode);
  return path;
};
const mode = (path: string): number => statSync(path).mode & 0o7777;

describe("assertPrivateStore", () => {
  it("accepts the invoking user's 0700 directory outside any git tree", () => {
    const path = dir();
    expect(assertPrivateStore(path)).toBe(path);
  });

  it.each([0o750, 0o705, 0o770, 0o707])("refuses a directory with mode %o (a group or other bit)", (bits) => {
    expect(() => assertPrivateStore(dir(bits))).toThrow(/permission bits.*must be 0700/);
  });

  it("refuses a relative path, a missing path, a file and a link", () => {
    expect(() => assertPrivateStore("relative/store")).toThrow(/absolute/);
    expect(() => assertPrivateStore(join(root, "missing"))).toThrow(/ENOENT/);
    const file = join(root, "file");
    writeFileSync(file, "x");
    expect(() => assertPrivateStore(file)).toThrow(/not a directory/);
    const real = dir();
    const link = join(root, "link");
    symlinkSync(real, link);
    expect(() => assertPrivateStore(link)).toThrow(/not a directory.*link/);
  });

  it("refuses a directory the invoking user does not own", () => {
    const path = dir();
    const uid = process.getuid?.() as number;
    vi.spyOn(process, "getuid").mockReturnValue(uid + 1);
    expect(() => assertPrivateStore(path)).toThrow(/not owned by the invoking user/);
  });

  it("refuses a store below a directory that another user owns", () => {
    const parent = dir();
    const inside = join(parent, "store");
    mkdirSync(inside, { mode: 0o700 });
    const me = process.getuid?.() as number;
    // the store passes as the first caller's, the parent then belongs to somebody else
    vi.spyOn(process, "getuid").mockReturnValueOnce(me).mockReturnValue(me + 1);
    expect(() => assertPrivateStore(inside)).toThrow(/owned by neither the invoking user nor root/);
  });

  it("refuses a platform without POSIX ownership", () => {
    const path = dir();
    const real = process.getuid;
    (process as { getuid?: unknown }).getuid = undefined;
    try {
      expect(() => assertPrivateStore(path)).toThrow(/POSIX file ownership/);
    } finally {
      (process as { getuid?: unknown }).getuid = real;
    }
  });

  it("refuses a directory inside a git working tree", () => {
    const tree = dir();
    mkdirSync(join(tree, ".git"));
    const inside = join(tree, "store");
    mkdirSync(inside, { mode: 0o700 });
    expect(() => assertPrivateStore(inside)).toThrow(/inside a git working tree/);
  });

  it("refuses a parent that group or other can write to, unless it is sticky like /tmp", () => {
    const parent = dir(0o770);
    const inside = join(parent, "store");
    mkdirSync(inside, { mode: 0o700 });
    expect(() => assertPrivateStore(inside)).toThrow(/writable by group or other/);
    chmodSync(parent, 0o1777);
    expect(assertPrivateStore(inside)).toBe(inside);
    chmodSync(parent, 0o700);
  });
});

describe("the files of the store", () => {
  it("creates directories 0700 and files 0600 whatever the umask, and never overwrites", () => {
    const old = process.umask(0o000);
    try {
      const base = dir();
      const sub = join(base, "sub");
      createPrivateDir(sub);
      expect(mode(sub)).toBe(0o700);
      const file = join(sub, "secret");
      writeSecretFile(file, "abc");
      writeSecretFile(join(sub, "bytes"), Buffer.from("def"));
      expect(mode(file)).toBe(0o600);
      expect(readSecretFile(file)).toBe("abc");
      expect(() => writeSecretFile(file, "other")).toThrow(/cannot be created: EEXIST/);
      expect(readSecretFile(file)).toBe("abc");
      expect(() => createPrivateDir(sub)).toThrow(/cannot be created: EEXIST/);
      createSecretFile(join(sub, "empty"));
      expect(readSecretFile(join(sub, "empty"))).toBe("");
    } finally {
      process.umask(old);
    }
  });

  it("ensures a private directory: creates it, accepts one that is private, refuses one that is not", () => {
    const base = dir();
    const sub = join(base, "sealed");
    ensurePrivateDir(sub);
    expect(mode(sub)).toBe(0o700);
    ensurePrivateDir(sub);
    chmodSync(sub, 0o755);
    expect(() => ensurePrivateDir(sub)).toThrow(/not a private directory/);
  });

  it("refuses to read a file that is loose, not a file, or not the user's", () => {
    const base = dir();
    const file = join(base, "key");
    writeSecretFile(file, "abc");
    chmodSync(file, 0o644);
    expect(() => readSecretFile(file)).toThrow(/group or other permission bits \(mode 644\)/);
    chmodSync(file, 0o600);
    expect(() => readSecretFile(base)).toThrow(/not a regular file/);
    const uid = process.getuid?.() as number;
    vi.spyOn(process, "getuid").mockReturnValue(uid + 1);
    expect(() => readSecretFile(file)).toThrow(/not owned by the invoking user/);
  });

  it("lets only the invoking user or root own the directories above the store", () => {
    expect(ownerOk(1000, 1000)).toBe(true);
    expect(ownerOk(0, 1000)).toBe(true);
    expect(ownerOk(1001, 1000)).toBe(false);
  });

  it("refuses a link where a secret file is expected", () => {
    const base = dir();
    const real = join(base, "real");
    writeSecretFile(real, "abc");
    symlinkSync(real, join(base, "link"));
    expect(() => readSecretFile(join(base, "link"))).toThrow(/ELOOP/);
    expect(() => readSecretFile(join(base, "missing"))).toThrow(/ENOENT/);
  });

  it("does not create a file in a missing directory", () => {
    expect(() => writeSecretFile(join(root, "nowhere", "f"), "x")).toThrow(SealedStoreError);
    expect(existsSync(join(root, "nowhere"))).toBe(false);
  });

  it("parses JSON without quoting the text in its error", () => {
    expect(parseSecretJson('{"a":1}', "a.json")).toEqual({ a: 1 });
    try {
      parseSecretJson('{"key": "deadbeef" oops', "a.json");
      expect.unreachable();
    } catch (error) {
      expect((error as Error).message).toBe("a.json is not valid JSON");
    }
  });
});
