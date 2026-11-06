import { describe, expect, it } from "vitest";
import { MAX_OBJECT_BYTES, ObjectStoreError } from "../src/object-store.ts";
import { InMemoryObjectStore } from "../src/memory-store.ts";
import { R2ObjectStore } from "../src/r2-client.ts";
import { CredentialError, R2Settings } from "../src/r2-settings.ts";
import { BUCKET, ENV, ENV_TOKEN, FakeR2, LAYOUT, objectKey, SESSION_TOKEN, settings } from "./support.ts";

const bytes = (text: string): Uint8Array => new TextEncoder().encode(text);
const credCode = (run: () => unknown): string => {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(CredentialError);
    return (error as CredentialError).code;
  }
  return "no error";
};
const storeCode = async (run: () => Promise<unknown>): Promise<string> => {
  try {
    await run();
  } catch (error) {
    expect(error).toBeInstanceOf(ObjectStoreError);
    return (error as ObjectStoreError).code;
  }
  return "no error";
};

describe("the optional session token", () => {
  it("signs x-amz-security-token (and signs it) when a token is set, and carries none without one", async () => {
    for (const withToken of [true, false]) {
      const fake = new FakeR2();
      const store = new R2ObjectStore(settings(withToken), LAYOUT, { fetch: fake.fetch });
      await store.putIfAbsent(objectKey("a"), bytes("x"));
      await store.get(objectKey("a"));
      await store.list("objects/");
      await store.overwriteCurrent(bytes("{}"));
      expect(fake.seen).toHaveLength(4);
      for (const seen of fake.seen) {
        expect(seen.headers.get("x-amz-security-token")).toBe(withToken ? SESSION_TOKEN : null);
        const signed = /SignedHeaders=([^,]+),/.exec(seen.headers.get("authorization") ?? "")?.[1]?.split(";") ?? [];
        expect(signed.includes("x-amz-security-token")).toBe(withToken);
        expect(seen.url).not.toContain(SESSION_TOKEN);
      }
    }
  });

  it("treats an empty AWS_SESSION_TOKEN as unset and refuses a malformed one", () => {
    expect(R2Settings.fromEnvironment({ ...ENV, AWS_SESSION_TOKEN: "" }, [], BUCKET)).toBeInstanceOf(R2Settings);
    expect(credCode(() => R2Settings.fromEnvironment({ ...ENV, AWS_SESSION_TOKEN: "short" }, [], BUCKET))).toBe("credentials_invalid");
    expect(credCode(() => R2Settings.fromEnvironment({ ...ENV, AWS_SESSION_TOKEN: `${SESSION_TOKEN} x y` }, [], BUCKET))).toBe("credentials_invalid");
  });

  it("refuses the token on the command line: by name and by value", () => {
    for (const argv of [["AWS_SESSION_TOKEN=abc"], ["--session-token", "x"], ["--token=x"], ["x", SESSION_TOKEN], [`--thing=${SESSION_TOKEN}`]]) {
      expect(credCode(() => R2Settings.fromEnvironment(ENV_TOKEN, argv, BUCKET)), argv.join(" ")).toBe("credentials_on_argv");
    }
  });
});

describe("the injected layout refuses an undeclared key before any request", () => {
  it("never calls fetch for a key, prefix or overwrite the layout does not declare", async () => {
    const fake = new FakeR2();
    const store = new R2ObjectStore(settings(), LAYOUT, { fetch: fake.fetch });
    for (const key of ["", "other.txt", "objects/", "objects/a/b", "/current.json", "snapshots/x"]) {
      expect(await storeCode(() => store.putIfAbsent(key, bytes("x"))), key).toBe("key_invalid");
      expect(await storeCode(() => store.get(key)), key).toBe("key_invalid");
      expect(await storeCode(() => store.head(key)), key).toBe("key_invalid");
    }
    expect(await storeCode(() => store.putIfAbsent("current.json", bytes("x")))).toBe("key_invalid");
    for (const prefix of ["", "objects", "snapshots/", "objects/a/"]) expect(await storeCode(() => store.list(prefix)), prefix).toBe("key_invalid");
    expect(fake.seen).toEqual([]);
  });

  it("holds for a second user's layout, and a store with no overwrite refuses overwriteCurrent before a request", async () => {
    const other = { allowsKey: (k: string) => /^runs\/[0-9]{1,6}\.json$/.test(k), allowsListPrefix: (p: string) => p === "runs/", overwriteKey: null };
    const fake = new FakeR2();
    const store = new R2ObjectStore(settings(), other, { fetch: fake.fetch });
    expect(await storeCode(() => store.putIfAbsent(objectKey("a"), bytes("x")))).toBe("key_invalid");
    expect(await storeCode(() => store.overwriteCurrent(bytes("x")))).toBe("key_invalid");
    expect(fake.seen).toEqual([]);
    await store.putIfAbsent("runs/12.json", bytes("x"));
    expect(await store.list("runs/")).toEqual(["runs/12.json"]);
    const memory = new InMemoryObjectStore(other);
    expect(await storeCode(() => memory.overwriteCurrent(bytes("x")))).toBe("key_invalid");
    expect(await storeCode(() => memory.get("current.json"))).toBe("key_invalid");
  });

  it("refuses a layout whose overwritable key is not one of its own keys", () => {
    const bad = { allowsKey: () => false, allowsListPrefix: () => true, overwriteKey: "current.json" };
    expect(() => new InMemoryObjectStore(bad)).toThrow(ObjectStoreError);
    expect(() => new R2ObjectStore(settings(), bad)).toThrow(ObjectStoreError);
  });

  it("gives the in-memory store the client's rules", async () => {
    const store = new InMemoryObjectStore(LAYOUT);
    await store.putIfAbsent(objectKey("a"), bytes("1"));
    expect(await storeCode(() => store.putIfAbsent(objectKey("a"), bytes("2")))).toBe("exists");
    expect(await storeCode(() => store.putIfAbsent(objectKey("b"), { byteLength: MAX_OBJECT_BYTES + 1 } as Uint8Array))).toBe("too_large");
    expect(await storeCode(() => store.putIfAbsent("current.json", bytes("x")))).toBe("key_invalid");
    await store.overwriteCurrent(bytes("p"));
    store.seed("other/x", bytes("s"));
    expect(store.keys()).toEqual(["current.json", "objects/a", "other/x"]);
    expect(await store.head(objectKey("a"))).toEqual({ bytes: 1 });
    expect(await store.head(objectKey("zz"))).toBeNull();
    expect(await store.list("objects/")).toEqual(["objects/a"]);
    expect(store.writes).toEqual(["objects/a", "current.json"]);
  });
});
