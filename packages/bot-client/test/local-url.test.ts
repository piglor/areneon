import { describe, expect, it } from "vitest";
import { assertLocal, LocalUrlError } from "../src/index.ts";

describe("assertLocal", () => {
  it.each([
    "ws://127.0.0.1:8000/showdown/websocket",
    "ws://localhost:8000/showdown/websocket",
    "ws://[::1]:8000/showdown/websocket",
    "wss://127.0.0.1:8443/showdown/websocket",
    "ws://localhost",
    "ws://127.0.0.1/?q=1",
  ])("accepts %s and returns the parsed URL", (url) => {
    expect(assertLocal(url).href.startsWith(url.replace(/\?.*$/, ""))).toBe(true);
  });

  it.each([
    // userinfo
    ["ws://evil@127.0.0.1", /userinfo/],
    ["ws://user:pass@localhost:8000", /userinfo/],
    // look-alike hosts
    ["ws://127.0.0.1.evil.com", /host 127\.0\.0\.1\.evil\.com refused/],
    ["ws://localhost.evil.com", /host localhost\.evil\.com refused/],
    ["ws://example.com", /host example\.com refused/],
    // IPv4 in decimal, octal, hex or short form: URL normalizes these to 127.0.0.1
    ["ws://2130706433", /exactly as ws:\/\/127\.0\.0\.1/],
    ["ws://0x7f.1", /exactly as/],
    ["ws://0x7f000001", /exactly as/],
    ["ws://0177.0.0.1", /exactly as/],
    ["ws://127.1", /exactly as/],
    // non-canonical IPv6 loopback
    ["ws://[0:0:0:0:0:0:0:1]", /exactly as ws:\/\/\[::1\]/],
    // other loopback or wildcard addresses
    ["ws://[::ffff:127.0.0.1]", /host \[::ffff:7f00:1\] refused/],
    ["ws://0.0.0.0", /host 0\.0\.0\.0 refused/],
    ["ws://127.0.0.2", /host 127\.0\.0\.2 refused/],
    ["ws://[::]", /host \[::\] refused/],
    // trailing dot
    ["ws://localhost.", /host localhost\. refused/],
    ["ws://127.0.0.1.:8000", /exactly as|refused/],
    // IPv6 zone ids do not parse
    ["ws://[fe80::1%25eth0]", /not a valid URL/],
    ["ws://[::1%25lo]", /not a valid URL/],
    // schemes
    ["http://127.0.0.1:8000", /scheme http: refused/],
    ["https://localhost", /scheme https: refused/],
    ["file:///tmp/socket", /scheme file: refused/],
    // spelling the URL normalizes away
    ["WS://127.0.0.1", /exactly as/],
    ["ws://LOCALHOST", /exactly as/],
    ["ws://127.0.0.1:80", /exactly as/],
    ["ws:127.0.0.1", /exactly as/],
    [" ws://127.0.0.1", /exactly as/],
    ["ws://127.0.0.1\n:8000", /exactly as/],
    // not a URL
    ["", /not a valid URL/],
    ["127.0.0.1:8000", /not a valid URL/],
    ["localhost:8000", /scheme localhost: refused/],
  ])("refuses %j", (url, message) => {
    expect(() => assertLocal(url)).toThrow(LocalUrlError);
    expect(() => assertLocal(url)).toThrow(message);
  });
});
