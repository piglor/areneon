// connect (ADR-007) against a real Postgres 18: `make db-up` first.

import { afterAll, describe, expect, it } from "vitest";
import { connect } from "../src/index.ts";

const HOST = process.env.RULES_DB_HOST ?? "127.0.0.1:55432";
// The throwaway local password `make db-up` sets (never the Coolify one).
const IMPORTER_URL = `postgres://importer:local-importer@${HOST}/areneon`;

const open: { end(): Promise<void> }[] = [];
afterAll(async () => {
  await Promise.all(open.map((client) => client.end()));
});

describe("connect", () => {
  it("opens a session as the URL's role, under the given application name", async () => {
    const client = await connect(IMPORTER_URL, "rules-store-test");
    open.push(client);
    const { rows } = await client.query("SELECT current_user AS u, current_setting('application_name') AS app");
    expect(rows).toEqual([{ u: "importer", app: "rules-store-test" }]);
  });

  it("survives an error on an idle connection; the next query reports it", async () => {
    const client = await connect(IMPORTER_URL);
    open.push(client);
    expect(client.emit("error", new Error("connection dropped"))).toBe(true);
    const { rows } = await client.query("SELECT current_setting('application_name') AS app");
    expect(rows).toEqual([{ app: "areneon-rules-store" }]);
  });

  it("rejects when the server refuses the login", async () => {
    await expect(connect(`postgres://importer:not-the-password@${HOST}/areneon`)).rejects.toThrow(
      /password authentication failed/,
    );
  });
});
