// The committed form of a ledger (#1457): one JCS line per entity, pair and revocation after a header line, so a ledger is
// read as a diff and a format bump adds no single-line blob. The lines read back to the very ledger (`admission_hash`
// unchanged), and the Python reader reads the same lines (tests/contracts/test_backend_admission.py).
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { AdmissionError, admissionHash, readBackendAdmission } from "../src/admission.ts";
import { admissionLines, readAdmissionLines } from "../src/admission-lines.ts";
import { PIN } from "./support.ts";

const DIR = fileURLToPath(new URL("../../../fixtures/contracts/v1/backend-admission/valid/", import.meta.url));
const samples = readdirSync(DIR).sort().map((name) => JSON.parse(readFileSync(join(DIR, name), "utf8")) as unknown);

describe("admission lines", () => {
  it.each(samples.map((s, i) => [i, s] as const))("sample %i round-trips to the same ledger and hash", (_i, sample) => {
    const ledger = readBackendAdmission(sample, { showdown_commit: PIN });
    const text = admissionLines(ledger);
    expect(text.endsWith("\n")).toBe(true);
    expect(text.split("\n").filter(Boolean).length).toBe(1 + ledger.entities.length + ledger.pairs.length + ledger.revoked.length);
    const back = readAdmissionLines(text, { showdown_commit: PIN });
    expect(back).toEqual(ledger);
    expect(admissionHash(back)).toBe(admissionHash(ledger));
  });

  it("refuses lines that are not a header and then entities, pairs and revocations in order", () => {
    const ledger = readBackendAdmission(samples[0], { showdown_commit: PIN });
    const lines = admissionLines(ledger).split("\n").filter(Boolean);
    const read = (ls: string[]) => () => readAdmissionLines(`${ls.join("\n")}\n`, { showdown_commit: PIN });
    expect(read([])).toThrow(AdmissionError);
    expect(read(lines.slice(1))).toThrow(/header/);
    expect(read([lines[0] as string, lines[0] as string])).toThrow(/one kind/);
    expect(read([...lines, '{"other":{}}'])).toThrow(/one kind/);
    expect(read([...lines, "not json"])).toThrow(AdmissionError);
    if (ledger.entities.length > 0 && ledger.pairs.length > 0) {
      const firstPair = lines.findIndex((l) => l.startsWith('{"pair"'));
      const swapped = [...lines];
      [swapped[1], swapped[firstPair]] = [swapped[firstPair] as string, swapped[1] as string];
      expect(read(swapped)).toThrow(/order/);
    }
    expect(read([...lines.slice(0, -1), ` ${lines.at(-1)}`])).toThrow(/canonical/);
    expect(() => readAdmissionLines(admissionLines(ledger).trimEnd(), { showdown_commit: PIN })).toThrow(/newline/);
  });
});
