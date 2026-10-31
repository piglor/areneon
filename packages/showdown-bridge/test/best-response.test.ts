// The responder's choice and the evaluation's arithmetic (ADR-029 section 2, #1257). Pure: no engine, no worker.
import { describe, expect, it } from "vitest";
import { BestResponseError, chooseResponse, columnWeights, responderStream } from "../src/best-response.ts";

const own = (n: number): { rank: number }[] => Array.from({ length: n }, (_, rank) => ({ rank }));
const table = (entries: (number | null)[][], ranks?: number[]): Parameters<typeof chooseResponse>[0] =>
  ({ own: (ranks ?? entries.map((_, i) => i)).map((rank) => ({ rank })), entries }) as unknown as Parameters<typeof chooseResponse>[0];
const opp = (columns: { mean_ppm: number; argmax_worlds?: number }[]): Parameters<typeof columnWeights>[0] => columns.map((c, i) => ({ leads: [1, i + 2] as [number, number], ...c }));

// Row 0 is the counter to column 0; rows 1 and 2 do better on average over all three columns.
const ENTRIES = [
  [900_000, 100_000, 100_000],
  [500_000, 500_000, 500_000],
  [100_000, 800_000, 800_000],
];

describe("chooseResponse", () => {
  it("takes the row with the highest expected entry under the column weights", () => {
    expect(chooseResponse(table(ENTRIES), [1_000_000, 0, 0])).toEqual({ rank: 0, expected_ppm: 900_000, columns_used: 1 });
    expect(chooseResponse(table(ENTRIES), [1, 1, 1])).toEqual({ rank: 2, expected_ppm: Math.floor(1_700_000 / 3), columns_used: 3 });
    // the floor of the weighted mean
    expect(chooseResponse(table(ENTRIES), [1, 2, 0])).toEqual({ rank: 2, expected_ppm: Math.floor((100_000 + 1_600_000) / 3), columns_used: 2 });
  });

  it("breaks ties by the strategy's canonical rank, not by position in the table", () => {
    const tie = [
      [400_000, 400_000],
      [400_000, 400_000],
    ];
    expect(chooseResponse(table(tie, [7, 3]), [1, 1])?.rank).toBe(1);
    expect(chooseResponse(table(tie, [3, 7]), [1, 1])?.rank).toBe(0);
  });

  it("leaves out, for every row, a column some row lacks or that carries no weight; none left is null", () => {
    const holes = [
      [900_000, null, 100_000],
      [100_000, 800_000, 950_000],
    ];
    expect(chooseResponse(table(holes), [1, 5, 1])).toEqual({ rank: 1, expected_ppm: 525_000, columns_used: 2 });
    expect(chooseResponse(table(holes), [0, 5, 0])).toBeNull();
    expect(chooseResponse(table(holes), [0, 0, 0])).toBeNull();
    expect(chooseResponse(table([]), [1])).toBeNull();
  });

  it("is a function of the table and the weights alone", () => {
    const a = chooseResponse(table(ENTRIES), [3, 1, 2]);
    expect(chooseResponse(table(structuredClone(ENTRIES)), [3, 1, 2])).toEqual(a);
    expect(own(3)).toHaveLength(3);
  });
});

describe("the weights of the evaluated side's play", () => {
  const columns = opp([
    { mean_ppm: 700_000, argmax_worlds: 3 },
    { mean_ppm: 200_000, argmax_worlds: 1 },
    { mean_ppm: 100_000, argmax_worlds: 0 },
  ]);
  it("sample weighs a lead pair by the network's mean mass, argmax by the worlds it wins, blind equally", () => {
    expect(columnWeights(columns, "sample")).toEqual([700_000, 200_000, 100_000]);
    expect(columnWeights(columns, "argmax")).toEqual([3, 1, 0]);
    expect(columnWeights(columns, "blind")).toEqual([1, 1, 1]);
  });

  it("refuses argmax weights from a table built without argmax columns", () => {
    expect(() => columnWeights(opp([{ mean_ppm: 1 }]), "argmax")).toThrow(BestResponseError);
    expect(columnWeights(opp([{ mean_ppm: 1 }]), "blind")).toEqual([1]);
  });

  // A test with teeth. An evaluated policy that always plays ONE fixed strategy puts all its mass on one column: the responder
  // counters it and the blind control, which averages over every column, does not, so the exploitation margin is positive. A
  // uniform evaluated policy has the blind control's own weights, so both pick the same row: the margin is exactly zero.
  it("a fixed-strategy policy shows a positive margin and a uniform one a margin of zero", () => {
    const truth = (row: number, column: number): number => ENTRIES[row]?.[column] as number;
    const fixedColumns = opp([{ mean_ppm: 1_000_000 }, { mean_ppm: 0 }, { mean_ppm: 0 }]);
    const responder = chooseResponse(table(ENTRIES), columnWeights(fixedColumns, "sample"));
    const blind = chooseResponse(table(ENTRIES), columnWeights(fixedColumns, "blind"));
    expect(responder?.rank).toBe(0);
    expect(blind?.rank).toBe(2);
    // against the policy that really plays column 0 the responder scores 900,000 and the blind control 100,000
    const margin = truth(responder?.rank as number, 0) - truth(blind?.rank as number, 0);
    expect(margin).toBe(800_000);

    const uniformColumns = opp([{ mean_ppm: 333_333 }, { mean_ppm: 333_333 }, { mean_ppm: 333_333 }]);
    const sameResponder = chooseResponse(table(ENTRIES), columnWeights(uniformColumns, "sample"));
    const sameBlind = chooseResponse(table(ENTRIES), columnWeights(uniformColumns, "blind"));
    expect(sameResponder?.rank).toBe(sameBlind?.rank);
    const average = (row: number): number => [0, 1, 2].reduce((sum, c) => sum + truth(row, c), 0);
    expect(average(sameResponder?.rank as number) - average(sameBlind?.rank as number)).toBe(0);
  });
});

describe("responderStream", () => {
  it("is the SHA-256 of the JCS of the labels, a vector computed with Python hashlib", () => {
    const stream = responderStream({ seed: "sodium,00112233445566778899aabbccddeeff", game: 3, player: "p2", policy_id: "policy:abc", request_seq: 1 });
    expect(Buffer.from(stream).toString("hex")).toBe("89cdb525cdc215d9fc45043859339555df1a0902ca63a00d458f7f7e8ed991a1");
  });

  it("changes with each label", () => {
    const base = { seed: "sodium,00112233445566778899aabbccddeeff", game: 3, player: "p2", policy_id: "policy:abc", request_seq: 1 };
    const hex = (parts: typeof base): string => Buffer.from(responderStream(parts)).toString("hex");
    for (const change of [{ seed: "sodium,ff" }, { game: 4 }, { player: "p1" }, { policy_id: "policy:abd" }, { request_seq: 2 }]) expect(hex({ ...base, ...change })).not.toBe(hex(base));
  });
});
