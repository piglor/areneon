import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";
import { CHOICE_CHANNELS_FILE, choiceChannels, choiceChannelsText } from "../src/choice-channels.ts";
import { type ChoiceCase, loadCorpus } from "../src/choice-corpus.ts";
import { loadEngine } from "../src/index.ts";

describe("the choice-corpus player channels", () => {
  it("equal a fresh replay of every committed log on the pinned engine, byte for byte", () => {
    const fresh = choiceChannels(loadEngine());
    expect(Object.keys(fresh)).toEqual(Object.keys(loadCorpus()));
    expect(readFileSync(CHOICE_CHANNELS_FILE, "utf8")).toBe(choiceChannelsText(fresh));
    for (const channel of Object.values(fresh)) {
      expect(channel.lines.at(-1)?.startsWith("|request|")).toBe(true);
    }
  });
});

describe("choiceChannels on doctored copies of a case", () => {
  const [file, found] = Object.entries(loadCorpus())[0] as [string, ChoiceCase];
  const dirWith = (change: (c: ChoiceCase) => void): string => {
    const dir = mkdtempSync(join(tmpdir(), "choice-channels-"));
    const copy = structuredClone(found);
    change(copy);
    writeFileSync(join(dir, file), JSON.stringify(copy));
    return dir;
  };

  it("reads the other seat's view when the case is for p2", () => {
    const dir = dirWith((c) => (c.player = "p2"));
    try {
      const channel = choiceChannels(loadEngine(), dir)[file] as { player: string; lines: string[] };
      expect(channel.player).toBe("p2");
      expect(channel.lines.at(-1)?.startsWith("|request|")).toBe(true);
    } finally {
      rmSync(dir, { recursive: true });
    }
  });

  it("refuses a logged choice the engine does not take", () => {
    const dir = dirWith((c) => (c.log = [...c.log.slice(0, -1), { ...c.log[c.log.length - 1], choice: "move 9, move 9" } as ChoiceCase["log"][number]]));
    try {
      expect(() => choiceChannels(loadEngine(), dir)).toThrow(/the engine refused a logged choice/);
    } finally {
      rmSync(dir, { recursive: true });
    }
  });
});
