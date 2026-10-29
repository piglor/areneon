import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { CHOICE_CHANNELS_FILE, choiceChannels, choiceChannelsText } from "../src/choice-channels.ts";
import { loadCorpus } from "../src/choice-corpus.ts";
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
