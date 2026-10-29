// Re-capture the player channels of the golden choice corpus (after a pin bump, an engine-facing change or a
// corpus change; run capture-choices.ts first):
//   node packages/showdown-bridge/scripts/capture-choice-channels.ts
// Replays each committed case log on the pinned engine and writes fixtures/showdown/<format>/choice-channels/
// channels.json, which tests/ml/model/test_target.py reads. Review the diff.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { CHOICE_CHANNELS_FILE, choiceChannels, choiceChannelsText } from "../src/choice-channels.ts";
import { loadEngine } from "../src/engine.ts";

mkdirSync(dirname(CHOICE_CHANNELS_FILE), { recursive: true });
writeFileSync(CHOICE_CHANNELS_FILE, choiceChannelsText(choiceChannels(loadEngine())));
