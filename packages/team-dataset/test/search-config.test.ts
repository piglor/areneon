// The committed search config names ADR-025's belief identity of the committed v2 belief (ADR-026 section 5):
// the identity is computed here by the composition root, which joins the model, the validator and the prior.
import { loadDamageAwareConfigV2, loadRulesSnapshot, loadSearchConfig, resolveFormat } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { readerSha256 } from "../../public-state/src/index.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { createBeliefWorlds } from "../src/damage-aware-worlds.ts";
import { engine, FORMAT } from "./support.ts";

describe("config/search/rerank-1.v1.json", () => {
  it("names the belief identity hash of the committed belief (a new record when the reader or the pin moves it)", () => {
    const vocabulary = goldenTeamVocabulary();
    const manifest = resolveFormat(engine, FORMAT, loadRulesSnapshot(engine.commit)).manifest;
    const port = createBeliefWorlds({ engine, vocabulary, config: loadDamageAwareConfigV2(vocabulary), reader_sha256: readerSha256(), manifest });
    expect(loadSearchConfig(FORMAT).belief_identity_hash).toBe(port.identity_hash);
  });
});
