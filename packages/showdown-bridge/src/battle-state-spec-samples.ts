/* eslint-disable @typescript-eslint/no-explicit-any */
// The BattleStateSpec's share of the sample corpus (ADR-008 section 2): the committed record, schema-invalid
// mutations and reader-invalid ones that pass the schema and that `readBattleStateSpec` refuses (each for the rule
// its name says; `battle-state-spec.test.ts` holds the message of each).

import type { Engine } from "./engine.ts";
import { loadBattleStateSpec } from "./battle-state-spec.ts";

type Cases = Record<string, unknown>;

const field = (copy: any, id: string): any => copy.fields.find((f: any) => f.id === id);
const column = (copy: any, name: string): any => copy.encoder.find((c: any) => c.column === name);

export function battleStateSpecSamples(engine: Engine, formatId: string): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const base = loadBattleStateSpec(engine, formatId);
  const change = (fn: (copy: any) => void): Record<string, unknown> => {
    const copy = structuredClone(base) as unknown as Record<string, unknown>;
    fn(copy);
    return copy;
  };
  return {
    valid: {
      "battle-state-spec/v1": base,
    },
    invalid: {
      "battle-state-spec/extra-field": change((c) => (c.note = "x")),
      "battle-state-spec/next-version": change((c) => (c.schema_version = 2)),
      "battle-state-spec/fractional-size": change((c) => (c.dims[0].size = 2.5)),
      "battle-state-spec/unknown-visibility": change((c) => (field(c, "battle.turn").visibility = "secret")),
      "battle-state-spec/unknown-read": change((c) => (field(c, "battle.turn").source.read = "guess")),
      "battle-state-spec/bad-engine-path": change((c) => (field(c, "battle.turn").source.path = "battle..turn")),
      "battle-state-spec/unknown-list": change((c) => (field(c, "roster.item").value.lists = ["natures"])),
      "battle-state-spec/missing-world-items": change((c) => delete c.world_items),
    },
    readerInvalid: {
      "battle-state-spec/unsorted-fields": change((c) => c.fields.reverse()),
      "battle-state-spec/group-mismatch": change((c) => (field(c, "battle.turn").group = "field")),
      "battle-state-spec/unknown-dim": change((c) => (field(c, "roster.move").extra = ["MOVES"])),
      "battle-state-spec/unused-dim": change((c) => c.dims.push({ kind: "count", symbol: "ZZZ", size: 1, source: "players" })),
      "battle-state-spec/groups-out-of-order": change((c) => c.groups.reverse()),
      "battle-state-spec/range-reversed": change((c) => Object.assign(field(c, "battle.turn").value, { min: 5, max: 1 })),
      "battle-state-spec/range-overflows-dtype": change((c) => (field(c, "battle.turn").dtype = "int8")),
      "battle-state-spec/seat-engine-source": change((c) => (field(c, "seat_field.request_kind").source = { kind: "engine", path: "battle.requestState", read: "value" })),
      "battle-state-spec/wrong-engine-root": change((c) => (field(c, "roster.hp").source.path = "battle.turn")),
      "battle-state-spec/seat-not-public": change((c) => (field(c, "seat_field.request_kind").visibility = "hidden")),
      "battle-state-spec/projection-on-public": change((c) => (field(c, "battle.turn").projection = "none")),
      "battle-state-spec/reveal-missing": change((c) => delete field(c, "roster.item").reveal),
      "battle-state-spec/reveal-wrong-mask": change((c) => (field(c, "roster.item").reveal = "seat_roster.move_shown")),
      "battle-state-spec/omitted-but-read": change((c) => {
        c.omitted.push({ level: "battle", key: "turn", reason: "cosmetic" });
        c.omitted.sort((x: any, y: any) => (`${x.level}.${x.key}` < `${y.level}.${y.key}` ? -1 : 1));
      }),
      "battle-state-spec/encoder-unknown-field": change((c) => (column(c, "turn").shared = ["battle.turns"])),
      "battle-state-spec/encoder-reads-hidden": change((c) => (column(c, "turn").shared = ["battle.prng", "battle.turn"])),
      "battle-state-spec/opponent-reads-unprojectable": change((c) => (column(c, "stat_*").opponent = ["roster.stats"])),
      "battle-state-spec/unsorted-reducer": change((c) => c.reducer.reverse()),
      "battle-state-spec/world-item-missing": change((c) => c.world_items.pop()),
      "battle-state-spec/world-item-unknown-field": change((c) => (c.world_items[0].fields = ["roster.nothing"])),
      "battle-state-spec/unsorted-out-of-scope": change((c) => c.out_of_scope.reverse()),
    },
  };
}
