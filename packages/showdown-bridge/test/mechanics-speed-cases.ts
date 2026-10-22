/* eslint-disable @typescript-eslint/no-explicit-any */
// The hand-worked speed_order fixtures (fixtures/mechanics/<format>/speed/cases.json)
// turned into queries; shared by the probe test and the real-battle test.

import { readFileSync } from "node:fs";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { type ParsedMechanicsQuery, readMechanicsQuery } from "../src/mechanics-query.ts";

export interface SpeedCase {
  id: string;
  description: string;
  worked: string[];
  query: {
    sides: { set: string; boosts?: Partial<Record<"atk" | "def" | "spa" | "spd" | "spe", number>>; status?: string; mega?: boolean }[][];
    actions: { side: number; position: number; move: string | null; mega?: boolean }[];
    state?: { weather?: string; trick_room?: boolean; tailwind?: number[] };
  };
  real_battle: boolean;
  expected:
    | { actions: { speed: number; priority: number; order: number }[]; groups: number[][]; gap_ids: string[] }
    | { unsupported: { reason: string; position: number }; gap_ids: string[] };
}

export interface SpeedFixture {
  sets: Record<string, string>;
  cases: SpeedCase[];
}

export const SPEED_FIXTURE = JSON.parse(readFileSync(new URL(`../../../fixtures/mechanics/${GOLDEN_FORMAT}/speed/cases.json`, import.meta.url), "utf8")) as SpeedFixture;

const ZERO_BOOSTS = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

/** The full contract query of a case: defaults filled in, set names replaced by their pastes. */
export function fullSpeedQuery(fixture: SpeedFixture, speedCase: SpeedCase, change: (query: Record<string, any>) => void = () => undefined): ParsedMechanicsQuery {
  const { sides, actions, state } = speedCase.query;
  const query = {
    kind: "speed_order",
    sides: sides.map((side) =>
      side.map((combatant) => ({ set: fixture.sets[combatant.set] as string, boosts: { ...ZERO_BOOSTS, ...combatant.boosts }, status: combatant.status ?? null, hp: null, mega: combatant.mega ?? false })),
    ),
    actions: actions.map((action) => ({ side: action.side, position: action.position, move: action.move, mega: action.mega ?? false })),
    state: {
      weather: state?.weather ?? null,
      terrain: null,
      trick_room: state?.trick_room ?? false,
      sides: [0, 1].map(() => ({ aurora_veil: false, light_screen: false, reflect: false, tailwind: false })).map((side, index) => ({ ...side, tailwind: (state?.tailwind ?? []).includes(index) })),
    },
  };
  change(query);
  return readMechanicsQuery({ schema_version: 1, data_class: "synthetic", format_id: GOLDEN_FORMAT, query });
}
