import { loadEngine } from "@areneon/showdown-bridge";
import type { TimerRules } from "../src/index.ts";

/**
 * The server's tick (`TICK_TIME` in the pinned `server/room-battle.ts`). It
 * is a server constant, not a format rule, so the rule table does not carry it.
 */
const SERVER_TICK_SECONDS = 5;

/** The VGC Timer of the M-C format, read from the pinned engine's rule table. */
export function vgcTimer(): TimerRules {
  const engine = loadEngine();
  const format = engine.sim.Dex.formats.get("gen9championsvgc2026regmc");
  const values = engine.sim.Dex.formats.getRuleTable(format).valueRules;
  const read = (rule: string): number => {
    const value = values.get(rule);
    if (value === undefined) throw new Error(`the format has no ${rule}`);
    return Number(value);
  };
  return {
    starting: read("timerstarting"),
    grace: read("timergrace"),
    add_per_turn: read("timeraddperturn"),
    max_per_turn: read("timermaxperturn"),
    max_first_turn: read("timermaxfirstturn"),
    tick: SERVER_TICK_SECONDS,
  };
}
