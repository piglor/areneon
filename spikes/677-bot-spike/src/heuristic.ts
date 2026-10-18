// SPIKE (#677): throwaway. A legal heuristic: an INTEGRATION FIXTURE, NOT AI.
// It builds the first plausible choice from the bot's own request only. It
// never decides legality: the server-side engine validates every choice, and
// on `|error|[Invalid choice]` the bot falls back to `/choose default`.
import type { JsonObject } from "../../../packages/showdown-bridge/src/index.ts";

interface Move {
  id: string;
  target?: string;
  disabled?: boolean | string;
}
interface Active {
  moves: Move[];
  commanding?: boolean;
}
interface SidePokemon {
  condition: string;
  active?: boolean;
}

/** Targets that need an explicit target slot in Doubles. */
const FOE_TARGETED = new Set(["normal", "any", "adjacentFoe"]);
/** Ally-only targets; skipped so the heuristic never helps the wrong side. */
const ALLY_TARGETED = new Set(["adjacentAlly", "adjacentAllyOrSelf"]);

const fainted = (p: SidePokemon | undefined): boolean => p?.condition.endsWith(" fnt") ?? true;

function sidePokemon(request: JsonObject): SidePokemon[] {
  return (request.side as unknown as { pokemon: SidePokemon[] }).pokemon;
}

/** Bring the first four in team order; the first two lead. */
export function previewChoice(maxChosen: number): string {
  return `team ${Array.from({ length: maxChosen }, (_, i) => i + 1).join("")}`;
}

export function moveChoice(request: JsonObject): string {
  const active = request.active as unknown as Active[];
  const team = sidePokemon(request);
  return active
    .map((slot, i) => {
      if (fainted(team[i]) || slot.commanding) return "pass";
      const index = slot.moves.findIndex(
        (m) => !m.disabled && !ALLY_TARGETED.has(m.target ?? ""),
      );
      if (index < 0) return "move 1"; // e.g. Struggle; the engine decides
      const move = slot.moves[index] as Move;
      return FOE_TARGETED.has(move.target ?? "") ? `move ${index + 1} 1` : `move ${index + 1}`;
    })
    .join(", ");
}

export function forceSwitchChoice(request: JsonObject): string {
  const slots = request.forceSwitch as unknown as boolean[];
  const team = sidePokemon(request);
  const used = new Set<number>();
  return slots
    .map((mustSwitch) => {
      if (!mustSwitch) return "pass";
      const bench = team.findIndex((p, j) => !p.active && !fainted(p) && !used.has(j));
      if (bench < 0) return "pass";
      used.add(bench);
      return `switch ${bench + 1}`;
    })
    .join(", ");
}
