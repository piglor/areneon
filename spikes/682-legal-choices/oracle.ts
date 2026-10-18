// Spike #682 (throwaway): ask the pinned engine whether a choice string is
// legal for one side, without committing it. The live battle is never
// touched: every verdict runs on a clone, because `emitChoiceError` can
// mutate the side's request and Pokémon flags.

/** The slice of Showdown's in-process Battle the spike relies on. */
export interface SimPokemon {
  position: number;
}
export interface SimAction {
  choice: string;
  pokemon?: SimPokemon;
  target?: SimPokemon;
  moveid?: string;
  targetLoc?: number;
  mega?: boolean;
  megax?: boolean;
  megay?: boolean;
  terastallize?: string;
  index?: number;
}
export interface SimSide {
  id: string;
  requestState: string;
  activeRequest: unknown;
  pokemon: SimPokemon[];
  choice: { actions: SimAction[]; error: string };
  choose(input: string): boolean;
  isChoiceDone(): boolean;
  clearChoice(): void;
}
export interface SimBattle {
  sides: SimSide[];
  ended: boolean;
  turn: number;
  toJSON(): unknown;
  choose(side: string, input: string): boolean;
  setPlayer(side: string, options: { name: string; team: string }): void;
}
export interface SimBattleClass {
  new (options: { formatid: string; seed: string }): SimBattle;
  fromJSON(state: unknown): SimBattle;
}

export interface Verdict {
  choice: string;
  legal: boolean;
  error: string;
  /** Engine-normalised action, used to compare differently spelled strings. */
  key: string;
}

function actionKey(side: SimSide, a: SimAction): string {
  if (a.choice === "team") return `team:${a.index}:${side.pokemon.indexOf(a.pokemon as SimPokemon)}`;
  if (a.choice === "pass") return "pass";
  if (a.choice === "switch" || a.choice === "instaswitch") {
    return `switch:${side.pokemon.indexOf(a.target as SimPokemon)}`;
  }
  const event = a.megax ? "megax" : a.megay ? "megay" : a.mega ? "mega" : a.terastallize ? "tera" : "";
  return `move:${a.moveid}:${a.targetLoc ?? 0}:${event}`;
}

/** Judge each choice on one clone of `battle`, resetting between verdicts. */
export function judge(Battle: SimBattleClass, battle: SimBattle, sideIndex: number, choices: string[]): Verdict[] {
  const clone = Battle.fromJSON(battle.toJSON());
  const side = clone.sides[sideIndex] as SimSide;
  return choices.map((choice) => {
    side.clearChoice();
    const legal = side.choose(choice) && side.isChoiceDone();
    const error = side.choice.error || (legal ? "" : "incomplete choice");
    const key = legal ? side.choice.actions.map((a) => actionKey(side, a)).join(" | ") : "";
    side.clearChoice();
    return { choice, legal, error, key };
  });
}

/** Option C: every syntactically plausible string, no request knowledge. */
export function bruteForceUniverse(requestState: string, teamSize: number): string[] {
  if (requestState === "teampreview") {
    const out: string[] = [];
    const walk = (prefix: number[]): void => {
      if (prefix.length > 0) out.push(`team ${prefix.join("")}`);
      if (prefix.length === teamSize) return;
      for (let i = 1; i <= teamSize; i++) if (!prefix.includes(i)) walk([...prefix, i]);
    };
    walk([]);
    return out;
  }
  const slot: string[] = ["pass"];
  for (let m = 1; m <= 4; m++) {
    for (const t of ["", " -2", " -1", " 1", " 2"]) {
      for (const e of ["", " mega", " megax", " megay", " terastallize"]) slot.push(`move ${m}${t}${e}`);
    }
  }
  for (let s = 1; s <= teamSize; s++) slot.push(`switch ${s}`);
  const out: string[] = [];
  for (const a of slot) for (const b of slot) out.push(`${a}, ${b}`);
  return out;
}
