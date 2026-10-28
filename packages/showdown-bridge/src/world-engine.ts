// The engine objects the world builder touches, narrowed here so the shared engine types stay as they are.

export interface EffectState {
  duration?: number;
  stage?: number;
  move?: string;
  [key: string]: unknown;
}

export interface ActiveMoveLike {
  id: string;
}

export interface WMon {
  name: string;
  hp: number;
  maxhp: number;
  status: string;
  statusState: EffectState;
  fainted: boolean;
  level: number;
  gender: string;
  item: string;
  itemState: EffectState;
  ability: string;
  abilityState: EffectState;
  baseAbility: string;
  boosts: Record<string, number>;
  species: { id: string; isMega?: boolean };
  moveSlots: { id: string; pp: number; maxpp: number }[];
  storedStats: Record<string, number>;
  volatiles: Record<string, EffectState>;
  activeTurns: number;
  activeMoveActions: number;
  lastMove: ActiveMoveLike | null;
  lastMoveUsed: ActiveMoveLike | null;
  canMegaEvo: unknown;
  side: WSide;
  faint(): unknown;
  setStatus(status: string, source: null, sourceEffect: null, ignoreImmunities: boolean): unknown;
  sethp(hp: number): unknown;
  deductPP(move: string, amount: number): number;
  formeChange(species: string, source: null, isPermanent: boolean): unknown;
  addVolatile(id: string, source?: WMon | null, sourceEffect?: null): unknown;
  getHealth(): { shared: string };
}

export interface WSide {
  id: string;
  pokemon: WMon[];
  totalFainted: number;
  faintedThisTurn: WMon | null;
  faintedLastTurn: WMon | null;
  activeRequest: unknown;
  addSideCondition(id: string, source: WMon): unknown;
  sideConditions: Record<string, (EffectState & { layers?: number }) | undefined>;
}

export interface WBattle {
  sides: WSide[];
  ended: boolean;
  turn: number;
  activeMove: ActiveMoveLike | null;
  field: {
    weather: string;
    weatherState: EffectState;
    terrain: string;
    terrainState: EffectState;
    setWeather(id: string, source: WMon): unknown;
    clearWeather(): unknown;
    setTerrain(id: string, source: WMon): unknown;
    clearTerrain(): unknown;
    addPseudoWeather(id: string, source: WMon): unknown;
    pseudoWeather: Record<string, EffectState | undefined>;
  };
  actions: { runMegaEvo(pokemon: WMon): boolean };
  dex: {
    conditions: { get(id: string): { exists: boolean; effectType: string } };
    items: { get(id: string): { exists: boolean; isChoice?: boolean } };
    getActiveMove(id: string): ActiveMoveLike;
  };
  queue: { list: unknown[] };
  initEffectState(state: { id: string; target: WMon }): EffectState;
  singleEvent(event: string, effect: unknown, state: EffectState, target: WMon, source: null, sourceEffect: null): unknown;
  faintMessages(): unknown;
  endTurn(): void;
}

/** What the engine says an id is (`Weather`, `Terrain`, `Status`, `Condition`), or `none` for an id it does not have. */
export const conditionKind = (battle: WBattle, id: string): string => {
  const found = battle.dex.conditions.get(id);
  return found.exists ? found.effectType : "none";
};

export const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
export const STAGES = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"] as const;
