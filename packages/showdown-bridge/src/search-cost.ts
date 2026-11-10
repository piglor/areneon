// Per-component cost lines of a searched decision (ADR-026 section 6, S0 measure 5). A `CostSink` is an optional callback the search
// reports wall-clock milliseconds to; it changes no value the search computes. Absent, nothing is timed.

/**
 * The components of a searched decision:
 * - `root_prior`: the worker's `evaluate` of the real observation (a round trip, reducer and encoder in the worker included);
 * - `world_building`: drawing the opponent worlds from the belief and building each in the scenario battle (all attempts);
 * - `max_damage_reply`: the probe's max-damage replies (clone, request, probe queries);
 * - `turn_simulation`: cloning the world and running one engine turn per (candidate, reply), wait requests included;
 * - `observation_build`: building the leaf's hypothetical-channel observation on this side;
 * - `worker_round_trip`: the leaf's value query to the worker and back (the worker's reducer, encoder and network are inside it).
 */
export const COST_COMPONENTS = ["root_prior", "world_building", "max_damage_reply", "turn_simulation", "observation_build", "worker_round_trip"] as const;
export type CostComponent = (typeof COST_COMPONENTS)[number];

export interface CostSink {
  add(component: CostComponent, ms: number): void;
}

// The components sum to the decision's wall time only while the simulations run one after another, as they do (a decision awaits
// each leaf in turn); a search that overlapped its leaves would count their time more than once. Time outside the components
// (candidate enumeration, the reader, the world request, bookkeeping) is the decision total minus the sum: the budget script's
// `other` line.
