// The coverage labeller (ADR-013 section 3, ADR-010 O6): a small pure
// interpreter of a `TeamLabelSet`. It holds no game id. Which team has a
// Trick Room setter, a Mega Stone or Fake Out is written in the label-set
// record; this file only knows how a leaf predicate and `all_of` are read.
//
// The labels are about the team's own sets, which are public information
// once a team is shown, never about an opponent's hidden state.
//
// A team's base cell is its label values read as one number: for each label
// in the label set's order, the position of its value among the label's listed
// values (the default is last), as a mixed-radix digit with the first label
// most significant. With 5 labels of 4, 2, 2, 2 and 2 values that is the 64
// cells of O6.

import type { PokemonSet, TeamLabelSet } from "@areneon/showdown-bridge";

export class LabelError extends Error {
  override name = "LabelError";
}

/** The most base cells a label set may have: a stratum number is at most 65535. */
export const MAX_CELLS = 65536;

type Label = TeamLabelSet["labels"][number];
type Predicate = Label["values"][number]["when"];

export interface LabelValue {
  label_id: string;
  value: string;
}

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/** What a predicate can ask about a team: the ids of its species, moves, items and abilities. */
export interface TeamFacts {
  species: ReadonlySet<string>;
  moves: ReadonlySet<string>;
  items: ReadonlySet<string>;
  abilities: ReadonlySet<string>;
}

export function teamFacts(sets: readonly PokemonSet[]): TeamFacts {
  return {
    species: new Set(sets.map((s) => toId(s.species))),
    moves: new Set(sets.flatMap((s) => s.moves.map(toId))),
    items: new Set(sets.filter((s) => s.item !== "").map((s) => toId(s.item))),
    abilities: new Set(sets.map((s) => toId(s.ability))),
  };
}

const anyIn = (ids: readonly string[], facts: ReadonlySet<string>): boolean => ids.some((id) => facts.has(id));

/** Whether a predicate holds: a leaf when one member of the team matches one of its ids; `all_of` when every leaf holds. */
export function holds(predicate: Predicate, facts: TeamFacts): boolean {
  switch (predicate.kind) {
    case "has_species":
      return anyIn(predicate.species, facts.species);
    case "has_move":
      return anyIn(predicate.moves, facts.moves);
    case "has_item":
      return anyIn(predicate.items, facts.items);
    case "has_ability":
      return anyIn(predicate.abilities, facts.abilities);
    case "all_of":
      return predicate.of.every((leaf) => holds(leaf, facts));
  }
}

/** A label's value for a team: the first listed value whose predicate holds, else the default. */
export function labelValue(label: Label, facts: TeamFacts): string {
  return label.values.find((v) => holds(v.when, facts))?.value ?? label.default;
}

/** A team's value for every label of the set, in the set's order. */
export function labelTeam(set: TeamLabelSet, sets: readonly PokemonSet[]): LabelValue[] {
  const facts = teamFacts(sets);
  return set.labels.map((label) => ({ label_id: label.label_id, value: labelValue(label, facts) }));
}

/** How many values a label can take: the listed ones and the default. */
const radix = (label: Label): number => label.values.length + 1;

/** The number of base cells of a label set. */
export function cellCount(set: TeamLabelSet): number {
  const n = set.labels.reduce((product, label) => product * radix(label), 1);
  if (n > MAX_CELLS) throw new LabelError(`the label set has ${n} cells, more than ${MAX_CELLS}`);
  return n;
}

/** The base cell of a team's labels (see the file comment). */
export function cellOf(set: TeamLabelSet, labels: readonly LabelValue[]): number {
  if (labels.length !== set.labels.length) throw new LabelError("labels are not one per label of the label set");
  return set.labels.reduce((cell, label, k) => {
    const entry = labels[k] as LabelValue;
    if (entry.label_id !== label.label_id) throw new LabelError(`labels[${k}] is ${entry.label_id}, not ${label.label_id}`);
    const at = entry.value === label.default ? label.values.length : label.values.findIndex((v) => v.value === entry.value);
    if (at < 0) throw new LabelError(`label ${label.label_id}: ${entry.value} is not one of its values`);
    return cell * radix(label) + at;
  }, 0);
}

/** The label values of a base cell: the inverse of `cellOf`. */
export function cellLabels(set: TeamLabelSet, cell: number): LabelValue[] {
  if (!Number.isInteger(cell) || cell < 0 || cell >= cellCount(set)) throw new LabelError(`cell ${cell} is outside the label space`);
  const out: LabelValue[] = [];
  let rest = cell;
  for (let k = set.labels.length - 1; k >= 0; k--) {
    const label = set.labels[k] as Label;
    const at = rest % radix(label);
    rest = (rest - at) / radix(label);
    out.unshift({ label_id: label.label_id, value: label.values[at]?.value ?? label.default });
  }
  return out;
}

/**
 * The radix of each label, first label first. The merge (plan.ts) drops the
 * last label first, which wildcards the last digit of a cell number, and so on.
 */
export const radices = (set: TeamLabelSet): number[] => set.labels.map(radix);
