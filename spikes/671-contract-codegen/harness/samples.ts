// Builds the spike's sample instances from the golden fixtures into
// <out>/samples/{valid,invalid}/<catalog-key>.<name>.json. Every language
// harness reads the same files. Run: node harness/samples.ts <out>
import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
type Obj = { [key: string]: Json };

const repo = join(import.meta.dirname, "..", "..", "..");
const fixtures = join(repo, "fixtures", "showdown", "gen9championsvgc2026regmc");
const out = process.argv[2];
if (out === undefined) throw new Error("usage: node harness/samples.ts <out>");

const read = (path: string): Obj => JSON.parse(readFileSync(path, "utf8")) as Obj;
const clone = <T>(value: T): T => structuredClone(value);
const preview = read(join(fixtures, "preview.request.json"));
const turn = read(join(fixtures, "turn.request.json"));
const forceSwitch = read(join(fixtures, "force-switch.request.json"));
const mega = read(join(fixtures, "choices", "mega.json"));
const teams = mega.teams as { p1: Obj[]; p2: Obj[] };

const identity = (fixture: Obj, rqid?: number): Obj => ({
  request_seq: fixture.request_seq as number,
  request_hash: fixture.request_hash as string,
  ...(rqid === undefined ? {} : { rqid }),
});

const timerRules: Obj = {
  starting: 420, grace: 90, add_per_turn: 0, max_per_turn: 55, max_first_turn: 90, tick: 1,
};
const previewContext: Obj = {
  timer_source: "shadow", timer_rules: timerRules, turn_seconds_left: 90, bank_seconds_left: 420,
  grace_seconds_left: 90, safety_margin_ms: 3000, submit_within_ms: 87000, policy_budget_ms: 80000,
};
const battleContext: Obj = {
  ...previewContext, timer_source: "server", turn_seconds_left: 55, grace_seconds_left: 0,
  remaining_decisions_estimate: 12, submit_within_ms: 52000, policy_budget_ms: 30000,
};

const previewObs: Obj = {
  kind: "preview", schema_version: 1, manifest: preview.manifest as Obj, player: "p1",
  identity: identity(preview), request: preview.request as Obj,
  opponent_preview: teams.p2.map((set) => ({ details: `${set.species as string}, L50` })),
  decision_context: previewContext,
};
const battleObs: Obj = {
  kind: "battle", schema_version: 1, manifest: turn.manifest as Obj, player: "p1",
  identity: identity(turn, 7), request: turn.request as Obj,
  public_log: ["|start", "|switch|p2a: Gengar|Gengar, L50, M|100/100", "|turn|1"],
  decision_context: battleContext,
};
const switchObs: Obj = {
  kind: "battle", schema_version: 1, manifest: forceSwitch.manifest as Obj, player: "p1",
  identity: identity(forceSwitch), request: forceSwitch.request as Obj, public_log: [],
};
const recommendation: Obj = {
  schema_version: 1, identity: identity(turn), choice: "move 1 1, move 3",
  source_policy_id: "legal-heuristic@0", elapsed_ms: 412, deadline_outcome: "in_budget",
  confidence_bp: 6250,
};
const decision: Obj = {
  identity: identity(turn, 7), choice: "move 1 1, move 3", decision_context: battleContext,
  deadline_outcome: "in_budget", submitted_after_ms: 530, bank_seconds_after: 418,
  server_auto_choose_detected: false,
};
const team: Obj = { schema_version: 1, team: teams.p1 };

const valid: Record<string, Json> = {
  "observation.preview": previewObs,
  "observation.battle-move": battleObs,
  "observation.battle-force-switch": switchObs,
  "legal_choice_set.turn": {
    request_seq: turn.request_seq as number, request_hash: turn.request_hash as string,
    showdown_commit: (turn.manifest as Obj).showdown_commit as string,
    choices: ["move 1 1, move 1 1", "move 1 1, move 3"], pending: [],
  },
  "recommendation.in-budget": recommendation,
  "decision_record.bank-known": decision,
  // Nullable (present, null) vs optional (absent): both at once.
  "decision_record.bank-null": (({ decision_context: _omit, ...rest }) => ({ ...rest, bank_seconds_after: null }))(decision),
  "private_team_spec.mega-p1": team,
  "request_fixture.preview": preview,
  "request_fixture.turn": turn,
  "request_fixture.force-switch": forceSwitch,
};

const mutate = (base: Obj, edit: (o: Obj) => void): Obj => {
  const copy = clone(base);
  edit(copy);
  return copy;
};
const firstPokemon = (o: Obj): Obj => ((o.request as Obj).side as { pokemon: Obj[] }).pokemon[0] as Obj;

const invalid: Record<string, Json> = {
  // Privacy: an omniscient artefact smuggled into an actor observation.
  "observation.omniscient-log": mutate(battleObs, (o) => { o.omniscient_log = ["|split|p2"]; }),
  // Privacy: opponent hidden information (item) in a CTS preview entry.
  "observation.opponent-item": mutate(previewObs, (o) => {
    ((o.opponent_preview as Obj[])[0] as Obj).item = "focussash";
  }),
  // Privacy: an opponent-only key inside the engine-owned request.
  "observation.request-unknown-key": mutate(battleObs, (o) => { firstPokemon(o).hiddenPower = "fire"; }),
  "observation.unknown-tag": mutate(battleObs, (o) => { o.kind = "spectator"; }),
  "observation.tag-body-mismatch": mutate(previewObs, (o) => { o.kind = "battle"; o.public_log = []; }),
  "observation.rqid-null": mutate(battleObs, (o) => { (o.identity as Obj).rqid = null; }),
  "observation.request-seq-zero": mutate(battleObs, (o) => { (o.identity as Obj).request_seq = 0; }),
  "recommendation.confidence-over-max": mutate(recommendation, (o) => { o.confidence_bp = 10001; }),
  "recommendation.unknown-enum": mutate(recommendation, (o) => { o.deadline_outcome = "late"; }),
  "recommendation.fractional-int": mutate(recommendation, (o) => { o.elapsed_ms = 1.5; }),
  "decision_record.nullable-missing": mutate(decision, (o) => { delete o.bank_seconds_after; }),
  "decision_record.auto-choose": mutate(decision, (o) => { o.server_auto_choose_detected = true; }),
  "private_team_spec.seven": mutate(team, (o) => { (o.team as Obj[]).push(clone(teams.p1[0] as Obj)); }),
  "private_team_spec.sps-key": mutate(team, (o) => { ((o.team as Obj[])[0] as Obj).sps = 4; }),
  "request_fixture.unknown-key": mutate(turn, (o) => { o.hidden_state = "x"; }),
};

rmSync(join(out, "samples"), { recursive: true, force: true });
for (const [kind, set] of [["valid", valid], ["invalid", invalid]] as const) {
  const dir = join(out, "samples", kind);
  mkdirSync(dir, { recursive: true });
  for (const [name, value] of Object.entries(set)) {
    writeFileSync(join(dir, `${name}.json`), `${JSON.stringify(value, null, 2)}\n`);
  }
}
console.log(`samples: ${readdirSync(join(out, "samples", "valid")).length} valid, ${
  readdirSync(join(out, "samples", "invalid")).length} invalid`);
