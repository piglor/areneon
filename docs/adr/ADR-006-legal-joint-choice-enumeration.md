# ADR-006: Legal joint-choice enumeration

**Status:** Accepted (revision 3, 2026-10-18) | **Wave:** 1 | **Deciders:** core team | **Date:** 2026-10-18

ADR issue: #682 · Implementation Epic: #711 · Related: [ADR-003](ADR-003-pinned-showdown-reference-simulator.md), [ADR-005](ADR-005-bot-opponent-runtime.md)

Decision evidence: spike `spikes/682-legal-choices/` (branch `682-legal-choices`; measurements at commit `b55fb20`, roster query committed at `1172617`) [S10]

**Revision history:** r1 (2026-10-18) Proposed with three open decisions. r2 (2026-10-18) records the owner's answers to all three (see *Owner decisions*) and amends the #682 acceptance criterion accordingly. r3 (2026-10-18) addresses the remaining review findings: the rejection column counts candidates, each related ADR is its own source, the spike's limits are stated, the roster query is committed, and `maybeLocked` is named in the corpus. No decision changed from r2. Approval of r3 as a whole is still required.

---

## Context

Every Battle Policy, Preview Policy and legal heuristic chooses from a set of **legal joint choices**: a complete Doubles choice for both active slots, validated by the reference engine (`CONTEXT.md`). ADR-003 makes the pinned Showdown engine (`7244fe8`, format `gen9championsvgc2026regmc`) the only legality authority [S9]. `packages/showdown-bridge` already decodes the player's own `|request|` and runs seeded headless battles (runtime A1).

The decision question: **how does Areneon produce the candidate set a policy chooses from, so that every candidate it accepts is validated by the pinned engine, on both runtimes, without leaking hidden information?**

Facts from the pinned engine and protocol:

- `Side.choose(input)` parses and checks one side's whole choice string (moves, targets, `mega`/`megax`/`megay`/`terastallize`, `switch`, `pass`, `team`) and stores it in `side.choice`. It does not commit the turn; `Battle.choose` commits only after `side.isChoiceDone()` and `allChoicesDone()` [S1][S2].
- Joint constraints are checked in the side's choice state: a slot may be switched in only once per choice (`switchIns`), Mega Evolution only once per choice (`choice.mega`), and forced switches must equal `min(slots that must switch, healthy bench)` (`forcedSwitchesLeft`, `forcedPassesLeft`) [S1].
- Targets are positional. `validTargetLoc` accepts a valid location even if that slot is now empty or fainted; the engine retargets later [S2].
- A rejected choice calls `emitChoiceError`, which writes `|error|` to the side's stream and can **mutate** the request and Pokémon flags (for example it sets `trapped`, clears `maybeLocked`, marks a move disabled) [S1].
- `Battle.toJSON()` / `Battle.fromJSON()` serialise and rebuild a complete battle; a rebuilt battle without a `send` option discards its output [S2][S3].
- The engine deliberately hides some legality from the player. When a foe might have a trapping ability the request carries `maybeTrapped` instead of `trapped`, because "canceling switches would leak information"; `maybeDisabled` and `maybeLocked` work the same way [S2][S4]. The trapping abilities are Shadow Tag, Arena Trap and Magnet Pull [S11]. In the `champions` mod data, Gengar-Mega is the only species not marked nonstandard that has one of them (Shadow Tag), per the committed roster query `spikes/682-legal-choices/roster.ts` against the pinned build [S10].
- On submission the engine answers `|error|[Invalid choice]`, or `|error|[Unavailable choice]` followed by an updated `|request|` when the attempt revealed hidden information [S5].
- On the server path (A2), Areneon holds only the player's own request, never the engine's `Battle` object [S12].

## Decision drivers

1. 100% of candidates Areneon accepts are judged by the pinned engine; Areneon never declares legality (#682 acceptance criterion as amended below, ADR-003).
2. Player-perspective only: a legality verdict must not reveal opponent hidden information to a policy, log or checkpoint (repository privacy rules).
3. A1 and A2 must offer a policy the same candidate set for the same request (ADR-005 gate: equal observations apart from `rqid` and timer fields).
4. Deterministic, canonical order so episodes replay and `request_hash` identifies the set.
5. Cost fits the VGC Timer budget (55 s per turn; ADR-005 deadlines) and A1 training throughput.
6. A golden corpus covers preview, targets, Mega, forced switches and mutually conflicting switches (#682).

## Evidence and alternatives

### Spike measurement [S10]

Setup: pinned engine `7244fe8`, Node 24.11 on WSL2. 30 seeded M-C battles (`sodium,<b as 32 hex>`, b = 0..29, at most 40 turns). Both sides are the CC0 synthetic golden team with Mega Stones swapped in (p1: Charizardite Y, Venusaurite; p2: Gengarite, Blastoisinite). Each side plays uniformly at random among engine-accepted candidates. For every request the spike (a) enumerates candidates from the request JSON, (b) judges all of them with `side.choose` on one clone, (c) judges a request-blind brute-force universe on another clone and compares the engine-normalised actions.

| Phase | Requests | Candidates p50 / max | Rejected by engine (candidates) | Engine-accepted actions missed | Clone p50 | Judge all candidates p50 / p95 | Brute force p50 / p95 |
|---|---|---|---|---|---|---|---|
| Preview | 60 | 360 / 360 | 0 | 0 (of 1,956 strings) | 0.50 ms | 0.73 / 1.24 ms | 1.92 / 3.57 ms |
| Move | 877 | 56 / 314 | 272 | 0 (of 11,025 strings) | 0.57 ms | 1.78 / 3.89 ms | 83.7 / 130.9 ms |
| Forced switch | 124 | 2 / 2 | 0 | 0 (of 11,025 strings) | 0.49 ms | 0.64 / 0.78 ms | 8.1 / 9.2 ms |

- Coverage in the run: 96 move requests offered Mega, 278 offered a double switch (both slots switching, so the same-slot conflict was exercised and excluded), 3 forced-switch requests had two forced slots.
- **All 272 rejected candidates were `Can't switch: The active Pokémon is trapped`, in the 24 requests that carried `maybeTrapped`.** No candidate was rejected in a request without a `maybe*` flag. A clone verdict on those switches is therefore exactly the hidden information the engine withholds from the player.
- Not exercised by the run: `maybeDisabled` (Imprison), `maybeLocked`, locked moves and Struggle, Commander, Revival Blessing, `megax`/`megay` counted separately, Terastallization (absent in M-C). These belong in the curated corpus. The pending rule for `maybeDisabled` and `maybeLocked` (rule 5) is an inference from the engine source [S4], not a measurement.
- Spike limits: the spike does not reject unknown request fields (rule 3 is for the real enumerator), and its random player picks among clone-accepted candidates, so clone verdicts on `maybeTrapped` switches steer play. Rule 5 forbids that for policies; it does not affect the counts above.

### Alternatives

| Option | Summary | Functional fit | Privacy | A1/A2 parity | Cost per move request | Verdict |
|---|---|---|---|---|---|---|
| **A. Request-derived enumeration + engine judge** | Pure TS enumerator over the own request; each candidate judged by `side.choose` on a battle clone in A1 and CI; by the engine at submission in A2; hidden-information candidates stay pending | Meets both criteria in A1; A2 meets them at submission | Withholds verdicts that depend on `maybe*` flags | Same candidate set on both | 1.8 ms p50, 3.9 ms p95 [S10] | **Selected** |
| B. Request-derived enumeration only | Enumerate from request and tracked state, submit without engine checks (upstream random AI [S6], poke-env `valid_orders` + `join_orders` [S7][S8]) | Fails "never declares legality" | Safe | Same | ~0 | Rejected |
| C. Brute-force strings through the engine | No enumerator; try every syntactic string on a clone, keep accepted ones | Meets criteria in A1 only | Leaks trapping like A without the pending rule | Impossible in A2 (no `Battle`) | 84 ms p50, 131 ms p95, 47x A [S10] | Rejected as runtime; **kept as the CI completeness oracle** |
| D. Dry run on the live battle | `side.choose` then `clearChoice` on the running battle | Works in A1 | Leaks as A | A1 only | Cheapest | Rejected: `emitChoiceError` writes an error line into the player's stream (the A1 runner aborts on it) and mutates request and Pokémon flags [S1] |
| E. Shadow battle rebuilt from the player's view | Reconstruct a `Battle` from own request and public log for A2 | Partial | Needs guessed opponent sets | Diverges | Unknown | Rejected: re-implements state tracking that ADR-003 leaves to the engine; opponent sets are unknown under CTS |

Why not B: the upstream AI and poke-env both encode legality rules in client code (target lists, one Mega, no duplicate switch) and never ask the engine [S6][S7][S8]. That is exactly the rule duplication #682 forbids. Their rule set is still useful prior art for the enumerator, written independently (both are MIT; no code copied).

Why C is kept for CI: it needs no knowledge of the request, so it is an independent oracle. Comparing engine-normalised actions from C and A proves the enumerator misses nothing the engine would accept. 47x slower is fine in CI, not in a 55 s turn with search.

## Proposed decision

*Inference from the facts above, for owner review.* Adopt **Option A**, in `packages/showdown-bridge` (TypeScript beside the sim, ADR-002).

```text
own |request| (player view)
      |
      v
enumerateCandidates()  -- pure, deterministic, canonical strings
      |  candidates + pending (maybeTrapped / maybeDisabled / maybeLocked)
      v
ChoiceJudge port
  |-- CloneJudge (A1, CI): Battle.fromJSON(battle.toJSON()) once per request,
  |     per candidate: clearChoice; choose && isChoiceDone; clearChoice
  |     verdicts on pending candidates are withheld from the policy
  |-- SubmissionJudge (A1, A2): engine accepts, or |error|[Invalid choice]
  |     -> enumerator defect, fail closed; [Unavailable choice] + new
  |     |request| -> new request_seq, enumerate again
      v
LegalChoiceSet {request_seq, request_hash, showdown_commit, choices, pending}
      |
      v
Preview Policy / Battle Policy / legal heuristic
```

1. **Enumerator.** `enumerateCandidates(request: DecodedRequest): CandidateSet` is a pure function of the player's own request. It proposes, it does not decide. Output strings use the engine's canonical numeric syntax: `team <digits>`, `move <slot> [<target>] [mega|megax|megay|terastallize]`, `switch <slot>`, `pass`, joined as `<slot 1>, <slot 2>` [S5]. A target is written only when the move's target type requires one; positional targets follow the engine (foes `1`, `2`; ally and self negative) [S2]. It emits engine-distinct actions only: all ordered preview picks (360 for 6 choose 4), and every legal target location including empty foe slots and allies. Collapsing equivalent actions belongs to the policy action encoding, not here (owner decision 3).
2. **Deterministic order.** Slot 1 outer, slot 2 inner. Within a slot: moves by slot index, then targets `1, 2, ally, self`, then events none, `mega`, `megax`, `megay`, `terastallize`; then switches by ascending slot; then `pass`. Preview picks in lexicographic order. The set is identified by `request_hash` plus `showdown_commit`.
3. **Fail closed.** An unknown request field that can change legality (for example Revival Blessing `reviving`, an unseen `forceSwitch` shape) raises an error rather than guessing.
4. **ChoiceJudge port.** `judge(candidates) -> Verdict[]` with `{choice, legal, error}`.
   - `CloneJudge` runs only where the bridge owns the `Battle` (A1 and CI). It clones once per request and never touches the live battle. It must not call `allChoicesDone`, which sets `cantUndo` [S2].
   - `SubmissionJudge` is the engine's own verdict when the chosen candidate is submitted, on both runtimes [S5].
5. **Hidden-information rule (privacy, owner decision 1).** A candidate is **pending** when its legality can depend on a `maybe*` flag in the request: switches of a `maybeTrapped` slot, moves of a `maybeDisabled` or `maybeLocked` slot. Pending candidates are offered to the policy without a pre-submission verdict, in A1 as in A2. The CloneJudge verdict for them goes only to the evaluation log, labelled as hidden state. An `[Unavailable choice]` answer is a legitimate, player-visible event; the runner treats it as a new request, not an abort.
6. **Legality contract (owner decision 2).** A **legal joint choice** is either judged legal by the CloneJudge (non-pending, A1) or accepted by the engine on submission (any runtime). A non-pending candidate rejected by any judge is an enumerator defect: the A1 runner aborts and CI fails. On A2, no candidate is engine-judged before submission; there the contract rests on the corpus conformance below plus the submission verdict.
7. **Golden choice corpus** (`fixtures/showdown/<format>/choices/`, CC0, synthetic teams). Each case is a seed, both teams and a choice log that reaches the request, plus the expected candidate list and engine verdicts. Mandatory cases: preview (6 choose 4 with leads); every target type in M-C (`normal`, `any`, `adjacentAlly`, `adjacentAllyOrSelf`, `adjacentFoe`, self, spread); Mega (`mega`, `megax`/`megay`, two Mega-capable actives); forced switch (one slot, two slots, fewer bench than forced slots); mutually conflicting switches; visible `trapped`; hidden `maybeTrapped` (Gengar-Mega), `maybeDisabled` and `maybeLocked`; a fainted active with no replacement; a locked move or Struggle. CI asserts, per case: every non-pending candidate is engine-accepted; the brute-force oracle (Option C) finds no engine-accepted action outside the enumerated set; the order is stable.
8. **Provenance.** Episodes record the chosen string, `request_seq`, `request_hash` and `showdown_commit`; the candidate set can be recomputed from them, so it is not stored.

## Consequences and risks

| Risk | Mitigation or acceptance |
|---|---|
| The enumerator holds rule knowledge (targets, one Mega, no duplicate switch) that could drift from the engine | It only proposes. The CloneJudge rejects any extra candidate and the brute-force oracle finds any missing one, on every pin bump (ADR-003 golden re-run) |
| On A2 candidates are not engine-judged before submission | Accepted trade-off, owner decision 2. The submission verdict still holds, and corpus conformance bounds the risk |
| Pending candidates may be refused at submission and cost time on the VGC Timer | Re-enumeration is about 2 ms [S10]; the deadline logic of ADR-005 already reserves a margin |
| Random play leaves rare mechanics unexercised (Imprison, Commander, Revival Blessing, locked moves) | Curated corpus cases are mandatory and are listed in the first implementation ticket; unknown request shapes fail closed |
| Clone cost grows with log length | Measured 0.5 to 0.6 ms p50 on battles up to 40 turns; re-measure in A1 throughput work |
| Candidate count reaches 314 per move request | Acceptable for search and masking; collapsing equivalents is a policy-encoding decision (non-goal, owner decision 3) |
| `Battle.toJSON`/`fromJSON` is an engine internal that may change | Used only through `showdown-bridge` and covered by the golden corpus on every pin bump |

Replay: candidate sets are a pure function of request JSON and pinned commit, so historical episodes replay without a live service. Licence: only the MIT pinned engine is called; poke-env and the upstream AI are read as prior art, nothing copied, no AGPL code involved. Security: no new data crosses a boundary; the hidden-information rule keeps engine-only verdicts out of policy inputs.

## Migration and rollout

1. Implementation Epic #711 on the Wave 1 roadmap version: #712 enumerator and ChoiceJudge port; #713 golden choice corpus with the brute-force oracle in CI; #714 runner handling of `[Unavailable choice]` and pending candidates.
2. Replace the golden battle's `default` choices with corpus-driven choices; keep the spike only until the corpus lands, then delete it.
3. No persisted data migrates; contracts in #671 gain an optional `pending` marker on candidates.

## Non-goals

- Choosing among candidates (Preview Policy, Battle Policy, legal heuristic).
- Fixed-size action encoding or action masks for training; collapsing equivalent actions.
- Team legality (ADR-003, Showdown team validator).
- Tera, Dynamax, Z-Moves and other formats: handled only as far as the request flags them; M-C has none in the golden requests.
- Validating the opponent's choices.

## Owner decisions (2026-10-18, on r1)

1. **Accepted:** the hidden-information rule. `maybe*`-dependent candidates are offered pending in A1 too, so A1 and A2 match and no engine-only verdict reaches a policy.
2. **Accepted:** on A2, engine validation happens at submission only (no engine state exists there), backed by corpus conformance.
3. **Accepted:** engine-distinct granularity (360 preview orderings, ally and empty-slot targets); collapsing is left to the action-encoding work.

Amended #682 acceptance criterion, following decisions 1 and 2: *every candidate Areneon accepts is validated by the pinned engine, either by the CloneJudge before submission (non-pending candidates in A1 and CI) or by the engine's verdict on submission (pending candidates and A2); Areneon never declares legality itself.*

These answers settle the open questions. They were not approval of the ADR as a whole. The owner accepted r3 as a whole on 2026-10-18 (#682 journal).

## Sources

All accessed 2026-10-18. Showdown sources are at the pinned commit `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b` (committed 2026-10-16), MIT; that commit date is their version date.

| Label | Source | Claims supported |
|---|---|---|
| S1 | smogon/pokemon-showdown `sim/side.ts` (`emitChoiceError` L527, `isChoiceDone` L539, `chooseMove` L552, `chooseSwitch` L915, `chooseTeam` L1031, `clearChoice` L1118, `choose` L1185): https://github.com/smogon/pokemon-showdown/blob/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b/sim/side.ts | `side.choose` checks without committing; joint constraints (`switchIns`, `choice.mega`, forced switches); `emitChoiceError` emits an error line and mutates request and flags |
| S2 | `sim/battle.ts` (`toJSON`/`fromJSON` L318, `send` default L277, `validTargetLoc` L2403, `choose` L2962, `allChoicesDone` L3058, maybe-trap comment L1735): https://github.com/smogon/pokemon-showdown/blob/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b/sim/battle.ts | Commit only after both sides done; positional targets; clone API; `allChoicesDone` sets `cantUndo`; `maybeTrapped` exists to avoid leaking information |
| S3 | `sim/state.ts` (`serializeBattle` L61, `deserializeBattle` L84): https://github.com/smogon/pokemon-showdown/blob/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b/sim/state.ts | A battle can be rebuilt in full from its serialised state |
| S4 | `sim/pokemon.ts` (`getMoveRequestData` L1083): https://github.com/smogon/pokemon-showdown/blob/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b/sim/pokemon.ts | Requests expose `maybeTrapped`, `maybeDisabled`, `maybeLocked` instead of the true state |
| S5 | `sim/SIM-PROTOCOL.md` (choice syntax, `[Invalid choice]`, `[Unavailable choice]`, `rqid`): https://github.com/smogon/pokemon-showdown/blob/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b/sim/SIM-PROTOCOL.md | Canonical choice strings and target numbering; submission verdicts and the updated request after a reveal |
| S6 | `sim/tools/random-player-ai.ts`: https://github.com/smogon/pokemon-showdown/blob/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b/sim/tools/random-player-ai.ts | Upstream builds choices from the request alone (Option B prior art) |
| S7 | hsahovic/poke-env `src/poke_env/battle/double_battle.py` `valid_orders`, commit `7ff6212` (2026-09-18), MIT: https://github.com/hsahovic/poke-env/blob/7ff6212e4e738efd46f1ea52035b37a8bba1e386/src/poke_env/battle/double_battle.py | Client-side Doubles enumeration from request and tracked state, no engine check |
| S8 | poke-env `src/poke_env/player/battle_order.py` `join_orders`, same commit (2026-09-18): https://github.com/hsahovic/poke-env/blob/7ff6212e4e738efd46f1ea52035b37a8bba1e386/src/poke_env/player/battle_order.py | Client re-implements one-Mega and no-duplicate-switch rules |
| S9 | [ADR-003](ADR-003-pinned-showdown-reference-simulator.md), Accepted 2026-10-18 (#695) | The pinned engine is the only legality authority |
| S10 | Areneon spike `spikes/682-legal-choices/` (`enumerate.ts`, `oracle.ts`, `measure.ts` at `b55fb20`; `roster.ts` at `1172617`), branch `682-legal-choices`, `piglor/areneon#9` (main's squash commit is recorded on #682 after merge); run `node spikes/682-legal-choices/measure.ts 30` and `node spikes/682-legal-choices/roster.ts` | All measurements in the table; all 272 rejected candidates were trapped switches under `maybeTrapped`; the roster query over `Dex.mod('champions')` species not marked nonstandard prints `Gengar-Mega: Shadow Tag` only |
| S11 | `data/abilities.ts` (`arenatrap` L196, `magnetpull` L2517, `shadowtag` L4157): https://github.com/smogon/pokemon-showdown/blob/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b/data/abilities.ts | The abilities that set `maybeTrapped` through `onFoeMaybeTrapPokemon` |
| S12 | [ADR-005](ADR-005-bot-opponent-runtime.md), Accepted (#697) | A2 is a bot client on a local Showdown server, holding only its own request; A1 runs the engine in-process |
