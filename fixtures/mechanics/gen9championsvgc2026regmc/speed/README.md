# speed_order fixtures

Synthetic (CC0) sets and hand-worked expected values for the `speed_order`
query of the mechanics probe (ADR-016 slice 2). `cases.json` holds the sets
(PokePaste in the engine's canonical form, all legal in the format), and
for each case the query in short form, the expected result and the working.

The expected numbers were worked on paper from the formulas below, not read
from the probe. The probe and the committed engine must reproduce them
(`packages/showdown-bridge/test/mechanics-speed.test.ts`), and for the cases
marked `real_battle` the order must equal the order of the `|move|` lines of a
real turn of the engine on the same state (`mechanics-speed-battle.test.ts`).

## The formulas used

All Pokemon are level 50. Stat Points replace EVs, IVs are 31 (champions mod).

- Speed stat: `floor((base Speed + Stat Points + 20) x nature)`, nature 1.1, 1.0 or 0.9. Serious is neutral.
- Stage `n`: the engine's table `1, 1.5, 2, 2.5, 3, 3.5, 4` for `|n| = 0..6`; Speed times the entry for `n > 0`, divided by it for `n < 0`, rounded down (`+6` is 4x, `-1` is `floor(x / 1.5)`).
- Modifiers: Tailwind x2, Choice Scarf x1.5, paralysis x0.5 (all chained, then rounded down for the values used here).
- Trick Room: the action speed is minus the Speed, with no wrap.
- A switch is order 103; a move is order 200; a move's priority is its own (Protect +4, Fake Out +3, Dragon Tail -6).
- The Speed order is by order (low first), priority (high first), action speed (high first). Equal on all three is a tie.

## Base Speeds and Speeds

| Pokemon | Base | Mega base | Nature, Stat Points in Speed | Speed | Mega Speed |
|---|---|---|---|---|---|
| Venusaur | 80 | | Serious, 0 | 80 + 20 = 100 | |
| Blastoise | 78 | | Serious, 0 | 78 + 20 = 98 | |
| Pikachu | 90 | | Timid (+Spe), 32 | floor((90 + 32 + 20) x 1.1) = floor(156.2) = 156 | |
| Garchomp | 102 | 92 | Serious, 0 | 102 + 20 = 122 | 92 + 20 = 112 |
| Garchomp (fast) | 102 | | Jolly (+Spe), 6 | floor((102 + 6 + 20) x 1.1) = floor(140.8) = 140 | |
| Gengar | 110 | 130 | Serious, 0 | 110 + 20 = 130 | 130 + 20 = 150 |
| Ninetales | 100 | | Serious, 0 | 100 + 20 = 120 | |

## Reading a case

`query.sides` names a set per Pokemon (with optional `boosts`, `status`, `mega`), `query.actions` the
actions (`move` null is a switch), `query.state` the field (`weather`, `trick_room`, and `tailwind`
as the list of sides that have it). The tests fill in the rest of the contract query. `expected.actions`
lists the speed, priority and order of each action in query order, `expected.groups` the tie groups in
execution order (a Mega Evolution is its own group before the moves, listing the same index again for the
move), and `worked` the arithmetic.

A case with `real_battle: false` has an exact tie (the real engine draws), a switch (the scenario has no
bench), or declares a state the real battle test does not reproduce.
