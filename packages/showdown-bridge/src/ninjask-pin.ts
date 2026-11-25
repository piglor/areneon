/**
 * The commit of the vendored Areneon fork of Ninjask (`vendor/ninjask`, https://github.com/piglor/ninjask), the CPU-native
 * engine of training play (#1530). Pokémon Showdown at `PINNED_SHOWDOWN_COMMIT` stays the referee and the only authority
 * on legality; the native engine is admitted for training games only, entity by entity, through the differential harness.
 *
 * A bump is its own pull request: it follows `vendor/ninjask/docs/repin.md` of the fork, moves the gitlink and this
 * constant together, and `test/ninjask-pin.test.ts` fails on any that is left behind. The fork's Showdown pin
 * (`vendor/ninjask/refsim/SHOWDOWN_COMMIT`) must equal `PINNED_SHOWDOWN_COMMIT`.
 */
export const PINNED_NINJASK_COMMIT = "4dfb2a9d3806a7824d40395b85f483e6920a0d1c";
