"""The copy and explore step of a population run (ADR-021 section 6, Figure 2, slice 8).

Pure functions over plain data: no I/O, torch or floats. :mod:`areneon_ml.loop.ladder_write` calls
:func:`decide` while it builds a generation's ``ladder.json``; :mod:`areneon_ml.loop.population`
applies the outcome at the start of the next generation (:mod:`areneon_ml.loop.slot_copy`).

**Decision (Figure 2).** From the strengths and the window's pair-score table of generation ``g``:

1. a member is *ready* when at least ``selection.ready_generations`` generations have passed since
   the generation of its last copy (generation 0 for a member never copied); none ready: ``none``,
   ``no_ready_member``;
2. the loser ``L`` is the ready member of least strength and the donor ``D`` the member of greatest
   strength, ties to the lower member index; fewer than ``n_min`` pairs of ``D`` against ``L`` is
   ``too_few_pairs`` (``D`` is ``L`` only when the ready member is also the strongest, for
   instance a population of one: then there is no head to head either);
3. the one-sided bootstrap lower bound (``alpha_ppm``, lane ``rating.LANE_COPY`` of the generation's
   bootstrap seed) of ``D``'s mean pair score against ``L`` must exceed ``500000 + margin_ppm``:
   else ``lower_bound``;
4. ``D``'s lowest anchor pair score (``floor(score_sum_ppm / pairs)`` over its anchor cells) must be
   at least ``L``'s: else ``anchor_minimum``;
5. after ``L`` takes ``D``'s root at least ``min_roots`` distinct roots must remain: else
   ``root_floor``.

At most one copy per generation: the first failing condition names a ``none``; all hold: a ``copy``.

**Explore.** The copy's settings are the donor's, each perturbed with integer draws from
``population-seed/1`` purpose ``explore`` (generation of the decision, member the loser, index
counting from 0): per tunable one coin (probability 1/2 by the draw's parity), and when it lands a
second draw picks the direction. ``learning_rate_micro``, ``entropy_coef_micro``, ``clip_micro``
and ``target_kl_micro`` (when the donor has one and the config bounds it; without a bound it is
left alone and consumes no draw) become ``v * 5 // 4`` or
``v * 4 // 5``; ``epochs`` moves by one; the opponent shares move ``SHARE_STEP_PPM`` from one kind
to another (the ordered pair is the draw modulo 6), as far as both bounds allow. The records hold
the explored shares; the plan reads the base ``population.shares`` until slice 10. Every value is
clamped to ``population.explore.bounds``; model, encoder, dataset, baselines, seeds and format are
never touched.

**Slot state.** A ladder record lists each member's root and effective settings as they were at the
decision, before its copy. :func:`slot_state` gives the state a generation starts from: the previous
record's, with its copy applied (the loser takes the donor's root and the copy's settings).
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from areneon_ml.learner.config import MICRO, PPOSettings, RunConfig
from areneon_ml.loop import rating
from areneon_ml.loop.population_seed import population_draw

#: The opponent shares move by this much between two kinds in one explore step.
SHARE_STEP_PPM = 50_000
SHARE_KINDS = ("ladder_ppm", "self_play_ppm", "past_pool_ppm")
#: Index of the ``member-learner`` draw that seeds a copied slot's learner generator.
LEARNER_SEED_INDEX = 0
_PPO_KEYS = ("learning_rate_micro", "entropy_coef_micro", "epochs", "clip_micro")

Settings = dict[str, Any]
Table = Mapping[tuple[str, str, str], Sequence[int]]


# -- settings ------------------------------------------------------------------------------------


def member_settings(config: RunConfig, index: int) -> Settings:
    """A member's starting PPO settings and opponent shares, as the record states them."""
    ppo = config.raw["ppo"]
    population = config.raw["population"]
    over: dict[str, Any] = next(
        (e for e in population.get("member_ppo", []) if e["member"] == index), {}
    )
    settings: Settings = {key: over.get(key, ppo[key]) for key in _PPO_KEYS}
    kl = over.get("target_kl_micro", ppo.get("target_kl_micro"))
    if kl is not None:
        settings["target_kl_micro"] = kl
    settings["shares"] = dict(population["shares"])
    return settings


def ppo_part(settings: Mapping[str, Any]) -> Settings:
    """The PPO half of a member's settings (everything but the shares): the sidecar's key."""
    return {key: value for key, value in settings.items() if key != "shares"}


def ppo_with(ppo: PPOSettings, over: Mapping[str, Any], *, exact: bool = False) -> PPOSettings:
    """``ppo`` with the keys of ``over`` (a ``member_ppo`` entry or a settings dict) applied.

    With ``exact`` ``over`` is a complete settings dict (a ladder record's): a target KL it does
    not name is cleared, so the effective settings equal the recorded ones (a donor without a
    target KL must not leave the loser's own).
    """
    changes: dict[str, Any] = {}
    if exact and "target_kl_micro" not in over:
        changes["target_kl"] = None
    if "learning_rate_micro" in over:
        changes["learning_rate"] = over["learning_rate_micro"] / MICRO
        if ppo.schedule is not None:
            changes["schedule"] = dataclasses.replace(
                ppo.schedule, start_micro=over["learning_rate_micro"]
            )
    if "entropy_coef_micro" in over:
        changes["entropy_coef"] = over["entropy_coef_micro"] / MICRO
        if ppo.entropy_schedule is not None:
            changes["entropy_schedule"] = dataclasses.replace(
                ppo.entropy_schedule, start_micro=over["entropy_coef_micro"]
            )
    if "epochs" in over:
        changes["epochs"] = over["epochs"]
    if "clip_micro" in over:
        changes["clip"] = over["clip_micro"] / MICRO
    if "target_kl_micro" in over:
        changes["target_kl"] = over["target_kl_micro"] / MICRO
    return dataclasses.replace(ppo, **changes)


# -- explore -------------------------------------------------------------------------------------


class _Draws:
    """The sequential ``explore`` draws of one decision."""

    def __init__(self, run_key: str, generation: int, member: str) -> None:
        self._args = (run_key, generation, member)
        self._index = 0

    def next(self) -> int:
        value = population_draw(*self._args, "explore", self._index)
        self._index += 1
        return value

    def coin(self) -> bool:
        return self.next() % 2 == 1

    def up(self) -> bool:
        return self.next() % 2 == 1


def _clamp(value: int, bound: Mapping[str, int] | None) -> int:
    if bound is None:
        return value
    return max(bound["min"], min(bound["max"], value))


def _scaled(value: int, up: bool) -> int:
    return value * 5 // 4 if up else value * 4 // 5


def explore(
    config: RunConfig, generation: int, loser: str, settings: Mapping[str, Any]
) -> Settings:
    """The donor's ``settings`` perturbed for the copy at ``loser`` (draws of ``generation``)."""
    bounds = config.raw["population"]["explore"]["bounds"]
    draws = _Draws(config.run_key, generation, loser)
    out: Settings = copy.deepcopy(dict(settings))
    for name in ("learning_rate_micro", "entropy_coef_micro"):
        if draws.coin():
            out[name] = _clamp(_scaled(out[name], draws.up()), bounds.get(name))
    if draws.coin():
        out["epochs"] = _clamp(out["epochs"] + (1 if draws.up() else -1), bounds.get("epochs"))
    if draws.coin():
        out["clip_micro"] = _clamp(_scaled(out["clip_micro"], draws.up()), bounds.get("clip_micro"))
    if "target_kl_micro" in out and "target_kl_micro" in bounds and draws.coin():
        out["target_kl_micro"] = _clamp(
            _scaled(out["target_kl_micro"], draws.up()), bounds["target_kl_micro"]
        )
    if draws.coin():
        pairs = [(a, b) for a in SHARE_KINDS for b in SHARE_KINDS if a != b]
        source, target = pairs[draws.next() % len(pairs)]
        shares = out["shares"]
        amount = max(
            0,
            min(
                SHARE_STEP_PPM,
                shares[source] - bounds[source]["min"],
                bounds[target]["max"] - shares[target],
            ),
        )
        shares[source] -= amount
        shares[target] += amount
    return out


# -- slot state ----------------------------------------------------------------------------------


def slot_state(
    config: RunConfig, previous: Mapping[str, Any] | None
) -> tuple[dict[str, str], dict[str, Settings]]:
    """Each member's root and effective settings at the start of a generation.

    ``previous`` is the committed record of the generation before (``None`` at generation 1, where
    the config decides): its members' state with its replacements (the copy and the health
    replacements, :func:`replacements`) applied. Every donor is read as it was at the decision.
    """
    if previous is None:
        labels = [f"m{k}" for k in range(config.raw["population"]["members"])]
        return (
            {label: label for label in labels},
            {label: member_settings(config, k) for k, label in enumerate(labels)},
        )
    base_roots = {m["member"]: m["root"] for m in previous["members"]}
    base_settings = {m["member"]: copy.deepcopy(m["settings"]) for m in previous["members"]}
    roots, settings = dict(base_roots), copy.deepcopy(base_settings)
    for found in replacements(config, previous):
        roots[found.loser] = base_roots[found.donor]
        settings[found.loser] = copy.deepcopy(found.settings)
    return roots, settings


# -- health replacements (slice 9) ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Replacement:
    """A slot that takes a donor's frozen state at the start of the generation after a decision.

    ``cause`` is ``copy`` for the Figure 2 copy and the health cause for a quarantined member.
    """

    loser: str
    donor: str
    settings: Settings
    seed_index: int
    cause: str


def quarantined(record: Mapping[str, Any]) -> dict[str, str]:
    """The members a ladder record quarantined, with their causes (its ``health`` selections)."""
    return {e["member"]: e["cause"] for e in record["selections"][1:]}


def replaced_slots(record: Mapping[str, Any]) -> tuple[str, ...]:
    """The slots a ladder record's decisions replace: the copy's loser and the quarantined."""
    first = record["selections"][0]
    return (*((first["loser"],) if first["kind"] == "copy" else ()), *quarantined(record))


def _member_number(member: str) -> int:
    return int(member[1:])


def health_donor(record: Mapping[str, Any]) -> str:
    """The top member that is not quarantined: greatest strength, ties to the lower index."""
    held = quarantined(record)
    live = [m for m in record["members"] if m["member"] not in held]
    return str(
        min(live, key=lambda m: (-m["strength"]["strength"], _member_number(m["member"])))["member"]
    )


def replacements(config: RunConfig, record: Mapping[str, Any]) -> list[Replacement]:
    """The replacements a committed ladder record asks for: its copy, then its health slots.

    A quarantined member's slot takes the top live member's frozen state with no statistical test
    (ADR-021 section 6); its explored settings are drawn from ``population-seed/1`` ``explore``
    for (this generation, the slot), exactly as a copy's, and its learner generator from the same
    ``member-learner`` draw. They are derived from the record, which stores only the cause.
    """
    out: list[Replacement] = []
    first = record["selections"][0]
    if first["kind"] == "copy":
        out.append(
            Replacement(
                first["loser"],
                first["donor"],
                copy.deepcopy(first["settings"]),
                first.get("learner_seed_index", LEARNER_SEED_INDEX),
                "copy",
            )
        )
    held = quarantined(record)
    if held:
        donor = health_donor(record)
        by_member = {m["member"]: m for m in record["members"]}
        for member, cause in held.items():
            settings = explore(config, record["generation"], member, by_member[donor]["settings"])
            out.append(Replacement(member, donor, settings, LEARNER_SEED_INDEX, cause))
    return out


# -- the decision --------------------------------------------------------------------------------


def anchor_minimum(table: Table, member: str, anchors: Sequence[str]) -> int:
    """The least ``floor(score_sum_ppm / pairs)`` over ``member``'s anchor cells."""
    found = []
    for anchor in anchors:
        pairs, total = rating.cell_totals(table[(member, "anchor", anchor)])
        found.append(total // pairs)
    return min(found)


def decide(  # noqa: PLR0913
    config: RunConfig,
    generation: int,
    *,
    strengths: Mapping[str, int],
    table: Table,
    anchors: Sequence[str],
    roots: Mapping[str, str],
    settings: Mapping[str, Settings],
    last_copy: Mapping[str, int],
    bootstrap: rating.BootstrapPlan,
    excluded: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """The first entry of the record's ``selections``: Figure 2 on this generation's results.

    ``strengths`` and ``roots`` are keyed by every member (in member order); ``last_copy`` maps a
    member to the generation of its last copy (absent: never copied); ``settings`` are the members'
    effective settings, of which the donor's seeds the explore step. ``excluded`` are the members a
    health stop quarantined in this generation: they are neither loser nor donor (their slots are
    replaced by the health rule, :func:`replacements`).
    """
    selection = config.raw["population"]["selection"]
    members = list(strengths)
    eligible = [m for m in members if m not in excluded]
    ready = [
        m for m in eligible if generation - last_copy.get(m, 0) >= selection["ready_generations"]
    ]
    if not ready:
        return {"kind": "none", "failed": "no_ready_member"}
    loser = min(ready, key=lambda m: (strengths[m], members.index(m)))
    donor = min(eligible, key=lambda m: (-strengths[m], members.index(m)))
    scores = table.get((donor, "member", loser)) if donor != loser else None
    pairs = 0 if scores is None else rating.cell_totals(scores)[0]
    if scores is None or pairs < selection["n_min"]:
        return {"kind": "none", "failed": "too_few_pairs"}
    lower = rating.head_to_head_lower(scores, bootstrap, rating.LANE_COPY)
    if lower <= rating.EVEN_PPM + selection["margin_ppm"]:
        return {"kind": "none", "failed": "lower_bound"}
    donor_min = anchor_minimum(table, donor, anchors)
    loser_min = anchor_minimum(table, loser, anchors)
    if donor_min < loser_min:
        return {"kind": "none", "failed": "anchor_minimum"}
    roots_before = len(set(roots.values()))
    roots_after = len({roots[donor] if m == loser else roots[m] for m in members})
    if roots_after < selection["min_roots"]:
        return {"kind": "none", "failed": "root_floor"}
    return {
        "kind": "copy",
        "loser": loser,
        "donor": donor,
        "pairs": pairs,
        "margin_ppm": selection["margin_ppm"],
        "lower_bound_ppm": lower,
        "donor_anchor_min_ppm": donor_min,
        "loser_anchor_min_ppm": loser_min,
        "roots_before": roots_before,
        "roots_after": roots_after,
        "learner_seed_index": LEARNER_SEED_INDEX,
        "settings": explore(config, generation, loser, settings[donor]),
    }


def learner_seed(config: RunConfig, generation: int, loser: str, index: int, limit: int) -> int:
    """The new learner generator seed of a copied slot: the ``member-learner`` draw, reduced."""
    return population_draw(config.run_key, generation, loser, "member-learner", index) & limit
