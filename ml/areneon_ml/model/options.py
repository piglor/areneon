"""Optional model options (ADR-010 erratum A4 item 2).

An option is an optional key of ``model.options`` (network) or ``model.encoder.options``
(encoder) of the ``TrainingRunConfig``. Absent keeps today's behaviour and today's bytes:
``EncoderConfig.as_dict`` and ``ModelConfig.as_dict`` omit the ``options`` key when no option
is set, so ``encoder_version``, the checkpoint metadata and every existing hash stay as they
were. Present, an option is part of the config JSON, so it moves the run config hash, the
``encoder_version`` and the checkpoint metadata.

This module is the seam and the registry: each experiment arm's ticket adds its own
``OptionSpec`` to ``REGISTRY`` (and a property to the contract's closed ``options`` object).
A name that is not registered is refused, never ignored.

Defined so far:

- ``candidate_matchup`` (encoder, layout changing; ADR-022 tier B): per move candidate, the
  engine-data type-chart cells of the move's type against the shown types of each position, and
  a same-type flag. Only ``true`` is a value: leave the key out to turn it off.
- ``candidate_features`` (encoder, layout changing; ADR-022 tier A): the value is the feature set
  the run reads, ``{"feature_set": "probe-damage/1" or "probe-damage/2", "feature_set_hash":
  <64 hex>}``. Each damaging
  move part of a battle candidate reads the engine's damage rows the Node side attaches to the
  query (``model/features.py``). The hash enters the config, so the ``encoder_version`` and the
  checkpoint metadata, and a pair with the option has a policy id with a ``features`` member.
- ``candidate_target`` (encoder, layout changing; ADR-010 erratum A4 item 7): per move candidate,
  the token of the Pokémon in the targeted active slot, which the Battle network gathers into the
  slot part. Only ``true`` is a value: leave the key out to turn it off.
- ``recent_history`` (encoder, layout changing; ADR-010 erratum A4 item 7): the reducer's recent
  history (``Recent``, ``elapsed``, ``condition_turns``) on the Pokémon and field features. Only
  ``true`` is a value: leave the key out to turn it off.
- ``exact_volatiles`` (encoder, layout changing; ADR-010 erratum A4 item 7): volatiles and field and
  side conditions as exact vocabulary ids (a multi-hot over the engine-data ids) instead of
  SHA-256 buckets, plus the types a ``typechange`` set. Needs a version 2 engine-data table. Only
  ``true`` is a value: leave the key out to turn it off.
- ``factored_preview`` (network, layout changing; ADR-010 erratum A4 item 4): the Preview network
  scores an order as a sum over places of ``u(place, Pokemon token)`` plus ``v(lead pair)``
  (``network.py``); the Battle network ignores it. Only ``true`` is a value.
- ``separate_value`` (network, layout changing; ADR-010 erratum A4 item 7): the value no longer
  shares gradient with the policy trunk. ``"network"``: a second trunk and value head
  (``value_net.*``) with no layer shared with the policy network, same encoder inputs and sizes;
  ``"detached"``: the value head reads the trunk's output with the gradient cut.
- ``policy_output_scale_ppm`` (network, layout changing; ADR-010 erratum A4 item 7): the scorer's
  last layer(s) start at ``value / 10^6`` of their default initial weights (10000 is 1/100), so
  the initial policy is near uniform. An integer from 1 to 999999 (1,000,000 is the default and is
  refused); it changes initial weights only, so the checkpoint records it in the config.
- ``shared_trunk`` (network, layout changing; ADR-010 erratum A4 item 5): the Preview and Battle
  networks share one embedding and transformer trunk (``network.py``: ``trunk_keys``,
  ``tie_trunk``), with role-specific candidate and value heads. Only ``true`` is a value. The
  learner trains the pair with one Adam over the union of parameters; each role's checkpoint is
  still a complete network and the two files hold equal trunk tensors. Not combined with a
  ``warm_start`` that names the preview role (two cloned trunks conflict); a Battle-only warm start
  gives the cloned Battle trunk to both roles.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

SCOPES = ("encoder", "network")


@dataclass(frozen=True)
class FeatureSetOption:
    """The value of ``candidate_features``: a feature set id and its identity hash."""

    feature_set: str
    feature_set_hash: str

    def as_dict(self) -> dict[str, str]:
        return {"feature_set": self.feature_set, "feature_set_hash": self.feature_set_hash}


OptionValue = bool | int | str | FeatureSetOption
Options = tuple[tuple[str, OptionValue], ...]  # sorted by name; hashable, so configs stay frozen


class OptionError(ValueError):
    """An option set is not usable (unknown name, wrong scope, invalid value)."""


@dataclass(frozen=True)
class OptionSpec:
    """One option: its value check and whether it changes the network's tensor layout."""

    name: str
    check: Callable[[Any], bool] | None
    layout_changing: bool = False
    #: For an option whose value is an object: reads it into a frozen value (``check`` is then
    #: not used, and a boolean or an integer is refused).
    parse: Callable[[Mapping[str, Any]], OptionValue] | None = None


CANDIDATE_MATCHUP = "candidate_matchup"
CANDIDATE_FEATURES = "candidate_features"
CANDIDATE_TARGET = "candidate_target"
RECENT_HISTORY = "recent_history"
EXACT_VOLATILES = "exact_volatiles"
FACTORED_PREVIEW = "factored_preview"
SEPARATE_VALUE = "separate_value"
SEPARATE_VALUE_NETWORK = "network"
SEPARATE_VALUE_DETACHED = "detached"
POLICY_OUTPUT_SCALE_PPM = "policy_output_scale_ppm"
SHARED_TRUNK = "shared_trunk"
MAX_OUTPUT_SCALE_PPM = 999_999  # 1,000,000 is the default: leave the option out
#: The feature sets the encoder can read (ADR-022 §3, erratum E-22.3: ``/2`` also carries the
#: threat, speed and priority arrays, which the encoder attaches to the own and foe tokens).
FEATURE_SETS = ("probe-damage/1", "probe-damage/2")
_HASH = re.compile(r"[0-9a-f]{64}")


def _parse_feature_set(source: Mapping[str, Any]) -> FeatureSetOption:
    if set(source) != {"feature_set", "feature_set_hash"}:
        raise OptionError("candidate_features must be exactly feature_set and feature_set_hash")
    name, digest = source["feature_set"], source["feature_set_hash"]
    if name not in FEATURE_SETS:
        raise OptionError(
            f"candidate_features names a feature set the encoder cannot read: {name!r}"
        )
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise OptionError("candidate_features.feature_set_hash must be 64 lowercase hex digits")
    return FeatureSetOption(name, digest)


REGISTRY: dict[str, dict[str, OptionSpec]] = {
    "encoder": {
        CANDIDATE_MATCHUP: OptionSpec(CANDIDATE_MATCHUP, lambda value: value is True, True),
        CANDIDATE_FEATURES: OptionSpec(CANDIDATE_FEATURES, None, True, parse=_parse_feature_set),
        CANDIDATE_TARGET: OptionSpec(CANDIDATE_TARGET, lambda value: value is True, True),
        RECENT_HISTORY: OptionSpec(RECENT_HISTORY, lambda value: value is True, True),
        EXACT_VOLATILES: OptionSpec(EXACT_VOLATILES, lambda value: value is True, True),
    },
    "network": {
        FACTORED_PREVIEW: OptionSpec(FACTORED_PREVIEW, lambda value: value is True, True),
        SHARED_TRUNK: OptionSpec(SHARED_TRUNK, lambda value: value is True, True),
        SEPARATE_VALUE: OptionSpec(
            SEPARATE_VALUE, lambda v: v in (SEPARATE_VALUE_NETWORK, SEPARATE_VALUE_DETACHED), True
        ),
        POLICY_OUTPUT_SCALE_PPM: OptionSpec(
            POLICY_OUTPUT_SCALE_PPM,
            lambda v: type(v) is int and 1 <= v <= MAX_OUTPUT_SCALE_PPM,
            True,
        ),
    },
}


def read_options(scope: str, source: object) -> Options:
    """The options of ``source`` (a JSON object) as a sorted tuple, refusing any unknown name."""
    if scope not in REGISTRY:
        raise OptionError(f"unknown option scope {scope!r}")
    if not isinstance(source, Mapping):
        raise OptionError(f"{scope} options must be an object")
    found: list[tuple[str, OptionValue]] = []
    for name in sorted(source):
        spec = REGISTRY[scope].get(name)
        if spec is None:
            raise OptionError(f"unknown {scope} option {name!r}")
        value = source[name]
        if spec.parse is not None:
            if not isinstance(value, Mapping):
                raise OptionError(f"{scope} option {name!r} must be an object")
            found.append((name, spec.parse(value)))
            continue
        if not isinstance(value, bool | int | str) or spec.check is None or not spec.check(value):
            raise OptionError(f"{scope} option {name!r} has an invalid value")
        found.append((name, value))
    return tuple(found)


def options_dict(options: Options) -> dict[str, Any]:
    """The JSON form of the options (an object value as its object)."""
    return {
        name: value.as_dict() if isinstance(value, FeatureSetOption) else value
        for name, value in options
    }


def with_options(body: dict[str, Any], options: Options) -> dict[str, Any]:
    """``body`` plus an ``options`` key, only when an option is set (absent keeps the bytes)."""
    if options:
        body["options"] = options_dict(options)
    return body


def layout_changing(scope: str, options: Options) -> bool:
    """Whether any set option changes the tensor layout of a written checkpoint."""
    return any(REGISTRY[scope][name].layout_changing for name, _ in options)


def is_set(options: Options, name: str) -> bool:
    """Whether the option ``name`` is present (a boolean option is present only as ``true``)."""
    return any(key == name and value for key, value in options)


def option_value(options: Options, name: str) -> OptionValue | None:
    """The value of the option ``name``, or None when it is not set."""
    for key, value in options:
        if key == name:
            return value
    return None


def feature_set(options: Options) -> FeatureSetOption | None:
    """The ``candidate_features`` value of an encoder's options, or None."""
    value = option_value(options, CANDIDATE_FEATURES)
    return value if isinstance(value, FeatureSetOption) else None
