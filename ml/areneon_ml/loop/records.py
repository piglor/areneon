"""What a generation records: the policy identity, team-set manifest and CheckpointRecords."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from areneon_contracts.contracts import ContractError, jcs, sha256_hex, validate

from areneon_ml.checkpoint.record import CheckpointRecord, read_record
from areneon_ml.learner.config import RunConfig
from areneon_ml.model.options import feature_set

# What the worker does with the pair when the loop plays it: sample at temperature 1 in float32
# (ADR-010 §2). They are part of the policy's identity, so a worker that changed them would be
# a different policy id.
MODE = "sample"
TEMPERATURE = 1
PRECISION = "float32"
FULL_TEMPERATURE_PPM = 1_000_000  # temperature 1 in millionths: the default, never written
DIGEST_CHARS = 64  # a SHA-256 digest in lowercase hex


def policy_id(  # noqa: PLR0913, PLR0917
    preview: str,
    battle: str,
    mode: str = MODE,
    temperature_ppm: int | None = None,
    truncation: dict[str, Any] | None = None,
    features: str | None = None,
    search: str | None = None,
    preview_solve: str | None = None,
) -> str:
    """``policy:`` + SHA-256 of ``JCS({preview, battle, mode, temperature, precision})``.

    A pair that reads a candidate feature set (ADR-022 §6) adds ``features: <feature_set_hash>``
    to the preimage; ``features`` is None for every other pair, whose id is unchanged.

    The loop plays ``sample`` (the default); the progress evaluation also plays ``argmax``,
    which is a different policy with a different id.

    A pair whose Team Preview is chosen by the solver (ADR-029 §4) adds
    ``preview_solve: <solver identity hash>``, independently of ``search`` (a pair with the solver,
    with search, or with both are three different policies); ``preview_solve`` is None for every
    other policy, whose id is unchanged.

    A pair played with decision-time search (ADR-026 §5) adds ``search: <search identity hash>``
    (label "learned pair with decision-time search (one-ply re-rank)"); ``search`` is None for every
    other policy, whose id is unchanged.

    Play settings (ADR-023 §1) enter the preimage only when set, so every earlier id is
    unchanged: ``temperature_ppm`` (1 to 999,999; the default 1,000,000 is never written)
    replaces ``temperature``, and ``truncation`` (``{"kind": "threshold", "min_ppm": n}``)
    is added. They apply to ``sample`` only.
    """
    body: dict[str, Any] = {
        "preview": preview,
        "battle": battle,
        "mode": mode,
        "temperature": TEMPERATURE,
        "precision": PRECISION,
    }
    for key, value, what in (
        ("preview_solve", preview_solve, "a Team Preview solver identity hash"),
        ("search", search, "a search identity hash"),
        ("features", features, "a feature_set_hash"),
    ):
        if value is not None:
            if len(value) != DIGEST_CHARS or value.strip("0123456789abcdef"):
                raise ValueError(f"{key} is {what}: 64 lowercase hex digits")
            body[key] = value
    if temperature_ppm is not None or truncation is not None:
        if mode != "sample":
            raise ValueError("a temperature or a truncation applies to sample mode only")
        if temperature_ppm is not None:
            if not 1 <= temperature_ppm < FULL_TEMPERATURE_PPM:
                raise ValueError("temperature_ppm must be 1 to 999999; the default is omitted")
            del body["temperature"]
            body["temperature_ppm"] = temperature_ppm
        if truncation is not None:
            if truncation.keys() != {"kind", "min_ppm"} or truncation["kind"] != "threshold":
                raise ValueError("an unknown truncation")
            if not 1 <= truncation["min_ppm"] < FULL_TEMPERATURE_PPM:
                raise ValueError("min_ppm must be 1 to 999999")
            body["truncation"] = dict(truncation)
    return "policy:" + sha256_hex(jcs(body))


SEARCH_IDENTITY_KEYS = frozenset(
    {
        "search_config_hash",
        "belief_identity_hash",
        "mechanics_config_hash",
        "mechanics_adapter_format",
        "damage_aware_config_hash",
        "damage_value_format",
        "world_writers_hash",
        "world_builder_format",
        "world_facts_format",
        "reply_model_format",
        "leaf_format",
        "reader_sha256",
        "version_manifest",
    }
)
SEARCH_HASH_KEYS = (
    "search_config_hash",
    "belief_identity_hash",
    "mechanics_config_hash",
    "damage_aware_config_hash",
    "world_writers_hash",
    "reader_sha256",
)


def search_identity_hash(identity: dict[str, Any]) -> str:
    """SHA-256 of the JCS of a search identity (ADR-026 §5), the ``search`` member of a policy id.

    ``identity`` holds exactly the thirteen inputs that change what a search plays
    (``search_config_hash`` is the SHA-256 of the ``SearchConfig``'s JCS bytes); the TS
    ``searchIdentityHash`` computes the same.
    """
    if identity.keys() != SEARCH_IDENTITY_KEYS:
        raise ValueError("a search identity holds exactly its thirteen inputs")
    for key in SEARCH_HASH_KEYS:
        value = identity[key]
        if (
            not isinstance(value, str)
            or len(value) != DIGEST_CHARS
            or value.strip("0123456789abcdef")
        ):
            raise ValueError(f"{key} is a SHA-256: 64 lowercase hex digits")
    return sha256_hex(jcs(identity))


PREVIEW_SOLVE_IDENTITY_KEYS = frozenset(
    {
        "preview_solve_config_hash",
        "belief_identity_hash",
        "reader_sha256",
        "version_manifest",
        "solver_format",
        "leaf_format",
        "hypothetical_battle_format",
    }
)
PREVIEW_SOLVE_HASH_KEYS = ("preview_solve_config_hash", "belief_identity_hash", "reader_sha256")
ITERATIVE_SOLVERS = frozenset({"regret-matching", "regret-matching-plus"})


def read_preview_solve_config(value: object) -> dict[str, Any]:
    """A ``PreviewSolveConfig`` (ADR-029 §4): the schema, then the rules it cannot say.

    ``min_valid`` may not exceed ``n``, and ``iterations`` is present exactly for the iterative
    solvers; the TS ``readPreviewSolveConfig`` applies the same rules.
    """
    try:
        validate("preview-solve-config", value)
    except ContractError as error:
        raise ValueError(f"invalid PreviewSolveConfig: {error}") from error
    config = cast("dict[str, Any]", value)
    if config["worlds"]["min_valid"] > config["worlds"]["n"]:
        raise ValueError("min_valid exceeds the worlds per decision")
    solver = config["solver"]
    if (solver["kind"] in ITERATIVE_SOLVERS) != ("iterations" in solver):
        raise ValueError("iterations is present exactly for the iterative solvers")
    return config


def preview_solve_config_hash(config: dict[str, Any]) -> str:
    """SHA-256 of a ``PreviewSolveConfig``'s JCS bytes (``preview_solve_config_hash``)."""
    return sha256_hex(jcs(config))


def preview_solve_identity_hash(identity: dict[str, Any]) -> str:
    """SHA-256 of the JCS of a Team Preview solver identity (ADR-029 §4).

    It is the ``preview_solve`` member of a policy id.

    ``identity`` holds exactly the seven inputs that change what the solver plays; the TS
    ``previewSolveIdentityHash`` computes the same.
    """
    if identity.keys() != PREVIEW_SOLVE_IDENTITY_KEYS:
        raise ValueError("a solver identity holds exactly its seven inputs")
    for key in PREVIEW_SOLVE_HASH_KEYS:
        value = identity[key]
        if (
            not isinstance(value, str)
            or len(value) != DIGEST_CHARS
            or value.strip("0123456789abcdef")
        ):
            raise ValueError(f"{key} is a SHA-256: 64 lowercase hex digits")
    return sha256_hex(jcs(identity))


def team_set_manifest(hashes: Iterable[str]) -> dict[str, Any]:
    """The content-addressed list of every team the run drew (protocol C7, T3)."""
    return {"schema_version": 1, "teams": sorted(set(hashes))}


def team_set_hash(manifest: dict[str, Any]) -> str:
    return sha256_hex(jcs(manifest))


@dataclass(frozen=True)
class RecordFacts:
    """What a CheckpointRecord says beyond the run config."""

    role: str
    checkpoint_id: str
    parent: str | None
    training_set_hashes: list[str]
    team_set_manifest_hash: str
    learner_state_hash: str
    encoder_version: str
    manifest: dict[str, Any]
    #: A population member's lineage label ``<lineage>.m<k>`` (ADR-021 section 7); None is the
    #: config's ``lineage_id``.
    lineage_id: str | None = None
    #: The learned checkpoints the member trained against (the other members' frozen pairs); empty
    #: for a single lineage.
    league_opponents: tuple[str, ...] = ()


def checkpoint_record(config: RunConfig, facts: RecordFacts) -> CheckpointRecord:
    """The record of one frozen checkpoint, validated by the contract and the reader rules.

    The loop's checkpoints are synthetic (the training loader refuses anything else) and
    play no league yet, so ``league_opponents`` is empty: a mirror match against the
    parent's pair is lineage, not a league opponent, and scripted baselines never count.
    """
    raw: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_id": facts.checkpoint_id,
        "role": facts.role,
        "lineage_id": config.lineage_id if facts.lineage_id is None else facts.lineage_id,
        "league_opponents": sorted(set(facts.league_opponents)),
        "training_set_hashes": sorted(facts.training_set_hashes),
        "team_set_manifest_hash": facts.team_set_manifest_hash,
        "run_config_hash": config.hash,
        "learner_state_hash": facts.learner_state_hash,
        "encoder_version": facts.encoder_version,
        "data_class": "synthetic",
        "manifest": facts.manifest,
    }
    if facts.parent is not None:
        raw["parent"] = facts.parent
    return read_record(raw)


def features_hash(config: RunConfig) -> str | None:
    """The run's ``candidate_features`` hash (the policy id's ``features``), or None."""
    named = feature_set(config.model.encoder.options)
    return None if named is None else named.feature_set_hash


#: How reports and the dashboard name a pair that reads a candidate feature set (ADR-022 §6): its
#: latency includes the probe, and a comparison with ``damage-aware@v1`` is between two users of
#: the same probe.
FEATURES_FAMILY = (
    "learned policy that consults the engine's damage calculation on assumed opponent sets "
    "at decision time"
)


#: The same for ``probe-damage/2``, whose arrays also carry the engine's turn order (E-22.3).
FEATURES_2_FAMILY = (
    "learned policy that consults the engine's damage calculation and turn order on assumed "
    "opponent sets at decision time"
)


#: ``probe-damage/3``: ``/2`` and the damage of moves the opponent belief says a foe may hold.
FEATURES_3_FAMILY = (
    "learned policy that consults the engine's damage calculation and turn order on assumed "
    "opponent sets at decision time: damage under the assumed foe set, move likelihood from the "
    "belief"
)


#: ``probe-damage-belief/1`` (ADR-025 slice 5): the same rows, averaged over sampled worlds.
FEATURES_BELIEF_FAMILY = (
    "learned policy that consults the engine's damage calculation on sampled opponent worlds "
    "(the belief) at decision time"
)


#: ``probe-outcome/1`` (ADR-022 erratum E-22.5): ``/2``'s arrays recomputed with the player's own
#: battle history declared, and whether each move works now.
FEATURES_OUTCOME_FAMILY = (
    "learned policy that consults the engine's damage calculation, turn order and move outcomes "
    "on assumed opponent sets and the player's own battle history at decision time"
)


def family_label(config: RunConfig) -> str | None:
    """The family label of the run's pair, or None for a pair without a feature set."""
    named = feature_set(config.model.encoder.options)
    if named is None:
        return None
    if named.feature_set == "probe-damage-belief/1":
        return FEATURES_BELIEF_FAMILY
    if named.feature_set == "probe-outcome/1":
        return FEATURES_OUTCOME_FAMILY
    if named.feature_set == "probe-damage/3":
        return FEATURES_3_FAMILY
    return FEATURES_2_FAMILY if named.feature_set == "probe-damage/2" else FEATURES_FAMILY
