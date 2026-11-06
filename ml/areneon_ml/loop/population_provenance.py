"""The provenance export of a population run (ADR-021 sections 7 and 11, slice 12).

A population generation holds one pair of ``CheckpointRecord``s per member
(``members/m<k>/records/``). The C1/C6 checker needs one record per checkpoint id, so this module
reads every member's records of every committed generation and keeps one per id: with a Battle-only
warm start every member's generation-0 Battle record is the same clone record (one checkpoint, one
file, byte for byte), which then is a common ancestor of every member's Battle lineage. Two
records for one id that differ are refused.

The sources registry maps each member's training set of generation ``g`` to that member's pair
frozen at ``g - 1`` (the pair that played the games); the other members the pair met enter
through ``league_opponents`` and a copy through ``parent``, both already in the records. So every
member and snapshot of one run ends up in the others' ancestry (C1 and C6: mutually *seen*), and a
separate run, which shares no checkpoint, stays independent.

The cumulative learner steps of a checkpoint are read from the optimiser step counts of the
learner-state sidecar written beside it. A copy takes the donor's Adam moments with their step
counts, so a copied slot's count is its donor's plus its own by construction; the counts follow
the ``parent`` edges that way without being summed here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from areneon_ml.checkpoint.record import CheckpointRecord, read_record
from areneon_ml.loop import layout
from areneon_ml.model.network import ROLES


class PopulationProvenanceError(ValueError):
    """A population run whose records or sidecars do not form a provenance graph."""


def _members(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    held = manifest.get("members")
    if not isinstance(held, dict) or not held:
        raise PopulationProvenanceError("the generation's manifest lists no members")
    return held


def _verified(run_dir: Path, newest: layout.Committed) -> list[dict[str, Any]]:
    """Every committed generation's manifest, each checked against its files."""
    return [
        layout.verify_generation(layout.generation_dir(run_dir, g), g)
        for g in range(newest.generation + 1)
    ]


def member_records(run_dir: Path, newest: layout.Committed) -> list[tuple[CheckpointRecord, Path]]:
    """One record per checkpoint id, with the file it came from, in generation then member order."""
    found: dict[str, tuple[CheckpointRecord, Path]] = {}
    for g, manifest in enumerate(_verified(run_dir, newest)):
        for member in sorted(_members(manifest)):
            for role in ROLES:
                path = (
                    layout.generation_dir(run_dir, g)
                    / "members"
                    / member
                    / "records"
                    / f"{role}.json"
                )
                record = read_record(layout.read_json(path))
                if record.checkpoint_id != manifest["members"][member]["checkpoints"][role]:
                    raise PopulationProvenanceError(
                        f"generation {g}: {member}'s {role} record names another checkpoint "
                        "than the manifest"
                    )
                held = found.get(record.checkpoint_id)
                if held is None:
                    found[record.checkpoint_id] = (record, path)
                elif held[1].read_bytes() != path.read_bytes():
                    raise PopulationProvenanceError(
                        f"checkpoint {record.checkpoint_id} has two different records"
                    )
    return list(found.values())


def population_sources(run_dir: Path, newest: layout.Committed) -> dict[str, list[str]]:
    """Training-set hash to the checkpoints that generated it, for every member."""
    registry: dict[str, list[str]] = {}
    manifests = _verified(run_dir, newest)
    for record, _ in member_records(run_dir, newest):
        for held in record.training_set_hashes:
            if held not in registry:
                registry[held] = []  # an outside source (the warm start's teacher)
    for g in range(1, len(manifests)):
        for member, entry in _members(manifests[g]).items():
            played = manifests[g - 1]["members"][member]["checkpoints"]
            held = entry["training_set_hash"]
            registry[held] = sorted({*registry.get(held, []), *played.values()})
    return registry


def learner_steps(run_dir: Path, newest: layout.Committed) -> dict[str, int]:
    """Cumulative optimiser steps per checkpoint id: the role's largest Adam step count.

    Read from ``members/m<k>/learner-<g>.json``. A policy-frozen update (a copy's value-only
    update) leaves the policy counts at the donor's and adds its own to the value head's, so the
    largest count is the number of updates the checkpoint's lineage has taken.
    """
    steps: dict[str, int] = {}
    for g, manifest in enumerate(_verified(run_dir, newest)):
        for member, entry in sorted(_members(manifest).items()):
            # The sidecar is the JCS bytes without a newline (``pack_state``), so not read_json.
            sidecar = json.loads(
                (
                    layout.generation_dir(run_dir, g) / "members" / member / f"learner-{g}.json"
                ).read_text(encoding="utf-8")
            )
            for role in ROLES:
                counts = sidecar["optimizer_steps"][role]
                total = max(counts.values(), default=0)
                checkpoint = entry["checkpoints"][role]
                if steps.setdefault(checkpoint, total) != total:
                    raise PopulationProvenanceError(
                        f"checkpoint {checkpoint} has two different step counts"
                    )
    return dict(sorted(steps.items()))
