"""The CPU reference packer and the player-view projection of the battle-state spec (ADR-030 G4).

``pack`` turns a point (the pinned engine's state at a request plus both seats' channels and teams)
into the spec's tensors; ``project`` is what one seat may read of them, through the visibility
classes alone; ``reduced_view`` rebuilds the reducer-state view the encoder reads from a view; and
``compare_encodings`` holds its encoding bit-equal to ``Encoder.encode`` of the seat's query.
``python -m areneon_ml.projection serve`` answers the TS side's frames (the corpus gate, the
invariance test and the differential harness's ``projection`` kind).
"""

from .check import CONFIGS, compare_encodings, encoders_for, query_of
from .packer import PackError, PackSession, pack, plant_engine
from .project import plant_unreadable, project, view_digest
from .spec import ProjectionError, SpecIndex
from .view import reduced_view

__all__ = [
    "CONFIGS",
    "PackError",
    "PackSession",
    "ProjectionError",
    "SpecIndex",
    "compare_encodings",
    "encoders_for",
    "pack",
    "plant_engine",
    "plant_unreadable",
    "project",
    "query_of",
    "reduced_view",
    "view_digest",
]
