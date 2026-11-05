"""``population-seed/1`` (ADR-021 section 5): the Python twin of ``population-seed.ts``.

Every draw of a population run that is not a battle seed or a team draw (those keep
``league-seed/1``) is SHA-256 over the JCS of ``{run_key, generation, member, purpose, index}``;
the first 8 bytes, big endian, are an unsigned 64-bit integer. How a use reduces it is that
use's rule. A seed never reaches a worker or an observation.
"""

from __future__ import annotations

import re
from typing import Literal, get_args

from areneon_contracts.contracts import jcs, sha256_hex

#: Names the derivation. A new byte layout gets a new number.
POPULATION_SEED_DERIVATION = "population-seed/1"

PopulationSeedPurpose = Literal[
    "member-init", "member-learner", "opponent-draw", "explore", "bootstrap"
]
POPULATION_SEED_PURPOSES: tuple[str, ...] = get_args(PopulationSeedPurpose)

# A learning member ``m<k>`` or an exploiter seat ``x<k>`` (ADR-021 section 2).
_MEMBER = re.compile(r"[mx](0|[1-9][0-9]{0,1})")


def population_draw(
    run_key: str, generation: int, member: str, purpose: PopulationSeedPurpose, index: int
) -> int:
    """The unsigned 64-bit draw of ``population-seed/1``; generation and index count from 0."""
    if any(isinstance(n, bool) or not isinstance(n, int) for n in (generation, index)):
        raise ValueError("generation and index are integers")
    if generation < 0 or index < 0:
        raise ValueError("generation and index count from 0")
    if _MEMBER.fullmatch(member) is None:
        raise ValueError("member must be m<k> or x<k>")
    if purpose not in POPULATION_SEED_PURPOSES:
        raise ValueError("purpose must be one of the population seed purposes")
    body = {
        "run_key": run_key,
        "generation": generation,
        "member": member,
        "purpose": purpose,
        "index": index,
    }
    return int(sha256_hex(jcs(body))[:16], 16)
