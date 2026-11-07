"""The single-lineage pages keep their bytes (slice 12b added the population pages beside them).

The digests are those of the committed ``inspector-run`` fixture's pages as built before the
population readers existed. A deliberate change to a page (the template, the style, a column)
changes them: print the new ones with ``ARENEON_PRINT_PAGE_DIGESTS=1`` and ``pytest -s`` on this
file, and review the diff.
"""

from __future__ import annotations

import hashlib
import os

from areneon_ml.inspector import render, replay, replay_page

from tests.ml.inspector.support import FIXTURE, ROOT, fixture_model
from tests.ml.inspector.test_replay_pages import table

PAGES = {
    "index.html": "d7673611d4e07259fe7fe17b9d64b1d18b9ad8be2e9dda622ed1c1f5a6a6eb3f",
    "distributions.html": "b580abfd7dfdb82086541cbd99a2ae4f6cf8f868aa9caf188b3c6fad5ba76419",
    "games-g000001.html": "dcc679a264b00f67a328e8ecc06a490d46bd57d79d51750827ac799bafd9d1f1",
    "games-g000002.html": "8c64827a1cba200161a5e7191dbdd7b69ed9e0870a23e1cfbcc17597346e62fa",
    "drawn-g000001.html": "ca56fb188a12fa6a9999c7260a4d2d861552bee81865c486774434010cdf028d",
    "drawn-g000002.html": "7f768c716ec62bce4c814c1fe0681a6a710089478a50a29d6a819bb8bdb65deb",
    "teams-0001.html": "5d98a711e0921ea830cfee41d0869c23bd271e540cf571eb103557dbb4b47450",
    "replays/g000001/game-000000.html": (
        "36302c8f2f58b2c3104ff4c1f967958ac735d117e9a2c7dc0d0285239abf172d"
    ),
    "replays/index.html": "edee6dde38b43b82a4d7d95c89184272ea88a99b142069c5d39afb090a3eb066",
}


def built() -> dict[str, str]:
    pages = render.build_pages(fixture_model(), {(1, 0), (2, 3)})
    (game,) = replay.read_games(FIXTURE, 1, [0])[1]
    pages["replays/g000001/game-000000.html"] = replay_page.replay_page(
        game, replay_page.dex_for(table(), game)
    )
    pages["replays/index.html"] = replay_page.index_page({(1, 0), (2, 3)}, "inspector-fixture")
    return pages


def test_the_single_lineage_pages_are_byte_for_byte_what_they_were() -> None:
    pages = built()
    digests = {name: hashlib.sha256(pages[name].encode()).hexdigest() for name in PAGES}
    if os.environ.get("ARENEON_PRINT_PAGE_DIGESTS") == "1":
        for name, digest in digests.items():
            print(f'    "{name}": "{digest}",')
    assert digests == PAGES
    assert (ROOT / "fixtures" / "loop" / "inspector-run").is_dir()
