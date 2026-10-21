"""What a run trained on, joined and checked (ADR-017 section 4a).

``build_model`` reads a run directory read-only: it verifies every generation's manifest, reads the
loop's files strictly, takes the pool the loop's own function rebuilds, and refuses on any mismatch:

* a ``plan.json`` team index outside the pool;
* a ``team-set.json`` or ``generation.json`` team hash that is not a pool team (or is not the
  teams the plan draws);
* an Episode whose own team does not hash to the team its plan seat names, whose result differs from
  ``generation.json``, or that is not a synthetic league Episode of this run.

Games are joined to Episodes by battle seed and seat, never by file order. Only counts and ids are
taken from an Episode; nothing of its channel or decisions is kept.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import jcs, sha256_hex

from areneon_ml.inspector.pool import Pool
from areneon_ml.inspector.readers import (
    SEATS,
    GameRow,
    GenerationFile,
    InspectError,
    PlanEntry,
    read_generation,
    read_plan,
    read_team_set,
)
from areneon_ml.learner.config import RunConfig, RunConfigError, read_run_config
from areneon_ml.loop import layout
from areneon_ml.loop.league import (
    LeagueRecordError,
    LeagueRunRecord,
    check_generation_plan,
    read_league_run_record,
)

CONFIG_NAME = layout.CONFIG
RESULTS = ("win", "tie", "loss", "forfeit")
#: A synthetic, public-lineage run only: these are the labels an Episode of such a run carries.
PUBLIC_CLASS = "synthetic"
PUBLIC_RIGHTS = "cc0-synthetic"


@dataclass(frozen=True)
class Game:
    """One planned game of a generation, with the team indices and hashes its seats played."""

    generation: int
    plan: PlanEntry
    row: GameRow
    hashes: dict[str, str]

    @property
    def kind(self) -> str:
        """``self-play`` or the baseline's policy id."""
        return "self-play" if self.plan.opponent == "self" else self.plan.opponent

    def played_by(self, seat: str) -> str:
        """``pair`` for a seat the frozen pair played, ``baseline`` for the scripted one."""
        return "baseline" if self.plan.policy_seat not in ("both", seat) else "pair"

    def result(self, seat: str) -> str:
        if self.row.outcome == "forfeit":
            return "forfeit"
        if self.row.winner is None:
            return "tie"
        return "win" if self.row.winner == seat else "loss"


@dataclass(frozen=True)
class Generation:
    number: int
    pair_policy_id: str
    games: tuple[Game, ...]
    #: Distinct pool hashes this generation drew (sorted); the size of the cumulative team set.
    drawn: tuple[str, ...]
    team_set_size: int
    forfeits: int
    forfeit_causes: dict[str, int] | None
    episodes: int
    examples: int


@dataclass
class TeamUse:
    """Descriptive counts for one pool team: one appearance per seat it played."""

    draws: int = 0
    generations: set[int] = field(default_factory=set)
    games: list[tuple[int, int, str]] = field(default_factory=list)
    #: (opponent kind, who played the team) -> counts of games and of each result.
    results: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )


@dataclass(frozen=True)
class InspectModel:
    run_key: str
    lineage_id: str
    format_id: str
    config_hash: str
    pool: Pool
    generations: tuple[Generation, ...]
    use: dict[str, TeamUse]
    manifest_hashes: dict[int, str]
    #: Generation 0 (the seeded pair) plays no games: the generations that did.
    cumulative_drawn: Counter[str]


def read_run_dir_config(run_dir: Path) -> RunConfig:
    try:
        return read_run_config(layout.read_json(run_dir / CONFIG_NAME))
    except (layout.RunDirError, RunConfigError, OSError, json.JSONDecodeError) as exc:
        raise InspectError(f"{CONFIG_NAME}: {exc}") from exc


def refuse_non_public(config: RunConfig) -> None:
    """Refuse a run whose lineage is named private (ADR-017 section 4a).

    This matches the lineage name only, so it is a guard against an obvious mistake and not the
    defence. The defence is the data: ``read_episode_facts`` refuses any Episode that is not
    labelled synthetic and ``cc0-synthetic`` (and its opponent team synthetic), and the pool comes
    from the loop's own function over a verified synthetic dataset or the committed fixture teams.
    """
    lineage = config.lineage_id.lower()
    if lineage.startswith("private") or "private" in lineage.split("-"):
        raise InspectError(
            "the run's lineage is not a public one: only synthetic public runs are inspected"
        )


def generation_numbers(run_dir: Path) -> list[int]:
    base = layout.generations_dir(run_dir)
    if not base.is_dir():
        raise InspectError("the run directory has no generations")
    numbers = []
    for path in sorted(base.iterdir()):
        if path.name.startswith(".") and path.name.endswith(".partial"):
            continue  # unfinished (a run still going): not committed, so not read
        found = layout._GENERATION.fullmatch(path.name)
        if found is None or not path.is_dir():
            raise InspectError(f"generations/{path.name} is not a generation of this run")
        numbers.append(int(found[1]))
    if not numbers or numbers != list(range(len(numbers))):
        raise InspectError(f"the committed generations are {numbers}, not 0..n with no gap")
    return numbers


def _json(path: Path) -> Any:
    try:
        return layout.read_json(path)
    except (layout.RunDirError, OSError, json.JSONDecodeError) as exc:
        raise InspectError(f"{path.parent.name}/{path.name}: {exc}") from exc


@dataclass(frozen=True)
class EpisodeFacts:
    team_hash: str
    winner: str | None
    turns: int


def read_episode_facts(path: Path, run_key: str) -> tuple[str, str, EpisodeFacts]:
    """(seed, player, facts) of one Episode file: its counts and ids, nothing of its channel."""
    data = path.read_bytes()
    if hashlib.sha256(data.rstrip(b"\n")).hexdigest() != path.stem:
        raise InspectError(f"episodes/{path.name} does not hash to its name")
    try:
        episode = json.loads(data)
        player = episode["player"]
        collection = episode["provenance"]["collection"]
        sets = episode["team"]["team"]
        result = episode["result"]
        label = (episode["data_class"], episode["rights"], episode["team"]["data_class"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise InspectError(f"episodes/{path.name} is not an Episode this inspector reads") from exc
    if label != (PUBLIC_CLASS, PUBLIC_RIGHTS, PUBLIC_CLASS):
        raise InspectError(f"episodes/{path.name} is not a synthetic cc0-synthetic Episode")
    if (
        player not in SEATS
        or collection.get("kind") != "league"
        or collection.get("run_id") != run_key
        or collection.get("opponent_team_data_class") != PUBLIC_CLASS
        or not isinstance(collection.get("seed"), str)
        or not isinstance(result, dict)
    ):
        raise InspectError(f"episodes/{path.name} is not a league Episode of this run")
    turns = result.get("turns")
    if not isinstance(turns, int) or result.get("winner") not in (*SEATS, None):
        raise InspectError(f"episodes/{path.name} has a result this inspector does not read")
    return (
        collection["seed"],
        player,
        EpisodeFacts(sha256_hex(jcs(sets)), result["winner"], turns),
    )


def _episodes(
    path: Path, expected: tuple[str, ...], run_key: str
) -> dict[tuple[str, str], EpisodeFacts]:
    folder = path / "episodes"
    present = sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []
    if present != sorted(expected) or len(set(expected)) != len(expected):
        raise InspectError(
            f"{path.name}: the Episode files are not the episode_hashes of generation.json"
        )
    found: dict[tuple[str, str], EpisodeFacts] = {}
    for name in sorted(expected):
        seed, player, facts = read_episode_facts(folder / f"{name}.json", run_key)
        if (seed, player) in found:
            raise InspectError(f"{path.name}: two Episodes for one seed and seat")
        found[(seed, player)] = facts
    return found


def _episodes_by_seed(
    path: Path, expected: tuple[str, ...], run_key: str
) -> dict[str, dict[str, Path]]:
    """Every Episode file of a generation by battle seed and seat, after the same checks as
    ``_episodes``: the file names are the ``episode_hashes``, each file hashes to its name, and each
    is a synthetic league Episode of this run. The replay reads their content."""
    folder = path / "episodes"
    present = sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []
    if present != sorted(expected) or len(set(expected)) != len(expected):
        raise InspectError(
            f"{path.name}: the Episode files are not the episode_hashes of generation.json"
        )
    by_seed: dict[str, dict[str, Path]] = defaultdict(dict)
    for name in sorted(expected):
        seed, player, _ = read_episode_facts(folder / f"{name}.json", run_key)
        if player in by_seed[seed]:
            raise InspectError(f"{path.name}: two Episodes for one seed and seat")
        by_seed[seed][player] = folder / f"{name}.json"
    return dict(by_seed)


def _check_generation_files(
    number: int, path: Path, config: RunConfig, pool: Pool
) -> tuple[LeagueRunRecord, tuple[PlanEntry, ...], Any, tuple[str, ...]]:
    plan_raw = _json(path / "plan.json")
    try:
        record = read_league_run_record(_json(path / "league-run.json"), config)
        check_generation_plan(record, plan_raw, config)
    except LeagueRecordError as exc:
        raise InspectError(f"g{number:06d}: {exc}") from exc
    if record.generation != number:
        raise InspectError(
            f"g{number:06d}: its league-run.json is generation {record.generation}'s"
        )
    plan = read_plan(plan_raw)
    for entry in plan:
        for seat in SEATS:
            if entry.teams[seat] >= len(pool.teams):
                where = f"g{number:06d} game {entry.game}"
                index, size = entry.teams[seat], len(pool.teams)
                raise InspectError(f"{where}: team index {index} is outside the pool of {size}")
    generation = read_generation(_json(path / "generation.json"), number)
    team_set = read_team_set(_json(path / "team-set.json"))
    return record, plan, generation, team_set


def _join(
    number: int, plan: tuple[PlanEntry, ...], generation: Any, pool: Pool, pair: str
) -> tuple[Game, ...]:
    if len(generation.games) != len(plan):
        raise InspectError(
            f"g{number:06d}: generation.json and plan.json list different numbers of games"
        )
    games = []
    for entry, row in zip(plan, generation.games, strict=True):
        if row.seed != entry.seed:
            where = f"g{number:06d} game {entry.game}"
            raise InspectError(f"{where}: its seed differs between plan.json and generation.json")
        if row.opponent != (pair if entry.opponent == "self" else entry.opponent):
            raise InspectError(
                f"g{number:06d} game {entry.game}: its opponent differs between the two files"
            )
        hashes = {seat: pool.teams[entry.teams[seat]].hash for seat in SEATS}
        games.append(Game(number, entry, row, hashes))
    return tuple(games)


def _check_episodes(
    number: int, games: tuple[Game, ...], facts: dict[tuple[str, str], EpisodeFacts]
) -> None:
    claimed = 0
    for game in games:
        mine = [(seat, facts.get((game.row.seed, seat))) for seat in SEATS]
        if game.row.outcome == "forfeit":
            if any(f is not None for _, f in mine):
                raise InspectError(
                    f"g{number:06d} game {game.plan.game}: a forfeited game has an Episode"
                )
            continue
        for seat, episode in mine:
            where = f"g{number:06d} game {game.plan.game} seat {seat}"
            if episode is None:
                raise InspectError(f"{where}: no Episode")
            claimed += 1
            if episode.team_hash != game.hashes[seat]:
                raise InspectError(
                    f"{where}: the Episode's own team is not the team its plan seat names"
                )
            if episode.winner != game.row.winner or episode.turns != game.row.turns:
                raise InspectError(f"{where}: the Episode's result differs from generation.json")
    if claimed != len(facts):
        raise InspectError(f"g{number:06d}: an Episode matches no game of the plan")


def _check_counts(
    number: int, games: tuple[Game, ...], gen_file: GenerationFile, record: LeagueRunRecord
) -> None:
    """generation.json, league-run.json and the plan's rows must tell one story."""
    played = sum(1 for g in games if g.row.outcome == "played")
    if (gen_file.forfeits, record.played, record.forfeits) != (
        len(games) - played,
        played,
        len(games) - played,
    ):
        raise InspectError(f"g{number:06d}: the forfeit and played counts of its files differ")
    if record.episodes != len(gen_file.episode_hashes):
        raise InspectError(
            f"g{number:06d}: league-run.json counts other Episodes than generation.json lists"
        )
    causes = Counter(g.row.forfeit.cause for g in games if g.row.forfeit is not None)
    if gen_file.forfeit_causes is not None and gen_file.forfeit_causes != dict(causes):
        raise InspectError(f"g{number:06d}: forfeit_causes are not the causes of its game rows")
    per_kind = Counter(
        g.row.opponent if g.plan.opponent != "self" else record.pair_policy_id for g in games
    )
    if sorted(per_kind.items()) != sorted(record.opponents):
        raise InspectError(f"g{number:06d}: the opponents of league-run.json are not the plan's")


def _tally(games: tuple[Game, ...], use: dict[str, TeamUse], cumulative: Counter[str]) -> None:
    for game in games:
        for seat in SEATS:
            team_use = use[game.hashes[seat]]
            team_use.draws += 1
            team_use.generations.add(game.generation)
            team_use.games.append((game.generation, game.plan.game, seat))
            tally = team_use.results[(game.kind, game.played_by(seat))]
            tally["games"] += 1
            tally[game.result(seat)] += 1
            cumulative[game.hashes[seat]] += 1


def _check_pool(pool: Pool, config: RunConfig) -> None:
    declared = config.raw.get("team_dataset")
    if declared is not None and declared["dataset_hash"] != pool.dataset_hash:
        raise InspectError("the pool's dataset is not the one the run config names")
    if (declared is None) != (pool.source == "interim"):
        raise InspectError("the pool's source does not match the run config")
    if tuple(pool.excluded_abilities) != tuple(config.generation.excluded_abilities):
        raise InspectError(
            "the pool was rebuilt with other excluded abilities than the run config's"
        )


def _read_played(
    number: int, path: Path, config: RunConfig, pool: Pool, seen: set[str]
) -> tuple[Generation, tuple[Game, ...]]:
    """One generation with games: its files read, joined to its Episodes and checked."""
    record, plan, gen_file, team_set = _check_generation_files(number, path, config, pool)
    games = _join(number, plan, gen_file, pool, record.pair_policy_id)
    _check_episodes(number, games, _episodes(path, gen_file.episode_hashes, config.run_key))
    drawn = tuple(sorted({h for g in games for h in g.hashes.values()}))
    if gen_file.team_hashes != drawn:
        raise InspectError(
            f"g{number:06d}: generation.json's team_hashes are not the teams its plan draws"
        )
    seen |= set(drawn)
    if not set(team_set) <= {t.hash for t in pool.teams}:
        raise InspectError(f"g{number:06d}: its team-set.json holds a team that is not in the pool")
    if set(team_set) != seen:
        raise InspectError(
            f"g{number:06d}: its team-set.json is not the union of the teams drawn so far"
        )
    _check_counts(number, games, gen_file, record)
    generation = Generation(
        number,
        record.pair_policy_id,
        games,
        drawn,
        len(team_set),
        gen_file.forfeits,
        gen_file.forfeit_causes,
        record.episodes,
        gen_file.examples,
    )
    return generation, games


def build_model(run_dir: Path, pool: Pool, config: RunConfig) -> InspectModel:
    refuse_non_public(config)
    _check_pool(pool, config)
    use: dict[str, TeamUse] = {t.hash: TeamUse() for t in pool.teams}
    cumulative: Counter[str] = Counter()
    seen: set[str] = set()
    manifest_hashes: dict[int, str] = {}
    generations: list[Generation] = []
    for number in generation_numbers(run_dir):
        path = layout.generation_dir(run_dir, number)
        try:
            layout.verify_generation(path, number)
        except (layout.RunDirError, OSError, json.JSONDecodeError) as exc:
            raise InspectError(str(exc)) from exc
        manifest_hashes[number] = sha256_hex((path / layout.MANIFEST).read_bytes().rstrip(b"\n"))
        if number == 0:
            if read_team_set(_json(path / "team-set.json")):
                raise InspectError("g000000 (no games) lists teams in its team-set.json")
            continue
        generation, games = _read_played(number, path, config, pool, seen)
        _tally(games, use, cumulative)
        generations.append(generation)
    return InspectModel(
        config.run_key,
        config.lineage_id,
        config.format_id,
        config.hash,
        pool,
        tuple(generations),
        use,
        manifest_hashes,
        cumulative,
    )
