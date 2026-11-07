"""What a population run trained on, joined and checked (ADR-021 sections 2, 7, 8; ADR-017 4a).

``build_population_model`` reads a population run directory read-only, as :mod:`.model` reads a
single lineage's. It verifies every generation's manifest, reads the loop's files strictly (the
closed-key readers of :mod:`.population_readers` for ``plan.json`` and ``generation.json``, the
loop's own readers for ``league-run.json`` (version 2), ``ladder.json``, ``past-pool.json`` and each
member's checkpoint records), takes the pool the loop's own function rebuilds, and refuses on any
mismatch:

* a plan seat or row that disagrees with the other file, a team index outside the pool, a past
  pair that names a snapshot the committed pool of the generation before does not hold;
* an Episode that is not in the member directory that owns its seat, whose own team is not the team
  its plan seat names, whose policy is not the one that holds the seat, or whose result differs
  from ``generation.json``; an Episode file no game claims;
* a manifest that names another ladder, past pool or checkpoint than the files hold.

Games are joined to Episodes by battle seed, seat, team and the policy that played the seat (the
two halves of a team pair share a seed), never by file order. Only counts and ids are taken from an
Episode; nothing of its channel or decisions is kept here.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, sha256_hex

from areneon_ml.checkpoint.record import CheckpointRecord, RecordError, read_record
from areneon_ml.inspector.model import (
    CONFIG_NAME,
    EpisodeFacts,
    TeamUse,
    _json,
    generation_numbers,
    read_episode_header,
    refuse_non_public,
)
from areneon_ml.inspector.pool import Pool
from areneon_ml.inspector.population_readers import (
    PopGameRow,
    PopGenerationFile,
    PopPlanEntry,
    PopSeat,
    manifest_member,
    read_population_generation,
    read_population_plan,
    snapshot_parts,
)
from areneon_ml.inspector.readers import SEATS, InspectError, read_team_set
from areneon_ml.learner.config import RunConfig, RunConfigError, read_run_config
from areneon_ml.loop import layout, past_pool
from areneon_ml.loop.ladder import LadderRecordError, read_ladder_record
from areneon_ml.loop.league import (
    LeagueRecordError,
    LeagueRunRecordV2,
    MemberEntry,
    check_population_plan,
    read_league_run_record_v2,
)
from areneon_ml.loop.selection import quarantined

ROLES = ("preview", "battle")


def is_population(raw: object) -> bool:
    """A run config that has a ``population`` section."""
    return isinstance(raw, dict) and "population" in raw


def read_population_config(run_dir: Path) -> RunConfig:
    """The run's config, which must be a population run's."""
    try:
        raw = layout.read_json(run_dir / CONFIG_NAME)
        if not is_population(raw):
            raise InspectError(f"{CONFIG_NAME}: the run has no population section")
        return read_run_config(raw)
    except (layout.RunDirError, RunConfigError, OSError, json.JSONDecodeError) as exc:
        raise InspectError(f"{CONFIG_NAME}: {exc}") from exc


def run_is_population(run_dir: Path) -> bool:
    """Whether the run directory's config is a population run's (unreadable: not, so the
    single-lineage reader reports it in its own words)."""
    try:
        return is_population(layout.read_json(run_dir / CONFIG_NAME))
    except (layout.RunDirError, OSError, json.JSONDecodeError):
        return False


def seat_class(seat: PopSeat) -> str:
    """``member``, ``past`` (a frozen snapshot) or ``baseline``."""
    return {"member": "member", "snapshot": "past", "baseline": "baseline"}[seat.kind]


@dataclass(frozen=True)
class PopGame:
    """One planned game of a population generation, with the team and policy of each seat."""

    generation: int
    plan: PopPlanEntry
    row: PopGameRow
    hashes: dict[str, str]
    #: The policy id that played each seat (the member's pair for a member seat).
    policies: dict[str, str]

    @property
    def kind(self) -> str:
        return self.plan.kind

    def played_by(self, seat: str) -> str:
        """Who played the seat, as the team tables name it."""
        held = self.plan.seats[seat]
        return held.label if held.kind != "baseline" else f"baseline {held.label}"

    def result(self, seat: str) -> str:
        if self.row.outcome == "forfeit":
            return "forfeit"
        if self.row.winner is None:
            return "tie"
        return "win" if self.row.winner == seat else "loss"

    def members_on(self) -> tuple[str, ...]:
        """The learning members that hold a seat, in seat order (a self-play game names one)."""
        return tuple(
            dict.fromkeys(
                s.label for s in (self.plan.seats[x] for x in SEATS) if s.kind == "member"
            )
        )


@dataclass(frozen=True)
class MemberState:
    """One member in one generation: the pair it played, the pair it froze, and the ladder.

    A generation plays the pair frozen at the end of the one before. ``copied_from`` is set when
    the copy step made the slot train from another member's frozen pair (the new records' parent
    is the donor's checkpoint). Generation 0 has no played pair (empty ids).
    """

    member: str
    pair_policy_id: str
    played_preview: str
    played_battle: str
    #: The checkpoints this generation froze for the member (from its sealed records).
    preview: str
    battle: str
    copied_from: str | None
    games: int
    played: int
    episodes: int
    examples: int
    root: str | None
    strength: tuple[int, int, int] | None
    #: The cause when the ladder record quarantines the member in this generation (ADR-021
    #: section 12): its update was discarded and its sealed training set is null.
    quarantined: str | None = None
    #: A quarantined member froze the very records of the generation before (no new checkpoint).
    held: bool = False


@dataclass(frozen=True)
class PopGeneration:
    number: int
    games: tuple[PopGame, ...]
    drawn: tuple[str, ...]
    team_set_size: int
    forfeits: int
    forfeit_causes: dict[str, int] | None
    episodes: int
    members: tuple[MemberState, ...]
    #: The decision of the ladder's first selection (the copy step), as the record states it.
    selection: dict[str, Any] | None
    #: The pool the generation PLAYED (the committed pool of the generation before): snapshots.
    snapshots: tuple[past_pool.Snapshot, ...]
    #: The pool the generation committed for the next one.
    committed_snapshots: tuple[str, ...]
    window: tuple[int, int]

    def snapshot(self, snapshot_id: str) -> past_pool.Snapshot:
        return next(s for s in self.snapshots if s.id == snapshot_id)


@dataclass(frozen=True)
class PopulationModel:
    run_key: str
    lineage_id: str
    format_id: str
    config_hash: str
    pool: Pool
    generations: tuple[PopGeneration, ...]
    use: dict[str, TeamUse]
    manifest_hashes: dict[int, str]
    cumulative_drawn: Counter[str]
    population: dict[str, Any]
    members: tuple[str, ...]
    #: Generation 0's members (the seeded pairs): they play no games.
    initial: tuple[MemberState, ...]


@dataclass(frozen=True)
class _Episode:
    owner: str
    facts: EpisodeFacts
    opponent_policy_id: str | None
    path: Path


def _records(
    path: Path, members: tuple[str, ...], config: RunConfig
) -> dict[str, dict[str, CheckpointRecord]]:
    out: dict[str, dict[str, CheckpointRecord]] = {}
    for member in members:
        out[member] = {}
        for role in ROLES:
            try:
                record = read_record(_json(path / "members" / member / "records" / f"{role}.json"))
            except (RecordError, ContractError) as exc:
                raise InspectError(f"{path.name}: {member}'s {role} record: {exc}") from exc
            if record.role != role or record.run_config_hash != config.hash:
                raise InspectError(
                    f"{path.name}: {member}'s {role} record is not this run's {role} record"
                )
            if record.lineage_id != f"{config.lineage_id}.{member}":
                raise InspectError(f"{path.name}: {member}'s {role} record names another lineage")
            if record.data_class != "synthetic":
                raise InspectError(f"{path.name}: {member}'s {role} record is not synthetic")
            out[member][role] = record
    return out


def _past_file(path: Path, number: int, config: RunConfig) -> past_pool.PastPool | None:
    if not past_pool.enabled(config):
        return None
    raw = _json(path / past_pool.FILE)
    manifest = layout.read_json(path / layout.MANIFEST)
    if "past_pool_hash" in manifest and manifest["past_pool_hash"] != sha256_hex(jcs(raw)):
        raise InspectError(f"g{number:06d}: its past-pool.json is not the one its manifest names")
    try:
        found = past_pool.read_file(raw, config)
    except past_pool.PastPoolError as exc:
        raise InspectError(f"g{number:06d}: past-pool.json: {exc}") from exc
    if found.generation != number:
        raise InspectError(f"g{number:06d}: its past-pool.json is of another generation")
    return found


def _members_of(config: RunConfig) -> tuple[str, ...]:
    count = config.raw["population"]["members"]
    if config.raw["population"]["exploiters"] != 0:
        raise InspectError("an exploiter seat is not supported by the inspector yet")
    return tuple(f"m{k}" for k in range(count))


def _episodes(
    path: Path, files: tuple[Any, ...], config: RunConfig
) -> dict[tuple[str, str, str, str], _Episode]:
    """Every Episode of a generation by (seed, seat, own team, policy that played the seat)."""
    found: dict[tuple[str, str, str, str], _Episode] = {}
    for entry in files:
        folder = path / "members" / entry.member / "episodes"
        present = sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []
        if present != list(entry.episode_hashes):
            raise InspectError(
                f"{path.name}: {entry.member}'s Episode files are not its episode_hashes"
            )
        for name in entry.episode_hashes:
            seed, player, facts, links = read_episode_header(
                folder / f"{name}.json", config.run_key
            )
            if links.behaviour_policy_id is None:
                raise InspectError(f"episodes/{name}.json names no behaviour policy")
            key = (seed, player, facts.team_hash, links.behaviour_policy_id)
            if key in found:
                raise InspectError(f"{path.name}: two Episodes for one seed, seat, team and policy")
            found[key] = _Episode(
                entry.member, facts, links.opponent_policy_id, folder / f"{name}.json"
            )
    return found


def _policies(
    plan: PopPlanEntry, record: LeagueRunRecordV2, pool: past_pool.PastPool | None
) -> dict[str, str]:
    """The policy id that plays each seat of the game."""
    pairs = {m.member: m.pair_policy_id for m in record.members}
    out = {}
    for seat in SEATS:
        held = plan.seats[seat]
        if held.kind == "member":
            if held.label not in pairs:
                raise InspectError(f"game {plan.game}: its seat names a member the record lacks")
            out[seat] = pairs[held.label]
            continue
        assert held.policy_id is not None
        out[seat] = held.policy_id
        if held.kind == "snapshot":
            snap = (
                None
                if pool is None
                else next((s for s in pool.snapshots if s.id == held.label), None)
            )
            if snap is None or snap.policy_id != held.policy_id:
                raise InspectError(
                    f"game {plan.game}: its past pair is not in the committed past pool of the "
                    "generation before"
                )
    return out


def _join(  # noqa: PLR0913, PLR0917
    number: int,
    plan: tuple[PopPlanEntry, ...],
    gen_file: PopGenerationFile,
    pool: Pool,
    record: LeagueRunRecordV2,
    played_pool: past_pool.PastPool | None,
) -> tuple[PopGame, ...]:
    if len(gen_file.games) != len(plan):
        raise InspectError(
            f"g{number:06d}: generation.json and plan.json list different numbers of games"
        )
    games = []
    for entry, row in zip(plan, gen_file.games, strict=True):
        where = f"g{number:06d} game {entry.game}"
        labels = {s: entry.seats[s].label for s in SEATS}
        if (row.seed, row.pair, row.half, row.kind, row.member) != (
            entry.seed,
            entry.pair,
            entry.half,
            entry.kind,
            entry.member,
        ) or row.seats != labels:
            raise InspectError(f"{where}: plan.json and generation.json differ")
        initiating = "p1" if entry.half == 0 else "p2"
        if entry.kind == "self":
            if set(labels.values()) != {entry.member} or row.opponent != entry.member:
                raise InspectError(f"{where}: a self-play game is the member against itself")
        else:
            other = "p2" if initiating == "p1" else "p1"
            if labels[initiating] != entry.member or row.opponent != labels[other]:
                raise InspectError(f"{where}: its initiating seat or opponent is not the row's")
        kinds = {s: entry.seats[s].kind for s in SEATS}
        wanted = {
            "ladder": {"member"},
            "self": {"member"},
            "past": {"member", "snapshot"},
            "anchor": {"member", "baseline"},
        }[entry.kind]
        if set(kinds.values()) != wanted:
            raise InspectError(f"{where}: its seats do not fit a {entry.kind} game")
        for seat in SEATS:
            if entry.teams[seat] >= len(pool.teams):
                size = len(pool.teams)
                raise InspectError(
                    f"{where}: team index {entry.teams[seat]} is outside the pool of {size}"
                )
        hashes = {seat: pool.teams[entry.teams[seat]].hash for seat in SEATS}
        games.append(PopGame(number, entry, row, hashes, _policies(entry, record, played_pool)))
    return tuple(games)


def _check_episodes(
    number: int, games: tuple[PopGame, ...], found: dict[tuple[str, str, str, str], _Episode]
) -> None:
    claimed = 0
    for game in games:
        entry, row = game.plan, game.row
        where = f"g{number:06d} game {entry.game}"
        keys = {seat: (row.seed, seat, game.hashes[seat], game.policies[seat]) for seat in SEATS}
        if row.outcome == "forfeit":
            if any(k in found for k in keys.values()):
                raise InspectError(f"{where}: a forfeited game has an Episode")
            continue
        for seat in SEATS:
            episode = found.get(keys[seat])
            if episode is None:
                raise InspectError(f"{where} seat {seat}: no Episode")
            claimed += 1
            held = entry.seats[seat]
            owner = held.label if held.kind == "member" else entry.member
            if episode.owner != owner:
                raise InspectError(
                    f"{where} seat {seat}: its Episode is not in the directory of {owner}"
                )
            if (episode.facts.winner, episode.facts.turns) != (row.winner, row.turns):
                raise InspectError(
                    f"{where} seat {seat}: the Episode's result differs from generation.json"
                )
            other = game.policies["p2" if seat == "p1" else "p1"]
            if episode.opponent_policy_id not in (None, other):
                raise InspectError(f"{where} seat {seat}: the Episode names another opponent")
    if claimed != len(found):
        raise InspectError(f"g{number:06d}: an Episode matches no game of the plan")


def _check_counts(
    number: int, games: tuple[PopGame, ...], gen_file: PopGenerationFile, record: LeagueRunRecordV2
) -> None:
    played = sum(1 for g in games if g.row.outcome == "played")
    if (gen_file.forfeits, record.played, record.forfeits, record.games) != (
        len(games) - played,
        played,
        len(games) - played,
        len(games),
    ):
        raise InspectError(f"g{number:06d}: the forfeit and played counts of its files differ")
    total = sum(len(m.episode_hashes) for m in gen_file.members)
    if record.episodes != total or total != 2 * played:
        raise InspectError(
            f"g{number:06d}: league-run.json counts other Episodes than generation.json lists"
        )
    causes = Counter(g.row.forfeit.cause for g in games if g.row.forfeit is not None)
    if (gen_file.forfeit_causes or {}) != dict(causes):
        raise InspectError(f"g{number:06d}: forfeit_causes are not the causes of its game rows")


def _origin(
    previous: Mapping[str, Mapping[str, CheckpointRecord]] | None,
    records: Mapping[str, Mapping[str, CheckpointRecord]],
    entry: MemberEntry | None,
    member: str,
) -> str | None:
    """The member whose frozen pair this member's slot trained from in the generation (a copy
    applied at its start: the new records' parent is the donor's checkpoint), else None.

    The pair a member PLAYS is its own, frozen the generation before; the copy changes what it
    trains from. A parent that is neither the played pair nor another member's frozen pair is
    refused.
    """
    if previous is None or entry is None:
        return None
    played = {"preview": entry.preview, "battle": entry.battle}
    found: set[str | None] = set()
    for role in ROLES:
        parent = records[member][role].parent
        if parent == played[role]:
            found.add(None)
            continue
        donors = [
            other
            for other, roles in previous.items()
            if other != member and roles[role].checkpoint_id == parent
        ]
        if not donors:
            raise InspectError(f"{member}'s {role} record names a parent no member froze")
        found.add(donors[0])
    if len(found) != 1:
        raise InspectError(f"{member}'s Preview and Battle records name different parents")
    return found.pop()


def _held(
    previous: Mapping[str, Mapping[str, CheckpointRecord]] | None,
    records: Mapping[str, Mapping[str, CheckpointRecord]],
    member: str,
) -> bool:
    """Whether the member's records are the generation before's, byte for byte (a discarded update
    freezes the pair the member already had). A quarantined member that was replaced by a copy in
    the very generation keeps the copy's ordinary records and is not held."""
    if previous is None:
        return False
    return all(
        records[member][role].record_hash == previous[member][role].record_hash for role in ROLES
    )


def _states(  # noqa: PLR0913, PLR0917
    members: tuple[str, ...],
    record: LeagueRunRecordV2 | None,
    records: Mapping[str, Mapping[str, CheckpointRecord]],
    previous: Mapping[str, Mapping[str, CheckpointRecord]] | None,
    gen_file: PopGenerationFile | None,
    ladder: dict[str, Any] | None,
    manifest: dict[str, Any],
    number: int,
) -> tuple[MemberState, ...]:
    entries = {m.member: m for m in (record.members if record else ())}
    files = {m.member: m for m in (gen_file.members if gen_file else ())}
    rated = {m["member"]: m for m in (ladder["members"] if ladder else ())}
    stopped = quarantined(ladder) if ladder else {}
    out = []
    for member in members:
        sealed = manifest_member(manifest, member)["checkpoints"]
        ids = {role: records[member][role].checkpoint_id for role in ROLES}
        if sealed != ids:
            raise InspectError(f"g{number:06d}: {member}'s records are not the checkpoints sealed")
        entry = entries.get(member)
        cause = stopped.get(member)
        held = cause is not None and _held(previous, records, member)
        copied = None if held else _origin(previous, records, entry, member)
        file = files.get(member)
        sealed_set = manifest_member(manifest, member).get("training_set_hash")
        if file is not None and (sealed_set is None) != (cause is not None):
            raise InspectError(
                f"g{number:06d}: {member}'s sealed training set is null exactly when the ladder "
                "record quarantines it"
            )
        if file is not None and sealed_set is not None and sealed_set != file.training_set_hash:
            raise InspectError(f"g{number:06d}: {member}'s training set is not the one sealed")
        mine = rated.get(member)
        strength = mine["strength"] if mine else None
        out.append(
            MemberState(
                member,
                entry.pair_policy_id if entry else "",
                entry.preview if entry else "",
                entry.battle if entry else "",
                ids["preview"],
                ids["battle"],
                copied,
                entry.games if entry else 0,
                entry.played if entry else 0,
                entry.episodes if entry else 0,
                file.examples if file else 0,
                mine["root"] if mine else None,
                None
                if strength is None
                else (strength["lower"], strength["strength"], strength["upper"]),
                cause,
                held,
            )
        )
    return tuple(out)


@dataclass
class _Walk:
    """What the generation loop carries from one generation to the next."""

    seen: set[str]
    previous_records: dict[str, dict[str, CheckpointRecord]] | None
    previous_ladder: Any
    previous_pool: past_pool.PastPool | None


def _read_generation(  # noqa: PLR0913, PLR0917
    number: int,
    path: Path,
    config: RunConfig,
    pool: Pool,
    members: tuple[str, ...],
    walk: _Walk,
) -> PopGeneration:
    manifest = layout.read_json(path / layout.MANIFEST)
    plan_raw = _json(path / "plan.json")
    try:
        record = read_league_run_record_v2(_json(path / "league-run.json"), config)
        check_population_plan(record, plan_raw, config)
    except LeagueRecordError as exc:
        raise InspectError(f"g{number:06d}: {exc}") from exc
    if record.generation != number:
        raise InspectError(
            f"g{number:06d}: its league-run.json is generation {record.generation}'s"
        )
    plan = read_population_plan(plan_raw)
    gen_file = read_population_generation(_json(path / "generation.json"), number)
    if tuple(m.member for m in gen_file.members) != members:
        raise InspectError(f"g{number:06d}: generation.json lists other members than the config")
    team_set = read_team_set(_json(path / "team-set.json"))
    ladder_raw = _json(path / "ladder.json")
    if manifest.get("ladder_hash") != sha256_hex(jcs(ladder_raw)):
        raise InspectError(f"g{number:06d}: its ladder.json is not the one its manifest names")
    try:
        ladder = read_ladder_record(ladder_raw, walk.previous_ladder, config).raw
    except LadderRecordError as exc:
        raise InspectError(f"g{number:06d}: ladder.json: {exc}") from exc
    if ladder["generation"] != number:
        raise InspectError(f"g{number:06d}: its ladder.json is of another generation")
    own_pool = _past_file(path, number, config)
    games = _join(number, plan, gen_file, pool, record, walk.previous_pool)
    _check_episodes(number, games, _episodes(path, gen_file.members, config))
    _check_counts(number, games, gen_file, record)
    drawn = tuple(sorted({h for g in games for h in g.hashes.values()}))
    if gen_file.team_hashes != drawn:
        raise InspectError(
            f"g{number:06d}: generation.json's team_hashes are not the teams its plan draws"
        )
    walk.seen |= set(drawn)
    if not set(team_set) <= {t.hash for t in pool.teams} or set(team_set) != walk.seen:
        raise InspectError(
            f"g{number:06d}: its team-set.json is not the union of the teams drawn so far"
        )
    records = _records(path, members, config)
    states = _states(
        members, record, records, walk.previous_records, gen_file, ladder, manifest, number
    )
    selection = ladder["selections"][0] if ladder["selections"] else None
    window = (ladder["window"]["first_generation"], ladder["window"]["last_generation"])
    generation = PopGeneration(
        number,
        games,
        drawn,
        len(team_set),
        gen_file.forfeits,
        gen_file.forfeit_causes,
        record.episodes,
        states,
        selection,
        walk.previous_pool.snapshots if walk.previous_pool else (),
        tuple(s.id for s in own_pool.snapshots) if own_pool else (),
        window,
    )
    walk.previous_records, walk.previous_ladder, walk.previous_pool = records, ladder_raw, own_pool
    return generation


def _tally(games: tuple[PopGame, ...], use: dict[str, TeamUse], cumulative: Counter[str]) -> None:
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


def build_population_model(run_dir: Path, pool: Pool, config: RunConfig) -> PopulationModel:
    refuse_non_public(config)
    declared = config.raw.get("team_dataset")
    if (declared is not None and declared["dataset_hash"] != pool.dataset_hash) or (
        (declared is None) != (pool.source == "interim")
    ):
        raise InspectError("the pool's source does not match the run config")
    if tuple(pool.excluded_abilities) != tuple(config.generation.excluded_abilities):
        raise InspectError(
            "the pool was rebuilt with other excluded abilities than the run config's"
        )
    members = _members_of(config)
    use: dict[str, TeamUse] = {t.hash: TeamUse() for t in pool.teams}
    cumulative: Counter[str] = Counter()
    manifest_hashes: dict[int, str] = {}
    generations: list[PopGeneration] = []
    walk = _Walk(set(), None, None, None)
    initial: tuple[MemberState, ...] = ()
    for number in generation_numbers(run_dir):
        path = layout.generation_dir(run_dir, number)
        try:
            manifest = layout.verify_generation(path, number)
        except (layout.RunDirError, OSError, json.JSONDecodeError) as exc:
            raise InspectError(str(exc)) from exc
        manifest_hashes[number] = sha256_hex((path / layout.MANIFEST).read_bytes().rstrip(b"\n"))
        if number == 0:
            # Generation 0 plays no training games; its team set is the anchor calibration's teams.
            calibration = set(read_team_set(_json(path / "team-set.json")))
            if not calibration <= {t.hash for t in pool.teams}:
                raise InspectError(
                    "g000000: its team-set.json holds a team that is not in the pool"
                )
            walk.seen |= calibration
            records = _records(path, members, config)
            initial = _states(members, None, records, None, None, None, manifest, 0)
            walk.previous_records, walk.previous_pool = records, _past_file(path, 0, config)
            continue
        generation = _read_generation(number, path, config, pool, members, walk)
        _tally(generation.games, use, cumulative)
        generations.append(generation)
    return PopulationModel(
        config.run_key,
        config.lineage_id,
        config.format_id,
        config.hash,
        pool,
        tuple(generations),
        use,
        manifest_hashes,
        cumulative,
        config.raw["population"],
        members,
        initial,
    )


__all__ = [
    "MemberState",
    "PopGame",
    "PopGeneration",
    "PopulationModel",
    "build_population_model",
    "is_population",
    "read_population_config",
    "run_is_population",
    "seat_class",
    "snapshot_parts",
]
