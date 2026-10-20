"""The entity-token encoder (ADR-010 §3): a player's observation to tensors.

Input is exactly what the policy boundary allows (ADR-010 §2): the
observation's own channel (reduced by the clean-room reducer), the player's
own team, and the candidate list with each candidate's ``pending`` flag. It
reads nothing else of a query: not ``decision_context``, not ``identity``
(so not ``rqid``), not the ``rqid`` inside the log's ``|request|`` lines, and
never a verdict of the CloneJudge or an opponent's hidden information, none of
which the observation holds. The A1 and A2 forms of one decision therefore
give identical tensors.

Tokens: one per own Pokémon, one per opponent Pokémon (species from Team
Preview, then only what the reducer extracted, with explicit unknown values),
and the network adds a field token (``field_*``) and an aggregation token. A
candidate is described by its parsed parts and by which tokens it names
(the acting Pokémon, the switch-in, the picked team positions), taken from the
player's own last request. Free-form ids (volatiles, field and side
conditions) are bucketed by SHA-256 into a configured number of buckets: the
engine owns those names, not this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from areneon_contracts.contracts import jcs, sha256_hex

from areneon_ml.model.candidates import (
    EVENTS,
    SLOTS,
    ParsedChoice,
    UnsupportedCandidate,
    parse_choice,
)
from areneon_ml.model.gamedata import NONE, UNKNOWN, GameData
from areneon_ml.reducer import BattleState, PokemonState, SideState, reduce_channel
from areneon_ml.reducer.state import STAT_KEYS

# Bump when the meaning of a feature changes without its name or width changing.
ENCODER_FORMAT = 1

REQUEST_KINDS = ("preview", "move", "forceSwitch")
GENDERS = (None, "M", "F", "N")
MAX_SLOT_LETTERS = 3  # triples is the widest gametype the reducer reads
TARGET_CODES = 20  # 0 untargeted, else 10 + the engine's target location (-9..9)
_REQUEST = "|request|"
_SIDE = re.compile(r"^p[1-4]: ")
STAT_NAMES = ("atk", "def", "spa", "spd", "spe")


class EncodingError(ValueError):
    """The observation or a candidate cannot be encoded; nothing is guessed."""


@dataclass(frozen=True)
class EncoderConfig:
    """Feature widths and scales: configuration, part of ``encoder_version``."""

    volatile_buckets: int = 32
    condition_buckets: int = 32
    max_moves: int = 4  # moves per Pokémon (the validator allows at most four)
    max_team: int = 6  # Pokémon per side
    max_picks: int = 4  # Team Preview positions an order names
    stat_scale: float = 500.0  # divisor of a Pokémon's exact stats
    turn_scale: float = 100.0  # divisor of the turn counter

    def __post_init__(self) -> None:
        for name in ("volatile_buckets", "condition_buckets", "max_moves", "max_team", "max_picks"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.stat_scale <= 0 or self.turn_scale <= 0:
            raise ValueError("scales must be positive")
        if self.max_picks > self.max_team:
            raise ValueError("max_picks cannot exceed max_team")

    def as_dict(self) -> dict[str, Any]:
        return {
            "volatile_buckets": self.volatile_buckets,
            "condition_buckets": self.condition_buckets,
            "max_moves": self.max_moves,
            "max_team": self.max_team,
            "max_picks": self.max_picks,
            "stat_scale": self.stat_scale,
            "turn_scale": self.turn_scale,
        }


@dataclass(frozen=True)
class Layout:
    """Names and kinds of every feature column, derived from the configuration."""

    mon_cat: tuple[tuple[str, str], ...]  # (column, vocabulary kind)
    mon_num: tuple[str, ...]
    field_cat: tuple[tuple[str, str], ...]
    field_num: tuple[str, ...]
    cand_cat: tuple[tuple[str, str], ...]

    @classmethod
    def of(cls, config: EncoderConfig) -> Layout:
        moves = tuple((f"move_{i + 1}", "move") for i in range(config.max_moves))
        buckets = config.condition_buckets
        return cls(
            mon_cat=(
                ("base_species", "species"),
                ("species", "species"),
                ("temp_species", "species"),
                ("item", "item"),
                ("mega_stone", "item"),
                ("ability", "ability"),
                ("status", "status"),
                ("gender", "gender"),
                ("active_slot", "slot"),
                *moves,
            ),
            mon_num=(
                "own",
                "hp_known",
                "hp_fraction",
                "fainted",
                "brought_known",
                "brought",
                "shiny",
                "mega",
                "item_known",
                "item_lost",
                "ability_suppressed",
                "can_mega_evo",
                "level",
                "stats_known",
                "hp_stat",
                *(f"stat_{name}" for name in STAT_NAMES),
                *(f"boost_{name}" for name in STAT_KEYS),
                *(f"pp_{i + 1}" for i in range(config.max_moves)),
                *(f"volatile_{i}" for i in range(config.volatile_buckets)),
            ),
            field_cat=(("weather", "weather"), ("terrain", "terrain"), ("request", "request")),
            field_num=(
                "turn",
                "started",
                *(f"field_{i}" for i in range(buckets)),
                *(f"own_side_{i}" for i in range(buckets)),
                *(f"opponent_side_{i}" for i in range(buckets)),
            ),
            cand_cat=(
                ("action", "action"),
                ("move", "move"),
                ("move_slot", "move_slot"),
                ("target", "target"),
                ("event", "event"),
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mon_cat": [list(c) for c in self.mon_cat],
            "mon_num": list(self.mon_num),
            "field_cat": [list(c) for c in self.field_cat],
            "field_num": list(self.field_num),
            "cand_cat": [list(c) for c in self.cand_cat],
        }


def encoder_version(game: GameData, config: EncoderConfig) -> str:
    """``enc<format>:`` + SHA-256 of the engine-data hash, the config and the feature layout.

    A pin bump changes the engine-data hash, so it yields a new encoder
    version, never a silent remap (ADR-010 §3).
    """
    body = {
        "encoder_format": ENCODER_FORMAT,
        "engine_data_hash": game.engine_data_hash,
        "config": config.as_dict(),
        "layout": Layout.of(config).as_dict(),
    }
    return f"enc{ENCODER_FORMAT}:{sha256_hex(jcs(body))}"


def bucket(text: str, buckets: int) -> int:
    """A stable bucket of an engine-owned name (SHA-256, not Python's salted hash)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") % buckets


@dataclass(frozen=True)
class EncodedQuery:
    """One query as tensors, without a batch dimension.

    ``mon_*`` rows are the own Pokémon first (``n_own``), then the opponent's.
    Candidate token columns hold a row of ``mon_*`` or -1 for none.
    """

    kind: str
    n_own: int
    mon_cat: torch.Tensor  # int64 [M, len(Layout.mon_cat)]
    mon_num: torch.Tensor  # float32 [M, len(Layout.mon_num)]
    field_cat: torch.Tensor  # int64 [len(Layout.field_cat)]
    field_num: torch.Tensor  # float32 [len(Layout.field_num)]
    cand_cat: torch.Tensor  # int64 [N, SLOTS, len(Layout.cand_cat)]
    cand_tok: torch.Tensor  # int64 [N, SLOTS, 2]: acting Pokémon, switch-in
    team_tok: torch.Tensor  # int64 [N, max_picks]: picked Pokémon in order (Team Preview)
    pending: torch.Tensor  # float32 [N]

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "mon_cat": self.mon_cat,
            "mon_num": self.mon_num,
            "field_cat": self.field_cat,
            "field_num": self.field_num,
            "cand_cat": self.cand_cat,
            "cand_tok": self.cand_tok,
            "team_tok": self.team_tok,
            "pending": self.pending,
        }


@dataclass
class _Request:
    """The parts of the player's own last request a candidate refers to."""

    order: list[int] = field(default_factory=list)  # own token of side.pokemon[0], [1], ...
    moves: list[list[int]] = field(default_factory=list)  # per active slot: engine move indices


class Encoder:
    """Observation + candidates to ``EncodedQuery``: pure and deterministic."""

    def __init__(self, game: GameData, config: EncoderConfig | None = None) -> None:
        self.game = game
        self.config = config or EncoderConfig()
        self.layout = Layout.of(self.config)
        self.version = encoder_version(game, self.config)

    # -- public --------------------------------------------------------------

    def encode(self, query: Mapping[str, Any]) -> EncodedQuery:
        """Encode ``query["observation"]`` and ``query["candidates"]``; nothing else is read."""
        observation = query["observation"]
        state = reduce_channel(
            observation["player"],
            observation["log"],
            observation["team"]["team"],
            self.game.vocabulary,
        )
        field_features = self._field(state)
        request = self._request(state, observation["log"])
        candidates = [self._candidate(c, request) for c in query["candidates"]]
        return self._assemble(observation["kind"], state, field_features, candidates)

    # -- tokens --------------------------------------------------------------

    def _slot_index(self, side: SideState, mon: PokemonState) -> int:
        for position, letter in enumerate(sorted(side.slots)):
            index = side.slots[letter]
            if index is not None and side.roster[index] is mon:
                return position + 1
        return 0

    def _mon_cat(self, side: SideState, mon: PokemonState) -> list[int]:
        game, own = self.game, mon.own
        if mon.item is not None:
            item = game.items.lookup(mon.item)
        else:
            item = NONE if mon.item_known else UNKNOWN
        if mon.ability is not None:
            ability = game.abilities.lookup(mon.ability)
        else:
            ability = NONE if own else UNKNOWN
        shown = mon.moves_own if own and mon.moves_own is not None else mon.moves_revealed
        pad = NONE if own else UNKNOWN
        moves = [game.moves.lookup(m) for m in shown[: self.config.max_moves]]
        moves += [pad] * (self.config.max_moves - len(moves))
        if mon.gender not in GENDERS:
            raise EncodingError(f"unsupported gender {mon.gender!r}")
        return [
            game.species.lookup(mon.base_species),
            game.species.lookup(mon.species) if mon.species else UNKNOWN,
            game.species.lookup(mon.temp_species) if mon.temp_species else NONE,
            item,
            game.items.lookup(mon.mega_stone) if mon.mega_stone else NONE,
            ability,
            game.statuses.lookup(mon.status) if mon.status else NONE,
            GENDERS.index(mon.gender),
            self._slot_index(side, mon),
            *moves,
        ]

    def _mon_num(self, mon: PokemonState) -> list[float]:
        config = self.config
        fraction = mon.hp_fraction
        stats = mon.stats or {}
        hp_stat = float(mon.hp_den) if mon.own and mon.hp_den else 0.0
        volatiles = [0.0] * config.volatile_buckets
        for name in sorted(mon.volatiles):
            volatiles[bucket(name, config.volatile_buckets)] += 1.0
        known_moves = mon.moves_own if mon.moves_own is not None else []
        pp = []
        for i in range(config.max_moves):
            move = known_moves[i] if i < len(known_moves) else None
            base = self.game.move_base_pp[self.game.moves.lookup(move)] if move else 0
            pp.append(mon.pp.get(move, 0) / base if move and base else 0.0)
        return [
            1.0 if mon.own else 0.0,
            0.0 if fraction is None else 1.0,
            0.0 if fraction is None else fraction,
            1.0 if mon.fainted else 0.0,
            0.0 if mon.brought is None else 1.0,
            1.0 if mon.brought else 0.0,
            1.0 if mon.shiny else 0.0,
            1.0 if mon.mega else 0.0,
            1.0 if mon.item_known else 0.0,
            1.0 if mon.item_lost else 0.0,
            1.0 if mon.ability_suppressed else 0.0,
            1.0 if mon.can_mega_evo else 0.0,
            0.0 if mon.level is None else mon.level / 100.0,
            1.0 if mon.stats else 0.0,
            hp_stat / config.stat_scale,
            *(stats.get(name, 0) / config.stat_scale for name in STAT_NAMES),
            *(mon.boosts[name] / 6.0 for name in STAT_KEYS),
            *pp,
            *volatiles,
        ]  # fmt: skip

    def _conditions(self, conditions: Mapping[str, int] | set[str]) -> list[float]:
        out = [0.0] * self.config.condition_buckets
        counts = (
            conditions if isinstance(conditions, Mapping) else dict.fromkeys(sorted(conditions), 1)
        )
        for name in sorted(counts):
            out[bucket(name, self.config.condition_buckets)] += float(counts[name])
        return out

    def _field(self, state: BattleState) -> tuple[list[int], list[float]]:
        game = self.game
        if state.request_kind not in REQUEST_KINDS:
            raise EncodingError(f"no decision to encode at a {state.request_kind!r} request")
        cat = [
            game.weathers.lookup(state.weather) if state.weather else NONE,
            game.terrains.lookup(state.terrain) if state.terrain else NONE,
            REQUEST_KINDS.index(state.request_kind) + 1,
        ]
        num = [
            state.turn / self.config.turn_scale,
            1.0 if state.started else 0.0,
            *self._conditions(state.field_conditions),
            *self._conditions(state.own.conditions),
            *self._conditions(state.opponent.conditions),
        ]
        return cat, num

    # -- the request ---------------------------------------------------------

    def _request(self, state: BattleState, log: Sequence[str]) -> _Request:
        """The player's own last ``|request|``: its Pokémon order and its moves.

        ``rqid`` is never read.
        """
        # The reducer saw a request (_field refused a state without one), so a line exists.
        raw = [line for line in log if line.startswith(_REQUEST)][-1]
        body = json.loads(raw[len(_REQUEST) :])
        by_name = {mon.name: i for i, mon in enumerate(state.own.roster)}
        out = _Request()
        for entry in body["side"]["pokemon"]:
            # The reducer refused a request naming a Pokémon outside the own team.
            out.order.append(by_name[_SIDE.sub("", str(entry["ident"]), count=1)])
        for active in body.get("active", []):
            out.moves.append([self.game.moves.lookup_or_unknown(m["id"]) for m in active["moves"]])
        return out

    # -- candidates ----------------------------------------------------------

    def _token(self, request: _Request, position: int) -> int:
        if not 1 <= position <= len(request.order):
            raise UnsupportedCandidate("a candidate names a team position the request lacks")
        return request.order[position - 1]

    def _candidate(
        self, candidate: Mapping[str, Any], request: _Request
    ) -> tuple[list[list[int]], list[list[int]], list[int], float]:
        parsed: ParsedChoice = parse_choice(candidate["choice"])
        config = self.config
        cat = [[0] * len(self.layout.cand_cat) for _ in range(SLOTS)]
        tok = [[-1, -1] for _ in range(SLOTS)]
        team = [-1] * config.max_picks
        if parsed.kind == "team":
            if len(parsed.team) > config.max_picks:
                raise UnsupportedCandidate("a team order picks more Pokémon than max_picks")
            for place, position in enumerate(parsed.team):
                team[place] = self._token(request, position)
        else:
            for slot, action in enumerate(parsed.slots):
                if action.kind == "pass":
                    continue
                cat[slot][0] = 1 if action.kind == "move" else 2
                tok[slot][0] = self._token(request, slot + 1)
                assert action.index is not None  # a move or switch always has its index
                if action.kind == "switch":
                    tok[slot][1] = self._token(request, action.index)
                    continue
                moves = request.moves[slot] if slot < len(request.moves) else []
                if action.index > min(len(moves), config.max_moves):
                    raise UnsupportedCandidate("a candidate names a move the request lacks")
                cat[slot][1] = moves[action.index - 1]
                cat[slot][2] = action.index
                cat[slot][3] = 0 if action.target is None else 10 + action.target
                cat[slot][4] = 0 if action.event is None else 1 + EVENTS.index(action.event)
        return cat, tok, team, 1.0 if candidate["pending"] else 0.0

    # -- assembly ------------------------------------------------------------

    def _assemble(
        self,
        kind: str,
        state: BattleState,
        field_features: tuple[list[int], list[float]],
        candidates: list[tuple[list[list[int]], list[list[int]], list[int], float]],
    ) -> EncodedQuery:
        own = [(state.own, mon) for mon in state.own.roster]
        opponent = [(state.opponent, mon) for mon in state.opponent.roster]
        if len(own) > self.config.max_team or len(opponent) > self.config.max_team:
            raise EncodingError("a side has more Pokémon than max_team")
        field_cat, field_num = field_features
        rows = [*own, *opponent]
        n = len(candidates)
        return EncodedQuery(
            kind=kind,
            n_own=len(own),
            mon_cat=torch.tensor([self._mon_cat(s, m) for s, m in rows], dtype=torch.int64),
            mon_num=torch.tensor([self._mon_num(m) for _, m in rows], dtype=torch.float32),
            field_cat=torch.tensor(field_cat, dtype=torch.int64),
            field_num=torch.tensor(field_num, dtype=torch.float32),
            cand_cat=torch.tensor([c[0] for c in candidates], dtype=torch.int64).reshape(
                n, SLOTS, len(self.layout.cand_cat)
            ),
            cand_tok=torch.tensor([c[1] for c in candidates], dtype=torch.int64).reshape(
                n, SLOTS, 2
            ),
            team_tok=torch.tensor([c[2] for c in candidates], dtype=torch.int64).reshape(
                n, self.config.max_picks
            ),
            pending=torch.tensor([c[3] for c in candidates], dtype=torch.float32),
        )
