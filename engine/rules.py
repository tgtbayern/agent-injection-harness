"""Rule surface. Deterministic by construction -- no LLM call may appear here.

The rules themselves are *not* this project's contribution, so they are not
hand-written: role behaviour, night-action priority and conflict resolution,
vote legality, tie handling and win detection all come from the MIT-licensed
`werewolf-engine` package. This module only pins the setup, maps between the
library's 0-based player ids and the 1-based ids agents see, and resolves a
seed into everything the world needs before the first night.

Things the library does not cover, and which this package adds on top:

1. a speech phase (the library only votes -- speech is where the attack lives),
2. abstention (the library requires every living player to vote; our recovery
   path needs "retries exhausted -> abstain"),
3. a round cap,
4. the sheriff: an elected office with a fractional ballot, control of speech
   order, and a badge that outlives its holder,
5. last words: a dying player's final speech, read by everyone and answerable
   by nobody.

Items 4 and 5 exist because they change *who is worth attacking*. Without
them every seat is an equally valuable injection target; with them, hijacking
the sheriff is worth more than hijacking a villager, and a corpse gets one
unrebuttable turn at the table.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from werewolf_engine import Game
from werewolf_engine.roles import get_role_class, register_role
from werewolf_engine.roles.doctor import Doctor as _LibDoctor
from werewolf_engine.roles.hunter import Hunter as _LibHunter

NUM_PLAYERS = 8   # the default variant's size; kept for callers that predate variants
MAX_ROUNDS = 6

SHERIFF_VOTE_WEIGHT = 1.5


# ---- roles the library does not ship in the shape this project needs ----

class Guard(_LibDoctor):
    """The library's Doctor under the name the 12-player setup uses.

    Same rule -- protect one living player a night, never the same one two
    nights running -- and that constraint stays the library's rather than
    being reimplemented here. Only the name differs.
    """

    name = "guard"


class Hunter(_LibHunter):
    """Revenge is played, not drawn.

    The library kills a *random* survivor when the hunter dies. That is the one
    thing this project cannot have: a terminal action with no reasoning behind
    it and no trace to read, in a harness whose whole purpose is to show why an
    agent did what it did. The hook is disabled here and the harness runs a
    real turn for the shot instead (`GameState.pending_hunter`).
    """

    name = "hunter"

    def on_player_died(self, game, dead_player, cause) -> None:  # noqa: D102
        return


register_role("guard", Guard)
register_role("hunter", Hunter)


class Role(str, Enum):
    WOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    GUARD = "guard"
    VILLAGER = "villager"


class Team(str, Enum):
    WOLF = "werewolf"
    VILLAGE = "village"


class Phase(str, Enum):
    """Orchestration phases. Finer-grained than the library's LOBBY/NIGHT/DAY/END
    because the day is split into speech and vote, and because three of these
    are interrupts rather than steps in a cycle."""

    NIGHT = "night"
    SHERIFF_CAMPAIGN = "sheriff_campaign"
    SHERIFF_VOTE = "sheriff_vote"
    DAY_SPEECH = "day_speech"
    DAY_VOTE = "day_vote"
    LAST_WORDS = "last_words"
    HUNTER_SHOOT = "hunter_shoot"
    OVER = "over"


# ---- variants ---------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    """A table size and its rule switches.

    Kept as data so a configuration change never needs a code change, and so
    the 8-player games every existing result was produced on stay bit-identical
    to what they were: `8p` is the old constants, moved.
    """

    name: str
    num_players: int
    role_counts: dict[str, int]
    sheriff: bool = False
    last_words: bool = False


VARIANTS: dict[str, Variant] = {
    # The original setup. Game richness was explicitly not a selling point.
    "8p": Variant(
        "8p", 8,
        {"werewolf": 2, "seer": 1, "witch": 1, "villager": 4},
    ),
    # The standard Chinese 12-player table: 4 wolves, 4 gods, 4 villagers.
    "12p": Variant(
        "12p", 12,
        {"werewolf": 4, "seer": 1, "witch": 1, "hunter": 1, "guard": 1, "villager": 4},
        sheriff=True,
        last_words=True,
    ),
    # 8 players with the offices switched on, so the sheriff's effect can be
    # read without the table size moving at the same time.
    "8p+sheriff": Variant(
        "8p+sheriff", 8,
        {"werewolf": 2, "seer": 1, "witch": 1, "villager": 4},
        sheriff=True,
        last_words=True,
    ),
}
DEFAULT_VARIANT = "8p"


def get_variant(variant: "str | Variant | None") -> Variant:
    if variant is None:
        return VARIANTS[DEFAULT_VARIANT]
    if isinstance(variant, Variant):
        return variant
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}; known: {sorted(VARIANTS)}"
        )
    return VARIANTS[variant]


# ---- id mapping ------------------------------------------------------
# Agents see players 1..N; the library numbers them 0..N-1.

def to_engine_id(player_id: int) -> int:
    return player_id - 1


def to_player_id(engine_id: int) -> int:
    return engine_id + 1


@dataclass(frozen=True)
class Setup:
    """Everything the seed decides, resolved once before the first night.

    Note what is *not* here: night targets. Who the seer checks, who the wolves
    kill and what the witch does with her potions are decisions the agents make
    in their own turns -- the seed fixes the world, not the play. The sheriff is
    not here either: that office is won at the table, not dealt.
    """

    roles: dict[int, Role]
    speech_order: list[int]
    fallback_order: list[int]
    variant: Variant = field(default_factory=lambda: VARIANTS[DEFAULT_VARIANT])

    @property
    def num_players(self) -> int:
        return self.variant.num_players

    def wolves(self) -> list[int]:
        return sorted(p for p, r in self.roles.items() if r is Role.WOLF)

    def seats(self) -> list[int]:
        return list(range(1, self.num_players + 1))


def build_game(seed: int, variant: "str | Variant | None" = None) -> tuple[Game, Setup]:
    """Create a seeded library game plus the resolved setup.

    Role assignment is done here rather than through the library's
    `assign_roles`, which shuffles the *global* RNG: doing it with a local
    `random.Random(seed)` keeps games reproducible and safe to run
    concurrently, while the role classes themselves still come from the
    library's registry.
    """
    v = get_variant(variant)
    rng = random.Random(seed)

    game = Game([str(to_player_id(i)) for i in range(v.num_players)])
    names: list[str] = []
    for role_name, count in v.role_counts.items():
        names.extend([role_name] * count)
    if len(names) != v.num_players:
        raise ValueError(
            f"variant {v.name}: role counts total {len(names)}, "
            f"expected {v.num_players}"
        )
    rng.shuffle(names)
    for player, role_name in zip(game.players, names):
        player.assign_role(get_role_class(role_name)())

    ids = [to_player_id(i) for i in range(v.num_players)]

    # Speech order is FIXED at 1..N while roles are randomised, rather than the
    # other way round. Two reasons, and the second is the important one:
    #
    #   * a transcript read back in seat order is legible; a shuffled order
    #     makes every review of a game an exercise in re-sorting.
    #   * it decouples the conformity axis. With position fixed and role
    #     random, "speaking fifth" and "being the seer" vary independently, so
    #     an order effect cannot be a role effect wearing a disguise.
    #
    # A sheriff perturbs this on purpose -- see `GameState.speech_order_this_round`
    # -- which is why sheriff games are a separate arm rather than a new default.
    speech_order = ids[:]

    # Used only when an agent's night turn cannot be salvaged (see recovery):
    # a seeded, reproducible target so a failed call never degenerates into
    # "the wolves did not kill anyone tonight".
    fallback_order = ids[:]
    rng.shuffle(fallback_order)

    setup = Setup(
        roles={to_player_id(p.id): Role(p.role.name) for p in game.players},
        speech_order=speech_order,
        fallback_order=fallback_order,
        variant=v,
    )
    return game, setup


def team_of(role: Role) -> Team:
    return Team.WOLF if role is Role.WOLF else Team.VILLAGE
