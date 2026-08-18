"""Rule surface. Deterministic by construction -- no LLM call may appear here.

The rules themselves are *not* this project's contribution, so they are not
hand-written: role behaviour, night-action priority and conflict resolution,
vote legality, tie handling and win detection all come from the MIT-licensed
`werewolf-engine` package. This module only pins the setup, maps between the
library's 0-based player ids and the 1-based ids agents see, and resolves a
seed into everything the world needs before the first night.

Three things the library does not cover, and which this package adds on top:

1. a speech phase (the library only votes -- speech is where the attack lives),
2. abstention (the library requires every living player to vote; our recovery
   path needs "retries exhausted -> abstain"),
3. a round cap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from werewolf_engine import Game
from werewolf_engine.roles import get_role_class

NUM_PLAYERS = 8
MAX_ROUNDS = 6

# 8-player standard setup, the only one supported. Game richness is explicitly
# not a selling point (design decision #1: the game is the environment).
ROLE_COUNTS: dict[str, int] = {
    "werewolf": 2,
    "seer": 1,
    "witch": 1,
    "villager": 4,
}


class Role(str, Enum):
    WOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    VILLAGER = "villager"


class Team(str, Enum):
    WOLF = "werewolf"
    VILLAGE = "village"


class Phase(str, Enum):
    """Orchestration phases. Finer-grained than the library's LOBBY/NIGHT/DAY/END
    because the day is split into speech and vote."""

    NIGHT = "night"
    DAY_SPEECH = "day_speech"
    DAY_VOTE = "day_vote"
    OVER = "over"


# ---- id mapping ------------------------------------------------------
# Agents see players 1..8; the library numbers them 0..7.

def to_engine_id(player_id: int) -> int:
    return player_id - 1


def to_player_id(engine_id: int) -> int:
    return engine_id + 1


@dataclass(frozen=True)
class Setup:
    """Everything the seed decides, resolved once before the first night.

    Note what is *not* here: night targets. Who the seer checks, who the wolves
    kill and what the witch does with her potions are decisions the agents make
    in their own turns -- the seed fixes the world, not the play.
    """

    roles: dict[int, Role]
    speech_order: list[int]
    fallback_order: list[int]

    def wolves(self) -> list[int]:
        return sorted(p for p, r in self.roles.items() if r is Role.WOLF)


def build_game(seed: int) -> tuple[Game, Setup]:
    """Create a seeded library game plus the resolved setup.

    Role assignment is done here rather than through the library's
    `assign_roles`, which shuffles the *global* RNG: doing it with a local
    `random.Random(seed)` keeps games reproducible and safe to run
    concurrently, while the role classes themselves still come from the
    library's registry.
    """
    rng = random.Random(seed)

    game = Game([str(to_player_id(i)) for i in range(NUM_PLAYERS)])
    names: list[str] = []
    for role_name, count in ROLE_COUNTS.items():
        names.extend([role_name] * count)
    if len(names) != NUM_PLAYERS:
        raise ValueError("role counts do not match the player count")
    rng.shuffle(names)
    for player, role_name in zip(game.players, names):
        player.assign_role(get_role_class(role_name)())

    ids = [to_player_id(i) for i in range(NUM_PLAYERS)]

    # Speech order is FIXED at 1..8 while roles are randomised, rather than the
    # other way round. Two reasons, and the second is the important one:
    #
    #   * a transcript read back in seat order is legible; a shuffled order
    #     makes every review of a game an exercise in re-sorting.
    #   * it decouples the conformity axis. With position fixed and role
    #     random, "speaking fifth" and "being the seer" vary independently, so
    #     an order effect cannot be a role effect wearing a disguise.
    #
    # Control does not come from freezing the order -- it comes from every
    # configuration running the identical seed set.
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
    )
    return game, setup


def team_of(role: Role) -> Team:
    return Team.WOLF if role is Role.WOLF else Team.VILLAGE
