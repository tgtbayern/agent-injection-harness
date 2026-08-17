"""Deterministic judge. Contains no LLM call, by rule.

Public surface (design decision #2 -- the referee is code, not a model):

    state = GameState.new(seed)
    view  = get_visible_state(state, player_id)
    state.apply_action(player_id, action)
"""

from .rules import (
    MAX_ROUNDS,
    NUM_PLAYERS,
    Phase,
    Role,
    Setup,
    Team,
    build_game,
    team_of,
    to_engine_id,
    to_player_id,
)
from .state import ActionError, CheckRecord, DeathRecord, GameState, Speech
from .visibility import assert_no_leak, get_visible_state

__all__ = [
    "ActionError",
    "CheckRecord",
    "DeathRecord",
    "GameState",
    "MAX_ROUNDS",
    "NUM_PLAYERS",
    "Phase",
    "Role",
    "Setup",
    "Speech",
    "Team",
    "assert_no_leak",
    "build_game",
    "get_visible_state",
    "team_of",
    "to_engine_id",
    "to_player_id",
]
