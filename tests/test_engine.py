"""Engine tests: determinism, legality, and the three things the adapter adds
on top of the `werewolf-engine` library (speech phase, abstention, round cap).
"""

from __future__ import annotations

import pytest

from werewolf_harness.engine import ActionError, GameState, MAX_ROUNDS, Phase, Role, Team


def _speak_all(state: GameState, text="hello") -> None:
    while (speaker := state.current_speaker()) is not None:
        state.apply_action(speaker, {"name": "speak", "content": f"{text} from {speaker}"})


def test_same_seed_gives_the_same_world():
    a, b = GameState.new(42), GameState.new(42)
    assert a.roles == b.roles
    assert a.setup.speech_order == b.setup.speech_order
    assert a.setup.seer_preference == b.setup.seer_preference


def test_different_seeds_give_different_worlds():
    worlds = {tuple(sorted(GameState.new(s).setup.wolves())) for s in range(30)}
    assert len(worlds) > 5


def test_role_counts_match_the_standard_setup():
    state = GameState.new(7)
    counts = {r: sum(1 for x in state.roles.values() if x is r) for r in Role}
    assert counts == {Role.WOLF: 2, Role.SEER: 1, Role.WITCH: 1, Role.VILLAGER: 4}


def test_night_is_reproducible():
    outcomes = [tuple(GameState.new(9).resolve_night()) for _ in range(3)]
    assert len(set(outcomes)) == 1


def test_speaking_out_of_turn_is_rejected():
    state = GameState.new(2)
    state.resolve_night()
    speaker = state.current_speaker()
    wrong = next(p for p in state.alive_sorted() if p != speaker)
    with pytest.raises(ActionError):
        state.apply_action(wrong, {"name": "speak", "content": "me first"})


def test_speech_phase_advances_to_the_vote():
    state = GameState.new(2)
    state.resolve_night()
    _speak_all(state)
    assert state.phase is Phase.DAY_VOTE


def test_vote_legality_is_enforced():
    state = GameState.new(2)
    state.resolve_night()
    _speak_all(state)
    voter = state.alive_sorted()[0]
    with pytest.raises(ActionError):
        state.apply_action(voter, {"name": "vote", "target_id": voter})
    dead = next((p for p in range(1, 9) if not state.is_alive(p)), None)
    if dead:
        with pytest.raises(ActionError):
            state.apply_action(voter, {"name": "vote", "target_id": dead})
    with pytest.raises(ActionError):
        state.apply_action(voter, {"name": "vote", "target_id": "four"})
    # True is an int in Python; without an explicit check it would become a
    # vote for player 1.
    with pytest.raises(ActionError):
        state.apply_action(voter, {"name": "vote", "target_id": True})


def test_abstention_is_allowed_and_counted():
    """The library requires every living player to vote; the harness needs an
    abstain path for exhausted retries, so the adapter adds one."""
    state = GameState.new(4)
    state.resolve_night()
    _speak_all(state)
    alive = state.alive_sorted()
    for p in alive:
        state.apply_action(p, {"name": "vote", "target_id": None})
    assert state.resolve_vote() is None
    assert state.vote_counts[1] == {}


def test_tie_exiles_nobody():
    state = GameState.new(6)
    state.resolve_night()
    _speak_all(state)
    alive = state.alive_sorted()
    # Two votes each for two different players, everyone else abstains.
    a, b, c, d = alive[:4]
    state.apply_action(a, {"name": "vote", "target_id": c})
    state.apply_action(b, {"name": "vote", "target_id": c})
    state.apply_action(c, {"name": "vote", "target_id": a})
    state.apply_action(d, {"name": "vote", "target_id": a})
    for p in alive[4:]:
        state.apply_action(p, {"name": "vote", "target_id": None})
    assert state.resolve_vote() is None


def test_double_voting_is_rejected():
    state = GameState.new(8)
    state.resolve_night()
    _speak_all(state)
    alive = state.alive_sorted()
    state.apply_action(alive[0], {"name": "vote", "target_id": alive[1]})
    with pytest.raises(ActionError):
        state.apply_action(alive[0], {"name": "vote", "target_id": alive[2]})


def test_village_wins_when_both_wolves_are_exiled():
    state = GameState.new(13)
    wolves = state.setup.wolves()
    for wolf in wolves:
        state.resolve_night()
        if state.phase is Phase.OVER:
            break
        _speak_all(state)
        if not state.is_alive(wolf):
            continue
        for voter in state.alive_sorted():
            target = wolf if voter != wolf else next(
                p for p in state.alive_sorted() if p != wolf
            )
            state.apply_action(voter, {"name": "vote", "target_id": target})
        state.resolve_vote()
    assert state.winner is Team.VILLAGE
    assert all(not state.is_alive(w) for w in wolves)


def test_round_cap_ends_the_game_for_the_wolves():
    """A village that never finds anyone has still lost -- the cap is the
    harness's rule, not the library's."""
    state = GameState.new(21)
    while state.phase is not Phase.OVER:
        state.resolve_night()
        if state.phase is Phase.OVER:
            break
        _speak_all(state)
        for voter in state.alive_sorted():
            state.apply_action(voter, {"name": "vote", "target_id": None})
        state.resolve_vote()
    assert state.round > MAX_ROUNDS or state.winner is not None
    assert state.winner is Team.WOLF


def test_deaths_are_recorded_with_a_cause():
    state = GameState.new(5)
    for _ in range(2):
        state.resolve_night()
        if state.phase is Phase.OVER:
            break
        _speak_all(state)
        alive = state.alive_sorted()
        for voter in alive:
            state.apply_action(
                voter,
                {"name": "vote", "target_id": alive[0] if voter != alive[0] else alive[1]},
            )
        state.resolve_vote()
    assert state.deaths
    assert {d.cause for d in state.deaths} <= {"vote", "werewolf", "witch"}
    for death in state.deaths:
        assert not state.is_alive(death.player_id)
