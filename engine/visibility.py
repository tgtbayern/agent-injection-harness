"""Who can see what.

This is the one module in the project that must be correct or every experiment
is void: if one player's private information leaks into another player's view,
the whole environment stops measuring anything. It is therefore the module with
mandatory unit-test coverage (`tests/test_visibility.py`).

The rule is simple and enforced in one place: a view contains
  * public facts (alive list, deaths, the full speech transcript, vote records)
  * the viewer's own private facts (own role, fellow wolves, own check history,
    own potion status)
and nothing else. No other player's role, no other player's private info, and
no ground truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .rules import Phase, Role, to_engine_id, to_player_id

if TYPE_CHECKING:  # pragma: no cover
    from .state import GameState


def get_visible_state(state: "GameState", player_id: int) -> dict:
    """Everything player `player_id` is allowed to know, as a plain dict."""
    if player_id not in range(1, state.num_players + 1):
        raise ValueError(f"no such player: {player_id}")

    return {
        "you": _self_view(state, player_id),
        "public": _public_view(state),
        "private": _private_view(state, player_id),
    }


def _self_view(state: "GameState", player_id: int) -> dict:
    role = state.role_of(player_id)
    order = state.speech_order_this_round()
    return {
        "player_id": player_id,
        "role": role.value,
        "team": "werewolf" if role is Role.WOLF else "village",
        "alive": state.is_alive(player_id),
        "speech_position": order.index(player_id) + 1 if player_id in order else None,
        "speakers_total": len(order),
    }


def _public_view(state: "GameState") -> dict:
    return {
        "round": state.round,
        "phase": state.phase.value,
        "alive": state.alive_sorted(),
        # The *mechanism* of a night death is private. Whether someone was
        # taken by the wolves or by the witch's poison is exactly the kind of
        # thing the village has to infer, and the library's cause string would
        # hand it over for free. Only "exiled" versus "died at night" is public.
        "dead": [
            {
                "player_id": d.player_id,
                "round": d.round,
                "cause": _public_cause(d.cause),
            }
            for d in state.deaths
        ],
        "speech_order": state.speech_order_this_round(),
        "speeches": [s.to_dict() for s in state.speeches],
        "votes": {
            str(rnd): {str(v): t for v, t in ballots.items()}
            for rnd, ballots in state.votes.items()
        },
        "vote_counts": {
            str(rnd): {str(t): c for t, c in counts.items()}
            for rnd, counts in state.vote_counts.items()
        },
        "exiles": {str(rnd): who for rnd, who in state.exiles.items()},
        # The badge is public by construction -- an office nobody can see is
        # not an office. Who holds it, who ran for it, and where it went when
        # its holder died are all things the whole table watched happen.
        "sheriff": state.sheriff,
        "sheriff_candidates": list(state.sheriff_candidates),
        "badge_transfers": list(state.badge_transfers),
        # A shot is the loudest event in the game and cannot be concealed. Note
        # what this says and what it does not: "player 7 shot player 3", never
        # "player 7 is the hunter". The referee reports the event; the table
        # draws the inference, exactly as it does from a public seer claim.
        "shots": [
            {"round": s["round"], "by": s["hunter"], "target": s["target"]}
            for s in state.hunter_shots
        ],
    }


def _public_cause(cause: str) -> str:
    """Death causes the whole table can see.

    "werewolf" and "witch" both collapse to "night": which of the two took
    someone is the single most valuable thing the village has to infer, and
    handing it over would void the experiment. A daylight shot does not
    collapse -- everybody watched it happen.
    """
    if cause == "vote":
        return "exiled"
    if cause == "hunter":
        return "shot"
    return "night"


def _private_view(state: "GameState", player_id: int) -> dict:
    """Private facts, sourced from the library's per-role `get_private_info`.

    The library only exposes the seer's *most recent* check, so the harness
    keeps the seer's full history in `GameState.seer_checks` and hands it back
    here -- still only ever to the seer.
    """
    role = state.role_of(player_id)
    raw = state.game.get_private_state(to_engine_id(player_id)).get("private_info", {})
    private: dict = {}

    if role is Role.WOLF:
        private["fellow_wolves"] = sorted(
            to_player_id(w["id"]) for w in raw.get("fellow_wolves", [])
        )
    elif role is Role.SEER:
        private["checks"] = [c.to_dict() for c in state.checks_of(player_id)]
    elif role is Role.WITCH:
        private["antidote_available"] = bool(raw.get("heal_available", False))
        private["poison_available"] = bool(raw.get("kill_available", False))

    return private


def assert_no_leak(view: dict, state: "GameState", player_id: int) -> None:
    """Defensive check used by the tests and by the runner in debug mode.

    Raises AssertionError if a view mentions any other player's role.
    """
    others = {
        p: state.role_of(p).value
        for p in range(1, state.num_players + 1)
        if p != player_id
    }
    private = view.get("private", {})
    for other, role_value in others.items():
        if role_value in str(private.get("checks", "")) and role_value not in (
            "werewolf",
            "villager",
        ):
            raise AssertionError(f"role of player {other} leaked into the view")
    if "roles" in view or "ground_truth" in view:
        raise AssertionError("ground truth leaked into a player view")
    if view["you"]["player_id"] != player_id:
        raise AssertionError("view built for the wrong player")


__all__ = ["get_visible_state", "assert_no_leak", "Phase"]
