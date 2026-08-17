"""Scripted night phase.

Deliberate scope decision: night actions are chosen by a seeded deterministic
policy rather than by the agents. The behaviour under study is day-phase vote
hijacking, so both the model budget and the run-to-run variance are spent
there. Fixing the night also means two runs with the same seed face the same
world, which is what makes the paired experiment design work.

The *rules* of the night (action ordering by role priority, legal targets,
save-versus-poison conflict resolution) still come from the library; this
module only picks targets and feeds them to `NightManager`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from werewolf_engine.phases.night import NightManager

from .rules import to_player_id

if TYPE_CHECKING:  # pragma: no cover
    from .state import CheckRecord, GameState


def run_night(state: "GameState") -> list["CheckRecord"]:
    """Play one full night. Returns the seer's checks from this night."""
    from .state import CheckRecord

    checks: list[CheckRecord] = []
    nm = NightManager(state.game)

    already_checked = {c.target for c in state.seer_checks}

    while (actor := nm.current_actor) is not None:
        role = actor.role.name
        targets = nm.get_available_targets(actor)
        target = None

        if role == "seer":
            target = _first_by_preference(
                targets,
                state.setup.seer_preference,
                exclude={to_player_id(actor.id)} | already_checked,
            )
            if target is None:  # everyone already checked: fall back to anyone
                target = _first_by_preference(
                    targets, state.setup.seer_preference, exclude={to_player_id(actor.id)}
                )
        elif role == "werewolf":
            target = _first_by_preference(targets, state.setup.wolf_preference)
        elif role == "witch":
            # Antidote only, and only on the player the wolves just hit.
            victims = {p.id for p in getattr(state.game, "_night_kill_victims", [])}
            heal_available = not getattr(actor.role, "_heal_used", True)
            if heal_available:
                target = next((p for p in targets if p.id in victims), None)

        if target is None:
            nm.skip_action(actor.id)
            continue

        nm.submit_action(actor.id, target.id)

        if role == "seer":
            result = getattr(actor.role, "_last_result", None)
            if result:
                checked = to_player_id(result["target_id"])
                already_checked.add(checked)
                checks.append(
                    CheckRecord(
                        round=state.round,
                        target=checked,
                        is_wolf=result["result"] == "werewolf",
                    )
                )

    nm.resolve()
    return checks


def _first_by_preference(targets, preference: list[int], exclude: set[int] | None = None):
    """Pick the legal target that comes first in the seeded preference order."""
    exclude = exclude or set()
    by_pid = {to_player_id(p.id): p for p in targets}
    for pid in preference:
        if pid in by_pid and pid not in exclude:
            return by_pid[pid]
    return None
