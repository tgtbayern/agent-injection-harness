"""The night, as a sequence of agent turns.

The night used to be scripted. It is not any more: the seer picks who to check,
the wolves pick who to kill and the witch decides what to do with her potions,
each in a real ReAct turn with its own reasoning trace. Control over the
experiment comes from every configuration running the identical seed set and
the identical procedure -- not from freezing what the agents do inside it.

This module owns the *turn order and legality* of the night; it never chooses a
target. Ordering by role priority, the save-versus-poison conflict and the
resolution of who actually dies still come from the `werewolf-engine` library.

Two rules the harness adds on top, both to remove genuine ambiguity:

* **The pack decides together.** Each living wolf names a target; the pack's
  choice is the majority, and a tie goes to the wolf who acted first. Only that
  single target is submitted to the library, so the last wolf to speak cannot
  silently overwrite the first one.
* **Saving and poisoning are separate actions.** The library infers which one
  the witch meant from whether the target is tonight's victim, which makes
  "poison the person who was already attacked" unrepresentable. The harness
  splits them into two named actions and rejects that case explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from werewolf_engine.phases.night import NightManager

from .rules import Role, to_engine_id, to_player_id

if TYPE_CHECKING:  # pragma: no cover
    from .state import GameState

# Terminal night actions, by the role allowed to take them.
NIGHT_ACTIONS: dict[str, set[str]] = {
    "seer": {"night_check", "night_skip"},
    "werewolf": {"night_kill", "night_skip"},
    "witch": {"night_save", "night_poison", "night_skip"},
    "guard": {"night_protect", "night_skip"},
    "villager": set(),
    # The hunter has no night turn. Its one action is a reaction to its own
    # death, and it happens in daylight (Phase.HUNTER_SHOOT).
    "hunter": set(),
}


class NightPhase:
    """Turn cursor over the library's night actors."""

    def __init__(self, state: "GameState"):
        self.state = state
        self.manager = NightManager(state.game)
        self.wolf_intents: dict[int, int] = {}   # wolf -> its chosen target
        self.wolf_order: list[int] = []          # the order they acted in
        self.records: list[dict] = []            # everything for the log

    # ---- who acts now --------------------------------------------------

    def current_actor(self) -> int | None:
        actor = self.manager.current_actor
        return to_player_id(actor.id) if actor is not None else None

    def actor_role(self) -> Role | None:
        actor = self.manager.current_actor
        return Role(actor.role.name) if actor is not None else None

    def allowed_actions(self) -> set[str]:
        """What this actor may actually call, right now.

        State-aware, not just role-aware: offering `night_save` when the
        antidote is already spent is the referee advertising an illegal move,
        and an agent that takes it up burns its whole turn being refused.
        """
        role = self.actor_role()
        if role is None:
            return set()
        actions = set(NIGHT_ACTIONS.get(role.value, set()))
        if role is Role.WITCH:
            actor = self.manager.current_actor
            if getattr(actor.role, "_heal_used", True) or not self.victims_tonight():
                actions.discard("night_save")
            if getattr(actor.role, "_kill_used", True):
                actions.discard("night_poison")
        if role is Role.SEER and not self.legal_targets(to_player_id(
                self.manager.current_actor.id)):
            actions.discard("night_check")
        if role is Role.GUARD and not self.legal_targets(to_player_id(
                self.manager.current_actor.id)):
            actions.discard("night_protect")
        return actions

    def legal_targets(self, player_id: int) -> list[int]:
        """Living players this actor may name, from the library's own rules."""
        actor = self.manager.current_actor
        if actor is None or to_player_id(actor.id) != player_id:
            return []
        targets = {to_player_id(p.id) for p in self.manager.get_available_targets(actor)}
        if self.actor_role() is Role.SEER:
            # Checking yourself is legal in the library and useless in the game.
            targets.discard(player_id)
            targets -= {c.target for c in self.state.seer_checks}
        return sorted(targets)

    def victims_tonight(self) -> list[int]:
        """Who the wolves have hit so far tonight. Private to the witch."""
        return sorted(
            to_player_id(p.id)
            for p in getattr(self.state.game, "_night_kill_victims", [])
        )

    def remaining_wolves(self) -> int:
        return sum(
            1 for p in self.manager._actors[self.manager._current_index :]
            if p.role and p.role.name == "werewolf"
        )

    # ---- taking a turn --------------------------------------------------

    def submit(self, player_id: int, name: str, target_id: int | None) -> dict:
        """Apply one night action. Raises ValueError with an actionable message."""
        actor = self.manager.current_actor
        if actor is None:
            raise ValueError("the night is already over")
        if to_player_id(actor.id) != player_id:
            raise ValueError(
                f"it is player {to_player_id(actor.id)}'s night turn, not player {player_id}"
            )
        if name not in self.allowed_actions():
            allowed = ", ".join(sorted(self.allowed_actions())) or "nothing"
            raise ValueError(f"{name} is not available to you tonight; you may: {allowed}")

        role = self.actor_role()
        if name == "night_skip":
            self.manager.skip_action(actor.id)
            return self._record(player_id, role, name, None, "skipped")

        if not isinstance(target_id, int) or isinstance(target_id, bool):
            raise ValueError(f"{name} needs an integer player id")

        if role is Role.WOLF:
            return self._wolf(player_id, target_id)
        if role is Role.WITCH:
            return self._witch(player_id, name, target_id)
        if role is Role.GUARD:
            return self._guard(player_id, target_id)
        return self._seer(player_id, target_id)

    # ---- per role -------------------------------------------------------

    def _seer(self, player_id: int, target_id: int) -> dict:
        legal = self.legal_targets(player_id)
        if target_id not in legal:
            raise ValueError(
                f"player {target_id} is not checkable tonight; you may check {legal}"
            )
        actor = self.manager.current_actor
        self.manager.submit_action(actor.id, to_engine_id(target_id))
        result = getattr(actor.role, "_last_result", None) or {}
        is_wolf = result.get("result") == "werewolf"
        return self._record(player_id, Role.SEER, "night_check", target_id,
                            "werewolf" if is_wolf else "not a werewolf",
                            extra={"is_wolf": is_wolf})

    def _guard(self, player_id: int, target_id: int) -> dict:
        """Protect one living player. The no-repeat rule is the library's.

        `legal_targets` already drops last night's charge, because the library's
        `get_available_targets` does -- so the refusal here is a backstop for a
        target that was never offered, not a second implementation of the rule.
        """
        legal = self.legal_targets(player_id)
        if target_id not in legal:
            raise ValueError(
                f"player {target_id} cannot be protected tonight "
                f"(you may protect {legal}; never the same player twice running)"
            )
        actor = self.manager.current_actor
        self.manager.submit_action(actor.id, to_engine_id(target_id))
        return self._record(player_id, Role.GUARD, "night_protect", target_id,
                            "protected")

    def _wolf(self, player_id: int, target_id: int) -> dict:
        legal = self.legal_targets(player_id)
        if target_id not in legal:
            raise ValueError(
                f"player {target_id} is not a legal kill; the pack may take {legal}"
            )
        self.wolf_intents[player_id] = target_id
        self.wolf_order.append(player_id)
        actor = self.manager.current_actor

        if self.remaining_wolves() > 1:
            # Hold the vote open: the pack's choice is submitted once, by the
            # last wolf to act, so nobody's pick is silently overwritten.
            self.manager.skip_action(actor.id)
            return self._record(player_id, Role.WOLF, "night_kill", target_id,
                                "noted; the pack decides together")

        chosen = self._pack_choice()
        self.manager.submit_action(actor.id, to_engine_id(chosen))
        return self._record(
            player_id, Role.WOLF, "night_kill", target_id,
            f"the pack attacks player {chosen}",
            extra={"pack_choice": chosen, "intents": dict(self.wolf_intents)},
        )

    def _pack_choice(self) -> int:
        """Majority; a tie goes to the wolf who acted first."""
        counts: dict[int, int] = {}
        for target in self.wolf_intents.values():
            counts[target] = counts.get(target, 0) + 1
        best = max(counts.values())
        leaders = [t for t, c in counts.items() if c == best]
        if len(leaders) == 1:
            return leaders[0]
        for wolf in self.wolf_order:
            if self.wolf_intents[wolf] in leaders:
                return self.wolf_intents[wolf]
        return sorted(leaders)[0]  # pragma: no cover -- unreachable

    def _witch(self, player_id: int, name: str, target_id: int) -> dict:
        actor = self.manager.current_actor
        victims = self.victims_tonight()
        heal_left = not getattr(actor.role, "_heal_used", True)
        poison_left = not getattr(actor.role, "_kill_used", True)

        if name == "night_save":
            if not heal_left:
                raise ValueError("your antidote is already used")
            if target_id not in victims:
                raise ValueError(
                    f"the antidote only works on tonight's victim; that is {victims or 'nobody'}"
                )
        else:  # night_poison
            if not poison_left:
                raise ValueError("your poison is already used")
            if target_id in victims:
                raise ValueError(
                    f"player {target_id} is already dying tonight; poisoning them "
                    "wastes the potion"
                )
            if target_id == player_id:
                raise ValueError("you cannot poison yourself")
            if target_id not in self.state.alive:
                raise ValueError(f"player {target_id} is not alive")

        self.manager.submit_action(actor.id, to_engine_id(target_id))
        return self._record(player_id, Role.WITCH, name, target_id,
                            "antidote used" if name == "night_save" else "poison used")

    # ---- bookkeeping ----------------------------------------------------

    def _record(self, player_id, role, name, target, outcome, extra=None) -> dict:
        record = {
            "round": self.state.round,
            "player_id": player_id,
            "role": role.value if role else None,
            "action": name,
            "target": target,
            "outcome": outcome,
            **(extra or {}),
        }
        self.records.append(record)
        return record

    def finished(self) -> bool:
        return self.manager.current_actor is None

    def resolve(self) -> None:
        """Apply the night: the library decides who actually dies."""
        self.manager.resolve()
