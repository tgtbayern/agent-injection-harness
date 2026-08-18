"""Human seat at the table.

The point of this is not that it is fun. It is that "34% of votes were
hijacked" is an uninterpretable number until something anchors it, and the only
available anchor is a person playing the same seat under the same rules with
the same information.

Three things about the baseline, all of which belong in the write-up rather
than being quietly hoped away:

1. A human has no `belief_state`, so the CLI makes one: a suspicion form each
   round, 0-5 per living player. It is coarser than what an agent maintains,
   and the comparison should be read with that in mind.
2. Sample size is tiny. Ten to fifteen games is an order-of-magnitude
   comparison, not a significance test, and it must be reported as such.
3. A person who knows they are being tested reads more carefully than one who
   does not. There is no fix; it goes in the limitations section.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..engine import GameState, get_visible_state
from ..harness.agent.loop import TurnResult
from ..evalkit.runner import RunConfig, run_game


class HumanPlayer:
    """Implements the same `run_turn` contract the agent loop does, so the
    runner cannot tell the difference and neither can the other players."""

    def __init__(self, player_id: int, input_fn=input, output_fn=print):
        self.player_id = player_id
        self.input = input_fn
        self.output = output_fn
        self._last_belief_round = 0

    def run_turn(self, state: GameState, player_id: int, belief, task: str) -> TurnResult:
        view = get_visible_state(state, player_id)
        result = TurnResult(
            player_id=player_id,
            round=state.round,
            task=task,
            is_human=True,
            belief_before=belief.snapshot(),
        )
        self._show(view, state)
        if state.round != self._last_belief_round:
            self._collect_beliefs(view, belief, state)
            self._last_belief_round = state.round

        if task == "night":
            self._night_turn(state, player_id, result)
        elif task == "speak":
            content = self._ask_text("Your speech (one line): ") or "I pass this round."
            applied = state.apply_action(player_id, {"name": "speak", "content": content})
            result.speech = content
            result.speech_order = applied.get("speech_order")
        else:
            alive = [p for p in state.alive_sorted() if p != player_id]
            target = self._ask_int(f"Vote for which player {alive}, or 0 to abstain: ",
                                   allowed=set(alive) | {0})
            target = None if target == 0 else target
            state.apply_action(player_id, {"name": "vote", "target_id": target})
            result.vote = target

        result.belief_after = belief.snapshot()
        result.react_trace = [
            {"step": 1, "thought": "(human)",
             "action": (result.night_action or {}).get("action", task), "args": {},
             "observation": "", "tokens": 0, "latency_ms": 0,
             "guard_blocked": False, "block_reason": None, "guard_detections": [],
             "injected": False}
        ]
        result.steps_used = 1
        return result

    def _night_turn(self, state: GameState, player_id: int, result: TurnResult) -> None:
        """A human's night turn.

        The same choice the agents get, offered the same way: the engine says
        what is legal, the person picks. A seat that has nothing to do tonight
        is told so and skipped rather than asked a meaningless question.
        """
        options = state.night_options(player_id)
        actions = [a for a in options["actions"] if a != "night_skip"]
        targets = options["targets"]

        if options.get("victims_tonight") is not None:
            self.output(f"  the wolves attacked: {options['victims_tonight'] or 'nobody'}")
        if options.get("pack_votes_so_far"):
            self.output(f"  your pack has named so far: {options['pack_votes_so_far']}")

        if not actions:
            state.apply_action(player_id, {"name": "night_skip", "target_id": None})
            result.night_action = {"action": "night_skip", "target": None}
            self.output("  nothing for you to do tonight")
            return

        chosen = actions[0]
        if len(actions) > 1:
            self.output(f"  tonight you may: {', '.join(actions)} (or skip)")
            for index, action in enumerate(actions, start=1):
                self.output(f"    {index}. {action}")
            pick = self._ask_int(
                f"  choose 1-{len(actions)}, or 0 to skip: ",
                allowed=set(range(0, len(actions) + 1)),
            )
            if pick == 0:
                state.apply_action(player_id, {"name": "night_skip", "target_id": None})
                result.night_action = {"action": "night_skip", "target": None}
                return
            chosen = actions[pick - 1]

        target = self._ask_int(
            f"  {chosen} on which player {targets}, or 0 to skip: ",
            allowed=set(targets) | {0},
        )
        if target == 0:
            state.apply_action(player_id, {"name": "night_skip", "target_id": None})
            result.night_action = {"action": "night_skip", "target": None}
            return
        applied = state.apply_action(player_id, {"name": chosen, "target_id": target})
        result.night_action = {"action": chosen, "target": target,
                               "outcome": applied.get("outcome")}
        self.output(f"  -> {applied.get('outcome')}")

    # ---- presentation ---------------------------------------------------

    def _show(self, view: dict, state: GameState) -> None:
        you, public, private = view["you"], view["public"], view["private"]
        self.output("\n" + "=" * 68)
        self.output(f"ROUND {public['round']} -- you are player {you['player_id']} "
                    f"({you['role']}), alive: {public['alive']}")
        if private.get("fellow_wolves"):
            self.output(f"  your fellow wolves: {private['fellow_wolves']}")
        for check in private.get("checks", []):
            self.output(f"  night {check['round']}: player {check['target']} is "
                        f"{'A WEREWOLF' if check['is_wolf'] else 'not a werewolf'}")
        if private.get("antidote_available") is not None:
            self.output(f"  antidote: {private['antidote_available']}, "
                        f"poison: {private['poison_available']}")
        if public["dead"]:
            self.output("  dead: " + ", ".join(
                f"p{d['player_id']} (r{d['round']}, "
                f"{'exiled' if d['cause'] == 'exiled' else 'night'})" for d in public["dead"]
            ))
        self.output("-" * 68)
        for speech in public["speeches"]:
            if speech["round"] != public["round"]:
                continue
            who = "you" if speech["player_id"] == you["player_id"] else f"p{speech['player_id']}"
            self.output(f"  [{who}] {speech['content']}")
        self.output("-" * 68)

    def _collect_beliefs(self, view: dict, belief, state: GameState) -> None:
        """A human's stand-in for a belief state: 0 = surely village, 5 = surely wolf."""
        self.output("Rate how likely each living player is a wolf (0-5, blank keeps "
                    "the current value):")
        for pid in view["public"]["alive"]:
            if pid == self.player_id:
                continue
            current = belief.get(pid)
            raw = self._ask_text(f"  player {pid} [{current.suspicion:.1f}]: ")
            if raw.strip() == "":
                continue
            try:
                score = max(0, min(5, int(raw.strip())))
            except ValueError:
                continue
            belief.update(pid, score / 5, "human rating", state.round)

    # ---- input ----------------------------------------------------------

    def _ask_text(self, prompt: str) -> str:
        try:
            return self.input(prompt)
        except EOFError:
            return ""

    def _ask_int(self, prompt: str, allowed: set[int], default: int = 0,
                 attempts: int = 5) -> int:
        """Bounded, so a closed stdin or a scripted run cannot spin forever.

        After `attempts` unusable answers the turn takes the same conservative
        default an exhausted agent takes -- abstention -- rather than blocking
        the game on a person who has walked away.
        """
        for _ in range(attempts):
            raw = self._ask_text(prompt).strip()
            if raw.isdigit() and int(raw) in allowed:
                return int(raw)
            if raw == "":
                break
            self.output(f"  pick one of {sorted(allowed)}")
        self.output(f"  no usable answer; defaulting to {default}")
        return default


def play(seed: int = 1, seat: int = 1, guard: tuple[str, ...] = ("L1", "L2"),
         attack: bool = True, out: str | None = None) -> int:
    """Run one game with a human in `seat`. Other players are not told."""
    cfg = RunConfig(
        seed=seed,
        guard_layers=tuple(guard),
        attack_enabled=attack,
        human_players=(seat,),
    )
    human = HumanPlayer(seat)
    log = run_game(cfg, human_ui=human)

    print("\n" + "=" * 68)
    print(f"winner: {log['outcome']['winner']}")
    print(f"roles:  {log['ground_truth']['roles']}")
    hijacked = [
        t for r in log["rounds"] for a in r["agents"]
        if a["player_id"] == seat and a["is_human"]
        for t in a.get("read_payloads", [])
    ]
    print(f"payloads you were exposed to: {len(hijacked)}")
    path = Path(out or f"human_game_{log['game_id']}.json")
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), "utf-8")
    print(f"log written to {path}")
    return 0
