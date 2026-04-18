"""
Game tree representation for poker CFR solving.

Models the extensive-form game tree with:
- Chance nodes (dealing cards)
- Player action nodes (fold/check/call/bet/raise)
- Terminal nodes (showdown or fold)
- Information sets (what a player actually knows)
"""

from enum import Enum, auto
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from .cards import Card


class NodeType(Enum):
    CHANCE = auto()
    PLAYER = auto()
    TERMINAL = auto()


class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "allin"


@dataclass
class Action:
    action_type: ActionType
    amount: float = 0.0  # bet/raise size (absolute chips)

    def __repr__(self):
        if self.action_type in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL):
            return self.action_type.value
        elif self.action_type == ActionType.ALL_IN:
            return f"allin({self.amount:.0f})"
        else:
            return f"{self.action_type.value}({self.amount:.0f})"

    def __hash__(self):
        return hash((self.action_type, round(self.amount, 2)))

    def __eq__(self, other):
        return (self.action_type == other.action_type and
                abs(self.amount - other.amount) < 0.01)

    def to_key(self) -> str:
        if self.action_type in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL):
            return self.action_type.value
        return f"{self.action_type.value}_{self.amount:.0f}"


@dataclass
class GameState:
    """Complete game state at any point in the hand."""
    street: int = 0  # 0=preflop, 1=flop, 2=turn, 3=river
    board: List[Card] = field(default_factory=list)
    pot: float = 0.0
    stacks: List[float] = field(default_factory=lambda: [100.0, 100.0])
    bets: List[float] = field(default_factory=lambda: [0.0, 0.0])
    acting_player: int = 0  # 0=OOP, 1=IP
    history: List[Action] = field(default_factory=list)
    street_history: List[Action] = field(default_factory=list)
    is_allin: bool = False
    n_actions_this_street: int = 0
    last_aggressor: int = -1

    def effective_stack(self) -> float:
        return min(self.stacks)

    def to_bet(self, player: int) -> float:
        """Amount already bet by player this street."""
        return self.bets[player]

    def amount_to_call(self) -> float:
        return abs(self.bets[0] - self.bets[1])

    def total_pot(self) -> float:
        return self.pot + self.bets[0] + self.bets[1]

    def copy(self) -> 'GameState':
        return GameState(
            street=self.street,
            board=list(self.board),
            pot=self.pot,
            stacks=list(self.stacks),
            bets=list(self.bets),
            acting_player=self.acting_player,
            history=list(self.history),
            street_history=list(self.street_history),
            is_allin=self.is_allin,
            n_actions_this_street=self.n_actions_this_street,
            last_aggressor=self.last_aggressor,
        )


class GameNode:
    """A node in the extensive-form game tree."""

    def __init__(self, node_type: NodeType, state: GameState):
        self.node_type = node_type
        self.state = state
        self.children: Dict[str, 'GameNode'] = {}  # action_key -> child
        self.parent: Optional['GameNode'] = None
        # For terminal nodes
        self.payoff_multiplier: float = 0.0  # +1 for winner, -1 for loser

    @property
    def is_terminal(self) -> bool:
        return self.node_type == NodeType.TERMINAL

    @property
    def is_chance(self) -> bool:
        return self.node_type == NodeType.CHANCE

    @property
    def is_player(self) -> bool:
        return self.node_type == NodeType.PLAYER


class GameTreeBuilder:
    """
    Build the game tree with betting abstraction.

    Supports configurable bet sizes as fractions of pot.
    """

    def __init__(self,
                 starting_pot: float,
                 starting_stacks: List[float],
                 board: List[Card],
                 bet_sizes: Dict[int, List[float]] = None,
                 raise_sizes: Dict[int, List[float]] = None,
                 max_raises_per_street: int = 3):
        """
        Args:
            starting_pot: Initial pot size
            starting_stacks: [OOP_stack, IP_stack]
            board: Community cards (determines which street we're on)
            bet_sizes: {street: [pot_fraction, ...]} e.g. {1: [0.33, 0.67, 1.0]}
            raise_sizes: {street: [pot_fraction, ...]} for raises
            max_raises_per_street: Max number of raises per street
        """
        self.starting_pot = starting_pot
        self.starting_stacks = list(starting_stacks)
        self.board = list(board)
        self.max_raises = max_raises_per_street

        # Default bet sizes if not provided
        default_bets = [0.33, 0.67, 1.0]
        default_raises = [0.5, 1.0]

        if board:
            street = len(board) - 2 if len(board) >= 3 else 0
            street = max(1, street)  # Flop=1, Turn=2, River=3
        else:
            street = 0

        self.street = street
        self.bet_sizes = bet_sizes or {s: default_bets for s in range(4)}
        self.raise_sizes = raise_sizes or {s: default_raises for s in range(4)}
        self.node_count = 0

    def build(self) -> GameNode:
        """Build and return the root of the game tree."""
        initial_state = GameState(
            street=self.street,
            board=list(self.board),
            pot=self.starting_pot,
            stacks=list(self.starting_stacks),
            bets=[0.0, 0.0],
            acting_player=0,  # OOP acts first postflop
            history=[],
            street_history=[],
            is_allin=False,
            n_actions_this_street=0,
            last_aggressor=-1,
        )
        root = self._build_node(initial_state, 0)
        return root

    def _build_node(self, state: GameState, n_raises: int) -> GameNode:
        """Recursively build game tree from state."""
        self.node_count += 1

        # Safety: limit tree size
        if self.node_count > 100000:
            node = GameNode(NodeType.TERMINAL, state)
            node.payoff_multiplier = 1.0
            return node

        # Check terminal conditions

        # 1. Fold happened -> terminal
        if state.history and state.history[-1].action_type == ActionType.FOLD:
            node = GameNode(NodeType.TERMINAL, state)
            node.payoff_multiplier = -1.0  # Folder loses
            return node

        # 2. All-in with equal bets -> showdown terminal
        if state.is_allin and state.bets[0] == state.bets[1]:
            node = GameNode(NodeType.TERMINAL, state)
            node.payoff_multiplier = 1.0
            return node

        # 3. Past the river -> terminal (showdown)
        if state.street > 3:
            node = GameNode(NodeType.TERMINAL, state)
            node.payoff_multiplier = 1.0
            return node

        # 4. All-in where opponent still needs to act
        if state.is_allin:
            # Opponent should have call/fold options
            pass

        # Generate available actions
        actions = self._get_actions(state, n_raises)

        if not actions:
            # No actions = terminal
            node = GameNode(NodeType.TERMINAL, state)
            node.payoff_multiplier = 1.0
            return node

        node = GameNode(NodeType.PLAYER, state)

        for action in actions:
            child_state = self._apply_action(state, action, n_raises)
            child_raises = n_raises
            if action.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                child_raises += 1
            # Reset raises on new street
            if child_state.street != state.street:
                child_raises = 0

            child = self._build_node(child_state, child_raises)
            child.parent = node
            node.children[action.to_key()] = child

        return node

    def _get_actions(self, state: GameState, n_raises: int) -> List[Action]:
        """Generate legal actions for current player."""
        actions = []
        player = state.acting_player
        to_call = state.amount_to_call()
        pot = state.total_pot()
        stack = state.stacks[player]

        if stack <= 0:
            return []

        if to_call > 0:
            # Facing a bet/raise
            actions.append(Action(ActionType.FOLD))

            if to_call >= stack:
                # Can only call all-in
                actions.append(Action(ActionType.CALL, min(to_call, stack)))
            else:
                actions.append(Action(ActionType.CALL, to_call))

                # Raises
                if n_raises < self.max_raises:
                    pot_after_call = pot + to_call
                    for frac in self.raise_sizes.get(state.street, [1.0]):
                        raise_amount = to_call + pot_after_call * frac
                        raise_amount = min(raise_amount, stack)
                        raise_amount = max(raise_amount, to_call * 2)  # Min raise
                        if raise_amount >= stack * 0.9:
                            # Close enough to all-in
                            actions.append(Action(ActionType.ALL_IN, stack))
                            break
                        actions.append(Action(ActionType.RAISE, round(raise_amount, 1)))

                    # Always include all-in option if not already added
                    if not any(a.action_type == ActionType.ALL_IN for a in actions):
                        if stack > to_call:
                            actions.append(Action(ActionType.ALL_IN, stack))
        else:
            # No bet to call
            actions.append(Action(ActionType.CHECK))

            if n_raises < self.max_raises:
                for frac in self.bet_sizes.get(state.street, [0.67]):
                    bet_amount = pot * frac
                    bet_amount = max(bet_amount, 1.0)  # Min bet = 1 chip
                    bet_amount = min(bet_amount, stack)
                    if bet_amount >= stack * 0.9:
                        actions.append(Action(ActionType.ALL_IN, stack))
                        break
                    actions.append(Action(ActionType.BET, round(bet_amount, 1)))

                if not any(a.action_type == ActionType.ALL_IN for a in actions):
                    if stack > 0:
                        actions.append(Action(ActionType.ALL_IN, stack))

        # Deduplicate
        seen = set()
        unique_actions = []
        for a in actions:
            key = a.to_key()
            if key not in seen:
                seen.add(key)
                unique_actions.append(a)

        return unique_actions

    def _apply_action(self, state: GameState, action: Action, n_raises: int) -> GameState:
        """Apply action and return new state."""
        new = state.copy()
        player = new.acting_player
        opponent = 1 - player

        new.history.append(action)
        new.street_history.append(action)
        new.n_actions_this_street += 1

        if action.action_type == ActionType.FOLD:
            # Terminal: opponent wins pot
            new_node_state = new
            # Mark as terminal by clearing children possibilities
            new.stacks[opponent] += new.pot + new.bets[0] + new.bets[1]
            new.pot = 0
            new.bets = [0, 0]
            return new

        elif action.action_type == ActionType.CHECK:
            # Check: either pass to next player or end street
            if new.n_actions_this_street >= 2:
                # Both players checked -> end of street
                self._advance_street(new)
            else:
                new.acting_player = opponent

        elif action.action_type == ActionType.CALL:
            call_amount = min(action.amount, new.stacks[player])
            new.stacks[player] -= call_amount
            new.bets[player] += call_amount

            if new.stacks[player] <= 0 or new.stacks[opponent] <= 0:
                new.is_allin = True
            else:
                # Call ends action on this street
                self._advance_street(new)

        elif action.action_type in (ActionType.BET, ActionType.RAISE):
            bet_amount = min(action.amount, new.stacks[player])
            new.stacks[player] -= bet_amount
            new.bets[player] += bet_amount
            new.acting_player = opponent
            new.last_aggressor = player

        elif action.action_type == ActionType.ALL_IN:
            allin_amount = new.stacks[player]
            new.stacks[player] = 0
            new.bets[player] += allin_amount
            new.is_allin = True
            new.last_aggressor = player

            # If opponent still needs to act
            if new.bets[player] > new.bets[opponent]:
                new.acting_player = opponent
                new.is_allin = False  # Opponent can still fold/call

        return new

    def _advance_street(self, state: GameState):
        """Move to next street: collect bets into pot, reset street state."""
        state.pot += state.bets[0] + state.bets[1]
        state.bets = [0.0, 0.0]
        state.street += 1
        state.street_history = []
        state.n_actions_this_street = 0
        state.acting_player = 0  # OOP acts first
        state.last_aggressor = -1


def make_info_set_key(player: int, hole_cards: Tuple[Card, Card],
                      board: List[Card], history: List[Action],
                      bucket: int = -1) -> str:
    """
    Create unique key for information set.

    An information set groups all game states that look identical to a player:
    same hole cards, same board, same action history.

    If bucket >= 0, uses bucket instead of hole cards (for abstraction).
    """
    if bucket >= 0:
        hand_key = f"B{bucket}"
    else:
        sorted_hole = sorted(hole_cards, key=lambda c: c.id, reverse=True)
        hand_key = f"{sorted_hole[0]}{sorted_hole[1]}"

    board_key = ":".join(str(c) for c in board) if board else "x"
    history_key = "/".join(a.to_key() for a in history) if history else "start"

    return f"P{player}|{hand_key}|{board_key}|{history_key}"
