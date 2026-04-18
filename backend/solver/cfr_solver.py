"""
Discounted CFR (DCFR) Solver for poker.

Implements the Discounted Counterfactual Regret Minimization algorithm
from Brown & Sandholm (2019). Converges ~10x faster than vanilla CFR+.

Key idea: discount past regrets and strategy sums by factors that
decrease over iterations, giving more weight to recent iterations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from ..core.cards import Card, all_hole_card_combos
from ..core.game_tree import (
    GameNode, GameTreeBuilder, GameState, NodeType,
    Action, ActionType, make_info_set_key
)
from ..core.evaluator import evaluate_hand


class InfoSetData:
    """Stores CFR data for a single information set."""

    __slots__ = ('actions', 'regret_sum', 'strategy_sum', 'n_actions')

    def __init__(self, actions: List[str]):
        self.actions = actions
        self.n_actions = len(actions)
        self.regret_sum = np.zeros(self.n_actions, dtype=np.float64)
        self.strategy_sum = np.zeros(self.n_actions, dtype=np.float64)

    def get_strategy(self) -> np.ndarray:
        """Regret Matching: convert regrets to strategy."""
        positive = np.maximum(self.regret_sum, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        else:
            return np.ones(self.n_actions) / self.n_actions

    def get_average_strategy(self) -> np.ndarray:
        """Get final average strategy (the NE approximation)."""
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        else:
            return np.ones(self.n_actions) / self.n_actions


class DCFRSolver:
    """
    Discounted CFR solver for single-street poker subgames.

    Solves for Nash Equilibrium strategies in an abstracted game tree.
    """

    def __init__(self,
                 root: GameNode,
                 board: List[Card],
                 oop_range: List[Tuple[Card, Card]] = None,
                 ip_range: List[Tuple[Card, Card]] = None,
                 alpha: float = 1.5,
                 beta: float = 0.5,
                 gamma: float = 2.0):
        """
        Args:
            root: Root of the game tree
            board: Community cards
            oop_range: OOP player's starting range
            ip_range: IP player's starting range
            alpha, beta, gamma: DCFR discount parameters
                alpha: discount for positive regret (default 1.5)
                beta: discount for negative regret (default 0.5)
                gamma: discount for strategy sum (default 2.0)
        """
        self.root = root
        self.board = board
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Info set storage
        self.info_sets: Dict[str, InfoSetData] = {}

        # Dead cards
        dead = set(c.id for c in board)

        # Player ranges
        if oop_range is None or ip_range is None:
            all_combos = all_hole_card_combos(board)
            oop_range = oop_range or all_combos
            ip_range = ip_range or all_combos

        self.oop_range = [(c1, c2) for c1, c2 in oop_range
                          if c1.id not in dead and c2.id not in dead]
        self.ip_range = [(c1, c2) for c1, c2 in ip_range
                         if c1.id not in dead and c2.id not in dead]

        # Iteration counter
        self.iteration = 0
        self.exploitability_history: List[float] = []

    def get_or_create_info_set(self, key: str, actions: List[str]) -> InfoSetData:
        """Get existing or create new information set."""
        if key not in self.info_sets:
            self.info_sets[key] = InfoSetData(actions)
        return self.info_sets[key]

    def solve(self, n_iterations: int = 1000, verbose: bool = False) -> Dict:
        """
        Run DCFR with external sampling for n_iterations.
        Uses Monte Carlo sampling of hand matchups for speed.

        Returns dict of info_set_key -> average strategy.
        """
        import random

        # Pre-filter valid hand matchups
        valid_pairs = []
        for oop_hand in self.oop_range:
            for ip_hand in self.ip_range:
                if (oop_hand[0].id != ip_hand[0].id and
                        oop_hand[0].id != ip_hand[1].id and
                        oop_hand[1].id != ip_hand[0].id and
                        oop_hand[1].id != ip_hand[1].id):
                    valid_pairs.append((oop_hand, ip_hand))

        if not valid_pairs:
            return self._extract_strategies()

        # Number of hand samples per iteration
        samples_per_iter = min(len(valid_pairs), max(10, len(valid_pairs) // 10))

        for t in range(1, n_iterations + 1):
            self.iteration = t

            # Sample a subset of hand matchups each iteration
            sampled = random.sample(valid_pairs, samples_per_iter) if len(valid_pairs) > samples_per_iter else valid_pairs

            for oop_hand, ip_hand in sampled:
                hands = [oop_hand, ip_hand]
                self._cfr(self.root, hands, 1.0, 1.0, t)

            # Apply discounting
            self._discount_regrets(t)

            if verbose and t % max(1, n_iterations // 10) == 0:
                exp = self._estimate_exploitability()
                self.exploitability_history.append(exp)
                print(f"  Iteration {t}/{n_iterations}, "
                      f"Info sets: {len(self.info_sets)}, "
                      f"Exploitability: {exp:.4f}")

        return self._extract_strategies()

    def _cfr(self, node: GameNode, hands: List[Tuple[Card, Card]],
             reach_0: float, reach_1: float, t: int) -> float:
        """
        Core CFR traversal.

        Args:
            node: Current game tree node
            hands: [oop_hand, ip_hand]
            reach_0: Probability player 0 reaches this node
            reach_1: Probability player 1 reaches this node
            t: Current iteration number

        Returns:
            Expected value for player 0 at this node.
        """
        state = node.state

        # Terminal node
        if node.is_terminal:
            return self._terminal_value(state, hands)

        if not node.children:
            return self._terminal_value(state, hands)

        player = state.acting_player
        action_keys = list(node.children.keys())

        # Get info set
        hand = hands[player]
        info_key = make_info_set_key(
            player, hand, state.board, state.history
        )
        info_set = self.get_or_create_info_set(info_key, action_keys)
        strategy = info_set.get_strategy()

        # Compute value for each action
        action_values = np.zeros(len(action_keys))
        node_value = 0.0

        for i, action_key in enumerate(action_keys):
            child = node.children[action_key]

            if player == 0:
                action_values[i] = self._cfr(
                    child, hands,
                    reach_0 * strategy[i], reach_1, t
                )
            else:
                action_values[i] = -self._cfr(
                    child, hands,
                    reach_0, reach_1 * strategy[i], t
                )

            node_value += strategy[i] * action_values[i]

        # Update regrets
        opponent_reach = reach_1 if player == 0 else reach_0
        for i in range(len(action_keys)):
            regret = opponent_reach * (action_values[i] - node_value)
            info_set.regret_sum[i] += regret

        # Update strategy sum (weighted by current player's reach)
        my_reach = reach_0 if player == 0 else reach_1
        info_set.strategy_sum += my_reach * strategy

        if player == 0:
            return node_value
        else:
            return -node_value

    def _terminal_value(self, state: GameState, hands: List[Tuple[Card, Card]]) -> float:
        """
        Compute terminal payoff from player 0's perspective.
        """
        # Check for fold
        if state.history and state.history[-1].action_type == ActionType.FOLD:
            folder = state.acting_player
            pot = state.pot + state.bets[0] + state.bets[1]

            # The player who DIDN'T fold wins
            if folder == 0:
                # P0 folded, P0 loses what they put in
                return -(state.pot / 2 + state.bets[0])
            else:
                # P1 folded, P0 wins
                return state.pot / 2 + state.bets[1]

        # Showdown
        board = state.board
        hand0 = list(hands[0])
        hand1 = list(hands[1])

        score0 = evaluate_hand(hand0, board)
        score1 = evaluate_hand(hand1, board)

        pot = state.pot + state.bets[0] + state.bets[1]
        half_pot = pot / 2

        if score0 > score1:
            return half_pot
        elif score0 < score1:
            return -half_pot
        else:
            return 0.0  # Split pot

    def _discount_regrets(self, t: int):
        """Apply DCFR discounting to all info sets."""
        # DCFR discount factors
        pos_discount = t ** self.alpha / (t ** self.alpha + 1)
        neg_discount = t ** self.beta / (t ** self.beta + 1)
        strat_discount = (t / (t + 1)) ** self.gamma

        for info_set in self.info_sets.values():
            # Discount positive and negative regrets separately
            for i in range(info_set.n_actions):
                if info_set.regret_sum[i] > 0:
                    info_set.regret_sum[i] *= pos_discount
                else:
                    info_set.regret_sum[i] *= neg_discount

            # Discount strategy sums
            info_set.strategy_sum *= strat_discount

    def _estimate_exploitability(self) -> float:
        """Rough estimate of strategy exploitability (sum of positive regrets)."""
        total_regret = 0
        count = 0
        for info_set in self.info_sets.values():
            total_regret += np.maximum(info_set.regret_sum, 0).sum()
            count += 1
        return total_regret / max(count, 1)

    def _extract_strategies(self) -> Dict[str, Dict[str, float]]:
        """Extract final average strategies for all info sets."""
        strategies = {}
        for key, info_set in self.info_sets.items():
            avg = info_set.get_average_strategy()
            strategy = {}
            for i, action in enumerate(info_set.actions):
                if avg[i] > 0.001:  # Filter out near-zero actions
                    strategy[action] = round(float(avg[i]), 4)
            # Normalize
            total = sum(strategy.values())
            if total > 0:
                strategy = {k: round(v / total, 4) for k, v in strategy.items()}
            strategies[key] = strategy
        return strategies

    def get_strategy_for_hand(self, player: int, hole_cards: Tuple[Card, Card],
                              board: List[Card], history: List[Action]) -> Dict[str, float]:
        """Get the solved strategy for a specific hand at a specific point."""
        key = make_info_set_key(player, hole_cards, board, history)
        if key in self.info_sets:
            info_set = self.info_sets[key]
            avg = info_set.get_average_strategy()
            result = {}
            for i, action in enumerate(info_set.actions):
                result[action] = round(float(avg[i]), 4)
            return result
        return {}
