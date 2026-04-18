"""
Solver engine - high-level interface for the CFR solver.

Provides easy-to-use API for:
1. Setting up poker scenarios
2. Running the solver
3. Querying results
"""

from typing import Dict, List, Tuple, Optional
from ..core.cards import Card, parse_cards, all_hole_card_combos
from ..core.game_tree import GameTreeBuilder, Action
from ..core.evaluator import evaluate_hand, get_hand_category
from .cfr_solver import DCFRSolver


class SolverEngine:
    """High-level solver interface."""

    def __init__(self):
        self.solver: Optional[DCFRSolver] = None
        self.tree_builder: Optional[GameTreeBuilder] = None
        self.last_result: Optional[Dict] = None

    def solve_spot(self,
                   board_str: str,
                   pot: float,
                   oop_stack: float,
                   ip_stack: float,
                   oop_range: List[str] = None,
                   ip_range: List[str] = None,
                   bet_sizes: List[float] = None,
                   raise_sizes: List[float] = None,
                   n_iterations: int = 500,
                   verbose: bool = False) -> Dict:
        """
        Solve a poker spot.

        Args:
            board_str: Board cards e.g. "As Kd 7h"
            pot: Current pot size
            oop_stack: OOP player remaining stack
            ip_stack: IP player remaining stack
            oop_range: List of hand strings for OOP range
            ip_range: List of hand strings for IP range
            bet_sizes: Bet sizes as pot fractions
            raise_sizes: Raise sizes as pot fractions
            n_iterations: Number of CFR iterations

        Returns:
            Dict with strategies and game info
        """
        board = parse_cards(board_str) if board_str.strip() else []

        # Build game tree
        bet_size_map = {s: (bet_sizes or [0.5, 1.0]) for s in range(4)}
        raise_size_map = {s: (raise_sizes or [1.0]) for s in range(4)}

        self.tree_builder = GameTreeBuilder(
            starting_pot=pot,
            starting_stacks=[oop_stack, ip_stack],
            board=board,
            bet_sizes=bet_size_map,
            raise_sizes=raise_size_map,
            max_raises_per_street=2
        )

        root = self.tree_builder.build()

        # Parse ranges
        oop_hands = self._parse_range(oop_range, board) if oop_range else None
        ip_hands = self._parse_range(ip_range, board) if ip_range else None

        # Create solver
        self.solver = DCFRSolver(
            root=root,
            board=board,
            oop_range=oop_hands,
            ip_range=ip_hands
        )

        # Solve
        strategies = self.solver.solve(n_iterations=n_iterations, verbose=verbose)

        # Organize results
        result = self._organize_results(strategies, board)
        self.last_result = result

        return result

    def _parse_range(self, range_strs: List[str], board: List[Card]) -> List[Tuple[Card, Card]]:
        """Parse range strings like ['AKs', 'QQ', 'JTs'] into card combos."""
        dead = set(c.id for c in board)
        combos = []

        for hand_str in range_strs:
            hand_str = hand_str.strip()
            if len(hand_str) == 4:
                # Specific hand like "AsKd"
                try:
                    cards = parse_cards(hand_str[:2] + " " + hand_str[2:])
                    if cards[0].id not in dead and cards[1].id not in dead:
                        combos.append((cards[0], cards[1]))
                except (ValueError, IndexError):
                    pass
            elif len(hand_str) == 2:
                # Pocket pair like "AA"
                rank1 = hand_str[0]
                combos.extend(self._expand_pair(rank1, dead))
            elif len(hand_str) == 3:
                rank1, rank2 = hand_str[0], hand_str[1]
                suited = hand_str[2].lower()
                if suited == 's':
                    combos.extend(self._expand_suited(rank1, rank2, dead))
                elif suited == 'o':
                    combos.extend(self._expand_offsuit(rank1, rank2, dead))

        return combos if combos else None

    def _expand_pair(self, rank_char: str, dead: set) -> List[Tuple[Card, Card]]:
        from ..core.cards import RANK_CHARS
        rank = RANK_CHARS.index(rank_char.upper()) + 2
        combos = []
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                c1 = Card(rank, s1)
                c2 = Card(rank, s2)
                if c1.id not in dead and c2.id not in dead:
                    combos.append((c1, c2))
        return combos

    def _expand_suited(self, r1: str, r2: str, dead: set) -> List[Tuple[Card, Card]]:
        from ..core.cards import RANK_CHARS
        rank1 = RANK_CHARS.index(r1.upper()) + 2
        rank2 = RANK_CHARS.index(r2.upper()) + 2
        combos = []
        for s in range(4):
            c1 = Card(rank1, s)
            c2 = Card(rank2, s)
            if c1.id not in dead and c2.id not in dead:
                combos.append((c1, c2))
        return combos

    def _expand_offsuit(self, r1: str, r2: str, dead: set) -> List[Tuple[Card, Card]]:
        from ..core.cards import RANK_CHARS
        rank1 = RANK_CHARS.index(r1.upper()) + 2
        rank2 = RANK_CHARS.index(r2.upper()) + 2
        combos = []
        for s1 in range(4):
            for s2 in range(4):
                if s1 != s2:
                    c1 = Card(rank1, s1)
                    c2 = Card(rank2, s2)
                    if c1.id not in dead and c2.id not in dead:
                        combos.append((c1, c2))
        return combos

    def _organize_results(self, strategies: Dict, board: List[Card]) -> Dict:
        """Organize raw strategy output into structured result."""
        oop_strategies = {}
        ip_strategies = {}

        for key, strategy in strategies.items():
            parts = key.split("|")
            player = parts[0]  # P0 or P1
            hand = parts[1]
            history = parts[3] if len(parts) > 3 else "root"

            entry = {
                "hand": hand,
                "history": history,
                "strategy": strategy,
            }

            if player == "P0":
                if history not in oop_strategies:
                    oop_strategies[history] = []
                oop_strategies[history].append(entry)
            else:
                if history not in ip_strategies:
                    ip_strategies[history] = []
                ip_strategies[history].append(entry)

        return {
            "board": [str(c) for c in board],
            "oop_strategies": oop_strategies,
            "ip_strategies": ip_strategies,
            "info_set_count": len(strategies),
            "tree_node_count": self.tree_builder.node_count if self.tree_builder else 0,
        }
