"""
Fast hand evaluator for 5-7 card poker hands.
Uses lookup-table based evaluation for speed.

Hand rankings (higher = better):
    8: Straight Flush
    7: Four of a Kind
    6: Full House
    5: Flush
    4: Straight
    3: Three of a Kind
    2: Two Pair
    1: One Pair
    0: High Card
"""

from typing import List, Tuple, Dict
from .cards import Card
import itertools


# Hand category names
HAND_CATEGORIES = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"
]


def evaluate_5cards(cards: List[Card]) -> int:
    """
    Evaluate exactly 5 cards. Returns integer score (higher = better).
    Score format: category * 1_000_000 + kicker_score
    Kicker score uses base-15 encoding (ranks are 2..14).
    """
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]

    is_flush = len(set(suits)) == 1
    is_straight, straight_high = _check_straight(ranks)

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    counts = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    if is_straight and is_flush:
        return 8 * 1_000_000 + straight_high
    elif counts[0][1] == 4:
        quad_rank = counts[0][0]
        kicker = counts[1][0]
        return 7 * 1_000_000 + quad_rank * 15 + kicker
    elif counts[0][1] == 3 and counts[1][1] == 2:
        trip_rank = counts[0][0]
        pair_rank = counts[1][0]
        return 6 * 1_000_000 + trip_rank * 15 + pair_rank
    elif is_flush:
        return 5 * 1_000_000 + _kicker_score(ranks)
    elif is_straight:
        return 4 * 1_000_000 + straight_high
    elif counts[0][1] == 3:
        trip_rank = counts[0][0]
        kickers = sorted([r for r, c in counts if c == 1], reverse=True)
        return 3 * 1_000_000 + trip_rank * 15 * 15 + _kicker_score(kickers)
    elif counts[0][1] == 2 and counts[1][1] == 2:
        high_pair = max(counts[0][0], counts[1][0])
        low_pair = min(counts[0][0], counts[1][0])
        kicker = counts[2][0]
        return 2 * 1_000_000 + high_pair * 225 + low_pair * 15 + kicker
    elif counts[0][1] == 2:
        pair_rank = counts[0][0]
        kickers = sorted([r for r, c in counts if c == 1], reverse=True)
        return 1 * 1_000_000 + pair_rank * 15 * 15 * 15 + _kicker_score(kickers)
    else:
        return 0 * 1_000_000 + _kicker_score(ranks)


def evaluate_hand(hole_cards: List[Card], board: List[Card]) -> int:
    """
    Evaluate best 5-card hand from hole cards + board (5-7 cards total).
    Returns integer score (higher = better).
    """
    all_cards = list(hole_cards) + list(board)
    if len(all_cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(all_cards)}")

    best = 0
    for combo in itertools.combinations(all_cards, 5):
        score = evaluate_5cards(list(combo))
        if score > best:
            best = score
    return best


def get_hand_category(score: int) -> str:
    """Get human readable hand category from score."""
    category = score // 1_000_000
    if category < 0 or category >= len(HAND_CATEGORIES):
        return "Unknown"
    return HAND_CATEGORIES[category]


def _check_straight(ranks: List[int]) -> Tuple[bool, int]:
    """Check if sorted (desc) ranks form a straight. Returns (is_straight, high_card)."""
    unique_ranks = sorted(set(ranks), reverse=True)
    if len(unique_ranks) < 5:
        return False, 0

    # Check normal straight
    for i in range(len(unique_ranks) - 4):
        if unique_ranks[i] - unique_ranks[i + 4] == 4:
            return True, unique_ranks[i]

    # Check wheel (A-2-3-4-5)
    if set([14, 2, 3, 4, 5]).issubset(set(unique_ranks)):
        return True, 5

    return False, 0


def _kicker_score(ranks: List[int]) -> int:
    """Convert a list of kicker ranks to a comparable integer using base-15."""
    score = 0
    for i, r in enumerate(ranks):
        score += r * (15 ** (len(ranks) - 1 - i))
    return score


def compute_equity(hole_cards: List[Card], board: List[Card],
                   villain_range: List[Tuple[Card, Card]] = None,
                   n_simulations: int = 0) -> float:
    """
    Compute equity of hole_cards against a range on a given board.
    If board is incomplete, runs Monte Carlo simulation.
    If villain_range is None, assumes all possible hands.

    Args:
        hole_cards: Player's two hole cards
        board: Community cards (0-5)
        villain_range: List of possible villain hands
        n_simulations: Number of MC simulations (0 = exact enumeration)

    Returns:
        Equity as float [0.0, 1.0]
    """
    import random as rng

    dead = set(c.id for c in hole_cards) | set(c.id for c in board)
    remaining_deck = [Card.from_id(i) for i in range(52) if i not in dead]

    if villain_range is None:
        villain_range = list(itertools.combinations(remaining_deck, 2))
    else:
        villain_range = [(c1, c2) for c1, c2 in villain_range
                         if c1.id not in dead and c2.id not in dead]

    cards_needed = 5 - len(board)
    wins = 0
    ties = 0
    total = 0

    if cards_needed == 0:
        # River - exact evaluation
        hero_score = evaluate_hand(hole_cards, board)
        for v1, v2 in villain_range:
            if v1.id in dead or v2.id in dead:
                continue
            villain_score = evaluate_hand([v1, v2], board)
            if hero_score > villain_score:
                wins += 1
            elif hero_score == villain_score:
                ties += 1
            total += 1
    elif n_simulations > 0:
        # Monte Carlo
        for _ in range(n_simulations):
            valid_villain = [(v1, v2) for v1, v2 in villain_range
                             if v1.id not in dead and v2.id not in dead]
            if not valid_villain:
                continue
            v1, v2 = rng.choice(valid_villain)
            v_dead = dead | {v1.id, v2.id}
            deck = [Card.from_id(i) for i in range(52) if i not in v_dead]
            rng.shuffle(deck)
            full_board = list(board) + deck[:cards_needed]

            hero_score = evaluate_hand(hole_cards, full_board)
            villain_score = evaluate_hand([v1, v2], full_board)
            if hero_score > villain_score:
                wins += 1
            elif hero_score == villain_score:
                ties += 1
            total += 1
    else:
        # Exact enumeration (only practical for river / turn with small ranges)
        for v1, v2 in villain_range:
            if v1.id in dead or v2.id in dead:
                continue
            v_dead = dead | {v1.id, v2.id}
            deck_remaining = [Card.from_id(i) for i in range(52) if i not in v_dead]
            for runout in itertools.combinations(deck_remaining, cards_needed):
                full_board = list(board) + list(runout)
                hero_score = evaluate_hand(hole_cards, full_board)
                villain_score = evaluate_hand([v1, v2], full_board)
                if hero_score > villain_score:
                    wins += 1
                elif hero_score == villain_score:
                    ties += 1
                total += 1

    if total == 0:
        return 0.5
    return (wins + ties * 0.5) / total


def compute_multiway_equity(hole_cards: List[Card], board: List[Card],
                            num_villains: int = 2,
                            n_simulations: int = 10000) -> Dict:
    """
    Compute equity in a multi-way pot via Monte Carlo simulation.

    Args:
        hole_cards: Hero's two hole cards
        board: Community cards (0-5)
        num_villains: Number of opponents (1-8)
        n_simulations: Number of MC simulations

    Returns:
        Dict with:
        - equity: float (win probability)
        - win_pct: float
        - tie_pct: float
        - lose_pct: float
        - per_villain_equities: list of equity vs each villain position
    """
    import random as rng

    num_villains = max(1, min(num_villains, 8))
    dead = set(c.id for c in hole_cards) | set(c.id for c in board)
    cards_needed = 5 - len(board)

    wins = 0
    ties = 0
    losses = 0
    total = 0

    for _ in range(n_simulations):
        # Build remaining deck
        deck = [Card.from_id(i) for i in range(52) if i not in dead]
        rng.shuffle(deck)

        # Deal villain hands
        idx = 0
        villain_hands = []
        valid = True
        for v in range(num_villains):
            if idx + 1 >= len(deck):
                valid = False
                break
            villain_hands.append((deck[idx], deck[idx + 1]))
            idx += 2
        if not valid:
            continue

        # Complete the board
        remaining_for_board = deck[idx:]
        full_board = list(board) + remaining_for_board[:cards_needed]

        if len(full_board) < 5:
            continue

        # Evaluate all hands
        hero_score = evaluate_hand(hole_cards, full_board)

        hero_wins = True
        hero_ties = False
        for vh in villain_hands:
            v_score = evaluate_hand(list(vh), full_board)
            if v_score > hero_score:
                hero_wins = False
                hero_ties = False
                break
            elif v_score == hero_score:
                hero_ties = True

        if hero_wins and not hero_ties:
            wins += 1
        elif hero_wins and hero_ties:
            ties += 1
        else:
            losses += 1
        total += 1

    if total == 0:
        eq = 1.0 / (num_villains + 1)
        return {
            "equity": eq,
            "win_pct": eq,
            "tie_pct": 0.0,
            "lose_pct": 1.0 - eq,
            "num_villains": num_villains,
            "simulations": 0,
        }

    win_pct = wins / total
    tie_pct = ties / total
    lose_pct = losses / total
    # Equity = win% + tie%/2 (simplified for multiway ties)
    equity = win_pct + tie_pct * 0.5

    return {
        "equity": round(equity, 4),
        "win_pct": round(win_pct, 4),
        "tie_pct": round(tie_pct, 4),
        "lose_pct": round(lose_pct, 4),
        "num_villains": num_villains,
        "simulations": total,
    }


def compute_equity_distribution(hole_cards: List[Card], board: List[Card],
                                buckets: int = 10, n_samples: int = 1000) -> List[float]:
    """
    Compute equity distribution (histogram) against random hands.
    Used for card abstraction / bucketing.

    Returns list of bucket frequencies summing to 1.0.
    """
    import random as rng

    dead = set(c.id for c in hole_cards) | set(c.id for c in board)
    remaining = [Card.from_id(i) for i in range(52) if i not in dead]

    equities = []
    cards_needed = 5 - len(board)

    for _ in range(n_samples):
        rng.shuffle(remaining)
        # Pick villain hand
        villain = remaining[:2]
        # Complete board
        v_dead = set(c.id for c in villain)
        deck = [c for c in remaining[2:] if c.id not in v_dead]
        if len(deck) < cards_needed:
            continue
        full_board = list(board) + deck[:cards_needed]

        hero = evaluate_hand(hole_cards, full_board)
        vill = evaluate_hand(villain, full_board)
        if hero > vill:
            equities.append(1.0)
        elif hero == vill:
            equities.append(0.5)
        else:
            equities.append(0.0)

    if not equities:
        return [1.0 / buckets] * buckets

    histogram = [0.0] * buckets
    for eq in equities:
        idx = min(int(eq * buckets), buckets - 1)
        histogram[idx] += 1

    total = sum(histogram)
    return [h / total for h in histogram]
