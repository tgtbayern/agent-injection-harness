"""
Preflop GTO strategy tables.

Contains standard preflop opening ranges, 3-bet ranges, and calling ranges
for 6-max No-Limit Hold'em at 100BB effective stacks.

Range format: each hand has action frequencies that sum to 1.0.
"""

from typing import Dict, Tuple

# 169 canonical starting hands
RANKS = 'AKQJT98765432'

# Standard 13x13 grid (row=card1, col=card2)
# Upper triangle = suited, lower triangle = offsuit, diagonal = pairs
ALL_HANDS = []
for i, r1 in enumerate(RANKS):
    for j, r2 in enumerate(RANKS):
        if i < j:
            ALL_HANDS.append(f"{r1}{r2}s")
        elif i > j:
            ALL_HANDS.append(f"{r2}{r1}o")
        else:
            ALL_HANDS.append(f"{r1}{r2}")

# Number of combos for each hand type
def hand_combos(hand: str) -> int:
    if len(hand) == 2:  # Pair
        return 6
    elif hand[-1] == 's':
        return 4
    else:
        return 12


# ============================================================
# 6-MAX PREFLOP RANGES (RFI = Raise First In)
# ============================================================

# UTG opening range (~15%)
UTG_RFI = {
    # Premium pairs
    'AA': {'raise': 1.0}, 'KK': {'raise': 1.0}, 'QQ': {'raise': 1.0},
    'JJ': {'raise': 1.0}, 'TT': {'raise': 1.0}, '99': {'raise': 1.0},
    '88': {'raise': 1.0}, '77': {'raise': 0.5, 'fold': 0.5},
    # Broadway
    'AKs': {'raise': 1.0}, 'AKo': {'raise': 1.0},
    'AQs': {'raise': 1.0}, 'AQo': {'raise': 1.0},
    'AJs': {'raise': 1.0}, 'AJo': {'raise': 0.5, 'fold': 0.5},
    'ATs': {'raise': 1.0},
    'KQs': {'raise': 1.0}, 'KQo': {'raise': 0.5, 'fold': 0.5},
    'KJs': {'raise': 1.0},
    'KTs': {'raise': 0.5, 'fold': 0.5},
    'QJs': {'raise': 1.0},
    'QTs': {'raise': 0.5, 'fold': 0.5},
    'JTs': {'raise': 1.0},
    # Suited aces
    'A9s': {'raise': 0.5, 'fold': 0.5},
    'A8s': {'raise': 0.3, 'fold': 0.7},
    'A5s': {'raise': 0.5, 'fold': 0.5},
    'A4s': {'raise': 0.3, 'fold': 0.7},
}

# HJ opening range (~19%)
HJ_RFI = {
    **{k: v for k, v in UTG_RFI.items()},
    '77': {'raise': 1.0}, '66': {'raise': 1.0},
    'AJo': {'raise': 1.0}, 'ATo': {'raise': 0.5, 'fold': 0.5},
    'ATs': {'raise': 1.0}, 'A9s': {'raise': 1.0},
    'KQo': {'raise': 1.0},
    'KJs': {'raise': 1.0}, 'KTs': {'raise': 1.0},
    'QTs': {'raise': 1.0}, 'QJs': {'raise': 1.0},
    'J9s': {'raise': 0.5, 'fold': 0.5},
    'T9s': {'raise': 1.0},
    '98s': {'raise': 0.5, 'fold': 0.5},
    'A8s': {'raise': 0.5, 'fold': 0.5},
    'A5s': {'raise': 1.0}, 'A4s': {'raise': 0.5, 'fold': 0.5},
}

# CO opening range (~27%)
CO_RFI = {
    **{k: v for k, v in HJ_RFI.items()},
    '66': {'raise': 1.0}, '55': {'raise': 1.0},
    'ATo': {'raise': 1.0}, 'A9o': {'raise': 0.5, 'fold': 0.5},
    'A8s': {'raise': 1.0}, 'A7s': {'raise': 1.0},
    'A6s': {'raise': 1.0}, 'A5s': {'raise': 1.0},
    'A4s': {'raise': 1.0}, 'A3s': {'raise': 0.5, 'fold': 0.5},
    'A2s': {'raise': 0.5, 'fold': 0.5},
    'KJo': {'raise': 1.0}, 'KTo': {'raise': 0.5, 'fold': 0.5},
    'K9s': {'raise': 1.0}, 'K8s': {'raise': 0.5, 'fold': 0.5},
    'QJo': {'raise': 1.0}, 'QTo': {'raise': 0.5, 'fold': 0.5},
    'Q9s': {'raise': 1.0},
    'J9s': {'raise': 1.0}, 'JTo': {'raise': 0.5, 'fold': 0.5},
    'T9s': {'raise': 1.0}, 'T8s': {'raise': 0.5, 'fold': 0.5},
    '98s': {'raise': 1.0}, '87s': {'raise': 1.0},
    '97s': {'raise': 0.5, 'fold': 0.5},
    '76s': {'raise': 0.5, 'fold': 0.5},
}

# BTN opening range (~45%)
BTN_RFI = {
    **{k: v for k, v in CO_RFI.items()},
    '55': {'raise': 1.0}, '44': {'raise': 1.0}, '33': {'raise': 1.0},
    '22': {'raise': 1.0},
    'A9o': {'raise': 1.0}, 'A8o': {'raise': 1.0},
    'A7o': {'raise': 0.5, 'fold': 0.5}, 'A6o': {'raise': 0.5, 'fold': 0.5},
    'A7s': {'raise': 1.0}, 'A6s': {'raise': 1.0},
    'A5s': {'raise': 1.0}, 'A4s': {'raise': 1.0},
    'A3s': {'raise': 1.0}, 'A2s': {'raise': 1.0},
    'KTo': {'raise': 1.0}, 'K9o': {'raise': 0.5, 'fold': 0.5},
    'K8s': {'raise': 1.0}, 'K7s': {'raise': 1.0},
    'K6s': {'raise': 1.0}, 'K5s': {'raise': 0.5, 'fold': 0.5},
    'QTo': {'raise': 1.0}, 'Q9o': {'raise': 0.5, 'fold': 0.5},
    'Q8s': {'raise': 1.0}, 'Q7s': {'raise': 0.5, 'fold': 0.5},
    'JTo': {'raise': 1.0}, 'J9o': {'raise': 0.5, 'fold': 0.5},
    'J8s': {'raise': 1.0}, 'J7s': {'raise': 0.5, 'fold': 0.5},
    'T8s': {'raise': 1.0}, 'T7s': {'raise': 0.5, 'fold': 0.5},
    'T9o': {'raise': 0.5, 'fold': 0.5},
    '97s': {'raise': 1.0}, '96s': {'raise': 0.5, 'fold': 0.5},
    '87s': {'raise': 1.0}, '86s': {'raise': 0.5, 'fold': 0.5},
    '76s': {'raise': 1.0}, '75s': {'raise': 0.5, 'fold': 0.5},
    '65s': {'raise': 1.0}, '64s': {'raise': 0.5, 'fold': 0.5},
    '54s': {'raise': 1.0}, '53s': {'raise': 0.5, 'fold': 0.5},
    '43s': {'raise': 0.5, 'fold': 0.5},
}

# SB opening range (limp or raise) (~40%)
SB_RFI = {
    **{k: v for k, v in BTN_RFI.items()},
    # SB has slightly tighter range due to positional disadvantage
}

# BB defense vs BTN
BB_DEFEND_VS_BTN = {
    # 3-bet range
    'AA': {'raise': 1.0}, 'KK': {'raise': 1.0}, 'QQ': {'raise': 1.0},
    'JJ': {'raise': 0.5, 'call': 0.5},
    'AKs': {'raise': 1.0}, 'AKo': {'raise': 1.0},
    'AQs': {'raise': 0.7, 'call': 0.3},
    'AQo': {'raise': 0.3, 'call': 0.7},
    'A5s': {'raise': 0.5, 'call': 0.5},
    'A4s': {'raise': 0.5, 'call': 0.5},
    # Calling range
    'TT': {'call': 1.0}, '99': {'call': 1.0}, '88': {'call': 1.0},
    '77': {'call': 1.0}, '66': {'call': 1.0}, '55': {'call': 1.0},
    '44': {'call': 0.5, 'fold': 0.5}, '33': {'call': 0.5, 'fold': 0.5},
    '22': {'call': 0.5, 'fold': 0.5},
    'AJs': {'call': 1.0}, 'ATs': {'call': 1.0},
    'A9s': {'call': 1.0}, 'A8s': {'call': 1.0},
    'A7s': {'call': 1.0}, 'A6s': {'call': 1.0},
    'A3s': {'call': 1.0}, 'A2s': {'call': 1.0},
    'AJo': {'call': 1.0}, 'ATo': {'call': 0.7, 'fold': 0.3},
    'A9o': {'call': 0.5, 'fold': 0.5},
    'KQs': {'raise': 0.3, 'call': 0.7}, 'KQo': {'call': 1.0},
    'KJs': {'call': 1.0}, 'KTs': {'call': 1.0},
    'K9s': {'call': 1.0}, 'K8s': {'call': 0.5, 'fold': 0.5},
    'KJo': {'call': 0.5, 'fold': 0.5},
    'QJs': {'call': 1.0}, 'QTs': {'call': 1.0},
    'Q9s': {'call': 1.0}, 'Q8s': {'call': 0.5, 'fold': 0.5},
    'QJo': {'call': 0.5, 'fold': 0.5},
    'JTs': {'call': 1.0}, 'J9s': {'call': 1.0},
    'J8s': {'call': 0.5, 'fold': 0.5},
    'JTo': {'call': 0.5, 'fold': 0.5},
    'T9s': {'call': 1.0}, 'T8s': {'call': 1.0},
    'T9o': {'call': 0.5, 'fold': 0.5},
    '98s': {'call': 1.0}, '97s': {'call': 0.5, 'fold': 0.5},
    '87s': {'call': 1.0}, '86s': {'call': 0.5, 'fold': 0.5},
    '76s': {'call': 1.0}, '75s': {'call': 0.5, 'fold': 0.5},
    '65s': {'call': 1.0}, '64s': {'call': 0.5, 'fold': 0.5},
    '54s': {'call': 1.0}, '53s': {'call': 0.5, 'fold': 0.5},
    '43s': {'call': 0.5, 'fold': 0.5},
}


# ============================================================
# Lookup functions
# ============================================================

POSITION_RANGES = {
    'UTG': UTG_RFI,
    'HJ': HJ_RFI,
    'CO': CO_RFI,
    'BTN': BTN_RFI,
    'SB': SB_RFI,
    'BB_vs_BTN': BB_DEFEND_VS_BTN,
}


def get_preflop_strategy(position: str, hand: str) -> Dict[str, float]:
    """
    Get preflop strategy for a hand at a given position.

    Args:
        position: 'UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB_vs_BTN'
        hand: Canonical hand like 'AKs', 'QQ', 'T9o'

    Returns:
        Dict of action -> frequency, e.g. {'raise': 0.7, 'fold': 0.3}
    """
    ranges = POSITION_RANGES.get(position, {})
    strategy = ranges.get(hand, {'fold': 1.0})
    return strategy


def get_full_range_matrix(position: str) -> Dict[str, Dict[str, float]]:
    """Get the complete 13x13 range matrix for a position."""
    matrix = {}
    for hand in ALL_HANDS:
        matrix[hand] = get_preflop_strategy(position, hand)
    return matrix


def canonicalize_hand(card1_rank: str, card2_rank: str, suited: bool) -> str:
    """Convert two rank chars and suited flag to canonical hand notation."""
    r1_idx = RANKS.index(card1_rank)
    r2_idx = RANKS.index(card2_rank)

    if r1_idx == r2_idx:
        return f"{card1_rank}{card2_rank}"
    elif r1_idx < r2_idx:
        return f"{card1_rank}{card2_rank}{'s' if suited else 'o'}"
    else:
        return f"{card2_rank}{card1_rank}{'s' if suited else 'o'}"


def hand_to_canonical(card1_str: str, card2_str: str) -> str:
    """Convert specific hand (e.g. 'Ah', 'Kd') to canonical form ('AKo')."""
    r1 = card1_str[0].upper()
    r2 = card2_str[0].upper()
    s1 = card1_str[1].lower()
    s2 = card2_str[1].lower()
    suited = s1 == s2
    return canonicalize_hand(r1, r2, suited)
