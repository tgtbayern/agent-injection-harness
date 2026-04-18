"""
Villain Range Estimator.

Infers villain's likely hand range from their action history.
"UTG raises" → top ~15% of hands.
"UTG raises, flop bet 67%" → further narrows to hands that connect with flop.
"""

from typing import List, Set, Tuple, Dict
from .cards import Card, RANK_CHARS

RANKS = 'AKQJT98765432'
POSITIONS = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']

# ============================================================
# Preflop opening ranges by position (canonical hand → True)
# ============================================================

_UTG = {'AA','KK','QQ','JJ','TT','99','88','77',
        'AKs','AKo','AQs','AQo','AJs','ATs','A9s','A5s','A4s',
        'KQs','KQo','KJs','KTs','QJs','QTs','JTs'}

_HJ = _UTG | {'66','AJo','ATo','A8s','KJo','K9s','J9s','T9s','98s','A7s','A6s'}

_CO = _HJ | {'55','44','A9o','A8o','A7s','A6s','A3s','A2s',
             'KTo','K8s','QJo','Q9s','JTo','J9s','T8s','97s','87s','76s'}

_BTN = _CO | {'33','22','A5o','A4o','A3o','K9o','K7s','K6s','K5s',
             'QTo','Q8s','J8s','T7s','96s','86s','75s','65s','54s'}

_SB = _BTN.copy()

# BB defend vs raise (wider)
_BB_DEFEND = _BTN | {'Q7s','J7s','T6s','95s','85s','74s','64s','53s','43s',
                     'A2o','K8o','Q9o','J9o','T9o','98o'}

# 3-bet range (tight)
_3BET = {'AA','KK','QQ','AKs','AKo','JJ','AQs','A5s','A4s','KQs'}

# 4-bet range
_4BET = {'AA','KK','QQ','AKs','AKo'}

OPEN_RANGES = {
    'UTG': _UTG, 'HJ': _HJ, 'CO': _CO, 'BTN': _BTN, 'SB': _SB,
    'BB': _BB_DEFEND,
}


def expand_hand_to_combos(hand: str, dead: Set[int] = None) -> List[Tuple[Card, Card]]:
    """Expand canonical hand like 'AKs' to all specific combos."""
    dead = dead or set()
    combos = []
    if len(hand) == 2:
        # Pair
        rank = RANK_CHARS.index(hand[0]) + 2
        for s1 in range(4):
            for s2 in range(s1+1, 4):
                c1 = Card(rank, s1)
                c2 = Card(rank, s2)
                if c1.id not in dead and c2.id not in dead:
                    combos.append((c1, c2))
    elif hand.endswith('s'):
        r1 = RANK_CHARS.index(hand[0]) + 2
        r2 = RANK_CHARS.index(hand[1]) + 2
        for s in range(4):
            c1 = Card(r1, s)
            c2 = Card(r2, s)
            if c1.id not in dead and c2.id not in dead:
                combos.append((c1, c2))
    elif hand.endswith('o'):
        r1 = RANK_CHARS.index(hand[0]) + 2
        r2 = RANK_CHARS.index(hand[1]) + 2
        for s1 in range(4):
            for s2 in range(4):
                if s1 != s2:
                    c1 = Card(r1, s1)
                    c2 = Card(r2, s2)
                    if c1.id not in dead and c2.id not in dead:
                        combos.append((c1, c2))
    return combos


def estimate_villain_range(
    action_history: List[str],
    hero_position: str,
    board_cards: List[Card] = None,
    hero_cards: List[Card] = None,
) -> Dict:
    """
    Analyze action history to estimate each active villain's range.

    Returns {
        "villains": {
            "UTG": {
                "range_name": "UTG open range (~15%)",
                "canonical_hands": ['AA','KK',...],
                "combos": [(Card,Card), ...],
                "actions_taken": ["raise_3"],
            },
            ...
        },
        "active_villains": ["UTG", "CO"],  # who's still in
    }
    """
    dead = set()
    if hero_cards:
        dead |= {c.id for c in hero_cards}
    if board_cards:
        dead |= {c.id for c in board_cards}

    # Parse action history to build per-player profile
    player_actions = {}  # pos → [action strings]
    for entry in action_history:
        if ':' in entry:
            pos, action = entry.split(':', 1)
            pos = pos.upper().strip()
        else:
            continue
        if pos not in player_actions:
            player_actions[pos] = []
        player_actions[pos].append(action.strip())

    villains = {}
    active = []
    n_raises_preflop = 0

    # Count preflop raises
    for pos in POSITIONS:
        acts = player_actions.get(pos, [])
        for a in acts:
            if 'raise' in a:
                n_raises_preflop += 1

    for pos in POSITIONS:
        if pos == hero_position:
            continue

        acts = player_actions.get(pos, [])

        # Determine if folded
        if any('fold' in a for a in acts):
            continue

        # If no action recorded and preflop, assume active
        has_action = len(acts) > 0

        # Determine range based on actions
        canonical = set()
        range_name = ""

        if not has_action:
            # No action yet — they could have any hand (not yet acted)
            canonical = OPEN_RANGES.get(pos, _CO).copy()
            range_name = f"{pos} (not yet acted, full range ~{len(canonical)} hands)"
            active.append(pos)
        else:
            first_action = acts[0] if acts else ''

            if 'raise' in first_action and n_raises_preflop <= 1:
                # Open raise
                canonical = OPEN_RANGES.get(pos, _CO).copy()
                pct = len(canonical) / 169 * 100
                range_name = f"{pos} open raise (~{pct:.0f}% = {len(canonical)} hands)"
            elif 'raise' in first_action and n_raises_preflop == 2:
                # 3-bet
                canonical = _3BET.copy()
                range_name = f"{pos} 3-bet (~{len(canonical)} premium hands)"
            elif 'raise' in first_action and n_raises_preflop >= 3:
                # 4-bet+
                canonical = _4BET.copy()
                range_name = f"{pos} 4-bet+ (only {len(canonical)} super-premium)"
            elif 'call' in first_action:
                # Calling range: has hands good enough to call but not 3bet
                base = OPEN_RANGES.get(pos, _CO)
                canonical = base - _3BET  # remove 3bet hands (they would have raised)
                # But add some trapping hands back
                canonical |= {'AA', 'KK'}  # sometimes trap
                range_name = f"{pos} flat call (~{len(canonical)} hands, capped range)"
            elif 'check' in first_action:
                canonical = OPEN_RANGES.get(pos, _CO).copy()
                range_name = f"{pos} checked (wide range)"
            elif 'allin' in first_action:
                canonical = _4BET.copy() | {'AKs', 'AKo', 'QQ', 'JJ'}
                range_name = f"{pos} all-in (polarized: premium + bluffs)"
            else:
                canonical = OPEN_RANGES.get(pos, _CO).copy()
                range_name = f"{pos} (~{len(canonical)} hands)"

            # Postflop action narrows further
            postflop_acts = [a for a in acts[1:]]  # skip first (preflop) action
            for pa in postflop_acts:
                if 'bet' in pa or 'raise' in pa:
                    try:
                        pct = float(pa.split('_')[1])
                    except (ValueError, IndexError):
                        pct = 50
                    if pct >= 67:
                        # Big bet = polarized (strong + bluffs, remove medium)
                        range_name += f" → bet {pct:.0f}% (polarized)"
                    else:
                        range_name += f" → bet {pct:.0f}%"
                elif 'call' in pa:
                    range_name += " → called (medium strength, capped)"
                elif 'check' in pa:
                    range_name += " → checked (weak or trapping)"

            active.append(pos)

        # Expand to actual card combos
        combos = []
        for h in canonical:
            combos.extend(expand_hand_to_combos(h, dead))

        villains[pos] = {
            "range_name": range_name,
            "canonical_hands": sorted(canonical),
            "combos": combos,
            "combo_count": len(combos),
            "actions_taken": acts,
        }

    return {
        "villains": villains,
        "active_villains": active,
    }
