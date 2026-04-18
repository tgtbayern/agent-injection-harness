"""
Hand Advisor Engine - Core decision recommendation system.

Provides GTO-based recommendations for any poker scenario:
- Preflop: open, 3bet, 4bet, call, fold decisions
- Postflop: check, bet (multiple sizings), raise, call, fold

Modeled after PioSolver / GTO Wizard style hand advisors.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import random
import math

# ============================================================
# Preflop comprehensive strategy database
# ============================================================

RANKS = 'AKQJT98765432'
POSITIONS = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']

# Hand strength tiers (1=premium, 8=trash)
def hand_tier(hand: str) -> int:
    """Classify hand into strength tier."""
    if len(hand) == 2:  # Pair
        r = RANKS.index(hand[0])
        if r <= 1: return 1   # AA, KK
        if r <= 3: return 2   # QQ, JJ
        if r <= 5: return 3   # TT, 99
        if r <= 7: return 4   # 88, 77
        if r <= 9: return 5   # 66, 55
        return 6              # 44-22

    r1 = RANKS.index(hand[0])
    r2 = RANKS.index(hand[1])
    suited = hand[-1] == 's'
    gap = r2 - r1

    if r1 == 0:  # Ax
        if r2 <= 1: return 1  # AK
        if r2 <= 2: return 2  # AQ
        if r2 <= 3 and suited: return 2  # AJs
        if r2 <= 3: return 3  # AJo
        if r2 <= 5 and suited: return 3  # ATs, A9s
        if r2 <= 5: return 4
        if suited: return 4   # Axs
        return 6              # Axo

    if r1 <= 1:  # Kx
        if r2 <= 2: return 2 if suited else 3
        if r2 <= 4 and suited: return 3
        if r2 <= 4: return 5
        if suited: return 5
        return 7

    if r1 <= 2:  # Qx
        if r2 <= 3 and suited: return 3
        if r2 <= 3: return 4
        if r2 <= 5 and suited: return 4
        if suited: return 6
        return 7

    # Connected / suited
    if gap <= 1 and suited and r1 <= 5: return 4  # JTs, T9s, 98s
    if gap <= 1 and suited and r1 <= 8: return 5  # 87s, 76s, 65s, 54s
    if gap <= 1 and suited: return 6               # 43s, 32s — too weak even suited
    if gap <= 2 and suited and r1 <= 6: return 5   # J9s, T8s
    if gap <= 1 and r1 <= 4: return 5              # JTo, T9o
    if suited and gap <= 3: return 6
    if gap <= 1: return 6
    return 8


def get_rfi_strategy(position: str, hand: str) -> Dict[str, float]:
    """Get Raise First In strategy."""
    tier = hand_tier(hand)
    suited = hand.endswith('s')
    pair = len(hand) == 2 and hand[0] == hand[1]

    # Position-dependent opening ranges
    thresholds = {
        'UTG': {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.5, 5: 0.0},
        'HJ':  {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.3},
        'CO':  {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.3},
        'BTN': {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 0.3},
        # SB: HU vs BB so range is wider, but still fold the worst junk
        # Tier 5 (low pockets, marginal suiteds) = 80%, tier 6 (weak offsuit) = 30%
        # Tier 7 (very weak) = fold
        'SB':  {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.8, 6: 0.3, 7: 0.0},
    }

    pos_thresh = thresholds.get(position, thresholds['CO'])
    open_freq = pos_thresh.get(tier, 0.0)

    if open_freq <= 0:
        return {'fold': 1.0}
    elif open_freq >= 1.0:
        return {'raise': 1.0}
    else:
        return {'raise': round(open_freq, 2), 'fold': round(1 - open_freq, 2)}


def get_vs_raise_strategy(position: str, hand: str, raiser_pos: str,
                          raise_size: float = 3.0) -> Dict[str, float]:
    """Get strategy facing a raise (3bet or call or fold)."""
    tier = hand_tier(hand)
    suited = hand.endswith('s')
    pair = len(hand) == 2 and hand[0] == hand[1]

    # Tighter vs earlier position raises
    pos_order = {p: i for i, p in enumerate(POSITIONS)}
    raiser_tightness = pos_order.get(raiser_pos, 3)  # 0=UTG(tight) to 5=BB

    if tier <= 1:
        # Premium: always 3bet (with some trapping)
        if pair and hand[0] == 'A':
            return {'raise': 0.85, 'call': 0.15}
        return {'raise': 1.0}
    elif tier == 2:
        if raiser_tightness <= 1:  # vs UTG/HJ
            return {'raise': 0.4, 'call': 0.6}
        return {'raise': 0.6, 'call': 0.4}
    elif tier == 3:
        if raiser_tightness <= 1:
            return {'call': 0.7, 'fold': 0.3}
        return {'call': 0.8, 'raise': 0.2}
    elif tier == 4:
        if raiser_tightness <= 2:
            return {'call': 0.4, 'fold': 0.6}
        return {'call': 0.7, 'fold': 0.3}
    elif tier == 5:
        if suited and pair:
            return {'call': 0.6, 'fold': 0.4}
        if raiser_tightness >= 3:
            return {'call': 0.5, 'fold': 0.5}
        return {'fold': 0.8, 'call': 0.2}
    elif tier == 6:
        # Some suited hands can bluff 3-bet vs late position opens
        # Use a deterministic mixed frequency instead of random.random()
        if suited and raiser_tightness >= 3:  # vs CO/BTN/SB only
            return {'raise': 0.15, 'fold': 0.85}
        return {'fold': 1.0}
    else:
        return {'fold': 1.0}


def get_vs_3bet_strategy(position: str, hand: str, threebettor_pos: str) -> Dict[str, float]:
    """Get strategy when facing a 3bet."""
    tier = hand_tier(hand)

    if tier <= 1:
        # Premium: 4bet or call
        if hand in ('AA', 'KK'):
            return {'raise': 0.6, 'call': 0.4}
        return {'raise': 0.4, 'call': 0.6}
    elif tier == 2:
        return {'call': 0.7, 'fold': 0.3}
    elif tier == 3:
        return {'call': 0.4, 'fold': 0.6}
    elif tier == 4:
        if hand.endswith('s'):
            return {'call': 0.3, 'fold': 0.7}
        return {'fold': 0.9, 'call': 0.1}
    else:
        return {'fold': 1.0}


def get_bb_defense_strategy(hand: str, raiser_pos: str,
                            raise_size: float = 2.5) -> Dict[str, float]:
    """BB facing a raise - wider defense range."""
    tier = hand_tier(hand)
    pos_order = {p: i for i, p in enumerate(POSITIONS)}
    raiser_tightness = pos_order.get(raiser_pos, 3)

    if tier <= 1:
        return {'raise': 0.7, 'call': 0.3}
    elif tier == 2:
        return {'raise': 0.35, 'call': 0.65}
    elif tier == 3:
        # vs tight EP/HJ: mostly call, rarely 3-bet
        # vs loose BTN/SB: mix more 3-bets in
        if raiser_tightness >= 3:  # vs CO/BTN/SB: wider opening range → 3-bet more
            return {'raise': 0.35, 'call': 0.65}
        return {'call': 0.8, 'raise': 0.2}  # vs UTG/HJ: respect the tighter range
    elif tier == 4:
        if raiser_tightness >= 3:  # vs late position
            return {'call': 0.9, 'raise': 0.1}
        return {'call': 0.6, 'fold': 0.4}
    elif tier == 5:
        if raiser_tightness >= 3:
            return {'call': 0.7, 'fold': 0.3}
        return {'call': 0.4, 'fold': 0.6}
    elif tier == 6:
        if raiser_tightness >= 4:  # vs BTN/SB
            return {'call': 0.5, 'fold': 0.5}
        return {'fold': 0.8, 'call': 0.2}
    else:
        return {'fold': 1.0}


# ============================================================
# Postflop Hand Advisor
# ============================================================

@dataclass
class BoardTexture:
    """Analyze board texture for strategy decisions."""
    high_card: int = 0      # Highest card rank
    is_monotone: bool = False    # 3+ same suit
    is_two_tone: bool = False    # 2 same suit
    is_rainbow: bool = False     # All different suits
    is_paired: bool = False      # Board has a pair
    is_connected: bool = False   # Consecutive cards
    has_straight_draw: bool = False
    has_flush_draw: bool = False
    wetness: float = 0.0    # 0=dry, 1=very wet


def analyze_board(board_ranks: List[int], board_suits: List[int]) -> BoardTexture:
    """Analyze board texture."""
    tex = BoardTexture()

    if not board_ranks:
        return tex

    tex.high_card = max(board_ranks)

    # Suit analysis
    suit_counts = {}
    for s in board_suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    max_suit = max(suit_counts.values()) if suit_counts else 0

    tex.is_monotone = max_suit >= 3
    tex.is_two_tone = max_suit == 2
    tex.is_rainbow = max_suit == 1

    # Pair check
    rank_counts = {}
    for r in board_ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    tex.is_paired = max(rank_counts.values()) >= 2

    # Connectivity: require a cluster of 3+ consecutive ranks within a 5-card window
    sorted_ranks = sorted(set(board_ranks))
    gaps = [sorted_ranks[i+1] - sorted_ranks[i] for i in range(len(sorted_ranks)-1)]
    # Count consecutive clusters (streaks of gap=1)
    consec_count = 1  # current streak length
    max_consec = 1
    for g in gaps:
        if g <= 1:
            consec_count += 1
            max_consec = max(max_consec, consec_count)
        else:
            consec_count = 1
    tex.is_connected = max_consec >= 3  # 3+ consecutive ranks (e.g., 7-8-9)
    # Straight draw: 3+ cards within a 5-rank window (opponent could have a draw)
    tex.has_straight_draw = False
    for i in range(len(sorted_ranks)):
        window_low = sorted_ranks[i]
        in_window = sum(1 for r in sorted_ranks if window_low <= r <= window_low + 4)
        if in_window >= 3:
            tex.has_straight_draw = True
            break

    # Flush draw potential
    tex.has_flush_draw = max_suit >= 2

    # Wetness score
    wetness = 0
    if tex.is_monotone: wetness += 0.4
    elif tex.is_two_tone: wetness += 0.2
    if tex.is_connected: wetness += 0.3
    if tex.has_straight_draw: wetness += 0.2
    if tex.is_paired: wetness -= 0.1
    tex.wetness = min(1.0, max(0.0, wetness))

    return tex


@dataclass
class HandStrength:
    """Categorize hand strength on a board."""
    category: str = ""          # "nuts", "strong", "medium", "weak", "air"
    made_hand: str = ""         # "top_pair", "overpair", "two_pair", "set", etc.
    draw: str = ""              # "flush_draw", "straight_draw", "combo_draw", ""
    draw_outs: int = 0          # Number of draw outs
    relative_strength: float = 0.0  # 0-1 scale


def classify_hand_strength(hand: str, board_str: str) -> HandStrength:
    """
    Classify hand strength relative to board.

    This is a heuristic classification used for strategy decisions.
    """
    from .cards import parse_cards
    from .evaluator import evaluate_hand, get_hand_category

    hs = HandStrength()

    if not board_str.strip():
        # Preflop
        tier = hand_tier(hand if len(hand) <= 3 else
                        canonicalize_from_cards(hand))
        hs.relative_strength = max(0, 1.0 - (tier - 1) * 0.12)
        hs.category = "premium" if tier <= 2 else "strong" if tier <= 3 else "medium" if tier <= 5 else "weak"
        return hs

    try:
        # Hand may come as "AhKd" (no space) or "Ah Kd"
        if len(hand) == 4 and ' ' not in hand:
            hand_cards = parse_cards(hand[:2] + ' ' + hand[2:])
        else:
            hand_cards = parse_cards(hand)
        board_cards = parse_cards(board_str)
    except (ValueError, IndexError):
        hs.category = "unknown"
        return hs

    if len(hand_cards) != 2 or len(board_cards) < 3:
        hs.category = "unknown"
        return hs

    score = evaluate_hand(hand_cards, board_cards)
    cat = get_hand_category(score)

    hr1, hr2 = hand_cards[0].rank, hand_cards[1].rank
    hs1, hs2 = hand_cards[0].suit, hand_cards[1].suit
    board_ranks = [c.rank for c in board_cards]
    board_suits = [c.suit for c in board_cards]
    top_board = max(board_ranks)

    # Made hand classification
    # First check if board is paired (affects two pair / full house evaluation)
    board_rank_counts = {}
    for br in board_ranks:
        board_rank_counts[br] = board_rank_counts.get(br, 0) + 1
    board_is_paired = max(board_rank_counts.values()) >= 2 if board_rank_counts else False

    if cat == "Straight Flush":
        hs.made_hand = "straight_flush"
        hs.category = "nuts"
        hs.relative_strength = 1.0
    elif cat == "Four of a Kind":
        hs.made_hand = "quads"
        hs.category = "nuts"
        hs.relative_strength = 0.99
    elif cat == "Full House":
        # Differentiate full house quality by trip rank vs pair rank.
        # Top full house (trips > pair, e.g. KKK-QQ) is much safer than
        # bottom full house (trips < pair, e.g. QQQ-KK where villain can have KKK-QQ).
        all_ranks_sorted = sorted([c.rank for c in hand_cards + board_cards], reverse=True)
        rank_counter: Dict[int, int] = {}
        for r in all_ranks_sorted:
            rank_counter[r] = rank_counter.get(r, 0) + 1
        trip_ranks = [r for r, cnt in rank_counter.items() if cnt >= 3]
        pair_ranks = [r for r, cnt in rank_counter.items() if cnt == 2]
        trip_rank = max(trip_ranks) if trip_ranks else 0
        pair_rank = max(pair_ranks) if pair_ranks else 0
        if trip_rank >= pair_rank:
            hs.made_hand = "full_house"        # top full: trips ≥ pair rank
            hs.relative_strength = 0.95
        else:
            hs.made_hand = "bottom_full_house"  # vulnerable: trips < pair rank
            hs.relative_strength = 0.87         # e.g. QQQ-KK where villain may have KKK
        hs.category = "nuts"
    elif cat == "Flush":
        # Determine flush suit, then check hero's contribution in that suit
        all_suit_counts: Dict[int, int] = {}
        for c in hand_cards + board_cards:
            all_suit_counts[c.suit] = all_suit_counts.get(c.suit, 0) + 1
        flush_suit = max(all_suit_counts, key=all_suit_counts.get)
        hero_flush_ranks = [c.rank for c in hand_cards if c.suit == flush_suit]
        if hero_flush_ranks:
            # Hero contributes at least one flush-suit card
            flush_high = max(hero_flush_ranks)
        else:
            # Hero plays the board flush — their best card is the top board flush card
            board_flush_ranks = [c.rank for c in board_cards if c.suit == flush_suit]
            flush_high = max(board_flush_ranks) if board_flush_ranks else 0
        if flush_high >= 14:
            hs.made_hand = "nut_flush"
            hs.category = "nuts"
            hs.relative_strength = 0.95
        elif flush_high >= 12:
            hs.made_hand = "flush"
            hs.category = "nuts"
            hs.relative_strength = 0.90
        else:
            hs.made_hand = "weak_flush"
            hs.category = "strong"
            hs.relative_strength = 0.82
    elif cat == "Straight":
        # Straight quality: nut straight (highest possible) >> bottom straight
        all_ranks = sorted(set([c.rank for c in hand_cards + board_cards]), reverse=True)
        # Find the highest 5-card straight that uses at least one hero card
        straight_high = 0
        hero_r = {c.rank for c in hand_cards}
        if 14 in hero_r:
            hero_r.add(1)
        all_r = set(all_ranks)
        if 14 in all_r:
            all_r.add(1)
        for bottom in range(10, 0, -1):
            window = set(range(bottom, bottom + 5))
            if window <= all_r and window & hero_r:
                straight_high = bottom + 4
                break
        # Scale: broadway(14)=0.88, middle(10)=0.82, bottom(5-6)=0.76, wheel(5)=0.72
        if straight_high >= 14:
            hs.made_hand = "nut_straight"
            hs.relative_strength = 0.88
        elif straight_high >= 11:
            hs.made_hand = "straight"
            hs.relative_strength = 0.84
        elif straight_high >= 9:
            hs.made_hand = "straight"
            hs.relative_strength = 0.80
        else:
            hs.made_hand = "weak_straight"
            hs.relative_strength = 0.76
        hs.category = "strong"
        # Penalty: straight on monotone/two-tone board is vulnerable to flush
        tex_temp = analyze_board(board_ranks, board_suits)
        if tex_temp.is_monotone:
            hs.relative_strength = max(0.60, hs.relative_strength - 0.12)
        elif tex_temp.is_two_tone:
            hs.relative_strength = max(0.65, hs.relative_strength - 0.05)
    elif cat == "Three of a Kind":
        if hr1 == hr2:
            # Differentiate top set vs middle/bottom set by how many overcards exist on board
            # Top set (e.g. KK on K-Q-7): 0 overcards → safest, rs=0.94
            # Middle set (e.g. QQ on K-Q-7): 1 overcard → slightly vulnerable, rs=0.91
            # Bottom set (e.g. 77 on K-Q-7): 2+ overcards → more vulnerable, rs=0.88
            overcards_to_set = sum(1 for br in board_ranks if br > hr1)
            if overcards_to_set == 0:
                hs.made_hand = "top_set"
                hs.relative_strength = 0.94
            elif overcards_to_set == 1:
                hs.made_hand = "middle_set"
                hs.relative_strength = 0.91
            else:
                hs.made_hand = "bottom_set"
                hs.relative_strength = 0.88
            hs.category = "nuts"  # all sets are near-nuts
        else:
            # Trips (board has the three-of-a-kind): kicker matters significantly
            kicker = max(hr1, hr2)  # best kicker card
            kicker_bonus = (kicker - 2) / 12.0 * 0.10  # 0.0 (2-kicker) to 0.10 (A-kicker)
            hs.made_hand = "trips"
            hs.category = "strong"
            hs.relative_strength = round(0.68 + kicker_bonus, 2)  # 0.68 (2-kicker) → 0.78 (A-kicker)
    elif cat == "Two Pair":
        # Check if one of the pairs comes from a board pair (not our hand)
        hero_paired_with_board = any(
            (hr1 == br or hr2 == br) for br in board_ranks
        )
        if board_is_paired and not hero_paired_with_board:
            # e.g. KK on JJ-7 — evaluator gives "Two Pair" (board pair JJ + hero's KK)
            # but strategically this is still an overpair / underpair
            if hr1 == hr2 and hr1 > top_board:
                # Genuine overpair — board pair is lower than hero's pair
                # rs is slightly lower than vs unpaired board (board pair = villain draw target)
                # but category should still be "strong"
                hs.made_hand = "overpair"
                hs.category = "strong"
                hs.relative_strength = 0.65  # was 0.55 — too low; KK on JJ7 is still strong
            elif hr1 == hr2:
                hs.made_hand = "underpair"
                hs.category = "weak"
                hs.relative_strength = 0.25
            else:
                hs.made_hand = "high_card_on_paired"
                hs.category = "weak"
                hs.relative_strength = 0.2
        elif hero_paired_with_board and hr1 != hr2:
            # Real two pair — rank quality by which board ranks hero pairs
            paired_with = sorted(
                [r for r in [hr1, hr2] if r in board_ranks], reverse=True
            )
            sorted_board = sorted(board_ranks, reverse=True)
            hs.category = "strong"
            # Deduplicate board ranks for position lookup (paired boards have repeated ranks)
            unique_board = sorted(set(board_ranks), reverse=True)
            if len(paired_with) >= 2:
                # Hero pairs two different board ranks
                pos0 = unique_board.index(paired_with[0]) if paired_with[0] in unique_board else 99
                pos1 = unique_board.index(paired_with[1]) if paired_with[1] in unique_board else 99
                if pos0 == 0 and pos1 == 1:
                    # Hero pairs the top TWO unique board ranks (e.g., AK on AK7)
                    hs.made_hand = "top_two_pair"
                    hs.relative_strength = 0.78
                elif pos0 == 0 or pos0 == 1:
                    # Hero pairs the top OR second board rank (+ lower) → top_bottom
                    hs.made_hand = "top_bottom_two_pair"
                    hs.relative_strength = 0.70
                else:
                    hs.made_hand = "bottom_two_pair"
                    hs.relative_strength = 0.63
            elif paired_with:
                # Only one hole card pairs the board (shouldn't reach Two Pair this way, but guard)
                pos0 = unique_board.index(paired_with[0]) if paired_with[0] in unique_board else 99
                if pos0 <= 1:
                    hs.made_hand = "top_bottom_two_pair"
                    hs.relative_strength = 0.70
                else:
                    hs.made_hand = "bottom_two_pair"
                    hs.relative_strength = 0.63
            else:
                hs.made_hand = "two_pair"
                hs.relative_strength = 0.65
        else:
            # Fallback: board two pair, hero's kicker plays
            hs.made_hand = "two_pair"
            hs.category = "strong"
            hs.relative_strength = 0.65
    elif cat == "One Pair":
        if hr1 == hr2 and hr1 > top_board:
            hs.made_hand = "overpair"
            hs.category = "strong"
            hs.relative_strength = 0.7
        elif hr1 == hr2:
            # Pocket pair that does NOT beat the board (overpair handled above).
            # Here the board has at least one overcard to our pair.
            overcards = sum(1 for br in board_ranks if br > hr1)
            undercards = sum(1 for br in board_ranks if br < hr1)
            close_overcard = overcards == 1 and top_board - hr1 <= 2 and top_board != 14
            if close_overcard and undercards >= 1:
                # One close overcard (e.g. 88 on T-6-3 where T is only 2 ranks above 8)
                hs.made_hand = "pocket_pair_mid"
                hs.category = "weak"
                hs.relative_strength = 0.32
            else:
                # Multiple overcards, Ace on board, or high broadway overcard
                hs.made_hand = "underpair"
                hs.category = "weak"
                hs.relative_strength = 0.25
        else:
            # Unpaired hero cards — find which hero card paired the board and where it sits.
            # Note: one hero card may be ABOVE the top board card (acts as kicker).
            paired_with = [r for r in [hr1, hr2] if r in board_ranks]
            if paired_with:
                paired_rank = max(paired_with)
                # The kicker is the hero card that did NOT pair the board
                kicker_rank = hr2 if paired_rank == hr1 else hr1
                sorted_board = sorted(board_ranks, reverse=True)
                if paired_rank == sorted_board[0]:
                    # Paired the TOP board card — top pair; classify by kicker strength.
                    # top_pair_top_kicker: kicker beats top board card (e.g., K+A on K-board,
                    #   or A+K on A-board).  Threshold = paired_rank itself (strictly better).
                    # top_pair_good_kicker: kicker is T–Q (strong broadway, does not beat top card).
                    # top_pair_weak_kicker: kicker below T.
                    # Special case: pairing an Ace — no kicker can exceed it, so
                    # top_pair_top_kicker requires kicker >= K (13).
                    # For non-Ace top pairs: top_pair_top_kicker requires kicker > paired_rank
                    # (i.e., an Ace overcard — the only card that beats the pair).
                    is_tptk = (paired_rank == 14 and kicker_rank >= 13) or \
                               (paired_rank < 14 and kicker_rank > paired_rank)
                    if is_tptk:
                        # Kicker is the highest possible → true TPTK
                        hs.made_hand = "top_pair_top_kicker"
                        hs.category = "strong"
                        hs.relative_strength = 0.68
                    elif kicker_rank >= 10:
                        hs.made_hand = "top_pair_good_kicker"
                        hs.category = "medium"
                        # Scale within the bucket: T(0.55), J(0.57), Q(0.59)
                        hs.relative_strength = round(0.53 + (kicker_rank - 10) * 0.02, 2)
                    else:
                        hs.made_hand = "top_pair_weak_kicker"
                        hs.category = "medium"
                        hs.relative_strength = 0.48
                elif len(sorted_board) > 1 and paired_rank == sorted_board[1]:
                    hs.made_hand = "second_pair"
                    hs.category = "medium"
                    hs.relative_strength = 0.4
                else:
                    hs.made_hand = "bottom_pair"
                    hs.category = "weak"
                    hs.relative_strength = 0.28
            else:
                hs.made_hand = "bottom_pair"
                hs.category = "weak"
                hs.relative_strength = 0.28
    else:
        # High card
        if max(hr1, hr2) >= 14:
            hs.made_hand = "ace_high"
            hs.category = "weak"
            hs.relative_strength = 0.2
        else:
            hs.made_hand = "high_card"
            hs.category = "air"
            hs.relative_strength = 0.1

    # Check for draws — only meaningful when there are cards still to come.
    # On the river (5 board cards), draws are resolved; skip draw detection entirely.
    # Also skip for already-strong made hands (flush, full house, quads, straight flush)
    # to avoid spurious draw annotations (e.g., broadway OESD on a made flush board).
    _strong_made = {"nut_flush", "flush", "weak_flush", "full_house", "quads", "straight_flush"}
    if len(board_cards) < 5 and hs.made_hand not in _strong_made:
        _check_draws(hs, hand_cards, board_cards)

        # Draws boost relative strength
        if hs.draw_outs >= 12:
            hs.category = "strong" if hs.category in ("air", "weak") else hs.category
            hs.relative_strength = max(hs.relative_strength, 0.65)
        elif hs.draw_outs >= 8:
            if hs.category in ("air", "weak"):
                hs.category = "medium"
            hs.relative_strength = max(hs.relative_strength, 0.5)
        elif hs.draw_outs >= 4:
            hs.relative_strength = max(hs.relative_strength, 0.35)

    return hs


def _check_draws(hs: HandStrength, hand_cards, board_cards):
    """Check for flush and straight draws."""
    all_cards = hand_cards + board_cards
    hand_ranks = set(c.rank for c in hand_cards)

    # ---- Flush draw ----
    suit_counts = {}
    for c in all_cards:
        suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1

    for suit, count in suit_counts.items():
        hero_has = any(c.suit == suit for c in hand_cards)
        if count == 4 and hero_has:
            hs.draw = "flush_draw"
            hs.draw_outs = 9
            # Nut flush draw?
            hero_suit_ranks = [c.rank for c in hand_cards if c.suit == suit]
            if 14 in hero_suit_ranks:
                hs.draw = "nut_flush_draw"
            break
        elif count >= 5 and hero_has:
            pass  # Already have a flush (made hand)

    # ---- Straight draw (check ALL possible straights) ----
    all_ranks_set = set(c.rank for c in all_cards)
    # Also add 1 as alias for Ace for wheel draws
    if 14 in all_ranks_set:
        all_ranks_set.add(1)
    hero_ranks_set = set(c.rank for c in hand_cards)
    if 14 in hero_ranks_set:
        hero_ranks_set.add(1)

    best_draw = None  # (outs, type)
    oesd_found = False
    gutshot_found = False

    for bottom in range(1, 11):  # bottom of 5-card straight: A(1) through T(10)
        straight = set(range(bottom, bottom + 5))
        have = straight & all_ranks_set
        miss = straight - all_ranks_set
        hero_contributes = bool(straight & hero_ranks_set)

        if len(miss) == 1 and hero_contributes:
            missing_rank = list(miss)[0]
            # OESD: missing card is at either end AND that end is reachable
            # (i.e., there's actually a card at that rank that can complete the draw).
            # Special case: broadway draw (T-J-Q-K-A, missing T at bottom=10) has no
            # card below T that extends to another straight via this window, so it is
            # a ONE-SIDED draw (gutshot, 4 outs), not an OESD.
            # Rule: the missing end is "open" only if the OTHER three consecutive cards
            # in the window are interior (i.e., have valid cards on both sides).
            # Simpler: OESD if the 4 present cards span exactly 4 consecutive ranks
            # with the missing card at one end AND the window doesn't touch rank 14
            # at the bottom OR rank 1 at the top (which would make it one-sided).
            is_oesd = False
            if missing_rank == bottom:
                # Missing the low end: valid OESD only if the window doesn't use Ace=14
                # as its high card (A-high straight like T-J-Q-K-A has no lower card to extend)
                is_oesd = (bottom + 4) < 14  # top end is not an Ace (rank 14)
            elif missing_rank == bottom + 4:
                # Missing the high end: valid OESD only if bottom > 1
                # (wheel A-2-3-4-5: if missing A at top, A can also alias as 1 — gutshot only)
                is_oesd = bottom > 1
            if is_oesd:
                if not oesd_found:
                    oesd_found = True
                    if best_draw is None or best_draw[0] < 8:
                        best_draw = (8, "oesd")
            else:
                # Gutshot (one-sided draw or inner miss)
                if not gutshot_found:
                    gutshot_found = True
                    if best_draw is None or best_draw[0] < 4:
                        best_draw = (4, "gutshot")

        # Double gutshot / wrap: check for 2-gap with hero contributing
        if len(miss) == 2 and hero_contributes and len(have) == 3:
            # Check if hero has both needed cards for a double-belly-buster
            pass

    if best_draw:
        outs, dtype = best_draw
        if hs.draw in ("flush_draw", "nut_flush_draw"):
            # Combo draw: flush + straight
            hs.draw = "combo_draw"
            hs.draw_outs += outs - 2  # subtract ~2 for straight-flush overlap outs
        elif hs.draw == "combo_draw":
            pass  # already combo
        else:
            hs.draw = "oesd" if dtype == "oesd" else "gutshot"
            hs.draw_outs += outs

    # ---- Backdoor draws (only add if no primary draw already found) ----
    if not hs.draw:
        # Backdoor flush: exactly 3 of one suit total, hero has ≥1 of them
        for suit, count in suit_counts.items():
            if count == 3 and any(c.suit == suit for c in hand_cards):
                hs.draw = "backdoor_flush"
                hs.draw_outs = 2  # worth ~1.5 extra outs
                break

    if not hs.draw:
        # Backdoor straight: 3 of 5 consecutive ranks present, hero contributes
        for bottom in range(1, 11):
            straight = set(range(bottom, bottom + 5))
            have = straight & all_ranks_set
            hero_contributes = bool(straight & hero_ranks_set)
            if len(have) == 3 and hero_contributes:
                hs.draw = "backdoor_straight"
                hs.draw_outs = 1
                break


def canonicalize_from_cards(hand_str: str) -> str:
    """Convert 'AhKd' to 'AKo' canonical form."""
    if len(hand_str) != 4:
        return hand_str
    r1, s1 = hand_str[0], hand_str[1]
    r2, s2 = hand_str[2], hand_str[3]

    ri1 = RANKS.index(r1.upper())
    ri2 = RANKS.index(r2.upper())
    suited = s1.lower() == s2.lower()

    if ri1 == ri2:
        return f"{r1.upper()}{r2.upper()}"
    elif ri1 < ri2:
        return f"{r1.upper()}{r2.upper()}{'s' if suited else 'o'}"
    else:
        return f"{r2.upper()}{r1.upper()}{'s' if suited else 'o'}"


# ============================================================
# Postflop strategy advisor
# ============================================================

def _detect_turn_completes(board_cards) -> str:
    """Detect if the turn card completed a flush or straight draw from the flop."""
    if len(board_cards) < 4:
        return ""

    flop_cards = board_cards[:3]
    turn_card = board_cards[3]

    flop_suits = [c.suit for c in flop_cards]
    flop_ranks = [c.rank for c in flop_cards]

    # Flush completed: turn suit appeared twice on flop
    if flop_suits.count(turn_card.suit) >= 2:
        return "flush_completed"

    # Straight completed: the turn card makes a 4-to-a-straight on board
    # (villain needs just one hole card to complete). This is a "scare card" scenario.
    flop_ranks_set = set(flop_ranks)
    if 14 in flop_ranks_set:
        flop_ranks_set.add(1)
    all_ranks_set = flop_ranks_set | {turn_card.rank}
    if turn_card.rank == 14:
        all_ranks_set.add(1)

    for bottom in range(1, 11):
        straight = set(range(bottom, bottom + 5))
        present = len(straight & all_ranks_set)
        flop_present = len(straight & flop_ranks_set)
        # 4 of 5 ranks are on board (straight possible with 1 hole card)
        # AND the turn card contributed a new rank (wasn't already 4/5 on flop)
        if present >= 4 and flop_present < 4:
            return "straight_completed"

    return ""


def get_postflop_strategy(
    hand: str,
    board: str,
    position: str,        # "OOP" or "IP"
    action_sequence: List[str],  # e.g. ["check", "bet_50"]
    pot_size: float = 10.0,
    stack_size: float = 100.0,
    street: str = "flop",  # "flop", "turn", "river"
    num_villains: int = 1,  # 1=HU, 2+=multiway
    hero_position: str = "",  # actual position name like "CO"
) -> Dict:
    """
    Get postflop GTO strategy recommendation.

    Returns dict with:
    - recommended_actions: {action: frequency}
    - reasoning: str
    - ev_estimates: {action: estimated_ev}
    - bet_sizes: available bet sizing options
    """
    hs = classify_hand_strength(hand, board)

    from .cards import parse_cards
    board_cards = parse_cards(board) if board.strip() else []
    board_ranks = [c.rank for c in board_cards]
    board_suits = [c.suit for c in board_cards]
    tex = analyze_board(board_ranks, board_suits)

    spr = stack_size / pot_size if pot_size > 0 else 10.0  # stack to pot ratio

    # Determine what we're facing — ONLY look at current street's actions
    # Preflop actions (raises/calls/folds before board) should be ignored
    facing_bet = False
    bet_fraction = 0.0
    num_bets_this_street = 0

    # Split actions into preflop and postflop.
    # Postflop starts at the first check/bet_ action.
    # If none found, all actions are preflop (postflop_start = end of list).
    postflop_start = len(action_sequence)  # default: all preflop
    for idx, raw_action in enumerate(action_sequence):
        action = raw_action.split(":")[-1].strip() if ":" in raw_action else raw_action
        if action.startswith("check") or action.startswith("bet_"):
            postflop_start = idx
            break
    postflop_actions = action_sequence[postflop_start:]

    # Track whether a check or call preceded a bet (needed to detect donk bets).
    # A donk bet is when hero leads into villain without a prior check from villain.
    # If the FIRST postflop action is a bet_ with no preceding check/call, it's a
    # donk bet by hero → facing_bet stays False (route to first-action strategy).
    had_check_or_call = False
    for raw_action in postflop_actions:
        action = raw_action.split(":")[-1].strip() if ":" in raw_action else raw_action
        if action.startswith("bet_") or action.startswith("raise_"):
            # Only set facing_bet if there was a prior check/call this street
            # (meaning villain acted first, then bet). A lone leading bet is a donk bet.
            if had_check_or_call:
                facing_bet = True
                try:
                    parts = action.replace("raise_", "").replace("bet_", "")
                    if parts:
                        bet_fraction = float(parts) / 100.0
                    else:
                        bet_fraction = 0.5
                except (ValueError, IndexError):
                    bet_fraction = 0.5
                num_bets_this_street += 1
        elif action == "check":
            facing_bet = False
            had_check_or_call = True
        elif action == "call":
            facing_bet = False
            had_check_or_call = True

    # Detect 3-bet pot (≥2 preflop raises before postflop actions start)
    preflop_raises = sum(
        1 for a in action_sequence[:postflop_start]
        if 'raise' in a.split(':')[-1].strip()
    )
    is_3bet_pot = preflop_raises >= 2

    # Detect turn completing a draw (texture change awareness)
    turn_completed = _detect_turn_completes(board_cards) if street == 'turn' else ""
    if turn_completed:
        # Scare card hit: increase wetness so strategy accounts for the changed board
        tex.wetness = min(1.0, tex.wetness + 0.25)

    # Build strategy based on hand strength + position + texture
    result = {
        "hand_analysis": {
            "made_hand": hs.made_hand,
            "draw": hs.draw,
            "draw_outs": hs.draw_outs,
            "category": hs.category,
            "relative_strength": round(hs.relative_strength, 2),
        },
        "board_analysis": {
            "wetness": round(tex.wetness, 2),
            "is_monotone": tex.is_monotone,
            "is_paired": tex.is_paired,
            "is_connected": tex.is_connected,
            "texture": "wet" if tex.wetness > 0.5 else "dry" if tex.wetness < 0.2 else "medium",
            "turn_completed": turn_completed,
        },
        "spr": round(spr, 1),
        "is_3bet_pot": is_3bet_pot,
        "facing_bet": facing_bet,
        "bet_fraction": round(bet_fraction, 2),
    }

    if facing_bet:
        strategy, reasoning = _facing_bet_strategy(hs, tex, position, bet_fraction, spr, street, is_3bet_pot)
    elif position == "OOP":
        strategy, reasoning = _oop_first_action(hs, tex, spr, street, is_3bet_pot)
    else:
        strategy, reasoning = _ip_first_action(hs, tex, spr, street, action_sequence, is_3bet_pot)

    # ---- Multi-way adjustments ----
    if num_villains >= 2:
        strategy, reasoning = _apply_multiway_adjustments(
            strategy, reasoning, hs, tex, num_villains, facing_bet, street
        )

    result["recommended_actions"] = strategy
    result["reasoning"] = reasoning
    result["available_bet_sizes"] = _get_bet_sizes(pot_size, stack_size, street)
    result["num_villains"] = num_villains

    # ---- Deep GTO Reasoning ----
    from .reasoning import build_dynamic_reasoning
    gto_reasoning = build_dynamic_reasoning(
        hero_hand_str=hand,
        board_str=board,
        recommended_actions=strategy,
        pot_size=pot_size,
        stack_size=stack_size,
        street=street,
        num_villains=num_villains,
        facing_bet=facing_bet,
        bet_fraction=bet_fraction,
        action_history=action_sequence,
        hero_position=hero_position if 'hero_position' in dir() else "",
    )
    result["gto_reasoning"] = gto_reasoning

    return result


def _apply_multiway_adjustments(
    strategy: Dict[str, float],
    reasoning: str,
    hs: HandStrength,
    tex: BoardTexture,
    num_villains: int,
    facing_bet: bool,
    street: str,
) -> Tuple[Dict[str, float], str]:
    """
    Adjust strategy for multi-way pots.

    Key multi-way principles from GTO theory:
    1. Bluff MUCH less (multiple opponents to call you down)
    2. Value bet more linearly (thin value goes up, fancy plays go down)
    3. Draws lose implied odds (harder to get paid from one specific player)
    4. Hand strength requirements go UP (need stronger hands to continue)
    5. Position matters even more
    6. Set-mining becomes more profitable (better implied odds from bigger pot)
    7. Pot odds improve (more dead money) but reverse implied odds increase
    """
    adjusted = dict(strategy)
    mw = num_villains  # shorthand

    # Multiplier: how much to tighten (higher = tighter)
    tighten_factor = 1.0 + (mw - 1) * 0.25  # 1.25 for 2 villains, 1.5 for 3, etc.

    if hs.category in ("nuts", "strong"):
        # Strong hands: bet MORE for value (multiple opponents to pay you off)
        # Reduce trapping, increase direct betting
        for k in list(adjusted.keys()):
            if k == "check":
                adjusted[k] *= max(0.3, 1.0 / tighten_factor)
            elif k.startswith("bet_") or k.startswith("raise_"):
                adjusted[k] = min(1.0, adjusted[k] * 1.15)

        reasoning = (
            f"[MULTIWAY {mw+1}-way] {hs.made_hand}: Strong hands gain value with "
            f"more opponents. Bet for value more often — less trapping in multi-way pots. "
            f"Multiple opponents increase chance someone has a calling hand."
        )

    elif hs.category == "medium":
        # Medium hands: much more cautious in multi-way
        # Reduce betting, increase checking/folding
        for k in list(adjusted.keys()):
            if k.startswith("bet_") or k.startswith("raise_"):
                adjusted[k] *= max(0.2, 1.0 / tighten_factor)
            if k == "check":
                adjusted[k] = min(1.0, adjusted[k] * tighten_factor)
            if k == "fold" and facing_bet:
                adjusted[k] = min(1.0, adjusted[k] * (1.0 + (mw - 1) * 0.15))

        reasoning = (
            f"[MULTIWAY {mw+1}-way] {hs.made_hand}: Medium hands decrease significantly "
            f"in multi-way pots. With {mw} opponents, someone likely has a better hand. "
            f"Play cautiously — check more and fold more to aggression."
        )

    elif hs.category == "weak":
        if hs.draw_outs >= 8:
            # Draws in multiway: better pot odds but can't bluff
            # Call more (better immediate odds), but don't raise (can't bluff)
            for k in list(adjusted.keys()):
                if k.startswith("raise_"):
                    adjusted[k] *= 0.3  # Almost never raise as semi-bluff
                if k == "call" and facing_bet:
                    adjusted[k] = min(1.0, adjusted[k] * 1.1)  # Better pot odds

            reasoning = (
                f"[MULTIWAY {mw+1}-way] {hs.draw} ({hs.draw_outs} outs): "
                f"Better pot odds in multi-way (more dead money). But do NOT semi-bluff "
                f"raise — you can't fold out {mw} opponents. Call if getting the right price."
            )
        else:
            # Weak hands without draws: fold much more
            for k in list(adjusted.keys()):
                if k == "fold":
                    adjusted[k] = min(1.0, adjusted.get(k, 0) + (mw - 1) * 0.15)
                elif k == "call":
                    adjusted[k] *= max(0.2, 1.0 / tighten_factor)
                elif k.startswith("bet_") or k.startswith("raise_"):
                    adjusted[k] *= 0.2

            reasoning = (
                f"[MULTIWAY {mw+1}-way] {hs.made_hand}: Weak hands are nearly unplayable "
                f"in multi-way pots. With {mw} opponents, fold equity drops to near zero. "
                f"Save your chips."
            )

    else:  # air
        if facing_bet:
            # When facing a bet with air in multiway: don't bluff-raise, but
            # still call at reduced frequency if base strategy had a call
            for k in list(adjusted.keys()):
                if k.startswith("raise_"):
                    adjusted[k] *= max(0.05, 0.15 / mw)
                if k == "call":
                    adjusted[k] *= max(0.4, 1.0 / tighten_factor)
                if k == "fold":
                    adjusted[k] = min(1.0, adjusted.get(k, 0) * 1.1)

            reasoning = (
                f"[MULTIWAY {mw+1}-way] Air facing bet: do not raise (bluff raises fail vs "
                f"{mw} opponents). Call only if getting the right price or with backdoor draws."
            )
        else:
            # Not facing a bet: NEVER bluff into multiway
            for k in list(adjusted.keys()):
                if k.startswith("bet_") or k.startswith("raise_"):
                    adjusted[k] *= max(0.05, 0.3 / mw)
                if k == "fold":
                    adjusted[k] = min(1.0, adjusted.get(k, 0) + (mw - 1) * 0.2)
                if k == "check":
                    adjusted[k] = min(1.0, adjusted.get(k, 0) * tighten_factor)

            reasoning = (
                f"[MULTIWAY {mw+1}-way] DO NOT BLUFF into {mw} opponents. "
                f"Multi-way bluffing is one of the biggest leaks in poker. "
                f"You need to fold out ALL opponents to win, and each one "
                f"reduces your fold equity multiplicatively."
            )

    # Normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: round(v / total, 4) for k, v in adjusted.items() if v > 0.005}
    # Re-normalize after filtering
    total = sum(adjusted.values())
    if total > 0 and abs(total - 1.0) > 0.01:
        adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}

    return adjusted, reasoning


def _get_bet_sizes(pot: float, stack: float, street: str) -> List[Dict]:
    """Get available bet sizes with descriptions."""
    sizes = []

    fractions = [
        (0.25, "1/4 pot", "Small probe bet"),
        (0.33, "1/3 pot", "Small bet / block bet"),
        (0.50, "1/2 pot", "Standard bet"),
        (0.67, "2/3 pot", "Medium bet"),
        (0.75, "3/4 pot", "Large bet"),
        (1.00, "Pot", "Pot-sized bet"),
        (1.25, "1.25x pot", "Overbet"),
        (1.50, "1.5x pot", "Large overbet"),
        (2.00, "2x pot", "Massive overbet"),
    ]

    for frac, label, desc in fractions:
        amount = round(pot * frac, 1)
        if amount <= stack and amount >= 1:
            sizes.append({
                "fraction": frac,
                "label": label,
                "description": desc,
                "amount": amount,
                "pct_label": f"{int(frac * 100)}%",
            })

    # Always include all-in
    if stack > 0:
        allin_fraction = round(stack / pot, 2) if pot > 0 else 999
        sizes.append({
            "fraction": allin_fraction,
            "label": "All-in",
            "description": f"All-in for {stack:.0f}",
            "amount": stack,
            "pct_label": "All-in",
        })

    return sizes


def _oop_first_action(hs: HandStrength, tex: BoardTexture,
                      spr: float, street: str,
                      is_3bet_pot: bool = False) -> Tuple[Dict, str]:
    """OOP strategy — math-driven by relative_strength."""
    s = hs.relative_strength  # 0-1
    w = tex.wetness            # 0-1
    d = hs.draw_outs           # 0-15

    # River: draws are resolved — treat draw contribution as zero
    if street == 'river':
        d = 0

    # Short stack: commit-or-fold, no fancy play
    if spr < 2:
        if s >= 0.55:
            return ({"bet_100": 1.0},
                    f"Short stack (SPR={spr:.1f}): {hs.made_hand} is strong — commit.")
        elif s >= 0.40 or hs.draw_outs >= 8:
            return ({"bet_75": 0.6, "check": 0.4},
                    f"Short stack (SPR={spr:.1f}): semi-commit with {hs.made_hand}.")
        else:
            return ({"check": 1.0},
                    f"Short stack (SPR={spr:.1f}): too weak to commit — check and give up.")

    # Base bet frequency scales with hand strength
    # Very strong (>0.85): bet often for value
    # Strong (0.65-0.85): bet on wet, check/bet on dry
    # Medium (0.4-0.65): mostly check (pot control)
    # Weak (<0.4): check, occasional bluff if draws
    # Air (<0.2): check, bluff with draws only
    # 3-bet pots: polarized ranges → larger default value sizing
    value_size = "75" if is_3bet_pot else ("67" if w > 0.4 else "50")

    if s >= 0.90:  # Nuts: flush, set, full house
        bet_big = 0.50 + w * 0.20   # 50-70% big bet on wet boards (OOP bets less than IP)
        if is_3bet_pot:
            bet_big = min(0.80, bet_big + 0.10)  # more aggression in 3-bet pots
        # Small blocking bet mixed in for balance (check-calling range protection)
        bet_small = 0.08
        check = max(0.05, 1 - bet_big - bet_small)
        strategy = {"bet_75": round(bet_big, 2), "bet_33": round(bet_small, 2), "check": round(check, 2)}
        reason = f"Very strong ({hs.made_hand}). Bet for value{', protect vs draws' if w > 0.3 else ', can slow-play on dry board'}.{' [3-bet pot: polarized sizing]' if is_3bet_pot else ''}"

    elif s >= 0.65:  # Strong: overpair, TPTK, two pair
        bet_freq = 0.3 + (s - 0.65) * 2.0 + w * 0.15  # 30-65%
        bet_freq = min(0.7, bet_freq)
        check = round(1 - bet_freq, 2)
        strategy = {f"bet_{value_size}": round(bet_freq, 2), "check": check}
        reason = f"Strong hand ({hs.made_hand}, strength {s:.0%}). {'Bet for value+protection on wet board' if w > 0.4 else 'Mix bet/check on dry board'}."

    elif s >= 0.40:  # Medium: TP weak kicker, second pair, middle pair
        bet_freq = max(0.1, 0.25 * (s - 0.4) / 0.25)  # 10-25%
        if d >= 8:
            bet_freq = 0.35  # semi-bluff with draw
        check = round(1 - bet_freq, 2)
        strategy = {"check": check, "bet_33": round(bet_freq, 2)}
        reason = f"Medium hand ({hs.made_hand}). Pot control — check most, thin bet sometimes."
        if d >= 8:
            reason = f"Medium hand + draw ({hs.draw}, {d} outs). Check-call or semi-bluff."

    elif s >= 0.20:  # Weak: bottom pair, underpair
        if d >= 8:
            strategy = {"check": 0.55, "bet_67": 0.45}
            reason = f"Weak made hand but strong draw ({hs.draw}, {d} outs). Semi-bluff."
        elif d >= 4:
            strategy = {"check": 0.80, "bet_50": 0.20}
            reason = f"Weak hand + gutshot ({d} outs). Mostly check, occasional semi-bluff."
        else:
            strategy = {"check": 0.92, "bet_33": 0.08}
            reason = f"Weak hand ({hs.made_hand}). Check and hope to see showdown."

    else:  # Air: no pair, no real draw
        if d >= 8:
            strategy = {"bet_67": 0.45, "check": 0.55}
            reason = f"Air but strong draw ({hs.draw}, {d} outs). Semi-bluff frequently."
        elif d >= 4:
            strategy = {"check": 0.70, "bet_50": 0.30}
            reason = f"Air with draw ({d} outs). Occasional semi-bluff."
        elif w < 0.3 and street == 'flop':
            strategy = {"check": 0.80, "bet_33": 0.20}
            reason = f"Air on dry board. Small bluff sometimes to balance range."
        elif street == 'river':
            strategy = {"check": 0.97, "bet_33": 0.03}
            reason = f"Air on river ({hs.made_hand}). Almost never bluff — no draws to represent."
        else:
            strategy = {"check": 0.95, "bet_33": 0.05}
            reason = f"Air ({hs.made_hand}). Give up most of the time."

    return strategy, reason


def _ip_first_action(hs: HandStrength, tex: BoardTexture,
                     spr: float, street: str,
                     action_seq: List[str],
                     is_3bet_pot: bool = False) -> Tuple[Dict, str]:
    """IP strategy (checked to) — math-driven."""
    s = hs.relative_strength
    w = tex.wetness
    d = hs.draw_outs

    # River: draws resolved
    if street == 'river':
        d = 0

    # Short stack: commit-or-fold
    if spr < 2:
        if s >= 0.55:
            return ({"bet_100": 1.0},
                    f"Short stack (SPR={spr:.1f}): {hs.made_hand} — commit for value.")
        elif s >= 0.40 or hs.draw_outs >= 8:
            return ({"bet_75": 0.6, "check": 0.4},
                    f"Short stack (SPR={spr:.1f}): semi-commit with {hs.made_hand}.")
        else:
            return ({"check": 1.0},
                    f"Short stack (SPR={spr:.1f}): too weak — check behind.")

    # 3-bet pots: use larger sizing for value bets
    value_size = "75" if is_3bet_pot else ("75" if w > 0.3 else "50")

    if s >= 0.90:
        bet_freq = 0.60 + w * 0.20
        if is_3bet_pot:
            bet_freq = min(0.90, bet_freq + 0.10)
        # River: IP nuts should never bet less than 75% — no slow-play value in position
        if street == 'river':
            bet_freq = max(bet_freq, 0.75)
        # Deep stack slow-play: at SPR>10, allow more checking to let villain catch up
        if spr > 10 and street != 'river':
            slow_play_bonus = min(0.20, (spr - 10) / 50)  # up to +20% check at SPR=20
            bet_freq = max(0.40, bet_freq - slow_play_bonus)
        check = max(0.05, round(1 - bet_freq, 2))
        strategy = {f"bet_{value_size}": round(bet_freq, 2), "check": check}
        spr_note = f" [deep SPR={spr:.0f}: slow-playing more]" if spr > 10 and street != 'river' else ""
        reason = f"Nuts in position ({hs.made_hand}). Bet for max value.{' [3-bet pot]' if is_3bet_pot else ''}{spr_note}"

    elif s >= 0.65:
        bet_freq = 0.50 + w * 0.15
        sizing = "75" if is_3bet_pot else ("67" if s > 0.75 else "50")
        strategy = {f"bet_{sizing}": round(bet_freq, 2), "check": round(1 - bet_freq, 2)}
        reason = f"Strong hand ({hs.made_hand}) in position. Bet for value."

    elif s >= 0.40:
        # On the river, medium hands lose most of their betting value: opponent ranges are
        # more defined, and betting often gets called only by better hands (reverse implied odds).
        if street == 'river':
            bet_freq = 0.10 + d * 0.01  # d is already 0 on river, kept for clarity
            strategy = {"check": round(1 - bet_freq, 2), "bet_33": round(bet_freq, 2)}
            reason = f"Medium hand ({hs.made_hand}) on river in position. Check behind — thin river value gets called by better hands."
        elif is_3bet_pot:
            # In 3-bet pots, ranges are more polarized and stack-to-pot ratios are lower.
            # Medium hands still have significant relative equity and should bet more aggressively.
            bet_freq = 0.45 + d * 0.02
            strategy = {"bet_75": round(min(0.6, bet_freq), 2), "check": round(max(0.4, 1 - bet_freq), 2)}
            reason = f"Medium hand ({hs.made_hand}) in 3-bet pot in position. Bet larger for value in polarized pot."
        else:
            bet_freq = 0.35 + d * 0.02  # draws add betting incentive
            strategy = {"bet_33": round(min(0.5, bet_freq), 2), "check": round(max(0.5, 1 - bet_freq), 2)}
            reason = f"Medium hand ({hs.made_hand}) in position. Thin value or check behind."

    elif s >= 0.20:
        if d >= 8:
            strategy = {"bet_67": 0.45, "check": 0.55}
            reason = f"Weak hand + draw ({hs.draw}, {d} outs) in position. Semi-bluff."
        else:
            strategy = {"check": 0.85, "bet_33": 0.15}
            reason = f"Weak hand ({hs.made_hand}). Mostly check behind for showdown value."

    else:
        if d >= 8:
            strategy = {"bet_67": 0.45, "check": 0.55}
            reason = f"Air with draw ({d} outs). Semi-bluff in position."
        elif d >= 4:
            strategy = {"bet_50": 0.30, "check": 0.70}
            reason = f"Air with backdoor draw. Occasional bluff."
        elif street == 'river':
            bet_bluff = max(0.10, 0.20 - w * 0.1)
            strategy = {"bet_50": round(bet_bluff, 2), "check": round(1 - bet_bluff, 2)}
            reason = f"Air on river. Bluff only {bet_bluff*100:.0f}% — no draws to balance against."
        else:
            bet_bluff = max(0.15, 0.35 - w * 0.3)
            strategy = {"bet_50": round(bet_bluff, 2), "check": round(1 - bet_bluff, 2)}
            reason = f"Air in position. Bluff {bet_bluff*100:.0f}% to balance range."

    return strategy, reason


def _facing_bet_strategy(hs: HandStrength, tex: BoardTexture,
                         position: str, bet_fraction: float,
                         spr: float, street: str,
                         is_3bet_pot: bool = False) -> Tuple[Dict, str]:
    """Strategy facing a bet — fully math-driven.

    Uses pot odds, MDF, relative strength, and draw outs
    to compute continuous call/raise/fold frequencies.
    """
    s = hs.relative_strength
    d = hs.draw_outs
    bf = max(bet_fraction, 0.01)

    # River: draws don't exist
    if street == 'river':
        d = 0

    # Short stack facing a bet: commit-or-fold, no passive calls
    if spr < 2:
        if s >= 0.45 or d >= 8:
            action_desc = "Check-raise all-in" if position == 'OOP' else "Raise all-in"
            return ({"raise_300": 1.0},
                    f"Short stack (SPR={spr:.1f}): {action_desc} — pot-committed with {hs.made_hand}.")
        else:
            return ({"fold": 1.0},
                    f"Short stack (SPR={spr:.1f}): fold — not worth committing with {hs.made_hand}.")

    # Core math
    pot_odds = bf / (1 + 2 * bf)             # equity needed to call
    mdf = 1.0 / (1 + bf)                     # minimum defense frequency
    ev_per_call = s - pot_odds               # simplified EV indicator

    # Draw equity on flop (2 cards to come) vs turn/river (1 card)
    draw_equity = min(0.6, d * 0.04) if street == 'flop' else min(0.45, d * 0.02)
    effective_equity = min(0.98, s + draw_equity * (1 - s) * 0.7)  # combine made+draw

    if effective_equity < 0.01:
        effective_equity = s

    # ---- Nuts/near-nuts (s >= 0.85): raise for value ----
    raise_key = "raise_300" if is_3bet_pot else "raise_250"
    if s >= 0.85:
        raise_freq = 0.4 + (s - 0.85) * 4.0  # 40-70%
        raise_freq = min(0.70, raise_freq)
        if position == 'OOP':
            # OOP must check-raise to protect range and deny free cards
            raise_freq = min(0.85, raise_freq + 0.15)
        if spr < 3:  # short stack → raise more (commit)
            raise_freq = min(0.90, raise_freq + 0.15)
        call_freq = round(1 - raise_freq, 2)
        strategy = {raise_key: round(raise_freq, 2), "call": call_freq}
        action_desc = "Check-raise" if position == 'OOP' else "Raise"
        pot_note = " [3-bet pot: larger sizing]" if is_3bet_pot else ""
        reason = f"Very strong ({hs.made_hand}, {s:.0%}). {action_desc} for value. SPR={spr:.1f}.{pot_note}"

    # ---- Strong (0.60-0.85): mostly call, sometimes raise ----
    # Guard: if rs was boosted by draws but made_hand is air, fall through to draw path.
    elif s >= 0.60 and not (hs.made_hand in ("high_card", "ace_high") and d >= 8):
        call_freq = 0.65
        raise_freq = max(0.05, (s - 0.60) * 1.5)  # 5-37%
        fold_freq = max(0, 1 - call_freq - raise_freq)
        # Bigger bets → more folds from medium hands
        if bf >= 1.0 and s < 0.75:
            fold_freq = max(fold_freq, 0.15)
            call_freq = 1 - raise_freq - fold_freq
        strategy = _norm({"call": call_freq, raise_key: raise_freq, "fold": fold_freq})
        reason = f"Strong hand ({hs.made_hand}, {s:.0%}) facing {bf*100:.0f}% pot. {'Raise/call' if raise_freq > 0.15 else 'Mostly call'}."

    # ---- Medium (0.35-0.60): call vs small, fold vs large ----
    # ---- Weak (s>=0.20 or d>=4): marginal defense ----
    # Merged into one smooth bucket to eliminate the cliff at the pot_odds boundary.
    elif effective_equity >= pot_odds or s >= 0.20 or d >= 4:
        if d >= 8 and effective_equity < pot_odds:
            # Good draw but below pot odds: semi-bluff raise option
            call_freq = 0.50
            fold_freq = 0.45
            raise_freq_bluff = 0.05
            strategy = _norm({"call": call_freq, "fold": fold_freq, raise_key: raise_freq_bluff})
            reason = f"Weak hand but {d}-out draw. Call for draw equity, occasional semi-bluff raise."
            return strategy, reason
        # Unified smooth formula: call_freq based on equity advantage/shortfall relative to pot odds.
        # equity_margin is positive when profitable (+EV), negative when -EV.
        equity_margin = effective_equity - pot_odds  # range: ~-0.4 to +0.4
        # Base call: scales from ~80% (strong +EV) down to ~5% (heavy -EV)
        call_freq = min(0.80, max(0.05, 0.40 + equity_margin * 2.5))
        # Additionally reduce for large bets (harder to make implied odds work)
        if bf > 0.75:
            call_freq = max(0.05, call_freq - (bf - 0.75) * 0.15)
        fold_freq = max(0.05, 1 - call_freq)
        raise_freq = 0
        if d >= 10 and s < 0.3:  # strong draw = sometimes raise
            raise_freq = 0.15
            fold_freq = max(0, fold_freq - 0.15)
        strategy = _norm({"call": call_freq, "fold": fold_freq, raise_key: raise_freq})
        if effective_equity >= pot_odds:
            reason = f"{hs.made_hand or 'Draw'} ({effective_equity:.0%} equity vs {pot_odds:.0%} needed). Profitable call."
            if d > 0:
                reason += f" Draw adds {d} outs."
        else:
            reason = f"{hs.made_hand} ({s:.0%}) below pot odds ({pot_odds:.0%}). {'Peel vs small bet' if bf <= 0.5 else 'Lean toward fold'}."

    # ---- Pure air ----
    else:
        if d >= 8:
            # Strong draw (e.g. combo draw)
            call_freq = 0.55
            raise_freq = 0.10
            fold_freq = 0.35
            strategy = _norm({"call": call_freq, raise_key: raise_freq, "fold": fold_freq})
            reason = f"Air but {d}-out draw ({hs.draw}). Enough draw equity to continue."
        elif d >= 4 and bf <= 0.5:
            strategy = {"call": 0.30, "fold": 0.70}
            reason = f"Air with gutshot ({d} outs) vs small bet. Peel one street."
        else:
            bluff_raise = max(0, 0.08 - bf * 0.05)
            strategy = _norm({"fold": max(0.85, 1 - bluff_raise), "raise_300": bluff_raise})
            reason = f"Air ({hs.made_hand}). Fold. No equity to continue."

    return strategy, reason


def _norm(d: Dict[str, float]) -> Dict[str, float]:
    """Normalize strategy dict so values sum to 1, remove near-zero."""
    d = {k: max(0, v) for k, v in d.items()}
    total = sum(d.values())
    if total <= 0:
        return {"check": 1.0}
    d = {k: round(v / total, 4) for k, v in d.items() if v / total > 0.01}
    total = sum(d.values())
    if total > 0 and abs(total - 1) > 0.01:
        d = {k: round(v / total, 4) for k, v in d.items()}
    return d


# ============================================================
# Full scenario advisor
# ============================================================

@dataclass
class PokerScenario:
    """Complete poker scenario for hand advisor."""
    hero_hand: str                      # e.g. "AhKd" or "AKo"
    hero_position: str                  # UTG, HJ, CO, BTN, SB, BB
    villain_position: str = ""          # Position of main opponent
    board: str = ""                     # Board cards
    pot_size: float = 1.5               # Current pot (preflop starts at 1.5 in most games)
    stack_size: float = 100.0           # Effective stack
    action_history: List[str] = field(default_factory=list)
    # Action format: "player:action" e.g. "UTG:raise_3", "HJ:fold", "CO:call", "BTN:raise_9"
    street: str = "preflop"
    num_villains: int = 1               # Number of opponents (1=HU, 2-8=multiway)


def advise_scenario(scenario: PokerScenario) -> Dict:
    """
    Main entry point: given a complete scenario, return strategy advice.

    Returns comprehensive advice including:
    - recommended actions with frequencies
    - reasoning
    - available bet sizes
    - hand strength analysis
    """
    if scenario.street == "preflop":
        return _advise_preflop(scenario)
    else:
        return _advise_postflop(scenario)


def _advise_preflop(scenario: PokerScenario) -> Dict:
    """Preflop advice based on action history."""
    hand = scenario.hero_hand.replace(" ", "")  # normalize "Ah Kd" → "AhKd"

    # Canonicalize hand if specific cards given
    if len(hand) == 4:
        canonical = canonicalize_from_cards(hand)
    else:
        canonical = hand

    pos = scenario.hero_position

    # Parse action history to determine scenario type
    raises = [a for a in scenario.action_history if 'raise' in a]
    calls = [a for a in scenario.action_history if 'call' in a]

    n_raises = len(raises)

    if n_raises == 0:
        # RFI situation
        strategy = get_rfi_strategy(pos, canonical)
        scenario_type = "RFI (Raise First In)"
        detail = f"First to open from {pos}"
    elif n_raises == 1:
        # Facing a raise
        raiser_pos = ""
        for a in scenario.action_history:
            if 'raise' in a:
                parts = a.split(':')
                raiser_pos = parts[0] if len(parts) > 1 else 'CO'
                break

        if pos == 'BB':
            strategy = get_bb_defense_strategy(canonical, raiser_pos)
            scenario_type = "BB Defense"
            detail = f"BB facing raise from {raiser_pos}"
        else:
            strategy = get_vs_raise_strategy(pos, canonical, raiser_pos)
            scenario_type = "Facing Raise"
            detail = f"{pos} facing raise from {raiser_pos}"
    elif n_raises == 2:
        # Facing 3bet
        threebettor = ""
        for a in scenario.action_history:
            if 'raise' in a:
                parts = a.split(':')
                threebettor = parts[0] if len(parts) > 1 else 'BB'

        strategy = get_vs_3bet_strategy(pos, canonical, threebettor)
        scenario_type = "Facing 3-Bet"
        detail = f"{pos} facing 3bet from {threebettor}"
    elif n_raises >= 3:
        # Facing 4bet+
        tier = hand_tier(canonical)
        if tier <= 1:
            strategy = {"raise": 0.5, "call": 0.5}
        elif tier <= 2:
            strategy = {"call": 0.6, "fold": 0.4}
        else:
            strategy = {"fold": 1.0}
        scenario_type = f"Facing {n_raises+1}-Bet"
        detail = f"Facing {n_raises+1}-bet with {canonical}"
    else:
        strategy = {"fold": 1.0}
        scenario_type = "Unknown"
        detail = "Could not determine scenario"

    # Raise sizing recommendation
    raise_sizes = []
    if 'raise' in strategy:
        if n_raises == 0:
            raise_sizes = [
                {"size": 2.5, "label": "2.5BB", "description": "Standard open", "frequency": 0.7},
                {"size": 3.0, "label": "3BB", "description": "Larger open", "frequency": 0.3},
            ]
            if pos == 'SB':
                raise_sizes = [
                    {"size": 3.0, "label": "3BB", "description": "Standard SB open", "frequency": 0.8},
                    {"size": 2.5, "label": "2.5BB", "description": "Min-raise", "frequency": 0.2},
                ]
        elif n_raises == 1:
            raise_sizes = [
                {"size": 3.0, "label": "3x raise", "description": "Standard 3-bet", "frequency": 0.5},
                {"size": 3.5, "label": "3.5x raise", "description": "Larger 3-bet", "frequency": 0.3},
                {"size": 4.0, "label": "4x raise", "description": "Large 3-bet (vs EP)", "frequency": 0.2},
            ]
        elif n_raises >= 2:
            raise_sizes = [
                {"size": 2.2, "label": "2.2x 3bet", "description": "Standard 4-bet", "frequency": 0.6},
                {"size": 2.5, "label": "2.5x 3bet", "description": "Larger 4-bet", "frequency": 0.3},
                {"size": 999, "label": "All-in", "description": "Shove", "frequency": 0.1},
            ]

    # Count callers from action history (for multi-way info)
    n_callers = len([a for a in scenario.action_history if 'call' in a])
    nv = max(scenario.num_villains, n_callers + (1 if n_raises > 0 else 0))

    # Multi-way preflop adjustment: tighten with more callers
    if nv >= 2 and n_raises == 0:
        # Squeeze opportunity or tighten open
        tier = hand_tier(canonical)
        if n_callers >= 2:
            # With limpers: tighten range but increase raise size
            scenario_type = f"RFI (vs {n_callers} limpers)"
            detail = f"Opening from {pos} with {n_callers} limpers"
            if tier <= 3:
                strategy = {'raise': 1.0}
            elif tier <= 4:
                strategy = {'raise': 0.7, 'fold': 0.3}
            else:
                strategy = {'fold': 1.0}
            raise_sizes = [
                {"size": 3.0 + n_callers, "label": f"{3+n_callers}BB", "description": f"Standard (3+{n_callers} limpers)", "frequency": 0.7},
                {"size": 4.0 + n_callers, "label": f"{4+n_callers}BB", "description": "Larger iso-raise", "frequency": 0.3},
            ]

    # Build GTO reasoning with villain matrices
    from .reasoning import build_dynamic_reasoning
    top_action = max(strategy, key=strategy.get)
    gto_reasoning = build_dynamic_reasoning(
        hero_hand_str=scenario.hero_hand,
        board_str="",
        recommended_actions=strategy,
        pot_size=scenario.pot_size,
        stack_size=scenario.stack_size,
        street="preflop",
        num_villains=nv,
        facing_bet=(n_raises > 0),
        bet_fraction=0,
        action_history=scenario.action_history,
        hero_position=pos,
    )

    return {
        "scenario_type": scenario_type,
        "detail": detail,
        "hand": canonical,
        "specific_hand": scenario.hero_hand,
        "position": pos,
        "recommended_actions": strategy,
        "raise_sizes": raise_sizes,
        "hand_tier": hand_tier(canonical),
        "reasoning": _preflop_reasoning(canonical, strategy, scenario_type, pos),
        "num_villains": nv,
        "gto_reasoning": gto_reasoning,
    }


def _preflop_reasoning(hand: str, strategy: Dict, scenario_type: str, pos: str) -> str:
    """Generate human readable reasoning for preflop decision."""
    tier = hand_tier(hand)
    main_action = max(strategy, key=strategy.get)
    freq = strategy[main_action]

    tier_desc = {1: "premium", 2: "strong", 3: "good", 4: "playable",
                 5: "marginal", 6: "weak", 7: "very weak", 8: "trash"}

    parts = [f"{hand} is a {tier_desc.get(tier, 'unknown')} hand (tier {tier})."]

    if main_action == "raise" and freq >= 0.9:
        parts.append(f"This is a clear {main_action} from {pos}.")
    elif main_action == "raise":
        parts.append(f"This hand raises {freq*100:.0f}% of the time from {pos}.")
    elif main_action == "call":
        parts.append(f"This hand calls {freq*100:.0f}% of the time.")
    elif main_action == "fold":
        if freq >= 0.9:
            parts.append(f"This hand should be folded from {pos}.")
        else:
            parts.append(f"This hand folds {freq*100:.0f}% of the time, but can be played at mixed frequency.")

    return " ".join(parts)


def _advise_postflop(scenario: PokerScenario) -> Dict:
    """Postflop advice."""
    is_ip = scenario.hero_position in ('BTN', 'CO') or \
            (scenario.hero_position == 'IP')
    position = "IP" if is_ip else "OOP"

    nv = max(1, scenario.num_villains)

    result = get_postflop_strategy(
        hand=scenario.hero_hand,
        board=scenario.board,
        position=position,
        action_sequence=scenario.action_history,
        pot_size=scenario.pot_size,
        stack_size=scenario.stack_size,
        street=scenario.street,
        num_villains=nv,
        hero_position=scenario.hero_position,
    )

    mw_label = f" ({nv+1}-way)" if nv >= 2 else ""
    result["scenario_type"] = f"Postflop ({scenario.street}){mw_label}"
    result["detail"] = f"{scenario.hero_position} ({position}) on {scenario.board}{mw_label}"
    result["hand"] = scenario.hero_hand
    result["position"] = scenario.hero_position
    result["num_villains"] = nv

    return result
