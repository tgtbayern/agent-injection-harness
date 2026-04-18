"""
Dynamic GTO Reasoning Engine v3.

Every calculation uses the villain's ESTIMATED RANGE (from their action history),
not random hands. Every sentence references the actual cards on the table.
Also shows how villain should respond to Hero's recommended action.
"""

from typing import Dict, List, Tuple, Optional
from .cards import Card, parse_cards, RANK_CHARS
from .evaluator import evaluate_hand, get_hand_category
from .ranges import estimate_villain_range
import itertools, random

RANK_NAMES = {14:'A',13:'K',12:'Q',11:'J',10:'T',9:'9',8:'8',7:'7',6:'6',5:'5',4:'4',3:'3',2:'2'}
RANK_WORDS = {14:'Ace',13:'King',12:'Queen',11:'Jack',10:'Ten',9:'Nine',8:'Eight',7:'Seven',6:'Six',5:'Five',4:'Four',3:'Three',2:'Two'}


def build_dynamic_reasoning(
    hero_hand_str: str,
    board_str: str,
    recommended_actions: Dict[str, float],
    pot_size: float,
    stack_size: float,
    street: str,
    num_villains: int,
    facing_bet: bool,
    bet_fraction: float,
    action_history: List[str],
    hero_position: str = "",
) -> Dict:
    try:
        if len(hero_hand_str) == 4 and ' ' not in hero_hand_str:
            hero_cards = parse_cards(hero_hand_str[:2] + ' ' + hero_hand_str[2:])
        else:
            hero_cards = parse_cards(hero_hand_str)
        board_cards = parse_cards(board_str) if board_str.strip() else []
    except Exception:
        return {"analysis": [], "math": [], "villain_hands": {}, "villain_ranges": {}}

    if len(hero_cards) != 2:
        return {"analysis": [], "math": [], "villain_hands": {}, "villain_ranges": {}}

    hr1, hr2 = hero_cards[0], hero_cards[1]
    spr = stack_size / pot_size if pot_size > 0 else 10
    top_action = max(recommended_actions, key=recommended_actions.get) if recommended_actions else 'check'

    analysis = []
    math_lines = []

    # ================================================================
    # 0. ESTIMATE VILLAIN RANGES from action history
    # ================================================================
    range_info = estimate_villain_range(
        action_history, hero_position, board_cards, hero_cards
    )
    active_villains = range_info["active_villains"]
    villain_data = range_info["villains"]

    # Collect all villain combos for equity calculation
    all_villain_combos = []
    range_descriptions = []
    for vpos in active_villains:
        vd = villain_data.get(vpos, {})
        combos = vd.get("combos", [])
        all_villain_combos.extend(combos)
        range_descriptions.append(f"**{vpos}**: {vd.get('range_name', 'unknown')}")

    # Show villain range estimates
    if range_descriptions:
        analysis.append("**Villain range estimates (from action history):**")
        for rd in range_descriptions:
            analysis.append(f"  • {rd}")

    # ================================================================
    # 1. BOARD INTERACTION
    # ================================================================
    if board_cards:
        board_ranks = sorted([c.rank for c in board_cards], reverse=True)
        board_str_pretty = ' '.join(str(c) for c in board_cards)
        score = evaluate_hand(hero_cards, board_cards)
        category = get_hand_category(score)

        analysis.append(f"**Your hand: {hr1}{hr2} on {board_str_pretty}**")
        analysis.append(f"You make: **{category}**")

        # Pair analysis
        for bc in board_cards:
            if hr1.rank == bc.rank:
                pos_on_board = "top" if bc.rank == board_ranks[0] else "second" if len(board_ranks)>1 and bc.rank == board_ranks[1] else "bottom"
                kicker = hr2 if hr1.rank == bc.rank else hr1
                analysis.append(f"Your {RANK_WORDS[hr1.rank]} pairs {bc} → **{pos_on_board} pair**, kicker {RANK_WORDS[kicker.rank]}")
                break
            if hr2.rank == bc.rank:
                pos_on_board = "top" if bc.rank == board_ranks[0] else "second" if len(board_ranks)>1 and bc.rank == board_ranks[1] else "bottom"
                kicker = hr1
                analysis.append(f"Your {RANK_WORDS[hr2.rank]} pairs {bc} → **{pos_on_board} pair**, kicker {RANK_WORDS[kicker.rank]}")
                break
        else:
            if hr1.rank == hr2.rank:
                if hr1.rank > board_ranks[0]:
                    analysis.append(f"Pocket {RANK_WORDS[hr1.rank]}s → **overpair**")
                else:
                    analysis.append(f"Pocket {RANK_WORDS[hr1.rank]}s → **underpair**")
            elif category == "High Card":
                analysis.append(f"No pair — {RANK_WORDS[max(hr1.rank,hr2.rank)]}-high, missed the board")

        # Draws
        for suit in {hr1.suit, hr2.suit}:
            bc_suit = sum(1 for c in board_cards if c.suit == suit)
            hc_suit = sum(1 for c in hero_cards if c.suit == suit)
            if bc_suit + hc_suit == 4:
                sname = {0:'clubs',1:'diamonds',2:'hearts',3:'spades'}[suit]
                fc = [c for c in hero_cards if c.suit == suit]
                analysis.append(f"**Flush draw** ({sname}): {' '.join(str(c) for c in fc)} + {bc_suit} on board → 9 outs")

        all_r = sorted(set(c.rank for c in hero_cards + board_cards))
        for start in range(1, 11):
            needed = list(range(start, start+5))
            if 1 in needed: needed = [14,2,3,4,5]
            have = [r for r in needed if r in all_r]
            miss = [r for r in needed if r not in all_r]
            hero_in = [r for r in needed if r in [hr1.rank, hr2.rank]]
            if len(miss) == 1 and hero_in:
                nc = RANK_NAMES.get(miss[0], '?')
                dt = "OESD" if miss[0] in (needed[0], needed[-1]) else "gutshot"
                outs = 8 if dt == "OESD" else 4
                analysis.append(f"**{dt}**: need **{nc}** for {''.join(RANK_NAMES.get(r,'?') for r in sorted(needed, reverse=True))} ({outs} outs)")
                break

        # ================================================================
        # 2. EQUITY VS VILLAIN'S RANGE (not random!)
        # ================================================================
        dead = set(c.id for c in hero_cards + board_cards)
        if all_villain_combos:
            # Filter valid combos
            valid = [(c1,c2) for c1,c2 in all_villain_combos
                     if c1.id not in dead and c2.id not in dead
                     and c1.id != c2.id]
            if len(valid) > 400:
                valid = random.sample(valid, 400)

            win = lose = tie = 0
            beat_examples = {}
            for v1, v2 in valid:
                try:
                    vs = evaluate_hand([v1,v2], board_cards)
                    vc = get_hand_category(vs)
                    if score > vs: win += 1
                    elif score < vs:
                        lose += 1
                        if vc not in beat_examples: beat_examples[vc] = f"{v1}{v2}"
                    else: tie += 1
                except: pass

            total = win + lose + tie
            if total > 0:
                equity = (win + tie*0.5) / total
                win_pct = win/total*100
                lose_pct = lose/total*100

                analysis.append(f"**Equity vs villain's range**: {equity*100:.1f}% (beat {win_pct:.0f}% / lose {lose_pct:.0f}% of their range)")

                if beat_examples:
                    threats = list(beat_examples.items())[:4]
                    analysis.append(f"**In their range that beats you**: {', '.join(f'{cat} ({ex})' for cat,ex in threats)}")
            else:
                equity = 0.5
        else:
            equity = 0.5

        # ================================================================
        # 3. WHY THIS ACTION — with actual equity vs range
        # ================================================================
        if facing_bet and bet_fraction > 0:
            bet_amt = pot_size * bet_fraction
            po = bet_amt / (pot_size + 2*bet_amt)
            mdf = pot_size / (pot_size + bet_amt)
            ev_call = equity * (pot_size + bet_amt) - (1-equity) * bet_amt

            math_lines.append(f"Villain bets {bet_amt:.1f} BB into {pot_size:.1f} BB ({bet_fraction*100:.0f}% pot)")
            math_lines.append(f"Pot odds: need **{po*100:.1f}%** equity to call")
            math_lines.append(f"Your equity vs their range: **{equity*100:.1f}%** → {'PROFITABLE' if equity > po else 'UNPROFITABLE'} call")
            math_lines.append(f"EV(call) = {equity*100:.1f}% × {pot_size+bet_amt:.1f} − {(1-equity)*100:.1f}% × {bet_amt:.1f} = **{ev_call:+.1f} BB**")
            math_lines.append(f"MDF: must defend **{mdf*100:.0f}%** of your range")

            if ev_call > 0:
                analysis.append(f"**Call is +EV** ({ev_call:+.1f} BB): your {equity*100:.0f}% equity vs their range beats the {po*100:.0f}% pot odds threshold.")
            else:
                analysis.append(f"**Call is -EV** ({ev_call:+.1f} BB): your {equity*100:.0f}% equity is below the {po*100:.0f}% threshold. Consider folding unless you have draws or need to defend MDF.")
        else:
            math_lines.append(f"SPR = {spr:.1f} (stack {stack_size:.0f} / pot {pot_size:.0f})")
            math_lines.append(f"Equity vs villain's range: **{equity*100:.0f}%**")

            if num_villains >= 2:
                feq = 0.5 ** num_villains
                math_lines.append(f"Fold equity vs {num_villains} players: 50%^{num_villains} = **{feq*100:.0f}%**")

            if top_action == 'check':
                reasons = []
                if equity < 0.5:
                    reasons.append(f"equity vs their range is only {equity*100:.0f}%")
                if num_villains >= 2:
                    reasons.append(f"fold equity only {0.5**num_villains*100:.0f}% vs {num_villains} opponents")
                if lose_pct > 30:
                    reasons.append(f"{lose_pct:.0f}% of their range already beats you")
                if reasons:
                    analysis.append(f"**Why check**: {'; '.join(reasons)}.")
            elif 'bet' in top_action:
                if equity > 0.6:
                    analysis.append(f"**Why bet**: {equity*100:.0f}% equity vs their range — you're ahead of most of their range, bet for value.")
                else:
                    analysis.append(f"**Thin value / semi-bluff**: {equity*100:.0f}% equity is marginal, but betting has merit with fold equity.")

        if num_villains >= 2:
            analysis.append(f"**{num_villains+1}-way**: each opponent having ~30% chance of beating you means ~{min(95,int(lose_pct*(1+(num_villains-1)*0.3)))}% chance at least one does.")

        # ================================================================
        # 4. PER-VILLAIN RESPONSE TO HERO'S ACTION
        # ================================================================
        per_villain_response = {}
        for vpos in active_villains:
            vd = villain_data.get(vpos, {})
            v_combos = vd.get("combos", [])
            if v_combos:
                per_villain_response[vpos] = _villain_response_to_hero(
                    hero_cards, board_cards, score, top_action,
                    pot_size, bet_fraction, v_combos, [vpos]
                )
                per_villain_response[vpos]["range_name"] = vd.get("range_name", "")
                per_villain_response[vpos]["canonical_hands"] = vd.get("canonical_hands", [])[:25]
                # Build 13x13 matrix for this villain
                per_villain_response[vpos]["matrix"] = _build_villain_matrix(
                    board_cards, hero_cards, score, top_action,
                    pot_size, bet_fraction,
                    vd.get("canonical_hands", []),
                )
        villain_response = per_villain_response
    else:
        # Preflop — build matrix from preflop ranges
        analysis.append(f"**Your hand: {hr1}{hr2}**")
        suited = hr1.suit == hr2.suit
        if hr1.rank == hr2.rank:
            analysis.append(f"Pocket {RANK_WORDS[hr1.rank]}s")
        elif suited:
            analysis.append(f"{RANK_WORDS[hr1.rank]}{RANK_WORDS[hr2.rank]} suited")
        else:
            analysis.append(f"{RANK_WORDS[hr1.rank]}{RANK_WORDS[hr2.rank]} offsuit")

        villain_response = {}
        for vpos in active_villains:
            vd = villain_data.get(vpos, {})
            v_range = set(vd.get("canonical_hands", []))
            if v_range:
                villain_response[vpos] = _build_preflop_villain_response(
                    vpos, v_range, top_action, action_history
                )
                villain_response[vpos]["range_name"] = vd.get("range_name", "")
                villain_response[vpos]["canonical_hands"] = vd.get("canonical_hands", [])[:25]
        equity = 0.5

    # Build range display info
    villain_range_display = {}
    for vpos in active_villains:
        vd = villain_data.get(vpos, {})
        villain_range_display[vpos] = {
            "range_name": vd.get("range_name", ""),
            "combo_count": vd.get("combo_count", 0),
            "sample_hands": vd.get("canonical_hands", [])[:20],
        }

    return {
        "analysis": analysis,
        "math": math_lines,
        "villain_hands": villain_response,  # now keyed by position: {"UTG": {...}, "CO": {...}}
        "villain_ranges": villain_range_display,
    }


def _villain_response_to_hero(
    hero_cards, board_cards, hero_score,
    hero_action, pot_size, bet_fraction,
    villain_combos, active_villains
):
    """
    Given hero's chosen action, how should each hand in villain's range respond?
    """
    if not villain_combos or not board_cards:
        return {}

    dead = set(c.id for c in hero_cards + board_cards)
    valid = [(c1,c2) for c1,c2 in villain_combos
             if c1.id not in dead and c2.id not in dead and c1.id != c2.id]
    if len(valid) > 500:
        valid = random.sample(valid, 500)

    # Determine what hero did
    if hero_action.startswith('bet') or hero_action.startswith('raise'):
        try:
            frac = float(hero_action.split('_')[1]) / 100.0
        except: frac = bet_fraction if bet_fraction > 0 else 0.67
        bet_amt = pot_size * frac
        new_pot = pot_size + bet_amt
        denom = pot_size + 2*bet_amt
        po = bet_amt / denom if denom > 0 else 0.5
        scenario = f"Villain facing Hero's **{frac*100:.0f}%** pot bet ({bet_amt:.1f} BB)"
        mdf = pot_size / (pot_size + bet_amt) if (pot_size + bet_amt) > 0 else 0.5
    elif hero_action == 'check':
        scenario = "Villain acts after Hero checks (in position)"
        bet_amt = 0
        po = 0
        mdf = 0
        frac = 0
    else:
        scenario = f"Villain responds to Hero's {hero_action}"
        bet_amt = 0; po = 0; mdf = 0; frac = 0

    # Evaluate each villain hand and categorize their optimal response
    raise_hands = []
    call_hands = []
    fold_hands = []

    for v1, v2 in valid:
        try:
            vs = evaluate_hand([v1,v2], board_cards)
            vc = get_hand_category(vs)
            label = f"{v1}{v2}"

            if hero_action == 'check':
                # Villain can bet or check behind
                if vc in ('Straight Flush','Four of a Kind','Full House','Flush','Straight','Three of a Kind'):
                    raise_hands.append((label, vc, vs))  # "bet" = raise category
                elif vc == 'Two Pair':
                    raise_hands.append((label, vc, vs))
                elif vc == 'One Pair':
                    br = sorted([c.rank for c in board_cards], reverse=True)
                    paired_top = (v1.rank == br[0] or v2.rank == br[0])
                    if paired_top or (v1.rank == v2.rank and v1.rank > br[0]):
                        call_hands.append((label, vc, vs))  # "check behind / thin bet"
                    else:
                        fold_hands.append((label, vc, vs))  # "check behind"
                else:
                    fold_hands.append((label, vc, vs))  # check behind
            else:
                # Facing hero's bet: raise / call / fold
                if vc in ('Straight Flush','Four of a Kind','Full House'):
                    raise_hands.append((label, vc, vs))
                elif vc in ('Flush','Straight','Three of a Kind'):
                    raise_hands.append((label, vc, vs))
                elif vc == 'Two Pair':
                    call_hands.append((label, vc, vs))
                elif vc == 'One Pair':
                    br = sorted([c.rank for c in board_cards], reverse=True)
                    paired_top = (v1.rank == br[0] or v2.rank == br[0])
                    overpair = (v1.rank == v2.rank and v1.rank > br[0])
                    if overpair or paired_top:
                        call_hands.append((label, vc, vs))
                    elif frac <= 0.5:
                        call_hands.append((label, vc, vs))  # small bet, can call wider
                    else:
                        fold_hands.append((label, vc, vs))
                else:
                    # Check for draws (simplified: if hand has some connection)
                    fold_hands.append((label, vc, vs))
        except:
            pass

    def examples(lst, n=8):
        seen = set()
        result = []
        for hand, cat, sc in sorted(lst, key=lambda x: -x[2]):
            if hand not in seen:
                result.append(f"{hand} ({cat})")
                seen.add(hand)
            if len(result) >= n: break
        return result

    total = len(raise_hands) + len(call_hands) + len(fold_hands)
    if total == 0:
        return {}

    if hero_action == 'check':
        raise_label = "Should bet for value"
        call_label = "Should check behind / thin value"
        fold_label = "Should check behind (weak)"
    else:
        raise_label = "Should raise"
        call_label = "Should call"
        fold_label = "Should fold"

    return {
        "scenario": scenario,
        "mdf": round(mdf, 3) if mdf else None,
        "pot_odds": round(po, 3) if po else None,
        "raise": {"pct": round(len(raise_hands)/total*100,1), "count": len(raise_hands),
                  "examples": examples(raise_hands), "label": raise_label},
        "call":  {"pct": round(len(call_hands)/total*100,1), "count": len(call_hands),
                  "examples": examples(call_hands), "label": call_label},
        "fold":  {"pct": round(len(fold_hands)/total*100,1), "count": len(fold_hands),
                  "examples": examples(fold_hands, 5), "label": fold_label},
        "total_combos": total,
    }


def _build_villain_matrix(
    board_cards: List[Card],
    hero_cards: List[Card],
    hero_score: int,
    hero_action: str,
    pot_size: float,
    bet_fraction: float,
    villain_canonical_range: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Build a 13×13 matrix: for each canonical hand (AA, AKs, AKo, ...),
    compute what % should raise / call / fold on this board.

    Returns: { "AKs": {"raise":0.0, "call":1.0, "fold":0.0, "in_range": true}, ... }
    """
    from .ranges import expand_hand_to_combos

    RANKS_STR = 'AKQJT98765432'
    dead = set(c.id for c in hero_cards + board_cards)
    range_set = set(villain_canonical_range)
    board_ranks = sorted([c.rank for c in board_cards], reverse=True)

    # Determine bet context
    is_facing_bet = hero_action.startswith('bet') or hero_action.startswith('raise')
    try:
        frac = float(hero_action.split('_')[1]) / 100.0 if is_facing_bet else (bet_fraction or 0)
    except:
        frac = bet_fraction or 0.5

    matrix = {}

    for i, r1 in enumerate(RANKS_STR):
        for j, r2 in enumerate(RANKS_STR):
            if i < j:
                hand = r1 + r2 + 's'
            elif i > j:
                hand = r2 + r1 + 'o'
            else:
                hand = r1 + r2

            in_range = hand in range_set

            if not in_range or not board_cards:
                matrix[hand] = {"raise": 0, "call": 0, "fold": 0, "in_range": in_range}
                continue

            # Expand to combos and evaluate each
            combos = expand_hand_to_combos(hand, dead)
            if not combos:
                matrix[hand] = {"raise": 0, "call": 0, "fold": 0, "in_range": in_range}
                continue

            r_count = c_count = f_count = 0
            for v1, v2 in combos:
                try:
                    vs = evaluate_hand([v1, v2], board_cards)
                    vc = get_hand_category(vs)

                    if hero_action == 'check':
                        # Villain can bet or check behind
                        if vc in ('Straight Flush', 'Four of a Kind', 'Full House', 'Flush',
                                  'Straight', 'Three of a Kind', 'Two Pair'):
                            r_count += 1
                        elif vc == 'One Pair':
                            paired_top = (v1.rank == board_ranks[0] or v2.rank == board_ranks[0])
                            overpair = (v1.rank == v2.rank and v1.rank > board_ranks[0])
                            if paired_top or overpair:
                                c_count += 1  # thin value / check
                            else:
                                f_count += 1  # check behind
                        else:
                            f_count += 1
                    else:
                        # Facing hero's bet
                        if vc in ('Straight Flush', 'Four of a Kind', 'Full House',
                                  'Flush', 'Straight', 'Three of a Kind'):
                            r_count += 1
                        elif vc == 'Two Pair':
                            c_count += 1
                        elif vc == 'One Pair':
                            paired_top = (v1.rank == board_ranks[0] or v2.rank == board_ranks[0])
                            overpair = (v1.rank == v2.rank and v1.rank > board_ranks[0])
                            if overpair or paired_top:
                                c_count += 1
                            elif frac <= 0.5:
                                c_count += 1
                            else:
                                f_count += 1
                        else:
                            f_count += 1
                except:
                    pass

            total = r_count + c_count + f_count
            if total > 0:
                matrix[hand] = {
                    "raise": round(r_count / total, 2),
                    "call": round(c_count / total, 2),
                    "fold": round(f_count / total, 2),
                    "in_range": True,
                }
            else:
                matrix[hand] = {"raise": 0, "call": 0, "fold": 0, "in_range": in_range}

    return matrix


def _build_preflop_villain_response(
    villain_pos: str,
    villain_range: set,
    hero_action: str,
    action_history: List[str],
) -> Dict:
    """
    Build 13×13 preflop matrix for villain.
    Shows what each hand should do facing hero's action.
    """
    from .advisor import hand_tier

    RANKS_STR = 'AKQJT98765432'

    # Count raises in history to determine scenario
    n_raises = sum(1 for a in action_history if 'raise' in a)

    # What is villain facing?
    if hero_action == 'raise' or 'raise' in hero_action:
        # Villain faces hero's raise: should they call, 3bet, or fold?
        scenario = f"{villain_pos} facing Hero's raise"
        def classify(hand, in_range):
            if not in_range:
                return {"raise": 0, "call": 0, "fold": 0, "in_range": False}
            tier = hand_tier(hand)
            if tier <= 1:
                return {"raise": 0.7, "call": 0.3, "fold": 0, "in_range": True}
            elif tier <= 2:
                return {"raise": 0.4, "call": 0.6, "fold": 0, "in_range": True}
            elif tier <= 3:
                return {"raise": 0.15, "call": 0.7, "fold": 0.15, "in_range": True}
            elif tier <= 4:
                return {"raise": 0, "call": 0.6, "fold": 0.4, "in_range": True}
            elif tier <= 5:
                return {"raise": 0, "call": 0.3, "fold": 0.7, "in_range": True}
            else:
                return {"raise": 0, "call": 0, "fold": 1.0, "in_range": True}
    elif hero_action == 'call':
        scenario = f"{villain_pos} after Hero calls"
        def classify(hand, in_range):
            if not in_range:
                return {"raise": 0, "call": 0, "fold": 0, "in_range": False}
            tier = hand_tier(hand)
            if tier <= 1:
                return {"raise": 0.8, "call": 0.2, "fold": 0, "in_range": True}  # squeeze
            elif tier <= 2:
                return {"raise": 0.5, "call": 0.5, "fold": 0, "in_range": True}
            elif tier <= 4:
                return {"raise": 0, "call": 0.7, "fold": 0.3, "in_range": True}
            elif tier <= 6:
                return {"raise": 0, "call": 0.4, "fold": 0.6, "in_range": True}
            else:
                return {"raise": 0, "call": 0, "fold": 1.0, "in_range": True}
    else:
        scenario = f"{villain_pos} responds to Hero's {hero_action}"
        def classify(hand, in_range):
            if not in_range:
                return {"raise": 0, "call": 0, "fold": 0, "in_range": False}
            tier = hand_tier(hand)
            if tier <= 2:
                return {"raise": 0.5, "call": 0.5, "fold": 0, "in_range": True}
            elif tier <= 4:
                return {"raise": 0, "call": 0.6, "fold": 0.4, "in_range": True}
            else:
                return {"raise": 0, "call": 0, "fold": 1.0, "in_range": True}

    matrix = {}
    r_total = c_total = f_total = 0

    for i, r1 in enumerate(RANKS_STR):
        for j, r2 in enumerate(RANKS_STR):
            if i < j:
                hand = r1 + r2 + 's'
            elif i > j:
                hand = r2 + r1 + 'o'
            else:
                hand = r1 + r2

            in_range = hand in villain_range
            result = classify(hand, in_range)
            matrix[hand] = result
            if in_range:
                r_total += result["raise"]
                c_total += result["call"]
                f_total += result["fold"]

    total = r_total + c_total + f_total
    r_pct = r_total / total * 100 if total else 0
    c_pct = c_total / total * 100 if total else 0
    f_pct = f_total / total * 100 if total else 0

    return {
        "scenario": scenario,
        "mdf": None,
        "pot_odds": None,
        "raise": {"pct": round(r_pct, 1), "count": int(r_total), "examples": [], "label": "3-Bet / Raise"},
        "call": {"pct": round(c_pct, 1), "count": int(c_total), "examples": [], "label": "Call"},
        "fold": {"pct": round(f_pct, 1), "count": int(f_total), "examples": [], "label": "Fold"},
        "total_combos": int(total),
        "matrix": matrix,
    }
