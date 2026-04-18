"""
FastAPI backend for GTO Poker Calculator.

Provides REST API endpoints for:
1. Preflop strategy lookup (range matrix)
2. Hand Advisor (preflop + postflop scenario analysis)
3. Equity calculation
4. Postflop CFR solving
"""

import os
import sys
import time
import asyncio
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.cards import Card, parse_cards, RANK_CHARS, SUIT_CHARS
from backend.core.evaluator import evaluate_hand, get_hand_category, compute_equity
from backend.core.advisor import (
    advise_scenario, PokerScenario,
    get_rfi_strategy, hand_tier, RANKS, POSITIONS,
    get_postflop_strategy, classify_hand_strength,
    _get_bet_sizes,
)
from backend.solver.preflop import (
    get_preflop_strategy, get_full_range_matrix,
    POSITION_RANGES, ALL_HANDS, hand_combos,
)
from backend.solver.engine import SolverEngine

# ============================================================
# App Setup
# ============================================================

app = FastAPI(title="GTO Poker Calculator", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)
solver_engine = SolverEngine()

# ============================================================
# Request/Response Models
# ============================================================

class PreflopResponse(BaseModel):
    position: str
    strategies: Dict[str, Dict[str, float]]
    total_combos: int
    open_percentage: float


class AdvisorRequest(BaseModel):
    hero_hand: str = Field(..., description="Hero hand e.g. 'AhKd' or 'AKo'")
    hero_position: str = Field(..., description="Hero position: UTG/HJ/CO/BTN/SB/BB")
    villain_position: str = Field("", description="Villain position")
    board: str = Field("", description="Board cards e.g. 'As Kd 7h'")
    pot_size: float = Field(1.5, description="Current pot size in BB")
    stack_size: float = Field(100.0, description="Effective stack in BB")
    action_history: List[str] = Field(default=[], description="Action history e.g. ['UTG:raise_3','HJ:fold']")
    street: str = Field("preflop", description="preflop/flop/turn/river")
    num_villains: int = Field(1, description="Number of opponents (1=HU, 2-8=multiway)", ge=1, le=8)


class EquityRequest(BaseModel):
    hand: str = Field(..., description="Hero hand, e.g. 'Ah Kd'")
    board: str = Field("", description="Board cards")
    simulations: int = Field(10000, description="Monte Carlo simulations")
    num_villains: int = Field(1, description="Number of opponents", ge=1, le=8)


class SolveRequest(BaseModel):
    board: str = Field(..., description="Board cards, e.g. 'As Kd 7h'")
    pot: float = Field(10.0, description="Current pot size")
    oop_stack: float = Field(100.0, description="OOP player stack")
    ip_stack: float = Field(100.0, description="IP player stack")
    oop_range: List[str] = Field(default=None)
    ip_range: List[str] = Field(default=None)
    bet_sizes: List[float] = Field(default=[0.33, 0.67, 1.0])
    raise_sizes: List[float] = Field(default=[0.5, 1.0])
    iterations: int = Field(100, ge=10, le=2000)


# ============================================================
# API Endpoints
# ============================================================

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/preflop/{position}")
async def get_preflop(position: str):
    """Get full preflop range matrix for a position."""
    position = position.upper()
    if position == "BB":
        position = "BB_vs_BTN"

    if position not in POSITION_RANGES and position != "BB_VS_BTN":
        raise HTTPException(400, f"Invalid position: {position}")

    matrix = get_full_range_matrix(position)

    total_combos = 0
    open_combos = 0
    for hand, strategy in matrix.items():
        n = hand_combos(hand)
        total_combos += n
        fold_freq = strategy.get('fold', 0)
        open_combos += n * (1 - fold_freq)

    return PreflopResponse(
        position=position,
        strategies=matrix,
        total_combos=total_combos,
        open_percentage=round(open_combos / total_combos * 100, 1) if total_combos else 0,
    )


@app.post("/api/advisor")
async def hand_advisor(req: AdvisorRequest):
    """
    Hand Advisor - the main feature.
    Given a scenario, returns GTO-based action recommendations.
    """
    try:
        scenario = PokerScenario(
            hero_hand=req.hero_hand,
            hero_position=req.hero_position.upper(),
            villain_position=req.villain_position.upper() if req.villain_position else "",
            board=req.board,
            pot_size=req.pot_size,
            stack_size=req.stack_size,
            action_history=req.action_history,
            street=req.street.lower(),
            num_villains=req.num_villains,
        )

        result = advise_scenario(scenario)

        # Also compute equity if we have board cards (multi-way aware)
        equity_data = None
        hand_category = None
        if req.board.strip() and len(req.hero_hand) == 4:
            try:
                hand_cards = parse_cards(req.hero_hand[:2] + " " + req.hero_hand[2:])
                board_cards = parse_cards(req.board)
                if len(hand_cards) == 2 and len(board_cards) >= 3:
                    loop = asyncio.get_event_loop()
                    if req.num_villains >= 2:
                        from backend.core.evaluator import compute_multiway_equity
                        nv = req.num_villains
                        equity_data = await loop.run_in_executor(
                            executor,
                            lambda: compute_multiway_equity(
                                hand_cards, board_cards,
                                num_villains=nv,
                                n_simulations=5000
                            )
                        )
                        equity = equity_data["equity"]
                    else:
                        equity = await loop.run_in_executor(
                            executor,
                            lambda: compute_equity(hand_cards, board_cards, n_simulations=5000)
                        )
                        equity = round(equity, 4)
                    score = evaluate_hand(hand_cards, board_cards)
                    hand_category = get_hand_category(score)
            except Exception:
                equity = None

        result["equity"] = equity_data["equity"] if equity_data else (equity if 'equity' in dir() else None)
        result["equity_detail"] = equity_data  # Full multiway breakdown
        result["hand_category"] = hand_category
        return result

    except Exception as e:
        raise HTTPException(500, f"Advisor error: {str(e)}")


@app.post("/api/equity")
async def calculate_equity(req: EquityRequest):
    """Calculate equity for a hand against random range(s)."""
    try:
        hand_cards = parse_cards(req.hand)
        board_cards = parse_cards(req.board) if req.board.strip() else []

        if len(hand_cards) != 2:
            raise HTTPException(400, "Hand must be exactly 2 cards")

        loop = asyncio.get_event_loop()
        nv = req.num_villains

        if nv >= 2:
            from backend.core.evaluator import compute_multiway_equity
            result = await loop.run_in_executor(
                executor,
                lambda: compute_multiway_equity(
                    hand_cards, board_cards,
                    num_villains=nv,
                    n_simulations=req.simulations
                )
            )
        else:
            equity = await loop.run_in_executor(
                executor,
                lambda: compute_equity(hand_cards, board_cards, n_simulations=req.simulations)
            )
            result = {
                "equity": round(equity, 4),
                "win_pct": round(equity, 4),
                "tie_pct": 0.0,
                "lose_pct": round(1.0 - equity, 4),
                "num_villains": 1,
                "simulations": req.simulations,
            }

        hand_category = None
        if len(board_cards) >= 3:
            score = evaluate_hand(hand_cards, board_cards)
            hand_category = get_hand_category(score)

        result["hand_category"] = hand_category
        return result

    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/solve")
async def solve_spot(req: SolveRequest):
    """Solve a postflop spot using DCFR."""
    start = time.time()

    try:
        default_range = [
            'AA', 'KK', 'QQ', 'JJ', 'TT',
            'AKs', 'AKo', 'AQs',
            'KQs', 'JTs',
        ]

        oop_range = req.oop_range or default_range
        ip_range = req.ip_range or default_range

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: solver_engine.solve_spot(
                board_str=req.board,
                pot=req.pot,
                oop_stack=req.oop_stack,
                ip_stack=req.ip_stack,
                oop_range=oop_range,
                ip_range=ip_range,
                bet_sizes=req.bet_sizes,
                raise_sizes=req.raise_sizes,
                n_iterations=req.iterations,
                verbose=False,
            )
        )

        elapsed = (time.time() - start) * 1000

        return {
            "board": result['board'],
            "oop_strategies": result['oop_strategies'],
            "ip_strategies": result['ip_strategies'],
            "info_set_count": result['info_set_count'],
            "tree_node_count": result['tree_node_count'],
            "solve_time_ms": round(elapsed, 1),
        }

    except Exception as e:
        raise HTTPException(500, f"Solver error: {str(e)}")


@app.get("/api/positions")
async def get_positions():
    return {
        "positions": [
            {"id": "UTG", "name": "Under The Gun", "order": 1},
            {"id": "HJ", "name": "Hijack", "order": 2},
            {"id": "CO", "name": "Cutoff", "order": 3},
            {"id": "BTN", "name": "Button", "order": 4},
            {"id": "SB", "name": "Small Blind", "order": 5},
            {"id": "BB", "name": "Big Blind", "order": 6},
        ]
    }


@app.post("/api/evaluate")
async def evaluate_cards(hand: str, board: str):
    try:
        hand_cards = parse_cards(hand)
        board_cards = parse_cards(board)
        score = evaluate_hand(hand_cards, board_cards)
        category = get_hand_category(score)
        return {"score": score, "category": category}
    except ValueError as e:
        raise HTTPException(400, str(e))


# ============================================================
# Static file serving
# ============================================================

FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "frontend"
)

if os.path.exists(FRONTEND_DIR):
    app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
