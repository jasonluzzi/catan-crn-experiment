"""Minimal paired-game runner + a deterministic scripted agent.

This is the Phase-1 scaffold: enough to construct a game from an injected
``GameRNG`` + fixed ``Layout`` and play it to termination while recording a
trace. The full experiment runner (parallel pairs, parquet logging, confounders)
is built in later phases on top of this.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

_ENGINE = Path(__file__).resolve().parent.parent / "engine"
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from src.game.game_state import GameState          # noqa: E402
from src.game.rules import Rules                   # noqa: E402
from src.game.constants import NUM_PLAYERS         # noqa: E402
from src.game.actions import (                     # noqa: E402
    ACTION_ROLL_DICE, ACTION_END_TURN, ACTION_BUY_DEV_CARD,
    ACTION_BANK_TRADE_START, ACTION_BANK_TRADE_END,
    ACTION_BUILD_ROAD_START, ACTION_BUILD_ROAD_END,
    ACTION_BUILD_SETTLEMENT_START, ACTION_BUILD_SETTLEMENT_END,
    ACTION_BUILD_CITY_START, ACTION_BUILD_CITY_END,
    ACTION_PROPOSE_TRADE_START, ACTION_PROPOSE_TRADE_END,
    ACTION_ACCEPT_TRADE, ACTION_REJECT_TRADE,
    ACTION_COUNTER_OFFER_START, ACTION_COUNTER_OFFER_END,
    ACTION_CANCEL_TRADE,
    ACTION_CONFIRM_TRADE_START, ACTION_CONFIRM_TRADE_END,
    ACTION_DECLINE_TRADE_START, ACTION_DECLINE_TRADE_END,
)


def mask_trading_actions(mask: np.ndarray) -> np.ndarray:
    """Zero out all P2P-trading actions (the no-trading regime used in evals)."""
    m = mask.copy()
    m[ACTION_PROPOSE_TRADE_START:ACTION_PROPOSE_TRADE_END] = False
    m[ACTION_ACCEPT_TRADE] = False
    m[ACTION_REJECT_TRADE] = False
    m[ACTION_COUNTER_OFFER_START:ACTION_COUNTER_OFFER_END] = False
    m[ACTION_CANCEL_TRADE] = False
    m[ACTION_CONFIRM_TRADE_START:ACTION_CONFIRM_TRADE_END] = False
    m[ACTION_DECLINE_TRADE_START:ACTION_DECLINE_TRADE_END] = False
    return m


class ScriptedAgent:
    """A fully deterministic, mask-only agent (consumes NO randomness).

    Priority mirrors a simple builder: roll -> city -> settlement -> dev card ->
    road -> (else lowest legal action, e.g. robber/steal/discard/setup/end).
    Because it never draws from any RNG, a game played by four of these is a pure
    function of the injected engine streams + layout -- ideal for determinism
    tests. It is NOT meant to play well.
    """

    def __init__(self, no_trading: bool = True):
        self.no_trading = no_trading

    def get_action(self, state) -> int:
        mask = Rules.get_legal_actions(state)
        if self.no_trading:
            m = mask_trading_actions(mask)
            if m.any():
                mask = m
        legal = np.flatnonzero(mask)

        def first_in(lo, hi):
            sel = legal[(legal >= lo) & (legal < hi)]
            return int(sel[0]) if len(sel) else None

        if mask[ACTION_ROLL_DICE]:
            return ACTION_ROLL_DICE
        for lo, hi in (
            (ACTION_BUILD_CITY_START, ACTION_BUILD_CITY_END),
            (ACTION_BUILD_SETTLEMENT_START, ACTION_BUILD_SETTLEMENT_END),
        ):
            a = first_in(lo, hi)
            if a is not None:
                return a
        if mask[ACTION_BUY_DEV_CARD]:
            return ACTION_BUY_DEV_CARD
        a = first_in(ACTION_BUILD_ROAD_START, ACTION_BUILD_ROAD_END)
        if a is not None:
            return a
        # Everything else (setup placement, robber, steal, discard, play dev):
        # take the lowest legal action, but prefer END_TURN last.
        non_end = legal[legal != ACTION_END_TURN]
        return int(non_end[0]) if len(non_end) else ACTION_END_TURN


def ranks_from_vp(vp: List[int]) -> List[int]:
    """Rank 1..4 by victory points (1 = most VP). Ties broken by lower seat id,
    which is arbitrary but deterministic; documented in LIMITATIONS.md."""
    order = sorted(range(len(vp)), key=lambda i: (-vp[i], i))
    rank = [0] * len(vp)
    for r, seat in enumerate(order, start=1):
        rank[seat] = r
    return rank


def run_setup(agents, rng, layout, no_trading: bool = True, max_steps: int = 2000):
    """Step a fresh game through initial placement only; return the GameState.

    Stops as soon as the setup phase ends (each player then has 2 settlements,
    0 cities). Used to compute end-of-initial-placement features without playing
    the whole game.
    """
    state = GameState(rng=rng, layout=layout)
    steps = 0
    while state.is_setup_phase and not state.is_game_over and steps < max_steps:
        p = state.current_player_idx
        a = agents[p].get_action(state)
        Rules.execute_action(state, a)
        steps += 1
    return state


def play_game(agents, rng, layout, no_trading: bool = True, max_steps: int = 20000,
              record_trace: bool = False, feature_every: int = 0,
              setup_agents=None) -> dict:
    """Play one full game with injected ``rng`` (GameRNG) and fixed ``layout``.

    Args beyond the obvious:
      feature_every : if >0, snapshot per-player features (diversity + ports) every
                      this many turns (and at game end) -> `trajectory` (mediator /
                      divergence logging).
      setup_agents  : optional separate agent list used DURING the setup phase only
                      (Arm A replays recorded placements here); the main `agents`
                      drive the rest of the game.

    Returns winner, per-seat VP/rank, dice sequence, maritime-trade counts,
    bank-scarcity events, and (optionally) the trace and feature trajectory.
    """
    from crn.features import compute_all_features   # lazy: avoid import cycle

    state = GameState(rng=rng, layout=layout)
    trace: List[tuple] = []
    dice_seq: List[tuple] = []
    maritime_trades = [0] * NUM_PLAYERS
    trajectory: List[dict] = []
    recorded_turns = set()
    steps = 0

    def snapshot():
        feats = compute_all_features(state)
        for f in feats:
            f["turn"] = int(state.turn_number)
        trajectory.append({"turn": int(state.turn_number), "players": feats})

    while not state.is_game_over and steps < max_steps:
        p = state.current_player_idx
        acting = (setup_agents if (setup_agents is not None and state.is_setup_phase)
                  else agents)
        a = acting[p].get_action(state)
        if record_trace:
            trace.append((p, a))
        if ACTION_BANK_TRADE_START <= a < ACTION_BANK_TRADE_END:
            maritime_trades[p] += 1
        Rules.execute_action(state, a)
        if a == ACTION_ROLL_DICE:
            dice_seq.append(tuple(state.last_dice_roll))
        if (feature_every and not state.is_setup_phase
                and state.turn_number % feature_every == 0
                and state.turn_number not in recorded_turns):
            recorded_turns.add(state.turn_number)
            snapshot()
        steps += 1

    if feature_every:
        snapshot()   # always capture the final state

    vp_total = [int(state.players[i].total_victory_points) for i in range(NUM_PLAYERS)]
    vp_visible = [int(state.players[i].victory_points) for i in range(NUM_PLAYERS)]
    return {
        "winner": state.winner,
        "completed": bool(state.is_game_over),
        "vp": vp_total,
        "vp_visible": vp_visible,
        "rank": ranks_from_vp(vp_total),
        "dice_seq": dice_seq,
        "maritime_trades": maritime_trades,
        "bank_scarcity_events": int(getattr(state, "bank_scarcity_events", 0)),
        "steps": steps,
        "trace": trace if record_trace else None,
        "trajectory": trajectory if feature_every else None,
        "final_features": compute_all_features(state),
    }
