"""Arm A / Arm B paired-run drivers.

Arm A (primary, causal): the agent places on a REFERENCE board; that placement is
recorded and replayed identically onto two equal-pip-swapped boards -- game A gets
the increase-swap (focal div_count d+1), game B the decrease-swap (d-1). Neither
played board is the reference, so the agent's optimal-placement advantage is
balanced across the pair and cancels in the within-pair difference. Everything
else (placement, dice, dev-deck, steals, all pip totals, board scarcity) is held
identical; only the focal player's touched resources differ. Contrast = +2 div_count.

Arm B (descriptive): the agent places freely on each of the same two boards, so
diversity is endogenous. Shows how differently the agent behaves when it controls
placement; not a causal estimate.

Both arms accept mode="crn" (paired games share dice/deck/steal) or
mode="independent" (the control set: those streams independent between A and B).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

_ENGINE = Path(__file__).resolve().parent.parent / "engine"
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from src.game.game_state import GameState       # noqa: E402
from src.game.rules import Rules                # noqa: E402
from src.game.constants import NUM_PLAYERS      # noqa: E402

from crn.rng import PairRNG                      # noqa: E402
from crn.layout import generate_reference_layout  # noqa: E402
from crn.runner import play_game, run_setup      # noqa: E402
from crn.features import (                        # noqa: E402
    focal_tiles, choose_diversity_swap, make_placebo_swaps, make_symmetric_swaps,
    features_on_layout, compute_all_features, compute_player_features, board_scarcity,
    RES_NAMES, opp_on_tile_map,
)


def _focal_treatment_log(init_A, init_B, focal, swap_inc, swap_dec):
    """Diversity/composition/port summary of the A(+1) vs B(-1) contrast."""
    fa, fb = init_A[focal], init_B[focal]
    gained_A = swap_inc["focal_new_resource"]   # new distinct resource on board A
    gained_B = swap_dec["focal_new_resource"]   # duplicate resource on board B
    return {
        "delta_div_count": fa["div_count"] - fb["div_count"],          # +2 by design
        "delta_div_entropy": round(fa["div_entropy"] - fb["div_entropy"], 6),
        "focal_gained_A": int(gained_A), "focal_lost_A": int(swap_inc["focal_old_resource"]),
        "focal_lost_B": int(swap_dec["focal_old_resource"]), "focal_gained_B": int(gained_B),
        # Does the focal hold a 2:1 port matching the resource it gains? (interference
        # of ports with the treatment; ~always False for iter17, logged regardless)
        "portmatch_gained_A": bool(fa.get(f"port_{RES_NAMES[gained_A]}", False)),
        "portmatch_gained_B": bool(fb.get(f"port_{RES_NAMES[gained_B]}", False)),
    }


class ReplayAgent:
    """Replays a recorded (player, action) setup sequence; one instance per game.

    Placed in all four seats (it ignores seat and pops the next recorded action),
    asserting the recorded player matches the state's current player -- which it
    always does, because the snake-draft order is shared across the pair.
    """

    def __init__(self, recorded: List[tuple]):
        self.recorded = recorded
        self.i = 0

    def get_action(self, state) -> int:
        p = state.current_player_idx
        rp, ra = self.recorded[self.i]
        assert rp == p, f"replay misalignment @{self.i}: recorded {rp}, state {p}"
        self.i += 1
        return ra


def record_placement(agent, rng, layout, max_steps: int = 2000):
    """Run the agent through initial placement on `layout`; return
    (recorded (player, action) list, end-of-setup GameState)."""
    state = GameState(rng=rng, layout=layout)
    recorded: List[tuple] = []
    steps = 0
    while state.is_setup_phase and not state.is_game_over and steps < max_steps:
        p = state.current_player_idx
        a = int(agent.get_action(state))
        recorded.append((p, a))
        Rules.execute_action(state, a)
        steps += 1
    return recorded, state


def _placements(state):
    sett = [list(state.players[i].settlements) for i in range(NUM_PLAYERS)]
    cities = [list(state.players[i].cities) for i in range(NUM_PLAYERS)]
    return sett, cities


def run_arm_a(pair_seed: int, agent, focal_seat: int, *, mode: str = "crn",
              feature_every: int = 10, require_equal_pips: bool = True) -> Dict:
    """Run one Arm-A pair. Returns a result dict (with skipped=True if the board/
    placement does not admit the symmetric equal-pip contrast)."""
    pr = PairRNG(pair_seed, mode)
    lg = pr.layout_gen()
    ref = generate_reference_layout(lg)

    recorded, setup_state = record_placement(agent, pr.ref_bundle(), ref)
    sett, cities = _placements(setup_state)
    ftiles = focal_tiles(setup_state, focal_seat)
    baseline_div = compute_player_features(setup_state, focal_seat)["div_count"]
    opp_tiles = opp_on_tile_map(setup_state, focal_seat)   # minimize-interference tiebreak

    # Build the symmetric pair inline so we can record WHY a seed is dropped
    # (eligibility characterization for the overlap caveat). Draw order (increase
    # then decrease) is fixed for reproducibility.
    inc = choose_diversity_swap(ref, ftiles, "increase", lg, require_equal_pips, opp_tiles)
    dec = choose_diversity_swap(ref, ftiles, "decrease", lg, require_equal_pips, opp_tiles)
    if inc is None or dec is None:
        if baseline_div >= 5:
            reason = "cap_high_div5"
        elif baseline_div <= 1:
            reason = "cap_low_div1"
        elif inc is None and dec is None:
            reason = "no_equal_pip_candidate_both"
        else:
            reason = "no_equal_pip_candidate_" + ("increase" if inc is None else "decrease")
        return {"arm": "A", "pair_seed": pair_seed, "focal": focal_seat, "mode": mode,
                "skipped": True, "reason": reason, "baseline_div": int(baseline_div)}
    L_inc, rec_inc = inc
    L_dec, rec_dec = dec
    swap_rec = {"increase": rec_inc, "decrease": rec_dec}

    # Initial (end-of-placement) treatment + confounders under the FIXED placement.
    init_A = features_on_layout(sett, cities, L_inc)   # focal div_count = d+1
    init_B = features_on_layout(sett, cities, L_dec)   # focal div_count = d-1

    agents = [agent] * NUM_PLAYERS
    gA = play_game(agents, pr.game_a(), L_inc, feature_every=feature_every,
                   setup_agents=[ReplayAgent(recorded)] * NUM_PLAYERS)
    gB = play_game(agents, pr.game_b(), L_dec, feature_every=feature_every,
                   setup_agents=[ReplayAgent(recorded)] * NUM_PLAYERS)

    return {
        "arm": "A", "pair_seed": pair_seed, "focal": focal_seat, "mode": mode,
        "skipped": False, "swap_records": swap_rec,
        "baseline_div": int(baseline_div),
        "focal_div_A": init_A[focal_seat]["div_count"],
        "focal_div_B": init_B[focal_seat]["div_count"],
        "treatment": _focal_treatment_log(init_A, init_B, focal_seat, rec_inc, rec_dec),
        "init_features": {"A": init_A, "B": init_B},
        "board_scarcity": {"ref": board_scarcity(ref),
                           "A": board_scarcity(L_inc), "B": board_scarcity(L_dec)},
        "gameA": gA, "gameB": gB,
    }


def run_placebo(pair_seed: int, agent, focal_seat: int, *, mode: str = "crn",
                feature_every: int = 0, require_equal_pips: bool = True) -> Dict:
    """Dose-ZERO placebo pair: two distinct neutral swaps (div_count unchanged on
    both), same replayed placement and CRN streams as Arm A. The A-B outcome
    difference here has zero diversity signal, so it estimates the 'swap a focal
    tile at all' (pure composition) effect -- the placebo for the Arm-A contrast."""
    pr = PairRNG(pair_seed, mode)
    lg = pr.layout_gen()
    ref = generate_reference_layout(lg)
    recorded, setup_state = record_placement(agent, pr.ref_bundle(), ref)
    sett, cities = _placements(setup_state)
    ftiles = focal_tiles(setup_state, focal_seat)
    baseline_div = compute_player_features(setup_state, focal_seat)["div_count"]
    opp_tiles = opp_on_tile_map(setup_state, focal_seat)

    plc = make_placebo_swaps(ref, ftiles, lg, require_equal_pips=require_equal_pips,
                             opp_on_tile=opp_tiles)
    if plc is None:
        return {"arm": "placebo", "pair_seed": pair_seed, "focal": focal_seat,
                "mode": mode, "skipped": True, "reason": "no_two_neutral_swaps",
                "baseline_div": int(baseline_div)}
    L1, L2, swap_rec = plc
    init_A = features_on_layout(sett, cities, L1)
    init_B = features_on_layout(sett, cities, L2)

    agents = [agent] * NUM_PLAYERS
    gA = play_game(agents, pr.game_a(), L1, feature_every=feature_every,
                   setup_agents=[ReplayAgent(recorded)] * NUM_PLAYERS)
    gB = play_game(agents, pr.game_b(), L2, feature_every=feature_every,
                   setup_agents=[ReplayAgent(recorded)] * NUM_PLAYERS)
    return {
        "arm": "placebo", "pair_seed": pair_seed, "focal": focal_seat, "mode": mode,
        "skipped": False, "swap_records": swap_rec, "baseline_div": int(baseline_div),
        "focal_div_A": init_A[focal_seat]["div_count"],
        "focal_div_B": init_B[focal_seat]["div_count"],   # equal by design
        "delta_div_entropy": round(init_A[focal_seat]["div_entropy"]
                                   - init_B[focal_seat]["div_entropy"], 6),
        "init_features": {"A": init_A, "B": init_B},
        "board_scarcity": {"A": board_scarcity(L1), "B": board_scarcity(L2)},
        "gameA": gA, "gameB": gB,
    }


def run_arm_b(pair_seed: int, agent, focal_seat: int, *, mode: str = "crn",
              feature_every: int = 10, require_equal_pips: bool = True) -> Dict:
    """Run one Arm-B pair: same two boards as Arm A, but the agent places freely on
    each (diversity endogenous -> descriptive only)."""
    pr = PairRNG(pair_seed, mode)
    lg = pr.layout_gen()
    ref = generate_reference_layout(lg)

    # Need the focal tiles to build the same swaps; use a throwaway reference setup.
    _, setup_state = record_placement(agent, pr.ref_bundle(), ref)
    ftiles = focal_tiles(setup_state, focal_seat)
    # NOTE: unlike run_arm_a / run_placebo, interference minimization (opp_on_tile) is
    # intentionally NOT applied here -- Arm B is descriptive only (endogenous placement),
    # not a causal estimate, so the swap-selection tiebreak doesn't need to match.
    sym = make_symmetric_swaps(ref, ftiles, lg, require_equal_pips=require_equal_pips)
    if sym is None:
        return {"arm": "B", "pair_seed": pair_seed, "focal": focal_seat,
                "mode": mode, "skipped": True, "reason": "no_symmetric_equal_pip_swap"}
    L_inc, L_dec, swap_rec = sym

    agents = [agent] * NUM_PLAYERS
    # Endogenous placement on each board (agent chooses); capture its init features.
    setup_A = run_setup(agents, pr.game_a(), L_inc)
    setup_B = run_setup(agents, pr.game_b(), L_dec)
    init_A = compute_all_features(setup_A)
    init_B = compute_all_features(setup_B)

    gA = play_game(agents, pr.game_a(), L_inc, feature_every=feature_every)
    gB = play_game(agents, pr.game_b(), L_dec, feature_every=feature_every)

    return {
        "arm": "B", "pair_seed": pair_seed, "focal": focal_seat, "mode": mode,
        "skipped": False, "swap_records": swap_rec,
        "focal_div_A": init_A[focal_seat]["div_count"],
        "focal_div_B": init_B[focal_seat]["div_count"],
        "init_features": {"A": init_A, "B": init_B},
        "board_scarcity": {"ref": board_scarcity(ref),
                           "A": board_scarcity(L_inc), "B": board_scarcity(L_dec)},
        "gameA": gA, "gameB": gB,
    }
