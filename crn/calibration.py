"""Pip calibration experiment: estimate delta, the smallest effect of interest for
the diversity study.

This reuses the diversity pipeline EXACTLY (reference board, replayed placement, CRN
streams, deterministic agent, focal-seat rotation) but the treatment is a NUMBER-TOKEN
swap instead of a resource swap:

  Board P+: swap the focal hex's token with a partner hex's token worth exactly +1 pip
            -> the focal player gains exactly 1 pip (resources, and hence distinct
            count, unchanged).
  Board P-: ... worth exactly -1 pip -> focal loses exactly 1 pip.

delta = mean( focal_rank(P-) - focal_rank(P+) ), signed so positive => pip helps.
It is the rank effect of a two-step (+1 vs -1) pip contrast.

Eligibility (per spec):
  - Focal hex: adjacent to the focal player and to NO other player, and to EXACTLY ONE
    focal building (so the focal pip change is exactly +/-1, not +/-2).
  - Partner hex: NOT focal-adjacent; pip differs from the focal hex by exactly 1 in the
    required direction. Prefer partners adjacent to no player; else allow opponent-
    adjacent partners and log the opponents' pip changes.
  - Reject any swap that puts two 6s/8s on adjacent hexes (red-number rule, whole board).
  - Never the desert.
  - Uniform-random pick from eligible candidates (from the seed); log full candidate set.
  - A replication is eligible iff BOTH a P+ and a P- swap exist.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np

_ENGINE = Path(__file__).resolve().parent.parent / "engine"
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from src.game.board import TILE_ADJACENCY           # noqa: E402
from src.game.constants import NUM_PLAYERS, NUM_TILES  # noqa: E402

from crn.rng import PairRNG                           # noqa: E402
from crn.layout import generate_reference_layout, swap_numbers, DESERT  # noqa: E402
from crn.runner import play_game                      # noqa: E402
from crn.arms import record_placement, _placements, ReplayAgent  # noqa: E402
from crn.features import (                            # noqa: E402
    pips_of, compute_player_features, features_on_layout, board_scarcity,
)

RED = {6, 8}


# --------------------------------------------------------------------- board helpers
def player_tiles(state, player_id) -> Set[int]:
    """Non-desert tiles adjacent to a player's buildings (count-agnostic set)."""
    from crn.features import focal_tiles
    return focal_tiles(state, player_id)


def focal_building_counts(state, focal) -> Dict[int, int]:
    """tile_id -> number of FOCAL buildings adjacent to it."""
    board = state.board
    counts: Dict[int, int] = {}
    p = state.players[focal]
    corners = [(c, 1) for c in p.settlements] + [(c, 2) for c in p.cities]
    for corner_id, w in corners:
        for t in board.corners[corner_id].adjacent_tiles:
            if t < len(board.tiles) and board.tiles[t].resource != DESERT:
                counts[t] = counts.get(t, 0) + w
    return counts


def red_number_ok(layout) -> bool:
    """No two ADJACENT tiles both carry a 6 or an 8."""
    num = layout.number_per_tile
    for t in range(NUM_TILES):
        if num[t] in RED:
            for adj in TILE_ADJACENCY.get(t, set()):
                if num[adj] in RED:
                    return False
    return True


def creates_same_number_adjacency(layout) -> bool:
    """Diagnostic only (not a rejection criterion per spec): two equal non-zero
    tokens on adjacent tiles."""
    num = layout.number_per_tile
    for t in range(NUM_TILES):
        if num[t] == 0:
            continue
        for adj in TILE_ADJACENCY.get(t, set()):
            if num[adj] == num[t]:
                return True
    return False


# --------------------------------------------------------------------- swap selection
def _pip_swap_candidates(layout, focal_excl_tiles, focal_adj_tiles,
                         tiles_by_player, delta: int):
    """All eligible (focal_hex, partner_hex) number-swaps for a given pip delta.

    Returns (clean, opp_affecting) candidate lists; each entry is a dict with the
    swap tiles, pips, and any opponents whose pips move (partner-adjacent)."""
    num = layout.number_per_tile
    non_desert = [t for t in range(NUM_TILES) if layout.resource_per_tile[t] != DESERT]
    # Which players (other than focal) touch each tile.
    opp_on_tile: Dict[int, List[int]] = {}
    for pid, tiles in tiles_by_player.items():
        for t in tiles:
            opp_on_tile.setdefault(t, []).append(pid)

    clean, opp = [], []
    for f in focal_excl_tiles:
        pf = pips_of(num[f])
        if pf == 0:
            continue
        for q in non_desert:
            if q == f or q in focal_adj_tiles:
                continue                       # partner must NOT touch the focal
            if pips_of(num[q]) != pf + delta:
                continue
            cand = swap_numbers(layout, f, q)
            if not red_number_ok(cand):
                continue
            opponents = [pid for pid in opp_on_tile.get(q, []) if pid is not None]
            # pip change for opponents on q: q goes from pip(pf+delta) -> pip(pf),
            # i.e. a change of -delta each (times their building count on q).
            rec = {
                "focal_hex": int(f), "partner_hex": int(q),
                "focal_old_num": int(num[f]), "focal_new_num": int(num[q]),
                "focal_pip_delta": int(delta),
                "partner_opponents": [int(o) for o in opponents],
                "opponent_pip_delta": int(-delta),
            }
            (opp if opponents else clean).append((cand, rec))
    return clean, opp


def choose_pip_swap(layout, focal_excl_tiles, focal_adj_tiles, tiles_by_player,
                    delta: int, gen: np.random.Generator):
    """Pick one eligible pip swap (prefer partners adjacent to no player)."""
    clean, opp = _pip_swap_candidates(layout, focal_excl_tiles, focal_adj_tiles,
                                      tiles_by_player, delta)
    pool = clean if clean else opp
    if not pool:
        return None
    idx = int(gen.integers(len(pool)))
    cand, rec = pool[idx]
    rec = dict(rec)
    rec["partner_adjacent_to_opponent"] = bool(not clean)
    rec["n_clean_candidates"] = len(clean)
    rec["n_opp_candidates"] = len(opp)
    rec["candidates"] = [{"focal_hex": r["focal_hex"], "partner_hex": r["partner_hex"]}
                         for _, r in (clean + opp)]
    return cand, rec


# --------------------------------------------------------------------- run one pair
def build_calibration_boards(pair_seed: int, agent, focal_seat: int, mode: str = "crn"):
    """Construct the reference / P+ / P- boards + placement WITHOUT playing games.

    Returns a dict (skipped=True if either direction unavailable) with the layouts,
    swap records, recorded placement, end-of-setup state, and the PairRNG handle so
    the caller can play the two games with the shared CRN streams."""
    pr = PairRNG(pair_seed, mode)
    lg = pr.layout_gen()
    ref = generate_reference_layout(lg)
    recorded, st = record_placement(agent, pr.ref_bundle(), ref)

    all_player_tiles = {j: player_tiles(st, j) for j in range(NUM_PLAYERS)}
    focal_adj = all_player_tiles[focal_seat]
    others = set().union(*[all_player_tiles[j] for j in range(NUM_PLAYERS)
                           if j != focal_seat])
    fb_counts = focal_building_counts(st, focal_seat)
    # Focal-exclusive AND exactly one focal building (so the pip change is exactly +/-1).
    focal_excl = [t for t in focal_adj if t not in others and fb_counts.get(t, 0) == 1]
    base_pips = compute_player_features(st, focal_seat)["total_pips"]

    pplus = choose_pip_swap(ref, focal_excl, focal_adj, all_player_tiles, +1, lg)
    pminus = choose_pip_swap(ref, focal_excl, focal_adj, all_player_tiles, -1, lg)
    if pplus is None or pminus is None:
        return {"skipped": True,
                "reason": ("no_P+" if pplus is None else "") + ("no_P-" if pminus is None else ""),
                "n_focal_exclusive": len(focal_excl), "base_total_pips": base_pips}
    return {"skipped": False, "pr": pr, "ref": ref,
            "L_plus": pplus[0], "L_minus": pminus[0],
            "rec_plus": pplus[1], "rec_minus": pminus[1],
            "recorded": recorded, "setup_state": st, "focal_excl": focal_excl,
            "base_total_pips": base_pips}


def run_calibration(pair_seed: int, agent, focal_seat: int, *, mode: str = "crn",
                    feature_every: int = 0) -> Dict:
    """Run one calibration pair (P+ vs P-) with the standard CRN + replayed placement."""
    b = build_calibration_boards(pair_seed, agent, focal_seat, mode)
    if b["skipped"]:
        return {"exp": "calib", "pair_seed": pair_seed, "focal": focal_seat, "mode": mode,
                "skipped": True, "reason": b["reason"],
                "n_focal_exclusive": b["n_focal_exclusive"],
                "base_total_pips": b["base_total_pips"]}
    pr, ref = b["pr"], b["ref"]
    L_plus, L_minus = b["L_plus"], b["L_minus"]
    rec_plus, rec_minus = b["rec_plus"], b["rec_minus"]
    recorded, st = b["recorded"], b["setup_state"]
    sett, cities = _placements(st)
    base_pips = b["base_total_pips"]

    init_plus = features_on_layout(sett, cities, L_plus)
    init_minus = features_on_layout(sett, cities, L_minus)

    agents = [agent] * NUM_PLAYERS
    gP = play_game(agents, pr.game_a(), L_plus, feature_every=feature_every,
                   setup_agents=[ReplayAgent(recorded)] * NUM_PLAYERS)
    gM = play_game(agents, pr.game_b(), L_minus, feature_every=feature_every,
                   setup_agents=[ReplayAgent(recorded)] * NUM_PLAYERS)

    return {
        "exp": "calib", "pair_seed": pair_seed, "focal": focal_seat, "mode": mode,
        "skipped": False,
        "swap_records": {"P+": rec_plus, "P-": rec_minus},
        "n_focal_exclusive": len(b["focal_excl"]),
        "base_total_pips": base_pips,
        "focal_pips_plus": init_plus[focal_seat]["total_pips"],
        "focal_pips_minus": init_minus[focal_seat]["total_pips"],
        "init_features": {"ref": features_on_layout(sett, cities, ref),
                          "P+": init_plus, "P-": init_minus},
        "board_scarcity": {"ref": board_scarcity(ref),
                           "P+": board_scarcity(L_plus), "P-": board_scarcity(L_minus)},
        "rank_plus": gP["rank"][focal_seat], "rank_minus": gM["rank"][focal_seat],
        "gameP": gP, "gameM": gM,
    }
