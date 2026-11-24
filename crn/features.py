"""Treatment (resource diversity) + confounder features at end of initial placement.

Two diversity definitions (per the design):
  div_count   : number of DISTINCT resource types adjacent to the player's
                settlements (1..5, desert excluded).
  div_entropy : Shannon entropy (bits) of the player's expected production
                distribution over the 5 resources, weighted by pips.

Confounders logged for EVERY player (so the analysis can adjust, not hide):
  * Composition: expected production per resource (exp_prod_<resource>).
  * total_pips : sum of production_probability over the player's buildings.
  * Ports      : generic (3:1) access, per-resource (2:1) port access, and
                 whether each resource port matches a resource the player produces.
Interference is handled by computing these for all four players each game.

"Pips" here are in units of production_probability (dots/36). Entropy is a
distribution over resources, so it is invariant to that scale.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from math import log2
from typing import Dict, List, Optional, Set
import numpy as np

_ENGINE = Path(__file__).resolve().parent.parent / "engine"
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from src.game.constants import Resource, PortType     # noqa: E402
from crn.layout import targeted_swap, DESERT          # noqa: E402

# pips (dots) by number token; production_probability == PIPS[n]/36.
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def pips_of(number: int) -> int:
    return PIPS.get(int(number), 0)


RESOURCES = [Resource.LUMBER, Resource.BRICK, Resource.WOOL, Resource.GRAIN, Resource.ORE]
RES_NAMES = ["lumber", "brick", "wool", "grain", "ore"]
# PortType value -> resource index (GENERIC has none). LUMBER port=1 .. ORE port=5.
PORT_TO_RES = {PortType.LUMBER: 0, PortType.BRICK: 1, PortType.WOOL: 2,
               PortType.GRAIN: 3, PortType.ORE: 4}


def player_building_corners(player):
    """[(corner_id, weight)] with settlement weight 1, city weight 2."""
    return ([(c, 1) for c in player.settlements]
            + [(c, 2) for c in player.cities])


def focal_tiles(state, player_id) -> Set[int]:
    """Non-desert tile ids adjacent to a player's settlements+cities."""
    board = state.board
    player = state.players[player_id]
    tiles: Set[int] = set()
    for corner_id, _ in player_building_corners(player):
        for t in board.corners[corner_id].adjacent_tiles:
            if t < len(board.tiles) and board.tiles[t].resource != Resource.DESERT:
                tiles.add(t)
    return tiles


def expected_production(state, player_id) -> np.ndarray:
    """Vector (len 5) of expected production per resource, weighted by pips."""
    board = state.board
    player = state.players[player_id]
    exp = np.zeros(5, dtype=float)
    for corner_id, weight in player_building_corners(player):
        for t in board.corners[corner_id].adjacent_tiles:
            if t >= len(board.tiles):
                continue
            tile = board.tiles[t]
            if tile.resource == Resource.DESERT:
                continue
            exp[int(tile.resource)] += weight * tile.production_probability
    return exp


def shannon_entropy_bits(weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    p = weights[weights > 0] / total
    return float(-(p * np.log2(p)).sum())


def board_scarcity(layout) -> Dict:
    """Board-level resource supply: total pips per resource + board entropy.

    Invariants (verified in tests): total board pips is fixed by construction
    (number tokens never move); the per-resource breakdown here is held fixed only
    under equal-pip swaps, and varies under a full reshuffle.
    """
    totals = np.zeros(5, dtype=float)
    for t in range(len(layout.resource_per_tile)):
        r = layout.resource_per_tile[t]
        if r == DESERT:
            continue
        totals[int(r)] += PIPS.get(int(layout.number_per_tile[t]), 0)
    feat = {
        "board_total_pips": round(float(totals.sum()), 6),
        "board_entropy": round(shannon_entropy_bits(totals), 6),
    }
    for r, name in enumerate(RES_NAMES):
        feat[f"board_pips_{name}"] = round(float(totals[r]), 6)
    return feat


def compute_player_features(state, player_id) -> Dict:
    """All diversity + confounder features for one player (flat, parquet-ready)."""
    board = state.board
    player = state.players[player_id]
    exp = expected_production(state, player_id)
    total = float(exp.sum())

    div_count = int((exp > 0).sum())
    div_entropy = shannon_entropy_bits(exp)

    # Ports accessible via the player's building corners.
    ports = []
    for corner_id, _ in player_building_corners(player):
        port = board.corners[corner_id].port
        if port is not None:
            ports.append(port)
    has_generic = any(p == PortType.GENERIC for p in ports)
    res_port_access = [False] * 5
    for p in ports:
        if p in PORT_TO_RES:
            res_port_access[PORT_TO_RES[p]] = True
    n_resource_ports = sum(res_port_access)
    n_matching = sum(1 for r in range(5) if res_port_access[r] and exp[r] > 0)

    feat = {
        "player_id": player_id,
        "div_count": div_count,
        "div_entropy": round(div_entropy, 6),
        "total_pips": round(total, 6),
        "n_settlements": len(player.settlements),
        "n_cities": len(player.cities),
        "has_generic_port": bool(has_generic),
        "n_resource_ports": int(n_resource_ports),
        "n_matching_ports": int(n_matching),
    }
    for r, name in enumerate(RES_NAMES):
        feat[f"exp_prod_{name}"] = round(float(exp[r]), 6)
        feat[f"port_{name}"] = bool(res_port_access[r])
    return feat


def compute_all_features(state) -> List[Dict]:
    """Features for all players (interference-visible)."""
    return [compute_player_features(state, i) for i in range(len(state.players))]


class _ShimPlayer:
    def __init__(self, settlements, cities):
        self.settlements = list(settlements)
        self.cities = list(cities)


class _ShimState:
    """Minimal state exposing .board and .players for feature computation."""
    def __init__(self, board, players):
        self.board = board
        self.players = players


def features_on_layout(settlements_per_player, cities_per_player, layout) -> List[Dict]:
    """Compute all-player features for a layout, holding placements fixed (Arm A).

    Lets us evaluate the treated layout under the SAME settlements as the reference
    game, so a targeted swap's effect (on focal and on others = interference) is a
    clean difference.
    """
    board = layout.to_board()
    players = [_ShimPlayer(s, c) for s, c in
               zip(settlements_per_player, cities_per_player)]
    return compute_all_features(_ShimState(board, players))


# --------------------------------------------------------------------------- swaps
def opp_on_tile_map(state, focal) -> Dict[int, Set[int]]:
    """tile_id -> set of OPPONENT seat ids whose buildings touch that tile.

    Used to minimize interference: swapping resources on a tile changes the
    composition of every player adjacent to it, so we prefer swaps whose two tiles
    touch the fewest distinct opponents."""
    board = state.board
    m: Dict[int, Set[int]] = {}
    for pid in range(len(state.players)):
        if pid == focal:
            continue
        p = state.players[pid]
        for corner_id in list(p.settlements) + list(p.cities):
            for t in board.corners[corner_id].adjacent_tiles:
                if t < len(board.tiles) and board.tiles[t].resource != DESERT:
                    m.setdefault(t, set()).add(pid)
    return m


def _interference(ti, tj, opp_on_tile) -> int:
    """# distinct opponents whose composition changes if (ti,tj) resources swap."""
    if opp_on_tile is None:
        return 0
    return len(opp_on_tile.get(ti, set()) | opp_on_tile.get(tj, set()))


def _select_swap(candidates, opp_on_tile, gen):
    """Pick a candidate (ti,tj): minimize opponent interference first, then uniform
    at random among the minimizers (drawn from the seed). Returns
    (idx_in_candidates, ti, tj, chosen_interference, per_candidate_interference)."""
    interf = [_interference(a, b, opp_on_tile) for a, b in candidates]
    best = min(interf)
    elig = [k for k, v in enumerate(interf) if v == best]
    pick = elig[int(gen.integers(len(elig)))]
    return pick, candidates[pick][0], candidates[pick][1], best, interf


def choose_diversity_swap(layout, focal_tile_set: Set[int], direction: str,
                          gen: np.random.Generator, require_equal_pips: bool = True,
                          opp_on_tile=None):
    """Pick a 2-tile resource swap that changes the FOCAL player's div_count by +/-1.

    direction="decrease": make a uniquely-contributed focal resource into one the
      focal already has elsewhere (distinct count -1).
    direction="increase": turn a duplicated focal resource into one the focal does
      not yet have (distinct count +1).

    Numbers/desert are never touched (resources-only swap), so every corner's pip
    total and the TOTAL board pips are invariant. With require_equal_pips=True, the
    two swapped tiles must also carry equal pips, which additionally holds each
    resource's BOARD-LEVEL total pips (board scarcity / board entropy) fixed -- the
    swap only moves which resource the focal touches. This is the primary design.

    Returns (new_layout, record) or None if no valid swap exists for this placement.
    """
    if direction not in ("increase", "decrease"):
        raise ValueError("direction must be 'increase' or 'decrease'")

    res = layout.resource_per_tile
    num = layout.number_per_tile
    focal = sorted(focal_tile_set)
    nd = layout.non_desert_tiles
    non_focal = [t for t in nd if t not in focal_tile_set]

    focal_res = [res[t] for t in focal]
    counts = Counter(focal_res)
    distinct = set(focal_res)

    candidates = []  # (t_i focal, t_j non-focal)
    for ti in focal:
        ri = res[ti]
        if direction == "decrease" and counts[ri] != 1:
            continue
        if direction == "increase" and counts[ri] < 2:
            continue
        for tj in non_focal:
            rj = res[tj]
            if require_equal_pips and pips_of(num[ti]) != pips_of(num[tj]):
                continue
            if direction == "decrease" and rj != ri and rj in distinct:
                candidates.append((ti, tj))
            elif direction == "increase" and rj not in distinct:
                candidates.append((ti, tj))

    if not candidates:
        return None
    # Selection: minimize opponent interference, then uniform-at-random among the
    # minimizers (from the pair seed). With opp_on_tile=None this reduces to pure
    # uniform. The FULL candidate set (with per-candidate interference) is logged so
    # the rule cannot silently shape the estimand.
    pick_idx, ti, tj, chosen_interf, interf = _select_swap(candidates, opp_on_tile, gen)
    new_layout = targeted_swap(layout, ti, tj)
    cand_log = [{"focal_tile": int(a), "other_tile": int(b),
                 "focal_old_resource": int(res[a]), "focal_new_resource": int(res[b]),
                 "pips": int(pips_of(num[a])), "interference": int(interf[k])}
                for k, (a, b) in enumerate(candidates)]
    record = {
        "direction": direction,
        "focal_tile": int(ti),
        "other_tile": int(tj),
        "focal_old_resource": int(res[ti]),   # resource the focal LOSES on its tile
        "focal_new_resource": int(res[tj]),    # resource the focal GAINS on its tile
        "swap_pips": int(pips_of(num[ti])),
        "equal_pip": bool(pips_of(num[ti]) == pips_of(num[tj])),
        "n_candidate_swaps": len(candidates),
        "picked_index": pick_idx,
        "n_opponents_affected": int(chosen_interf),
        "opponents_affected": sorted(
            (opp_on_tile or {}).get(ti, set()) | (opp_on_tile or {}).get(tj, set())),
        "min_possible_interference": int(min(interf)),
        "candidates": cand_log,
    }
    return new_layout, record


def choose_neutral_swap(layout, focal_tile_set: Set[int], gen: np.random.Generator,
                        require_equal_pips: bool = True, opp_on_tile=None,
                        exclude=None):
    """Dose-ZERO (placebo) swap: change WHICH resource the focal touches on one of
    its tiles WITHOUT changing its distinct-resource count.

    Construction: take a focal tile carrying a DUPLICATED focal resource ri (so the
    focal still has ri after) and swap it with an equal-pip non-focal tile whose
    resource rj (!= ri) is ALREADY in the focal's set (so no new distinct type is
    added). div_count is unchanged; composition/pip-entropy shift. Touches a focal
    tile, so the board is still one swap off the reference (no placement-fit leak).

    Minimizes opponent interference like the diversity swap. `exclude` is an optional
    set of (ti,tj) pairs to skip (so a second neutral swap differs from the first).
    """
    res = layout.resource_per_tile
    num = layout.number_per_tile
    focal = sorted(focal_tile_set)
    non_focal = [t for t in layout.non_desert_tiles if t not in focal_tile_set]
    counts = Counter(res[t] for t in focal)
    distinct = set(res[t] for t in focal)
    exclude = exclude or set()

    candidates = []
    for ti in focal:
        ri = res[ti]
        if counts[ri] < 2:          # keep ri present after losing this copy
            continue
        for tj in non_focal:
            rj = res[tj]
            if require_equal_pips and pips_of(num[ti]) != pips_of(num[tj]):
                continue
            if rj != ri and rj in distinct and (ti, tj) not in exclude:
                candidates.append((ti, tj))
    if not candidates:
        return None
    pick_idx, ti, tj, chosen_interf, interf = _select_swap(candidates, opp_on_tile, gen)
    record = {
        "direction": "neutral", "focal_tile": int(ti), "other_tile": int(tj),
        "focal_old_resource": int(res[ti]), "focal_new_resource": int(res[tj]),
        "swap_pips": int(pips_of(num[ti])), "n_candidate_swaps": len(candidates),
        "picked_index": pick_idx,
        "n_opponents_affected": int(chosen_interf),
        "opponents_affected": sorted(
            (opp_on_tile or {}).get(ti, set()) | (opp_on_tile or {}).get(tj, set())),
        "min_possible_interference": int(min(interf)),
        "candidates": [{"focal_tile": int(a), "other_tile": int(b),
                        "interference": int(interf[k])}
                       for k, (a, b) in enumerate(candidates)],
    }
    return targeted_swap(layout, ti, tj), record


def make_placebo_swaps(layout, focal_tile_set: Set[int], gen: np.random.Generator,
                       require_equal_pips: bool = True, opp_on_tile=None):
    """Placebo pair: two DISTINCT dose-zero swaps (both leave div_count unchanged),
    so the A-B contrast has zero diversity signal but a real composition difference.
    Returns (layout_1, layout_2, record) or None if two distinct neutral swaps are
    unavailable."""
    first = choose_neutral_swap(layout, focal_tile_set, gen, require_equal_pips,
                                opp_on_tile=opp_on_tile)
    if first is None:
        return None
    l1, rec1 = first
    second = choose_neutral_swap(layout, focal_tile_set, gen, require_equal_pips,
                                 opp_on_tile=opp_on_tile,
                                 exclude={(rec1["focal_tile"], rec1["other_tile"])})
    if second is None:
        return None
    l2, rec2 = second
    return l1, l2, {"swap1": rec1, "swap2": rec2}


def make_symmetric_swaps(layout, focal_tile_set: Set[int], gen: np.random.Generator,
                         require_equal_pips: bool = True):
    """Build the symmetric Arm-A pair: an increase-swap and a decrease-swap, both
    one swap off the reference (so placement staleness is balanced; neither game is
    the board the agent placed on).

    Returns (layout_increase, layout_decrease, record) or None if either direction
    is unavailable for this (board, placement). `record` carries both swap records
    and the realized focal div_count contrast (should be +2).
    """
    inc = choose_diversity_swap(layout, focal_tile_set, "increase", gen, require_equal_pips)
    if inc is None:
        return None
    dec = choose_diversity_swap(layout, focal_tile_set, "decrease", gen, require_equal_pips)
    if dec is None:
        return None
    l_inc, rec_inc = inc
    l_dec, rec_dec = dec
    return l_inc, l_dec, {"increase": rec_inc, "decrease": rec_dec}
