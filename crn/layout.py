"""Board layout representation + treatment generators for the paired experiment.

A ``Layout`` separates the two things the experiment must treat differently:

  * number_per_tile + desert position  -> HELD FIXED within a pair
  * resource_per_tile (which resource sits on each tile) -> THE TREATMENT

Because the desert tile and every number token are fixed, each corner's pip
total (sum of production_probability over adjacent tiles) is identical within a
pair; only *which resource* those pips produce changes. This is the invariant
the whole design rests on, and it is enforced structurally here rather than
hoped for (the stale engine coupled resource-shuffle -> desert -> numbers).

Tile ids and the number/resource vectors are length-19, indexed by tile_id.
Resource values are stored as ints (see engine ``constants.Resource``); the
desert tile has resource == Resource.DESERT and number 0.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import numpy as np

# Make the vendored engine importable as the `src` package.
_ENGINE = Path(__file__).resolve().parent.parent / "engine"
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from src.game.board import Board                       # noqa: E402
from src.game.constants import Resource, NUM_TILES     # noqa: E402

DESERT = int(Resource.DESERT)


@dataclass(frozen=True)
class Layout:
    """An exact board assignment: resource and number token per tile."""
    resource_per_tile: Tuple[int, ...]   # length 19, ints (Resource)
    number_per_tile: Tuple[int, ...]     # length 19, token (0 on the desert)

    def __post_init__(self):
        assert len(self.resource_per_tile) == NUM_TILES, "need 19 resources"
        assert len(self.number_per_tile) == NUM_TILES, "need 19 numbers"

    @property
    def desert_tile(self) -> int:
        return self.resource_per_tile.index(DESERT)

    @property
    def non_desert_tiles(self) -> List[int]:
        return [i for i, r in enumerate(self.resource_per_tile) if r != DESERT]

    def to_board(self) -> Board:
        """Build a fully-constructed engine Board from this layout (no RNG)."""
        return Board(layout=self)


def board_to_layout(board: Board) -> Layout:
    """Extract a Layout from a constructed engine Board."""
    return Layout(
        resource_per_tile=tuple(int(t.resource) for t in board.tiles),
        number_per_tile=tuple(int(t.number) for t in board.tiles),
    )


def generate_reference_layout(layout_gen: np.random.Generator) -> Layout:
    """Generate a valid random reference layout using the layout RNG stream.

    Reuses the engine's own rule-respecting generation (6/8 not adjacent, same
    number not adjacent) by building a seeded Board, then reads it back.
    """
    board = Board(randomize=True, rng=layout_gen)
    return board_to_layout(board)


def full_reshuffle(layout: Layout, layout_gen: np.random.Generator) -> Layout:
    """Treatment (a): permute resource types across the NON-desert tiles.

    Desert position and every number token are preserved, so pip totals are
    unchanged within the pair; only the resource->tile mapping changes.
    """
    res = list(layout.resource_per_tile)
    nd = layout.non_desert_tiles
    vals = [res[i] for i in nd]
    perm = layout_gen.permutation(len(vals))
    for k, i in enumerate(nd):
        res[i] = vals[perm[k]]
    return Layout(tuple(res), layout.number_per_tile)


def swap_numbers(layout: Layout, tile_i: int, tile_j: int) -> Layout:
    """Swap the NUMBER TOKENS on two tiles; resources stay put. Used by the pip
    calibration (changes a tile's production frequency without moving resources, so
    distinct-resource counts are unchanged). Neither tile may be the desert."""
    if DESERT in (layout.resource_per_tile[tile_i], layout.resource_per_tile[tile_j]):
        raise ValueError("swap_numbers tiles must both be non-desert")
    num = list(layout.number_per_tile)
    num[tile_i], num[tile_j] = num[tile_j], num[tile_i]
    return Layout(layout.resource_per_tile, tuple(num))


def targeted_swap(layout: Layout, tile_i: int, tile_j: int) -> Layout:
    """Treatment (b): swap the resources on two specific non-desert tiles.

    Used to make a minimal, controlled change to the focal player's diversity
    while logging the effect on others. Numbers/desert unchanged.
    """
    if DESERT in (layout.resource_per_tile[tile_i], layout.resource_per_tile[tile_j]):
        raise ValueError("targeted_swap tiles must both be non-desert")
    res = list(layout.resource_per_tile)
    res[tile_i], res[tile_j] = res[tile_j], res[tile_i]
    return Layout(tuple(res), layout.number_per_tile)
