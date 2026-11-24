"""Board representation for Settlers of Catan."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import random

from .constants import (
    Resource, PortType, NUM_TILES, NUM_CORNERS, NUM_EDGES,
    TILE_DISTRIBUTION, NUMBER_DISTRIBUTION
)


# Pre-computed tile adjacency map (tiles that share an edge)
# Built from get_verified_tile_corners() - tiles sharing 2 corners are adjacent
# Updated for Colonist.io board topology
TILE_ADJACENCY: Dict[int, Set[int]] = {
    0: {1, 2, 3},
    1: {0, 2, 16, 17},
    2: {0, 1, 3, 4, 17, 18},
    3: {0, 2, 4, 5},
    4: {2, 3, 5, 6, 7, 18},
    5: {3, 4, 6},
    6: {4, 5, 7, 8},
    7: {4, 6, 8, 9, 10, 18},
    8: {6, 7, 9},
    9: {7, 8, 10, 11},
    10: {7, 9, 11, 12, 13, 18},
    11: {9, 10, 12},
    12: {10, 11, 13, 14},
    13: {10, 12, 14, 15, 17, 18},
    14: {12, 13, 15},
    15: {13, 14, 16, 17},
    16: {1, 15, 17},
    17: {1, 2, 13, 15, 16, 18},
    18: {2, 4, 7, 10, 13, 17},
}


@dataclass
class Tile:
    """A hexagonal tile on the board."""
    id: int
    resource: Resource
    number: int
    has_robber: bool = False

    @property
    def production_probability(self) -> float:
        """Probability of this tile producing (out of 36)."""
        if self.number == 0 or self.number == 7:
            return 0.0
        probs = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        return probs.get(self.number, 0) / 36.0


@dataclass
class Corner:
    """A corner (vertex) where tiles meet."""
    id: int
    adjacent_tiles: List[int] = field(default_factory=list)
    adjacent_corners: List[int] = field(default_factory=list)
    adjacent_edges: List[int] = field(default_factory=list)
    building: Optional[str] = None
    owner: Optional[int] = None
    port: Optional[PortType] = None


@dataclass
class Edge:
    """An edge between two corners."""
    id: int
    corners: Tuple[int, int] = (0, 0)
    adjacent_tiles: List[int] = field(default_factory=list)
    has_road: bool = False
    owner: Optional[int] = None


def _shuffle(seq: list, rng=None) -> None:
    """Shuffle a list in place.

    CRN refactor: if `rng` (a numpy Generator) is supplied, use it so the draw is
    reproducible and stream-separated; otherwise fall back to the legacy global
    `random` module (unchanged original behavior).
    """
    if rng is None:
        random.shuffle(seq)
    else:
        perm = rng.permutation(len(seq))
        seq[:] = [seq[i] for i in perm]


class Board:
    """The Catan game board."""

    def __init__(self, randomize: bool = True, rng=None, layout=None):
        """Initialize the board.

        CRN refactor (added params, original default behavior preserved):
          - `rng`: optional numpy Generator for the *layout* stream. When given (and
            no explicit `layout`), resource/number shuffles draw from it instead of
            the global `random` module, so board generation is reproducible and
            independent of all other randomness.
          - `layout`: optional Layout (crn.layout.Layout) giving an exact
            resource-per-tile and number-per-tile assignment. When given, the board
            is built deterministically from it and NO randomness is consumed. This
            is how the paired experiment holds the desert tile + number tokens fixed
            while permuting only the resource types (keeping each corner's pip total
            identical within a pair).
        """
        self.tiles: List[Tile] = []
        self.corners: List[Corner] = []
        self.edges: List[Edge] = []
        if layout is not None:
            self._build_from_layout(layout)
        else:
            self._build_board(randomize, rng)
        self._assign_ports()
        self._build_tile_corner_map()

    def _build_graph(self) -> None:
        """Create corner/edge objects and adjacencies (independent of tile layout)."""
        for corner_id in range(NUM_CORNERS):
            self.corners.append(Corner(id=corner_id))
        for edge_id in range(NUM_EDGES):
            self.edges.append(Edge(id=edge_id))
        self._build_adjacencies()

    def _build_from_layout(self, layout) -> None:
        """Build tiles directly from a Layout (deterministic, no RNG).

        layout.resource_per_tile[tile_id] -> Resource (ints ok), and
        layout.number_per_tile[tile_id] -> int token (0 for the desert tile).
        """
        for tile_id in range(NUM_TILES):
            resource = Resource(layout.resource_per_tile[tile_id])
            if resource == Resource.DESERT:
                self.tiles.append(Tile(tile_id, resource, 0, True))
            else:
                self.tiles.append(
                    Tile(tile_id, resource, int(layout.number_per_tile[tile_id]))
                )
        self._build_graph()

        # Cached longest road per player (invalidated on road placement)
        self._longest_road_cache: Dict[int, int] = {}
        self._road_cache_valid: bool = True

        # Cached corner production rankings. These are a pure function of the tile
        # layout (production_probability / number / resource), which never changes
        # after board generation, so they are computed once and then shared by
        # reference across every board copy in the game (see copy()). This removes a
        # per-MCTS-node O(54 log 54) recomputation from the hottest search path.
        self._corner_rankings_cache = None

    def _build_tile_corner_map(self) -> None:
        """Build precomputed mapping from tile_id to corner_ids for O(1) lookup."""
        self._tile_to_corners: Dict[int, List[int]] = {i: [] for i in range(NUM_TILES)}
        for corner in self.corners:
            for tile_id in corner.adjacent_tiles:
                if tile_id < NUM_TILES:
                    self._tile_to_corners[tile_id].append(corner.id)

    def get_tile_corners(self, tile_id: int) -> List[int]:
        """Get corner IDs adjacent to a tile (O(1) cached lookup)."""
        return self._tile_to_corners.get(tile_id, [])

    def _build_board(self, randomize: bool, rng=None) -> None:
        """Build the board structure with valid number placement.

        Enforces standard Catan rules:
        - 6 and 8 cannot be adjacent
        - Same numbers cannot be adjacent

        `rng` (CRN refactor): optional numpy Generator for the layout stream.
        """
        resources = []
        for resource, count in TILE_DISTRIBUTION.items():
            resources.extend([resource] * count)

        if randomize:
            _shuffle(resources, rng)

        # Find which tile is the desert (gets no number)
        desert_tile_id = resources.index(Resource.DESERT)

        # Get valid number placement
        numbers = NUMBER_DISTRIBUTION.copy()
        if randomize:
            number_placement = self._get_valid_number_placement(
                numbers, desert_tile_id, rng=rng
            )
        else:
            number_placement = numbers

        # Create tiles with assigned numbers
        number_idx = 0
        for tile_id, resource in enumerate(resources):
            if resource == Resource.DESERT:
                self.tiles.append(Tile(tile_id, resource, 0, True))
            else:
                self.tiles.append(Tile(tile_id, resource, number_placement[number_idx]))
                number_idx += 1

        self._build_graph()

    def _get_valid_number_placement(
        self,
        numbers: List[int],
        desert_tile_id: int,
        max_attempts: int = 1000,
        rng=None
    ) -> List[int]:
        """Generate a valid number placement respecting Catan rules.

        Rules:
        - 6 and 8 cannot be on adjacent tiles
        - Same numbers cannot be on adjacent tiles

        Uses backtracking with random restarts.

        Args:
            numbers: List of number tokens to place (18 numbers for 18 non-desert tiles)
            desert_tile_id: Which tile is the desert (skipped)
            max_attempts: Maximum placement attempts before giving up

        Returns:
            Valid number placement list (18 numbers)
        """
        # Map from non-desert tile indices to actual tile IDs
        non_desert_tiles = [i for i in range(NUM_TILES) if i != desert_tile_id]

        for attempt in range(max_attempts):
            _shuffle(numbers, rng)
            placement = [0] * 18  # Numbers for each non-desert tile

            if self._try_place_numbers(numbers, non_desert_tiles, placement, 0, rng=rng):
                return placement

        # Fallback: return shuffled numbers (may have violations)
        # This shouldn't happen with reasonable max_attempts
        _shuffle(numbers, rng)
        return numbers

    def _try_place_numbers(
        self,
        numbers: List[int],
        non_desert_tiles: List[int],
        placement: List[int],
        idx: int,
        rng=None
    ) -> bool:
        """Recursively try to place numbers using backtracking.

        Args:
            numbers: Remaining numbers to place
            non_desert_tiles: Mapping from placement index to tile ID
            placement: Current placement (modified in place)
            idx: Current position in placement

        Returns:
            True if valid placement found, False otherwise
        """
        if idx >= len(non_desert_tiles):
            return True  # All numbers placed successfully

        tile_id = non_desert_tiles[idx]

        # Get numbers already placed on adjacent tiles
        adjacent_numbers = set()
        for adj_tile in TILE_ADJACENCY.get(tile_id, set()):
            if adj_tile in non_desert_tiles:
                adj_idx = non_desert_tiles.index(adj_tile)
                if adj_idx < idx:  # Already placed
                    adjacent_numbers.add(placement[adj_idx])

        # Try each remaining number
        available = list(numbers)
        _shuffle(available, rng)  # Randomize order for variety

        for num in available:
            # Check if this number is valid here
            if not self._is_valid_number_placement(num, adjacent_numbers):
                continue

            # Place the number
            placement[idx] = num
            remaining = numbers.copy()
            remaining.remove(num)

            # Recurse
            if self._try_place_numbers(remaining, non_desert_tiles, placement, idx + 1, rng=rng):
                return True

        return False  # No valid placement found from this state

    def _is_valid_number_placement(self, num: int, adjacent_numbers: Set[int]) -> bool:
        """Check if placing a number is valid given adjacent numbers.

        Args:
            num: Number to place
            adjacent_numbers: Numbers already on adjacent tiles

        Returns:
            True if placement is valid
        """
        # Rule 1: Same numbers cannot be adjacent
        if num in adjacent_numbers:
            return False

        # Rule 2: 6 and 8 cannot be adjacent
        if num == 6 and 8 in adjacent_numbers:
            return False
        if num == 8 and 6 in adjacent_numbers:
            return False

        return True

    def _build_adjacencies(self) -> None:
        """Build adjacency relationships using a verified mapping."""
        # Use the pre-computed correct mapping
        tile_corners = get_verified_tile_corners()

        # Build tile-to-corner relationships
        for tile_id, corners in enumerate(tile_corners):
            for corner_id in corners:
                if tile_id not in self.corners[corner_id].adjacent_tiles:
                    self.corners[corner_id].adjacent_tiles.append(tile_id)

        # Build corner-to-corner adjacencies
        corner_neighbors: Dict[int, Set[int]] = {i: set() for i in range(NUM_CORNERS)}
        for corners in tile_corners:
            for i in range(6):
                c1, c2 = corners[i], corners[(i + 1) % 6]
                corner_neighbors[c1].add(c2)
                corner_neighbors[c2].add(c1)

        for corner_id, neighbors in corner_neighbors.items():
            self.corners[corner_id].adjacent_corners = sorted(neighbors)

        # Build edges
        edge_id = 0
        seen = set()
        for corner_id in range(NUM_CORNERS):
            for neighbor_id in sorted(corner_neighbors[corner_id]):
                pair = tuple(sorted([corner_id, neighbor_id]))
                if pair not in seen and edge_id < NUM_EDGES:
                    seen.add(pair)
                    self.edges[edge_id].corners = (corner_id, neighbor_id)
                    t1 = set(self.corners[corner_id].adjacent_tiles)
                    t2 = set(self.corners[neighbor_id].adjacent_tiles)
                    self.edges[edge_id].adjacent_tiles = list(t1 & t2)
                    self.corners[corner_id].adjacent_edges.append(edge_id)
                    self.corners[neighbor_id].adjacent_edges.append(edge_id)
                    edge_id += 1

    def _assign_ports(self) -> None:
        """Assign ports to coastal corners (Colonist topology)."""
        # Port assignments for Colonist board topology
        # Colonist has 6 adjacent coastal corner pairs + 6 isolated coastal corners
        # We assign ports to pairs and some isolated corners to get 9 ports total
        # Adjacent pairs: (0,5), (12,13), (20,21), (28,29), (35,36), (42,43)
        # Isolated: 9, 17, 25, 32, 39, 46
        port_assignments = [
            ([0, 5], PortType.GENERIC),      # Pair 1
            ([12, 13], PortType.WOOL),       # Pair 2
            ([20, 21], PortType.GENERIC),    # Pair 3
            ([28, 29], PortType.ORE),        # Pair 4
            ([35, 36], PortType.GENERIC),    # Pair 5
            ([42, 43], PortType.GRAIN),      # Pair 6
            ([9], PortType.GENERIC),         # Isolated 1
            ([25], PortType.BRICK),          # Isolated 2
            ([39], PortType.LUMBER),         # Isolated 3
        ]
        for corner_ids, port_type in port_assignments:
            for cid in corner_ids:
                if cid < len(self.corners):
                    self.corners[cid].port = port_type

    def get_robber_tile(self) -> int:
        for tile in self.tiles:
            if tile.has_robber:
                return tile.id
        return 0

    def move_robber(self, tile_id: int) -> None:
        for tile in self.tiles:
            tile.has_robber = False
        if 0 <= tile_id < len(self.tiles):
            self.tiles[tile_id].has_robber = True

    def get_tiles_for_number(self, number: int) -> List[int]:
        return [t.id for t in self.tiles if t.number == number]

    def get_corner_production_value(self, corner_id: int) -> float:
        if corner_id >= len(self.corners):
            return 0.0
        return sum(self.tiles[tid].production_probability
                   for tid in self.corners[corner_id].adjacent_tiles
                   if tid < len(self.tiles))

    def get_corner_resources(self, corner_id: int) -> List[Resource]:
        """Get list of resources produced at this corner (PHASE 3: position-invariant features).

        Args:
            corner_id: The corner to analyze.

        Returns:
            List of Resource types produced at this corner (may have duplicates).
        """
        if corner_id >= len(self.corners):
            return []
        resources = []
        for tid in self.corners[corner_id].adjacent_tiles:
            if tid < len(self.tiles):
                tile = self.tiles[tid]
                if tile.resource != Resource.DESERT and tile.number > 0:
                    resources.append(tile.resource)
        return resources

    def get_corner_has_good_number(self, corner_id: int) -> bool:
        """Check if corner touches a tile with 6 or 8 (best probabilities).

        Args:
            corner_id: The corner to check.

        Returns:
            True if corner touches a tile numbered 6 or 8.
        """
        if corner_id >= len(self.corners):
            return False
        for tid in self.corners[corner_id].adjacent_tiles:
            if tid < len(self.tiles):
                if self.tiles[tid].number in (6, 8):
                    return True
        return False

    def compute_corner_rankings(self) -> Dict[int, Dict[str, float]]:
        """Compute production rankings for all corners (PHASE 3: position-invariant features).

        This computes relative rankings that are board-invariant, meaning the agent learns
        to value "best corner" rather than "corner 32" (fixing position memorization).

        Returns:
            Dict mapping corner_id to:
              - production_rank: 0.0 (best) to 1.0 (worst) based on production value
              - resource_diversity: Number of unique resources / 5 (max 1.0)
              - has_6_or_8: 1.0 if touches a 6 or 8, else 0.0
        """
        # Board layout is immutable after generation, so this is computed once and
        # cached (and shared across copies via copy()). Returns the same dict object
        # on every call — callers only read it, never mutate it.
        if self._corner_rankings_cache is not None:
            return self._corner_rankings_cache

        # Compute production values for all corners
        productions = []
        for corner_id in range(NUM_CORNERS):
            prod = self.get_corner_production_value(corner_id)
            resources = self.get_corner_resources(corner_id)
            has_good = self.get_corner_has_good_number(corner_id)
            productions.append({
                'corner_id': corner_id,
                'production': prod,
                'diversity': len(set(resources)),
                'has_6_or_8': has_good
            })

        # Sort by production value (descending)
        sorted_corners = sorted(productions, key=lambda x: x['production'], reverse=True)

        # Create ranking dict
        rankings = {}
        for rank, corner_data in enumerate(sorted_corners):
            corner_id = corner_data['corner_id']
            rankings[corner_id] = {
                'production_rank': rank / (NUM_CORNERS - 1),  # 0.0 = best, 1.0 = worst
                'resource_diversity': corner_data['diversity'] / 5.0,  # Max 5 unique resources
                'has_6_or_8': 1.0 if corner_data['has_6_or_8'] else 0.0
            }
        self._corner_rankings_cache = rankings
        return rankings

    def get_adjacent_players(self, tile_id: int) -> Set[int]:
        players = set()
        for corner in self.corners:
            if tile_id in corner.adjacent_tiles and corner.owner is not None:
                players.add(corner.owner)
        return players

    def is_valid_settlement_location(self, corner_id: int, player_id: int,
                                     is_setup: bool = False,
                                     owned_edges: Set[int] = None) -> bool:
        if corner_id >= len(self.corners):
            return False
        corner = self.corners[corner_id]
        if corner.building is not None:
            return False
        for adj_id in corner.adjacent_corners:
            if self.corners[adj_id].building is not None:
                return False
        if not is_setup and owned_edges:
            if not any(eid in owned_edges for eid in corner.adjacent_edges):
                return False
        return True

    def is_valid_road_location(self, edge_id: int, player_id: int,
                               owned_edges: Set[int] = None,
                               owned_corners: Set[int] = None,
                               is_setup: bool = False,
                               setup_settlement: int = None) -> bool:
        if edge_id >= len(self.edges):
            return False
        edge = self.edges[edge_id]
        if edge.has_road:
            return False
        if is_setup and setup_settlement is not None:
            return setup_settlement in edge.corners
        owned_edges = owned_edges or set()
        owned_corners = owned_corners or set()
        for cid in edge.corners:
            if cid in owned_corners:
                return True
            for adj_eid in self.corners[cid].adjacent_edges:
                if adj_eid in owned_edges:
                    return True
        return False

    def is_valid_city_location(self, corner_id: int, player_id: int) -> bool:
        if corner_id >= len(self.corners):
            return False
        c = self.corners[corner_id]
        return c.building == 'settlement' and c.owner == player_id

    def place_settlement(self, corner_id: int, player_id: int) -> bool:
        if corner_id >= len(self.corners):
            return False
        self.corners[corner_id].building = 'settlement'
        self.corners[corner_id].owner = player_id
        return True

    def place_city(self, corner_id: int, player_id: int) -> bool:
        if corner_id >= len(self.corners):
            return False
        self.corners[corner_id].building = 'city'
        return True

    def place_road(self, edge_id: int, player_id: int) -> bool:
        if edge_id >= len(self.edges):
            return False
        self.edges[edge_id].has_road = True
        self.edges[edge_id].owner = player_id
        # Invalidate longest road cache when roads change
        # BUG FIX: Must also clear the cache dictionary, not just the valid flag
        # Otherwise, stale values persist and get returned after the first player's
        # calculation re-validates the cache
        self._road_cache_valid = False
        self._longest_road_cache = {}
        return True

    def calculate_longest_road(self, player_id: int) -> int:
        """Calculate longest road for a player with caching.

        Cache is invalidated when any road is placed.
        """
        # Return cached value if valid
        if self._road_cache_valid and player_id in self._longest_road_cache:
            return self._longest_road_cache[player_id]

        player_edges = [e for e in self.edges if e.owner == player_id]
        if not player_edges:
            result = 0
        else:
            corner_to_edges: Dict[int, List[int]] = {}
            for edge in player_edges:
                for cid in edge.corners:
                    if cid not in corner_to_edges:
                        corner_to_edges[cid] = []
                    corner_to_edges[cid].append(edge.id)

            def dfs(cid: int, visited: Set[int]) -> int:
                max_len = 0
                for eid in corner_to_edges.get(cid, []):
                    if eid in visited:
                        continue
                    edge = self.edges[eid]
                    next_cid = edge.corners[1] if edge.corners[0] == cid else edge.corners[0]
                    visited.add(eid)
                    max_len = max(max_len, 1 + dfs(next_cid, visited))
                    visited.remove(eid)
                return max_len

            result = max(dfs(cid, set()) for cid in corner_to_edges.keys())

        # Cache the result
        self._longest_road_cache[player_id] = result
        # Mark cache as valid (will be invalidated on next road placement)
        self._road_cache_valid = True
        return result

    def get_player_ports(self, player_id: int) -> List[PortType]:
        # Use set for O(1) duplicate checking instead of O(n) list search
        ports_set = set()
        for c in self.corners:
            if c.owner == player_id and c.port:
                ports_set.add(c.port)
        return list(ports_set)

    # === Strategic Analysis Methods ===

    # Resource weights for strategic evaluation
    RESOURCE_WEIGHTS = {
        Resource.GRAIN: 1.35,
        Resource.ORE: 1.33,
        Resource.LUMBER: 0.78,
        Resource.BRICK: 0.78,
        Resource.WOOL: 0.76,
    }

    # Synergy bonuses for resource pairs that work together
    SYNERGY_BONUSES = {
        (Resource.ORE, Resource.GRAIN): 0.4,      # Cities + dev cards
        (Resource.GRAIN, Resource.ORE): 0.4,
        (Resource.LUMBER, Resource.BRICK): 0.3,   # Expansion (roads + settlements)
        (Resource.BRICK, Resource.LUMBER): 0.3,
        (Resource.WOOL, Resource.GRAIN): 0.25,    # Settlements + dev cards
        (Resource.GRAIN, Resource.WOOL): 0.25,
    }

    def get_tile_production_impact(self, tile_id: int) -> Dict[int, float]:
        """Calculate the production impact of blocking a tile with the robber.

        Args:
            tile_id: The tile to analyze.

        Returns:
            Dict mapping player_id to production value lost if robber placed here.
        """
        if tile_id >= len(self.tiles):
            return {}

        tile = self.tiles[tile_id]
        production_prob = tile.production_probability
        resource = tile.resource

        if resource == Resource.DESERT or production_prob == 0:
            return {}

        # Calculate impact per player
        impact = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        resource_weight = self.RESOURCE_WEIGHTS.get(resource, 1.0)

        for corner in self.corners:
            if tile_id in corner.adjacent_tiles and corner.owner is not None:
                multiplier = 2 if corner.building == 'city' else 1
                impact[corner.owner] += production_prob * multiplier * resource_weight

        return impact

    def get_corner_synergy_score(self, corner_id: int) -> float:
        """Calculate the resource synergy score for a corner.

        Synergy occurs when a corner produces resource pairs that work together:
        - Ore + Grain: Cities and dev cards
        - Lumber + Brick: Expansion (roads + settlements)
        - Wool + Grain: Settlements and dev cards

        Args:
            corner_id: The corner to analyze.

        Returns:
            Total synergy bonus for the corner.
        """
        if corner_id >= len(self.corners):
            return 0.0

        corner = self.corners[corner_id]

        # Get resources produced by adjacent tiles
        resources_produced = []
        for tile_id in corner.adjacent_tiles:
            if tile_id < len(self.tiles):
                tile = self.tiles[tile_id]
                if tile.resource != Resource.DESERT and tile.production_probability > 0:
                    resources_produced.append((tile.resource, tile.production_probability))

        # Calculate synergy bonuses
        synergy_score = 0.0
        for i, (res1, prob1) in enumerate(resources_produced):
            for res2, prob2 in resources_produced[i+1:]:
                pair = (res1, res2)
                if pair in self.SYNERGY_BONUSES:
                    # Weighted by production probability
                    synergy_score += self.SYNERGY_BONUSES[pair] * min(prob1, prob2)
                pair_rev = (res2, res1)
                if pair_rev in self.SYNERGY_BONUSES and pair_rev != pair:
                    synergy_score += self.SYNERGY_BONUSES[pair_rev] * min(prob1, prob2)

        return synergy_score

    def get_corner_number_coordination(
        self, corner_id: int, player_id: int, exclude_corner_id: int = None
    ) -> float:
        """Calculate number coordination bonus for a corner relative to player's buildings.

        Number coordination occurs when a player has matching numbers on different
        resource types, enabling coordinated production.

        Args:
            corner_id: The corner being evaluated.
            player_id: The player to check coordination with.
            exclude_corner_id: Corner to exclude from player's buildings (use when
                evaluating a corner that was just placed to avoid self-matching).

        Returns:
            Coordination bonus (0.3 per matching number pair).
        """
        if corner_id >= len(self.corners):
            return 0.0

        corner = self.corners[corner_id]

        # Get numbers at this corner
        corner_numbers = set()
        for tile_id in corner.adjacent_tiles:
            if tile_id < len(self.tiles):
                tile = self.tiles[tile_id]
                if tile.number > 0 and tile.resource != Resource.DESERT:
                    corner_numbers.add(tile.number)

        # Get numbers from player's existing buildings (excluding the corner being evaluated)
        player_numbers = set()
        for idx, c in enumerate(self.corners):
            # Skip the corner being evaluated to avoid self-matching
            if idx == exclude_corner_id:
                continue
            if c.owner == player_id:
                for tile_id in c.adjacent_tiles:
                    if tile_id < len(self.tiles):
                        tile = self.tiles[tile_id]
                        if tile.number > 0 and tile.resource != Resource.DESERT:
                            player_numbers.add(tile.number)

        # Count matching numbers (coordination)
        matching = corner_numbers & player_numbers
        return 0.3 * len(matching)

    def get_weighted_production_value(self, corner_id: int) -> float:
        """Calculate weighted production value for a corner.

        Uses resource weights to value high-demand resources more.

        Args:
            corner_id: The corner to analyze.

        Returns:
            Weighted production value.
        """
        if corner_id >= len(self.corners):
            return 0.0

        corner = self.corners[corner_id]
        weighted_value = 0.0

        for tile_id in corner.adjacent_tiles:
            if tile_id < len(self.tiles):
                tile = self.tiles[tile_id]
                if tile.resource != Resource.DESERT:
                    weight = self.RESOURCE_WEIGHTS.get(tile.resource, 1.0)
                    weighted_value += tile.production_probability * weight

        return weighted_value

    def get_reachable_ports(self, corner_id: int, max_distance: int = 3) -> List[Tuple[PortType, int]]:
        """Find ports reachable from a corner within a road distance.

        Args:
            corner_id: Starting corner.
            max_distance: Maximum road segments to traverse.

        Returns:
            List of (port_type, distance) tuples, sorted by distance.
        """
        if corner_id >= len(self.corners):
            return []

        # BFS to find reachable ports
        from collections import deque

        visited = {corner_id}
        queue = deque([(corner_id, 0)])
        ports_found = []

        while queue:
            current_corner, distance = queue.popleft()

            # Check for port at this corner
            if current_corner < len(self.corners):
                corner = self.corners[current_corner]
                if corner.port is not None:
                    ports_found.append((corner.port, distance))

            # Continue BFS if within distance limit
            if distance < max_distance:
                corner = self.corners[current_corner]
                for adj_corner_id in corner.adjacent_corners:
                    if adj_corner_id not in visited:
                        visited.add(adj_corner_id)
                        # Only traverse if not blocked by opponent building
                        adj_corner = self.corners[adj_corner_id]
                        # (In setup, assume all paths are open)
                        queue.append((adj_corner_id, distance + 1))

        # Sort by distance and remove duplicates
        unique_ports = {}
        for port_type, dist in ports_found:
            if port_type not in unique_ports or dist < unique_ports[port_type]:
                unique_ports[port_type] = dist

        return sorted(unique_ports.items(), key=lambda x: x[1])

    def get_road_usefulness(self, edge_id: int, player_id: int) -> Tuple[float, bool, int]:
        """Analyze the strategic usefulness of a road placement.

        Args:
            edge_id: The edge to analyze.
            player_id: The player considering the road.

        Returns:
            Tuple of (usefulness_score, is_dead_end, continuation_potential).
        """
        if edge_id >= len(self.edges):
            return (0.0, True, 0)

        edge = self.edges[edge_id]
        owned_corners = set()
        owned_edges = set()

        for c in self.corners:
            if c.owner == player_id:
                owned_corners.add(c.id)
        for e in self.edges:
            if e.owner == player_id:
                owned_edges.add(e.id)

        usefulness = 0.0
        continuation_potential = 0
        dead_end = True

        # Check each endpoint of the road
        for corner_id in edge.corners:
            if corner_id >= len(self.corners):
                continue
            corner = self.corners[corner_id]

            # Does this enable a valid settlement spot?
            can_settle = self.is_valid_settlement_location(
                corner_id, player_id, is_setup=False, owned_edges=owned_edges | {edge_id}
            )
            if can_settle:
                prod_value = self.get_corner_production_value(corner_id)
                usefulness += 0.5 + prod_value
                dead_end = False

            # Does this connect to more road building options?
            for adj_corner_id in corner.adjacent_corners:
                if adj_corner_id < len(self.corners):
                    adj_corner = self.corners[adj_corner_id]
                    # Check if there's an edge to this adjacent corner we could build
                    for adj_edge_id in corner.adjacent_edges:
                        if adj_edge_id != edge_id and adj_edge_id not in owned_edges:
                            adj_edge = self.edges[adj_edge_id]
                            if not adj_edge.has_road:
                                continuation_potential += 1
                                dead_end = False

        usefulness += 0.1 * continuation_potential

        return (usefulness, dead_end, continuation_potential)

    def get_road_blocking_value(self, edge_id: int, player_id: int) -> float:
        """Calculate the value of a road for blocking opponents.

        Args:
            edge_id: The edge to analyze.
            player_id: The player considering the road.

        Returns:
            Blocking value based on how much it restricts opponents.
        """
        if edge_id >= len(self.edges):
            return 0.0

        edge = self.edges[edge_id]
        blocking_value = 0.0

        # Check each endpoint
        for corner_id in edge.corners:
            if corner_id >= len(self.corners):
                continue
            corner = self.corners[corner_id]

            # Count nearby opponent roads that this blocks expansion from
            for adj_edge_id in corner.adjacent_edges:
                if adj_edge_id < len(self.edges):
                    adj_edge = self.edges[adj_edge_id]
                    if adj_edge.owner is not None and adj_edge.owner != player_id:
                        # This road blocks an opponent's expansion path
                        prod_value = self.get_corner_production_value(corner_id)
                        blocking_value += 0.15 + 0.1 * prod_value

        return blocking_value

    def copy(self) -> 'Board':
        new_board = Board.__new__(Board)
        new_board.tiles = [Tile(t.id, t.resource, t.number, t.has_robber) for t in self.tiles]
        new_board.corners = [
            Corner(c.id, c.adjacent_tiles.copy(), c.adjacent_corners.copy(),
                   c.adjacent_edges.copy(), c.building, c.owner, c.port)
            for c in self.corners
        ]
        new_board.edges = [
            Edge(e.id, e.corners, e.adjacent_tiles.copy(), e.has_road, e.owner)
            for e in self.edges
        ]
        # Copy cached mappings
        new_board._tile_to_corners = {k: v.copy() for k, v in self._tile_to_corners.items()}
        new_board._longest_road_cache = self._longest_road_cache.copy()
        new_board._road_cache_valid = self._road_cache_valid
        # Share the corner-ranking cache by reference: the copied board has the same
        # (immutable) tile layout, and the dict is read-only, so this is safe and makes
        # the ranking a compute-once-per-game cost instead of once-per-node.
        new_board._corner_rankings_cache = self._corner_rankings_cache
        return new_board


def get_verified_tile_corners() -> List[List[int]]:
    """Return tile-to-corner mapping for Colonist.io board topology.

    This mapping was derived from Colonist's corner adjacency graph to ensure
    compatibility with BC training data collected from Colonist games.

    - Each tile has exactly 6 unique corners
    - Adjacent tiles share exactly 2 corners (1 edge)
    - Total edges = 72
    - 18 corners touch 1 tile (outer ring)
    - 12 corners touch 2 tiles (middle ring)
    - 24 corners touch 3 tiles (inner corners)
    """
    return [
        [0, 5, 4, 3, 2, 1],          # Tile 0
        [1, 46, 45, 44, 47, 2],      # Tile 1
        [2, 47, 48, 49, 6, 3],       # Tile 2
        [3, 6, 7, 8, 9, 4],          # Tile 3
        [6, 49, 50, 14, 10, 7],      # Tile 4
        [7, 10, 11, 12, 13, 8],      # Tile 5
        [10, 14, 15, 16, 17, 11],    # Tile 6
        [14, 50, 51, 22, 18, 15],    # Tile 7
        [15, 18, 19, 20, 21, 16],    # Tile 8
        [18, 22, 23, 24, 25, 19],    # Tile 9
        [22, 51, 52, 33, 26, 23],    # Tile 10
        [23, 26, 27, 28, 29, 24],    # Tile 11
        [26, 33, 30, 31, 32, 27],    # Tile 12
        [30, 37, 40, 53, 52, 33],    # Tile 13
        [30, 37, 34, 35, 36, 31],    # Tile 14
        [34, 39, 38, 41, 40, 37],    # Tile 15
        [38, 43, 42, 45, 44, 41],    # Tile 16
        [40, 53, 48, 47, 44, 41],    # Tile 17
        [48, 53, 52, 51, 50, 49],    # Tile 18
    ]
