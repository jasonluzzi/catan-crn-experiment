"""State encoder for converting Catan game state to neural network input.

OPTIMIZATION NOTES:
- Pre-allocate observation buffer and reuse it
- Use numpy vectorized operations where possible
- Cache frequently computed values
- Minimize Python loops in hot paths
"""

import numpy as np
from typing import List, Optional, Dict, Tuple

from ..game.constants import (
    Resource, DevCardType, GamePhase, PortType,
    NUM_PLAYERS, NUM_TILES, NUM_CORNERS, NUM_EDGES,
    ROLL_PROBABILITY, INITIAL_BANK_RESOURCES, MAX_COUNTER_OFFER_DEPTH,
    DEV_CARD_COUNTS
)
from ..game.game_state import GameState, RollHistoryEntry
from ..game.trade_templates import TRADE_TEMPLATES, NUM_TRADE_TEMPLATES


# Pre-compute resource list for faster iteration
_TRADEABLE_RESOURCES = list(Resource.tradeable())
_DEV_CARD_TYPES = list(DevCardType)
_PORT_TYPES = list(PortType)


class StateEncoder:
    """Encodes Catan game state into a fixed-size observation array.

    Observation shape: (176, 65)
    - Row 0: Game context (turn, phase, longest road holder, trade state)
    - Rows 1-19: Tile features (resource, number, robber, adjacent buildings)
    - Rows 20-73: Corner features (54 corners)
    - Rows 74-145: Edge features (72 edges)
    - Rows 146-149: Player features (4 players)
    - Rows 150-154: Bank resources (5 resource types)
    - Rows 155-174: Roll history (last 20 dice rolls)
    - Row 175: Pending trade offer details
    """

    # Feature dimensions
    FEATURE_DIM = 65
    GAME_CONTEXT_ROWS = 1
    TILE_ROWS = 19
    CORNER_ROWS = 54
    EDGE_ROWS = 72
    PLAYER_ROWS = 4
    BANK_ROWS = 5
    ROLL_HISTORY_ROWS = 20
    TRADE_ROWS = 1
    TOTAL_ROWS = (GAME_CONTEXT_ROWS + TILE_ROWS + CORNER_ROWS + EDGE_ROWS +
                  PLAYER_ROWS + BANK_ROWS + ROLL_HISTORY_ROWS + TRADE_ROWS)  # 176

    def __init__(self):
        """Initialize the state encoder."""
        self.observation_shape = (self.TOTAL_ROWS, self.FEATURE_DIM)
        # OPTIMIZATION: Pre-allocate observation buffer for reuse
        self._obs_buffer = np.zeros(self.observation_shape, dtype=np.float32)
        # Cache for player order computation
        self._player_order_cache: Dict[int, List[int]] = {
            i: [(i + j) % NUM_PLAYERS for j in range(NUM_PLAYERS)]
            for i in range(NUM_PLAYERS)
        }
        # Cache for tile->corners mapping (lazy-loaded from board)
        self._tile_corners_cache: Dict[int, List[int]] = {}
        self._tile_corners_cache_initialized: bool = False

    def encode(self, state: GameState, player_perspective: int) -> np.ndarray:
        """Encode the game state from a player's perspective.

        Args:
            state: The game state to encode.
            player_perspective: The player ID from whose perspective to encode.

        Returns:
            Normalized observation array of shape (150, 65).

        OPTIMIZATION: Reuses pre-allocated buffer and uses cached player order.
        """
        # OPTIMIZATION: Reuse buffer instead of allocating new array each time
        obs = self._obs_buffer
        obs.fill(0.0)  # Reset to zeros - faster than allocating new array

        row = 0

        # Game context (row 0)
        self._encode_game_context_inplace(obs[row], state, player_perspective)
        row += 1

        # OPTIMIZATION: Pre-compute commonly accessed values
        board = state.board
        tiles = board.tiles
        corners = board.corners
        edges = board.edges
        player = state.players[player_perspective]
        owned_edges = player.get_owned_edges()
        owned_corners = player.get_owned_corners()

        # Lazy-load tile->corners cache from board (built once per encoder)
        if not self._tile_corners_cache_initialized:
            self._tile_corners_cache = board._tile_to_corners
            self._tile_corners_cache_initialized = True

        # Tile features (rows 1-19)
        for tile_id in range(NUM_TILES):
            self._encode_tile_inplace(obs[row], tiles, tile_id, corners, player_perspective)
            row += 1

        # PHASE 3: Compute corner rankings for position-invariant features
        # This helps the agent learn "best production" vs "corner 32" (fixes memorization)
        corner_rankings = board.compute_corner_rankings()

        # Corner features (rows 20-73)
        for corner_id in range(NUM_CORNERS):
            self._encode_corner_inplace(obs[row], state, board, corners, tiles,
                                        corner_id, player_perspective, owned_edges,
                                        corner_rankings)
            row += 1

        # Edge features (rows 74-145)
        for edge_id in range(NUM_EDGES):
            self._encode_edge_inplace(obs[row], board, edges, corners,
                                      edge_id, player_perspective, owned_edges, owned_corners,
                                      state.is_setup_phase, state.last_settlement_placed)
            row += 1

        # Player features (rows 146-149)
        # OPTIMIZATION: Use cached player order
        player_order = self._player_order_cache[player_perspective]
        for player_id in player_order:
            self._encode_player_inplace(obs[row], state, player_id, player_perspective)
            row += 1

        # Bank resources (rows 150-154)
        for i, resource in enumerate(_TRADEABLE_RESOURCES):
            self._encode_bank_resource_inplace(obs[row], state, resource)
            row += 1

        # Roll history (rows 155-174)
        for i in range(self.ROLL_HISTORY_ROWS):
            if i < len(state.roll_history):
                self._encode_roll_history_inplace(obs[row], state.roll_history[-(i+1)], state, i, player_perspective)
            # else: row stays as zeros
            row += 1

        # Pending trade (row 175)
        self._encode_trade_inplace(obs[row], state, player_perspective)
        row += 1

        # Return a copy to avoid buffer reuse issues with async envs
        return obs.copy()

    def _encode_game_context_inplace(self, features: np.ndarray, state: GameState, perspective: int) -> None:
        """Encode game-level context features directly into pre-allocated array.

        OPTIMIZATION: Writes directly to provided array instead of allocating new one.
        """
        idx = 0

        # Turn number (normalized)
        features[idx] = min(state.turn_number * 0.01, 1.0)  # Avoid division
        idx += 1

        # Game phase (one-hot, 12 phases: includes TRADE_RESPONSE, TRADE_CONFIRM)
        phase_idx = int(state.phase)
        if idx + phase_idx < self.FEATURE_DIM:
            features[idx + phase_idx] = 1.0
        idx += 12

        # Current player relative to perspective
        relative_player = (state.current_player_idx - perspective) % NUM_PLAYERS
        features[idx] = relative_player / (NUM_PLAYERS - 1)
        idx += 1

        # Is it my turn?
        features[idx] = 1.0 if state.current_player_idx == perspective else 0.0
        idx += 1

        # Dice rolled this turn?
        features[idx] = 1.0 if state.dice_rolled else 0.0
        idx += 1

        # Last dice roll (normalized) - use multiplication instead of division
        features[idx] = (state.last_dice_roll[0] - 1) * 0.2
        idx += 1
        features[idx] = (state.last_dice_roll[1] - 1) * 0.2
        idx += 1
        features[idx] = (state.last_dice_roll[0] + state.last_dice_roll[1] - 2) * 0.1
        idx += 1

        # Longest road holder (relative)
        if state.longest_road_holder is not None:
            relative = (state.longest_road_holder - perspective) % NUM_PLAYERS
            features[idx] = (relative + 1) / NUM_PLAYERS
        idx += 1

        # Largest army holder (relative)
        if state.largest_army_holder is not None:
            relative = (state.largest_army_holder - perspective) % NUM_PLAYERS
            features[idx] = (relative + 1) / NUM_PLAYERS
        idx += 1

        # Dev cards remaining in deck (normalized)
        features[idx] = len(state.dev_card_deck) * 0.04  # 1/25 = 0.04
        idx += 1

        # Is setup phase?
        features[idx] = 1.0 if state.is_setup_phase else 0.0
        idx += 1

        # Setup phase index
        features[idx] = state.setup_phase_idx * 0.125  # 1/8 = 0.125
        idx += 1

        # Robber location tile ID
        robber_tile = state.board.get_robber_tile()
        features[idx] = robber_tile / (NUM_TILES - 1)
        idx += 1

        # === Global Dev Card State ===
        dev_cards_remaining = state.get_dev_cards_remaining()

        # Monopolies remaining (0-2)
        features[idx] = dev_cards_remaining.get(DevCardType.MONOPOLY, 0) / 2.0
        idx += 1

        # Road buildings remaining (0-2)
        features[idx] = dev_cards_remaining.get(DevCardType.ROAD_BUILDING, 0) / 2.0
        idx += 1

        # Knights remaining (0-14)
        features[idx] = dev_cards_remaining.get(DevCardType.KNIGHT, 0) / 14.0
        idx += 1

        # Year of Plenty remaining (0-2)
        features[idx] = dev_cards_remaining.get(DevCardType.YEAR_OF_PLENTY, 0) / 2.0
        idx += 1

        # Estimated VP cards drawn (for hidden VP threat assessment)
        features[idx] = state.get_estimated_vp_cards_drawn() / 5.0

    def _encode_game_context(self, state: GameState, perspective: int) -> np.ndarray:
        """Encode game-level context features (legacy method for compatibility)."""
        features = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        self._encode_game_context_inplace(features, state, perspective)
        return features

    def _encode_tile_inplace(self, features: np.ndarray, tiles: list, tile_id: int, corners: list, perspective: int) -> None:
        """Encode features for a single tile directly into pre-allocated array.

        OPTIMIZATION: Accepts pre-fetched tiles and corners lists to avoid repeated attribute access.
        BUG FIX: Now encodes adjacent buildings in relative player order (not absolute).
        """
        idx = 0

        if tile_id >= len(tiles):
            return

        tile = tiles[tile_id]

        # Resource type (one-hot, 6 types including desert)
        res_idx = int(tile.resource)
        if idx + res_idx < self.FEATURE_DIM:
            features[idx + res_idx] = 1.0
        idx += 6

        # Number token (normalized) - use multiplication
        features[idx] = tile.number * 0.0833  # 1/12 approx
        idx += 1

        # Roll probability
        features[idx] = tile.production_probability
        idx += 1

        # Has robber?
        features[idx] = 1.0 if tile.has_robber else 0.0
        idx += 1

        # Adjacent buildings count per player - OPTIMIZED: use precomputed tile->corners mapping
        # Initialize counts
        player_settlements = [0, 0, 0, 0]
        player_cities = [0, 0, 0, 0]

        # Use precomputed mapping for O(6) instead of O(54) iteration
        for corner_id in self._tile_corners_cache.get(tile_id, []):
            corner = corners[corner_id]
            if corner.owner is not None:
                if corner.building == 'settlement':
                    player_settlements[corner.owner] += 1
                elif corner.building == 'city':
                    player_cities[corner.owner] += 1

        # BUG FIX: Encode in RELATIVE player order (perspective first, then others)
        # This prevents the model from learning "slot 0 = P0" which caused P0 bias
        for rel_idx in range(NUM_PLAYERS):
            player_id = (perspective + rel_idx) % NUM_PLAYERS
            features[idx] = player_settlements[player_id] * 0.1667  # 1/6 approx
            idx += 1
            features[idx] = player_cities[player_id] * 0.1667
            idx += 1

    def _encode_tile(self, state: GameState, tile_id: int, perspective: int = 0) -> np.ndarray:
        """Encode features for a single tile (legacy method for compatibility)."""
        features = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        self._encode_tile_inplace(features, state.board.tiles, tile_id, state.board.corners, perspective)
        return features

    def _encode_corner_inplace(self, features: np.ndarray, state: GameState, board,
                               corners: list, tiles: list, corner_id: int,
                               perspective: int, owned_edges: set,
                               corner_rankings: dict = None) -> None:
        """Encode features for a single corner directly into pre-allocated array.

        OPTIMIZATION: Accepts pre-fetched data structures and pre-computed owned_edges.

        PHASE 3: Now includes position-invariant features (production_rank, resource_diversity,
        has_6_or_8) to fix position memorization where agent always picks "corner 32".
        """
        idx = 0

        if corner_id >= len(corners):
            return

        corner = corners[corner_id]

        # Building type (one-hot: empty, settlement, city)
        if corner.building is None:
            features[idx] = 1.0
        elif corner.building == 'settlement':
            features[idx + 1] = 1.0
        else:  # city
            features[idx + 2] = 1.0
        idx += 3

        # Owner (relative to perspective, one-hot including no owner)
        if corner.owner is not None:
            relative = (corner.owner - perspective) % NUM_PLAYERS
            features[idx + relative + 1] = 1.0
        else:
            features[idx] = 1.0  # No owner
        idx += NUM_PLAYERS + 1

        # Port type (one-hot: none, generic, lumber, brick, wool, grain, ore)
        if corner.port is None:
            features[idx] = 1.0
        else:
            port_idx = int(corner.port) + 1
            if idx + port_idx < self.FEATURE_DIM:
                features[idx + port_idx] = 1.0
        idx += 7

        # Production value (sum of probabilities from adjacent tiles) - OPTIMIZED: inline calculation
        prod_value = 0.0
        for tid in corner.adjacent_tiles:
            if tid < len(tiles):
                prod_value += tiles[tid].production_probability
        features[idx] = prod_value
        idx += 1

        # Is valid for settlement (current player)? - OPTIMIZED: use pre-computed owned_edges
        is_valid = board.is_valid_settlement_location(
            corner_id, perspective,
            is_setup=state.is_setup_phase,
            owned_edges=owned_edges
        )
        features[idx] = 1.0 if is_valid else 0.0
        idx += 1

        # Adjacent to my buildings?
        my_adjacent = False
        for adj_corner_id in corner.adjacent_corners:
            if adj_corner_id < len(corners):
                if corners[adj_corner_id].owner == perspective:
                    my_adjacent = True
                    break
        features[idx] = 1.0 if my_adjacent else 0.0
        idx += 1

        # Resource production per type from adjacent tiles - OPTIMIZED: single pass
        # Initialize all to 0 (already done by caller)
        for res in _TRADEABLE_RESOURCES:
            prod = 0.0
            for tile_id in corner.adjacent_tiles:
                if tile_id < len(tiles):
                    tile = tiles[tile_id]
                    if tile.resource == res:
                        prod += tile.production_probability
            features[idx] = prod
            idx += 1

        # === PHASE 3: Position-invariant features (fixes position memorization) ===
        # These features teach the agent to value "best production corner" rather than
        # memorizing "corner 32" which varies with board randomization.
        if corner_rankings is not None and corner_id in corner_rankings:
            ranking = corner_rankings[corner_id]
            # Production rank: 0.0 = best corner, 1.0 = worst corner
            # Inverted so higher = better (consistent with other features)
            features[idx] = 1.0 - ranking['production_rank']
            idx += 1
            # Resource diversity: how many unique resources (0.0 to 1.0)
            features[idx] = ranking['resource_diversity']
            idx += 1
            # Has 6 or 8: touches high-probability tile
            features[idx] = ranking['has_6_or_8']
            idx += 1
        else:
            idx += 3  # Skip if no rankings provided

    def _encode_corner(self, state: GameState, corner_id: int,
                       perspective: int) -> np.ndarray:
        """Encode features for a single corner (legacy method for compatibility)."""
        features = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        player = state.players[perspective]
        corner_rankings = state.board.compute_corner_rankings()
        self._encode_corner_inplace(
            features, state, state.board, state.board.corners, state.board.tiles,
            corner_id, perspective, player.get_owned_edges(), corner_rankings
        )
        return features

    def _encode_edge_inplace(self, features: np.ndarray, board, edges: list, corners: list,
                             edge_id: int, perspective: int, owned_edges: set,
                             owned_corners: set, is_setup: bool, last_settlement_placed) -> None:
        """Encode features for a single edge directly into pre-allocated array.

        OPTIMIZATION: Accepts pre-fetched data and pre-computed owned sets.
        """
        idx = 0

        if edge_id >= len(edges):
            return

        edge = edges[edge_id]

        # Has road?
        features[idx] = 1.0 if edge.has_road else 0.0
        idx += 1

        # Owner (relative to perspective, one-hot including no owner)
        if edge.owner is not None:
            relative = (edge.owner - perspective) % NUM_PLAYERS
            features[idx + relative + 1] = 1.0
        else:
            features[idx] = 1.0  # No owner
        idx += NUM_PLAYERS + 1

        # Is valid for road (current player)? - OPTIMIZED: use pre-computed sets
        is_valid = board.is_valid_road_location(
            edge_id, perspective,
            owned_edges=owned_edges,
            owned_corners=owned_corners,
            is_setup=is_setup,
            setup_settlement=last_settlement_placed
        )
        features[idx] = 1.0 if is_valid else 0.0
        idx += 1

        # Connected to my building? - OPTIMIZED: use pre-computed owned_corners
        connects_to_building = edge.corners[0] in owned_corners or edge.corners[1] in owned_corners
        features[idx] = 1.0 if connects_to_building else 0.0
        idx += 1

        # Connected to my road? - OPTIMIZED: use pre-computed owned_edges
        connects_to_road = False
        for corner_id in edge.corners:
            if corner_id < len(corners):
                corner = corners[corner_id]
                for adj_edge_id in corner.adjacent_edges:
                    if adj_edge_id != edge_id and adj_edge_id in owned_edges:
                        connects_to_road = True
                        break
                if connects_to_road:
                    break
        features[idx] = 1.0 if connects_to_road else 0.0

    def _encode_edge(self, state: GameState, edge_id: int,
                     perspective: int) -> np.ndarray:
        """Encode features for a single edge (legacy method for compatibility)."""
        features = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        player = state.players[perspective]
        self._encode_edge_inplace(
            features, state.board, state.board.edges, state.board.corners,
            edge_id, perspective, player.get_owned_edges(), player.get_owned_corners(),
            state.is_setup_phase, state.last_settlement_placed
        )
        return features

    def _encode_player_inplace(self, features: np.ndarray, state: GameState,
                               player_id: int, perspective: int) -> None:
        """Encode features for a single player directly into pre-allocated array.

        OPTIMIZATION: Writes directly to provided array, uses pre-computed constants.
        """
        idx = 0

        player = state.players[player_id]
        is_self = player_id == perspective

        # Relative position to perspective
        relative = (player_id - perspective) % NUM_PLAYERS
        features[idx] = relative / (NUM_PLAYERS - 1)
        idx += 1

        # Is this the perspective player?
        features[idx] = 1.0 if is_self else 0.0
        idx += 1

        # Victory points (visible) - use multiplication
        features[idx] = player.victory_points * 0.1
        idx += 1

        # Total VP (only for self)
        if is_self:
            features[idx] = player.total_victory_points * 0.1
        idx += 1

        # Resources (only known for self)
        if is_self:
            for res in _TRADEABLE_RESOURCES:
                features[idx] = min(player.resources.get(res, 0) * 0.1, 1.0)
                idx += 1
            features[idx] = min(player.total_resources * 0.05, 1.0)
            idx += 1
        else:
            idx += 6
            # Total resources (visible to all)
            features[idx] = min(player.total_resources * 0.05, 1.0)
            idx += 1

        # Dev cards (only known for self)
        if is_self:
            for card_type in _DEV_CARD_TYPES:
                features[idx] = min(player.dev_cards.get(card_type, 0) * 0.2, 1.0)
                idx += 1
        else:
            # Total dev cards (visible)
            total_dev = sum(player.dev_cards.values())
            features[idx] = min(total_dev * 0.1, 1.0)
            idx += 5

        # Knights played
        features[idx] = player.knights_played * 0.2
        idx += 1

        # Has longest road?
        features[idx] = 1.0 if player.has_longest_road else 0.0
        idx += 1

        # Has largest army?
        features[idx] = 1.0 if player.has_largest_army else 0.0
        idx += 1

        # Longest road length
        features[idx] = player.longest_road_length * 0.0667  # 1/15
        idx += 1

        # Buildings count
        features[idx] = len(player.settlements) * 0.2
        idx += 1
        features[idx] = len(player.cities) * 0.25
        idx += 1
        features[idx] = len(player.roads) * 0.0667
        idx += 1

        # Ports (for self)
        if is_self:
            player_ports_set = set(player.ports)  # Convert to set for O(1) lookup
            for port_type in _PORT_TYPES:
                features[idx] = 1.0 if port_type in player_ports_set else 0.0
                idx += 1
        else:
            idx += len(_PORT_TYPES)

        # Resource production (settlements + 2*cities adjacent to each number)
        # OPTIMIZATION: Single pass over buildings, accumulate per-resource production
        if is_self:
            corners = state.board.corners
            tiles = state.board.tiles

            # Initialize production accumulators
            production = {res: 0.0 for res in _TRADEABLE_RESOURCES}

            # Single pass: iterate buildings once, accumulate all resource production
            for corner_id in player.settlements:
                if corner_id < len(corners):
                    for tile_id in corners[corner_id].adjacent_tiles:
                        if tile_id < len(tiles):
                            tile = tiles[tile_id]
                            if tile.resource in production and not tile.has_robber:
                                production[tile.resource] += tile.production_probability

            for corner_id in player.cities:
                if corner_id < len(corners):
                    for tile_id in corners[corner_id].adjacent_tiles:
                        if tile_id < len(tiles):
                            tile = tiles[tile_id]
                            if tile.resource in production and not tile.has_robber:
                                production[tile.resource] += tile.production_probability * 2

            # Write to features array
            for res in _TRADEABLE_RESOURCES:
                features[idx] = min(production[res], 1.0)
                idx += 1

        # === Dev Card Inference Features (visible for all players) ===
        # These help infer hidden non-knight dev cards

        # Dev card hold duration (average turns held, normalized by 20)
        hold_duration = player.get_dev_card_hold_duration(state.turn_number)
        features[idx] = min(hold_duration / 20.0, 1.0)
        idx += 1

        # Knight play ratio (times played knight when blocked)
        knight_ratio = player.get_knight_play_ratio()
        features[idx] = knight_ratio
        idx += 1

        # Estimated non-knight dev cards (normalized by total)
        likely_non_knights = player.get_likely_hidden_non_knights()
        total_dev = player.total_dev_cards
        if total_dev > 0:
            features[idx] = likely_non_knights / total_dev
        else:
            features[idx] = 0.0
        idx += 1

        # Times blocked with cards (normalized)
        features[idx] = min(player.times_blocked_with_cards / 10.0, 1.0)
        idx += 1

        # Total dev cards held (visible to all)
        features[idx] = min(player.total_dev_cards / 10.0, 1.0)

    def _encode_player(self, state: GameState, player_id: int,
                       perspective: int) -> np.ndarray:
        """Encode features for a single player (legacy method for compatibility)."""
        features = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        self._encode_player_inplace(features, state, player_id, perspective)
        return features

    def _get_player_order(self, perspective: int) -> List[int]:
        """Get player order starting from perspective player."""
        return [(perspective + i) % NUM_PLAYERS for i in range(NUM_PLAYERS)]

    def _encode_bank_resource_inplace(self, features: np.ndarray, state: GameState,
                                       resource: Resource) -> None:
        """Encode bank resource information for a single resource type."""
        idx = 0

        # Resource type (one-hot, 5 types)
        res_idx = int(resource)
        if res_idx < 5:
            features[idx + res_idx] = 1.0
        idx += 5

        # Amount in bank (normalized by initial amount)
        bank_amount = state.bank_resources.get(resource, INITIAL_BANK_RESOURCES)
        features[idx] = bank_amount / INITIAL_BANK_RESOURCES
        idx += 1

        # Is scarce? (less than 5 remaining)
        features[idx] = 1.0 if bank_amount < 5 else 0.0
        idx += 1

        # Is depleted? (0 remaining)
        features[idx] = 1.0 if bank_amount == 0 else 0.0

    def _encode_roll_history_inplace(self, features: np.ndarray,
                                      roll_entry: 'RollHistoryEntry',
                                      state: GameState, age: int,
                                      perspective: int) -> None:
        """Encode a single roll history entry.

        Args:
            features: Output feature array.
            roll_entry: The roll history entry.
            state: Current game state (for context).
            age: How many rolls ago (0 = most recent).
            perspective: The player perspective (for relative encoding).
        """
        idx = 0

        # Dice total (normalized 2-12)
        features[idx] = (roll_entry.dice_total - 2) / 10.0
        idx += 1

        # Age of roll (normalized, 0 = most recent)
        features[idx] = age / self.ROLL_HISTORY_ROWS
        idx += 1

        # Robber moved?
        features[idx] = 1.0 if roll_entry.robber_moved else 0.0
        idx += 1

        # Turn number when this roll occurred (normalized)
        features[idx] = min(roll_entry.turn_number * 0.01, 1.0)
        idx += 1

        # Resources gained per player (4 players × 5 resources = 20 features)
        # BUG FIX: Encode in RELATIVE player order (perspective first, then others)
        for rel_idx in range(NUM_PLAYERS):
            player_id = (perspective + rel_idx) % NUM_PLAYERS
            for resource in _TRADEABLE_RESOURCES:
                gained = roll_entry.resources_gained.get(player_id, {}).get(resource, 0)
                features[idx] = min(gained * 0.25, 1.0)  # Normalize by 4 (max from city)
                idx += 1

        # Total resources gained this roll
        total_gained = sum(
            sum(res.values())
            for res in roll_entry.resources_gained.values()
        )
        features[idx] = min(total_gained * 0.1, 1.0)

    def _encode_trade_inplace(self, features: np.ndarray, state: GameState,
                               perspective: int) -> None:
        """Encode pending trade information."""
        idx = 0

        # Is there a pending trade?
        has_trade = state.pending_trade is not None
        features[idx] = 1.0 if has_trade else 0.0
        idx += 1

        if not has_trade:
            return

        trade = state.pending_trade

        # Trade phase encoding (one-hot)
        in_response = state.phase == GamePhase.TRADE_RESPONSE
        in_confirm = state.phase == GamePhase.TRADE_CONFIRM
        features[idx] = 1.0 if in_response else 0.0
        idx += 1
        features[idx] = 1.0 if in_confirm else 0.0
        idx += 1

        # Am I the proposer?
        features[idx] = 1.0 if trade.proposer_id == perspective else 0.0
        idx += 1

        # Am I the current responder?
        features[idx] = 1.0 if state.current_responder_idx == perspective else 0.0
        idx += 1

        # Counter depth (normalized)
        features[idx] = trade.counter_depth / MAX_COUNTER_OFFER_DEPTH
        idx += 1

        # Resources being offered (5 values)
        for resource in _TRADEABLE_RESOURCES:
            features[idx] = min(trade.offering.get(resource, 0) * 0.25, 1.0)
            idx += 1

        # Resources being requested (5 values)
        for resource in _TRADEABLE_RESOURCES:
            features[idx] = min(trade.requesting.get(resource, 0) * 0.25, 1.0)
            idx += 1

        # Number of accepting players
        features[idx] = len(state.accepting_players) / 3.0
        idx += 1

        # Am I in the accepting players?
        features[idx] = 1.0 if perspective in state.accepting_players else 0.0
        idx += 1

        # Number of responders remaining
        features[idx] = len(state.responders_remaining) / 3.0
