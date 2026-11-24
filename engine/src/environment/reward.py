"""Reward calculation for the Catan RL environment."""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from ..game.game_state import GameState
from ..game.board import Board
from ..game.player import Player
from ..game.constants import ActionType, Resource, PortType, NUM_PLAYERS


@dataclass
class RewardConfig:
    """Configuration for reward calculation.

    PHASE 3 REBALANCING (Reward Hacking Fix):
    - Win/lose rewards increased 5x (10/-5 -> 50/-25) to dominate dense rewards
    - Dense scale reduced 4x (0.2 -> 0.05) to prevent dense reward exploitation
    - Trade exploit rewards removed (were enabling counter-offer spam)
    - Counter-offer penalty added to discourage spam (212/game -> target <5)
    - VP velocity rewards added to encourage pushing for 10 VP

    PHASE 3.1 REWARD SHAPING (City Building Fix):
    - City reward doubled (2.0 -> 4.0) to make cities more attractive than roads
    - Setup pip bonus quadrupled (0.5 -> 2.0) for better opening play
    - Resource production rewards added to create immediate feedback
    - Monopoly/Year of Plenty rewards scaled by cards gained
    """
    # Win/lose rewards (5x increase to dominate dense rewards)
    win_reward: float = 50.0                   # was 10.0
    lose_reward: float = -25.0                 # was -5.0

    # Dense reward scale (4x reduction to prevent exploitation)
    dense_scale: float = 0.05                  # was 0.2

    # Building rewards (before scaling)
    settlement_reward: float = 1.0
    city_reward: float = 4.0                   # was 2.0 - doubled to encourage city building
    road_reward: float = 0.1

    # Development card rewards
    dev_card_bought_reward: float = 0.2
    dev_card_played_reward: float = 0.5

    # Dev card effect scaling (Phase 3.1)
    year_of_plenty_reward: float = 0.50        # Fixed reward (always 2 resources)
    monopoly_base_reward: float = 0.10         # Base reward for playing monopoly
    monopoly_per_resource_reward: float = 0.05 # Per resource stolen

    # VP rewards with escalation near victory
    vp_gained_reward: float = 1.0
    vp_high_multiplier: float = 2.0            # NEW: multiplier for VP >= 6
    vp_final_push_multiplier: float = 3.0      # NEW: multiplier for VP >= 8
    longest_road_reward: float = 2.0
    largest_army_reward: float = 2.0

    # Penalty
    resource_discarded_penalty: float = -0.1

    # Setup phase bonus (encourages good initial placements)
    setup_production_bonus_scale: float = 2.0  # was 0.5 - quadrupled for better openings

    # Resource production rewards (Phase 3.1)
    # Rewards when settlements/cities produce resources on dice rolls
    resource_production_reward: float = 0.10   # Per resource produced on roll

    # Player-to-player trading rewards - EXPLOIT REMOVED
    # These were enabling counter-offer spam (212 per game)
    successful_trade_reward: float = 0.0       # was 0.2 - REMOVED (exploit)
    favorable_trade_bonus: float = 0.0         # was 0.1 - REMOVED (exploit)
    trade_rejected_penalty: float = 0.0        # No penalty - rejection isn't inherently bad
    trade_accepted_bonus: float = 0.0          # was 0.05 - REMOVED (THE EXPLOIT)

    # Counter-offer handling (penalty removed - spam was symptom of position memorization)
    counter_offer_penalty: float = 0.0         # No penalty - counter-offers are valid negotiation
    max_counter_offers_per_trade: int = 2      # Hard cap still enforced in rules.py

    # === Strategic Reward Shaping Parameters ===

    # Settlement shaping
    synergy_bonus_scale: float = 1.0           # Scale for resource synergy bonuses
    number_coordination_bonus: float = 0.3     # Bonus per matching number pair
    weighted_production_scale: float = 0.5     # Scale for weighted production value
    setup_complementarity_scale: float = 0.4   # Bonus for complementary setup settlements

    # Road shaping
    road_usefulness_scale: float = 0.2         # Scale for road usefulness score
    dead_end_penalty: float = -0.1             # Penalty for dead-end roads
    blocking_value_scale: float = 0.15         # Scale for blocking opponent expansion

    # Robber shaping
    robber_threat_scale: float = 0.5           # Scale for threat-weighted robber targeting
    sole_source_bonus: float = 0.3             # Bonus for blocking >50% of target's resource
    city_blocking_bonus: float = 0.2           # Bonus per city affected

    # Port rewards
    port_acquisition_scale: float = 0.3        # Scale for port acquisition bonus
    port_path_bonus_scale: float = 0.2         # Scale for setup port path availability

    # Hoarding penalty
    hoarding_penalty_base: float = -0.1        # Base penalty per card over 6

    # === COMPREHENSIVE SETUP REWARD PARAMETERS ===
    # Based on research from competitive Catan strategy guides
    # These replace the simple production bonus for setup phase
    #
    # DESIGN PHILOSOPHY:
    # - Rewards guide early learning but decay over time (curriculum)
    # - Strategic exceptions prevent penalizing valid advanced strategies (OWS, ports)
    # - Strong positive signals for good plays, softer penalties for "bad" plays
    #   (since "bad" might be intentional advanced strategy)

    # Enable comprehensive setup rewards (set False to use old simple production bonus)
    use_comprehensive_setup_rewards: bool = True

    # === CURRICULUM DECAY (disabled by default) ===
    # Setup rewards decay over training to let win/lose signal dominate
    # Enable this after validating harsh penalties work for basic learning
    use_setup_reward_decay: bool = False        # Disabled by default
    setup_reward_decay_start: int = 500_000     # Start decaying after 500K steps
    setup_reward_decay_end: int = 2_000_000     # Fully decayed by 2M steps
    setup_reward_min_scale: float = 0.2         # Keep 20% to maintain basic guidance

    # Resource value weights (normalized, based on competitive analysis)
    # Wheat and Ore are critical for cities/dev cards
    resource_weight_wheat: float = 1.35
    resource_weight_ore: float = 1.33
    resource_weight_brick: float = 0.85  # Boosted slightly for scarcity (3 hexes)
    resource_weight_wood: float = 0.78
    resource_weight_sheep: float = 0.76

    # Weighted pip reward (per weighted pip)
    setup_weighted_pip_reward: float = 0.8

    # Raw pip thresholds (per settlement)
    setup_excellent_pip_threshold: int = 12
    setup_excellent_pip_bonus: float = 3.0
    setup_good_pip_threshold: int = 10
    setup_good_pip_bonus: float = 1.5
    setup_poor_pip_threshold: int = 7
    setup_poor_pip_penalty: float = -3.0

    # Number diversity rewards (unique numbers across both settlements)
    # 5-6 unique numbers is ideal for consistent production
    setup_number_diversity_6: float = 4.0   # Excellent - roll hits you 2/3 of time
    setup_number_diversity_5: float = 2.5   # Good
    setup_number_diversity_4: float = 0.0   # Neutral
    setup_number_diversity_3: float = -3.0  # Bad - too dependent on few rolls
    setup_number_diversity_2: float = -5.0  # Very bad

    # Resource diversity rewards (unique resources across both settlements)
    setup_resource_diversity_5: float = 5.0   # All resources - very flexible
    setup_resource_diversity_4: float = 2.5   # Good - can trade for 5th
    setup_resource_diversity_3: float = 0.0   # Neutral
    setup_resource_diversity_2: float = -4.0  # Bad - too trade dependent
    setup_resource_diversity_1: float = -8.0  # Terrible

    # Critical resource penalties
    setup_missing_wheat_penalty: float = -5.0  # Wheat needed for settlements, cities, dev cards
    setup_missing_ore_penalty: float = -3.0    # Ore needed for cities and dev cards

    # Strategic resource combo bonuses
    setup_wheat_ore_combo_bonus: float = 3.0   # Core combo for cities
    setup_ows_strategy_bonus: float = 2.0      # Ore/Wheat/Sheep enables dev card strategy
    setup_wood_brick_combo_bonus: float = 1.5  # Enables expansion

    # Number coordination (same number on both settlements)
    # Synergistic pairs are good, other doubles add variance
    setup_wheat_ore_same_number_bonus: float = 2.5   # Great synergy
    setup_wood_brick_same_number_bonus: float = 2.0  # Good synergy for roads
    setup_generic_double_penalty: float = -1.0       # Other doubles add variance

    # Position rewards
    setup_inner_ring_bonus: float = 2.0        # 3 hexes, good expansion
    setup_coast_without_port_penalty: float = -2.5  # Only 2 hexes, no port benefit
    setup_coast_with_port_bonus: float = 0.5   # Somewhat compensated

    # Port rewards
    setup_2to1_port_with_2_sources: float = 2.5  # Good if you have production
    setup_2to1_port_with_1_source: float = 0.5
    setup_2to1_port_missing_resource: float = 2.0 # 2:1 port for missing resource (used with strategic exceptions)
    setup_3to1_port_bonus: float = 1.0
    setup_3to1_port_high_production: float = 2.0  # 3:1 with 10+ pips of one resource

    # Expansion potential (reachable good spots within 3 roads)
    setup_expansion_spot_value: float = 0.4    # Per reachable 8+ pip spot
    setup_expansion_to_wheat_bonus: float = 1.5  # Can expand to wheat if missing


class RewardCalculator:
    """Calculates rewards for the Catan RL environment."""

    # Port value factors based on resource traded away
    PORT_VALUE_FACTORS = {
        PortType.LUMBER: 1.2,   # Trading away less valuable
        PortType.WOOL: 1.2,
        PortType.BRICK: 1.0,
        PortType.GRAIN: 0.8,   # Trading away valuable resources
        PortType.ORE: 0.8,
        PortType.GENERIC: 1.0,
    }

    # Resource weight mapping
    RESOURCE_WEIGHTS = {
        Resource.GRAIN: 1.35,   # Wheat - universal currency
        Resource.ORE: 1.33,     # Critical for cities/dev cards
        Resource.BRICK: 0.85,   # Boosted for scarcity (3 hexes)
        Resource.LUMBER: 0.78,  # Early game expansion
        Resource.WOOL: 0.76,    # Lowest but enables dev cards
    }

    # Pip values for each dice number
    PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

    def __init__(self, config: Optional[RewardConfig] = None):
        """Initialize the reward calculator.

        Args:
            config: Reward configuration. Uses defaults if not provided.
        """
        self.config = config or RewardConfig()
        # Update resource weights from config
        self.RESOURCE_WEIGHTS = {
            Resource.GRAIN: self.config.resource_weight_wheat,
            Resource.ORE: self.config.resource_weight_ore,
            Resource.BRICK: self.config.resource_weight_brick,
            Resource.LUMBER: self.config.resource_weight_wood,
            Resource.WOOL: self.config.resource_weight_sheep,
        }

    # === Comprehensive Setup Reward Methods ===

    def calculate_comprehensive_setup_reward(
        self,
        board: Board,
        corner_id: int,
        player_id: int,
        is_second_settlement: bool = False,
        first_settlement_corner: Optional[int] = None
    ) -> float:
        """Calculate comprehensive reward for setup settlement placement.

        Based on competitive Catan strategy research, this evaluates:
        - Weighted pip production (resource-adjusted)
        - Number diversity (unique numbers across settlements)
        - Resource diversity (unique resources)
        - Critical resource access (wheat, ore)
        - Strategic combos (wheat+ore, wood+brick)
        - Number coordination (synergistic doubles)
        - Board position (inner ring vs coast)
        - Port value
        - Expansion potential

        Args:
            board: The game board.
            corner_id: The corner where settlement is placed.
            player_id: The player placing the settlement.
            is_second_settlement: Whether this is the second setup settlement.
            first_settlement_corner: First settlement corner (for complementarity).

        Returns:
            Total comprehensive setup reward.
        """
        reward = 0.0
        cfg = self.config

        # Get corner info
        if corner_id >= len(board.corners):
            return 0.0
        corner = board.corners[corner_id]

        # Collect tile info for this corner
        corner_numbers = []
        corner_resources = []
        raw_pips = 0
        weighted_pips = 0.0

        for tile_id in corner.adjacent_tiles:
            if tile_id < len(board.tiles):
                tile = board.tiles[tile_id]
                if tile.number > 0 and tile.resource != Resource.DESERT:
                    pips = self.PIPS.get(tile.number, 0)
                    raw_pips += pips
                    weighted_pips += pips * self.RESOURCE_WEIGHTS.get(tile.resource, 1.0)
                    corner_numbers.append(tile.number)
                    corner_resources.append(tile.resource)

        # === 1. WEIGHTED PIP REWARD ===
        reward += weighted_pips * cfg.setup_weighted_pip_reward

        # === 2. RAW PIP THRESHOLDS ===
        if raw_pips >= cfg.setup_excellent_pip_threshold:
            reward += cfg.setup_excellent_pip_bonus
        elif raw_pips >= cfg.setup_good_pip_threshold:
            reward += cfg.setup_good_pip_bonus
        elif raw_pips < cfg.setup_poor_pip_threshold:
            reward += cfg.setup_poor_pip_penalty

        # For second settlement, evaluate combined stats
        if is_second_settlement and first_settlement_corner is not None:
            first_corner = board.corners[first_settlement_corner]
            first_numbers = []
            first_resources = []

            for tile_id in first_corner.adjacent_tiles:
                if tile_id < len(board.tiles):
                    tile = board.tiles[tile_id]
                    if tile.number > 0 and tile.resource != Resource.DESERT:
                        first_numbers.append(tile.number)
                        first_resources.append(tile.resource)

            all_numbers = first_numbers + corner_numbers
            all_resources = first_resources + corner_resources

            # === 3. NUMBER DIVERSITY ===
            unique_numbers = len(set(all_numbers))
            if unique_numbers >= 6:
                reward += cfg.setup_number_diversity_6
            elif unique_numbers == 5:
                reward += cfg.setup_number_diversity_5
            elif unique_numbers == 4:
                reward += cfg.setup_number_diversity_4
            elif unique_numbers == 3:
                reward += cfg.setup_number_diversity_3
            else:
                reward += cfg.setup_number_diversity_2

            # === 4. RESOURCE DIVERSITY ===
            unique_resources = len(set(all_resources))
            if unique_resources >= 5:
                reward += cfg.setup_resource_diversity_5
            elif unique_resources == 4:
                reward += cfg.setup_resource_diversity_4
            elif unique_resources == 3:
                reward += cfg.setup_resource_diversity_3
            elif unique_resources == 2:
                reward += cfg.setup_resource_diversity_2
            else:
                reward += cfg.setup_resource_diversity_1

            # === 5. STRATEGIC COMBO BONUSES (check first for OWS detection) ===
            resource_set = set(all_resources)
            has_ows = (Resource.GRAIN in resource_set and
                       Resource.ORE in resource_set and
                       Resource.WOOL in resource_set)

            if Resource.GRAIN in resource_set and Resource.ORE in resource_set:
                reward += cfg.setup_wheat_ore_combo_bonus
                if has_ows:
                    reward += cfg.setup_ows_strategy_bonus  # OWS is top-tier strategy
            if Resource.LUMBER in resource_set and Resource.BRICK in resource_set:
                reward += cfg.setup_wood_brick_combo_bonus

            # === 6. CRITICAL RESOURCE PENALTIES (with strategic exceptions) ===
            # Check for port access that can compensate for missing resources
            has_wheat_port = self._has_port_for_resource(board, corner_id, first_settlement_corner, Resource.GRAIN)
            has_ore_port = self._has_port_for_resource(board, corner_id, first_settlement_corner, Resource.ORE)
            has_3to1_port = self._has_generic_port(board, corner_id, first_settlement_corner)
            can_expand_to_wheat = self._can_expand_to_resource(board, corner_id, Resource.GRAIN)

            if Resource.GRAIN not in resource_set:
                # Waive or reduce penalty if compensated
                if has_wheat_port:
                    reward += cfg.setup_2to1_port_missing_resource  # Bonus instead of penalty
                elif has_3to1_port or can_expand_to_wheat:
                    reward += cfg.setup_missing_wheat_penalty * 0.3  # Reduced penalty
                else:
                    reward += cfg.setup_missing_wheat_penalty

            if Resource.ORE not in resource_set:
                if has_ore_port:
                    reward += cfg.setup_2to1_port_missing_resource * 0.8  # Slightly less valuable
                elif has_3to1_port:
                    reward += cfg.setup_missing_ore_penalty * 0.5  # Reduced penalty
                else:
                    reward += cfg.setup_missing_ore_penalty

            # === 7. NUMBER COORDINATION (Synergistic Doubles) ===
            # Find numbers that appear on both settlements
            first_number_set = set(first_numbers)
            corner_number_set = set(corner_numbers)
            shared_numbers = first_number_set & corner_number_set

            for num in shared_numbers:
                # Get resources on this number from each settlement
                first_res_on_num = [r for n, r in zip(first_numbers, first_resources) if n == num]
                corner_res_on_num = [r for n, r in zip(corner_numbers, corner_resources) if n == num]
                all_res_on_num = set(first_res_on_num + corner_res_on_num)

                # Check for synergistic pairs
                if Resource.GRAIN in all_res_on_num and Resource.ORE in all_res_on_num:
                    reward += cfg.setup_wheat_ore_same_number_bonus
                elif Resource.LUMBER in all_res_on_num and Resource.BRICK in all_res_on_num:
                    reward += cfg.setup_wood_brick_same_number_bonus
                else:
                    # Generic double - adds variance without synergy
                    reward += cfg.setup_generic_double_penalty

        # === 8. BOARD POSITION ===
        num_adjacent_tiles = len([t for t in corner.adjacent_tiles if t < len(board.tiles)])
        has_port = corner.port is not None

        if num_adjacent_tiles == 3:
            # Inner position with 3 hexes
            reward += cfg.setup_inner_ring_bonus
        elif num_adjacent_tiles <= 2:
            # Coastal position
            if has_port:
                reward += cfg.setup_coast_with_port_bonus
            else:
                reward += cfg.setup_coast_without_port_penalty

        # === 9. PORT VALUE ===
        if has_port:
            port_type = corner.port  # corner.port is a PortType enum directly
            port_resource = self._get_port_resource(port_type)

            if port_type == PortType.GENERIC:
                # 3:1 port - check for high production of any resource
                max_resource_pips = self._get_max_resource_pips(
                    board, corner_id, first_settlement_corner
                )
                if max_resource_pips >= 10:
                    reward += cfg.setup_3to1_port_high_production  # Strong combo
                else:
                    reward += cfg.setup_3to1_port_bonus
            else:
                # 2:1 port - value depends on sources and whether it fills a gap
                sources = self._count_resource_sources(
                    board, corner_id, player_id, port_resource, first_settlement_corner
                )
                if sources >= 2:
                    reward += cfg.setup_2to1_port_with_2_sources
                elif sources == 1:
                    reward += cfg.setup_2to1_port_with_1_source
                # Note: 2:1 for missing resource bonus handled in critical resource section

        # === 10. EXPANSION POTENTIAL ===
        expansion_spots = self._count_good_expansion_spots(board, corner_id)
        reward += expansion_spots * cfg.setup_expansion_spot_value

        return reward

    def _get_port_resource(self, port_type: PortType) -> Optional[Resource]:
        """Get the resource type for a 2:1 port."""
        port_to_resource = {
            PortType.LUMBER: Resource.LUMBER,
            PortType.BRICK: Resource.BRICK,
            PortType.WOOL: Resource.WOOL,
            PortType.GRAIN: Resource.GRAIN,
            PortType.ORE: Resource.ORE,
        }
        return port_to_resource.get(port_type)

    def _count_resource_sources(
        self,
        board: Board,
        corner_id: int,
        player_id: int,
        resource: Optional[Resource],
        first_settlement_corner: Optional[int]
    ) -> int:
        """Count how many hexes produce a given resource for player."""
        if resource is None:
            return 0

        sources = 0
        corners_to_check = [corner_id]
        if first_settlement_corner is not None:
            corners_to_check.append(first_settlement_corner)

        for cid in corners_to_check:
            if cid >= len(board.corners):
                continue
            corner = board.corners[cid]
            for tile_id in corner.adjacent_tiles:
                if tile_id < len(board.tiles):
                    tile = board.tiles[tile_id]
                    if tile.resource == resource and tile.number > 0:
                        sources += 1

        return sources

    def _count_good_expansion_spots(self, board: Board, corner_id: int) -> int:
        """Count expansion spots with 8+ pips reachable within 3 roads."""
        # Simplified: just count adjacent valid corners with good production
        good_spots = 0
        corner = board.corners[corner_id]

        # Check corners reachable via adjacent edges
        for edge_id in corner.adjacent_edges:
            if edge_id >= len(board.edges):
                continue
            edge = board.edges[edge_id]
            for adj_corner_id in edge.corners:
                if adj_corner_id == corner_id or adj_corner_id >= len(board.corners):
                    continue
                adj_corner = board.corners[adj_corner_id]
                # Check if this would be a valid settlement spot
                if adj_corner.building is None and adj_corner.owner is None:
                    # Calculate pips
                    pips = 0
                    for tile_id in adj_corner.adjacent_tiles:
                        if tile_id < len(board.tiles):
                            tile = board.tiles[tile_id]
                            pips += self.PIPS.get(tile.number, 0)
                    if pips >= 8:
                        good_spots += 1

        return good_spots

    def _has_port_for_resource(
        self,
        board: Board,
        corner_id: int,
        first_settlement_corner: Optional[int],
        resource: Resource
    ) -> bool:
        """Check if player has a 2:1 port for the specified resource."""
        resource_to_port = {
            Resource.LUMBER: PortType.LUMBER,
            Resource.BRICK: PortType.BRICK,
            Resource.WOOL: PortType.WOOL,
            Resource.GRAIN: PortType.GRAIN,
            Resource.ORE: PortType.ORE,
        }
        target_port = resource_to_port.get(resource)
        if target_port is None:
            return False

        corners_to_check = [corner_id]
        if first_settlement_corner is not None:
            corners_to_check.append(first_settlement_corner)

        for cid in corners_to_check:
            if cid >= len(board.corners):
                continue
            corner = board.corners[cid]
            if corner.port == target_port:
                return True
        return False

    def _has_generic_port(
        self,
        board: Board,
        corner_id: int,
        first_settlement_corner: Optional[int]
    ) -> bool:
        """Check if player has a 3:1 generic port."""
        corners_to_check = [corner_id]
        if first_settlement_corner is not None:
            corners_to_check.append(first_settlement_corner)

        for cid in corners_to_check:
            if cid >= len(board.corners):
                continue
            corner = board.corners[cid]
            if corner.port == PortType.GENERIC:
                return True
        return False

    def _can_expand_to_resource(
        self,
        board: Board,
        corner_id: int,
        resource: Resource
    ) -> bool:
        """Check if there's a good expansion path to a resource within 2 roads."""
        corner = board.corners[corner_id]

        # Check adjacent corners (1 road away)
        for edge_id in corner.adjacent_edges:
            if edge_id >= len(board.edges):
                continue
            edge = board.edges[edge_id]
            for adj_corner_id in edge.corners:
                if adj_corner_id == corner_id or adj_corner_id >= len(board.corners):
                    continue
                adj_corner = board.corners[adj_corner_id]
                # Check if buildable and has the resource
                if adj_corner.building is None and adj_corner.owner is None:
                    for tile_id in adj_corner.adjacent_tiles:
                        if tile_id < len(board.tiles):
                            tile = board.tiles[tile_id]
                            if tile.resource == resource and tile.number > 0:
                                # Check if decent production (5+ pips on that resource)
                                pips = self.PIPS.get(tile.number, 0)
                                if pips >= 3:  # At least a 5, 9 or better
                                    return True
        return False

    def _get_max_resource_pips(
        self,
        board: Board,
        corner_id: int,
        first_settlement_corner: Optional[int]
    ) -> int:
        """Get the maximum pips for any single resource across settlements."""
        resource_pips: Dict[Resource, int] = {}

        corners_to_check = [corner_id]
        if first_settlement_corner is not None:
            corners_to_check.append(first_settlement_corner)

        for cid in corners_to_check:
            if cid >= len(board.corners):
                continue
            corner = board.corners[cid]
            for tile_id in corner.adjacent_tiles:
                if tile_id < len(board.tiles):
                    tile = board.tiles[tile_id]
                    if tile.resource != Resource.DESERT and tile.number > 0:
                        pips = self.PIPS.get(tile.number, 0)
                        resource_pips[tile.resource] = resource_pips.get(tile.resource, 0) + pips

        return max(resource_pips.values()) if resource_pips else 0

    def get_setup_reward_scale(self, timestep: int) -> float:
        """Get the curriculum decay scale for setup rewards.

        Returns a scale factor that starts at 1.0 and decays to min_scale
        between decay_start and decay_end timesteps.
        """
        cfg = self.config
        if timestep < cfg.setup_reward_decay_start:
            return 1.0
        elif timestep >= cfg.setup_reward_decay_end:
            return cfg.setup_reward_min_scale
        else:
            # Linear decay
            progress = (timestep - cfg.setup_reward_decay_start) / (
                cfg.setup_reward_decay_end - cfg.setup_reward_decay_start
            )
            return 1.0 - progress * (1.0 - cfg.setup_reward_min_scale)

    # === Strategic Reward Shaping Methods ===

    def calculate_settlement_shaping_reward(
        self,
        board: Board,
        corner_id: int,
        player_id: int,
        is_setup: bool = False,
        first_settlement_corner: Optional[int] = None
    ) -> float:
        """Calculate strategic reward for settlement placement.

        Args:
            board: The game board.
            corner_id: The corner where settlement is placed.
            player_id: The player placing the settlement.
            is_setup: Whether this is setup phase.
            first_settlement_corner: First settlement corner (for complementarity).

        Returns:
            Total shaping reward for the settlement placement.
        """
        reward = 0.0

        # Synergy bonus
        synergy = board.get_corner_synergy_score(corner_id)
        reward += synergy * self.config.synergy_bonus_scale

        # Number coordination (with existing buildings)
        # Exclude corner_id to avoid self-matching when using current_state.board
        coordination = board.get_corner_number_coordination(
            corner_id, player_id, exclude_corner_id=corner_id
        )
        reward += coordination

        # Weighted production value
        weighted_prod = board.get_weighted_production_value(corner_id)
        reward += weighted_prod * self.config.weighted_production_scale

        # Setup complementarity bonus
        if is_setup and first_settlement_corner is not None:
            complementarity = self._calculate_setup_complementarity(
                board, corner_id, first_settlement_corner
            )
            reward += complementarity * self.config.setup_complementarity_scale

        return reward

    def _calculate_setup_complementarity(
        self,
        board: Board,
        corner_id: int,
        first_corner_id: int
    ) -> float:
        """Calculate complementarity bonus for second setup settlement.

        Rewards unique resources and unique numbers not covered by first settlement.

        Args:
            board: The game board.
            corner_id: Second settlement corner.
            first_corner_id: First settlement corner.

        Returns:
            Complementarity bonus.
        """
        # Get resources and numbers from first settlement
        first_resources = set()
        first_numbers = set()
        first_corner = board.corners[first_corner_id]
        for tile_id in first_corner.adjacent_tiles:
            if tile_id < len(board.tiles):
                tile = board.tiles[tile_id]
                if tile.resource != Resource.DESERT:
                    first_resources.add(tile.resource)
                    if tile.number > 0:
                        first_numbers.add(tile.number)

        # Get resources and numbers from second settlement
        second_resources = set()
        second_numbers = set()
        second_corner = board.corners[corner_id]
        for tile_id in second_corner.adjacent_tiles:
            if tile_id < len(board.tiles):
                tile = board.tiles[tile_id]
                if tile.resource != Resource.DESERT:
                    second_resources.add(tile.resource)
                    if tile.number > 0:
                        second_numbers.add(tile.number)

        # Bonus for unique resources
        unique_resources = second_resources - first_resources
        unique_numbers = second_numbers - first_numbers

        bonus = len(unique_resources) * 0.2 + len(unique_numbers) * 0.1
        return bonus

    def calculate_road_shaping_reward(
        self,
        board: Board,
        edge_id: int,
        player_id: int
    ) -> float:
        """Calculate strategic reward for road placement.

        Args:
            board: The game board.
            edge_id: The edge where road is placed.
            player_id: The player placing the road.

        Returns:
            Total shaping reward for the road placement.
        """
        usefulness, is_dead_end, continuation = board.get_road_usefulness(edge_id, player_id)
        blocking = board.get_road_blocking_value(edge_id, player_id)

        reward = 0.0

        # Usefulness reward
        reward += usefulness * self.config.road_usefulness_scale

        # Dead end penalty
        if is_dead_end:
            reward += self.config.dead_end_penalty

        # Blocking value
        reward += blocking * self.config.blocking_value_scale

        return reward

    def calculate_robber_shaping_reward(
        self,
        state: GameState,
        tile_id: int,
        player_id: int,
    ) -> float:
        """Calculate strategic reward for robber placement.

        Uses VP-based threat heuristic to weight production impact.
        The policy's strategic context (competition features) provides
        richer threat signals to the actor/critic for decision-making.

        Args:
            state: Current game state.
            tile_id: The tile where robber is placed.
            player_id: The player moving the robber.

        Returns:
            Total shaping reward for the robber placement.
        """
        board = state.board
        production_impact = board.get_tile_production_impact(tile_id)

        reward = 0.0

        # VP-based threat heuristic
        threat_levels = {}
        for p in state.players:
            if p.id != player_id:
                # Threat based on VP progress and dev card potential
                threat = p.total_victory_points / 10.0
                threat += p.total_dev_cards * 0.1
                threat_levels[p.id] = min(threat, 1.0)

        # Reward for production impact weighted by threat
        for opponent_id, impact in production_impact.items():
            if opponent_id != player_id and impact > 0:
                threat = threat_levels.get(opponent_id, 0.5)
                reward += impact * threat * self.config.robber_threat_scale

        # Sole-source blocking bonus
        tile = board.tiles[tile_id]
        for opponent_id in range(NUM_PLAYERS):
            if opponent_id != player_id:
                opponent = state.players[opponent_id]
                if self._is_sole_source_block(board, tile_id, opponent_id):
                    reward += self.config.sole_source_bonus

        # City blocking bonus
        for corner in board.corners:
            if tile_id in corner.adjacent_tiles:
                if corner.building == 'city' and corner.owner != player_id:
                    reward += self.config.city_blocking_bonus

        return reward

    def _is_sole_source_block(
        self,
        board: Board,
        tile_id: int,
        player_id: int
    ) -> bool:
        """Check if blocking this tile removes >50% of player's production of a resource.

        Args:
            board: The game board.
            tile_id: The tile being blocked.
            player_id: The player to check.

        Returns:
            True if this is a sole-source block.
        """
        tile = board.tiles[tile_id]
        resource = tile.resource

        if resource == Resource.DESERT:
            return False

        # Calculate total production of this resource for the player
        total_production = 0.0
        blocked_production = 0.0

        for corner in board.corners:
            if corner.owner == player_id:
                multiplier = 2 if corner.building == 'city' else 1
                for adj_tile_id in corner.adjacent_tiles:
                    if adj_tile_id < len(board.tiles):
                        adj_tile = board.tiles[adj_tile_id]
                        if adj_tile.resource == resource:
                            prod = adj_tile.production_probability * multiplier
                            total_production += prod
                            if adj_tile_id == tile_id:
                                blocked_production += prod

        if total_production == 0:
            return False

        return blocked_production / total_production > 0.5

    def calculate_port_acquisition_reward(
        self,
        board: Board,
        corner_id: int,
        player_id: int
    ) -> float:
        """Calculate reward for acquiring a port.

        Args:
            board: The game board.
            corner_id: The corner with the port.
            player_id: The player acquiring the port.

        Returns:
            Port acquisition reward.
        """
        corner = board.corners[corner_id]
        if corner.port is None:
            return 0.0

        port_type = corner.port

        # Base value: 2:1 ports worth more than 3:1
        if port_type == PortType.GENERIC:
            base_value = 1.0
        else:
            base_value = 1.5

        # Resource factor - trading away less valuable resources is better
        resource_factor = self.PORT_VALUE_FACTORS.get(port_type, 1.0)

        return base_value * resource_factor * self.config.port_acquisition_scale

    def calculate_hoarding_penalty(self, player: Player) -> float:
        """Calculate penalty for holding too many resource cards.

        Args:
            player: The player to check.

        Returns:
            Penalty for resource hoarding.
        """
        total_cards = player.total_resources
        if total_cards <= 6:
            return 0.0

        # Penalty increases with cards over 6
        over_limit = total_cards - 6
        return self.config.hoarding_penalty_base * over_limit

    def calculate_reward(
        self,
        prev_state: GameState,
        current_state: GameState,
        action_type: ActionType,
        player_id: int
    ) -> float:
        """Calculate the reward for a state transition.

        Args:
            prev_state: Game state before action.
            current_state: Game state after action.
            action_type: The type of action taken.
            player_id: The player receiving the reward.

        Returns:
            The reward value.
        """
        reward = 0.0

        # Check for game end
        if current_state.is_game_over:
            if current_state.winner == player_id:
                return self.config.win_reward
            else:
                return self.config.lose_reward

        # Dense rewards (scaled)
        dense_reward = self._calculate_dense_reward(
            prev_state, current_state, action_type, player_id
        )
        reward += dense_reward * self.config.dense_scale

        return reward

    def _calculate_dense_reward(
        self,
        prev_state: GameState,
        current_state: GameState,
        action_type: ActionType,
        player_id: int,
        action_context: Optional[Dict] = None
    ) -> float:
        """Calculate dense (auxiliary) rewards including strategic shaping.

        Args:
            prev_state: State before action.
            current_state: State after action.
            action_type: Type of action taken.
            player_id: Player receiving reward.
            action_context: Optional dict with corner_id, edge_id, tile_id.
        """
        reward = 0.0
        prev_player = prev_state.players[player_id]
        curr_player = current_state.players[player_id]
        action_context = action_context or {}

        # Building rewards with strategic shaping
        if action_type == ActionType.BUILD_SETTLEMENT or action_type == ActionType.PLACE_SETTLEMENT:
            new_settlements = len(curr_player.settlements) - len(prev_player.settlements)
            reward += new_settlements * self.config.settlement_reward

            # Find the newly placed settlement corner
            new_settlement_ids = set(curr_player.settlements) - set(prev_player.settlements)
            for corner_id in new_settlement_ids:
                first_settlement = prev_player.settlements[0] if prev_player.settlements else None
                is_second_settlement = first_settlement is not None

                if action_type == ActionType.PLACE_SETTLEMENT and prev_state.is_setup_phase:
                    # Setup phase rewards
                    if self.config.use_comprehensive_setup_rewards:
                        # Use comprehensive setup rewards based on competitive strategy research
                        setup_reward = self.calculate_comprehensive_setup_reward(
                            prev_state.board,  # Use prev_state to avoid self-matching
                            corner_id,
                            player_id,
                            is_second_settlement=is_second_settlement,
                            first_settlement_corner=first_settlement
                        )
                        reward += setup_reward
                    else:
                        # Legacy: simple production bonus
                        production_value = current_state.board.get_corner_production_value(corner_id)
                        reward += production_value * self.config.setup_production_bonus_scale

                        # Legacy strategic shaping
                        shaping_reward = self.calculate_settlement_shaping_reward(
                            prev_state.board,
                            corner_id,
                            player_id,
                            is_setup=True,
                            first_settlement_corner=first_settlement
                        )
                        reward += shaping_reward

                        # Port acquisition bonus
                        port_reward = self.calculate_port_acquisition_reward(
                            current_state.board, corner_id, player_id
                        )
                        reward += port_reward
                else:
                    # Non-setup settlement placement
                    # Strategic shaping for settlement placement
                    # NOTE: Use prev_state.board to avoid counting new settlement against itself
                    shaping_reward = self.calculate_settlement_shaping_reward(
                        prev_state.board,
                        corner_id,
                        player_id,
                        is_setup=False,
                        first_settlement_corner=first_settlement
                    )
                    reward += shaping_reward

                    # Port acquisition bonus
                    port_reward = self.calculate_port_acquisition_reward(
                        current_state.board, corner_id, player_id
                    )
                    reward += port_reward

        if action_type == ActionType.BUILD_CITY:
            new_cities = len(curr_player.cities) - len(prev_player.cities)
            reward += new_cities * self.config.city_reward

        if action_type == ActionType.BUILD_ROAD or action_type == ActionType.PLACE_ROAD:
            new_roads = len(curr_player.roads) - len(prev_player.roads)
            reward += new_roads * self.config.road_reward

            # Strategic shaping for road placement
            new_road_ids = set(curr_player.roads) - set(prev_player.roads)
            for edge_id in new_road_ids:
                road_shaping = self.calculate_road_shaping_reward(
                    current_state.board, edge_id, player_id
                )
                reward += road_shaping

        # Robber placement shaping
        if action_type == ActionType.MOVE_ROBBER:
            # Find which tile the robber moved to
            robber_tile = current_state.board.get_robber_tile()
            robber_shaping = self.calculate_robber_shaping_reward(
                current_state, robber_tile, player_id
            )
            reward += robber_shaping

        # Resource production rewards (Phase 3.1)
        # Reward when dice roll produces resources for this player
        if action_type == ActionType.ROLL_DICE:
            resources_gained = curr_player.total_resources - prev_player.total_resources
            if resources_gained > 0:
                reward += resources_gained * self.config.resource_production_reward

        # Development card rewards
        if action_type == ActionType.BUY_DEV_CARD:
            reward += self.config.dev_card_bought_reward

        # Dev card play rewards with scaling (Phase 3.1)
        if action_type == ActionType.PLAY_KNIGHT:
            reward += self.config.dev_card_played_reward

        if action_type == ActionType.PLAY_ROAD_BUILDING:
            reward += self.config.dev_card_played_reward

        if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            # Fixed reward - always gives 2 resources
            reward += self.config.year_of_plenty_reward

        if action_type == ActionType.PLAY_MONOPOLY:
            # Scaled reward based on resources stolen
            resources_gained = curr_player.total_resources - prev_player.total_resources
            reward += self.config.monopoly_base_reward
            if resources_gained > 0:
                reward += resources_gained * self.config.monopoly_per_resource_reward

        # VP change reward with escalation near victory
        vp_change = curr_player.victory_points - prev_player.victory_points
        if vp_change > 0:
            curr_vp = curr_player.total_victory_points
            # Escalating multiplier for VP near victory (encourages final push)
            if curr_vp >= 8:
                multiplier = self.config.vp_final_push_multiplier  # 3x for final push
            elif curr_vp >= 6:
                multiplier = self.config.vp_high_multiplier  # 2x
            else:
                multiplier = 1.0
            reward += vp_change * self.config.vp_gained_reward * multiplier

        # Longest road / largest army
        if not prev_player.has_longest_road and curr_player.has_longest_road:
            reward += self.config.longest_road_reward

        if not prev_player.has_largest_army and curr_player.has_largest_army:
            reward += self.config.largest_army_reward

        # Resource discard penalty
        if action_type == ActionType.DISCARD:
            resources_lost = prev_player.total_resources - curr_player.total_resources
            reward += resources_lost * self.config.resource_discarded_penalty

        # Hoarding penalty (only on END_TURN to avoid penalizing every action)
        if action_type == ActionType.END_TURN:
            hoarding_penalty = self.calculate_hoarding_penalty(curr_player)
            reward += hoarding_penalty

        # Counter-offers: no penalty (spam was symptom of position memorization, now fixed)
        # The hard cap in rules.py still limits counter-offer depth

        # Player-to-player trade rewards (REMOVED - were enabling exploitation)
        # Note: successful_trade_reward, favorable_trade_bonus, trade_accepted_bonus
        # are now all 0.0 by default to prevent reward hacking via trade spam

        return reward

    def calculate_reward_fast(
        self,
        prev_vp: int,
        prev_resources: int,
        prev_settlements: int,
        prev_cities: int,
        prev_roads: int,
        prev_longest_road: bool,
        prev_largest_army: bool,
        current_state: GameState,
        action_type: ActionType,
        agent_player_id: int,
        prev_settlement_ids: Optional[List[int]] = None,
        prev_road_ids: Optional[List[int]] = None,
    ) -> float:
        """Calculate reward using lightweight snapshot instead of full state copy.

        OPTIMIZATION: Avoids expensive state.copy() by accepting individual field values.

        Robber rewards use VP-based threat heuristic. The policy's strategic context
        (competition features from victory_path_head) provides richer threat signals
        to the actor/critic for decision-making.

        Args:
            prev_vp: Previous total victory points.
            prev_resources: Previous total resources.
            prev_settlements: Previous settlement count.
            prev_cities: Previous city count.
            prev_roads: Previous road count.
            prev_longest_road: Whether player had longest road.
            prev_largest_army: Whether player had largest army.
            current_state: Current game state after action.
            action_type: The type of action taken.
            agent_player_id: The agent's player ID.
            prev_settlement_ids: List of previous settlement corner IDs.
            prev_road_ids: List of previous road edge IDs.

        Returns:
            The reward value.
        """
        # Check for game end
        if current_state.is_game_over:
            if current_state.winner == agent_player_id:
                return self.config.win_reward
            else:
                return self.config.lose_reward

        curr_player = current_state.players[agent_player_id]
        reward = 0.0

        # Building rewards with strategic shaping
        if action_type == ActionType.BUILD_SETTLEMENT or action_type == ActionType.PLACE_SETTLEMENT:
            new_settlements = len(curr_player.settlements) - prev_settlements
            reward += new_settlements * self.config.settlement_reward

            # Find the newest settlement
            if curr_player.settlements and new_settlements > 0:
                # Get the new settlement(s)
                prev_set = set(prev_settlement_ids) if prev_settlement_ids else set()
                new_corner_ids = set(curr_player.settlements) - prev_set

                for corner_id in new_corner_ids:
                    first_settlement = prev_settlement_ids[0] if prev_settlement_ids else None
                    is_second_settlement = first_settlement is not None

                    # Setup phase rewards
                    if action_type == ActionType.PLACE_SETTLEMENT:
                        if self.config.use_comprehensive_setup_rewards:
                            # Use comprehensive setup rewards
                            # Note: Using current_state.board here (fast path doesn't have prev_state)
                            setup_reward = self.calculate_comprehensive_setup_reward(
                                current_state.board,
                                corner_id,
                                agent_player_id,
                                is_second_settlement=is_second_settlement,
                                first_settlement_corner=first_settlement
                            )
                            reward += setup_reward
                        else:
                            # Legacy: simple production bonus
                            production_value = current_state.board.get_corner_production_value(corner_id)
                            reward += production_value * self.config.setup_production_bonus_scale

                            # Legacy strategic shaping
                            shaping_reward = self.calculate_settlement_shaping_reward(
                                current_state.board,
                                corner_id,
                                agent_player_id,
                                is_setup=True,
                                first_settlement_corner=first_settlement
                            )
                            reward += shaping_reward

                            # Port acquisition
                            port_reward = self.calculate_port_acquisition_reward(
                                current_state.board, corner_id, agent_player_id
                            )
                            reward += port_reward
                    else:
                        # Non-setup settlement
                        is_setup = current_state.is_setup_phase
                        shaping_reward = self.calculate_settlement_shaping_reward(
                            current_state.board,
                            corner_id,
                            agent_player_id,
                            is_setup=is_setup,
                            first_settlement_corner=first_settlement
                        )
                        reward += shaping_reward

                        # Port acquisition
                        port_reward = self.calculate_port_acquisition_reward(
                            current_state.board, corner_id, agent_player_id
                        )
                        reward += port_reward

        if action_type == ActionType.BUILD_CITY:
            new_cities = len(curr_player.cities) - prev_cities
            reward += new_cities * self.config.city_reward

        if action_type == ActionType.BUILD_ROAD or action_type == ActionType.PLACE_ROAD:
            new_roads = len(curr_player.roads) - prev_roads
            reward += new_roads * self.config.road_reward

            # Strategic shaping for roads
            if curr_player.roads and new_roads > 0:
                prev_set = set(prev_road_ids) if prev_road_ids else set()
                new_edge_ids = set(curr_player.roads) - prev_set

                for edge_id in new_edge_ids:
                    road_shaping = self.calculate_road_shaping_reward(
                        current_state.board, edge_id, agent_player_id
                    )
                    reward += road_shaping

        # Robber placement shaping (PHASE 3: now uses learned threat levels if available)
        if action_type == ActionType.MOVE_ROBBER:
            robber_tile = current_state.board.get_robber_tile()
            robber_shaping = self.calculate_robber_shaping_reward(
                current_state, robber_tile, agent_player_id
            )
            reward += robber_shaping

        # Resource production rewards (Phase 3.1)
        # Reward when dice roll produces resources for this player
        if action_type == ActionType.ROLL_DICE:
            resources_gained = curr_player.total_resources - prev_resources
            if resources_gained > 0:
                reward += resources_gained * self.config.resource_production_reward

        # Development card rewards
        if action_type == ActionType.BUY_DEV_CARD:
            reward += self.config.dev_card_bought_reward

        # Dev card play rewards with scaling (Phase 3.1)
        if action_type == ActionType.PLAY_KNIGHT:
            reward += self.config.dev_card_played_reward

        if action_type == ActionType.PLAY_ROAD_BUILDING:
            reward += self.config.dev_card_played_reward

        if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            # Fixed reward - always gives 2 resources
            reward += self.config.year_of_plenty_reward

        if action_type == ActionType.PLAY_MONOPOLY:
            # Scaled reward based on resources stolen
            resources_gained = curr_player.total_resources - prev_resources
            reward += self.config.monopoly_base_reward
            if resources_gained > 0:
                reward += resources_gained * self.config.monopoly_per_resource_reward

        # VP change reward with escalation near victory
        # Use total_victory_points to include hidden VP dev cards
        vp_change = curr_player.total_victory_points - prev_vp
        if vp_change > 0:
            curr_vp = curr_player.total_victory_points
            # Escalating multiplier for VP near victory (encourages final push)
            if curr_vp >= 8:
                multiplier = self.config.vp_final_push_multiplier  # 3x for final push
            elif curr_vp >= 6:
                multiplier = self.config.vp_high_multiplier  # 2x
            else:
                multiplier = 1.0
            reward += vp_change * self.config.vp_gained_reward * multiplier

        # Longest road / largest army
        if not prev_longest_road and curr_player.has_longest_road:
            reward += self.config.longest_road_reward

        if not prev_largest_army and curr_player.has_largest_army:
            reward += self.config.largest_army_reward

        # Resource discard penalty
        if action_type == ActionType.DISCARD:
            resources_lost = prev_resources - curr_player.total_resources
            reward += resources_lost * self.config.resource_discarded_penalty

        # Hoarding penalty (only on END_TURN to avoid penalizing every action)
        if action_type == ActionType.END_TURN:
            hoarding_penalty = self.calculate_hoarding_penalty(curr_player)
            reward += hoarding_penalty

        # Player-to-player trade rewards (REMOVED - were enabling exploitation)
        # Note: successful_trade_reward, favorable_trade_bonus, trade_accepted_bonus
        # are now all 0.0 by default to prevent reward hacking via trade spam

        # Apply dense scale to normal rewards
        scaled_reward = reward * self.config.dense_scale

        return scaled_reward

    def calculate_terminal_reward(
        self,
        state: GameState,
        player_id: int
    ) -> float:
        """Calculate terminal reward at end of game.

        PHASE 3: Includes VP-based partial credit for losses.
        This encourages the agent to push for more VP even when losing,
        rather than stalling at 4-5 VP.

        Args:
            state: Final game state.
            player_id: The player receiving the reward.

        Returns:
            Terminal reward.
        """
        if state.winner == player_id:
            return self.config.win_reward

        # VP-based partial credit for losses (encourages pushing for VP)
        final_vp = state.players[player_id].total_victory_points
        if final_vp >= 8:
            # Close game - only half penalty (was competitive)
            return self.config.lose_reward * 0.5
        elif final_vp >= 6:
            # Decent attempt - 75% penalty
            return self.config.lose_reward * 0.75
        else:
            # Poor performance - full penalty
            return self.config.lose_reward

    def get_info_dict(
        self,
        prev_state: Optional[GameState],
        current_state: GameState,
        action_type: ActionType,
        player_id: int
    ) -> dict:
        """Get detailed reward breakdown for logging.

        Args:
            prev_state: Game state before action.
            current_state: Game state after action.
            action_type: The type of action taken.
            player_id: The player.

        Returns:
            Dictionary with reward components.
        """
        info = {
            "total_reward": 0.0,
            "is_winner": False,
            "vp_current": current_state.players[player_id].victory_points,
            "vp_total": current_state.players[player_id].total_victory_points,
        }

        if current_state.is_game_over:
            info["is_winner"] = current_state.winner == player_id
            info["total_reward"] = self.calculate_terminal_reward(current_state, player_id)
        elif prev_state is not None:
            info["total_reward"] = self.calculate_reward(
                prev_state, current_state, action_type, player_id
            )
            info["dense_reward"] = self._calculate_dense_reward(
                prev_state, current_state, action_type, player_id
            )

        return info
