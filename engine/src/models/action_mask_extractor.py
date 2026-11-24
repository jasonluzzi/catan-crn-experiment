"""
Extract action masks from observations for behavioral cloning.

This module derives full action masks from the encoded observation,
enabling proper action masking without needing the full game state.

The observation already encodes:
- Player resources (for self)
- Corner validity for settlements
- Edge validity for roads
- Game phase

We use this to derive which actions are actually legal.
"""

import torch
import numpy as np
from typing import Dict, Set, Tuple, Optional

from .action_decomposer import (
    ActionDecomposer,
    HierarchicalActionType,
    LocationHead,
    TYPE_TO_LOCATION_HEAD,
    NUM_ACTION_TYPES,
)


# =============================================================================
# Observation Layout Constants
# =============================================================================

# Row indices in observation (shape: 176 x 65)
CONTEXT_ROW = 0
TILE_START = 1
TILE_END = 20
CORNER_START = 20
CORNER_END = 74
EDGE_START = 74
EDGE_END = 146
PLAYER_START = 146
PLAYER_END = 150
BANK_START = 150
BANK_END = 155

# Context row indices (row 0)
TURN_IDX = 0
PHASE_START_IDX = 1
NUM_GAME_PHASES = 12

# Player row indices (relative to each player row)
# Row 146 = current player (from their perspective)
PLAYER_RELATIVE_POS_IDX = 0
PLAYER_IS_SELF_IDX = 1
PLAYER_VP_IDX = 2
PLAYER_TOTAL_VP_IDX = 3
# Resources (only for self, indices 4-8)
PLAYER_LUMBER_IDX = 4
PLAYER_BRICK_IDX = 5
PLAYER_WOOL_IDX = 6
PLAYER_GRAIN_IDX = 7
PLAYER_ORE_IDX = 8
PLAYER_TOTAL_RESOURCES_IDX = 9
# Dev cards (only for self, indices 10-14)
PLAYER_KNIGHT_IDX = 10
PLAYER_ROAD_BUILDING_IDX = 11
PLAYER_YEAR_OF_PLENTY_IDX = 12
PLAYER_MONOPOLY_IDX = 13
PLAYER_VP_CARD_IDX = 14

# Corner feature indices (relative to each corner row)
CORNER_EMPTY_IDX = 0
CORNER_SETTLEMENT_IDX = 1
CORNER_CITY_IDX = 2
CORNER_NO_OWNER_IDX = 3
CORNER_OWNER_START_IDX = 4  # Owner 0-3 relative at indices 4-7
CORNER_IS_VALID_IDX = 16  # is_valid_settlement_location

# Edge feature indices (relative to each edge row)
EDGE_HAS_ROAD_IDX = 0
EDGE_NO_OWNER_IDX = 1
EDGE_OWNER_START_IDX = 2  # Owner 0-3 relative at indices 2-5
EDGE_IS_VALID_IDX = 6  # is_valid_road_location

# Building costs (in terms of resources)
# Resource order: LUMBER, BRICK, WOOL, GRAIN, ORE
ROAD_COST = [1, 1, 0, 0, 0]      # 1 lumber, 1 brick
SETTLEMENT_COST = [1, 1, 1, 1, 0]  # 1 lumber, 1 brick, 1 wool, 1 grain
CITY_COST = [0, 0, 0, 2, 3]      # 2 grain, 3 ore
DEV_CARD_COST = [0, 0, 1, 1, 1]  # 1 wool, 1 grain, 1 ore

# Counts
NUM_CORNERS = 54
NUM_EDGES = 72
NUM_TILES = 19
NUM_PLAYERS = 4
NUM_TRADE_TEMPLATES = 50


# =============================================================================
# Phase -> Valid Action Type Mapping
# =============================================================================

PHASE_TO_VALID_TYPES: Dict[int, Set[HierarchicalActionType]] = {
    0: {HierarchicalActionType.SETUP_SETTLEMENT},  # SETUP_SETTLEMENT_1
    1: {HierarchicalActionType.SETUP_ROAD},        # SETUP_ROAD_1
    2: {HierarchicalActionType.SETUP_SETTLEMENT},  # SETUP_SETTLEMENT_2
    3: {HierarchicalActionType.SETUP_ROAD},        # SETUP_ROAD_2
    4: {HierarchicalActionType.ROLL_DICE},         # ROLL_DICE
    5: {HierarchicalActionType.DISCARD},           # DISCARD
    6: {HierarchicalActionType.MOVE_ROBBER},       # MOVE_ROBBER
    7: {HierarchicalActionType.STEAL},             # STEAL
    8: {  # MAIN phase
        HierarchicalActionType.END_TURN,
        HierarchicalActionType.BUILD_ROAD,
        HierarchicalActionType.BUILD_SETTLEMENT,
        HierarchicalActionType.BUILD_CITY,
        HierarchicalActionType.BUY_DEV_CARD,
        HierarchicalActionType.PLAY_DEV_CARD,
        HierarchicalActionType.BANK_TRADE,
        HierarchicalActionType.PROPOSE_TRADE,
    },
    9: {  # TRADE_RESPONSE
        HierarchicalActionType.ACCEPT_TRADE,
        HierarchicalActionType.REJECT_TRADE,
        HierarchicalActionType.COUNTER_OFFER,
    },
    10: {  # TRADE_CONFIRM
        HierarchicalActionType.CONFIRM_TRADE,
        HierarchicalActionType.DECLINE_TRADE,
        HierarchicalActionType.CANCEL_TRADE,
    },
    11: set(),  # GAME_OVER
}


# =============================================================================
# Resource and Validity Extraction
# =============================================================================

def extract_game_phase(observation: torch.Tensor) -> torch.Tensor:
    """
    Extract game phase from observation.

    Args:
        observation: (batch, 176, 65) or (176, 65) tensor

    Returns:
        (batch,) or scalar tensor with phase index (0-11)
    """
    if observation.dim() == 2:
        context = observation[CONTEXT_ROW, :]
        phase_onehot = context[PHASE_START_IDX:PHASE_START_IDX + NUM_GAME_PHASES]
        return phase_onehot.argmax()
    else:
        context = observation[:, CONTEXT_ROW, :]
        phase_onehot = context[:, PHASE_START_IDX:PHASE_START_IDX + NUM_GAME_PHASES]
        return phase_onehot.argmax(dim=1)


def extract_player_resources(observation: torch.Tensor) -> torch.Tensor:
    """
    Extract current player's resources from observation.

    Args:
        observation: (batch, 176, 65) or (176, 65) tensor

    Returns:
        (batch, 5) or (5,) tensor with resource counts [lumber, brick, wool, grain, ore]
        Values are actual counts (denormalized from observation encoding).
    """
    if observation.dim() == 2:
        player_row = observation[PLAYER_START, :]  # Current player is first
        resources = player_row[PLAYER_LUMBER_IDX:PLAYER_ORE_IDX + 1]
        # Denormalize: values were encoded as count * 0.1
        return (resources * 10).round()
    else:
        player_rows = observation[:, PLAYER_START, :]
        resources = player_rows[:, PLAYER_LUMBER_IDX:PLAYER_ORE_IDX + 1]
        return (resources * 10).round()


def extract_player_dev_cards(observation: torch.Tensor) -> torch.Tensor:
    """
    Extract current player's dev cards from observation.

    Args:
        observation: (batch, 176, 65) tensor

    Returns:
        (batch, 5) tensor with dev card counts [knight, road_building, yop, monopoly, vp]
        Values are actual counts (denormalized).
    """
    if observation.dim() == 2:
        player_row = observation[PLAYER_START, :]
        dev_cards = player_row[PLAYER_KNIGHT_IDX:PLAYER_VP_CARD_IDX + 1]
        return (dev_cards * 5).round()  # Encoded as count * 0.2
    else:
        player_rows = observation[:, PLAYER_START, :]
        dev_cards = player_rows[:, PLAYER_KNIGHT_IDX:PLAYER_VP_CARD_IDX + 1]
        return (dev_cards * 5).round()


def extract_corner_validity(observation: torch.Tensor) -> torch.Tensor:
    """
    Extract which corners are valid settlement locations.

    Args:
        observation: (batch, 176, 65) tensor

    Returns:
        (batch, 54) boolean tensor indicating valid corners
    """
    if observation.dim() == 2:
        corner_rows = observation[CORNER_START:CORNER_END, :]
        return corner_rows[:, CORNER_IS_VALID_IDX] > 0.5
    else:
        corner_rows = observation[:, CORNER_START:CORNER_END, :]
        return corner_rows[:, :, CORNER_IS_VALID_IDX] > 0.5


def extract_edge_validity(observation: torch.Tensor) -> torch.Tensor:
    """
    Extract which edges are valid road locations.

    Args:
        observation: (batch, 176, 65) tensor

    Returns:
        (batch, 72) boolean tensor indicating valid edges
    """
    if observation.dim() == 2:
        edge_rows = observation[EDGE_START:EDGE_END, :]
        return edge_rows[:, EDGE_IS_VALID_IDX] > 0.5
    else:
        edge_rows = observation[:, EDGE_START:EDGE_END, :]
        return edge_rows[:, :, EDGE_IS_VALID_IDX] > 0.5


def extract_corner_ownership(observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract corner ownership and building type.

    Args:
        observation: (batch, 176, 65) tensor

    Returns:
        Tuple of:
        - (batch, 54) tensor: -1 if empty, 0-3 for owner (relative to current player)
        - (batch, 54) tensor: 0=empty, 1=settlement, 2=city
    """
    if observation.dim() == 2:
        corner_rows = observation[CORNER_START:CORNER_END, :]
        # Building type
        building_type = (
            corner_rows[:, CORNER_SETTLEMENT_IDX] * 1 +
            corner_rows[:, CORNER_CITY_IDX] * 2
        ).long()
        # Owner: -1 if no owner, else relative player index
        has_owner = corner_rows[:, CORNER_NO_OWNER_IDX] < 0.5
        owner = torch.full((NUM_CORNERS,), -1, dtype=torch.long, device=observation.device)
        for rel_player in range(NUM_PLAYERS):
            is_owner = corner_rows[:, CORNER_OWNER_START_IDX + rel_player] > 0.5
            owner[is_owner] = rel_player
        return owner, building_type
    else:
        batch_size = observation.shape[0]
        corner_rows = observation[:, CORNER_START:CORNER_END, :]
        building_type = (
            corner_rows[:, :, CORNER_SETTLEMENT_IDX] * 1 +
            corner_rows[:, :, CORNER_CITY_IDX] * 2
        ).long()
        owner = torch.full((batch_size, NUM_CORNERS), -1, dtype=torch.long, device=observation.device)
        for rel_player in range(NUM_PLAYERS):
            is_owner = corner_rows[:, :, CORNER_OWNER_START_IDX + rel_player] > 0.5
            owner[is_owner] = rel_player
        return owner, building_type


def can_afford(resources: torch.Tensor, cost: list) -> torch.Tensor:
    """
    Check if player can afford a building.

    Args:
        resources: (batch, 5) or (5,) tensor of resource counts
        cost: List of 5 integers [lumber, brick, wool, grain, ore]

    Returns:
        (batch,) or scalar boolean tensor
    """
    cost_tensor = torch.tensor(cost, dtype=resources.dtype, device=resources.device)
    if resources.dim() == 1:
        return (resources >= cost_tensor).all()
    else:
        return (resources >= cost_tensor.unsqueeze(0)).all(dim=1)


# =============================================================================
# Full Action Mask Extraction
# =============================================================================

def get_action_type_mask(
    observation: torch.Tensor,
    phase: Optional[torch.Tensor] = None,
    use_resource_masking: bool = True,
) -> torch.Tensor:
    """
    Get mask of valid action types based on game phase and resources.

    This applies:
    1. Phase-based masking: only allow actions valid for current phase
    2. Resource-based masking (if enabled): check if player can afford actions

    Args:
        observation: (batch, 176, 65) tensor
        phase: Optional pre-computed phase tensor
        use_resource_masking: Whether to apply resource-based masking

    Returns:
        (batch, NUM_ACTION_TYPES) boolean mask of valid action types
    """
    if observation.dim() == 2:
        observation = observation.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = observation.shape[0]
    device = observation.device

    # Extract phase if not provided
    if phase is None:
        phase = extract_game_phase(observation)

    # Extract resources for affordability checks
    if use_resource_masking:
        resources = extract_player_resources(observation)

    # Initialize mask (all False)
    mask = torch.zeros(batch_size, NUM_ACTION_TYPES, dtype=torch.bool, device=device)

    for i in range(batch_size):
        p = phase[i].item()

        # Get phase-valid types
        phase_valid = PHASE_TO_VALID_TYPES.get(p, set())

        for action_type in phase_valid:
            # Start with True (phase-valid)
            is_valid = True

            # Apply resource checks if enabled and in MAIN phase
            if use_resource_masking and p == 8:  # MAIN phase
                res = resources[i]

                if action_type == HierarchicalActionType.BUILD_ROAD:
                    is_valid = can_afford(res, ROAD_COST).item()
                elif action_type == HierarchicalActionType.BUILD_SETTLEMENT:
                    is_valid = can_afford(res, SETTLEMENT_COST).item()
                elif action_type == HierarchicalActionType.BUILD_CITY:
                    is_valid = can_afford(res, CITY_COST).item()
                elif action_type == HierarchicalActionType.BUY_DEV_CARD:
                    is_valid = can_afford(res, DEV_CARD_COST).item()
                # Note: BANK_TRADE, PROPOSE_TRADE need at least 1 resource to give
                elif action_type in (HierarchicalActionType.BANK_TRADE,
                                    HierarchicalActionType.PROPOSE_TRADE):
                    # Need at least 1 of some resource to trade
                    is_valid = (res.sum() > 0).item()
                # END_TURN, PLAY_DEV_CARD always valid if phase allows

            mask[i, action_type.value] = is_valid

        # Safety: if no types valid, allow all phase-valid types (ignore resources)
        if not mask[i].any():
            for action_type in phase_valid:
                mask[i, action_type.value] = True
            # If still none, allow all
            if not mask[i].any():
                mask[i, :] = True

    if squeeze_output:
        mask = mask.squeeze(0)

    return mask


def get_location_masks(
    observation: torch.Tensor,
    action_types: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Get location masks for each location head based on action type.

    Args:
        observation: (batch, 176, 65) tensor
        action_types: (batch,) tensor of predicted action type indices

    Returns:
        Dict of location head name -> (batch, head_size) boolean mask
    """
    if observation.dim() == 2:
        observation = observation.unsqueeze(0)
        action_types = action_types.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = observation.shape[0]
    device = observation.device

    # Extract validity info
    corner_valid = extract_corner_validity(observation)
    edge_valid = extract_edge_validity(observation)
    corner_owner, building_type = extract_corner_ownership(observation)

    # Initialize masks for each head
    masks = {
        'corner': torch.zeros(batch_size, NUM_CORNERS, dtype=torch.bool, device=device),
        'edge': torch.zeros(batch_size, NUM_EDGES, dtype=torch.bool, device=device),
        'tile': torch.ones(batch_size, NUM_TILES, dtype=torch.bool, device=device),
        'player': torch.ones(batch_size, NUM_PLAYERS, dtype=torch.bool, device=device),
        'trade_template': torch.ones(batch_size, NUM_TRADE_TEMPLATES, dtype=torch.bool, device=device),
        'small_select': torch.ones(batch_size, 25, dtype=torch.bool, device=device),
    }

    for i in range(batch_size):
        action_type = HierarchicalActionType(action_types[i].item())
        head = TYPE_TO_LOCATION_HEAD[action_type]

        if head == LocationHead.CORNER:
            if action_type in (HierarchicalActionType.BUILD_SETTLEMENT,
                              HierarchicalActionType.SETUP_SETTLEMENT):
                masks['corner'][i] = corner_valid[i]
            elif action_type == HierarchicalActionType.BUILD_CITY:
                # Can only upgrade own settlements
                masks['corner'][i] = (corner_owner[i] == 0) & (building_type[i] == 1)

        elif head == LocationHead.EDGE:
            if action_type in (HierarchicalActionType.BUILD_ROAD,
                              HierarchicalActionType.SETUP_ROAD):
                masks['edge'][i] = edge_valid[i]

        elif head == LocationHead.TILE:
            # Move robber: can move to any tile except current robber location
            # We don't have robber location easily, so allow all tiles
            # In practice the model will learn to avoid current robber tile
            masks['tile'][i] = True

        elif head == LocationHead.PLAYER:
            # Steal: can steal from players at robber location
            # For simplicity, allow all non-self players
            masks['player'][i, 0] = False  # Can't steal from self (relative position 0)
            masks['player'][i, 1:] = True

    if squeeze_output:
        for key in masks:
            masks[key] = masks[key].squeeze(0)

    return masks


def get_full_hierarchical_mask(
    observation: torch.Tensor,
    use_resource_masking: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Get complete hierarchical action masks.

    Args:
        observation: (batch, 176, 65) tensor

    Returns:
        Tuple of:
        - (batch, NUM_ACTION_TYPES) type mask
        - Dict of location head name -> (batch, head_size) location mask
    """
    type_mask = get_action_type_mask(observation, use_resource_masking=use_resource_masking)

    # For location masks, we need to know which action type will be selected
    # During training, we use the ground truth type
    # During inference, we use the predicted type
    # For now, return masks for all heads (to be filtered by actual type)

    batch_size = observation.shape[0] if observation.dim() == 3 else 1
    device = observation.device

    corner_valid = extract_corner_validity(observation)
    edge_valid = extract_edge_validity(observation)
    corner_owner, building_type = extract_corner_ownership(observation)

    # Prepare location masks that can be indexed by action type
    location_masks = {
        'corner': corner_valid,  # For BUILD_SETTLEMENT, SETUP_SETTLEMENT
        'corner_city': (corner_owner == 0) & (building_type == 1) if observation.dim() == 3
                       else ((corner_owner == 0) & (building_type == 1)).unsqueeze(0),
        'edge': edge_valid,  # For BUILD_ROAD, SETUP_ROAD
        'tile': torch.ones(batch_size if observation.dim() == 3 else 1, NUM_TILES,
                          dtype=torch.bool, device=device),
        'player': torch.ones(batch_size if observation.dim() == 3 else 1, NUM_PLAYERS,
                            dtype=torch.bool, device=device),
        'trade_template': torch.ones(batch_size if observation.dim() == 3 else 1,
                                     NUM_TRADE_TEMPLATES, dtype=torch.bool, device=device),
        'small_select': torch.ones(batch_size if observation.dim() == 3 else 1, 25,
                                   dtype=torch.bool, device=device),
    }

    # Set self-steal mask
    location_masks['player'][:, 0] = False if observation.dim() == 3 else False

    return type_mask, location_masks
