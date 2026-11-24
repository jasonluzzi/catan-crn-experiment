"""Action decoder for converting action indices to/from game actions."""

import numpy as np
from typing import List, Tuple

from ..game.actions import (
    Action, TOTAL_ACTIONS, index_to_action, action_to_index,
    ACTION_ROLL_DICE, ACTION_END_TURN,
    ACTION_BUILD_ROAD_START, ACTION_BUILD_ROAD_END,
    ACTION_BUILD_SETTLEMENT_START, ACTION_BUILD_SETTLEMENT_END,
    ACTION_BUILD_CITY_START, ACTION_BUILD_CITY_END,
    ACTION_BUY_DEV_CARD,
    ACTION_PLAY_KNIGHT, ACTION_PLAY_ROAD_BUILDING,
    ACTION_PLAY_YEAR_OF_PLENTY, ACTION_PLAY_MONOPOLY,
    ACTION_BANK_TRADE_START, ACTION_BANK_TRADE_END,
    ACTION_MOVE_ROBBER_START, ACTION_MOVE_ROBBER_END,
    ACTION_STEAL_START, ACTION_STEAL_END,
    ACTION_PLACE_SETTLEMENT_START, ACTION_PLACE_SETTLEMENT_END,
    ACTION_PLACE_ROAD_START, ACTION_PLACE_ROAD_END,
)
from ..game.constants import ActionType, Resource


class ActionDecoder:
    """Handles conversion between action indices and game actions."""

    def __init__(self):
        """Initialize the action decoder."""
        self.action_space_size = TOTAL_ACTIONS

        # Pre-compute action type ranges for efficiency
        self.action_ranges = [
            (ACTION_ROLL_DICE, ACTION_ROLL_DICE + 1, ActionType.ROLL_DICE),
            (ACTION_END_TURN, ACTION_END_TURN + 1, ActionType.END_TURN),
            (ACTION_BUILD_ROAD_START, ACTION_BUILD_ROAD_END, ActionType.BUILD_ROAD),
            (ACTION_BUILD_SETTLEMENT_START, ACTION_BUILD_SETTLEMENT_END, ActionType.BUILD_SETTLEMENT),
            (ACTION_BUILD_CITY_START, ACTION_BUILD_CITY_END, ActionType.BUILD_CITY),
            (ACTION_BUY_DEV_CARD, ACTION_BUY_DEV_CARD + 1, ActionType.BUY_DEV_CARD),
            (ACTION_PLAY_KNIGHT, ACTION_PLAY_KNIGHT + 1, ActionType.PLAY_KNIGHT),
            (ACTION_PLAY_ROAD_BUILDING, ACTION_PLAY_ROAD_BUILDING + 1, ActionType.PLAY_ROAD_BUILDING),
            (ACTION_PLAY_YEAR_OF_PLENTY, ACTION_PLAY_YEAR_OF_PLENTY + 1, ActionType.PLAY_YEAR_OF_PLENTY),
            (ACTION_PLAY_MONOPOLY, ACTION_PLAY_MONOPOLY + 1, ActionType.PLAY_MONOPOLY),
            (ACTION_BANK_TRADE_START, ACTION_BANK_TRADE_END, ActionType.BANK_TRADE),
            (ACTION_MOVE_ROBBER_START, ACTION_MOVE_ROBBER_END, ActionType.MOVE_ROBBER),
            (ACTION_STEAL_START, ACTION_STEAL_END, ActionType.STEAL),
            (ACTION_PLACE_SETTLEMENT_START, ACTION_PLACE_SETTLEMENT_END, ActionType.PLACE_SETTLEMENT),
            (ACTION_PLACE_ROAD_START, ACTION_PLACE_ROAD_END, ActionType.PLACE_ROAD),
        ]

    def decode(self, action_idx: int) -> Action:
        """Convert action index to Action object."""
        return index_to_action(action_idx)

    def encode(self, action: Action) -> int:
        """Convert Action object to action index."""
        return action_to_index(action)

    def get_action_type(self, action_idx: int) -> ActionType:
        """Get the type of an action from its index."""
        for start, end, action_type in self.action_ranges:
            if start <= action_idx < end:
                return action_type
        raise ValueError(f"Invalid action index: {action_idx}")

    def get_action_description(self, action_idx: int) -> str:
        """Get a human-readable description of an action."""
        action = self.decode(action_idx)
        return str(action)

    def get_actions_by_type(self, action_type: ActionType) -> List[int]:
        """Get all action indices of a given type."""
        for start, end, at in self.action_ranges:
            if at == action_type:
                return list(range(start, end))
        return []

    def get_trade_resources(self, action_idx: int) -> Tuple[Resource, Resource]:
        """Get the give/receive resources for a trade action."""
        if not (ACTION_BANK_TRADE_START <= action_idx < ACTION_BANK_TRADE_END):
            raise ValueError(f"Action {action_idx} is not a trade action")

        trade_idx = action_idx - ACTION_BANK_TRADE_START
        resources = list(Resource.tradeable())
        give_idx = trade_idx // 5
        receive_idx = trade_idx % 5
        return resources[give_idx], resources[receive_idx]

    def get_location_id(self, action_idx: int) -> int:
        """Get the location (corner, edge, or tile) ID for a location-based action."""
        action = self.decode(action_idx)

        if action.corner_id is not None:
            return action.corner_id
        if action.edge_id is not None:
            return action.edge_id
        if action.tile_id is not None:
            return action.tile_id

        raise ValueError(f"Action {action_idx} does not have a location")

    def mask_to_actions(self, mask: np.ndarray) -> List[int]:
        """Convert action mask to list of valid action indices."""
        return list(np.where(mask)[0])

    def actions_to_mask(self, actions: List[int]) -> np.ndarray:
        """Convert list of valid action indices to action mask."""
        mask = np.zeros(self.action_space_size, dtype=bool)
        for action_idx in actions:
            mask[action_idx] = True
        return mask

    def get_action_category_mask(self, mask: np.ndarray) -> dict:
        """Get which action categories are available in a mask."""
        categories = {}
        for start, end, action_type in self.action_ranges:
            categories[action_type.name] = mask[start:end].any()
        return categories

    def sample_from_mask(self, mask: np.ndarray) -> int:
        """Randomly sample a valid action from a mask."""
        valid_actions = self.mask_to_actions(mask)
        if not valid_actions:
            raise ValueError("No valid actions in mask")
        return np.random.choice(valid_actions)


# Singleton instance
action_decoder = ActionDecoder()
