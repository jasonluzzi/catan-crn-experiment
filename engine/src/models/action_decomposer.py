"""
Action decomposer for hierarchical action structure.

Converts between flat 475-action space and hierarchical (type, location) representation.

Hierarchical Structure:
- Stage 1: 20 action types
- Stage 2: 6 location heads (corner, edge, tile, player, trade_template, small_select)
"""

from enum import IntEnum
from typing import Tuple, Dict, Optional
import numpy as np

from ..game.actions import (
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
    ACTION_DISCARD_START, ACTION_DISCARD_END,
    ACTION_PROPOSE_TRADE_START, ACTION_PROPOSE_TRADE_END,
    ACTION_ACCEPT_TRADE, ACTION_REJECT_TRADE,
    ACTION_COUNTER_OFFER_START, ACTION_COUNTER_OFFER_END,
    ACTION_CANCEL_TRADE,
    ACTION_CONFIRM_TRADE_START, ACTION_CONFIRM_TRADE_END,
    ACTION_DECLINE_TRADE_START, ACTION_DECLINE_TRADE_END,
    TOTAL_ACTIONS,
)


class HierarchicalActionType(IntEnum):
    """20 hierarchical action types."""
    ROLL_DICE = 0
    END_TURN = 1
    BUILD_ROAD = 2
    BUILD_SETTLEMENT = 3
    BUILD_CITY = 4
    BUY_DEV_CARD = 5
    PLAY_DEV_CARD = 6  # Groups all 4 dev card types
    BANK_TRADE = 7
    MOVE_ROBBER = 8
    STEAL = 9
    SETUP_SETTLEMENT = 10
    SETUP_ROAD = 11
    DISCARD = 12
    PROPOSE_TRADE = 13
    ACCEPT_TRADE = 14
    REJECT_TRADE = 15
    COUNTER_OFFER = 16
    CANCEL_TRADE = 17
    CONFIRM_TRADE = 18
    DECLINE_TRADE = 19


class LocationHead(IntEnum):
    """6 location/parameter heads."""
    NONE = 0          # No location needed
    CORNER = 1        # 54 corners
    EDGE = 2          # 72 edges
    TILE = 3          # 19 tiles
    PLAYER = 4        # 4 players (or 3 for confirm/decline)
    TRADE_TEMPLATE = 5  # 50 trade templates
    SMALL_SELECT = 6  # Variable (25 max): bank trade, dev card, discard


# Total number of hierarchical action types
NUM_ACTION_TYPES = 20

# Location head output sizes
LOCATION_HEAD_SIZES = {
    LocationHead.NONE: 0,
    LocationHead.CORNER: 54,
    LocationHead.EDGE: 72,
    LocationHead.TILE: 19,
    LocationHead.PLAYER: 4,
    LocationHead.TRADE_TEMPLATE: 50,
    LocationHead.SMALL_SELECT: 25,  # Max size (bank trade = 25, dev card = 4, discard = 5)
}

# Mapping from action type to location head
TYPE_TO_LOCATION_HEAD: Dict[HierarchicalActionType, LocationHead] = {
    HierarchicalActionType.ROLL_DICE: LocationHead.NONE,
    HierarchicalActionType.END_TURN: LocationHead.NONE,
    HierarchicalActionType.BUILD_ROAD: LocationHead.EDGE,
    HierarchicalActionType.BUILD_SETTLEMENT: LocationHead.CORNER,
    HierarchicalActionType.BUILD_CITY: LocationHead.CORNER,
    HierarchicalActionType.BUY_DEV_CARD: LocationHead.NONE,
    HierarchicalActionType.PLAY_DEV_CARD: LocationHead.SMALL_SELECT,  # 4 types
    HierarchicalActionType.BANK_TRADE: LocationHead.SMALL_SELECT,      # 25 combos
    HierarchicalActionType.MOVE_ROBBER: LocationHead.TILE,
    HierarchicalActionType.STEAL: LocationHead.PLAYER,
    HierarchicalActionType.SETUP_SETTLEMENT: LocationHead.CORNER,
    HierarchicalActionType.SETUP_ROAD: LocationHead.EDGE,
    HierarchicalActionType.DISCARD: LocationHead.SMALL_SELECT,         # 5 resources
    HierarchicalActionType.PROPOSE_TRADE: LocationHead.TRADE_TEMPLATE,
    HierarchicalActionType.ACCEPT_TRADE: LocationHead.NONE,
    HierarchicalActionType.REJECT_TRADE: LocationHead.NONE,
    HierarchicalActionType.COUNTER_OFFER: LocationHead.TRADE_TEMPLATE,
    HierarchicalActionType.CANCEL_TRADE: LocationHead.NONE,
    HierarchicalActionType.CONFIRM_TRADE: LocationHead.PLAYER,         # 3 players
    HierarchicalActionType.DECLINE_TRADE: LocationHead.PLAYER,         # 3 players
}

# Flat action ranges - (start, end, hierarchical_type)
FLAT_RANGES = [
    (ACTION_ROLL_DICE, ACTION_ROLL_DICE + 1, HierarchicalActionType.ROLL_DICE),
    (ACTION_END_TURN, ACTION_END_TURN + 1, HierarchicalActionType.END_TURN),
    (ACTION_BUILD_ROAD_START, ACTION_BUILD_ROAD_END, HierarchicalActionType.BUILD_ROAD),
    (ACTION_BUILD_SETTLEMENT_START, ACTION_BUILD_SETTLEMENT_END, HierarchicalActionType.BUILD_SETTLEMENT),
    (ACTION_BUILD_CITY_START, ACTION_BUILD_CITY_END, HierarchicalActionType.BUILD_CITY),
    (ACTION_BUY_DEV_CARD, ACTION_BUY_DEV_CARD + 1, HierarchicalActionType.BUY_DEV_CARD),
    # Dev cards: KNIGHT=183, ROAD_BUILDING=184, YEAR_OF_PLENTY=185, MONOPOLY=186
    (ACTION_PLAY_KNIGHT, ACTION_PLAY_MONOPOLY + 1, HierarchicalActionType.PLAY_DEV_CARD),
    (ACTION_BANK_TRADE_START, ACTION_BANK_TRADE_END, HierarchicalActionType.BANK_TRADE),
    (ACTION_MOVE_ROBBER_START, ACTION_MOVE_ROBBER_END, HierarchicalActionType.MOVE_ROBBER),
    (ACTION_STEAL_START, ACTION_STEAL_END, HierarchicalActionType.STEAL),
    (ACTION_PLACE_SETTLEMENT_START, ACTION_PLACE_SETTLEMENT_END, HierarchicalActionType.SETUP_SETTLEMENT),
    (ACTION_PLACE_ROAD_START, ACTION_PLACE_ROAD_END, HierarchicalActionType.SETUP_ROAD),
    (ACTION_DISCARD_START, ACTION_DISCARD_END, HierarchicalActionType.DISCARD),
    (ACTION_PROPOSE_TRADE_START, ACTION_PROPOSE_TRADE_END, HierarchicalActionType.PROPOSE_TRADE),
    (ACTION_ACCEPT_TRADE, ACTION_ACCEPT_TRADE + 1, HierarchicalActionType.ACCEPT_TRADE),
    (ACTION_REJECT_TRADE, ACTION_REJECT_TRADE + 1, HierarchicalActionType.REJECT_TRADE),
    (ACTION_COUNTER_OFFER_START, ACTION_COUNTER_OFFER_END, HierarchicalActionType.COUNTER_OFFER),
    (ACTION_CANCEL_TRADE, ACTION_CANCEL_TRADE + 1, HierarchicalActionType.CANCEL_TRADE),
    (ACTION_CONFIRM_TRADE_START, ACTION_CONFIRM_TRADE_END, HierarchicalActionType.CONFIRM_TRADE),
    (ACTION_DECLINE_TRADE_START, ACTION_DECLINE_TRADE_END, HierarchicalActionType.DECLINE_TRADE),
]


class ActionDecomposer:
    """
    Converts between flat action indices and hierarchical (type, location) pairs.

    The hierarchical structure separates "what to do" (action type) from
    "where/how to do it" (location/parameter).
    """

    def __init__(self):
        # Build lookup table for O(1) flat -> hierarchical conversion
        self._flat_to_type = np.zeros(TOTAL_ACTIONS, dtype=np.int32)
        self._flat_to_location = np.zeros(TOTAL_ACTIONS, dtype=np.int32)
        self._type_range_starts = {}

        for start, end, action_type in FLAT_RANGES:
            self._type_range_starts[action_type] = start
            for flat_idx in range(start, end):
                self._flat_to_type[flat_idx] = action_type
                self._flat_to_location[flat_idx] = flat_idx - start

        # Build reverse lookup for O(1) hierarchical -> flat conversion
        # We need to handle special cases for SMALL_SELECT head
        self._type_to_flat_start = np.zeros(NUM_ACTION_TYPES, dtype=np.int32)
        for start, end, action_type in FLAT_RANGES:
            self._type_to_flat_start[action_type] = start

    def flat_to_hierarchical(self, action_idx: int) -> Tuple[int, int]:
        """
        Convert flat action index to hierarchical (type_id, location_id).

        Args:
            action_idx: Flat action index (0-474)

        Returns:
            Tuple of (type_id, location_id). location_id is 0 for actions
            that don't require a location.
        """
        if action_idx < 0 or action_idx >= TOTAL_ACTIONS:
            raise ValueError(f"Invalid action index: {action_idx}")

        return int(self._flat_to_type[action_idx]), int(self._flat_to_location[action_idx])

    def hierarchical_to_flat(self, type_id: int, location_id: int = 0) -> int:
        """
        Convert hierarchical (type_id, location_id) to flat action index.

        Args:
            type_id: Action type (0-19)
            location_id: Location within type (0 for non-location actions)

        Returns:
            Flat action index (0-474)
        """
        if type_id < 0 or type_id >= NUM_ACTION_TYPES:
            raise ValueError(f"Invalid type_id: {type_id}")

        flat_start = int(self._type_to_flat_start[type_id])
        return flat_start + location_id

    def get_type_mask(self, flat_mask: np.ndarray) -> np.ndarray:
        """
        Convert flat action mask (475,) to type mask (20,).

        A type is valid if ANY action of that type is valid in the flat mask.

        Args:
            flat_mask: Boolean array of shape (475,)

        Returns:
            Boolean array of shape (20,) indicating valid action types
        """
        type_mask = np.zeros(NUM_ACTION_TYPES, dtype=bool)

        for start, end, action_type in FLAT_RANGES:
            # A type is valid if any action in its range is valid
            if flat_mask[start:end].any():
                type_mask[action_type] = True

        return type_mask

    def get_location_mask(self, type_id: int, flat_mask: np.ndarray) -> np.ndarray:
        """
        Get location mask for a specific action type.

        Args:
            type_id: Action type (0-19)
            flat_mask: Boolean array of shape (475,)

        Returns:
            Boolean array with shape depending on the location head for this type.
            For NONE head, returns empty array.
        """
        location_head = TYPE_TO_LOCATION_HEAD[HierarchicalActionType(type_id)]

        if location_head == LocationHead.NONE:
            # No location needed - return single True
            return np.array([True], dtype=bool)

        # Find the range for this action type
        flat_start = int(self._type_to_flat_start[type_id])

        # Get the size of this type's action range
        for start, end, action_type in FLAT_RANGES:
            if action_type == type_id:
                size = end - start
                break

        # Extract the relevant portion of the flat mask
        location_mask = flat_mask[flat_start:flat_start + size].copy()

        return location_mask

    def get_location_head(self, type_id: int) -> LocationHead:
        """Get the location head used by an action type."""
        return TYPE_TO_LOCATION_HEAD[HierarchicalActionType(type_id)]

    def requires_location(self, type_id: int) -> bool:
        """Check if an action type requires a location selection."""
        return TYPE_TO_LOCATION_HEAD[HierarchicalActionType(type_id)] != LocationHead.NONE

    def get_location_size(self, type_id: int) -> int:
        """Get the number of possible locations for an action type."""
        for start, end, action_type in FLAT_RANGES:
            if action_type == type_id:
                return end - start
        raise ValueError(f"Unknown type_id: {type_id}")

    def batch_flat_to_hierarchical(self, action_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch convert flat action indices to hierarchical.

        Args:
            action_indices: Array of flat action indices

        Returns:
            Tuple of (type_ids, location_ids) arrays
        """
        return (
            self._flat_to_type[action_indices].copy(),
            self._flat_to_location[action_indices].copy()
        )

    def batch_hierarchical_to_flat(self, type_ids: np.ndarray, location_ids: np.ndarray) -> np.ndarray:
        """
        Batch convert hierarchical to flat action indices.

        Args:
            type_ids: Array of action types
            location_ids: Array of location indices

        Returns:
            Array of flat action indices
        """
        flat_starts = self._type_to_flat_start[type_ids]
        return flat_starts + location_ids

    def decompose_flat_actions_for_training(
        self,
        flat_actions: np.ndarray,
        flat_masks: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Decompose a batch of flat actions and masks for hierarchical training.

        Args:
            flat_actions: (batch_size,) flat action indices
            flat_masks: (batch_size, 475) flat action masks

        Returns:
            Dict containing:
                - type_labels: (batch_size,) action type labels
                - location_labels: (batch_size,) location labels
                - type_masks: (batch_size, 20) type masks
                - location_head_ids: (batch_size,) which location head to use
        """
        batch_size = len(flat_actions)

        type_labels, location_labels = self.batch_flat_to_hierarchical(flat_actions)

        type_masks = np.zeros((batch_size, NUM_ACTION_TYPES), dtype=bool)
        location_head_ids = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            type_masks[i] = self.get_type_mask(flat_masks[i])
            location_head_ids[i] = TYPE_TO_LOCATION_HEAD[HierarchicalActionType(type_labels[i])]

        return {
            'type_labels': type_labels,
            'location_labels': location_labels,
            'type_masks': type_masks,
            'location_head_ids': location_head_ids,
        }


# Singleton instance for convenience
action_decomposer = ActionDecomposer()


def flat_to_hierarchical(action_idx: int) -> Tuple[int, int]:
    """Convenience function using singleton decomposer."""
    return action_decomposer.flat_to_hierarchical(action_idx)


def hierarchical_to_flat(type_id: int, location_id: int = 0) -> int:
    """Convenience function using singleton decomposer."""
    return action_decomposer.hierarchical_to_flat(type_id, location_id)


def get_type_mask(flat_mask: np.ndarray) -> np.ndarray:
    """Convenience function using singleton decomposer."""
    return action_decomposer.get_type_mask(flat_mask)


def get_location_mask(type_id: int, flat_mask: np.ndarray) -> np.ndarray:
    """Convenience function using singleton decomposer."""
    return action_decomposer.get_location_mask(type_id, flat_mask)
