"""Game constants and enumerations for Settlers of Catan."""

from enum import Enum, IntEnum
from typing import Dict


class Resource(IntEnum):
    """Resource types in Catan."""
    LUMBER = 0
    BRICK = 1
    WOOL = 2
    GRAIN = 3
    ORE = 4
    DESERT = 5  # No resource (for desert tile)

    @classmethod
    def tradeable(cls) -> list:
        """Return list of tradeable resources (excludes desert)."""
        return [cls.LUMBER, cls.BRICK, cls.WOOL, cls.GRAIN, cls.ORE]


class DevCardType(IntEnum):
    """Development card types."""
    KNIGHT = 0
    ROAD_BUILDING = 1
    YEAR_OF_PLENTY = 2
    MONOPOLY = 3
    VICTORY_POINT = 4


class GamePhase(IntEnum):
    """Game phases."""
    SETUP_SETTLEMENT_1 = 0  # First settlement placement
    SETUP_ROAD_1 = 1        # First road placement
    SETUP_SETTLEMENT_2 = 2  # Second settlement placement
    SETUP_ROAD_2 = 3        # Second road placement
    ROLL_DICE = 4           # Must roll dice
    DISCARD = 5             # Players must discard (after 7 rolled)
    MOVE_ROBBER = 6         # Must move robber
    STEAL = 7               # Must steal from adjacent player
    MAIN = 8                # Main phase - build/trade/play cards
    TRADE_RESPONSE = 9      # Opponents respond to trade offer (accept/reject/counter)
    TRADE_CONFIRM = 10      # Proposer confirms trade with accepting player
    GAME_OVER = 11          # Game ended


class ActionType(IntEnum):
    """Types of actions a player can take."""
    ROLL_DICE = 0
    END_TURN = 1
    BUILD_ROAD = 2
    BUILD_SETTLEMENT = 3
    BUILD_CITY = 4
    BUY_DEV_CARD = 5
    PLAY_KNIGHT = 6
    PLAY_ROAD_BUILDING = 7
    PLAY_YEAR_OF_PLENTY = 8
    PLAY_MONOPOLY = 9
    BANK_TRADE = 10
    PORT_TRADE = 11
    MOVE_ROBBER = 12
    STEAL = 13
    DISCARD = 14
    PLACE_SETTLEMENT = 15  # Setup phase
    PLACE_ROAD = 16        # Setup phase
    # Player-to-player trading
    PROPOSE_TRADE = 17      # Propose a trade offer to all opponents
    ACCEPT_TRADE = 18       # Accept the pending trade offer
    REJECT_TRADE = 19       # Reject the pending trade offer
    COUNTER_OFFER = 20      # Counter with a different trade
    CANCEL_TRADE = 21       # Cancel pending trade (proposer only)
    CONFIRM_TRADE = 22      # Confirm trade with specific accepting player
    DECLINE_TRADE = 23      # Decline trade with specific accepting player


class PortType(IntEnum):
    """Port types for trading."""
    GENERIC = 0    # 3:1 any resource
    LUMBER = 1     # 2:1 lumber
    BRICK = 2      # 2:1 brick
    WOOL = 3       # 2:1 wool
    GRAIN = 4      # 2:1 grain
    ORE = 5        # 2:1 ore


# Building costs
BUILDING_COSTS: Dict[str, Dict[Resource, int]] = {
    'road': {
        Resource.LUMBER: 1,
        Resource.BRICK: 1,
    },
    'settlement': {
        Resource.LUMBER: 1,
        Resource.BRICK: 1,
        Resource.WOOL: 1,
        Resource.GRAIN: 1,
    },
    'city': {
        Resource.GRAIN: 2,
        Resource.ORE: 3,
    },
    'dev_card': {
        Resource.WOOL: 1,
        Resource.GRAIN: 1,
        Resource.ORE: 1,
    },
}

# Development card distribution (25 total)
DEV_CARD_COUNTS: Dict[DevCardType, int] = {
    DevCardType.KNIGHT: 14,
    DevCardType.ROAD_BUILDING: 2,
    DevCardType.YEAR_OF_PLENTY: 2,
    DevCardType.MONOPOLY: 2,
    DevCardType.VICTORY_POINT: 5,
}

# Resource tile distribution (19 total)
TILE_DISTRIBUTION: Dict[Resource, int] = {
    Resource.LUMBER: 4,
    Resource.BRICK: 3,
    Resource.WOOL: 4,
    Resource.GRAIN: 4,
    Resource.ORE: 3,
    Resource.DESERT: 1,
}

# Number token distribution (18 tokens, one for each non-desert tile)
# The desert gets no number
NUMBER_DISTRIBUTION = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

# Probability of rolling each number (out of 36)
ROLL_PROBABILITY = {
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
    8: 5, 9: 4, 10: 3, 11: 2, 12: 1
}

# Game configuration
NUM_PLAYERS = 4
NUM_TILES = 19
NUM_CORNERS = 54
NUM_EDGES = 72
VICTORY_POINTS_TO_WIN = 10
MAX_SETTLEMENTS = 5
MAX_CITIES = 4
MAX_ROADS = 15
BANK_TRADE_RATIO = 4
GENERIC_PORT_RATIO = 3
SPECIAL_PORT_RATIO = 2
MAX_HAND_SIZE_ON_7 = 7
# Player-to-player trading limits
# PHASE 3: Reduced from 3 to 2 to prevent counter-offer spam (was 212 per game)
MAX_COUNTER_OFFER_DEPTH = 2  # Max number of counter-offers in a negotiation chain
MAX_TRADE_RECEIVE = 2        # Max resources received in a trade (best case: 4:2 or 3:1)
INITIAL_BANK_RESOURCES = 19  # Starting resources per type in the bank

# Standard board layout - hex grid with axial coordinates
# Center tile is (0, 0), tiles arranged in rings
TILE_COORDS = [
    (0, 0),   # Center
    # Ring 1 (6 tiles)
    (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1),
    # Ring 2 (12 tiles)
    (2, -2), (2, -1), (2, 0), (1, 1), (0, 2), (-1, 2),
    (-2, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2),
]

# Port locations - which corners have which ports
# Format: (corner_indices, port_type)
PORT_ASSIGNMENTS = [
    ([0, 1], PortType.GENERIC),
    ([3, 4], PortType.LUMBER),
    ([7, 8], PortType.GENERIC),
    ([14, 15], PortType.BRICK),
    ([26, 27], PortType.GENERIC),
    ([37, 38], PortType.WOOL),
    ([45, 46], PortType.GENERIC),
    ([50, 51], PortType.GRAIN),
    ([52, 53], PortType.ORE),
]
