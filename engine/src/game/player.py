"""Player state management for Settlers of Catan."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from collections import Counter

from .constants import (
    Resource, DevCardType, PortType,
    MAX_SETTLEMENTS, MAX_CITIES, MAX_ROADS,
    BUILDING_COSTS, BANK_TRADE_RATIO, GENERIC_PORT_RATIO, SPECIAL_PORT_RATIO
)


@dataclass
class Player:
    """Represents a player in the game."""
    id: int
    resources: Dict[Resource, int] = field(default_factory=lambda: {r: 0 for r in Resource.tradeable()})
    dev_cards: Dict[DevCardType, int] = field(default_factory=lambda: {d: 0 for d in DevCardType})
    dev_cards_bought_this_turn: Dict[DevCardType, int] = field(default_factory=lambda: {d: 0 for d in DevCardType})
    played_dev_card_this_turn: bool = False
    settlements: List[int] = field(default_factory=list)  # Corner IDs
    cities: List[int] = field(default_factory=list)  # Corner IDs
    roads: List[int] = field(default_factory=list)  # Edge IDs
    ports: List[PortType] = field(default_factory=list)
    knights_played: int = 0
    longest_road_length: int = 0
    has_longest_road: bool = False
    has_largest_army: bool = False
    victory_points_from_dev_cards: int = 0

    # Dev card inference tracking (for hidden VP/Monopoly/etc. prediction)
    dev_card_purchase_turns: List[int] = field(default_factory=list)  # Turn each dev card was bought
    times_blocked_with_cards: int = 0  # Times robbed while holding dev cards
    times_played_knight_when_blocked: int = 0  # Times played knight after being robbed

    @property
    def total_resources(self) -> int:
        """Total number of resource cards."""
        return sum(self.resources.values())

    @property
    def victory_points(self) -> int:
        """Calculate total visible victory points."""
        vp = 0
        vp += len(self.settlements)  # 1 VP each
        vp += len(self.cities) * 2   # 2 VP each
        if self.has_longest_road:
            vp += 2
        if self.has_largest_army:
            vp += 2
        vp += self.victory_points_from_dev_cards
        return vp

    @property
    def hidden_victory_points(self) -> int:
        """Victory points from VP dev cards (hidden until game end)."""
        return self.dev_cards.get(DevCardType.VICTORY_POINT, 0)

    @property
    def total_victory_points(self) -> int:
        """Total VP including hidden VP cards."""
        return self.victory_points + self.hidden_victory_points

    def can_afford(self, building_type: str) -> bool:
        """Check if player can afford to build something."""
        if building_type not in BUILDING_COSTS:
            return False

        costs = BUILDING_COSTS[building_type]
        for resource, amount in costs.items():
            if self.resources.get(resource, 0) < amount:
                return False
        return True

    def pay_for(self, building_type: str) -> bool:
        """Pay the cost for a building. Returns False if cannot afford."""
        if not self.can_afford(building_type):
            return False

        costs = BUILDING_COSTS[building_type]
        for resource, amount in costs.items():
            self.resources[resource] -= amount
        return True

    def add_resource(self, resource: Resource, amount: int = 1) -> None:
        """Add resources to the player."""
        if resource in self.resources:
            self.resources[resource] += amount

    def remove_resource(self, resource: Resource, amount: int = 1) -> bool:
        """Remove resources from the player. Returns False if not enough."""
        if resource not in self.resources or self.resources[resource] < amount:
            return False
        self.resources[resource] -= amount
        return True

    def has_resources(self, resources: Dict[Resource, int]) -> bool:
        """Check if player has the specified resources."""
        for resource, amount in resources.items():
            if self.resources.get(resource, 0) < amount:
                return False
        return True

    def can_build_settlement(self) -> bool:
        """Check if player can build a settlement (resources + pieces)."""
        return (
            self.can_afford('settlement') and
            len(self.settlements) < MAX_SETTLEMENTS
        )

    def can_build_city(self) -> bool:
        """Check if player can build a city (resources + pieces + settlements)."""
        return (
            self.can_afford('city') and
            len(self.cities) < MAX_CITIES and
            len(self.settlements) > 0  # Must have a settlement to upgrade
        )

    def can_build_road(self) -> bool:
        """Check if player can build a road (resources + pieces)."""
        return (
            self.can_afford('road') and
            len(self.roads) < MAX_ROADS
        )

    def can_buy_dev_card(self) -> bool:
        """Check if player can buy a development card."""
        return self.can_afford('dev_card')

    def can_play_dev_card(self, card_type: DevCardType) -> bool:
        """Check if player can play a development card."""
        # VP cards are never played
        if card_type == DevCardType.VICTORY_POINT:
            return False

        # Can only play one dev card per turn
        if self.played_dev_card_this_turn:
            return False

        # Must have the card and not have bought it this turn
        available = self.dev_cards.get(card_type, 0) - self.dev_cards_bought_this_turn.get(card_type, 0)
        return available > 0

    def add_dev_card(self, card_type: DevCardType, bought_this_turn: bool = True, turn_number: int = 0) -> None:
        """Add a development card to the player.

        Args:
            card_type: The type of dev card.
            bought_this_turn: Whether the card was bought this turn (can't play until next).
            turn_number: The turn number when the card was bought (for age tracking).
        """
        self.dev_cards[card_type] = self.dev_cards.get(card_type, 0) + 1
        if bought_this_turn:
            self.dev_cards_bought_this_turn[card_type] = \
                self.dev_cards_bought_this_turn.get(card_type, 0) + 1
        # Track purchase turn for dev card age inference
        self.dev_card_purchase_turns.append(turn_number)

    def play_dev_card(self, card_type: DevCardType) -> bool:
        """Play a development card. Returns False if cannot play."""
        if not self.can_play_dev_card(card_type):
            return False

        self.dev_cards[card_type] -= 1
        self.played_dev_card_this_turn = True

        if card_type == DevCardType.KNIGHT:
            self.knights_played += 1

        # Remove oldest purchase turn (FIFO - oldest cards played first assumption)
        if self.dev_card_purchase_turns:
            self.dev_card_purchase_turns.pop(0)

        return True

    @property
    def total_dev_cards(self) -> int:
        """Total number of dev cards held."""
        return sum(self.dev_cards.values())

    def get_dev_card_hold_duration(self, current_turn: int) -> float:
        """Calculate average turns dev cards have been held.

        Args:
            current_turn: The current turn number.

        Returns:
            Average turns held, or 0 if no cards.
        """
        if not self.dev_card_purchase_turns:
            return 0.0
        ages = [current_turn - turn for turn in self.dev_card_purchase_turns]
        return sum(ages) / len(ages)

    def get_knight_play_ratio(self) -> float:
        """Calculate ratio of times knight was played when blocked.

        Returns:
            Ratio in [0, 1], or 0.5 if no data.
        """
        if self.times_blocked_with_cards == 0:
            return 0.5  # Unknown, assume neutral
        return self.times_played_knight_when_blocked / self.times_blocked_with_cards

    def get_likely_hidden_non_knights(self) -> float:
        """Estimate number of non-knight dev cards held.

        Uses knight play ratio to infer card composition.
        If player rarely plays knights when blocked, cards are likely non-knights.

        Returns:
            Estimated non-knight count (0 to total_dev_cards).
        """
        total = self.total_dev_cards
        if total == 0:
            return 0.0

        # If low knight play ratio, assume more non-knights
        # knight_ratio = 1.0 means all knights, 0.0 means no knights
        knight_ratio = self.get_knight_play_ratio()

        # Estimate: non_knight_fraction = 1 - knight_ratio
        # But we need to account for VP cards that can't be played
        estimated_non_knights = total * (1.0 - knight_ratio)

        return estimated_non_knights

    # Class-level constant for port-to-resource mapping (computed once)
    _PORT_RESOURCE_MAP = {
        PortType.LUMBER: Resource.LUMBER,
        PortType.BRICK: Resource.BRICK,
        PortType.WOOL: Resource.WOOL,
        PortType.GRAIN: Resource.GRAIN,
        PortType.ORE: Resource.ORE,
    }

    def get_trade_ratio(self, resource: Resource) -> int:
        """Get the best trade ratio for a resource (cached after first call)."""
        # Lazy-build cache on first call or when ports change
        if not hasattr(self, '_trade_ratio_cache') or self._trade_ratio_cache_ports != tuple(self.ports):
            self._rebuild_trade_ratio_cache()

        return self._trade_ratio_cache.get(resource, BANK_TRADE_RATIO)

    def _rebuild_trade_ratio_cache(self) -> None:
        """Rebuild the trade ratio cache based on current ports."""
        self._trade_ratio_cache = {}
        self._trade_ratio_cache_ports = tuple(self.ports)

        # Default all to 4:1
        for res in Resource.tradeable():
            self._trade_ratio_cache[res] = BANK_TRADE_RATIO

        # Apply 3:1 generic port if available
        if PortType.GENERIC in self.ports:
            for res in Resource.tradeable():
                self._trade_ratio_cache[res] = GENERIC_PORT_RATIO

        # Apply 2:1 specific ports (overrides 3:1)
        for port in self.ports:
            if port in self._PORT_RESOURCE_MAP:
                res = self._PORT_RESOURCE_MAP[port]
                self._trade_ratio_cache[res] = SPECIAL_PORT_RATIO

    def can_bank_trade(self, give_resource: Resource, receive_resource: Resource) -> bool:
        """Check if player can do a bank trade."""
        if give_resource == receive_resource:
            return False
        ratio = self.get_trade_ratio(give_resource)
        return self.resources.get(give_resource, 0) >= ratio

    def do_bank_trade(self, give_resource: Resource, receive_resource: Resource) -> bool:
        """Execute a bank trade. Returns False if cannot trade."""
        if not self.can_bank_trade(give_resource, receive_resource):
            return False

        ratio = self.get_trade_ratio(give_resource)
        self.resources[give_resource] -= ratio
        self.resources[receive_resource] += 1
        return True

    def discard_half(self, resources_to_discard: Dict[Resource, int]) -> bool:
        """Discard resources (when 7 rolled with >7 cards).

        Args:
            resources_to_discard: Dict mapping resource to amount to discard.

        Returns:
            True if valid discard, False otherwise.
        """
        # Verify the discard is valid
        total_to_discard = sum(resources_to_discard.values())
        required_discard = self.total_resources // 2

        if total_to_discard != required_discard:
            return False

        # Verify player has all resources
        for resource, amount in resources_to_discard.items():
            if self.resources.get(resource, 0) < amount:
                return False

        # Execute discard
        for resource, amount in resources_to_discard.items():
            self.resources[resource] -= amount

        return True

    def end_turn(self) -> None:
        """Reset turn-specific state."""
        self.dev_cards_bought_this_turn = {d: 0 for d in DevCardType}
        self.played_dev_card_this_turn = False

    def get_owned_edges(self) -> Set[int]:
        """Get set of edge IDs where player has roads."""
        return set(self.roads)

    def get_owned_corners(self) -> Set[int]:
        """Get set of corner IDs where player has buildings."""
        return set(self.settlements + self.cities)

    def copy(self) -> 'Player':
        """Create a deep copy of the player."""
        new_player = Player(self.id)
        new_player.resources = self.resources.copy()
        new_player.dev_cards = self.dev_cards.copy()
        new_player.dev_cards_bought_this_turn = self.dev_cards_bought_this_turn.copy()
        new_player.played_dev_card_this_turn = self.played_dev_card_this_turn
        new_player.settlements = self.settlements.copy()
        new_player.cities = self.cities.copy()
        new_player.roads = self.roads.copy()
        new_player.ports = self.ports.copy()
        new_player.knights_played = self.knights_played
        new_player.longest_road_length = self.longest_road_length
        new_player.has_longest_road = self.has_longest_road
        new_player.has_largest_army = self.has_largest_army
        new_player.victory_points_from_dev_cards = self.victory_points_from_dev_cards
        # Dev card inference tracking
        new_player.dev_card_purchase_turns = self.dev_card_purchase_turns.copy()
        new_player.times_blocked_with_cards = self.times_blocked_with_cards
        new_player.times_played_knight_when_blocked = self.times_played_knight_when_blocked
        return new_player
