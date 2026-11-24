"""Resource tracking and inference for Settlers of Catan.

Tracks known and inferred resources for all players, enabling strategic
decision-making based on opponent resource states.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum

from .constants import Resource, NUM_PLAYERS, BUILDING_COSTS


class ResourceConfidence(IntEnum):
    """Confidence level for resource knowledge."""
    EXACT = 0       # Know exactly (self or observed)
    HIGH = 1        # Very confident (e.g., just gained from dice)
    MEDIUM = 2      # Somewhat confident (inferred from actions)
    LOW = 3         # Uncertain (could have traded/stolen)
    UNKNOWN = 4     # No information


@dataclass
class PlayerResourceEstimate:
    """Estimated resources for a single player."""
    player_id: int
    # Known minimum resources (from observed gains)
    min_resources: Dict[Resource, int] = field(
        default_factory=lambda: {r: 0 for r in Resource.tradeable()}
    )
    # Estimated maximum resources (upper bound)
    max_resources: Dict[Resource, int] = field(
        default_factory=lambda: {r: 19 for r in Resource.tradeable()}
    )
    # Known total resource count
    total_count: int = 0
    # Confidence level
    confidence: ResourceConfidence = ResourceConfidence.UNKNOWN

    def copy(self) -> 'PlayerResourceEstimate':
        """Create a deep copy."""
        estimate = PlayerResourceEstimate(self.player_id)
        estimate.min_resources = self.min_resources.copy()
        estimate.max_resources = self.max_resources.copy()
        estimate.total_count = self.total_count
        estimate.confidence = self.confidence
        return estimate


@dataclass
class ResourceTracker:
    """Tracks resources for all players with inference capabilities.

    For the agent's own resources, tracks exact values.
    For opponents, maintains estimates based on observed events:
    - Dice rolls and settlements (resource gains)
    - Building actions (resource expenditures)
    - Trades (known transfers)
    - Discards (resource losses)
    - Steals (uncertain transfers)
    """
    agent_player_id: int
    estimates: Dict[int, PlayerResourceEstimate] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize estimates for all players."""
        if not self.estimates:
            for player_id in range(NUM_PLAYERS):
                self.estimates[player_id] = PlayerResourceEstimate(player_id)

    def update_exact(self, player_id: int, resources: Dict[Resource, int]) -> None:
        """Update with exact resource knowledge (e.g., for self)."""
        estimate = self.estimates[player_id]
        for resource in Resource.tradeable():
            count = resources.get(resource, 0)
            estimate.min_resources[resource] = count
            estimate.max_resources[resource] = count
        estimate.total_count = sum(resources.values())
        estimate.confidence = ResourceConfidence.EXACT

    def record_resource_gain(
        self,
        player_id: int,
        resource: Resource,
        amount: int
    ) -> None:
        """Record that a player gained resources (from dice roll)."""
        estimate = self.estimates[player_id]
        estimate.min_resources[resource] = min(
            estimate.min_resources[resource] + amount, 19
        )
        estimate.total_count += amount
        if estimate.confidence != ResourceConfidence.EXACT:
            estimate.confidence = ResourceConfidence.HIGH

    def record_resource_spent(
        self,
        player_id: int,
        resources: Dict[Resource, int]
    ) -> None:
        """Record that a player spent resources (from building)."""
        estimate = self.estimates[player_id]
        for resource, amount in resources.items():
            estimate.min_resources[resource] = max(
                0, estimate.min_resources[resource] - amount
            )
        estimate.total_count = max(0, estimate.total_count - sum(resources.values()))

    def record_build_action(self, player_id: int, building_type: str) -> None:
        """Record a build action and infer resource expenditure."""
        if building_type in BUILDING_COSTS:
            self.record_resource_spent(player_id, BUILDING_COSTS[building_type])

    def record_trade(
        self,
        player_id: int,
        gave: Dict[Resource, int],
        received: Dict[Resource, int]
    ) -> None:
        """Record a completed trade."""
        estimate = self.estimates[player_id]

        # Decrease for given resources
        for resource, amount in gave.items():
            estimate.min_resources[resource] = max(
                0, estimate.min_resources[resource] - amount
            )

        # Increase for received resources
        for resource, amount in received.items():
            estimate.min_resources[resource] = min(
                estimate.min_resources[resource] + amount, 19
            )

        # Update total
        net_change = sum(received.values()) - sum(gave.values())
        estimate.total_count = max(0, estimate.total_count + net_change)

    def record_discard(self, player_id: int, resource: Resource, amount: int) -> None:
        """Record that a player discarded resources."""
        estimate = self.estimates[player_id]
        estimate.min_resources[resource] = max(
            0, estimate.min_resources[resource] - amount
        )
        estimate.total_count = max(0, estimate.total_count - amount)

    def record_steal(
        self,
        stealer_id: int,
        victim_id: int,
        stolen_resource: Optional[Resource] = None
    ) -> None:
        """Record a steal action.

        If stolen_resource is known (agent involved), update precisely.
        Otherwise, reduce confidence and adjust bounds.
        """
        if stolen_resource is not None:
            # Known steal (agent was involved)
            self.estimates[victim_id].min_resources[stolen_resource] = max(
                0, self.estimates[victim_id].min_resources[stolen_resource] - 1
            )
            self.estimates[victim_id].total_count = max(
                0, self.estimates[victim_id].total_count - 1
            )
            self.estimates[stealer_id].min_resources[stolen_resource] = min(
                self.estimates[stealer_id].min_resources[stolen_resource] + 1, 19
            )
            self.estimates[stealer_id].total_count += 1
        else:
            # Unknown steal (between other players)
            self.estimates[victim_id].total_count = max(
                0, self.estimates[victim_id].total_count - 1
            )
            self.estimates[stealer_id].total_count += 1
            # Reduce confidence since we don't know what was stolen
            if self.estimates[victim_id].confidence != ResourceConfidence.EXACT:
                self.estimates[victim_id].confidence = ResourceConfidence.LOW
            if self.estimates[stealer_id].confidence != ResourceConfidence.EXACT:
                self.estimates[stealer_id].confidence = ResourceConfidence.LOW

    def get_estimate(self, player_id: int) -> PlayerResourceEstimate:
        """Get the resource estimate for a player."""
        return self.estimates[player_id]

    def get_min_resources(self, player_id: int) -> Dict[Resource, int]:
        """Get the minimum known resources for a player."""
        return self.estimates[player_id].min_resources.copy()

    def get_total_count(self, player_id: int) -> int:
        """Get the total known resource count for a player."""
        return self.estimates[player_id].total_count

    def can_player_afford(self, player_id: int, building_type: str) -> bool:
        """Check if a player can definitely afford a building (conservative).

        Returns True only if min resources are sufficient.
        """
        if building_type not in BUILDING_COSTS:
            return False

        min_res = self.estimates[player_id].min_resources
        for resource, amount in BUILDING_COSTS[building_type].items():
            if min_res.get(resource, 0) < amount:
                return False
        return True

    def might_player_afford(self, player_id: int, building_type: str) -> bool:
        """Check if a player might be able to afford a building (optimistic).

        Returns True if total resources might be sufficient.
        """
        if building_type not in BUILDING_COSTS:
            return False

        total_needed = sum(BUILDING_COSTS[building_type].values())
        return self.estimates[player_id].total_count >= total_needed

    def copy(self) -> 'ResourceTracker':
        """Create a deep copy of the tracker."""
        tracker = ResourceTracker(self.agent_player_id)
        tracker.estimates = {
            pid: est.copy() for pid, est in self.estimates.items()
        }
        return tracker


def create_tracker_from_game_state(agent_player_id: int, game_state) -> ResourceTracker:
    """Create a ResourceTracker initialized from current game state.

    Args:
        agent_player_id: The ID of the agent player (has exact knowledge).
        game_state: The current GameState object.

    Returns:
        Initialized ResourceTracker.
    """
    tracker = ResourceTracker(agent_player_id)

    # Set exact knowledge for the agent
    agent = game_state.players[agent_player_id]
    tracker.update_exact(agent_player_id, agent.resources)

    # For opponents, initialize from total resource counts
    for player in game_state.players:
        if player.id != agent_player_id:
            estimate = tracker.estimates[player.id]
            estimate.total_count = player.total_resources
            estimate.confidence = ResourceConfidence.MEDIUM

    return tracker
