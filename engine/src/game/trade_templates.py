"""Trade templates for player-to-player trading in Settlers of Catan.

Defines 50 standard trade templates used for proposing and counter-offering trades.
Templates are designed to represent reasonable trade ratios commonly seen in play.
"""

from typing import Dict, List, Tuple, NamedTuple
from .constants import Resource


class TradeTemplate(NamedTuple):
    """A trade template defining what resources are offered and requested."""
    offering: Dict[Resource, int]
    requesting: Dict[Resource, int]


def generate_trade_templates() -> List[TradeTemplate]:
    """Generate the 50 standard trade templates.

    Template distribution:
    - 20 templates: 1:1 trades (5 give × 4 receive, excluding same resource)
    - 20 templates: 2:1 trades (5 give × 4 receive, excluding same resource)
    - 10 templates: 3:1 trades (subset of most useful combinations)

    Returns:
        List of 50 TradeTemplate objects.
    """
    templates = []
    tradeable = Resource.tradeable()

    # 1:1 trades (20 templates)
    # Give 1 resource, receive 1 different resource
    for give_resource in tradeable:
        for receive_resource in tradeable:
            if give_resource != receive_resource:
                templates.append(TradeTemplate(
                    offering={give_resource: 1},
                    requesting={receive_resource: 1}
                ))

    # 2:1 trades (20 templates)
    # Give 2 of one resource, receive 1 of another
    for give_resource in tradeable:
        for receive_resource in tradeable:
            if give_resource != receive_resource:
                templates.append(TradeTemplate(
                    offering={give_resource: 2},
                    requesting={receive_resource: 1}
                ))

    # 3:1 trades (10 templates)
    # Give 3 of one resource, receive 1 of another
    # Include high-value combinations (getting rare resources)
    # Plus 2 common resource trades to reach 50 templates
    valuable_receives = [Resource.ORE, Resource.GRAIN]  # Most needed for cities/dev cards
    for give_resource in tradeable:
        for receive_resource in valuable_receives:
            if give_resource != receive_resource:
                templates.append(TradeTemplate(
                    offering={give_resource: 3},
                    requesting={receive_resource: 1}
                ))

    # Add 2 more 3:1 trades to reach 50 (LUMBER->BRICK and BRICK->LUMBER)
    templates.append(TradeTemplate(
        offering={Resource.LUMBER: 3},
        requesting={Resource.BRICK: 1}
    ))
    templates.append(TradeTemplate(
        offering={Resource.BRICK: 3},
        requesting={Resource.LUMBER: 1}
    ))

    assert len(templates) == 50, f"Expected 50 templates, got {len(templates)}"
    return templates


# Pre-computed list of all trade templates
TRADE_TEMPLATES: List[TradeTemplate] = generate_trade_templates()

# Number of trade templates
NUM_TRADE_TEMPLATES = len(TRADE_TEMPLATES)


def get_template_by_index(idx: int) -> TradeTemplate:
    """Get a trade template by its index.

    Args:
        idx: Template index (0-49).

    Returns:
        The TradeTemplate at that index.

    Raises:
        IndexError: If idx is out of range.
    """
    return TRADE_TEMPLATES[idx]


def get_template_index(offering: Dict[Resource, int], requesting: Dict[Resource, int]) -> int:
    """Find the index of a trade template matching the given offer/request.

    Args:
        offering: Resources being offered.
        requesting: Resources being requested.

    Returns:
        Template index, or -1 if no matching template.
    """
    target = TradeTemplate(offering=offering, requesting=requesting)

    for idx, template in enumerate(TRADE_TEMPLATES):
        if template.offering == target.offering and template.requesting == target.requesting:
            return idx

    return -1


def is_valid_trade(
    offering: Dict[Resource, int],
    requesting: Dict[Resource, int],
    player_resources: Dict[Resource, int]
) -> bool:
    """Check if a trade is valid.

    Validates:
    1. Player has sufficient resources to offer
    2. No same resource on both sides
    3. Request value doesn't exceed offer value + 2 (max 3:1 or 4:2 ratio)

    Args:
        offering: Resources being offered.
        requesting: Resources being requested.
        player_resources: The player's current resources.

    Returns:
        True if the trade is valid, False otherwise.
    """
    # Check player has resources to offer
    for resource, amount in offering.items():
        if player_resources.get(resource, 0) < amount:
            return False

    # Check no same resource on both sides
    for resource in offering:
        if resource in requesting and offering[resource] > 0 and requesting[resource] > 0:
            return False

    # Check request value limit (max +2 from offer)
    total_offering = sum(offering.values())
    total_requesting = sum(requesting.values())

    if total_requesting > total_offering + 2:
        return False

    return True


def get_trade_description(template: TradeTemplate) -> str:
    """Get a human-readable description of a trade template.

    Args:
        template: The trade template.

    Returns:
        String description like "2 LUMBER -> 1 ORE".
    """
    offer_parts = []
    for resource, amount in template.offering.items():
        if amount > 0:
            offer_parts.append(f"{amount} {resource.name}")

    request_parts = []
    for resource, amount in template.requesting.items():
        if amount > 0:
            request_parts.append(f"{amount} {resource.name}")

    return f"{', '.join(offer_parts)} -> {', '.join(request_parts)}"
