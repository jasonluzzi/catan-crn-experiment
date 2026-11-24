"""Action representations for Settlers of Catan."""

from dataclasses import dataclass
from typing import Optional, Dict, List

from .constants import ActionType, Resource, DevCardType

# Pre-computed for O(1) lookups instead of list.index() O(n)
_TRADEABLE_RESOURCES = list(Resource.tradeable())
_RESOURCE_TO_INDEX = {r: i for i, r in enumerate(_TRADEABLE_RESOURCES)}


@dataclass
class Action:
    """Represents an action a player can take."""
    type: ActionType
    # Location-based actions
    edge_id: Optional[int] = None      # For BUILD_ROAD, PLACE_ROAD
    corner_id: Optional[int] = None    # For BUILD_SETTLEMENT, BUILD_CITY, PLACE_SETTLEMENT
    tile_id: Optional[int] = None      # For MOVE_ROBBER
    # Trade actions (bank/port)
    give_resource: Optional[Resource] = None   # For BANK_TRADE, PORT_TRADE
    receive_resource: Optional[Resource] = None
    # Dev card actions
    dev_card_type: Optional[DevCardType] = None
    # Monopoly
    monopoly_resource: Optional[Resource] = None
    # Year of Plenty
    year_of_plenty_resources: Optional[List[Resource]] = None  # 2 resources
    # Road Building
    road_edges: Optional[List[int]] = None  # 2 edge IDs
    # Steal
    steal_from_player: Optional[int] = None
    # Discard
    discard_resources: Optional[Dict[Resource, int]] = None
    # Player-to-player trading
    trade_template_idx: Optional[int] = None  # Index into TRADE_TEMPLATES (0-49)
    target_player_id: Optional[int] = None    # For CONFIRM_TRADE/DECLINE_TRADE

    def __repr__(self) -> str:
        """Human-readable representation."""
        parts = [f"Action({self.type.name}"]
        if self.edge_id is not None:
            parts.append(f"edge={self.edge_id}")
        if self.corner_id is not None:
            parts.append(f"corner={self.corner_id}")
        if self.tile_id is not None:
            parts.append(f"tile={self.tile_id}")
        if self.give_resource is not None:
            parts.append(f"give={self.give_resource.name}")
        if self.receive_resource is not None:
            parts.append(f"receive={self.receive_resource.name}")
        if self.steal_from_player is not None:
            parts.append(f"steal_from={self.steal_from_player}")
        if self.trade_template_idx is not None:
            parts.append(f"template={self.trade_template_idx}")
        if self.target_player_id is not None:
            parts.append(f"target={self.target_player_id}")
        return ", ".join(parts) + ")"


# Action space indices
# We use a flat action space with the following layout:
#
# Index Range      | Action Type
# -----------------|------------------
# 0                | ROLL_DICE
# 1                | END_TURN
# 2-73             | BUILD_ROAD (72 edges)
# 74-127           | BUILD_SETTLEMENT (54 corners)
# 128-181          | BUILD_CITY (54 corners)
# 182              | BUY_DEV_CARD
# 183              | PLAY_KNIGHT
# 184              | PLAY_ROAD_BUILDING
# 185              | PLAY_YEAR_OF_PLENTY
# 186              | PLAY_MONOPOLY
# 187-211          | BANK_TRADE (5 give × 5 receive = 25, excluding same)
# 212-230          | MOVE_ROBBER (19 tiles)
# 231-234          | STEAL (from 4 players, including current)
# 235-288          | PLACE_SETTLEMENT (54 corners, setup phase)
# 289-360          | PLACE_ROAD (72 edges, setup phase)
# 361-365          | DISCARD (5 resources: LUMBER, BRICK, WOOL, GRAIN, ORE)
# === Player-to-Player Trading (new in Phase 2) ===
# 366-415          | PROPOSE_TRADE (50 trade templates)
# 416              | ACCEPT_TRADE (accept pending offer)
# 417              | REJECT_TRADE (reject pending offer)
# 418-467          | COUNTER_OFFER (50 trade templates as counter)
# 468              | CANCEL_TRADE (cancel pending offer, proposer only)
# 469-471          | CONFIRM_TRADE_WITH_P0/P1/P2 (confirm with specific accepting player)
# 472-474          | DECLINE_TRADE_WITH_P0/P1/P2 (decline specific accepting player)

ACTION_ROLL_DICE = 0
ACTION_END_TURN = 1
ACTION_BUILD_ROAD_START = 2
ACTION_BUILD_ROAD_END = 74  # exclusive
ACTION_BUILD_SETTLEMENT_START = 74
ACTION_BUILD_SETTLEMENT_END = 128
ACTION_BUILD_CITY_START = 128
ACTION_BUILD_CITY_END = 182
ACTION_BUY_DEV_CARD = 182
ACTION_PLAY_KNIGHT = 183
ACTION_PLAY_ROAD_BUILDING = 184
ACTION_PLAY_YEAR_OF_PLENTY = 185
ACTION_PLAY_MONOPOLY = 186
ACTION_BANK_TRADE_START = 187
ACTION_BANK_TRADE_END = 212
ACTION_MOVE_ROBBER_START = 212
ACTION_MOVE_ROBBER_END = 231
ACTION_STEAL_START = 231
ACTION_STEAL_END = 235
ACTION_PLACE_SETTLEMENT_START = 235
ACTION_PLACE_SETTLEMENT_END = 289
ACTION_PLACE_ROAD_START = 289
ACTION_PLACE_ROAD_END = 361
ACTION_DISCARD_START = 361
ACTION_DISCARD_END = 366  # 5 resources

# Player-to-player trading actions
ACTION_PROPOSE_TRADE_START = 366
ACTION_PROPOSE_TRADE_END = 416       # 50 trade templates
ACTION_ACCEPT_TRADE = 416
ACTION_REJECT_TRADE = 417
ACTION_COUNTER_OFFER_START = 418
ACTION_COUNTER_OFFER_END = 468       # 50 trade templates
ACTION_CANCEL_TRADE = 468
ACTION_CONFIRM_TRADE_START = 469
ACTION_CONFIRM_TRADE_END = 472       # 3 other players (excluding self)
ACTION_DECLINE_TRADE_START = 472
ACTION_DECLINE_TRADE_END = 475       # 3 other players (excluding self)

TOTAL_ACTIONS = 475


def index_to_action(action_idx: int) -> Action:
    """Convert a flat action index to an Action object."""
    if action_idx == ACTION_ROLL_DICE:
        return Action(type=ActionType.ROLL_DICE)

    if action_idx == ACTION_END_TURN:
        return Action(type=ActionType.END_TURN)

    if ACTION_BUILD_ROAD_START <= action_idx < ACTION_BUILD_ROAD_END:
        edge_id = action_idx - ACTION_BUILD_ROAD_START
        return Action(type=ActionType.BUILD_ROAD, edge_id=edge_id)

    if ACTION_BUILD_SETTLEMENT_START <= action_idx < ACTION_BUILD_SETTLEMENT_END:
        corner_id = action_idx - ACTION_BUILD_SETTLEMENT_START
        return Action(type=ActionType.BUILD_SETTLEMENT, corner_id=corner_id)

    if ACTION_BUILD_CITY_START <= action_idx < ACTION_BUILD_CITY_END:
        corner_id = action_idx - ACTION_BUILD_CITY_START
        return Action(type=ActionType.BUILD_CITY, corner_id=corner_id)

    if action_idx == ACTION_BUY_DEV_CARD:
        return Action(type=ActionType.BUY_DEV_CARD)

    if action_idx == ACTION_PLAY_KNIGHT:
        return Action(type=ActionType.PLAY_KNIGHT, dev_card_type=DevCardType.KNIGHT)

    if action_idx == ACTION_PLAY_ROAD_BUILDING:
        return Action(type=ActionType.PLAY_ROAD_BUILDING, dev_card_type=DevCardType.ROAD_BUILDING)

    if action_idx == ACTION_PLAY_YEAR_OF_PLENTY:
        return Action(type=ActionType.PLAY_YEAR_OF_PLENTY, dev_card_type=DevCardType.YEAR_OF_PLENTY)

    if action_idx == ACTION_PLAY_MONOPOLY:
        return Action(type=ActionType.PLAY_MONOPOLY, dev_card_type=DevCardType.MONOPOLY)

    if ACTION_BANK_TRADE_START <= action_idx < ACTION_BANK_TRADE_END:
        trade_idx = action_idx - ACTION_BANK_TRADE_START
        # Map trade_idx to give/receive combination
        # We skip same-resource trades, so we have 5*4=20 valid combinations
        # But for simplicity, we use 5*5=25 and mask invalid ones
        give_idx = trade_idx // 5
        receive_idx = trade_idx % 5
        return Action(
            type=ActionType.BANK_TRADE,
            give_resource=_TRADEABLE_RESOURCES[give_idx],
            receive_resource=_TRADEABLE_RESOURCES[receive_idx]
        )

    if ACTION_MOVE_ROBBER_START <= action_idx < ACTION_MOVE_ROBBER_END:
        tile_id = action_idx - ACTION_MOVE_ROBBER_START
        return Action(type=ActionType.MOVE_ROBBER, tile_id=tile_id)

    if ACTION_STEAL_START <= action_idx < ACTION_STEAL_END:
        player_id = action_idx - ACTION_STEAL_START
        return Action(type=ActionType.STEAL, steal_from_player=player_id)

    if ACTION_PLACE_SETTLEMENT_START <= action_idx < ACTION_PLACE_SETTLEMENT_END:
        corner_id = action_idx - ACTION_PLACE_SETTLEMENT_START
        return Action(type=ActionType.PLACE_SETTLEMENT, corner_id=corner_id)

    if ACTION_PLACE_ROAD_START <= action_idx < ACTION_PLACE_ROAD_END:
        edge_id = action_idx - ACTION_PLACE_ROAD_START
        return Action(type=ActionType.PLACE_ROAD, edge_id=edge_id)

    if ACTION_DISCARD_START <= action_idx < ACTION_DISCARD_END:
        resource_idx = action_idx - ACTION_DISCARD_START
        discard_resource = _TRADEABLE_RESOURCES[resource_idx]
        return Action(
            type=ActionType.DISCARD,
            discard_resources={discard_resource: 1}
        )

    # Player-to-player trading actions
    if ACTION_PROPOSE_TRADE_START <= action_idx < ACTION_PROPOSE_TRADE_END:
        template_idx = action_idx - ACTION_PROPOSE_TRADE_START
        return Action(type=ActionType.PROPOSE_TRADE, trade_template_idx=template_idx)

    if action_idx == ACTION_ACCEPT_TRADE:
        return Action(type=ActionType.ACCEPT_TRADE)

    if action_idx == ACTION_REJECT_TRADE:
        return Action(type=ActionType.REJECT_TRADE)

    if ACTION_COUNTER_OFFER_START <= action_idx < ACTION_COUNTER_OFFER_END:
        template_idx = action_idx - ACTION_COUNTER_OFFER_START
        return Action(type=ActionType.COUNTER_OFFER, trade_template_idx=template_idx)

    if action_idx == ACTION_CANCEL_TRADE:
        return Action(type=ActionType.CANCEL_TRADE)

    if ACTION_CONFIRM_TRADE_START <= action_idx < ACTION_CONFIRM_TRADE_END:
        # Maps to players 0, 1, 2 (relative to current player)
        relative_idx = action_idx - ACTION_CONFIRM_TRADE_START
        return Action(type=ActionType.CONFIRM_TRADE, target_player_id=relative_idx)

    if ACTION_DECLINE_TRADE_START <= action_idx < ACTION_DECLINE_TRADE_END:
        relative_idx = action_idx - ACTION_DECLINE_TRADE_START
        return Action(type=ActionType.DECLINE_TRADE, target_player_id=relative_idx)

    raise ValueError(f"Invalid action index: {action_idx}")


def action_to_index(action: Action) -> int:
    """Convert an Action object to a flat action index."""
    if action.type == ActionType.ROLL_DICE:
        return ACTION_ROLL_DICE

    if action.type == ActionType.END_TURN:
        return ACTION_END_TURN

    if action.type == ActionType.BUILD_ROAD:
        return ACTION_BUILD_ROAD_START + action.edge_id

    if action.type == ActionType.BUILD_SETTLEMENT:
        return ACTION_BUILD_SETTLEMENT_START + action.corner_id

    if action.type == ActionType.BUILD_CITY:
        return ACTION_BUILD_CITY_START + action.corner_id

    if action.type == ActionType.BUY_DEV_CARD:
        return ACTION_BUY_DEV_CARD

    if action.type == ActionType.PLAY_KNIGHT:
        return ACTION_PLAY_KNIGHT

    if action.type == ActionType.PLAY_ROAD_BUILDING:
        return ACTION_PLAY_ROAD_BUILDING

    if action.type == ActionType.PLAY_YEAR_OF_PLENTY:
        return ACTION_PLAY_YEAR_OF_PLENTY

    if action.type == ActionType.PLAY_MONOPOLY:
        return ACTION_PLAY_MONOPOLY

    if action.type == ActionType.BANK_TRADE:
        give_idx = _RESOURCE_TO_INDEX[action.give_resource]
        receive_idx = _RESOURCE_TO_INDEX[action.receive_resource]
        return ACTION_BANK_TRADE_START + give_idx * 5 + receive_idx

    if action.type == ActionType.MOVE_ROBBER:
        return ACTION_MOVE_ROBBER_START + action.tile_id

    if action.type == ActionType.STEAL:
        return ACTION_STEAL_START + action.steal_from_player

    if action.type == ActionType.PLACE_SETTLEMENT:
        return ACTION_PLACE_SETTLEMENT_START + action.corner_id

    if action.type == ActionType.PLACE_ROAD:
        return ACTION_PLACE_ROAD_START + action.edge_id

    if action.type == ActionType.DISCARD:
        # Get the resource being discarded (should be exactly one with count 1)
        for res, count in action.discard_resources.items():
            if count > 0:
                return ACTION_DISCARD_START + _RESOURCE_TO_INDEX[res]

    # Player-to-player trading actions
    if action.type == ActionType.PROPOSE_TRADE:
        return ACTION_PROPOSE_TRADE_START + action.trade_template_idx

    if action.type == ActionType.ACCEPT_TRADE:
        return ACTION_ACCEPT_TRADE

    if action.type == ActionType.REJECT_TRADE:
        return ACTION_REJECT_TRADE

    if action.type == ActionType.COUNTER_OFFER:
        return ACTION_COUNTER_OFFER_START + action.trade_template_idx

    if action.type == ActionType.CANCEL_TRADE:
        return ACTION_CANCEL_TRADE

    if action.type == ActionType.CONFIRM_TRADE:
        return ACTION_CONFIRM_TRADE_START + action.target_player_id

    if action.type == ActionType.DECLINE_TRADE:
        return ACTION_DECLINE_TRADE_START + action.target_player_id

    raise ValueError(f"Cannot convert action to index: {action}")
