"""Rules engine for Settlers of Catan - legal move generation and execution.

OPTIMIZATION NOTES:
- Pre-compute owned edges/corners once per get_legal_actions call
- Pass pre-computed sets to validation functions
- Avoid redundant method calls in hot paths
"""

from typing import List, Set, Optional
import numpy as np

from .constants import (
    GamePhase, ActionType, Resource, DevCardType,
    NUM_TILES, NUM_CORNERS, NUM_EDGES, NUM_PLAYERS,
    MAX_COUNTER_OFFER_DEPTH, BUILDING_COSTS
)
from .board import Board
from .player import Player
from .game_state import GameState, TradeOffer
from .actions import (
    Action, TOTAL_ACTIONS, index_to_action,
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
)
from .trade_templates import TRADE_TEMPLATES, is_valid_trade

# Pre-compute tradeable resources list to avoid repeated creation
_TRADEABLE_RESOURCES = list(Resource.tradeable())


class Rules:
    """Rules engine for Catan game logic."""

    @staticmethod
    def get_legal_actions(state: GameState) -> np.ndarray:
        """Get a mask of legal actions for the current player.

        Returns:
            Boolean array of shape (TOTAL_ACTIONS,) where True = legal.
        """
        mask = np.zeros(TOTAL_ACTIONS, dtype=bool)
        player = state.current_player
        board = state.board

        # Setup phases
        if state.phase == GamePhase.SETUP_SETTLEMENT_1 or state.phase == GamePhase.SETUP_SETTLEMENT_2:
            Rules._add_setup_settlement_actions(mask, state, player, board)
            return mask

        if state.phase == GamePhase.SETUP_ROAD_1 or state.phase == GamePhase.SETUP_ROAD_2:
            Rules._add_setup_road_actions(mask, state, player, board)
            return mask

        # Discard phase - player must discard resources one at a time
        if state.phase == GamePhase.DISCARD:
            Rules._add_discard_actions(mask, player)
            return mask

        # Move robber phase
        if state.phase == GamePhase.MOVE_ROBBER:
            Rules._add_move_robber_actions(mask, state, board)
            return mask

        # Steal phase
        if state.phase == GamePhase.STEAL:
            Rules._add_steal_actions(mask, state, board)
            return mask

        # Roll dice phase
        if state.phase == GamePhase.ROLL_DICE:
            mask[ACTION_ROLL_DICE] = True
            # Can play knight before rolling
            if player.can_play_dev_card(DevCardType.KNIGHT):
                mask[ACTION_PLAY_KNIGHT] = True
            return mask

        # Main phase
        if state.phase == GamePhase.MAIN:
            # Can always end turn
            mask[ACTION_END_TURN] = True

            # OPTIMIZATION: Pre-compute owned sets once for all build checks
            owned_edges = player.get_owned_edges()
            owned_corners = player.get_owned_corners()

            # Building actions (pass pre-computed sets)
            Rules._add_build_actions(mask, state, player, board, owned_edges, owned_corners)

            # Trading actions (bank/port)
            Rules._add_trade_actions(mask, player)

            # Player-to-player trade proposals
            Rules._add_propose_trade_actions(mask, state, player)

            # Development card actions
            Rules._add_dev_card_actions(mask, player)

        # Trade response phase (opponent responding to offer)
        if state.phase == GamePhase.TRADE_RESPONSE:
            Rules._add_trade_response_actions(mask, state, player)
            return mask

        # Trade confirm phase (proposer confirms with accepting player)
        if state.phase == GamePhase.TRADE_CONFIRM:
            Rules._add_trade_confirm_actions(mask, state, player)
            return mask

        return mask

    @staticmethod
    def _add_setup_settlement_actions(mask: np.ndarray, state: GameState,
                                       player: Player, board: Board) -> None:
        """Add legal settlement placement actions for setup phase."""
        for corner_id in range(NUM_CORNERS):
            if board.is_valid_settlement_location(corner_id, player.id, is_setup=True):
                mask[ACTION_PLACE_SETTLEMENT_START + corner_id] = True

    @staticmethod
    def _add_setup_road_actions(mask: np.ndarray, state: GameState,
                                 player: Player, board: Board) -> None:
        """Add legal road placement actions for setup phase."""
        settlement_corner = state.last_settlement_placed
        if settlement_corner is None:
            return

        for edge_id in range(NUM_EDGES):
            if board.is_valid_road_location(
                edge_id, player.id,
                is_setup=True,
                setup_settlement=settlement_corner
            ):
                mask[ACTION_PLACE_ROAD_START + edge_id] = True

    @staticmethod
    def _add_move_robber_actions(mask: np.ndarray, state: GameState, board: Board) -> None:
        """Add legal robber movement actions."""
        current_robber_tile = board.get_robber_tile()
        for tile_id in range(NUM_TILES):
            # Cannot stay on same tile
            if tile_id != current_robber_tile:
                mask[ACTION_MOVE_ROBBER_START + tile_id] = True

    @staticmethod
    def _add_steal_actions(mask: np.ndarray, state: GameState, board: Board) -> None:
        """Add legal steal actions."""
        robber_tile = state.pending_robber_tile
        if robber_tile is None:
            robber_tile = board.get_robber_tile()

        # Find players adjacent to robber tile with resources
        adjacent_players = board.get_adjacent_players(robber_tile)
        current_player = state.current_player_idx

        for player_id in adjacent_players:
            if player_id != current_player:
                if state.players[player_id].total_resources > 0:
                    mask[ACTION_STEAL_START + player_id] = True

        # If no one to steal from, allow "steal from self" as a no-op
        if not mask[ACTION_STEAL_START:ACTION_STEAL_END].any():
            mask[ACTION_STEAL_START + current_player] = True

    @staticmethod
    def _add_build_actions(mask: np.ndarray, state: GameState,
                           player: Player, board: Board,
                           owned_edges: Set[int] = None,
                           owned_corners: Set[int] = None) -> None:
        """Add legal building actions.

        OPTIMIZATION: Accepts pre-computed owned sets to avoid repeated computation.
        """
        # OPTIMIZATION: Use pre-computed sets if provided
        if owned_edges is None:
            owned_edges = player.get_owned_edges()
        if owned_corners is None:
            owned_corners = player.get_owned_corners()

        # Roads
        if player.can_build_road():
            for edge_id in range(NUM_EDGES):
                if board.is_valid_road_location(
                    edge_id, player.id,
                    owned_edges=owned_edges,
                    owned_corners=owned_corners
                ):
                    mask[ACTION_BUILD_ROAD_START + edge_id] = True

        # Settlements
        if player.can_build_settlement():
            for corner_id in range(NUM_CORNERS):
                if board.is_valid_settlement_location(
                    corner_id, player.id,
                    is_setup=False,
                    owned_edges=owned_edges
                ):
                    mask[ACTION_BUILD_SETTLEMENT_START + corner_id] = True

        # Cities
        if player.can_build_city():
            for corner_id in player.settlements:
                mask[ACTION_BUILD_CITY_START + corner_id] = True

        # Development cards
        if player.can_buy_dev_card() and len(state.dev_card_deck) > 0:
            mask[ACTION_BUY_DEV_CARD] = True

    @staticmethod
    def _add_trade_actions(mask: np.ndarray, player: Player) -> None:
        """Add legal trading actions.

        OPTIMIZATION: Uses pre-computed resource list and caches trade ratios.
        """
        # OPTIMIZATION: Pre-compute trade ratios for all resources once
        trade_ratios = {}
        for res in _TRADEABLE_RESOURCES:
            trade_ratios[res] = player.get_trade_ratio(res)

        for i, give_res in enumerate(_TRADEABLE_RESOURCES):
            # Check if player has enough of this resource to trade
            ratio = trade_ratios[give_res]
            if player.resources.get(give_res, 0) >= ratio:
                for j, receive_res in enumerate(_TRADEABLE_RESOURCES):
                    if give_res != receive_res:
                        action_idx = ACTION_BANK_TRADE_START + i * 5 + j
                        mask[action_idx] = True

    @staticmethod
    def _add_dev_card_actions(mask: np.ndarray, player: Player) -> None:
        """Add legal development card play actions."""
        if player.can_play_dev_card(DevCardType.KNIGHT):
            mask[ACTION_PLAY_KNIGHT] = True
        if player.can_play_dev_card(DevCardType.ROAD_BUILDING):
            mask[ACTION_PLAY_ROAD_BUILDING] = True
        if player.can_play_dev_card(DevCardType.YEAR_OF_PLENTY):
            mask[ACTION_PLAY_YEAR_OF_PLENTY] = True
        if player.can_play_dev_card(DevCardType.MONOPOLY):
            mask[ACTION_PLAY_MONOPOLY] = True

    @staticmethod
    def _add_discard_actions(mask: np.ndarray, player: Player) -> None:
        """Add legal discard actions based on resources the player has."""
        for i, res in enumerate(_TRADEABLE_RESOURCES):
            if player.resources.get(res, 0) > 0:
                mask[ACTION_DISCARD_START + i] = True

    @staticmethod
    def _add_propose_trade_actions(mask: np.ndarray, state: GameState, player: Player) -> None:
        """Add legal trade proposal actions (using trade templates)."""
        for i, template in enumerate(TRADE_TEMPLATES):
            if is_valid_trade(template.offering, template.requesting, player.resources):
                mask[ACTION_PROPOSE_TRADE_START + i] = True

    @staticmethod
    def _add_trade_response_actions(mask: np.ndarray, state: GameState, player: Player) -> None:
        """Add legal trade response actions (accept/reject/counter).

        Called when an opponent is responding to a trade offer.
        """
        if state.pending_trade is None:
            return

        # Can always reject
        mask[ACTION_REJECT_TRADE] = True

        # Check if player can accept (has resources requested by proposer)
        offer = state.pending_trade
        # In response phase, the offer's "offering" is what proposer gives,
        # and "requesting" is what the proposer wants (what responder must have)
        can_accept = True
        for resource, amount in offer.requesting.items():
            if player.resources.get(resource, 0) < amount:
                can_accept = False
                break

        if can_accept:
            mask[ACTION_ACCEPT_TRADE] = True

        # Can counter-offer if depth limit not reached
        if offer.counter_depth < MAX_COUNTER_OFFER_DEPTH:
            for i, template in enumerate(TRADE_TEMPLATES):
                # Skip templates already used in this negotiation
                if i in state.trade_offer_history:
                    continue
                # For counter-offer, player offers template.offering and requests template.requesting
                if is_valid_trade(template.offering, template.requesting, player.resources):
                    mask[ACTION_COUNTER_OFFER_START + i] = True

    @staticmethod
    def _add_trade_confirm_actions(mask: np.ndarray, state: GameState, player: Player) -> None:
        """Add legal trade confirm/decline actions.

        Called when the proposer is confirming which accepting player to trade with.
        """
        if state.pending_trade is None or not state.accepting_players:
            # No one accepted, should have returned to MAIN
            return

        # Can cancel the trade entirely
        mask[ACTION_CANCEL_TRADE] = True

        # For each accepting player, can confirm or decline
        for accepting_player_id in state.accepting_players:
            # Map accepting player ID to relative index (0, 1, or 2)
            # The relative index maps to the 3 "other" players from current player's perspective
            relative_idx = Rules._player_id_to_relative_idx(state.current_player_idx, accepting_player_id)
            if relative_idx is not None:
                mask[ACTION_CONFIRM_TRADE_START + relative_idx] = True
                mask[ACTION_DECLINE_TRADE_START + relative_idx] = True

    @staticmethod
    def _player_id_to_relative_idx(current_player_id: int, target_player_id: int) -> Optional[int]:
        """Convert an absolute player ID to a relative index (0, 1, 2) for the 3 other players."""
        if target_player_id == current_player_id:
            return None

        # Build list of other players in order
        other_players = [(current_player_id + 1 + i) % NUM_PLAYERS for i in range(3)]
        try:
            return other_players.index(target_player_id)
        except ValueError:
            return None

    @staticmethod
    def _relative_idx_to_player_id(current_player_id: int, relative_idx: int) -> int:
        """Convert a relative index (0, 1, 2) back to an absolute player ID."""
        return (current_player_id + 1 + relative_idx) % NUM_PLAYERS

    @staticmethod
    def execute_action(state: GameState, action_idx: int) -> GameState:
        """Execute an action and return the new game state.

        Args:
            state: Current game state (will be modified in place).
            action_idx: Index of the action to execute.

        Returns:
            The modified game state.
        """
        action = index_to_action(action_idx)
        player = state.current_player
        board = state.board

        if action.type == ActionType.ROLL_DICE:
            Rules._execute_roll_dice(state)

        elif action.type == ActionType.END_TURN:
            state.next_turn()

        elif action.type == ActionType.PLACE_SETTLEMENT:
            Rules._execute_place_settlement(state, action, player, board)

        elif action.type == ActionType.PLACE_ROAD:
            Rules._execute_place_road(state, action, player, board)

        elif action.type == ActionType.BUILD_ROAD:
            Rules._execute_build_road(state, action, player, board)

        elif action.type == ActionType.BUILD_SETTLEMENT:
            Rules._execute_build_settlement(state, action, player, board)

        elif action.type == ActionType.BUILD_CITY:
            Rules._execute_build_city(state, action, player, board)

        elif action.type == ActionType.BUY_DEV_CARD:
            Rules._execute_buy_dev_card(state, player)

        elif action.type == ActionType.PLAY_KNIGHT:
            Rules._execute_play_knight(state, player)

        elif action.type == ActionType.PLAY_ROAD_BUILDING:
            Rules._execute_play_road_building(state, player)

        elif action.type == ActionType.PLAY_YEAR_OF_PLENTY:
            Rules._execute_year_of_plenty(state, player)

        elif action.type == ActionType.PLAY_MONOPOLY:
            Rules._execute_monopoly(state, player)

        elif action.type == ActionType.BANK_TRADE:
            Rules._execute_bank_trade(state, action, player)

        elif action.type == ActionType.MOVE_ROBBER:
            Rules._execute_move_robber(state, action, board)

        elif action.type == ActionType.STEAL:
            Rules._execute_steal(state, action, player)

        elif action.type == ActionType.DISCARD:
            Rules._execute_discard(state, action, player)

        # Player-to-player trading actions
        elif action.type == ActionType.PROPOSE_TRADE:
            Rules._execute_propose_trade(state, action, player)

        elif action.type == ActionType.ACCEPT_TRADE:
            Rules._execute_accept_trade(state, player)

        elif action.type == ActionType.REJECT_TRADE:
            Rules._execute_reject_trade(state, player)

        elif action.type == ActionType.COUNTER_OFFER:
            Rules._execute_counter_offer(state, action, player)

        elif action.type == ActionType.CANCEL_TRADE:
            Rules._execute_cancel_trade(state)

        elif action.type == ActionType.CONFIRM_TRADE:
            Rules._execute_confirm_trade(state, action, player)

        elif action.type == ActionType.DECLINE_TRADE:
            Rules._execute_decline_trade(state, action)

        return state

    @staticmethod
    def _execute_roll_dice(state: GameState) -> None:
        """Execute dice roll action."""
        die1, die2 = state.roll_dice()
        total = die1 + die2

        # Store who rolled (for returning to after discard phase)
        state.roller_player_idx = state.current_player_idx

        if total == 7:
            # Check for discards
            must_discard = state.check_seven_rolled()
            if must_discard:
                # Initialize discard tracking
                state.players_needing_discard = list(must_discard)
                state.discard_counts = {}
                for player_id in must_discard:
                    player = state.players[player_id]
                    state.discard_counts[player_id] = player.total_resources // 2

                # Enter DISCARD phase, starting with first player who needs to discard
                state.phase = GamePhase.DISCARD
                state.current_player_idx = must_discard[0]
            else:
                # No one needs to discard, move robber
                state.phase = GamePhase.MOVE_ROBBER
        else:
            # Distribute resources
            state.distribute_resources(total)
            state.phase = GamePhase.MAIN

    @staticmethod
    def _execute_discard(state: GameState, action: Action, player: Player) -> None:
        """Execute a discard action - remove one resource from player's hand."""
        # Get the resource to discard
        for res, count in action.discard_resources.items():
            if count > 0 and player.resources.get(res, 0) > 0:
                player.resources[res] -= 1
                # Return discarded resource to bank
                state.bank_resources[res] = state.bank_resources.get(res, 0) + 1
                break

        # Decrement remaining discard count for this player
        player_id = player.id
        if player_id in state.discard_counts:
            state.discard_counts[player_id] -= 1

            # Check if this player is done discarding
            if state.discard_counts[player_id] <= 0:
                # Remove from list of players needing to discard
                if player_id in state.players_needing_discard:
                    state.players_needing_discard.remove(player_id)

                # Check if all players are done discarding
                if len(state.players_needing_discard) == 0:
                    # All done, move to robber phase
                    state.phase = GamePhase.MOVE_ROBBER
                    state.current_player_idx = state.roller_player_idx
                    state.discard_counts = {}
                else:
                    # Move to next player who needs to discard
                    state.current_player_idx = state.players_needing_discard[0]

    @staticmethod
    def _execute_place_settlement(state: GameState, action: Action,
                                   player: Player, board: Board) -> None:
        """Execute settlement placement in setup phase."""
        corner_id = action.corner_id
        board.place_settlement(corner_id, player.id)
        player.settlements.append(corner_id)

        # Update player's ports
        corner = board.corners[corner_id]
        if corner.port is not None and corner.port not in player.ports:
            player.ports.append(corner.port)

        # Grant initial resources for second settlement
        if state.phase == GamePhase.SETUP_SETTLEMENT_2:
            state.grant_initial_resources(player.id, corner_id)

        state.last_settlement_placed = corner_id

        # Increment setup phase index and set road phase
        state.setup_phase_idx += 1
        if state.phase == GamePhase.SETUP_SETTLEMENT_1:
            state.phase = GamePhase.SETUP_ROAD_1
        elif state.phase == GamePhase.SETUP_SETTLEMENT_2:
            state.phase = GamePhase.SETUP_ROAD_2

    @staticmethod
    def _execute_place_road(state: GameState, action: Action,
                             player: Player, board: Board) -> None:
        """Execute road placement in setup phase."""
        edge_id = action.edge_id
        board.place_road(edge_id, player.id)
        player.roads.append(edge_id)

        state.last_settlement_placed = None
        state.advance_setup_phase()

    @staticmethod
    def _execute_build_road(state: GameState, action: Action,
                             player: Player, board: Board) -> None:
        """Execute road building action."""
        edge_id = action.edge_id
        if player.pay_for('road'):
            # Return spent resources to bank
            state.return_resources_to_bank(player.id, BUILDING_COSTS['road'])
            board.place_road(edge_id, player.id)
            player.roads.append(edge_id)
            state.update_longest_road()
            state.check_winner()

    @staticmethod
    def _execute_build_settlement(state: GameState, action: Action,
                                   player: Player, board: Board) -> None:
        """Execute settlement building action."""
        corner_id = action.corner_id
        if player.pay_for('settlement'):
            # Return spent resources to bank
            state.return_resources_to_bank(player.id, BUILDING_COSTS['settlement'])
            board.place_settlement(corner_id, player.id)
            player.settlements.append(corner_id)

            # Update player's ports
            corner = board.corners[corner_id]
            if corner.port is not None and corner.port not in player.ports:
                player.ports.append(corner.port)

            # Settlement can break opponent's road chain
            state.update_longest_road()
            state.check_winner()

    @staticmethod
    def _execute_build_city(state: GameState, action: Action,
                             player: Player, board: Board) -> None:
        """Execute city building action."""
        corner_id = action.corner_id
        if player.pay_for('city'):
            # Return spent resources to bank
            state.return_resources_to_bank(player.id, BUILDING_COSTS['city'])
            board.place_city(corner_id, player.id)
            player.settlements.remove(corner_id)
            player.cities.append(corner_id)
            state.check_winner()

    @staticmethod
    def _execute_buy_dev_card(state: GameState, player: Player) -> None:
        """Execute development card purchase."""
        if player.pay_for('dev_card'):
            # Return spent resources to bank
            state.return_resources_to_bank(player.id, BUILDING_COSTS['dev_card'])
            card = state.draw_dev_card()
            if card is not None:
                player.add_dev_card(card, bought_this_turn=True, turn_number=state.turn_number)
                if card == DevCardType.VICTORY_POINT:
                    state.check_winner()

    @staticmethod
    def _execute_play_knight(state: GameState, player: Player) -> None:
        """Execute knight card play."""
        if player.play_dev_card(DevCardType.KNIGHT):
            state.record_dev_card_played(DevCardType.KNIGHT)

            # Track if player is responding to being blocked (robber on their tile)
            robber_tile = state.board.get_robber_tile()
            player_on_robber_tile = player.id in state.board.get_adjacent_players(robber_tile)
            if player_on_robber_tile and player.times_blocked_with_cards > player.times_played_knight_when_blocked:
                player.times_played_knight_when_blocked += 1

            state.update_largest_army()
            # Check for winner (Largest Army could give winning VP)
            if state.check_winner() is None:
                state.phase = GamePhase.MOVE_ROBBER

    @staticmethod
    def _execute_play_road_building(state: GameState, player: Player) -> None:
        """Execute road building card.

        For simplicity, this grants resources to build 2 roads.
        A full implementation would have a multi-step action.
        """
        if player.play_dev_card(DevCardType.ROAD_BUILDING):
            state.record_dev_card_played(DevCardType.ROAD_BUILDING)
            # Grant resources for 2 roads
            player.add_resource(Resource.LUMBER, 2)
            player.add_resource(Resource.BRICK, 2)

    @staticmethod
    def _execute_year_of_plenty(state: GameState, player: Player) -> None:
        """Execute Year of Plenty card.

        For simplicity, gives 1 of each of 2 most common resources.
        A full implementation would let player choose.
        """
        if player.play_dev_card(DevCardType.YEAR_OF_PLENTY):
            state.record_dev_card_played(DevCardType.YEAR_OF_PLENTY)
            # Give 2 resources from bank (check availability)
            resources_to_give = [Resource.LUMBER, Resource.BRICK]
            for resource in resources_to_give:
                if state.bank_resources.get(resource, 0) > 0:
                    state.bank_resources[resource] -= 1
                    player.add_resource(resource)

    @staticmethod
    def _execute_monopoly(state: GameState, player: Player) -> None:
        """Execute Monopoly card.

        For simplicity, takes the most abundant resource.
        A full implementation would let player choose.
        """
        if player.play_dev_card(DevCardType.MONOPOLY):
            state.record_dev_card_played(DevCardType.MONOPOLY)
            # Find most abundant resource across other players
            best_resource = Resource.LUMBER
            best_count = 0
            for res in Resource.tradeable():
                count = sum(p.resources[res] for p in state.players if p.id != player.id)
                if count > best_count:
                    best_count = count
                    best_resource = res

            # Take all of that resource
            for p in state.players:
                if p.id != player.id:
                    amount = p.resources[best_resource]
                    p.resources[best_resource] = 0
                    player.add_resource(best_resource, amount)

    @staticmethod
    def _execute_bank_trade(state: GameState, action: Action,
                             player: Player) -> None:
        """Execute bank trade action."""
        give_resource = action.give_resource
        receive_resource = action.receive_resource
        ratio = player.get_trade_ratio(give_resource)

        if player.do_bank_trade(give_resource, receive_resource):
            # Update bank: receive resources player gave, give 1 resource to player
            state.bank_resources[give_resource] = state.bank_resources.get(give_resource, 0) + ratio
            state.bank_resources[receive_resource] = state.bank_resources.get(receive_resource, 0) - 1

    @staticmethod
    def _execute_move_robber(state: GameState, action: Action,
                              board: Board) -> None:
        """Execute robber movement."""
        tile_id = action.tile_id
        board.move_robber(tile_id)
        state.pending_robber_tile = tile_id

        # Check if there's anyone to steal from
        adjacent_players = board.get_adjacent_players(tile_id)
        can_steal = False
        for pid in adjacent_players:
            if pid != state.current_player_idx:
                if state.players[pid].total_resources > 0:
                    can_steal = True
                    break

        if can_steal:
            state.phase = GamePhase.STEAL
        else:
            state.pending_robber_tile = None
            state.phase = GamePhase.MAIN

    @staticmethod
    def _execute_steal(state: GameState, action: Action,
                        player: Player) -> None:
        """Execute steal action."""
        victim_id = action.steal_from_player

        if victim_id != player.id:
            victim = state.players[victim_id]
            if victim.total_resources > 0:
                # Track that victim was robbed while holding dev cards
                # (for knight play ratio inference)
                if victim.total_dev_cards > 0:
                    victim.times_blocked_with_cards += 1

                # Steal random resource.
                # `available` is a multiset of the victim's cards, ordered by
                # Resource.tradeable(). CRN refactor: map a single uniform from the
                # steal stream to a card index, so the draw is coupled across paired
                # games wherever the victim's hand composition matches (it decouples
                # once hands diverge -- noted in LIMITATIONS.md).
                available = []
                for res in Resource.tradeable():
                    available.extend([res] * victim.resources[res])
                if available:
                    rng = getattr(state, "rng", None)
                    if rng is None:
                        import random
                        stolen = random.choice(available)
                    else:
                        u = float(rng.steal.random())
                        idx = min(int(u * len(available)), len(available) - 1)
                        stolen = available[idx]
                    victim.remove_resource(stolen)
                    player.add_resource(stolen)

        state.pending_robber_tile = None
        state.phase = GamePhase.MAIN

    # === Player-to-Player Trading Execution ===

    @staticmethod
    def _execute_propose_trade(state: GameState, action: Action, player: Player) -> None:
        """Execute trade proposal - broadcast offer to all opponents."""
        template_idx = action.trade_template_idx
        template = TRADE_TEMPLATES[template_idx]

        # Create the pending trade offer
        state.pending_trade = TradeOffer(
            proposer_id=player.id,
            offering=template.offering.copy(),
            requesting=template.requesting.copy(),
            counter_depth=0
        )

        # Clear and initialize trade history for this negotiation
        state.trade_offer_history.clear()
        state.trade_offer_history.add(template_idx)

        # Set up responders (all other players in order)
        state.responders_remaining = [
            (player.id + 1 + i) % NUM_PLAYERS
            for i in range(NUM_PLAYERS - 1)
        ]
        state.accepting_players = []
        state.original_proposer_idx = player.id

        # Move to first responder
        state.current_responder_idx = state.responders_remaining[0]
        state.current_player_idx = state.current_responder_idx
        state.phase = GamePhase.TRADE_RESPONSE

    @staticmethod
    def _execute_accept_trade(state: GameState, player: Player) -> None:
        """Execute trade acceptance - add player to accepting list."""
        if state.pending_trade is None:
            return

        state.accepting_players.append(player.id)
        Rules._advance_trade_response(state)

    @staticmethod
    def _execute_reject_trade(state: GameState, player: Player) -> None:
        """Execute trade rejection - move to next responder."""
        Rules._advance_trade_response(state)

    @staticmethod
    def _execute_counter_offer(state: GameState, action: Action, player: Player) -> None:
        """Execute counter-offer - swap proposer/responder roles."""
        if state.pending_trade is None:
            return

        template_idx = action.trade_template_idx
        template = TRADE_TEMPLATES[template_idx]

        # Add this template to history (prevents re-using same offer)
        state.trade_offer_history.add(template_idx)

        # Create new trade with swapped roles and incremented depth
        old_proposer = state.pending_trade.proposer_id
        new_depth = state.pending_trade.counter_depth + 1

        state.pending_trade = TradeOffer(
            proposer_id=player.id,
            offering=template.offering.copy(),
            requesting=template.requesting.copy(),
            counter_depth=new_depth
        )

        # The old proposer is now the only responder for the counter-offer
        state.responders_remaining = [old_proposer]
        state.accepting_players = []
        state.current_responder_idx = old_proposer
        state.current_player_idx = old_proposer
        state.phase = GamePhase.TRADE_RESPONSE

    @staticmethod
    def _execute_cancel_trade(state: GameState) -> None:
        """Execute trade cancellation - return to MAIN phase."""
        # Save proposer BEFORE clearing (clear sets it to None)
        proposer = state.original_proposer_idx
        Rules._clear_trade_state(state)
        state.current_player_idx = proposer if proposer is not None else 0
        state.phase = GamePhase.MAIN

    @staticmethod
    def _execute_confirm_trade(state: GameState, action: Action, player: Player) -> None:
        """Execute trade confirmation - complete trade with chosen player."""
        if state.pending_trade is None:
            return

        # Get the actual player ID from relative index
        target_relative_idx = action.target_player_id
        target_player_id = Rules._relative_idx_to_player_id(player.id, target_relative_idx)

        # Verify target is in accepting players
        if target_player_id not in state.accepting_players:
            return

        offer = state.pending_trade
        proposer = state.players[offer.proposer_id]
        acceptor = state.players[target_player_id]

        # Execute the trade
        # Proposer gives their offering to acceptor
        for resource, amount in offer.offering.items():
            proposer.remove_resource(resource, amount)
            acceptor.add_resource(resource, amount)

        # Acceptor gives the requested resources to proposer
        for resource, amount in offer.requesting.items():
            acceptor.remove_resource(resource, amount)
            proposer.add_resource(resource, amount)

        # Clear trade state and return to MAIN
        # Save proposer BEFORE clearing (clear sets it to None)
        proposer = state.original_proposer_idx
        Rules._clear_trade_state(state)
        state.current_player_idx = proposer if proposer is not None else 0
        state.phase = GamePhase.MAIN

    @staticmethod
    def _execute_decline_trade(state: GameState, action: Action) -> None:
        """Execute trade decline - reject a specific accepting player."""
        if state.pending_trade is None or not state.accepting_players:
            return

        # Get the actual player ID from relative index
        current_player = state.current_player_idx
        target_relative_idx = action.target_player_id
        target_player_id = Rules._relative_idx_to_player_id(current_player, target_relative_idx)

        # Remove from accepting players
        if target_player_id in state.accepting_players:
            state.accepting_players.remove(target_player_id)

        # If no more accepting players, return to MAIN
        if not state.accepting_players:
            # Save proposer BEFORE clearing (clear sets it to None)
            proposer = state.original_proposer_idx
            Rules._clear_trade_state(state)
            state.current_player_idx = proposer if proposer is not None else 0
            state.phase = GamePhase.MAIN

    @staticmethod
    def _advance_trade_response(state: GameState) -> None:
        """Advance to next responder or move to confirmation phase."""
        if not state.responders_remaining:
            # All responded, check for acceptors
            if state.accepting_players:
                # Move to confirm phase with proposer
                state.current_player_idx = state.pending_trade.proposer_id
                state.phase = GamePhase.TRADE_CONFIRM
            else:
                # No one accepted, return to MAIN
                # Save proposer BEFORE clearing (clear sets it to None)
                proposer = state.original_proposer_idx
                Rules._clear_trade_state(state)
                state.current_player_idx = proposer if proposer is not None else 0
                state.phase = GamePhase.MAIN
            return

        # Remove current responder from remaining
        if state.current_responder_idx in state.responders_remaining:
            state.responders_remaining.remove(state.current_responder_idx)

        if state.responders_remaining:
            # Move to next responder
            state.current_responder_idx = state.responders_remaining[0]
            state.current_player_idx = state.current_responder_idx
        else:
            # All responded
            if state.accepting_players:
                state.current_player_idx = state.pending_trade.proposer_id
                state.phase = GamePhase.TRADE_CONFIRM
            else:
                # Save proposer BEFORE clearing (clear sets it to None)
                proposer = state.original_proposer_idx
                Rules._clear_trade_state(state)
                state.current_player_idx = proposer if proposer is not None else 0
                state.phase = GamePhase.MAIN

    @staticmethod
    def _clear_trade_state(state: GameState) -> None:
        """Clear all trade-related state."""
        state.pending_trade = None
        state.responders_remaining = []
        state.current_responder_idx = None
        state.accepting_players = []
        state.original_proposer_idx = None
        state.trade_offer_history.clear()
