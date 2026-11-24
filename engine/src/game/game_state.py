"""Game state management for Settlers of Catan."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
import random

from .constants import (
    Resource, DevCardType, GamePhase, NUM_PLAYERS,
    DEV_CARD_COUNTS, VICTORY_POINTS_TO_WIN, INITIAL_BANK_RESOURCES,
    MAX_COUNTER_OFFER_DEPTH
)
from .board import Board
from .player import Player


@dataclass
class TradeOffer:
    """Represents a pending trade offer between players."""
    proposer_id: int
    offering: Dict[Resource, int]  # Resources being offered
    requesting: Dict[Resource, int]  # Resources being requested
    counter_depth: int = 0  # Number of counter-offers in chain (0-3)

    def copy(self) -> 'TradeOffer':
        """Create a deep copy of the trade offer."""
        return TradeOffer(
            proposer_id=self.proposer_id,
            offering=self.offering.copy(),
            requesting=self.requesting.copy(),
            counter_depth=self.counter_depth
        )


@dataclass
class RollHistoryEntry:
    """Records a dice roll and its effects."""
    dice_total: int
    turn_number: int
    resources_gained: Dict[int, Dict[Resource, int]]  # player_id -> resources
    robber_moved: bool = False

    def copy(self) -> 'RollHistoryEntry':
        """Create a deep copy of the roll history entry."""
        return RollHistoryEntry(
            dice_total=self.dice_total,
            turn_number=self.turn_number,
            resources_gained={
                pid: res.copy() for pid, res in self.resources_gained.items()
            },
            robber_moved=self.robber_moved
        )


@dataclass
class GameState:
    """Complete state of a Catan game."""
    board: Optional[Board] = None
    players: List[Player] = field(default_factory=list)
    current_player_idx: int = 0
    phase: GamePhase = GamePhase.SETUP_SETTLEMENT_1
    turn_number: int = 0
    dice_rolled: bool = False
    last_dice_roll: Tuple[int, int] = (0, 0)
    dev_card_deck: List[DevCardType] = field(default_factory=list)
    longest_road_holder: Optional[int] = None
    largest_army_holder: Optional[int] = None
    winner: Optional[int] = None
    setup_order: List[int] = field(default_factory=list)
    setup_phase_idx: int = 0
    pending_robber_tile: Optional[int] = None
    players_needing_discard: List[int] = field(default_factory=list)
    discard_counts: Dict[int, int] = field(default_factory=dict)  # player_id -> cards left to discard
    roller_player_idx: int = 0  # Who rolled the dice (for returning after discard)
    last_settlement_placed: Optional[int] = None  # For setup road placement

    # Player-to-player trading state
    pending_trade: Optional[TradeOffer] = None
    responders_remaining: List[int] = field(default_factory=list)  # Opponents yet to respond
    current_responder_idx: Optional[int] = None  # Current responder
    accepting_players: List[int] = field(default_factory=list)  # Players who accepted
    original_proposer_idx: Optional[int] = None  # Original proposer (for counter-offer chains)
    trade_offer_history: Set[int] = field(default_factory=set)  # Template indices used in current negotiation

    # Bank resource tracking
    bank_resources: Dict[Resource, int] = field(default_factory=dict)

    # Roll history for strategic inference
    roll_history: List[RollHistoryEntry] = field(default_factory=list)
    max_roll_history: int = 20  # Keep last 20 rolls

    # Global dev card tracking (for threat assessment)
    dev_cards_played: Dict[DevCardType, int] = field(default_factory=dict)

    # --- CRN refactor: injected randomness + fixed layout (duck-typed, no import) ---
    # `rng`: an object exposing numpy Generators as attributes .layout/.dice/
    #   .dev_deck/.steal/.setup (crn.rng.GameRNG). When None, all engine randomness
    #   falls back to the legacy global `random` module (original behavior).
    # `layout`: an optional crn.layout.Layout; when set, the board is built exactly
    #   from it (desert tile + number tokens fixed, resources as specified).
    rng: Optional[object] = None
    layout: Optional[object] = None

    def __post_init__(self):
        """Initialize the game."""
        # CRN refactor: build the board, threading either an explicit fixed layout
        # or the injected layout RNG stream. None/None reproduces original behavior.
        if self.board is None:
            if self.layout is not None:
                self.board = Board(layout=self.layout)
            elif self.rng is not None:
                self.board = Board(rng=self.rng.layout)
            else:
                self.board = Board()

        if not self.players:
            self.players = [Player(id=i) for i in range(NUM_PLAYERS)]

        if not self.dev_card_deck:
            self._init_dev_card_deck()

        if not self.setup_order:
            # Snake draft with randomized player order
            # E.g., if order is [2,0,3,1], snake is [2,0,3,1,1,3,0,2]
            player_order = list(range(NUM_PLAYERS))
            self._rng_shuffle(player_order, "setup")
            self.setup_order = player_order + player_order[::-1]
            # CRITICAL: Set current_player to first player in setup order
            self.current_player_idx = self.setup_order[0]

        if not self.bank_resources:
            self._init_bank_resources()

        if not self.dev_cards_played:
            self._init_dev_cards_played()

    def _init_bank_resources(self) -> None:
        """Initialize bank with 19 of each resource."""
        self.bank_resources = {
            r: INITIAL_BANK_RESOURCES for r in Resource.tradeable()
        }

    def _init_dev_cards_played(self) -> None:
        """Initialize dev cards played tracking."""
        self.dev_cards_played = {
            DevCardType.KNIGHT: 0,
            DevCardType.MONOPOLY: 0,
            DevCardType.ROAD_BUILDING: 0,
            DevCardType.YEAR_OF_PLENTY: 0,
            # VP cards are never played, they're revealed at end
        }

    def _rng_shuffle(self, seq: list, stream: str) -> None:
        """Shuffle `seq` in place using the named CRN stream, or global `random`."""
        if self.rng is None:
            random.shuffle(seq)
        else:
            gen = getattr(self.rng, stream)
            perm = gen.permutation(len(seq))
            seq[:] = [seq[i] for i in perm]

    def _init_dev_card_deck(self) -> None:
        """Initialize and shuffle the development card deck."""
        self.dev_card_deck = []
        for card_type, count in DEV_CARD_COUNTS.items():
            self.dev_card_deck.extend([card_type] * count)
        self._rng_shuffle(self.dev_card_deck, "dev_deck")

    @property
    def current_player(self) -> Player:
        """Get the current player."""
        return self.players[self.current_player_idx]

    @property
    def is_setup_phase(self) -> bool:
        """Check if game is in setup phase."""
        return self.phase in [
            GamePhase.SETUP_SETTLEMENT_1,
            GamePhase.SETUP_ROAD_1,
            GamePhase.SETUP_SETTLEMENT_2,
            GamePhase.SETUP_ROAD_2,
        ]

    @property
    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self.phase == GamePhase.GAME_OVER

    def get_setup_player_idx(self) -> int:
        """Get the player index for the current setup phase."""
        if self.setup_phase_idx < len(self.setup_order):
            return self.setup_order[self.setup_phase_idx]
        return 0

    def roll_dice(self) -> Tuple[int, int]:
        """Roll the dice and return the result."""
        if self.rng is None:
            die1 = random.randint(1, 6)
            die2 = random.randint(1, 6)
        else:
            # Draw both dice from a single stream; integers(1, 7) -> {1..6}.
            die1 = int(self.rng.dice.integers(1, 7))
            die2 = int(self.rng.dice.integers(1, 7))
        self.last_dice_roll = (die1, die2)
        self.dice_rolled = True
        return (die1, die2)

    def get_dice_total(self) -> int:
        """Get the total of the last dice roll."""
        return self.last_dice_roll[0] + self.last_dice_roll[1]

    def distribute_resources(self, dice_total: int) -> Dict[int, Dict[Resource, int]]:
        """Distribute resources based on dice roll with bank scarcity rules.

        Bank scarcity rule: If the bank doesn't have enough of a resource to
        fulfill all claims, that resource is not distributed to anyone
        (unless only one player is affected, in which case they get remaining).

        Returns:
            Dict mapping player_id to resources gained.
        """
        # First pass: calculate what everyone would get
        resources_needed = {i: {r: 0 for r in Resource.tradeable()} for i in range(NUM_PLAYERS)}

        # Get tiles with this number
        tiles = self.board.get_tiles_for_number(dice_total)

        for tile_id in tiles:
            tile = self.board.tiles[tile_id]

            # Skip if robber is here
            if tile.has_robber:
                continue

            # Skip desert
            if tile.resource == Resource.DESERT:
                continue

            # Find adjacent corners with buildings
            for corner in self.board.corners:
                if tile_id in corner.adjacent_tiles and corner.owner is not None:
                    amount = 2 if corner.building == 'city' else 1
                    resources_needed[corner.owner][tile.resource] += amount

        # Second pass: check bank capacity and distribute
        resources_gained = {i: {r: 0 for r in Resource.tradeable()} for i in range(NUM_PLAYERS)}

        for resource in Resource.tradeable():
            total_needed = sum(resources_needed[p][resource] for p in range(NUM_PLAYERS))

            if total_needed == 0:
                continue

            bank_available = self.bank_resources.get(resource, 0)

            if total_needed > bank_available:
                # CRN instrumentation (behavior-preserving): count a bank-scarcity
                # event whenever the 19-cap binds for a resource on this roll. Logged
                # as a post-treatment mediator; does not change any distribution.
                self.bank_scarcity_events = getattr(self, "bank_scarcity_events", 0) + 1

            if total_needed <= bank_available:
                # Bank can fulfill all claims
                for player_id in range(NUM_PLAYERS):
                    amount = resources_needed[player_id][resource]
                    if amount > 0:
                        resources_gained[player_id][resource] = amount
                        self.players[player_id].add_resource(resource, amount)
                        self.bank_resources[resource] -= amount
            else:
                # Bank scarcity: check if only one player affected
                players_needing = [p for p in range(NUM_PLAYERS) if resources_needed[p][resource] > 0]

                if len(players_needing) == 1:
                    # Single player gets remaining cards
                    player_id = players_needing[0]
                    amount = min(resources_needed[player_id][resource], bank_available)
                    if amount > 0:
                        resources_gained[player_id][resource] = amount
                        self.players[player_id].add_resource(resource, amount)
                        self.bank_resources[resource] -= amount
                # else: No one gets this resource (scarcity rule)

        # Record in roll history
        self._record_roll(dice_total, resources_gained, robber_moved=(dice_total == 7))

        return resources_gained

    def _record_roll(
        self,
        dice_total: int,
        resources_gained: Dict[int, Dict[Resource, int]],
        robber_moved: bool = False
    ) -> None:
        """Record a dice roll in the history."""
        entry = RollHistoryEntry(
            dice_total=dice_total,
            turn_number=self.turn_number,
            resources_gained=resources_gained,
            robber_moved=robber_moved
        )
        self.roll_history.append(entry)

        # Trim history to max size
        if len(self.roll_history) > self.max_roll_history:
            self.roll_history = self.roll_history[-self.max_roll_history:]

    def return_resources_to_bank(self, player_id: int, resources: Dict[Resource, int]) -> None:
        """Return resources from a player to the bank (e.g., when building)."""
        for resource, amount in resources.items():
            if amount > 0:
                self.bank_resources[resource] = self.bank_resources.get(resource, 0) + amount

    def take_resources_from_bank(self, player_id: int, resources: Dict[Resource, int]) -> bool:
        """Take resources from the bank for a player. Returns False if bank insufficient."""
        # Check bank has enough
        for resource, amount in resources.items():
            if self.bank_resources.get(resource, 0) < amount:
                return False

        # Execute transfer
        for resource, amount in resources.items():
            if amount > 0:
                self.bank_resources[resource] -= amount
                self.players[player_id].add_resource(resource, amount)

        return True

    def check_seven_rolled(self) -> List[int]:
        """Check which players must discard after a 7 is rolled.

        Returns:
            List of player IDs who must discard.
        """
        from .constants import MAX_HAND_SIZE_ON_7

        must_discard = []
        for player in self.players:
            if player.total_resources > MAX_HAND_SIZE_ON_7:
                must_discard.append(player.id)

        return must_discard

    def draw_dev_card(self) -> Optional[DevCardType]:
        """Draw a development card from the deck."""
        if not self.dev_card_deck:
            return None
        return self.dev_card_deck.pop()

    def record_dev_card_played(self, card_type: DevCardType) -> None:
        """Record that a dev card was played (for global tracking).

        Args:
            card_type: The type of card that was played.
        """
        if card_type in self.dev_cards_played:
            self.dev_cards_played[card_type] += 1

    def get_dev_cards_remaining(self) -> Dict[DevCardType, int]:
        """Get remaining dev cards by type (in deck + held by players).

        Returns:
            Dict mapping DevCardType to remaining count.
        """
        remaining = {
            DevCardType.KNIGHT: DEV_CARD_COUNTS[DevCardType.KNIGHT],
            DevCardType.MONOPOLY: DEV_CARD_COUNTS[DevCardType.MONOPOLY],
            DevCardType.ROAD_BUILDING: DEV_CARD_COUNTS[DevCardType.ROAD_BUILDING],
            DevCardType.YEAR_OF_PLENTY: DEV_CARD_COUNTS[DevCardType.YEAR_OF_PLENTY],
            DevCardType.VICTORY_POINT: DEV_CARD_COUNTS[DevCardType.VICTORY_POINT],
        }

        # Subtract played cards
        for card_type, played_count in self.dev_cards_played.items():
            if card_type in remaining:
                remaining[card_type] -= played_count

        return remaining

    def get_estimated_vp_cards_drawn(self) -> int:
        """Estimate how many VP cards have been drawn (hidden VP potential).

        VP cards = total bought - (revealed VPs + other types played)
        Since VP cards can't be played, any dev card that hasn't been
        revealed as a played knight/monopoly/etc. could be a VP card.

        Returns:
            Estimated VP cards drawn and held by players.
        """
        # Total dev cards bought = initial deck - remaining in deck
        initial_deck_size = sum(DEV_CARD_COUNTS.values())  # 25
        cards_in_deck = len(self.dev_card_deck)
        total_drawn = initial_deck_size - cards_in_deck

        # Total played (revealed) cards
        total_played = sum(self.dev_cards_played.values())

        # Cards held by players
        cards_held = total_drawn - total_played

        # Some held cards are known (revealed VP cards for winner)
        revealed_vps = sum(p.victory_points_from_dev_cards for p in self.players)

        # Remaining held cards could be VP cards or unplayed knights/etc.
        # Best estimate: at most 5 VP cards exist, so cap at that
        max_vp_possible = DEV_CARD_COUNTS[DevCardType.VICTORY_POINT] - revealed_vps
        estimated_hidden_vps = min(cards_held - total_played, max_vp_possible)

        return max(0, estimated_hidden_vps)

    def update_longest_road(self) -> Optional[int]:
        """Update longest road holder.

        Returns:
            Player ID of new holder, or None if no change.
        """
        min_length = 5  # Minimum to claim

        # Calculate lengths for all players
        for player in self.players:
            player.longest_road_length = self.board.calculate_longest_road(player.id)

        # Find current longest
        current_longest = 0
        current_holder = None

        for player in self.players:
            if player.longest_road_length >= min_length:
                if player.longest_road_length > current_longest:
                    current_longest = player.longest_road_length
                    current_holder = player.id

        # Update holder
        old_holder = self.longest_road_holder

        if current_holder is not None:
            if old_holder is None:
                # New holder
                self.longest_road_holder = current_holder
                self.players[current_holder].has_longest_road = True
            elif current_holder != old_holder:
                # Only change if strictly longer
                old_length = self.players[old_holder].longest_road_length
                if current_longest > old_length:
                    self.players[old_holder].has_longest_road = False
                    self.longest_road_holder = current_holder
                    self.players[current_holder].has_longest_road = True

        return self.longest_road_holder if self.longest_road_holder != old_holder else None

    def update_largest_army(self) -> Optional[int]:
        """Update largest army holder.

        Returns:
            Player ID of new holder, or None if no change.
        """
        min_knights = 3  # Minimum to claim
        old_holder = self.largest_army_holder

        # Count to beat: the CURRENT HOLDER's played knights (or min_knights-1 if unheld).
        # A challenger takes the badge only by playing STRICTLY MORE knights than the
        # current holder; a tie keeps it with the holder. Comparing against the holder's
        # count (rather than a scan-max reset to min-1) fixes a seat-order bug where a tie
        # transferred the badge to the lowest seat index, stealing it from a higher-seat holder.
        threshold = (self.players[old_holder].knights_played
                     if old_holder is not None else min_knights - 1)

        new_holder = None
        best = threshold
        for player in self.players:
            if player.knights_played >= min_knights and player.knights_played > best:
                best = player.knights_played  # among challengers who beat the holder, take the most
                new_holder = player.id

        if new_holder is not None and new_holder != old_holder:
            if old_holder is not None:
                self.players[old_holder].has_largest_army = False
            self.largest_army_holder = new_holder
            self.players[new_holder].has_largest_army = True
            return new_holder

        return None

    def check_winner(self) -> Optional[int]:
        """Check if the current player has won.

        In Catan, you can only win on YOUR turn. This prevents bias from
        checking players in fixed order [0,1,2,3] when multiple have 10+ VP.

        Returns:
            Player ID of winner (current player), or None if no winner yet.
        """
        current_player = self.players[self.current_player_idx]
        if current_player.total_victory_points >= VICTORY_POINTS_TO_WIN:
            self.winner = current_player.id
            self.phase = GamePhase.GAME_OVER
            return current_player.id
        return None

    def advance_setup_phase(self) -> None:
        """Advance to the next step in setup phase.

        This is called after placing a road. setup_phase_idx has already been
        incremented once for the settlement and will be incremented again here
        for the road.

        Setup phase order (snake draft):
        - Round 1: Players 0,1,2,3 each place settlement then road
        - Round 2: Players 3,2,1,0 each place settlement then road

        setup_phase_idx tracks completed actions (0-15 for 4 players).
        After this increment, setup_phase_idx will be even (pointing to next settlement).
        """
        self.setup_phase_idx += 1

        # Check if setup is complete (4 players × 4 actions = 16)
        total_steps = NUM_PLAYERS * 4
        if self.setup_phase_idx >= total_steps:
            # Setup complete, start normal play
            self.phase = GamePhase.ROLL_DICE
            # Main game turn order starts with first player from setup_order
            # and cycles clockwise (0→1→2→3 wrapping)
            self.current_player_idx = self.setup_order[0]
            self.turn_number = 1
            return

        # setup_phase_idx now points to the NEXT action
        # Even indices = settlement, Odd indices = road
        action_type = self.setup_phase_idx % 2  # 0=settlement, 1=road

        # Which player's turn (based on setup_order)
        # Each pair of actions (settlement+road) belongs to one player
        pair_idx = self.setup_phase_idx // 2
        self.current_player_idx = self.setup_order[pair_idx]

        # Determine round (first or second settlement/road for each player)
        round_num = pair_idx // NUM_PLAYERS  # 0 or 1

        if action_type == 0:
            # Settlement phase
            self.phase = GamePhase.SETUP_SETTLEMENT_1 if round_num == 0 else GamePhase.SETUP_SETTLEMENT_2
        else:
            # Road phase
            self.phase = GamePhase.SETUP_ROAD_1 if round_num == 0 else GamePhase.SETUP_ROAD_2

    def advance_discard_phase(self) -> None:
        """Advance discard phase after a player has discarded.

        Called after auto-discarding for a player during the DISCARD phase.
        Moves to the next player who needs to discard, or transitions to
        MOVE_ROBBER phase when all players have discarded.
        """
        if self.phase != GamePhase.DISCARD:
            return

        # Check if any player still needs to discard
        # Move to next player and check
        start_idx = self.current_player_idx
        for i in range(NUM_PLAYERS):
            next_idx = (start_idx + 1 + i) % NUM_PLAYERS
            if self.players[next_idx].total_resources > 7:
                self.current_player_idx = next_idx
                return

        # No one needs to discard anymore, move to robber phase
        # The player who rolled the 7 moves the robber
        self.current_player_idx = self.roller_player_idx if hasattr(self, 'roller_player_idx') else 0
        self.phase = GamePhase.MOVE_ROBBER

    def next_turn(self) -> None:
        """Advance to the next player's turn."""
        # Check for winner BEFORE advancing (you win on your turn)
        if self.check_winner():
            return  # Game over, don't advance

        self.current_player.end_turn()
        # BUG FIX: Use randomized turn order from setup, not fixed [0,1,2,3]
        # Turn order is first 4 elements of setup_order (before snake reversal)
        turn_order = self.setup_order[:NUM_PLAYERS]
        current_pos = turn_order.index(self.current_player_idx)
        next_pos = (current_pos + 1) % NUM_PLAYERS
        self.current_player_idx = turn_order[next_pos]
        self.turn_number += 1
        self.dice_rolled = False
        self.phase = GamePhase.ROLL_DICE

    def grant_initial_resources(self, player_id: int, corner_id: int) -> None:
        """Grant resources for second settlement placement."""
        player = self.players[player_id]
        corner = self.board.corners[corner_id]

        for tile_id in corner.adjacent_tiles:
            if tile_id < len(self.board.tiles):
                tile = self.board.tiles[tile_id]
                if tile.resource != Resource.DESERT:
                    player.add_resource(tile.resource)

    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        new_state = GameState.__new__(GameState)
        # CRN refactor: carry rng/layout by REFERENCE. The CRN experiment runner does
        # not use copy() during play (only MCTS determinization does, which is out of
        # scope); sharing the Generator by reference means draws from the copy would
        # advance the original's stream. Documented in LIMITATIONS.md.
        new_state.rng = self.rng
        new_state.layout = self.layout
        new_state.board = self.board.copy()
        new_state.players = [p.copy() for p in self.players]
        new_state.current_player_idx = self.current_player_idx
        new_state.phase = self.phase
        new_state.turn_number = self.turn_number
        new_state.dice_rolled = self.dice_rolled
        new_state.last_dice_roll = self.last_dice_roll
        new_state.dev_card_deck = self.dev_card_deck.copy()
        new_state.longest_road_holder = self.longest_road_holder
        new_state.largest_army_holder = self.largest_army_holder
        new_state.winner = self.winner
        new_state.setup_order = self.setup_order.copy()
        new_state.setup_phase_idx = self.setup_phase_idx
        new_state.pending_robber_tile = self.pending_robber_tile
        new_state.players_needing_discard = self.players_needing_discard.copy()
        new_state.discard_counts = self.discard_counts.copy()
        new_state.roller_player_idx = self.roller_player_idx
        new_state.last_settlement_placed = self.last_settlement_placed
        # Trade state
        new_state.pending_trade = self.pending_trade.copy() if self.pending_trade else None
        new_state.responders_remaining = self.responders_remaining.copy()
        new_state.current_responder_idx = self.current_responder_idx
        new_state.accepting_players = self.accepting_players.copy()
        new_state.original_proposer_idx = self.original_proposer_idx
        new_state.trade_offer_history = self.trade_offer_history.copy()
        # Bank and history
        new_state.bank_resources = self.bank_resources.copy()
        new_state.roll_history = [r.copy() for r in self.roll_history]
        new_state.max_roll_history = self.max_roll_history
        # Dev card tracking
        new_state.dev_cards_played = self.dev_cards_played.copy()
        return new_state
