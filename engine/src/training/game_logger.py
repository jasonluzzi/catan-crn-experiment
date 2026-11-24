"""Game logging for replay and animation."""

import fcntl
import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from enum import Enum

from ..game.game_state import GameState
from ..game.board import Board
from ..game.constants import Resource, GamePhase, ActionType


@dataclass
class ActionRecord:
    """Record of a single action taken in the game."""
    turn: int
    player: int
    phase: str
    action_type: str
    action_index: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp_ms: int = 0


@dataclass
class VPBreakdown:
    """Victory point breakdown for a player."""
    player_id: int
    settlements: int = 0      # 1 VP each
    cities: int = 0           # 2 VP each
    longest_road: bool = False  # 2 VP if true
    largest_army: bool = False  # 2 VP if true
    vp_cards: int = 0         # 1 VP each
    total: int = 0


@dataclass
class TurnSnapshot:
    """PHASE 3: Full-information snapshot of game state at end of turn.

    Used for analytics mode - contains all visible AND hidden information
    for post-training analysis without leaking to the training loop.
    """
    turn: int
    active_player: int
    dice_roll: Optional[tuple] = None
    robber_tile: int = 0

    # Per-player full state (indexed 0-3)
    player_resources: List[Dict[str, int]] = field(default_factory=list)
    player_dev_cards: List[Dict[str, int]] = field(default_factory=list)
    player_vp: List[int] = field(default_factory=list)
    player_hidden_vp: List[int] = field(default_factory=list)  # VP cards (hidden)
    player_settlements: List[List[int]] = field(default_factory=list)
    player_cities: List[List[int]] = field(default_factory=list)
    player_roads: List[List[int]] = field(default_factory=list)
    player_road_lengths: List[int] = field(default_factory=list)
    player_knights_played: List[int] = field(default_factory=list)

    # Special achievements
    longest_road_holder: Optional[int] = None
    largest_army_holder: Optional[int] = None

    # Bank state
    bank_resources: Dict[str, int] = field(default_factory=dict)

    # Actions taken this turn (for detailed analysis)
    actions_this_turn: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GameRecord:
    """Complete record of a game for replay."""
    game_id: str
    game_number: int
    timestamp: str

    # Board setup (immutable after game start)
    board_setup: Dict[str, Any] = field(default_factory=dict)

    # All actions taken (or filtered subset)
    actions: List[ActionRecord] = field(default_factory=list)

    # Game outcome
    winner: Optional[int] = None
    final_scores: List[int] = field(default_factory=list)
    total_turns: int = 0
    final_robber_tile: int = 0

    # VP breakdown per player
    vp_breakdown: List[VPBreakdown] = field(default_factory=list)

    # PHASE 3: Per-turn snapshots for analytics mode
    turn_snapshots: List[TurnSnapshot] = field(default_factory=list)

    # Metadata
    agent_player_id: int = 0
    opponent_type: str = "random"
    opponent_infos: List[Dict] = field(default_factory=list)  # Per-opponent details
    training_step: int = 0


class LogLevel(Enum):
    """What level of detail to log."""
    NONE = 0           # Don't log anything
    OUTCOME_ONLY = 1   # Just winner and final scores
    KEY_ACTIONS = 2    # Setup + buildings + dev cards
    FULL = 3           # Every action
    ANALYTICS = 4      # Full information per-turn snapshots for analytics


@dataclass
class GameLoggerConfig:
    """Configuration for game logging."""

    # Sampling - only log every N games
    log_every_n_games: int = 1

    # What to log
    log_level: LogLevel = LogLevel.KEY_ACTIONS

    # Always log these regardless of log_level
    always_log_setup: bool = True           # Initial placements
    always_log_settlements: bool = True     # Settlement builds
    always_log_cities: bool = True          # City upgrades
    always_log_dev_cards: bool = True       # Dev card buys and plays
    always_log_robber: bool = True          # Robber moves
    always_log_trades: bool = True          # Bank and port trades

    # Storage settings
    save_dir: str = "./game_logs"
    save_format: str = "jsonl"              # "jsonl" or "json"
    max_games_per_file: int = 1000          # For jsonl, create new file after N games
    shared_file_path: Optional[str] = None  # If set, all loggers write to this file

    # Include board state snapshots (larger files but easier replay)
    include_state_snapshots: bool = False
    snapshot_every_n_actions: int = 10

    @classmethod
    def for_analytics(
        cls,
        save_dir: str = "./game_logs/analytics",
        log_every_n_games: int = 10,
    ) -> 'GameLoggerConfig':
        """PHASE 3: Create config for full-information analytics logging.

        Logs complete game state at each turn for post-training analysis.
        Includes all hidden information (opponent resources, dev cards).

        Args:
            save_dir: Directory to save analytics logs.
            log_every_n_games: Only log every N games to manage file size.

        Returns:
            GameLoggerConfig configured for analytics mode.
        """
        return cls(
            log_every_n_games=log_every_n_games,
            log_level=LogLevel.ANALYTICS,
            always_log_setup=True,
            always_log_settlements=True,
            always_log_cities=True,
            always_log_dev_cards=True,
            always_log_robber=True,
            always_log_trades=True,
            save_dir=save_dir,
            save_format="jsonl",
            max_games_per_file=100,  # Smaller files since analytics data is large
            include_state_snapshots=False,  # Turn snapshots replace these
        )


class GameLogger:
    """Logs games for replay and analysis."""

    # Action types that are always considered "key actions"
    KEY_ACTION_TYPES = {
        ActionType.PLACE_SETTLEMENT,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_CITY,
        ActionType.BUILD_ROAD,
        ActionType.BUY_DEV_CARD,
        ActionType.PLAY_KNIGHT,
        ActionType.PLAY_ROAD_BUILDING,
        ActionType.PLAY_YEAR_OF_PLENTY,
        ActionType.PLAY_MONOPOLY,
        ActionType.MOVE_ROBBER,
        ActionType.BANK_TRADE,
        ActionType.PORT_TRADE,
    }

    def __init__(self, config: Optional[GameLoggerConfig] = None):
        """Initialize the game logger.

        Args:
            config: Logging configuration. Uses defaults if None.
        """
        self.config = config or GameLoggerConfig()
        self.game_count = 0
        self.current_game: Optional[GameRecord] = None
        self.current_file_game_count = 0
        self.current_file_path: Optional[str] = None

        # Create save directory
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Action counter for current game (for snapshots)
        self._action_count = 0

    def start_game(
        self,
        game_state: GameState,
        agent_player_id: int = 0,
        opponent_type: str = "random",
        opponent_infos: List[Dict] = None,
        training_step: int = 0
    ) -> bool:
        """Start logging a new game.

        Args:
            game_state: Initial game state.
            agent_player_id: Which player is the RL agent.
            opponent_type: Type of opponents being used.
            opponent_infos: Per-opponent details (type, model_path, elo).
            training_step: Current training step.

        Returns:
            True if this game will be logged, False if skipped.
        """
        self.game_count += 1
        self._action_count = 0

        # Check if we should log this game
        if self.config.log_level == LogLevel.NONE:
            self.current_game = None
            return False

        if self.config.log_every_n_games > 1:
            if self.game_count % self.config.log_every_n_games != 0:
                self.current_game = None
                return False

        # Create new game record
        self.current_game = GameRecord(
            game_id=str(uuid.uuid4())[:8],
            game_number=self.game_count,
            timestamp=datetime.now().isoformat(),
            agent_player_id=agent_player_id,
            opponent_type=opponent_type,
            opponent_infos=opponent_infos or [],
            training_step=training_step,
        )

        # Record board setup
        self.current_game.board_setup = self._encode_board_setup(game_state.board)

        return True

    def log_action(
        self,
        game_state: GameState,
        action_index: int,
        action_type: ActionType,
        player_id: int,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an action taken in the game.

        Args:
            game_state: Current game state.
            action_index: The action index that was taken.
            action_type: Type of action.
            player_id: Player who took the action.
            details: Additional action details (corner_id, edge_id, etc.)
        """
        if self.current_game is None:
            return

        self._action_count += 1

        # Determine if we should log this action
        should_log = self._should_log_action(game_state, action_type)

        if not should_log:
            return

        # Create action record
        record = ActionRecord(
            turn=game_state.turn_number,
            player=player_id,
            phase=game_state.phase.name,
            action_type=action_type.name,
            action_index=action_index,
            details=details or {},
            timestamp_ms=self._action_count,  # Relative ordering
        )

        # Add state snapshot if configured
        if (self.config.include_state_snapshots and
            self._action_count % self.config.snapshot_every_n_actions == 0):
            record.details["state_snapshot"] = self._encode_state_snapshot(game_state)

        self.current_game.actions.append(record)

    def log_setup_placement(
        self,
        game_state: GameState,
        player_id: int,
        placement_type: str,  # "settlement" or "road"
        location: int,        # corner_id or edge_id
        is_second_placement: bool = False
    ) -> None:
        """Log a setup phase placement (convenience method).

        Args:
            game_state: Current game state.
            player_id: Player who placed.
            placement_type: "settlement" or "road".
            location: Corner ID or Edge ID.
            is_second_placement: Whether this is the second settlement/road.
        """
        if self.current_game is None:
            return

        if not self.config.always_log_setup and self.config.log_level != LogLevel.FULL:
            return

        action_type = (ActionType.PLACE_SETTLEMENT if placement_type == "settlement"
                       else ActionType.BUILD_ROAD)

        details = {
            "location": location,
            "placement_round": 2 if is_second_placement else 1,
        }

        # For second settlement, record initial resources granted
        if placement_type == "settlement" and is_second_placement:
            corner = game_state.board.corners[location]
            resources_granted = []
            for tile_id in corner.adjacent_tiles:
                if tile_id < len(game_state.board.tiles):
                    tile = game_state.board.tiles[tile_id]
                    if tile.resource != Resource.DESERT:
                        resources_granted.append(tile.resource.name)
            details["initial_resources"] = resources_granted

        record = ActionRecord(
            turn=0,
            player=player_id,
            phase=game_state.phase.name,
            action_type=f"SETUP_{placement_type.upper()}",
            action_index=-1,  # Setup actions don't use standard action space
            details=details,
            timestamp_ms=self._action_count,
        )

        self._action_count += 1
        self.current_game.actions.append(record)

    def log_dice_roll(
        self,
        game_state: GameState,
        player_id: int,
        dice_result: tuple
    ) -> None:
        """Log a dice roll (convenience method).

        Args:
            game_state: Current game state.
            player_id: Player who rolled.
            dice_result: Tuple of (die1, die2).
        """
        if self.current_game is None:
            return

        # Only log dice rolls in FULL/ANALYTICS mode or if it's a 7
        if self.config.log_level not in {LogLevel.FULL, LogLevel.ANALYTICS} and sum(dice_result) != 7:
            return

        record = ActionRecord(
            turn=game_state.turn_number,
            player=player_id,
            phase=game_state.phase.name,
            action_type="ROLL_DICE",
            action_index=0,
            details={
                "dice": list(dice_result),
                "total": sum(dice_result),
            },
            timestamp_ms=self._action_count,
        )

        self._action_count += 1
        self.current_game.actions.append(record)

    def log_turn_end(
        self,
        game_state: GameState,
        dice_result: Optional[tuple] = None,
        actions_this_turn: Optional[List[Dict]] = None
    ) -> None:
        """PHASE 3: Log full turn snapshot for analytics mode.

        Captures all visible AND hidden information at end of turn.
        This is only logged in ANALYTICS mode for post-training analysis.

        Args:
            game_state: Game state at end of turn.
            dice_result: Dice roll for this turn (if any).
            actions_this_turn: List of actions taken this turn.
        """
        if self.current_game is None:
            return

        if self.config.log_level != LogLevel.ANALYTICS:
            return

        from ..game.constants import DevCardType

        snapshot = TurnSnapshot(
            turn=game_state.turn_number,
            active_player=game_state.current_player_idx,
            dice_roll=dice_result,
            robber_tile=game_state.board.get_robber_tile(),
            longest_road_holder=game_state.longest_road_holder,
            largest_army_holder=game_state.largest_army_holder,
            actions_this_turn=actions_this_turn or [],
        )

        # Full player state - ALL information including hidden
        for player in game_state.players:
            # Resources (would be hidden in real game for opponents)
            snapshot.player_resources.append(
                {r.name: c for r, c in player.resources.items()}
            )

            # Dev cards (hidden in real game)
            snapshot.player_dev_cards.append(
                {c.name: n for c, n in player.dev_cards.items()}
            )

            # VP breakdown
            snapshot.player_vp.append(player.victory_points)
            snapshot.player_hidden_vp.append(
                player.dev_cards.get(DevCardType.VICTORY_POINT, 0)
            )

            # Buildings and roads
            snapshot.player_settlements.append(list(player.settlements))
            snapshot.player_cities.append(list(player.cities))
            snapshot.player_roads.append(list(player.roads))
            snapshot.player_road_lengths.append(player.longest_road_length)
            snapshot.player_knights_played.append(player.knights_played)

        # Bank resources
        snapshot.bank_resources = {
            r.name: c for r, c in game_state.bank_resources.items()
        }

        self.current_game.turn_snapshots.append(snapshot)

    def end_game(
        self,
        game_state: GameState,
        winner: Optional[int] = None
    ) -> Optional[GameRecord]:
        """End logging for the current game.

        Args:
            game_state: Final game state.
            winner: Winner player ID (or None if no winner).

        Returns:
            The completed GameRecord, or None if game wasn't logged.
        """
        if self.current_game is None:
            return None

        # Record final state
        self.current_game.winner = winner
        self.current_game.final_scores = [
            p.total_victory_points for p in game_state.players
        ]
        self.current_game.total_turns = game_state.turn_number

        # Record final robber location
        self.current_game.final_robber_tile = game_state.board.get_robber_tile()

        # Record VP breakdown for each player
        from ..game.constants import DevCardType
        self.current_game.vp_breakdown = []
        for player in game_state.players:
            breakdown = VPBreakdown(
                player_id=player.id,
                settlements=len(player.settlements),
                cities=len(player.cities),
                longest_road=player.has_longest_road,
                largest_army=player.has_largest_army,
                vp_cards=player.dev_cards.get(DevCardType.VICTORY_POINT, 0),
                total=player.total_victory_points,
            )
            self.current_game.vp_breakdown.append(breakdown)

        # Save the game
        self._save_game(self.current_game)

        completed_game = self.current_game
        self.current_game = None

        return completed_game

    def _should_log_action(self, game_state: GameState, action_type: ActionType) -> bool:
        """Determine if an action should be logged based on config."""

        # Always log in FULL or ANALYTICS mode
        if self.config.log_level in {LogLevel.FULL, LogLevel.ANALYTICS}:
            return True

        # Check "always log" settings
        if self.config.always_log_setup and game_state.is_setup_phase:
            return True

        if self.config.always_log_settlements and action_type in {
            ActionType.PLACE_SETTLEMENT, ActionType.BUILD_SETTLEMENT
        }:
            return True

        if self.config.always_log_cities and action_type == ActionType.BUILD_CITY:
            return True

        if self.config.always_log_dev_cards and action_type in {
            ActionType.BUY_DEV_CARD,
            ActionType.PLAY_KNIGHT,
            ActionType.PLAY_ROAD_BUILDING,
            ActionType.PLAY_YEAR_OF_PLENTY,
            ActionType.PLAY_MONOPOLY,
        }:
            return True

        if self.config.always_log_robber and action_type == ActionType.MOVE_ROBBER:
            return True

        if self.config.always_log_trades and action_type in {
            ActionType.BANK_TRADE, ActionType.PORT_TRADE
        }:
            return True

        # In KEY_ACTIONS mode, log key action types
        if self.config.log_level == LogLevel.KEY_ACTIONS:
            return action_type in self.KEY_ACTION_TYPES

        return False

    def _encode_board_setup(self, board: Board) -> Dict[str, Any]:
        """Encode the board setup for replay.

        Includes:
        - Tile layout (resource types and number tokens)
        - Port locations (which corners have which port types)
        - Initial robber position (desert tile)
        """
        # Group ports by type, collecting corner pairs
        ports_by_type: Dict[str, List[int]] = {}
        for c in board.corners:
            if c.port is not None:
                port_name = c.port.name
                if port_name not in ports_by_type:
                    ports_by_type[port_name] = []
                ports_by_type[port_name].append(c.id)

        # Format ports as list with corner pairs
        ports_list = [
            {
                "port_type": port_type,
                "corners": sorted(corners),
            }
            for port_type, corners in ports_by_type.items()
        ]

        return {
            "tiles": [
                {
                    "id": t.id,
                    "resource": t.resource.name,
                    "number": t.number,
                    "has_robber": t.has_robber,
                }
                for t in board.tiles
            ],
            "ports": ports_list,
            # Also include flat list for easy lookup
            "port_corners": {
                c.id: c.port.name
                for c in board.corners if c.port is not None
            },
            "robber_start_tile": board.get_robber_tile(),
        }

    def _encode_state_snapshot(self, game_state: GameState) -> Dict[str, Any]:
        """Encode a snapshot of the current game state."""
        return {
            "turn": game_state.turn_number,
            "phase": game_state.phase.name,
            "current_player": game_state.current_player_idx,
            "robber_tile": game_state.board.get_robber_tile(),
            "players": [
                {
                    "id": p.id,
                    "vp": p.victory_points,
                    "total_vp": p.total_victory_points,
                    "settlements": list(p.settlements),
                    "cities": list(p.cities),
                    "roads": list(p.roads),
                    "resources": {r.name: c for r, c in p.resources.items() if c > 0},
                    "dev_cards": {c.name: n for c, n in p.dev_cards.items() if n > 0},
                    "knights_played": p.knights_played,
                }
                for p in game_state.players
            ],
            "longest_road_holder": game_state.longest_road_holder,
            "largest_army_holder": game_state.largest_army_holder,
        }

    def _save_game(self, game: GameRecord) -> None:
        """Save a completed game to disk."""
        if self.config.log_level == LogLevel.NONE:
            return

        # Convert to dict
        game_dict = {
            "game_id": game.game_id,
            "game_number": game.game_number,
            "timestamp": game.timestamp,
            "board_setup": game.board_setup,
            "actions": [asdict(a) for a in game.actions],
            "winner": game.winner,
            "final_scores": game.final_scores,
            "total_turns": game.total_turns,
            "final_robber_tile": game.final_robber_tile,
            "vp_breakdown": [asdict(vp) for vp in game.vp_breakdown],
            "agent_player_id": game.agent_player_id,
            "opponent_type": game.opponent_type,
            "opponent_infos": game.opponent_infos,
            "training_step": game.training_step,
            # PHASE 3: Include turn snapshots for analytics mode
            "turn_snapshots": [asdict(ts) for ts in game.turn_snapshots] if game.turn_snapshots else [],
        }

        # Convert numpy types to Python native types for JSON serialization
        game_dict = self._convert_to_native_types(game_dict)

        if self.config.save_format == "jsonl":
            self._save_jsonl(game_dict)
        else:
            self._save_json(game_dict)

    def _convert_to_native_types(self, obj):
        """Recursively convert numpy types to Python native types."""
        import numpy as np

        if isinstance(obj, dict):
            return {k: self._convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def _save_jsonl(self, game_dict: Dict) -> None:
        """Save game to JSONL file (one game per line).

        If shared_file_path is set, all loggers write to that file with
        file locking to handle concurrent writes from SubprocVecEnv.
        Otherwise, creates timestamped files per logger instance.
        """
        # Use shared file path if configured (for SubprocVecEnv compatibility)
        if self.config.shared_file_path:
            file_path = self.config.shared_file_path
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Append with file locking for concurrent writes
            with open(file_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
                try:
                    f.write(json.dumps(game_dict) + "\n")
                    f.flush()  # Ensure data is written before releasing lock
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
            return

        # Standard per-instance file handling
        # Create new file if needed
        if (self.current_file_path is None or
            self.current_file_game_count >= self.config.max_games_per_file):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_file_path = os.path.join(
                self.config.save_dir,
                f"games_{timestamp}.jsonl"
            )
            self.current_file_game_count = 0

        # Append to file
        with open(self.current_file_path, "a") as f:
            f.write(json.dumps(game_dict) + "\n")

        self.current_file_game_count += 1

    def _save_json(self, game_dict: Dict) -> None:
        """Save game to individual JSON file."""
        file_path = os.path.join(
            self.config.save_dir,
            f"game_{game_dict['game_id']}.json"
        )
        with open(file_path, "w") as f:
            json.dump(game_dict, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "total_games_seen": self.game_count,
            "games_logged": self.current_file_game_count,
            "log_every_n": self.config.log_every_n_games,
            "log_level": self.config.log_level.name,
            "current_file": self.current_file_path,
        }


class GameReplayer:
    """Replay logged games for visualization or analysis."""

    def __init__(self):
        """Initialize the replayer."""
        pass

    def load_game(self, file_path: str, game_id: Optional[str] = None) -> GameRecord:
        """Load a game from file.

        Args:
            file_path: Path to the game file.
            game_id: Specific game ID to load (for JSONL files).

        Returns:
            The loaded GameRecord.
        """
        if file_path.endswith(".jsonl"):
            return self._load_from_jsonl(file_path, game_id)
        else:
            return self._load_from_json(file_path)

    def _load_from_jsonl(self, file_path: str, game_id: Optional[str] = None) -> GameRecord:
        """Load a game from JSONL file."""
        with open(file_path, "r") as f:
            for line in f:
                game_dict = json.loads(line.strip())
                if game_id is None or game_dict["game_id"] == game_id:
                    return self._dict_to_record(game_dict)
        raise ValueError(f"Game {game_id} not found in {file_path}")

    def _load_from_json(self, file_path: str) -> GameRecord:
        """Load a game from JSON file."""
        with open(file_path, "r") as f:
            game_dict = json.load(f)
        return self._dict_to_record(game_dict)

    def _dict_to_record(self, game_dict: Dict) -> GameRecord:
        """Convert a dictionary to GameRecord."""
        actions = [
            ActionRecord(**a) for a in game_dict.get("actions", [])
        ]
        vp_breakdown = [
            VPBreakdown(**vp) for vp in game_dict.get("vp_breakdown", [])
        ]
        return GameRecord(
            game_id=game_dict["game_id"],
            game_number=game_dict["game_number"],
            timestamp=game_dict["timestamp"],
            board_setup=game_dict["board_setup"],
            actions=actions,
            winner=game_dict.get("winner"),
            final_scores=game_dict.get("final_scores", []),
            total_turns=game_dict.get("total_turns", 0),
            final_robber_tile=game_dict.get("final_robber_tile", 0),
            vp_breakdown=vp_breakdown,
            agent_player_id=game_dict.get("agent_player_id", 0),
            opponent_type=game_dict.get("opponent_type", "unknown"),
            opponent_infos=game_dict.get("opponent_infos", []),
            training_step=game_dict.get("training_step", 0),
        )

    def list_games(self, file_path: str) -> List[Dict[str, Any]]:
        """List all games in a JSONL file.

        Args:
            file_path: Path to JSONL file.

        Returns:
            List of game summaries.
        """
        games = []
        with open(file_path, "r") as f:
            for line in f:
                game_dict = json.loads(line.strip())
                games.append({
                    "game_id": game_dict["game_id"],
                    "game_number": game_dict["game_number"],
                    "winner": game_dict.get("winner"),
                    "final_scores": game_dict.get("final_scores"),
                    "total_turns": game_dict.get("total_turns"),
                    "num_actions": len(game_dict.get("actions", [])),
                })
        return games

    def reconstruct_game_state(
        self,
        game: GameRecord,
        up_to_action: int = -1
    ) -> GameState:
        """Reconstruct a GameState from a game record.

        Args:
            game: The game record to reconstruct.
            up_to_action: Reconstruct up to this action index (-1 = all).

        Returns:
            The reconstructed GameState.
        """
        from ..game.game_state import GameState
        from ..game.rules import Rules

        # Create initial state with the recorded board
        state = GameState()

        # Set up the board with recorded tile configuration
        for i, tile_data in enumerate(game.board_setup["tiles"]):
            if i < len(state.board.tiles):
                state.board.tiles[i].resource = Resource[tile_data["resource"]]
                state.board.tiles[i].number = tile_data["number"]
                state.board.tiles[i].has_robber = tile_data["has_robber"]

        # Replay actions
        actions_to_replay = game.actions if up_to_action < 0 else game.actions[:up_to_action]

        for action_record in actions_to_replay:
            if action_record.action_index >= 0:
                # Standard action - execute it
                Rules.execute_action(state, action_record.action_index)

        return state

    def get_action_frames(self, game: GameRecord) -> List[Dict[str, Any]]:
        """Get frames for animation, one per logged action.

        Args:
            game: The game record.

        Returns:
            List of frame data for animation.
        """
        frames = []

        for i, action in enumerate(game.actions):
            frame = {
                "frame_number": i,
                "turn": action.turn,
                "player": action.player,
                "action_type": action.action_type,
                "phase": action.phase,
                "details": action.details,
            }

            # Include state snapshot if available
            if "state_snapshot" in action.details:
                frame["state"] = action.details["state_snapshot"]

            frames.append(frame)

        return frames
