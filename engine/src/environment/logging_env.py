"""Logging wrapper for Catan environment."""

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import gymnasium as gym

from .catan_env import CatanEnv, CatanMultiAgentEnv
from .reward import RewardConfig
from ..game.actions import index_to_action
from ..game.rules import Rules
from ..game.constants import ActionType, GamePhase, DevCardType, Resource, PortType
from ..training.game_logger import GameLogger, GameLoggerConfig, LogLevel


class LoggingCatanEnv(CatanEnv):
    """Catan environment with game logging for replay.

    This wrapper logs game actions for later replay and animation.
    It integrates seamlessly with the training pipeline.
    """

    def __init__(
        self,
        opponent_policies: Optional[List[Any]] = None,
        reward_config: Optional[RewardConfig] = None,
        max_turns: int = 500,
        render_mode: Optional[str] = None,
        agent_position: Optional[int] = None,
        opponent_sampler: Optional[callable] = None,
        # Logging parameters
        game_logger: Optional[GameLogger] = None,
        log_every_n_games: int = 1,
        log_level: str = "key_actions",
        save_dir: str = "./game_logs",
        shared_file_path: Optional[str] = None,
        always_log_setup: bool = True,
        always_log_settlements: bool = True,
        always_log_cities: bool = True,
        always_log_dev_cards: bool = True,
        always_log_robber: bool = True,
        include_state_snapshots: bool = False,
        opponent_type: str = "random",
    ):
        """Initialize the logging environment.

        Args:
            opponent_policies: List of opponent policies.
            reward_config: Reward configuration.
            max_turns: Maximum turns per episode.
            render_mode: Rendering mode.
            agent_position: Fixed position for agent (0-3), or None for random.
            opponent_sampler: Optional callable for sampling opponents on each reset.
            game_logger: Existing GameLogger to use (if None, creates new one).
            log_every_n_games: Only log every N games.
            log_level: Detail level - "none", "outcome_only", "key_actions", "full".
            save_dir: Directory to save game logs.
            shared_file_path: If set, all loggers write to this file with locking.
            always_log_setup: Always log initial placements.
            always_log_settlements: Always log settlement builds.
            always_log_cities: Always log city upgrades.
            always_log_dev_cards: Always log dev card purchases/plays.
            always_log_robber: Always log robber moves.
            include_state_snapshots: Include periodic state snapshots.
            opponent_type: Type of opponents (for metadata).
        """
        super().__init__(
            opponent_policies=opponent_policies,
            reward_config=reward_config,
            max_turns=max_turns,
            render_mode=render_mode,
            agent_position=agent_position,
            opponent_sampler=opponent_sampler,
        )

        self.opponent_type = opponent_type

        # Create or use provided logger
        if game_logger is not None:
            self.game_logger = game_logger
        else:
            level_map = {
                "none": LogLevel.NONE,
                "outcome_only": LogLevel.OUTCOME_ONLY,
                "key_actions": LogLevel.KEY_ACTIONS,
                "full": LogLevel.FULL,
            }
            config = GameLoggerConfig(
                log_every_n_games=log_every_n_games,
                log_level=level_map.get(log_level.lower(), LogLevel.KEY_ACTIONS),
                always_log_setup=always_log_setup,
                always_log_settlements=always_log_settlements,
                always_log_cities=always_log_cities,
                always_log_dev_cards=always_log_dev_cards,
                always_log_robber=always_log_robber,
                save_dir=save_dir,
                shared_file_path=shared_file_path,
                include_state_snapshots=include_state_snapshots,
            )
            self.game_logger = GameLogger(config)

        # Track training step for metadata
        self.training_step = 0
        self._logging_this_game = False

    def set_training_step(self, step: int) -> None:
        """Update the current training step for metadata."""
        self.training_step = step

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and start logging a new game.

        This overrides the parent reset() to ensure opponent actions during
        setup are logged (they happen before the agent's first turn).
        """
        # Call gymnasium base reset (not CatanEnv.reset)
        gym.Env.reset(self, seed=seed)

        # Sample new opponents if sampler provided (for self-play)
        if self.opponent_sampler is not None:
            self.opponent_policies, self.opponent_infos = self.opponent_sampler()
            # Update opponent_type for logging based on sampled opponents
            has_pool = any(
                info.get("agent_type") == "pool" for info in self.opponent_infos
            )
            self.opponent_type = "self_play_pool" if has_pool else "self_play_random"

        # Initialize new game state
        from ..game.game_state import GameState
        from ..game.constants import NUM_PLAYERS
        self.state = GameState()

        # Determine agent position for this episode
        if options and "agent_position" in options:
            self.agent_player_id = options["agent_position"]
        elif self._agent_position_config is not None:
            self.agent_player_id = self._agent_position_config
        else:
            self.agent_player_id = np.random.randint(0, NUM_PLAYERS)

        self._episode_reward = 0.0
        self._episode_length = 0

        # Start logging BEFORE playing opponents
        self._logging_this_game = self.game_logger.start_game(
            game_state=self.state,
            agent_player_id=self.agent_player_id,
            opponent_type=self.opponent_type,
            opponent_infos=self.opponent_infos,
            training_step=self.training_step,
        )

        # Now play opponents until agent's turn (these will be logged)
        self._play_opponents_until_agent_turn()

        observation = self._get_observation()
        info = self._get_info()
        info["agent_position"] = self.agent_player_id

        return observation, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step and log the action.

        This overrides the parent step() to ensure proper logging order:
        agent action is logged BEFORE opponent actions are played.
        """
        assert self.state is not None, "Must call reset() before step()"

        # Get action details before executing
        action_obj = index_to_action(action)
        player_id = self.state.current_player_idx
        prev_phase = self.state.phase
        prev_state = self.state.copy()

        # Capture pre-action state for dev card logging
        pre_action_info = {}
        if self._logging_this_game:
            pre_action_info = self._capture_pre_action_state(action_obj, player_id)

        # Validate action
        action_mask = Rules.get_legal_actions(self.state)
        if not action_mask[action]:
            # Invalid action - return negative reward and don't change state
            observation = self._get_observation()
            info = self._get_info()
            info["invalid_action"] = True
            return observation, -1.0, False, False, info

        # Execute agent's action
        Rules.execute_action(self.state, action)

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            prev_state, self.state, action_obj.type, self.agent_player_id
        )
        self._episode_reward += reward

        # Log the agent's action BEFORE playing opponents
        if self._logging_this_game:
            self._log_action(action, action_obj, player_id, prev_phase, pre_action_info)

            # Log dice roll result
            if action_obj.type == ActionType.ROLL_DICE:
                self.game_logger.log_dice_roll(
                    self.state,
                    player_id,
                    self.state.last_dice_roll
                )

        # Check for termination
        terminated = self.state.is_game_over
        truncated = self._episode_length >= self.max_turns

        if not terminated and not truncated:
            # Play opponents until it's the agent's turn again
            # This will log opponent actions in the correct order (after agent's action)
            self._play_opponents_until_agent_turn()

            # Check again after opponents played
            terminated = self.state.is_game_over
            truncated = self._episode_length >= self.max_turns

        self._episode_length += 1

        # End game logging if episode finished
        if terminated or truncated:
            if self._logging_this_game:
                self.game_logger.end_game(
                    game_state=self.state,
                    winner=self.state.winner
                )
            self._logging_this_game = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _capture_pre_action_state(self, action_obj: Any, player_id: int) -> Dict[str, Any]:
        """Capture state before action execution for enhanced logging."""
        pre_info = {}
        player = self.state.players[player_id]

        if action_obj.type == ActionType.BUY_DEV_CARD:
            # Capture dev cards before purchase to determine what was drawn
            pre_info["dev_cards_before"] = dict(player.dev_cards)

        elif action_obj.type == ActionType.PLAY_KNIGHT:
            # Capture knights played count
            pre_info["knights_before"] = player.knights_played

        elif action_obj.type == ActionType.MOVE_ROBBER:
            # Capture current robber location before move
            pre_info["robber_from_tile"] = self.state.board.get_robber_tile()

        elif action_obj.type == ActionType.PLAY_MONOPOLY:
            # Capture all players' resources to determine what was taken
            pre_info["all_resources_before"] = {
                p.id: dict(p.resources) for p in self.state.players
            }

        return pre_info

    def _log_action(
        self,
        action_index: int,
        action_obj: Any,
        player_id: int,
        prev_phase: GamePhase,
        pre_action_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an action with appropriate details."""
        details = {}
        pre_action_info = pre_action_info or {}
        player = self.state.players[player_id]

        # Add location details for building actions
        if action_obj.type in {ActionType.PLACE_SETTLEMENT, ActionType.BUILD_SETTLEMENT}:
            details["corner_id"] = action_obj.corner_id

            # Check if this is setup phase
            if prev_phase in {GamePhase.SETUP_SETTLEMENT_1, GamePhase.SETUP_SETTLEMENT_2}:
                is_second = prev_phase == GamePhase.SETUP_SETTLEMENT_2
                self.game_logger.log_setup_placement(
                    self.state,
                    player_id,
                    "settlement",
                    action_obj.corner_id,
                    is_second_placement=is_second
                )
                return

        elif action_obj.type == ActionType.BUILD_CITY:
            details["corner_id"] = action_obj.corner_id

        elif action_obj.type in {ActionType.BUILD_ROAD, ActionType.PLACE_ROAD}:
            details["edge_id"] = action_obj.edge_id

            # Check if this is setup phase
            if prev_phase in {GamePhase.SETUP_ROAD_1, GamePhase.SETUP_ROAD_2}:
                is_second = prev_phase == GamePhase.SETUP_ROAD_2
                self.game_logger.log_setup_placement(
                    self.state,
                    player_id,
                    "road",
                    action_obj.edge_id,
                    is_second_placement=is_second
                )
                return

        elif action_obj.type == ActionType.MOVE_ROBBER:
            details["to_tile"] = action_obj.tile_id
            details["from_tile"] = pre_action_info.get("robber_from_tile")
            # Get tile resource type
            if action_obj.tile_id < len(self.state.board.tiles):
                tile = self.state.board.tiles[action_obj.tile_id]
                details["tile_resource"] = tile.resource.name
                details["tile_number"] = tile.number
            # Get players adjacent to new robber location
            adjacent_players = self.state.board.get_adjacent_players(action_obj.tile_id)
            details["adjacent_players"] = [p for p in adjacent_players if p != player_id]

        elif action_obj.type == ActionType.BANK_TRADE:
            if action_obj.give_resource is not None:
                details["give_resource"] = action_obj.give_resource.name
                details["get_resource"] = action_obj.receive_resource.name
                details["trade_ratio"] = "4:1"

        elif action_obj.type == ActionType.PORT_TRADE:
            if action_obj.give_resource is not None:
                details["give_resource"] = action_obj.give_resource.name
                details["get_resource"] = action_obj.receive_resource.name
                # Determine trade ratio based on player's ports
                trade_ratio = player.get_trade_ratio(action_obj.give_resource)
                details["trade_ratio"] = f"{trade_ratio}:1"
                # Check if using specific port or generic
                port_resource_map = {
                    PortType.LUMBER: Resource.LUMBER,
                    PortType.BRICK: Resource.BRICK,
                    PortType.WOOL: Resource.WOOL,
                    PortType.GRAIN: Resource.GRAIN,
                    PortType.ORE: Resource.ORE,
                }
                port_type = "GENERIC"
                for port, res in port_resource_map.items():
                    if res == action_obj.give_resource and port in player.ports:
                        port_type = port.name
                        break
                details["port_type"] = port_type

        elif action_obj.type == ActionType.STEAL:
            details["steal_from_player"] = action_obj.steal_from_player

        # Enhanced dev card logging
        elif action_obj.type == ActionType.BUY_DEV_CARD:
            # Determine which card was drawn by comparing before/after
            dev_cards_before = pre_action_info.get("dev_cards_before", {})
            dev_cards_after = dict(player.dev_cards)
            card_drawn = None
            for card_type in DevCardType:
                before_count = dev_cards_before.get(card_type, 0)
                after_count = dev_cards_after.get(card_type, 0)
                if after_count > before_count:
                    card_drawn = card_type.name
                    break
            details["card_drawn"] = card_drawn
            # Track if it's a VP card (hidden until game end)
            if card_drawn == "VICTORY_POINT":
                details["is_vp_card"] = True
                details["total_vp_cards"] = dev_cards_after.get(DevCardType.VICTORY_POINT, 0)

        elif action_obj.type == ActionType.PLAY_KNIGHT:
            details["knights_played"] = player.knights_played
            details["has_largest_army"] = player.has_largest_army

        elif action_obj.type == ActionType.PLAY_ROAD_BUILDING:
            # Currently grants resources for 2 roads (simplified implementation)
            details["resources_granted"] = {"LUMBER": 2, "BRICK": 2}
            details["note"] = "Grants resources for 2 roads"

        elif action_obj.type == ActionType.PLAY_YEAR_OF_PLENTY:
            # Currently hardcoded to give LUMBER + BRICK (simplified)
            details["resources_received"] = ["LUMBER", "BRICK"]
            details["note"] = "Simplified: always grants LUMBER + BRICK"

        elif action_obj.type == ActionType.PLAY_MONOPOLY:
            # Determine which resource was monopolized and how much was taken
            all_resources_before = pre_action_info.get("all_resources_before", {})
            player_resources_after = dict(player.resources)

            # Find which resource increased the most (excluding player's own)
            best_resource = None
            best_gain = 0
            for res in Resource.tradeable():
                before_total = sum(
                    all_resources_before.get(p.id, {}).get(res, 0)
                    for p in self.state.players if p.id != player_id
                )
                gain = before_total  # This is what we took
                if gain > best_gain:
                    best_gain = gain
                    best_resource = res

            if best_resource:
                details["resource_monopolized"] = best_resource.name
                details["total_taken"] = best_gain
            details["note"] = "Simplified: takes most abundant resource"

        # Log the action
        self.game_logger.log_action(
            game_state=self.state,
            action_index=action_index,
            action_type=action_obj.type,
            player_id=player_id,
            details=details
        )

    def _play_opponents_until_agent_turn(self) -> None:
        """Play opponent turns until it's the agent's turn, logging all actions."""
        max_opponent_actions = 1000  # Safety limit
        actions_taken = 0

        while (
            not self.state.is_game_over and
            self.state.current_player_idx != self.agent_player_id and
            actions_taken < max_opponent_actions
        ):
            opponent_id = self.state.current_player_idx
            prev_phase = self.state.phase
            action_mask = Rules.get_legal_actions(self.state)

            # Get opponent action
            opponent_policy = self.opponent_policies[opponent_id - 1] if opponent_id > 0 else None

            if opponent_policy is not None:
                # Use opponent policy
                from .state_encoder import StateEncoder
                encoder = StateEncoder()
                obs = encoder.encode(self.state, opponent_id)
                action = opponent_policy.predict(obs, action_mask)
            else:
                # Random action
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) == 0:
                    break
                action = np.random.choice(valid_actions)

            # Get action object for logging
            action_obj = index_to_action(action)

            # Capture pre-action state for enhanced logging
            pre_action_info = {}
            if self._logging_this_game:
                pre_action_info = self._capture_pre_action_state(action_obj, opponent_id)

            # Execute opponent action
            Rules.execute_action(self.state, action)
            actions_taken += 1

            # Log the opponent's action
            if self._logging_this_game:
                self._log_action(action, action_obj, opponent_id, prev_phase, pre_action_info)

                # Log dice roll result if applicable
                if action_obj.type == ActionType.ROLL_DICE:
                    self.game_logger.log_dice_roll(
                        self.state,
                        opponent_id,
                        self.state.last_dice_roll
                    )

    def get_game_logger(self) -> GameLogger:
        """Get the game logger instance."""
        return self.game_logger


def make_logging_env(
    opponent_policies: Optional[List] = None,
    reward_config: Optional[RewardConfig] = None,
    game_logger: Optional[GameLogger] = None,
    log_every_n_games: int = 100,
    log_level: str = "key_actions",
    save_dir: str = "./game_logs",
    shared_file_path: Optional[str] = None,
    rank: int = 0,
    seed: int = 0,
    opponent_type: str = "random",
    agent_position: Optional[int] = None,
    opponent_sampler: Optional[callable] = None,
):
    """Create a logging Catan environment.

    This is a factory function for use with vectorized environments.

    Args:
        opponent_policies: List of opponent policies.
        reward_config: Reward configuration.
        game_logger: Shared GameLogger instance.
        log_every_n_games: Only log every N games.
        log_level: Detail level.
        save_dir: Directory to save logs.
        shared_file_path: If set, all loggers write to this file with locking.
        rank: Environment rank (for parallel envs).
        seed: Random seed.
        opponent_type: Type of opponents.
        agent_position: Fixed position for agent (0-3), or None for random.
        opponent_sampler: Optional callable for sampling opponents on each reset.

    Returns:
        A function that creates the environment.
    """
    from stable_baselines3.common.monitor import Monitor
    from sb3_contrib.common.wrappers import ActionMasker

    def mask_fn(env):
        """Get action mask, handling wrapped environments."""
        if hasattr(env, 'action_masks'):
            return env.action_masks()
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'action_masks'):
            return env.unwrapped.action_masks()
        elif hasattr(env, 'env') and hasattr(env.env, 'action_masks'):
            return env.env.action_masks()
        else:
            raise AttributeError(f"Cannot find action_masks method on {type(env)}")

    def _init():
        env = LoggingCatanEnv(
            opponent_policies=opponent_policies,
            reward_config=reward_config,
            agent_position=agent_position,
            opponent_sampler=opponent_sampler,
            game_logger=game_logger,
            log_every_n_games=log_every_n_games,
            log_level=log_level,
            save_dir=save_dir,
            shared_file_path=shared_file_path,
            opponent_type=opponent_type,
        )
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed + rank)
        return env

    return _init
